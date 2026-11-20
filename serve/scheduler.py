"""GPU 작업자: 세션마다 쌓인 조각을 묶어 엔진에 넣는다 (engine_handoff 3.3).

- 스레드 하나만 engine.step을 부른다 (파이프라인 상태는 스레드 안전하다고 가정하지 않는다)
- 한 번에 세션마다 가장 오래된 조각 하나씩, 최대 batch_max개
- 첫 조각이 생기고 step_wait_ms가 지나면 모인 만큼 돌린다
- 결과는 이벤트 루프로 넘긴다 (loop.call_soon_threadsafe)
"""

import threading
import time
from collections import deque

import numpy as np

from .engine import Req


class StepScheduler:
    def __init__(self, engine, batch_max, step_wait_ms, loop, log=None):
        self.engine, self.batch_max, self.wait_s = engine, batch_max, step_wait_ms / 1000
        self.loop, self.log = loop, log or (lambda *a, **k: None)
        self.cv = threading.Condition()
        self.queues = {}  # stream_id -> deque[(Req, callback, 넣은 시각)]
        self.seen = set()  # 조각을 하나라도 받은 stream (is_last를 처리하면 지운다)
        self.stats = deque(maxlen=300)  # (step_ms, batch)
        self.running = False
        self.thread = None

    # ---- 이벤트 루프 쪽 ------------------------------------------------------
    def submit(self, req, callback):
        """callback(req, out, err)는 이벤트 루프에서 불린다."""
        with self.cv:
            self.seen.add(req.stream_id)
            self.queues.setdefault(req.stream_id, deque()).append((req, callback, time.perf_counter()))
            self.cv.notify()

    def abort(self, stream_id):
        """세션이 비정상으로 끝났다. 남은 조각을 버리고, 파이프라인 상태를 지우는 마지막 조각만 넣는다."""
        with self.cv:
            if stream_id not in self.seen:
                return
            q = self.queues.get(stream_id)
            if q and q[0][0].is_first:  # 파이프라인이 아직 이 stream을 모른다
                del self.queues[stream_id]
                self.seen.discard(stream_id)
                return
            n = self.engine.chunk_samples
            last = Req(stream_id, np.zeros(n, np.float32), False, True, n, 0)
            self.queues[stream_id] = deque([(last, None, time.perf_counter())])
            self.cv.notify()

    def backlog(self):
        with self.cv:
            return max((len(q) for q in self.queues.values()), default=0)

    def metrics(self):
        st = list(self.stats)
        if not st:
            return {"step_ms_p50": None, "step_ms_p95": None, "batch_mean": None}
        ms = sorted(s for s, _ in st)
        return {"step_ms_p50": round(ms[len(ms) // 2], 1), "step_ms_p95": round(ms[int(len(ms) * 0.95)], 1),
                "batch_mean": round(sum(b for _, b in st) / len(st), 2)}

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, name="gpu-step", daemon=True)
        self.thread.start()

    def stop(self):
        with self.cv:
            self.running = False
            self.cv.notify()
        if self.thread:
            self.thread.join(timeout=5)

    # ---- GPU 스레드 ----------------------------------------------------------
    def _take(self):
        with self.cv:
            while self.running and not self.queues:
                self.cv.wait()
            if not self.running:
                return None
            # 다른 세션 조각을 잠깐 기다린다 (묶을수록 GPU 한 번에 여러 세션)
            deadline = time.perf_counter() + self.wait_s
            while self.running and len(self.queues) < self.batch_max:
                left = deadline - time.perf_counter()
                if left <= 0:
                    break
                self.cv.wait(left)
            heads = sorted(self.queues.items(), key=lambda kv: kv[1][0][2])[: self.batch_max]
            batch = []
            for sid, q in heads:
                batch.append(q.popleft())
                if not q:
                    del self.queues[sid]
            return batch

    def _run(self):
        while True:
            batch = self._take()
            if batch is None:
                return
            reqs = [b[0] for b in batch]
            t = time.perf_counter()
            try:
                outs, err = self.engine.step(reqs), None
            except Exception as e:  # noqa: BLE001
                outs, err = [None] * len(reqs), e
                self.log("step_failed", status="error", message=repr(e), batch=len(reqs))
            self.stats.append(((time.perf_counter() - t) * 1000, len(reqs)))
            with self.cv:
                for r in reqs:
                    if r.is_last:
                        self.seen.discard(r.stream_id)
            for (req, cb, _), out in zip(batch, outs):
                if cb is not None:
                    self.loop.call_soon_threadsafe(cb, req, out, err)
