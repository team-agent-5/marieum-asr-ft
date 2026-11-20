"""GPU 없이 서버를 시험하는 가짜 엔진과 가짜 VAD (engine_handoff 6절 3단계).

FakeEngine은 NeMo 파이프라인 규칙을 흉내 낸다:
  - stream마다 상태가 있고 is_first에서 만들고 is_last 뒤 지운다. 모르는 stream의 조각이 오면 오류
  - 한 번에 같은 stream 조각이 둘 오면 오류
  - 상태 칸(num_slots)이 모자라면 오류 (NeMo context_manager: "No free slots available")
  - 소리가 있는 조각마다 partial에 글자 하나를 붙인다
  - 소리 뒤 무음 조각이 eou_chunks개 이어지면(0.32초 × 3 = 960ms ≥ 800ms) partial을 확정한다
"""

import time

import numpy as np

from serve.engine import StepOut

CHUNK = 5120  # 0.32초
SYL = "가나다라마바사아자차카타파하"


class FakeEngine:
    def __init__(self, eou_chunks=3, delay_s=0.0, revise_at=None, slots=32):
        self.chunk_samples = CHUNK
        self.eou_chunks, self.delay_s, self.revise_at, self.slots = eou_chunks, delay_s, revise_at, slots
        self.states = {}
        self.max_states = 0  # 한꺼번에 살아 있던 stream 수의 최대
        self.steps = []  # 각 step의 stream_id 목록
        self.load_s = 0.0
        self.gpu = {"name": "fake"}

    def step(self, reqs):
        ids = [r.stream_id for r in reqs]
        assert len(ids) == len(set(ids)), "같은 stream 조각 두 개를 한 번에 넣었다"
        self.steps.append(ids)
        if self.delay_s:
            time.sleep(self.delay_s)
        outs = []
        for r in reqs:
            assert len(r.samples) == CHUNK
            if r.is_first:
                assert r.stream_id not in self.states, "is_first가 두 번 왔다"
                if len(self.states) >= self.slots:
                    raise RuntimeError("No free slots available")
                self.states[r.stream_id] = {"partial": "", "sil": 0, "n": 0}
                self.max_states = max(self.max_states, len(self.states))
            st = self.states.get(r.stream_id)
            assert st is not None, f"is_first 없이 조각이 왔다: {r.stream_id}"
            loud = float(np.abs(r.samples[: r.length]).mean()) > 0.01 if r.length else False
            final = ""
            if loud:
                st["partial"] += SYL[st["n"] % len(SYL)]
                st["n"] += 1
                st["sil"] = 0
                if self.revise_at is not None and st["n"] == self.revise_at:
                    st["partial"] = "X" + st["partial"][1:]  # 앞에서 보낸 글자를 바꾼다
            elif st["partial"]:
                st["sil"] += 1
                if st["sil"] >= self.eou_chunks:
                    final, st["partial"] = " " + st["partial"], ""
            if r.is_last and st["partial"]:
                final, st["partial"] = final + " " + st["partial"], ""
            segs = [(0.0, r.audio_end_ms / 1000, 0.9)] if final else []
            outs.append(StepOut(r.stream_id, final, segs, st["partial"]))
            if r.is_last:
                del self.states[r.stream_id]
        return outs


class FakeVad:
    def __call__(self, windows):
        return [1.0 if float(np.abs(w).mean()) > 0.01 else 0.0 for w in windows]

    def reset(self):
        pass


SR = 16000


def speech(sec):
    t = np.arange(int(sec * SR)) / SR
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(sec):
    return np.zeros(int(sec * SR), np.float32)


def pcm16(x):
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
