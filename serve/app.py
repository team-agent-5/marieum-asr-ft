"""Nemotron ft_v3 ML 서버: WebSocket /asr-stream/v2 + POST /asr + /healthz + /livez.

스펙은 docs/engine_handoff.md 9절, 서버 구조는 3.3. marieum-ai(Whisper, v1)와 같은 포트·로그 형식을 쓴다.

실행 (워커 1개 고정. GPU 작업자와 파이프라인 상태가 프로세스 하나에 있다):
  ML_NEMO_MODEL=/path/to/ft_v3.nemo ML_NEMO_REPO=/path/to/NeMo \
  uvicorn serve.app:app --host 0.0.0.0 --port 8001 --workers 1
"""

import asyncio
import io
import itertools
import json
import re
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from colab.marieum_stream import CONVERSATION, MODES, SR

from .config import settings as S
from .scheduler import StepScheduler
from .stream_v2 import StreamV2, file_reqs

ENDPOINT = "/asr-stream/v2"
TRACE_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
MAX_TEXT = 1000

vad_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vad")


def log(event, trace_id="unbound", endpoint=ENDPOINT, **kw):
    """JSON 한 줄. 전사 문장은 찍지 않는다 (글자 수만)."""
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
           "trace_id": trace_id, "endpoint": endpoint, "event": event,
           "model_version": S.model_version, **kw}
    print(json.dumps(rec, ensure_ascii=False), flush=True)


# ---- VAD -----------------------------------------------------------------
class SileroVad:
    """Silero VAD는 앞 소리를 내부에 기억하므로 세션마다 따로 쓴다 (v1과 같다)."""

    def __init__(self):
        from silero_vad import load_silero_vad
        self.model = load_silero_vad()

    def __call__(self, windows):
        import torch
        with torch.no_grad():
            return [float(self.model(torch.from_numpy(np.ascontiguousarray(w)), SR)) for w in windows]

    def reset(self):
        self.model.reset_states()


class VadPool:
    def __init__(self, factory):
        self.factory, self.free = factory, []

    def acquire(self):
        return self.free.pop() if self.free else self.factory()

    def release(self, v):
        v.reset()
        self.free.append(v)


def resample(x, sr):
    if sr == SR or not len(x):
        return x
    import torch
    import torchaudio
    return torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(x)), sr, SR).numpy()


def default_engine():
    from .engine import NemotronEngine
    eng = NemotronEngine(S)
    t = time.perf_counter()
    eng.warmup(S.batch_max)
    log("warmup_done", warmup_s=round(time.perf_counter() - t, 2))
    return eng


# ---- 연결과 세션 -----------------------------------------------------------
class Conn:
    """연결 하나. 메시지는 대기열 하나로 순서대로 보낸다 (GPU 결과 콜백에서도 바로 넣을 수 있게)."""

    def __init__(self, ws):
        self.ws = ws
        self.q = asyncio.Queue()
        self.session = None
        self.open = True
        self.writer = asyncio.create_task(self._write())

    def push(self, msg):
        if self.open:
            self.q.put_nowait(msg)

    async def _write(self):
        while True:
            msg = await self.q.get()
            if msg is None:
                return
            try:
                await self.ws.send_text(json.dumps(msg, ensure_ascii=False))
            except Exception:  # noqa: BLE001 - 끊긴 연결
                self.open = False
                return

    def error(self, trace_id, code, message, fatal=False):
        """세션 밖 오류 (seq 없음)."""
        log("error", trace_id, status="error", code=code, message=message)
        self.push({"type": "error", "trace_id": trace_id, "code": code, "message": message, "fatal": fatal})

    async def drain(self):
        self.q.put_nowait(None)
        try:
            await asyncio.wait_for(self.writer, 5)
        except Exception:  # noqa: BLE001
            pass


class Session:
    live = {}  # stream_id -> Session

    def __init__(self, srv, conn, trace_id, patient_id, mode, sr, vad):
        self.srv, self.conn, self.trace_id, self.patient_id, self.mode, self.sr = srv, conn, trace_id, patient_id, mode, sr
        self.vad = vad
        self.sid = next(srv.stream_ids)
        self.st = StreamV2(trace_id, mode, S, self.sid, srv.engine.chunk_samples, vad)
        self.odd = b""
        self.bytes_total = 0
        self.t_start = time.perf_counter()
        self.arrivals = []  # (받은 오디오 끝 ms, 도착 시각) 지연 로그용
        self.seg_log = {}  # segment_id -> 발화 로그 (9.6)
        self.vad_busy = False  # 작업 스레드가 self.vad를 쓰는 중
        self.closed = False
        Session.live[self.sid] = self
        conn.session = self
        chunk_ms = srv.engine.chunk_samples * 1000 // SR
        self.send([self.st.msg("ready", engine="nemotron", model_version=S.model_version, chunk_ms=chunk_ms,
                               sample_rate=sr)])

    def send(self, msgs):
        for m in msgs:
            self.conn.push(m)
            if m["type"] == "segment_start":
                self.seg_log[m["segment_id"]] = {"start_ms": m["start_ms"], "deltas": 0, "revised": 0,
                                                 "first_delta_ms": None}
            elif m["type"] == "transcript_delta":
                g = self.seg_log[m["segment_id"]]
                g["deltas"] += 1
                g["revised"] += m["revised"]
                if g["first_delta_ms"] is None:  # 말 시작 오디오가 도착한 때 → 첫 delta
                    g["first_delta_ms"] = round((time.perf_counter() - self.arrival(g["start_ms"])) * 1000)
            elif m["type"] == "segment_final":
                g = self.seg_log.pop(m["segment_id"], {})
                final_ms = round((time.perf_counter() - self.arrival(m["end_ms"])) * 1000)  # 말 끝 → final
                log("segment_final", self.trace_id, status="ok", segment_id=m["segment_id"], reason=m["reason"],
                    chars=len(m["text"]), confidence=m["confidence"], is_last=m["is_last"],
                    speech_ms=m["end_ms"] - m["start_ms"], first_delta_ms=g.get("first_delta_ms"),
                    final_ms=final_ms, deltas=g.get("deltas", 0), revised=g.get("revised", 0),
                    latency_ms=final_ms)
            elif m["type"] == "session_end":
                self.cleanup("ok")

    def arrival(self, audio_ms):
        for end_ms, t in reversed(self.arrivals):
            if end_ms <= audio_ms + 1:
                return t
        return self.t_start

    def decode(self, data):
        data = self.odd + data
        self.odd = data[len(data) // 2 * 2:]
        x = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        return resample(x, self.sr)

    async def on_audio(self, data):
        if self.closed or self.st.ending:
            return
        self.bytes_total += len(data)

        def work():  # 작업 스레드: 디코딩과 VAD만. 구간 상태는 이벤트 루프에서 바꾼다
            x = self.decode(data)
            windows = self.st.seg.take_windows(x)
            return x, windows, (self.vad(windows) if len(windows) else [])

        self.vad_busy = True
        try:
            x, windows, probs = await asyncio.get_running_loop().run_in_executor(vad_pool, work)
        finally:
            self.vad_busy = False
        if self.closed:  # 작업 중에 세션이 끝났다 (INTERNAL). cleanup이 미뤄 둔 VAD를 이제 돌려준다
            self.release_vad()
            return
        reqs, msgs = self.st.push_audio(x, windows, probs)
        self.arrivals.append((self.st.received_ms, time.perf_counter()))
        if len(self.arrivals) > 400:
            del self.arrivals[:200]
        self.submit(reqs)
        self.send(msgs)
        if not self.closed and not self.st.ending and self.st.backlog_ms > S.max_backlog_ms:
            self.fail("OVERLOADED", f"처리 못 한 오디오가 {S.max_backlog_ms}ms를 넘었습니다. 실제 속도로 보내세요.")

    def end(self):
        if self.closed:
            return
        reqs, msgs = self.st.finish()
        self.submit(reqs)
        self.send(msgs)

    def submit(self, reqs):
        for r in reqs:
            self.srv.scheduler.submit(r, self.on_step)

    def on_step(self, req, out, err):  # 이벤트 루프
        if self.closed:
            return
        if err is not None:
            self.fail("INTERNAL", "전사 중 서버 오류가 발생했습니다.")
            return
        self.send(self.st.on_step(req, out))

    def fail(self, code, message):
        log("error", self.trace_id, status="error", code=code, message=message)
        self.conn.push(self.st.msg("error", code=code, message=message, fatal=True))
        self.srv.scheduler.abort(self.sid)
        self.cleanup(code)

    def abort(self):
        """연결이 끊겼다."""
        if not self.closed:
            self.srv.scheduler.abort(self.sid)
            self.cleanup("disconnected")

    def release_vad(self):
        if self.vad is not None:
            self.srv.vads.release(self.vad)
            self.vad = None

    def cleanup(self, status):
        if self.closed:
            return
        self.closed = True
        Session.live.pop(self.sid, None)
        if self.conn.session is self:
            self.conn.session = None
        if not self.vad_busy:  # 작업 스레드가 쓰는 중이면 on_audio가 끝난 뒤 돌려준다
            self.release_vad()
        log("session_closed", self.trace_id, status=status, patient_id=self.patient_id, mode=self.mode,
            audio_ms=self.st.received_ms, audio_bytes_total=self.bytes_total, segments=self.st.segments_sent,
            session_duration_ms=round((time.perf_counter() - self.t_start) * 1000))


# ---- 앱 -------------------------------------------------------------------
class Server:
    def __init__(self, engine_factory, vad_factory):
        self.engine_factory, self.vads = engine_factory, VadPool(vad_factory)
        self.status, self.error, self.started = "loading", None, time.monotonic()
        self.engine = self.scheduler = None
        self.stream_ids = itertools.count(1)
        self.file_slots = asyncio.Semaphore(S.asr_max_concurrent)  # POST /asr 동시 처리 한도

    async def boot(self):
        loop = asyncio.get_running_loop()
        try:
            self.engine = await loop.run_in_executor(None, self.engine_factory)
            self.vads.free.extend([self.vads.acquire() for _ in range(4)])  # 세션 시작 지연 줄이기
        except Exception as e:  # noqa: BLE001
            self.status, self.error = "error", repr(e)
            log("model_load_failed", status="error", message=repr(e))
            return
        self.scheduler = StepScheduler(self.engine, S.batch_max, S.step_wait_ms, loop, log)
        self.scheduler.start()
        self.status = "ok"
        log("model_loaded", load_s=round(getattr(self.engine, "load_s", 0.0), 1),
            chunk_ms=self.engine.chunk_samples * 1000 // SR)

    async def stats_loop(self):
        while True:
            await asyncio.sleep(10)
            if self.scheduler:
                log("stats", sessions=len(Session.live), max_backlog_frames=self.scheduler.backlog(),
                    **self.scheduler.metrics())


def create_app(engine_factory=default_engine, vad_factory=SileroVad):
    srv = Server(engine_factory, vad_factory)

    @asynccontextmanager
    async def lifespan(app):
        tasks = [asyncio.create_task(srv.boot()), asyncio.create_task(srv.stats_loop())]
        yield
        for t in tasks:
            t.cancel()
        if srv.scheduler:
            srv.scheduler.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.srv = srv

    @app.get("/healthz")
    async def healthz():
        up = int(time.monotonic() - srv.started)
        if srv.status != "ok":
            return JSONResponse({"status": srv.status, "engine": "nemotron", "uptime_sec": up,
                                 "message": srv.error or "모델 로딩 중"}, status_code=503)
        return {"status": "ok", "engine": "nemotron", "models": ["asr"],
                "model_version": {"asr": S.model_version}, "gpu": getattr(srv.engine, "gpu", {}),
                "uptime_sec": up, "sessions": len(Session.live),
                "gpu_pending": srv.scheduler.backlog(), **srv.scheduler.metrics()}

    @app.get("/livez")
    async def livez():
        return {"status": "alive"}

    @app.websocket(ENDPOINT)
    async def asr_stream_v2(ws: WebSocket):
        await ws.accept()
        conn = Conn(ws)
        if srv.status != "ok":
            conn.error("unbound", "MODEL_UNAVAILABLE", "모델 로딩 중입니다. 잠시 후 다시 연결하세요.", fatal=True)
            await conn.drain()
            await ws.close(1013)
            return
        try:
            while conn.open:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    s = conn.session
                    if s is None:
                        conn.error("unbound", "NO_SESSION", "start 없이 오디오를 받았습니다.")
                        continue
                    await s.on_audio(msg["bytes"])
                    continue
                await handle_text(srv, conn, msg.get("text") or "")
        except WebSocketDisconnect:
            pass
        finally:
            conn.open = False
            if conn.session:
                conn.session.abort()
            conn.writer.cancel()

    @app.post("/asr")
    async def asr(request: Request, sample_rate: int | None = None, trace_id: str | None = None):
        return await asr_file(srv, request, sample_rate, trace_id)

    return app


async def handle_text(srv, conn, raw):
    try:
        p = json.loads(raw)
        kind, trace = p.get("type"), p.get("trace_id")
    except (ValueError, AttributeError):
        conn.error("unbound", "INVALID_FORMAT", "JSON 메시지가 아닙니다.")
        return
    if kind == "ping":
        conn.push({"type": "pong", "trace_id": trace or "unbound"})
        return
    if not isinstance(trace, str) or not TRACE_RE.match(trace):
        conn.error("unbound", "INVALID_FORMAT", "trace_id가 없거나 형식이 맞지 않습니다.")
        return
    s = conn.session
    if kind == "start":
        if s is not None:
            conn.error(trace, "SESSION_BUSY", f"진행 중인 세션({s.trace_id})이 끝난 뒤 시작하세요.")
            return
        await start_session(srv, conn, p, trace)
    elif kind == "end":
        if s is None or s.trace_id != trace:
            conn.error(trace, "NO_SESSION", "시작하지 않은 세션입니다.")
            return
        s.end()
    else:
        conn.error(trace, "INVALID_FORMAT", f"알 수 없는 메시지 type: {kind!r}")


async def start_session(srv, conn, p, trace):
    enc, sr, mode = p.get("encoding", "pcm16"), p.get("sample_rate"), p.get("mode", CONVERSATION)
    if enc != "pcm16":
        return conn.error(trace, "INVALID_FORMAT", "encoding은 pcm16만 지원합니다.")
    if not isinstance(sr, int) or isinstance(sr, bool) or not 8000 <= sr <= 48000:
        return conn.error(trace, "INVALID_FORMAT", "sample_rate는 8000–48000 정수여야 합니다.")
    if mode not in MODES:
        return conn.error(trace, "INVALID_FORMAT", f"mode는 {MODES} 중 하나여야 합니다.")
    if p.get("language", "ko") != "ko":
        return conn.error(trace, "INVALID_FORMAT", "language는 ko만 지원합니다.")
    if len(Session.live) >= S.max_sessions:
        return conn.error(trace, "OVERLOADED", f"동시 세션이 한도({S.max_sessions})에 찼습니다.")
    vad = await asyncio.get_running_loop().run_in_executor(vad_pool, srv.vads.acquire)
    s = Session(srv, conn, trace, str(p.get("patient_id", ""))[:64], mode, sr, vad)
    log("start", trace, status="ok", patient_id=s.patient_id, mode=mode, sample_rate=sr, stream_id=s.sid)


# ---- POST /asr: 녹음 파일 한 번에 전사 --------------------------------------
class AsrInputError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def decode_upload(body, sample_rate):
    """WAV(PCM 8/16/24/32비트, 채널 수 무관) 또는 PCM16 LE 모노(sample_rate 쿼리 필요) → 16kHz float32 모노.
    marieum-ai server/app.py와 같은 규칙이다."""
    if body[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(body)) as w:
                sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
                raw = w.readframes(w.getnframes())
        except (wave.Error, EOFError) as e:
            raise AsrInputError(415, "INVALID_FORMAT", f"PCM WAV만 지원합니다 ({e}).") from None
        if width == 3:  # 24비트 → 32비트로 늘려서 읽는다
            b = np.frombuffer(raw[: len(raw) // 3 * 3], np.uint8).reshape(-1, 3)
            raw = np.concatenate([np.zeros((len(b), 1), np.uint8), b], axis=1).tobytes()
            width = 4
        dtype = {1: np.uint8, 2: "<i2", 4: "<i4"}.get(width)
        if dtype is None:
            raise AsrInputError(415, "INVALID_FORMAT", f"지원하지 않는 WAV 샘플 크기: {width}바이트")
        x = np.frombuffer(raw[: len(raw) // (width * ch) * width * ch], dtype).astype(np.float32)
        x = (x - 128) / 128 if width == 1 else x / float(2 ** (8 * width - 1))
        x = x.reshape(-1, ch).mean(axis=1) if ch > 1 else x
    elif sample_rate is not None:
        sr = sample_rate
        x = np.frombuffer(body[: len(body) // 2 * 2], "<i2").astype(np.float32) / 32768.0
    else:
        raise AsrInputError(415, "INVALID_FORMAT",
                            "WAV 파일을 보내거나, PCM16 LE 모노 바이트와 ?sample_rate=를 함께 보내세요.")
    if not 8000 <= sr <= 48000:
        raise AsrInputError(400, "INVALID_FORMAT", "sample_rate는 8000–48000이어야 합니다.")
    if len(x) / sr > S.asr_max_s:
        raise AsrInputError(413, "AUDIO_TOO_LONG", f"음성은 {S.asr_max_s:g}초까지 받습니다.")
    return resample(x, sr)


def asr_error(status, code, message, trace):
    log("error", trace, endpoint="/asr", status="error", code=code, message=message)
    return JSONResponse({"code": code, "message": message, "trace_id": trace}, status_code=status)


async def run_file(srv, x):
    """파일 오디오를 stream 하나로 파이프라인에 넣고 조각마다 (Req, StepOut)을 모은다."""
    sid = next(srv.stream_ids)
    done = asyncio.get_running_loop().create_future()
    outs = []

    def on_step(req, out, err):
        if done.done():
            return
        if err is not None:
            done.set_exception(err)
            return
        outs.append((req, out))
        if req.is_last:
            done.set_result(None)

    for r in file_reqs(sid, x, srv.engine.chunk_samples):
        srv.scheduler.submit(r, on_step)
    try:
        await done
    except BaseException:
        srv.scheduler.abort(sid)
        raise
    return outs


async def asr_file(srv, request, sample_rate, trace_id):
    """파일 하나를 stream 하나로 보고 같은 파이프라인에 조각을 차례로 넣는다 (3.7). 구간마다 따로 돌리지 않는다 (EXP-006).
    segments는 파이프라인 확정(stop_history_eou) 단위다."""
    trace = trace_id or request.headers.get("x-trace-id") or str(uuid.uuid4())
    if not TRACE_RE.fullmatch(trace):
        return asr_error(400, "INVALID_FORMAT", "trace_id는 영문·숫자·_- 64자 이내여야 합니다.", "unbound")
    if srv.status != "ok":
        return asr_error(503, "MODEL_UNAVAILABLE", "모델 로딩 중입니다. 잠시 후 다시 시도하세요.", trace)
    max_bytes = int(S.asr_max_s * 48000 * 2 * 4) + 1024
    if int(request.headers.get("content-length") or 0) > max_bytes:
        return asr_error(413, "AUDIO_TOO_LONG", f"음성은 {S.asr_max_s:g}초까지 받습니다.", trace)
    body = await request.body()
    if not body:
        return asr_error(400, "INVALID_FORMAT", "본문에 음성이 없습니다.", trace)
    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    try:
        x = await loop.run_in_executor(None, decode_upload, body, sample_rate)
    except AsrInputError as e:
        return asr_error(e.status, e.code, e.message, trace)

    try:
        async with srv.file_slots:
            outs = await run_file(srv, x)
    except Exception as e:  # noqa: BLE001
        log("asr_failed", trace, endpoint="/asr", status="error", message=repr(e))
        return asr_error(500, "INTERNAL", "전사 중 서버 오류가 발생했습니다.", trace)

    segs, prev_ms = [], 0
    for req, out in outs:
        text = out.final_text.strip()
        if not text:
            continue
        g = out.final_segments
        start_ms = round(min(a for a, _, _ in g) * 1000) if g else prev_ms
        end_ms = round(max(b for _, b, _ in g) * 1000) if g else req.audio_end_ms
        conf = round(sum(c for _, _, c in g) / len(g), 4) if g else 0.0
        segs.append({"text": text[:MAX_TEXT], "confidence": conf, "start_ms": start_ms, "end_ms": end_ms})
        prev_ms = req.audio_end_ms
    total = sum(max(1, s["end_ms"] - s["start_ms"]) for s in segs)
    conf = round(sum(s["confidence"] * max(1, s["end_ms"] - s["start_ms"]) for s in segs) / total, 4) if segs else 0.0
    body = {"trace_id": trace, "text": " ".join(s["text"] for s in segs)[:MAX_TEXT], "confidence": conf,
            "duration_ms": len(x) * 1000 // SR, "model_version": S.model_version, "segments": segs}
    log("transcript_file", trace, endpoint="/asr", status="ok", latency_ms=round((time.perf_counter() - t0) * 1000),
        confidence=conf, duration_ms=body["duration_ms"], segments=len(segs), chars=len(body["text"]))
    return body


app = create_app()
