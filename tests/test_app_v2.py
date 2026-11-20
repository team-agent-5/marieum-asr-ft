"""서버 통합 테스트: 실제 FastAPI 앱 + GPU 작업자 스레드 + 가짜 엔진 (engine_handoff 6절 3단계)."""

import io
import json
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from serve import app as appmod
from serve.config import settings as S

from fakes import FakeEngine, FakeVad, pcm16, silence, speech


def make_client(engine):
    app = appmod.create_app(engine_factory=lambda: engine, vad_factory=FakeVad)
    c = TestClient(app)
    c.__enter__()
    for _ in range(100):
        if c.get("/healthz").status_code == 200:
            break
        time.sleep(0.02)
    return c


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def client(engine):
    c = make_client(engine)
    yield c
    c.__exit__(None, None, None)
    appmod.Session.live.clear()


def recv_until(ws, kind):
    out = []
    while True:
        m = json.loads(ws.receive_text())
        out.append(m)
        if m["type"] == kind:
            return out


def send_audio(ws, x, frame=3200):
    for i in range(0, len(x), frame):
        ws.send_bytes(pcm16(x[i:i + frame]))


def test_healthz(client):
    h = client.get("/healthz").json()
    assert h["status"] == "ok" and h["engine"] == "nemotron" and h["sessions"] == 0
    assert client.get("/livez").json() == {"status": "alive"}


def test_session_roundtrip_and_reuse(client, engine):
    with client.websocket_connect("/asr-stream/v2") as ws:
        for trace in ("s1", "s2"):  # 같은 연결에서 세션을 차례로 두 번
            ws.send_text(json.dumps({"type": "start", "trace_id": trace, "sample_rate": 16000}))
            ready = json.loads(ws.receive_text())
            assert ready["type"] == "ready" and ready["seq"] == 1 and ready["chunk_ms"] == 320
            send_audio(ws, np.concatenate([speech(1.0), silence(2.5), speech(0.8)]))
            ws.send_text(json.dumps({"type": "end", "trace_id": trace}))
            msgs = recv_until(ws, "session_end")
            assert all(m["trace_id"] == trace for m in msgs)
            assert [m["seq"] for m in msgs] == list(range(2, len(msgs) + 2))
            finals = [m for m in msgs if m["type"] == "segment_final"]
            assert [f["reason"] for f in finals] == ["silence", "end"] and finals[-1]["is_last"]
            assert all(f["text"] for f in finals)
            assert msgs[-1]["segments"] == 2
    deadline = time.time() + 2
    while engine.states and time.time() < deadline:
        time.sleep(0.01)
    assert engine.states == {}  # 파이프라인 상태가 남지 않는다


def test_segment_log_fields(client, capsys):
    """발화마다 로그에 9.6 필드가 남고, 글자는 남지 않는다."""
    with client.websocket_connect("/asr-stream/v2") as ws:
        ws.send_text(json.dumps({"type": "start", "trace_id": "lg", "sample_rate": 16000}))
        json.loads(ws.receive_text())
        x = np.concatenate([speech(1.0), silence(2.5), speech(0.8)])
        for i in range(0, len(x), 3200):  # 파이프라인이 따라오게 조금씩 (한꺼번에 보내면 VAD가 먼저 구간을 닫는다)
            ws.send_bytes(pcm16(x[i:i + 3200]))
            time.sleep(0.02)
        ws.send_text(json.dumps({"type": "end", "trace_id": "lg"}))
        msgs = recv_until(ws, "session_end")
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    segs = [r for r in logs if r["event"] == "segment_final" and r["trace_id"] == "lg"]
    deltas = [m for m in msgs if m["type"] == "transcript_delta"]
    assert len(segs) == 2
    for r in segs:
        assert {"speech_ms", "first_delta_ms", "final_ms", "chars", "deltas", "revised", "reason"} <= r.keys()
        assert "text" not in r
    assert sum(r["deltas"] for r in segs) == len(deltas)
    assert segs[0]["deltas"] > 0
    assert all((r["first_delta_ms"] is None) == (r["deltas"] == 0) for r in segs)  # delta 없이 끝난 발화는 null


def test_errors(client):
    with client.websocket_connect("/asr-stream/v2") as ws:
        ws.send_bytes(pcm16(speech(0.1)))
        assert json.loads(ws.receive_text())["code"] == "NO_SESSION"
        ws.send_text("not json")
        assert json.loads(ws.receive_text())["code"] == "INVALID_FORMAT"
        ws.send_text(json.dumps({"type": "start", "trace_id": "a", "sample_rate": 100}))
        assert json.loads(ws.receive_text())["code"] == "INVALID_FORMAT"
        ws.send_text(json.dumps({"type": "ping", "trace_id": "a"}))
        assert json.loads(ws.receive_text()) == {"type": "pong", "trace_id": "a"}
        ws.send_text(json.dumps({"type": "start", "trace_id": "a", "sample_rate": 16000}))
        assert json.loads(ws.receive_text())["type"] == "ready"
        ws.send_text(json.dumps({"type": "start", "trace_id": "b", "sample_rate": 16000}))
        err = json.loads(ws.receive_text())
        assert err["code"] == "SESSION_BUSY" and err["fatal"] is False and "seq" not in err
        ws.send_text(json.dumps({"type": "end", "trace_id": "b"}))
        assert json.loads(ws.receive_text())["code"] == "NO_SESSION"


def test_overloaded_backlog(client):
    with client.websocket_connect("/asr-stream/v2") as ws:
        ws.send_text(json.dumps({"type": "start", "trace_id": "fast", "sample_rate": 16000}))
        json.loads(ws.receive_text())
        ws.send_bytes(pcm16(speech(8.0)))  # 실제 속도보다 8초 앞서 보낸다
        msgs = recv_until(ws, "error")
        assert msgs[-1]["code"] == "OVERLOADED" and msgs[-1]["fatal"] and msgs[-1]["seq"] >= 2
        ws.send_text(json.dumps({"type": "start", "trace_id": "again", "sample_rate": 16000}))
        assert json.loads(ws.receive_text())["type"] == "ready"  # 세션이 닫혀서 새로 시작할 수 있다


def test_max_sessions(client, monkeypatch):
    monkeypatch.setattr(S, "max_sessions", 1)
    with client.websocket_connect("/asr-stream/v2") as a, client.websocket_connect("/asr-stream/v2") as b:
        a.send_text(json.dumps({"type": "start", "trace_id": "a", "sample_rate": 16000}))
        assert json.loads(a.receive_text())["type"] == "ready"
        b.send_text(json.dumps({"type": "start", "trace_id": "b", "sample_rate": 16000}))
        err = json.loads(b.receive_text())
        assert err["code"] == "OVERLOADED" and err["fatal"] is False


def test_disconnect_frees_pipeline_state(client, engine):
    with client.websocket_connect("/asr-stream/v2") as ws:
        ws.send_text(json.dumps({"type": "start", "trace_id": "gone", "sample_rate": 16000}))
        json.loads(ws.receive_text())
        send_audio(ws, speech(1.0))
        time.sleep(0.2)
    deadline = time.time() + 2
    while (engine.states or appmod.Session.live) and time.time() < deadline:
        time.sleep(0.01)
    assert engine.states == {} and appmod.Session.live == {}


def test_internal_error_keeps_vad_until_worker_done():
    """step이 실패해 세션이 INTERNAL로 끝나도, 작업 스레드가 쓰는 중인 VAD는 다 쓴 뒤에 돌려준다."""
    bad = []

    class SlowVad(FakeVad):
        busy = False

        def __call__(self, windows):
            self.busy = True
            time.sleep(0.05)
            self.busy = False
            return super().__call__(windows)

        def reset(self):
            if self.busy:
                bad.append("reset while busy")

    class FailEngine(FakeEngine):
        def step(self, reqs):
            raise RuntimeError("boom")

    app = appmod.create_app(engine_factory=FailEngine, vad_factory=SlowVad)
    with TestClient(app) as c:
        for _ in range(100):
            if c.get("/healthz").status_code == 200:
                break
            time.sleep(0.02)
        with c.websocket_connect("/asr-stream/v2") as ws:
            ws.send_text(json.dumps({"type": "start", "trace_id": "x", "sample_rate": 16000}))
            json.loads(ws.receive_text())
            send_audio(ws, speech(2.0))
            assert recv_until(ws, "error")[-1]["code"] == "INTERNAL"
            time.sleep(0.2)
    appmod.Session.live.clear()
    assert bad == []


def test_concurrent_sessions_are_batched():
    eng = FakeEngine(delay_s=0.02)
    c = make_client(eng)
    try:
        with c.websocket_connect("/asr-stream/v2") as a, c.websocket_connect("/asr-stream/v2") as b:
            for ws, t in ((a, "a"), (b, "b")):
                ws.send_text(json.dumps({"type": "start", "trace_id": t, "sample_rate": 16000}))
                json.loads(ws.receive_text())
            x = np.concatenate([speech(1.0), silence(2.0)])
            for i in range(0, len(x), 3200):
                a.send_bytes(pcm16(x[i:i + 3200]))
                b.send_bytes(pcm16(x[i:i + 3200]))
            for ws, t in ((a, "a"), (b, "b")):
                ws.send_text(json.dumps({"type": "end", "trace_id": t}))
                recv_until(ws, "session_end")
        assert max(len(s) for s in eng.steps) == 2  # 두 세션 조각이 한 step에 묶였다
    finally:
        c.__exit__(None, None, None)
        appmod.Session.live.clear()


def test_resample_8k(client, monkeypatch):
    calls = []

    def fake_resample(x, sr):
        calls.append(sr)
        return np.repeat(x, 2) if sr == 8000 else x

    monkeypatch.setattr(appmod, "resample", fake_resample)
    with client.websocket_connect("/asr-stream/v2") as ws:
        ws.send_text(json.dumps({"type": "start", "trace_id": "n", "sample_rate": 8000}))
        assert json.loads(ws.receive_text())["sample_rate"] == 8000
        x = np.concatenate([speech(1.0), silence(2.0)])[::2]
        send_audio(ws, x, frame=1600)
        ws.send_text(json.dumps({"type": "end", "trace_id": "n"}))
        msgs = recv_until(ws, "session_end")
        assert calls and set(calls) == {8000}
        assert msgs[-1]["audio_ms"] >= 2900


def test_post_asr_wav(client):
    x = np.concatenate([speech(1.0), silence(1.5), speech(0.7), silence(0.3)])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16(x))
    r = client.post("/asr?trace_id=f1", content=buf.getvalue(), headers={"content-type": "audio/wav"})
    assert r.status_code == 200
    body = r.json()
    assert body["trace_id"] == "f1" and body["duration_ms"] == 3500
    assert len(body["segments"]) == 2 and body["text"] == " ".join(s["text"] for s in body["segments"])
    assert body["model_version"] == S.model_version


def test_post_asr_concurrency_limit(monkeypatch):
    """파일 요청이 몰려도 동시에 파이프라인에 들어가는 파일은 한도까지만. 나머지는 기다렸다가 처리된다."""
    monkeypatch.setattr(S, "asr_max_concurrent", 2)
    eng = FakeEngine(delay_s=0.01)
    c = make_client(eng)
    try:
        body = pcm16(np.concatenate([speech(1.0), silence(1.0)]))
        with ThreadPoolExecutor(6) as ex:
            rs = list(ex.map(lambda i: c.post(f"/asr?sample_rate=16000&trace_id=f{i}", content=body), range(6)))
        assert [r.status_code for r in rs] == [200] * 6
        assert all(r.json()["text"] for r in rs)
        assert eng.max_states <= 2
    finally:
        c.__exit__(None, None, None)


def test_post_asr_errors(client):
    assert client.post("/asr", content=b"").status_code == 400
    assert client.post("/asr", content=b"\x00\x01").status_code == 415  # sample_rate 없는 PCM
    long = pcm16(silence(61.0))
    assert client.post("/asr?sample_rate=16000", content=long).status_code == 413


def test_model_unavailable():
    def broken():
        raise RuntimeError("no model")

    app = appmod.create_app(engine_factory=broken, vad_factory=FakeVad)
    with TestClient(app) as c:
        time.sleep(0.05)
        h = c.get("/healthz")
        assert h.status_code == 503 and h.json()["status"] == "error"
        with c.websocket_connect("/asr-stream/v2") as ws:
            err = json.loads(ws.receive_text())
            assert err["code"] == "MODEL_UNAVAILABLE" and err["fatal"]
