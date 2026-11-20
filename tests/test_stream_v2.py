"""StreamV2 (세션 상태) 단위 테스트: 가짜 파이프라인으로 메시지 순서와 글자 규칙을 본다 (engine_handoff 9.4)."""

import numpy as np

from colab.marieum_stream import CONVERSATION, UTTERANCE
from serve.config import Settings
from serve.stream_v2 import StreamV2

from fakes import CHUNK, FakeEngine, FakeVad, silence, speech

FRAME = 1600  # 100ms씩 보낸다


class Driver:
    """오디오를 100ms씩 넣고, 나온 조각을 바로 가짜 파이프라인에 돌려 결과를 되먹인다."""

    def __init__(self, mode=CONVERSATION, engine=None, lag=0):
        self.eng = engine or FakeEngine()
        self.vad = FakeVad()
        self.st = StreamV2("t1", mode, Settings(), 7, CHUNK, self.vad)
        self.msgs = []
        self.pending = []
        self.lag = lag  # 파이프라인이 몇 조각 늦게 결과를 내는지

    def _run(self, reqs, flush=False):
        self.pending += reqs
        while len(self.pending) > (0 if flush else self.lag):
            r = self.pending.pop(0)
            (out,) = self.eng.step([r])
            self.msgs += self.st.on_step(r, out)

    def feed(self, x):
        for i in range(0, len(x), FRAME):
            part = x[i:i + FRAME]
            windows = self.st.seg.take_windows(part)
            reqs, msgs = self.st.push_audio(part, windows, self.vad(windows) if len(windows) else [])
            self.msgs += msgs
            self._run(reqs)

    def end(self):
        reqs, msgs = self.st.finish()
        self.msgs += msgs
        self._run(reqs, flush=True)

    def types(self):
        return [m["type"] for m in self.msgs]


def check_protocol(msgs):
    """seq는 1씩, 발화마다 segment_start → delta* → segment_final, 마지막은 is_last final → session_end."""
    assert [m["seq"] for m in msgs] == list(range(1, len(msgs) + 1))
    open_id = None
    expect_id = 0
    for m in msgs:
        if m["type"] == "segment_start":
            assert open_id is None and m["segment_id"] == expect_id
            open_id = m["segment_id"]
        elif m["type"] == "transcript_delta":
            assert m["segment_id"] == open_id
        elif m["type"] == "segment_final":
            assert m["segment_id"] == open_id
            open_id, expect_id = None, expect_id + 1
    assert msgs[-1]["type"] == "session_end"
    assert msgs[-2]["type"] == "segment_final" and msgs[-2]["is_last"]
    assert sum(1 for m in msgs if m.get("is_last")) == 1


def check_text(msgs):
    """append를 이어 붙이면 text가 되고, final text는 마지막 delta text로 시작한다.
    (구간을 닫는 조각에서 새로 확정된 글자는 delta 없이 final에만 들어갈 수 있다. 앱은 final text로 맞춘다)"""
    by = {}
    for m in msgs:
        if m["type"] == "transcript_delta":
            prev = by.get(m["segment_id"], "")
            if not m["revised"]:
                assert prev + m["append"] == m["text"]
            by[m["segment_id"]] = m["text"]
        elif m["type"] == "segment_final" and m["segment_id"] in by:
            assert m["text"].startswith(by[m["segment_id"]])


def test_conversation_two_utterances():
    d = Driver()
    d.feed(np.concatenate([silence(0.5), speech(1.0), silence(2.5), speech(1.0), silence(0.3)]))
    d.end()
    check_protocol(d.msgs)
    check_text(d.msgs)
    finals = [m for m in d.msgs if m["type"] == "segment_final"]
    assert [f["reason"] for f in finals] == ["silence", "end"]
    assert all(f["text"] for f in finals)
    assert finals[0]["start_ms"] < 500 < finals[0]["end_ms"] <= 1600
    assert finals[0]["confidence"] > 0 and finals[0]["no_speech_prob"] == 0.0
    # 첫 글자는 말이 끝나기 전에 delta로 나간다
    first_delta = next(m for m in d.msgs if m["type"] == "transcript_delta")
    assert first_delta["audio_end_ms"] <= 1500
    assert d.eng.states == {}


def test_final_waits_for_pipeline():
    """VAD가 구간을 닫아도, 파이프라인이 그 위치까지 처리하기 전에는 segment_final을 보내지 않는다."""
    d = Driver(lag=10)
    d.feed(np.concatenate([speech(1.0), silence(2.5)]))
    assert "segment_final" not in d.types()
    d._run([], flush=True)
    assert "segment_final" in d.types()
    f = next(m for m in d.msgs if m["type"] == "segment_final")
    assert f["text"]


def test_noise_is_not_a_segment():
    """150ms보다 짧은 소리는 VAD 구간을 열지 않는다 (segment_start 없음)."""
    d = Driver()
    d.feed(np.concatenate([silence(0.5), speech(0.064), silence(2.5)]))
    assert "segment_start" not in d.types()


def test_silence_only_session():
    d = Driver()
    d.feed(silence(3.0))
    d.end()
    check_protocol(d.msgs)
    f = [m for m in d.msgs if m["type"] == "segment_final"]
    assert len(f) == 1 and f[0]["reason"] == "end" and f[0]["text"] == "" and f[0]["no_speech_prob"] == 1.0


def test_utterance_mode_single_final():
    d = Driver(mode=UTTERANCE)
    d.feed(np.concatenate([silence(0.3), speech(1.0), silence(2.5), speech(0.6)]))
    d.end()
    check_protocol(d.msgs)
    check_text(d.msgs)
    f = [m for m in d.msgs if m["type"] == "segment_final"]
    assert len(f) == 1 and f[0]["reason"] == "end" and f[0]["text"]


def test_too_long_closes_session():
    d = Driver(mode=UTTERANCE)
    d.feed(speech(10.5))
    assert d.st.ending
    d._run([], flush=True)
    types = d.types()
    assert "error" in types
    err = next(m for m in d.msgs if m["type"] == "error")
    assert err["code"] == "AUDIO_TOO_LONG" and err["fatal"]
    f = [m for m in d.msgs if m["type"] == "segment_final"]
    assert f[-1]["reason"] == "too_long" and f[-1]["is_last"]
    assert types[-1] == "session_end"
    d.feed(speech(1.0))  # 끝난 뒤 오디오는 버린다
    assert d.types()[-1] == "session_end"


def test_revised_replaces_text():
    d = Driver(engine=FakeEngine(revise_at=3))
    d.feed(np.concatenate([speech(1.5), silence(2.5)]))
    d.end()
    check_protocol(d.msgs)
    rev = [m for m in d.msgs if m["type"] == "transcript_delta" and m["revised"]]
    assert rev and rev[0]["append"] == "" and rev[0]["text"].startswith("X")
    f = next(m for m in d.msgs if m["type"] == "segment_final")
    assert f["text"].startswith("X")


def test_long_speech_is_cut_at_max_length():
    d = Driver()
    d.feed(np.concatenate([speech(16.0), silence(2.5)]))
    d.end()
    check_protocol(d.msgs)
    reasons = [m["reason"] for m in d.msgs if m["type"] == "segment_final"]
    assert reasons[0] == "max_length"
    # 15초 자르기에서 확정 전 글자는 다음 구간으로 넘어간다. 글자는 빠지지 않는다
    total = sum(len(m["text"].replace(" ", "")) for m in d.msgs if m["type"] == "segment_final")
    assert total == 50  # 16초 = 0.32초 조각 50개, 가짜 엔진은 소리 조각마다 한 글자


def test_last_chunk_padding():
    d = Driver()
    d.feed(speech(0.5))  # 0.32초 조각 하나 + 0.18초 남음
    reqs, _ = d.st.finish()
    assert len(reqs) == 1 and reqs[0].is_last and reqs[0].length == 8000 - CHUNK
    assert len(reqs[0].samples) == CHUNK and not reqs[0].samples[reqs[0].length:].any()
