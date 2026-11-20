"""/asr-stream/v2 세션 한 개의 상태 (engine_handoff 3.3, 3.4, 9.4). 웹소켓·GPU를 모른다.

입력 세 가지, 출력은 늘 보낼 메시지 목록이다 (seq는 여기서 붙인다):
  push_audio(x, windows, probs)  16kHz 오디오와 그 VAD 결과 → (파이프라인에 넣을 Req 목록, 메시지)
  on_step(req, out)              파이프라인 결과 → 메시지
  finish(reason)                 end / too_long → (마지막 Req, 메시지)

문장 끊기는 섞은 방식 (3.4, EXP-017):
  - 글자는 파이프라인이 만든다. 파이프라인은 stop_history_eou(800ms)마다 글자를 확정한다 (final_text)
  - 앱에 보이는 문장 경계(segment_id, segment_final)는 VAD 쉼(ML_END_SILENCE_MS, 기본 0.8초, Segmenter)으로 나눈다
  - 구간이 close_ms에서 닫히면 파이프라인이 close_ms까지 처리하기를 기다렸다가, 그때까지 나온 글자를
    모두(확정 + 확정 전 partial) 그 구간 몫으로 segment_final에 넣는다
  - VAD 쉼이 확정(말 끝 뒤 p50 1.1초)보다 짧으면 닫을 때 글자가 아직 partial에 있다. 이 글자는 나중에 파이프라인이
    확정하면서 한 번 더 나오므로(final_text), 이미 보낸 partial(sent_partial)을 다음 글자 앞에서 뺀다
  - confidence: 확정 글자는 final_segments의 값, 확정 전 글자는 그 토큰들의 confidence(partial_confs)로 낸다
"""

from dataclasses import dataclass, field

import numpy as np

from colab.marieum_stream import SR, UTTERANCE, Segmenter, _ms

from .engine import Req


@dataclass
class VSeg:
    """앱에 보이는 문장 하나 (VAD 구간)."""
    start_ms: int
    seg_id: int = -1  # 확정(confirm)될 때 붙인다. 숨소리처럼 버려지는 구간은 번호를 받지 않는다
    close_ms: int | None = None  # VAD가 구간을 닫은 위치 (쉼 포함)
    speech_end_ms: int | None = None
    reason: str = ""
    is_last: bool = False
    started: bool = False  # segment_start를 보냈다
    sent: str = ""  # 이 구간에 transcript_delta로 보낸 글자 전체
    deltas: int = 0
    revised: int = 0


@dataclass
class Fin:
    emit_ms: int
    text: str
    confs: list = field(default_factory=list)  # [(conf, 토큰 수)]


def _join(parts):
    """파이프라인 글자를 그대로 잇는다. 띄어쓰기는 글자 안에 이미 있다: 새 단어로 시작하는 final·partial만
    앞에 공백이 붙고, 단어 중간에서 확정되면 다음 글자는 공백 없이 온다 (' 안' + '녕하세요').
    조각마다 공백을 지우고 다시 넣으면 단어 중간에 공백이 생긴다. 양 끝만 지운다."""
    return "".join(p for p in parts if p).strip()


def token_conf(pairs):
    """[(conf, 토큰 수)] → 토큰 수 가중 평균. 확정마다 단순 평균하면 1토큰 확정(예: 0.35)이 13토큰 확정(0.76)과
    같은 무게가 되어 문장 점수가 0.55로 내려간다. 11-20 TTS 360문장 실험에서 가중 평균이 틀린 문장을 더 잘 가려냈다
    (AUROC CER>0.2: 0.758 vs 0.731, Spearman -0.594 vs -0.562, 글자가 나온 102문장)."""
    total = sum(k for _, k in pairs)
    return round(sum(c * k for c, k in pairs) / total, 4) if total else 0.0


def file_reqs(stream_id, x, n):
    """녹음 파일 하나 → 조각 목록 (POST /asr). 마지막 조각은 0으로 채우고 length에 실제 길이."""
    reqs = []
    for i in range(0, max(len(x), 1), n):
        part = x[i:i + n]
        samples = np.zeros(n, np.float32)
        samples[: len(part)] = part
        reqs.append(Req(stream_id, samples, i == 0, i + n >= len(x), len(part) or n, (i + len(part)) * 1000 // SR))
    return reqs


class StreamV2:
    def __init__(self, trace_id, mode, cfg, stream_id, chunk_samples, vad=None):
        self.trace_id, self.mode, self.stream_id, self.n = trace_id, mode, stream_id, chunk_samples
        self.seg = Segmenter(mode, vad or (lambda w: [0.0] * len(w)), cfg)
        self.min_speech_win = self.seg.min_speech_win
        self.seq = 0
        self.next_id = 0
        self.vsegs = []  # 아직 segment_final을 안 보낸 구간 (앞에서부터 차례로 보낸다)
        self.fins = []  # 아직 어느 구간에도 붙지 않은 파이프라인 확정
        self.partial = ""
        self.partial_confs = []  # partial 토큰들의 confidence
        self.sent_partial = ""  # 닫은 구간에 이미 넣은, 아직 확정되지 않은 글자 (뒤에 오는 글자 앞에서 뺀다)
        self.buf = np.zeros(0, np.float32)
        self.sent_samples = 0  # 파이프라인에 넘긴 오디오 (조각 단위)
        self.received_ms = 0
        self.processed_ms = 0  # 파이프라인 결과가 돌아온 오디오 끝
        self.ending = False  # 마지막 Req를 만들었다
        self.drained = False  # 마지막 Req의 결과가 왔다
        self.done = False  # session_end를 보냈다
        self.last_speech_ms = 0
        if mode == UTTERANCE:  # 발화 모드는 세션 시작부터 구간 하나
            self.vsegs.append(VSeg(start_ms=0))

    # ---- 메시지 ------------------------------------------------------------
    def msg(self, kind, **kw):
        self.seq += 1
        return {"type": kind, "trace_id": self.trace_id, "seq": self.seq, **kw}

    @property
    def backlog_ms(self):
        return self.received_ms - self.processed_ms

    @property
    def segments_sent(self):
        return self.next_id

    # ---- 오디오 ------------------------------------------------------------
    def push_audio(self, x, windows, probs):
        """x: 16kHz float32. windows, probs: Segmenter.take_windows(x)와 그 VAD 결과 (작업 스레드에서 만든다)."""
        if self.ending:
            return [], []
        self.received_ms += len(x) * 1000 // SR
        self.buf = np.concatenate([self.buf, x])
        self._push_vad(windows, probs)
        if self.seg.done:  # 길이 상한 (too_long). 남은 발화를 닫고 세션을 끝낸다
            limit = self.seg.limit_win * 512 // SR
            err = self.msg("error", code="AUDIO_TOO_LONG", message=f"녹음은 {limit}초를 초과할 수 없습니다.",
                           fatal=True)
            return self._last_reqs(), [err] + self._pump()
        reqs = []
        while len(self.buf) >= self.n:
            reqs.append(self._req(self.buf[: self.n], self.n, is_last=False))
            self.buf = self.buf[self.n:]
        return reqs, self._pump()

    def finish(self):
        """end 수신: 남은 오디오를 마지막 조각으로 보내고, 마지막 구간을 닫는다."""
        if self.ending:
            return [], []
        s = self.seg.finish()
        if s is not None:
            self._close(s)
        return self._last_reqs(), self._pump()

    def _req(self, samples, length, is_last):
        r = Req(self.stream_id, np.ascontiguousarray(samples, np.float32), self.sent_samples == 0, is_last,
                length, 0)
        self.sent_samples += length
        r.audio_end_ms = self.sent_samples * 1000 // SR
        return r

    def _last_reqs(self):
        self.ending = True
        reqs = []
        while len(self.buf) > self.n:
            reqs.append(self._req(self.buf[: self.n], self.n, is_last=False))
            self.buf = self.buf[self.n:]
        tail = np.zeros(self.n, np.float32)
        tail[: len(self.buf)] = self.buf
        # 남은 오디오가 없으면 무음 한 조각으로 끝낸다 (길이 0 조각은 파이프라인에서 확인하지 않았다)
        reqs.append(self._req(tail, len(self.buf) or self.n, is_last=True))
        self.buf = np.zeros(0, np.float32)
        return reqs

    # ---- VAD 구간 ----------------------------------------------------------
    def _open(self):
        return self.vsegs[-1] if self.vsegs and self.vsegs[-1].close_ms is None else None

    def _confirm(self, v):
        if v.seg_id < 0:
            v.seg_id = self.next_id
            self.next_id += 1

    def _close(self, s):
        v = self._open()
        if v is None:  # 말소리 없이 끝남 (end, too_long): 빈 구간 하나를 만든다
            v = VSeg(start_ms=s.end_ms - s.duration_ms)
            self.vsegs.append(v)
        self._confirm(v)
        v.close_ms, v.reason = s.end_ms, s.reason
        v.speech_end_ms = min(self.last_speech_ms, s.end_ms) if s.has_speech else s.end_ms
        v.is_last = s.reason in ("end", "too_long")

    def _push_vad(self, windows, probs):
        for w, p in zip(windows, probs):
            if p >= self.seg.thr:
                self.last_speech_ms = self.seg.elapsed_ms + _ms(1)
            for s in self.seg.push_windows([w], [p]):
                self._close(s)
            if self.seg.done:
                return
            v = self._open()
            if self.seg.in_seg:
                if v is None:
                    v = VSeg(start_ms=_ms(self.seg.seg_start_win))
                    self.vsegs.append(v)
                if self.seg.speech_win >= self.min_speech_win:
                    self._confirm(v)
            elif v is not None and v.seg_id < 0:  # 숨소리·클릭음: Segmenter가 구간을 버렸다
                self.vsegs.pop()

    # ---- 파이프라인 결과 ----------------------------------------------------
    def on_step(self, req, out):
        self.processed_ms = max(self.processed_ms, req.audio_end_ms)
        if out.final_text.strip():
            self.fins.append(Fin(req.audio_end_ms, out.final_text, [(c, k) for _, _, c, k in out.final_segments]))
        self.partial = out.partial_text or ""
        self.partial_confs = list(out.partial_confs or [])
        if req.is_last:
            self.drained = True
            self.partial, self.partial_confs = "", []
        return self._pump()

    def _pump(self):
        """보낼 수 있는 것을 순서대로 보낸다: 앞 구간의 segment_final이 다음 구간의 segment_start보다 먼저."""
        out = []
        while self.vsegs:
            v = self.vsegs[0]
            if v.seg_id < 0:
                break
            if not v.started:
                v.started = True
                out.append(self.msg("segment_start", segment_id=v.seg_id, start_ms=v.start_ms))
            if v.close_ms is not None and (self.drained or (not v.is_last and self.processed_ms >= v.close_ms)):
                out.append(self._final(v))
                self.vsegs.pop(0)
                continue
            if v.close_ms is None:
                out += self._delta(v)
            break
        if self.drained and not self.vsegs and not self.done:
            self.done = True
            out.append(self.msg("session_end", segments=self.next_id, audio_ms=self.received_ms))
        return out

    def _pending(self):
        """아직 어느 구간에도 안 들어간 글자 조각 [(글자, [(conf, 토큰 수)])]. 확정들 + partial.
        앞에서 이미 보낸 partial(sent_partial)과 겹치는 글자는 뺀다. 남은 sent_partial도 돌려준다."""
        pieces = [(f.text, f.confs) for f in self.fins]
        if self.partial.strip():
            n = len(self.partial_confs)
            pieces.append((self.partial, [(sum(self.partial_confs) / n, n)] if n else []))
        skip, out = self.sent_partial.strip(), []
        for text, pairs in pieces:
            if skip:
                t = text.lstrip()
                k = min(len(skip), len(t))
                if t[:k] != skip[:k]:  # 앞서 보낸 글자가 바뀌었다 (revised). 더 빼지 않는다
                    skip = ""
                else:
                    skip, rest = skip[k:], t[k:]
                    if not rest.strip():
                        continue  # 이미 보낸 글자뿐인 조각
                    # 일부만 겹친 조각: 남은 글자만, confidence는 남은 글자 비율만큼의 무게로
                    pairs = [(c, max(1, round(n * len(rest.strip()) / max(1, len(t.strip()))))) for c, n in pairs]
                    text = rest
            out.append((text, pairs))
        return out, skip

    def _delta(self, v):
        text = _join([t for t, _ in self._pending()[0]])
        if text == v.sent:
            return []
        revised = not text.startswith(v.sent)
        append = "" if revised else text[len(v.sent):]
        v.sent, v.deltas, v.revised = text, v.deltas + 1, v.revised + revised
        return [self.msg("transcript_delta", segment_id=v.seg_id, append=append, text=text, revised=revised,
                         audio_end_ms=self.processed_ms)]

    def _final(self, v):
        pieces, _ = self._pending()
        text = _join([t for t, _ in pieces])
        conf = token_conf([p for _, pairs in pieces for p in pairs]) if text else 0.0
        # 지금까지의 글자는 모두 이 구간 몫이다. 확정 전 글자는 나중에 확정되어 다시 오므로 기억해 두고 뺀다
        self.fins = []
        self.sent_partial = "" if v.is_last else self.partial
        return self.msg("segment_final", segment_id=v.seg_id, text=text, start_ms=v.start_ms,
                        end_ms=v.speech_end_ms, reason=v.reason, confidence=conf,
                        no_speech_prob=0.0 if text else 1.0, is_last=v.is_last)
