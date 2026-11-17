"""Colab GPU VM: 실시간 지표를 잰다 (11-17). 글자가 화면에 언제 뜨고, 한번 뜬 글자가 얼마나 바뀌는지.

    MODE=whisper OUT_NAME=rt_yoonaj                                  python streaming_metrics.py
    MODE=nemo MODEL=.../ft_v3.nemo ATT_CONTEXT=[56,3] OUT_NAME=rt_ft_v3_320ms python streaming_metrics.py

  whisper  yoona-J를 운영 서버(marieum-ai) 방식 그대로 흉내 낸다. 쉼 1.75초마다 구간을 닫고 final을 낸다.
           구간 안에서는 0.5초마다 지금까지의 구간 전체를 다시 받아써서 interim을 내고, 두 번 연달아 같은 앞부분만 확정한다
           (LocalAgreement-2). 구간 나누기·확정 코드는 운영 코드를 그대로 복사한 marieum_stream.py를 쓴다
  nemo     Nemotron cache-aware 스트리밍. 0.32초(또는 1.12초) 조각마다 글자를 덧붙인다.
           STREAMS개 녹음을 한 번에 돌린다 (동시 세션 STREAMS개와 같은 GPU 부담)

공통
- 녹음마다 앞 CLIP_S초만 쓴다. 말 구간(시작·끝)은 운영 Segmenter로 한 번 구해 두 모드가 같이 쓴다 (rt_segments.json)
- 시간: 소리가 실제 속도로 들어온다고 보고, 계산이 끝난 시각에 화면에 뜬다고 본다. 계산 시간은 GPU에서 실제로 잰 값이다.
  앞 계산이 안 끝났으면 기다린다 (밀림이 반영된다). 네트워크와 앱은 빠져 있다
- whisper는 세션 1개를 혼자 돌린다 (운영은 4세션을 묶어 돌려 더 느릴 수 있다). nemo는 STREAMS개를 묶는다

구간마다 재는 것
  first_ms  말 시작 → 그 구간 첫 글자가 화면에 뜨기까지
  done_ms   말 끝 → 그 구간 글자가 다 뜨기까지 (whisper: final 도착, nemo: 그 구간에서 마지막으로 글자가 붙은 때).
            쉼으로 닫힌 구간만 센다
  erased    화면에서 지워지거나 바뀐 글자 수 (앞서 보인 글자와 달라진 만큼, 공백·문장부호 빼고)
  changes   글자가 지워진 갱신 횟수
출력  /content/out/<OUT_NAME>/segments.csv, summary.json (숫자만, 문장은 저장하지 않는다). Drive runs/eval/<OUT_NAME>/에도 복사
"""

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zeroshot_nemotron as Z  # noqa: E402  unpack(), install(), 경로
from marieum_stream import CONVERSATION, SR, WIN, Agreement, Segmenter  # noqa: E402

MODE = os.environ.get("MODE", "nemo")
MODEL = os.path.expanduser(os.environ.get(
    "MODEL", "yoona-J/ASR_Whisper_Disease_General" if MODE == "whisper" else "nvidia/nemotron-3.5-asr-streaming-0.6b"))
ATT_CONTEXT = os.environ.get("ATT_CONTEXT", "[56,3]")
STREAMS = int(os.environ.get("STREAMS", "4"))
CLIP_S = float(os.environ.get("CLIP_S", "180"))
OUT = Z.WORK / "out" / os.environ.get("OUT_NAME", f"rt_{MODE}")
SEGS = Z.WORK / "out" / "rt_segments.json"
WIN_MS = WIN * 1000 / SR  # 32ms


class Cfg:
    """marieum-ai server/config.py 기본값 (commit 2ccebf6)."""
    vad_threshold = 0.5
    speech_pad_ms = 200
    min_speech_ms = 150
    end_silence_ms = 1750
    utterance_max_s = 10.0
    max_segment_s = 15.0
    session_max_s = 600.0
    interim_step_ms = 500
    min_interim_speech_ms = 300
    interim_max_new_tokens = 96
    final_max_new_tokens = 128
    token_cap_rate = 14.0
    token_cap_floor = 8
    no_speech_threshold = 0.6
    min_interim_gap_s = 0.3  # app.py MIN_INTERIM_GAP_S


def token_cap(hard, audio_ms):  # scheduler.py token_cap
    return min(hard, math.ceil(audio_ms * Cfg.token_cap_rate / 1000) + Cfg.token_cap_floor)


_DROP = re.compile(r"[\s.,?!~·…]")


def norm(t):
    return _DROP.sub("", t)


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def load_clips():
    import soundfile as sf
    rows = [json.loads(l) for l in open(Z.DATA / "manifest.jsonl", encoding="utf-8")]
    clips = []
    for r in rows:
        wav, sr = sf.read(Z.DATA / r["audio_filepath"], dtype="float32", frames=int(CLIP_S * SR))
        assert sr == SR
        clips.append((r["speaker"], wav))
    return clips


# ---- 구간: 운영 Segmenter (대화 모드) -------------------------------------------------
def vad_probs(model, wav):
    import torch
    n = len(wav) // WIN
    model.reset_states()
    with torch.no_grad():
        return [float(model(torch.from_numpy(np.ascontiguousarray(wav[i * WIN:(i + 1) * WIN])), SR)) for i in range(n)]


def run_segmenter(probs, wav, on_window=None):
    """창(32ms)마다 Segmenter에 넣는다. on_window(k, seg, closed)를 창마다 부른다 (whisper 흉내에 쓴다)."""
    seg = Segmenter(CONVERSATION, None, Cfg)
    out = []
    for k, p in enumerate(probs):
        closed = seg.push_windows([wav[k * WIN:(k + 1) * WIN]], [p])
        if on_window:
            on_window(k, seg, closed)
        out += closed
    last = seg.finish()
    if on_window and last is not None:
        on_window(len(probs), seg, [last])
    if last is not None:
        out.append(last)
    return out


def describe(segments, probs):
    """Segment마다 말 시작·끝(첫·마지막 말소리 창)과 닫힌 시각."""
    res = []
    for s in segments:
        if not s.has_speech:
            continue
        a = int(round((s.end_ms - s.duration_ms) / WIN_MS))
        b = int(round(s.end_ms / WIN_MS))
        idx = [i for i in range(max(0, a), min(b, len(probs))) if probs[i] >= Cfg.vad_threshold]
        if not idx:
            continue
        res.append({"seg_id": s.seg_id, "start_ms": round(idx[0] * WIN_MS), "end_ms": round((idx[-1] + 1) * WIN_MS),
                    "close_ms": s.end_ms, "reason": s.reason})
    return res


def make_segments(clips):
    if SEGS.exists():
        return json.load(open(SEGS))
    try:
        import silero_vad  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "silero-vad"], check=True)
    from silero_vad import load_silero_vad
    vad = load_silero_vad()
    out = {}
    for spk, wav in clips:
        probs = vad_probs(vad, wav)
        out[spk] = {"probs": [round(p, 3) for p in probs], "segments": describe(run_segmenter(probs, wav), probs)}
    SEGS.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(SEGS, "w"))
    return out


# ---- 화면 기록 → 지표 -------------------------------------------------------------
def erasure(texts):
    """화면 글자 목록이 차례로 바뀔 때 지워진 글자 수와 지워진 갱신 횟수."""
    erased = changes = 0
    prev = ""
    for t in texts:
        t = norm(t)
        e = len(prev) - lcp(prev, t)
        if e > 0:
            erased += e
            changes += 1
        prev = t
    return erased, changes


# ---- whisper: 운영 서버 흉내 ---------------------------------------------------------
class Whisper:
    def __init__(self):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch = torch
        self.proc = WhisperProcessor.from_pretrained(MODEL)
        self.model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
        tok = self.proc.tokenizer
        self.sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
        self.nsp_id = next(i for i in (tok.convert_tokens_to_ids(t) for t in ("<|nospeech|>", "<|nocaptions|>"))
                           if i is not None and i != tok.unk_token_id)

    def __call__(self, audio, hard):
        """운영 engine.py transcribe()와 같은 계산 (한 줄). (문장, 걸린 초)."""
        torch = self.torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            feats = self.proc([audio], sampling_rate=SR, return_tensors="pt").input_features.to("cuda", torch.float16)
            enc = self.model.model.encoder(feats)
            dec = self.model.model.decoder(input_ids=torch.full((1, 1), self.sot, device="cuda"),
                                           encoder_hidden_states=enc.last_hidden_state)
            nsp = torch.softmax(self.model.proj_out(dec.last_hidden_state[:, -1]).float(), -1)[0, self.nsp_id].item()
            # 운영은 forced_decoder_ids(ko, transcribe). 최신 transformers에서는 language/task가 같은 프롬프트다
            from transformers.modeling_outputs import BaseModelOutput
            out = self.model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=enc.last_hidden_state),
                                      language="ko", task="transcribe", num_beams=1, do_sample=False,
                                      max_new_tokens=token_cap(hard, len(audio) * 1000 // SR))
            torch.cuda.synchronize()
        text = self.proc.batch_decode(out, skip_special_tokens=True)[0].strip()
        # 운영: no_speech_prob > 0.6 이고 avg_logprob <= 0.0 이면 빈 전사. logprob 조건은 항상 참이라 no_speech만 본다
        return ("" if nsp > Cfg.no_speech_threshold else text), time.perf_counter() - t0


def simulate_whisper(asr, probs, wav, segs_info):
    """창마다 운영 서버의 interim/final 규칙을 따라간다. 구간별 화면 기록 [(시각 ms, 글자, 종류)]과 계산 시간들."""
    S = {"gpu_free": 0.0, "inflight": None, "last_sent": -1e9, "agree": {}, "final_done": set()}
    disp = {}  # seg_id -> [(ms, text, kind)]
    calls = {"interim": [], "final": []}

    def deliver(now):
        f = S["inflight"]
        if f and f[0] <= now:
            S["inflight"] = None
            ready, sid, text = f
            if sid not in S["final_done"]:  # 이미 final을 낸 구간의 interim은 버린다 (운영과 같다)
                ag = S["agree"].setdefault(sid, Agreement())
                stable, unstable = ag.update(text)
                disp.setdefault(sid, []).append((ready, " ".join(t for t in (stable, unstable) if t), "interim"))
            S["last_sent"] = ready

    def on_window(k, seg, closed):
        now = (k + 1) * WIN_MS
        deliver(now)
        for s in closed:
            S["final_done"].add(s.seg_id)
            if S["inflight"] and S["inflight"][1] == s.seg_id:
                S["inflight"] = None  # cancel_interim
            if s.has_speech:
                text, c = asr(s.audio, Cfg.final_max_new_tokens)
                calls["final"].append(c)
                start = max(now, S["gpu_free"])
                S["gpu_free"] = start + c * 1000
                disp.setdefault(s.seg_id, []).append((S["gpu_free"], text, "final"))
        if (S["inflight"] is None and seg.interim_ready()
                and now - S["last_sent"] >= Cfg.min_interim_gap_s * 1000):
            snap = seg.snapshot()
            text, c = asr(snap.audio, Cfg.interim_max_new_tokens)
            calls["interim"].append(c)
            start = max(now, S["gpu_free"])
            S["gpu_free"] = start + c * 1000
            S["inflight"] = (S["gpu_free"], snap.seg_id, text)

    run_segmenter(probs, wav, on_window)
    rows = []
    for info in segs_info:
        d = sorted(disp.get(info["seg_id"], []))
        shown = [x for x in d if norm(x[1])]
        final = [x for x in d if x[2] == "final"]
        erased, changes = erasure([x[1] for x in d])
        rows.append({"first_ms": round(shown[0][0] - info["start_ms"]) if shown else "",
                     "done_ms": round(final[-1][0] - info["end_ms"]) if final and info["reason"] == "silence" else "",
                     "final_chars": len(norm(final[-1][1])) if final else 0, "erased": erased, "changes": changes,
                     "updates": len(d)})
    return rows, calls


# ---- nemo: cache-aware 스트리밍 -------------------------------------------------------
def load_nemo():
    import torch
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
    m = (nemo_asr.models.ASRModel.restore_from(MODEL, map_location="cuda") if MODEL.endswith(".nemo")
         else nemo_asr.models.ASRModel.from_pretrained(MODEL, map_location="cuda"))
    m.encoder.set_default_att_context_size(att_context_size=json.loads(ATT_CONTEXT))
    # 평가 스크립트(speech_to_text_cache_aware_streaming_infer.py, decoder_type=rnnt)와 같은 순서·설정
    dec = RNNTDecodingConfig(fused_batch_size=-1)
    if hasattr(m, "cur_decoder"):
        m.change_decoding_strategy(dec, decoder_type="rnnt")
    else:
        m.change_decoding_strategy(dec)
    if hasattr(m, "set_inference_prompt"):
        m.set_inference_prompt("ko-KR")
        m.decoding.set_strip_lang_tags(True)
    m = m.to("cuda").eval()
    torch.set_grad_enabled(False)
    return m


def chunk_size_at(buf):
    cs = buf.streaming_cfg.chunk_size
    if isinstance(cs, list):
        return cs[1] if (buf.buffer_idx > 0 or buf.pad_and_drop_preencoded) else cs[0]
    return cs


def stream_batch(m, paths, amp_dtype):
    """paths를 한 번에 스트리밍. 조각마다 (소리 끝 ms, 계산 초, [녹음별 글자])."""
    import torch
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    buf = CacheAwareStreamingAudioBuffer(model=m, online_normalization=False, pad_and_drop_preencoded=True)
    for p in paths:
        buf.append_audio_file(str(p), stream_id=-1)
    stride_ms = float(m.cfg.preprocessor.window_stride) * 1000
    cache = m.encoder.get_initial_cache_state(batch_size=len(paths))
    prev_hyp, prev_out, steps = None, None, []
    it = iter(buf)
    k = 0
    with torch.autocast("cuda", dtype=amp_dtype):
        while True:
            end_frame = min(buf.buffer_idx + chunk_size_at(buf), buf.buffer.size(-1))
            try:
                chunk, lens = next(it)
            except StopIteration:
                break
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                prev_out, hyps, c0, c1, c2, prev_hyp = m.conformer_stream_step(
                    processed_signal=chunk, processed_signal_length=lens,
                    cache_last_channel=cache[0], cache_last_time=cache[1], cache_last_channel_len=cache[2],
                    keep_all_outputs=buf.is_buffer_empty(), previous_hypotheses=prev_hyp, previous_pred_out=prev_out,
                    drop_extra_pre_encoded=m.encoder.streaming_cfg.drop_extra_pre_encoded,
                    return_transcription=True)
            torch.cuda.synchronize()
            cache = (c0, c1, c2)
            steps.append((end_frame * stride_ms, time.perf_counter() - t0,
                          [h.text if hasattr(h, "text") else str(h) for h in hyps]))
            k += 1
    return steps


def nemo_rows(steps, j, segs_info, clip_ms):
    """녹음 j의 화면 기록을 실제 속도 기준 시각으로 바꾸고 구간별 지표를 낸다."""
    shown, finish = [], 0.0
    for audio_end, c, texts in steps:
        finish = max(audio_end, finish) + c * 1000  # 이 조각 소리가 다 들어와야 시작, 앞 계산이 끝나야 시작
        shown.append((audio_end, finish, norm(texts[j])))
    rows = []
    for i, info in enumerate(segs_info):
        nxt = segs_info[i + 1]["start_ms"] if i + 1 < len(segs_info) else clip_ms + 10_000
        ev = [(a, f, t) for a, f, t in shown if info["start_ms"] < a <= nxt]
        before = [t for a, f, t in shown if a <= info["start_ms"]]
        base = before[-1] if before else ""
        grow, prev = [], base
        for a, f, t in ev:
            if len(t) > len(prev) or t != prev:
                grow.append((a, f, t))
            prev = t
        texts = [base] + [t for _, _, t in ev]
        erased, changes = erasure(texts)
        final_chars = (len(ev[-1][2]) - len(base)) if ev else 0
        rows.append({"first_ms": round(grow[0][1] - info["start_ms"]) if grow else "",
                     "done_ms": round(grow[-1][1] - info["end_ms"]) if grow and info["reason"] == "silence" else "",
                     "final_chars": final_chars, "erased": erased, "changes": changes, "updates": len(grow)})
    return rows


# ---- 요약 ------------------------------------------------------------------------
def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(math.ceil(q / 100 * len(xs))) - 1)]) if xs else None


def summarize(rows, extra):
    first = [r["first_ms"] for r in rows if r["first_ms"] != ""]
    done = [r["done_ms"] for r in rows if r["done_ms"] != ""]
    chars = sum(r["final_chars"] for r in rows)
    with_text = [r for r in rows if r["final_chars"] > 0]
    return {"mode": MODE, "model": MODEL, "att_context_size": ATT_CONTEXT if MODE == "nemo" else None,
            "clip_s": CLIP_S, "segments": len(rows), "segments_with_text": len(with_text),
            "first_ms_p50": pct(first, 50), "first_ms_p95": pct(first, 95),
            "done_ms_p50": pct(done, 50), "done_ms_p95": pct(done, 95), "done_n": len(done),
            "final_chars": chars, "erased_chars": sum(r["erased"] for r in rows),
            "erased_per_100_chars": round(100 * sum(r["erased"] for r in rows) / max(1, chars), 1),
            "segments_with_change_pct": round(100 * sum(r["changes"] > 0 for r in with_text) / max(1, len(with_text)), 1),
            **extra}


def main():
    if MODE == "nemo":
        try:
            import nemo.collections.asr  # noqa: F401
        except ImportError:
            Z.install()  # 설치한 프로세스에서는 바로 import가 안 될 수 있어 새로 띄운다
            os.execv(sys.executable, [sys.executable] + sys.argv)
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    Z.unpack()
    clips = load_clips()
    segs = make_segments(clips)
    print(f"구간 준비 {sum(len(v['segments']) for v in segs.values())}개 ({(time.time() - t0) / 60:.0f}분)", flush=True)

    rows = []
    if MODE == "whisper":
        asr = Whisper()
        asr(np.zeros(SR, np.float32), 8)  # 워밍업
        calls = {"interim": [], "final": []}
        for spk, wav in clips:
            r, c = simulate_whisper(asr, segs[spk]["probs"], wav, segs[spk]["segments"])
            rows += [{"speaker": spk, **x} for x in r]
            for k in calls:
                calls[k] += c[k]
        extra = {"interim_calls": len(calls["interim"]), "final_calls": len(calls["final"]),
                 "interim_gpu_ms_p50": pct([c * 1000 for c in calls["interim"]], 50),
                 "interim_gpu_ms_p95": pct([c * 1000 for c in calls["interim"]], 95),
                 "final_gpu_ms_p50": pct([c * 1000 for c in calls["final"]], 50),
                 "final_gpu_ms_p95": pct([c * 1000 for c in calls["final"]], 95), "streams": 1}
    else:
        import soundfile as sf
        import torch
        amp = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        m = load_nemo()
        cdir = Z.WORK / "rt_clips"
        cdir.mkdir(exist_ok=True)
        paths = []
        for spk, wav in clips:
            p = cdir / f"{spk}.wav"
            sf.write(p, wav, SR)
            paths.append(p)
        stream_batch(m, paths[:1], amp)  # 워밍업
        step_ms, behind = [], 0
        for b in range(0, len(clips), STREAMS):
            group = list(range(b, min(b + STREAMS, len(clips))))
            steps = stream_batch(m, [paths[i] for i in group], amp)
            chunk_ms = float(np.median(np.diff([e for e, _, _ in steps]))) if len(steps) > 2 else 0.0
            step_ms += [c * 1000 for _, c, _ in steps]
            behind += sum(c * 1000 > chunk_ms for _, c, _ in steps[1:])
            for j, i in enumerate(group):
                spk, wav = clips[i]
                rows += [{"speaker": spk, **x} for x in nemo_rows(steps, j, segs[spk]["segments"], len(wav) * 1000 / SR)]
        extra = {"streams": STREAMS, "chunk_ms": round(chunk_ms), "step_ms_p50": pct(step_ms, 50),
                 "step_ms_p95": pct(step_ms, 95), "steps": len(step_ms), "steps_slower_than_chunk": behind,
                 "amp_dtype": str(amp).replace("torch.", ""), "preprocessor_normalize": str(m.cfg.preprocessor.get("normalize"))}
        shutil.rmtree(cdir, ignore_errors=True)

    info = summarize(rows, {**extra, "total_s": round(time.time() - t0)})
    with open(OUT / "segments.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(info, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(info, ensure_ascii=False), flush=True)
    drive = Path("/content/drive/MyDrive/cdy-hack-ag5/marieum-asr-ft")
    if drive.exists():
        keep = drive / "runs/eval" / OUT.name
        keep.mkdir(parents=True, exist_ok=True)
        for name in ("segments.csv", "summary.json"):
            shutil.copy(OUT / name, keep / name)
        print(f"Drive에도 저장: {keep}", flush=True)


if __name__ == "__main__":
    main()
