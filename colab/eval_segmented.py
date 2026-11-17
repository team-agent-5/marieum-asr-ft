"""Colab GPU VM: test_v2를 운영 서버처럼 쉼마다 끊어서 받아쓴다 (0단계 실험 A·B).

운영 서버(marieum-ai 대화 모드)는 말소리 감지로 쉼(1.75초)마다 구간을 끊고, 구간마다 모델을 새로 돌린다.
zeroshot_nemotron.py는 녹음 전체를 한 번에 흘려 넣는다. 두 방식의 차이와 공정한 yoona-J 기준선을 본다.

    MODE=whisper OUT_NAME=yoonaj_vad                        python eval_segmented.py   # A: 운영 방식 yoona-J
    MODE=nemo MODEL=.../ft_v1.nemo OUT_NAME=ft_v1_vad_320ms python eval_segmented.py   # B: 같은 구간을 ft_v1로

구간 나누기는 marieum-ai server/config.py 기본값을 따른다 (Silero VAD 0.5, 앞뒤 200ms, 쉼 1750ms, 최대 15초).
운영 서버의 구간 경계와 똑같지는 않다 (서버는 32ms 창 단위 상태 기계, 여기는 silero get_speech_timestamps).
두 모드는 같은 구간 파일(segments.jsonl)을 쓴다. 먼저 돈 쪽이 만들고 뒤의 쪽이 읽는다.

입력  /content/upload/test_v2.tar.part-*  (zeroshot_nemotron.py와 같다)
출력  /content/out/<OUT_NAME>/hyps.jsonl      {"speaker", "text"}  구간 받아쓰기를 띄어 이은 것
      /content/out/<OUT_NAME>/seg_hyps.jsonl  {"speaker", "start", "end", "text"}
      /content/out/<OUT_NAME>/run_info.json
      Drive runs/eval/<OUT_NAME>/ 에도 복사 (Drive가 연결돼 있으면)
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zeroshot_nemotron as Z  # noqa: E402  unpack(), install(), 경로

MODE = os.environ.get("MODE", "nemo")
MODEL = os.path.expanduser(os.environ.get(
    "MODEL", "yoona-J/ASR_Whisper_Disease_General" if MODE == "whisper" else "nvidia/nemotron-3.5-asr-streaming-0.6b"))
ATT_CONTEXT = os.environ.get("ATT_CONTEXT", "[56,3]")
OUT = Z.WORK / "out" / os.environ.get("OUT_NAME", f"{MODE}_vad")
SEGS = Z.WORK / "out" / "vad_segments.jsonl"
CLIPS = Z.WORK / "vad_clips"
SR = 16000
# marieum-ai server/config.py 기본값 (2026-11-16)
VAD_THRESHOLD = 0.5
SPEECH_PAD_MS = 200
MIN_SPEECH_MS = 150
END_SILENCE_MS = int(os.environ.get("END_SILENCE_MS", "1750"))
MAX_SEGMENT_S = 15.0
MIN_CLIP_S = 0.5  # NeMo 스트리밍 추론이 아주 짧은 조각에서 실패하지 않게. 두 모드 모두 같은 기준으로 뺀다
NO_SPEECH_THRESHOLD = 0.6  # 운영: no_speech_prob > 0.6 이면 빈 전사 (logprob 조건은 꺼져 있다)
FINAL_MAX_NEW_TOKENS = 128


def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def make_segments(rows):
    """녹음마다 말소리 구간을 찾는다. 이미 있으면 그대로 쓴다 (A·B가 같은 구간을 쓰게)."""
    if SEGS.exists():
        return [json.loads(l) for l in open(SEGS, encoding="utf-8")]
    try:
        import silero_vad  # noqa: F401
    except ImportError:
        pip("silero-vad")
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    vad = load_silero_vad()
    segs = []
    for r in rows:
        wav, sr = sf.read(Z.DATA / r["audio_filepath"], dtype="float32")
        assert sr == SR, f"{r['audio_filepath']}: {sr}Hz"
        ts = get_speech_timestamps(torch.from_numpy(wav), vad, sampling_rate=SR, threshold=VAD_THRESHOLD,
                                   min_speech_duration_ms=MIN_SPEECH_MS, min_silence_duration_ms=END_SILENCE_MS,
                                   speech_pad_ms=SPEECH_PAD_MS, max_speech_duration_s=MAX_SEGMENT_S)
        for t in ts:
            s, e = round(t["start"] / SR, 3), round(t["end"] / SR, 3)
            if e - s >= MIN_CLIP_S:
                segs.append({"speaker": r["speaker"], "audio_filepath": r["audio_filepath"], "start": s, "end": e})
    SEGS.parent.mkdir(parents=True, exist_ok=True)
    with open(SEGS, "w", encoding="utf-8") as f:
        for s in segs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return segs


def load_clips(segs):
    import soundfile as sf
    cache, out = {}, []
    for s in segs:
        if s["audio_filepath"] not in cache:
            cache = {s["audio_filepath"]: sf.read(Z.DATA / s["audio_filepath"], dtype="float32")[0]}
        out.append(cache[s["audio_filepath"]][int(s["start"] * SR):int(s["end"] * SR)])
    return out


def run_whisper(segs):
    """운영 engine.py와 같은 방식: 강제 프롬프트(ko, transcribe), greedy, fp16, 무음 토큰 확률로 거르기."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
    tok = proc.tokenizer
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    nsp_id = next(i for i in (tok.convert_tokens_to_ids(t) for t in ("<|nospeech|>", "<|nocaptions|>"))
                  if i is not None and i != tok.unk_token_id)
    clips, texts = load_clips(segs), []
    with torch.inference_mode():
        for i in range(0, len(clips), 16):
            batch = clips[i:i + 16]
            feats = proc(batch, sampling_rate=SR, return_tensors="pt").input_features.to("cuda", torch.float16)
            enc = model.model.encoder(feats)
            dec = model.model.decoder(input_ids=torch.full((len(batch), 1), sot, device="cuda"),
                                      encoder_hidden_states=enc.last_hidden_state)
            nsp = torch.softmax(model.proj_out(dec.last_hidden_state[:, -1]).float(), -1)[:, nsp_id].tolist()
            # 운영은 forced_decoder_ids(ko, transcribe)를 쓴다. 최신 transformers에서는 language/task가 같은 프롬프트다
            out = model.generate(input_features=feats, language="ko", task="transcribe", num_beams=1, do_sample=False,
                                 max_new_tokens=FINAL_MAX_NEW_TOKENS)
            for t, p in zip(proc.batch_decode(out, skip_special_tokens=True), nsp):
                texts.append("" if p > NO_SPEECH_THRESHOLD else t.strip())
    return texts


def run_nemo(segs):
    """구간마다 따로 스트리밍 추론 (구간이 바뀌면 모델 상태를 새로 시작한다)."""
    import soundfile as sf
    man = OUT / "manifest_clips.jsonl"
    with open(man, "w", encoding="utf-8") as f:
        for k, (s, wav) in enumerate(zip(segs, load_clips(segs))):
            dst = CLIPS / s["speaker"] / f"{k:05d}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dst, wav, SR)
            f.write(json.dumps({"audio_filepath": str(dst), "duration": round(len(wav) / SR, 3), "text": "가",
                                "target_lang": "ko-KR"}, ensure_ascii=False) + "\n")
    script = Z.NEMO / "examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"
    import torch
    amp_dtype = os.environ.get("AMP_DTYPE") or ("bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16")
    for old in OUT.glob("streaming_out_*.json"):
        old.unlink()
    Z.sh([sys.executable, str(script),
          f"model_path={MODEL}" if MODEL.endswith(".nemo") else f"pretrained_name={MODEL}",
          f"dataset_manifest={man}", f"output_path={OUT}", f"att_context_size={ATT_CONTEXT}", "batch_size=32",
          "decoder_type=rnnt", "pad_and_drop_preencoded=true", "target_lang=ko-KR", "strip_lang_tags=true",
          "amp=true", f"amp_dtype={amp_dtype}", "cuda=0"], log=OUT / "run.log")
    out = sorted(OUT.glob("streaming_out_*.json"))[-1]
    texts = [json.loads(l)["pred_text"].strip() for l in open(out, encoding="utf-8")]
    out.unlink()  # 자리표시 정답이 섞인 파일
    shutil.rmtree(CLIPS, ignore_errors=True)
    return texts


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    Z.unpack()
    if MODE == "nemo":
        Z.install()
    else:
        try:
            import transformers  # noqa: F401
        except ImportError:
            pip("transformers", "accelerate")
    rows = [json.loads(l) for l in open(Z.DATA / "manifest.jsonl", encoding="utf-8")]
    segs = make_segments(rows)
    print(f"구간 {len(segs)}개, {sum(s['end'] - s['start'] for s in segs) / 60:.0f}분 ({(time.time() - t0) / 60:.0f}분)",
          flush=True)
    t1 = time.time()
    texts = run_whisper(segs) if MODE == "whisper" else run_nemo(segs)
    infer_s = time.time() - t1
    if len(texts) != len(segs):
        raise SystemExit(f"결과 {len(texts)}개 != 구간 {len(segs)}개")

    by = {r["speaker"]: [] for r in rows}
    with open(OUT / "seg_hyps.jsonl", "w", encoding="utf-8") as f:
        for s, t in zip(segs, texts):
            by[s["speaker"]].append(t)
            f.write(json.dumps({"speaker": s["speaker"], "start": s["start"], "end": s["end"], "text": t},
                               ensure_ascii=False) + "\n")
    with open(OUT / "hyps.jsonl", "w", encoding="utf-8") as f:
        for spk, ts in by.items():
            f.write(json.dumps({"speaker": spk, "text": " ".join(t for t in ts if t)}, ensure_ascii=False) + "\n")
    audio_min = sum(r["duration"] for r in rows) / 60
    info = {"mode": MODE, "model": MODEL, "att_context_size": ATT_CONTEXT if MODE == "nemo" else None,
            "end_silence_ms": END_SILENCE_MS, "max_segment_s": MAX_SEGMENT_S, "segments": len(segs),
            "empty_segments": sum(not t for t in texts), "files": len(rows), "audio_min": round(audio_min, 1),
            "infer_s": round(infer_s), "total_s": round(time.time() - t0)}
    json.dump(info, open(OUT / "run_info.json", "w"), indent=1)
    print(json.dumps(info, ensure_ascii=False), flush=True)
    drive = Path("/content/drive/MyDrive/cdy-hack-ag5/marieum-asr-ft")
    if drive.exists():
        keep = drive / "runs/eval" / OUT.name
        keep.mkdir(parents=True, exist_ok=True)
        for name in ("hyps.jsonl", "seg_hyps.jsonl", "run_info.json"):
            shutil.copy(OUT / name, keep / name)
        print(f"Drive에도 저장: {keep}", flush=True)


if __name__ == "__main__":
    main()
