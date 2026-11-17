"""Colab GPU VM: 학습 녹음을 말소리 구간으로 자르고 yoona-J로 받아쓴다 (문장 나누기 1단계).

정답은 VM에 없다. 여기서는 "어디가 말소리이고, 대충 뭐라고 했는지"만 만든다.
이 PC의 data_prep/align_segments.py가 이 결과를 녹음 전체 정답과 정렬해 구간마다 정답 문장을 붙인다.

입력  /content/train_audio/<speaker>/<stem>.flac   (Drive data/의 tar를 푼 것, docs/colab_runbook.md)
출력  {OUT}/<speaker>.jsonl  줄마다 {"speaker","stem","start","end","hyp"}  (화자 단위로 이어서 할 수 있다)
      {OUT}/progress.txt     처리한 화자 수

긴 작업이라 colab exec로 직접 돌리지 않고 VM 안에서 백그라운드로 띄운다 (30분 넘으면 CLI가 타임아웃된다):
    nohup python /content/segment_transcribe.py > {OUT}/run.log 2>&1 &
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

AUDIO = Path(os.environ.get("AUDIO", "/content/train_audio"))
OUT = Path(os.environ.get("OUT", "/content/drive/MyDrive/cdy-hack-ag5/marieum-asr-ft/runs/segments"))
MODEL = "yoona-J/ASR_Whisper_Disease_General"
SR = 16000
MAX_SEG_S = 20.0  # 구간 최대 길이
MIN_SEG_S = 0.8
MERGE_GAP_S = 0.6  # 이보다 짧은 쉼은 한 구간으로 (음절끊김 화자 대비 넉넉히)
PAD_S = 0.2

try:
    import silero_vad  # noqa: F401
    import transformers  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "silero-vad", "transformers", "accelerate",
                    "soundfile"], check=True)

import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad
from transformers import pipeline


def segments(wav, vad):
    """VAD 구간을 이어 붙이고 MAX_SEG_S로 나눈다. (시작초, 끝초) 목록."""
    ts = get_speech_timestamps(torch.from_numpy(wav), vad, sampling_rate=SR, min_silence_duration_ms=300,
                               speech_pad_ms=int(PAD_S * 1000))
    spans = [(t["start"] / SR, t["end"] / SR) for t in ts]
    merged = []
    for s, e in spans:
        if merged and s - merged[-1][1] < MERGE_GAP_S and e - merged[-1][0] <= MAX_SEG_S:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    out = []
    for s, e in merged:  # VAD 한 구간이 20초를 넘으면 균등하게 나눈다
        n = int(np.ceil((e - s) / MAX_SEG_S))
        step = (e - s) / n
        out += [(s + i * step, s + (i + 1) * step) for i in range(n)]
    return [(round(s, 2), round(e, 2)) for s, e in out if e - s >= MIN_SEG_S]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vad = load_silero_vad()
    asr = pipeline("automatic-speech-recognition", model=MODEL, torch_dtype=torch.float16, device=0, batch_size=16)
    gen = {"language": "ko", "task": "transcribe", "num_beams": 1}
    speakers = sorted(p.name for p in AUDIO.iterdir() if p.is_dir())
    t0 = time.time()
    for k, spk in enumerate(speakers, 1):
        dst = OUT / f"{spk}.jsonl"
        if dst.exists():
            continue
        rows = []
        for flac in sorted((AUDIO / spk).glob("*.flac")):
            wav, sr = sf.read(flac, dtype="float32")
            assert sr == SR, f"{flac}: {sr}Hz"
            segs = segments(wav, vad)
            clips = [{"raw": wav[int(s * SR):int(e * SR)], "sampling_rate": SR} for s, e in segs]
            hyps = asr(clips, generate_kwargs=gen) if clips else []
            rows += [{"speaker": spk, "stem": flac.stem, "start": s, "end": e, "hyp": h["text"].strip()}
                     for (s, e), h in zip(segs, hyps)]
        tmp = dst.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.rename(dst)  # 화자 하나가 끝까지 끝났을 때만 결과로 친다
        speech_min = sum(r["end"] - r["start"] for r in rows) / 60
        msg = f"{k}/{len(speakers)} {spk} 구간 {len(rows)}개 말소리 {speech_min:.1f}분 경과 {(time.time() - t0) / 60:.0f}분"
        print(msg, flush=True)
        (OUT / "progress.txt").write_text(msg + "\n", encoding="utf-8")
    (OUT / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    print("완료", flush=True)


if __name__ == "__main__":
    main()
