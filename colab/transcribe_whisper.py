"""Colab GPU VM에서 기존 모델 yoona-J(Whisper-small)로 test_v2를 전사한다. 같은 테스트로 비교하기 위한 기준선.

    colab --auth=oauth2 exec -s nemo -f colab/transcribe_whisper.py

- 30초 단위로 잘라 한 번에 전사하는 오프라인 방식이다. 운영 서버(0.5초마다 다시 전사)와 다르다.
  운영 방식 수치(EXP-000, CER 37.6%)는 test_v1 기준이고, 이건 test_v2 30명을 같은 조건으로 비교하는 용도다
- 전제: zeroshot_nemotron.py와 같은 /content/test_v2 (없으면 그 스크립트의 unpack 단계를 먼저 돌린다)
- 출력: /content/out/whisper_yoonaj/hyps.jsonl
"""

import json
import subprocess
import sys
import time
from pathlib import Path

MODEL = "yoona-J/ASR_Whisper_Disease_General"
DATA = Path("/content/test_v2")
OUT = Path("/content/out/whisper_yoonaj")

try:
    import transformers  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "accelerate"], check=True)

import torch
from transformers import pipeline

if not (DATA / "manifest.jsonl").exists():
    raise SystemExit(f"{DATA}/manifest.jsonl 이 없다. zeroshot_nemotron.py를 먼저 돌려 압축을 푼다")

OUT.mkdir(parents=True, exist_ok=True)
asr = pipeline("automatic-speech-recognition", model=MODEL, torch_dtype=torch.float16, device=0,
               chunk_length_s=30, batch_size=8)
rows = [json.loads(l) for l in open(DATA / "manifest.jsonl", encoding="utf-8")]
t0 = time.time()
with open(OUT / "hyps.jsonl", "w", encoding="utf-8") as f:
    for i, r in enumerate(rows, 1):
        out = asr(str(DATA / r["audio_filepath"]), generate_kwargs={"language": "ko", "task": "transcribe"})
        f.write(json.dumps({"speaker": r["speaker"], "text": out["text"]}, ensure_ascii=False) + "\n")
        print(f"{i}/{len(rows)} {r['speaker']}", flush=True)
audio_s = sum(r["duration"] for r in rows)
info = {"model": MODEL, "mode": "offline chunk30s", "files": len(rows), "audio_min": round(audio_s / 60, 1),
        "infer_s": round(time.time() - t0), "rtf": round((time.time() - t0) / audio_s, 4)}
json.dump(info, open(OUT / "run_info.json", "w"), indent=1)
print(json.dumps(info, ensure_ascii=False))
