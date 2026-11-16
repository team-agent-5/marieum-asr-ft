"""Colab GPU VM: 0단계 실험 세 개를 차례로 돌린다. 하나가 실패해도 다음 것은 돈다.

    python /content/run_stage0.py

  B  ft_v1을 쉼마다 끊어서 (eval_segmented.py, MODE=nemo)       -> out/ft_v1_vad_320ms
  A  yoona-J를 운영 서버처럼 쉼마다 끊어서 (MODE=whisper)      -> out/yoonaj_vad
  C  ft_v1을 녹음 전체로, 지연 1.12초 (zeroshot_nemotron.py)    -> out/ft_v1_1120ms

B가 NeMo 설치와 구간 나누기를 먼저 하므로 맨 앞에 둔다. 결과는 각 스크립트가 Drive runs/eval/에도 복사한다.
전제: /content/upload/test_v2.tar.part-*, /content/eval_segmented.py, /content/zeroshot_nemotron.py, Drive 연결
"""

import os
import subprocess
import time

FT = "/content/drive/MyDrive/marieum-asr-ft/ckpt/ft_v1/v1/checkpoints/ft_v1.nemo"
RUNS = [
    ("B", "eval_segmented.py", {"MODE": "nemo", "MODEL": FT, "OUT_NAME": "ft_v1_vad_320ms"}),
    ("A", "eval_segmented.py", {"MODE": "whisper", "OUT_NAME": "yoonaj_vad"}),
    ("C", "zeroshot_nemotron.py", {"MODEL": FT, "ATT_CONTEXT": "[56,13]", "OUT_NAME": "ft_v1_1120ms"}),
]

if not os.path.exists(FT):
    raise SystemExit(f"{FT}가 없다. colab drivemount를 먼저 한다 (사용자)")
for name, script, env in RUNS:
    t = time.time()
    p = subprocess.run(["python", f"/content/{script}"], env={**os.environ, **env})
    print(f"[{name}] exit {p.returncode}, {(time.time() - t) / 60:.0f}분", flush=True)
