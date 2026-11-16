"""Colab A100 VM: ft_v2 학습과 평가를 쉬지 않고 이어서 한다 (VM을 놀리지 않게).

    python /content/run_ft_v2.py

1. 학습: ft_v1에서 이어서 2000 step. 학습 목록 v4 (앞뒤 여유 + 이어 붙이기 확대). 검증은 ft_v1과 같은 val_v3
2. 평가: 지연 1.12초 (ft_v1 1.12초 EXP-007과 비교) -> out/ft_v2_1120ms
3. 평가: 지연 0.32초 (ft_v1 EXP-004와 비교)          -> out/ft_v2_320ms

학습이 실패하면 평가는 하지 않는다. 평가 결과는 zeroshot_nemotron.py가 Drive runs/eval/에도 복사한다.
전제: Drive 연결, Drive data/manifests/train_v4.jsonl·val_v3.jsonl, /content/upload/test_v2.tar.part-*,
      /content/train_nemotron.py, /content/zeroshot_nemotron.py
"""

import glob
import os
import subprocess
import time

DRIVE = "/content/drive/MyDrive/marieum-asr-ft"
FT1 = f"{DRIVE}/ckpt/ft_v1/v1/checkpoints/ft_v1.nemo"
TRAIN = {"BASE_MODEL": FT1, "RUN_NAME": "ft_v2", "MAX_STEPS": "2000", "LR": "2e-5", "WARMUP": "100",
         "TRAIN_MANIFEST": "train_v4.jsonl", "VAL_MANIFEST": "val_v3.jsonl", "MAX_DURATION": "25"}


def run(name, script, env):
    t = time.time()
    p = subprocess.run(["python", f"/content/{script}"], env={**os.environ, **env})
    print(f"[{name}] exit {p.returncode}, {(time.time() - t) / 60:.0f}분", flush=True)
    return p.returncode


if not os.path.exists(FT1):
    raise SystemExit(f"{FT1}가 없다. colab drivemount를 먼저 한다 (사용자)")
if run("train", "train_nemotron.py", TRAIN) != 0:
    raise SystemExit("학습 실패. 평가는 하지 않는다")
nemos = sorted(glob.glob(f"{DRIVE}/ckpt/ft_v2/v1/checkpoints/*.nemo"))
if not nemos:
    raise SystemExit("ft_v2 .nemo가 없다")
for name, ctx in (("ft_v2_1120ms", "[56,13]"), ("ft_v2_320ms", "[56,3]")):
    run(name, "zeroshot_nemotron.py", {"MODEL": nemos[-1], "ATT_CONTEXT": ctx, "OUT_NAME": name})
