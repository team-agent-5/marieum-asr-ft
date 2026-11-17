"""Colab A100 VM: ft_v3 학습과 평가를 쉬지 않고 이어서 한다 (VM을 놀리지 않게).

    python /content/run_ft_v3.py

1. 학습: ft_v2에서 이어서 2000 step, 학습률 1.5e-5. 학습 목록 v5 (앞뒤 쉼 최대 1.5초, 이어 붙이기 쉼 10초·확률 90%·최대 20초).
   검증은 ft_v1·ft_v2와 같은 val_v3
2. 평가: 지연 0.32초 (ft_v2 EXP-008과 비교) -> out/ft_v3_320ms
3. 평가: 지연 1.12초 (ft_v2 EXP-009와 비교) -> out/ft_v3_1120ms

ft_v2 모델(ckpt/ft_v2/)은 건드리지 않는다. 다시 돌리면 ckpt/ft_v3/의 마지막 체크포인트에서 이어 간다. ft_v3는 ckpt/ft_v3/에 따로 저장된다.
학습이 실패하면 평가는 하지 않는다. 평가 결과는 zeroshot_nemotron.py가 Drive runs/eval/에도 복사한다.
전제: Drive 연결, Drive data/manifests/train_v5.jsonl·val_v3.jsonl, /content/upload/test_v2.tar.part-*,
      /content/train_nemotron.py, /content/zeroshot_nemotron.py
"""

import os
import subprocess
import time

DRIVE = "/content/drive/MyDrive/cdy-hack-ag5/marieum-asr-ft"
FT2 = f"{DRIVE}/ckpt/ft_v2/v1/checkpoints/ft_v2.nemo"
TRAIN = {"BASE_MODEL": FT2, "RUN_NAME": "ft_v3", "MAX_STEPS": "2000", "LR": "1.5e-5", "WARMUP": "100",
         "TRAIN_MANIFEST": "train_v5.jsonl", "VAL_MANIFEST": "val_v3.jsonl", "MAX_DURATION": "25"}


def run(name, script, env):
    t = time.time()
    p = subprocess.run(["python", f"/content/{script}"], env={**os.environ, **env})
    print(f"[{name}] exit {p.returncode}, {(time.time() - t) / 60:.0f}분", flush=True)
    return p.returncode


for need in (FT2, f"{DRIVE}/data/manifests/train_v5.jsonl"):
    if not os.path.exists(need):
        raise SystemExit(f"{need}가 없다. drivemount와 train_v5.jsonl 업로드를 먼저 한다 (사용자)")
if run("train", "train_nemotron.py", TRAIN) != 0:
    raise SystemExit("학습 실패. 평가는 하지 않는다")
# 학습이 끝날 때 마지막 모델이 ft_v3.nemo로 저장된다 (예전 것은 ft_v3-v1.nemo처럼 이름이 바뀌어 남는다)
FT3 = f"{DRIVE}/ckpt/ft_v3/v1/checkpoints/ft_v3.nemo"
if not os.path.exists(FT3):
    raise SystemExit("ft_v3.nemo가 없다")
for name, ctx in (("ft_v3_320ms", "[56,3]"), ("ft_v3_1120ms", "[56,13]")):
    run(name, "zeroshot_nemotron.py", {"MODEL": FT3, "ATT_CONTEXT": ctx, "OUT_NAME": name})
