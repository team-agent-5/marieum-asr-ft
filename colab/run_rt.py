"""Colab L4 VM: 실시간 지표 세 가지를 차례로 잰다. 하나가 실패해도 다음 것은 돈다 (11-17).

    python /content/job.py start run_rt.py   (그다음 python /content/job.py wait)

  1. yoona-J, 운영 서버 방식 (쉼 1.75초마다 final, 0.5초마다 interim)  -> out/rt_yoonaj
  2. ft_v3, 0.32초 조각, 4세션 묶음                                    -> out/rt_ft_v3_320ms
  3. ft_v3, 1.12초 조각, 4세션 묶음                                    -> out/rt_ft_v3_1120ms

녹음마다 앞 3분 (30개, 90분). 결과(숫자만)는 streaming_metrics.py가 Drive runs/eval/에도 복사한다.
전제: Drive 연결, /content/upload/test_v2.tar.part-*, /content/{streaming_metrics,marieum_stream,zeroshot_nemotron,job}.py
"""

import os
import subprocess
import time

FT3 = "/content/drive/MyDrive/cdy-hack-ag5/marieum-asr-ft/ckpt/ft_v3/v1/checkpoints/ft_v3.nemo"
RUNS = [
    ("yoonaj", {"MODE": "whisper", "OUT_NAME": "rt_yoonaj"}),
    ("ft_v3_320ms", {"MODE": "nemo", "MODEL": FT3, "ATT_CONTEXT": "[56,3]", "OUT_NAME": "rt_ft_v3_320ms"}),
    ("ft_v3_1120ms", {"MODE": "nemo", "MODEL": FT3, "ATT_CONTEXT": "[56,13]", "OUT_NAME": "rt_ft_v3_1120ms"}),
]

if not os.path.exists(FT3):
    raise SystemExit(f"{FT3}가 없다. colab drivemount를 먼저 한다 (사용자)")
for name, env in RUNS:
    t = time.time()
    p = subprocess.run(["python", "/content/streaming_metrics.py"], env={**os.environ, **env})
    print(f"[{name}] exit {p.returncode}, {(time.time() - t) / 60:.0f}분", flush=True)
