"""Colab GPU VM: Nemotron 3.5 ASR Streaming 0.6B를 구음장애 데이터로 파인튜닝한다.

    colab --auth=oauth2 upload -s train colab/train_nemotron.py /content/train_nemotron.py
    echo 'import subprocess; p=subprocess.run("python /content/train_nemotron.py", shell=True); print("exit", p.returncode)' | colab --auth=oauth2 exec -s train

- 커널이 끝까지 기다리게 실행한다 (백그라운드로 띄우면 커널이 쉬어서 VM이 회수된다, 11-15)
- 설정은 환경변수로 바꾼다. 시험 학습: MAX_STEPS=1000, 본 학습: MAX_STEPS=3000 (기본). 중증은 3번 반복이라 더 길게 돌리면 외울 위험이 크다
- 체크포인트는 Drive에 저장하고, 같은 RUN_NAME으로 다시 돌리면 마지막 체크포인트에서 이어서 한다
- 근거: docs/nemotron_finetune_cases.md (인코더 전체 학습, bf16-mixed, 작은 학습률, 고정 step, target_lang)

전제 (docs/colab_runbook.md "파인튜닝")
  Drive marieum-asr-ft/data/train_audio.tar.part-*      학습 녹음 (문장 나누기 때 올린 것)
  Drive marieum-asr-ft/data/manifests/train_v1.jsonl, val_v1.jsonl   (build_train_manifests.py 출력, 브라우저로 올림)
출력
  Drive marieum-asr-ft/ckpt/<RUN_NAME>/v1/checkpoints/   *.nemo (가장 좋은 것), *.ckpt (이어하기용)
  Drive marieum-asr-ft/runs/train/<RUN_NAME>.log         학습 로그 (진행 확인용)
"""

import math
import os
import subprocess
import sys
import time
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/marieum-asr-ft")
AUDIO = Path("/content/train_audio")
NEMO = Path("/content/NeMo")
BASE = os.environ.get("BASE_MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b")
RUN = os.environ.get("RUN_NAME", "ft_v1")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "3000"))  # 48h·400s 배치면 한 바퀴 약 430 step, 약 7바퀴
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "500"))
LR = float(os.environ.get("LR", "3e-5"))
WARMUP = int(os.environ.get("WARMUP", "200"))
EFFECTIVE_BATCH_S = float(os.environ.get("EFFECTIVE_BATCH_S", "400"))  # 한 번 갱신에 들어가는 음성 초
LOG = DRIVE / "runs" / "train" / f"{RUN}.log"
MAX_DURATION = float(os.environ.get("MAX_DURATION", "22"))  # 이보다 긴 구간은 로더가 버린다. 속도 변형 0.9배도 생각한다
TRAIN_MANIFEST = os.environ.get("TRAIN_MANIFEST", "train_v2.jsonl")
VAL_MANIFEST = os.environ.get("VAL_MANIFEST", "val_v2.jsonl")
CONF_DIR = "examples/asr/conf/fastconformer/cache_aware_streaming"
CONF_NAME = "fastconformer_transducer_bpe_streaming_prompt"


def sh(cmd, **kw):
    print("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, shell=isinstance(cmd, str), **kw)


def gpu_mem_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    name, mem = out.split(",")
    return name.strip(), int(mem) / 1024


def install():
    try:
        import nemo.collections.asr  # noqa: F401
        if (NEMO / "examples").exists():
            return
    except ImportError:
        pass
    sh(["apt-get", "-qq", "install", "-y", "libsndfile1", "ffmpeg"])
    if not NEMO.exists():
        sh(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/NeMo.git", str(NEMO)])
    sh([sys.executable, "-m", "pip", "install", "-q", "Cython", "packaging"])
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", f"{NEMO}[asr]"])


def unpack():
    if AUDIO.exists() and any(AUDIO.iterdir()):
        return
    if not DRIVE.exists():
        raise SystemExit(f"{DRIVE}가 없다. colab drivemount를 먼저 한다 (사용자)")
    sh(f"cat {DRIVE}/data/train_audio.tar.part-* | tar -x -C /content")


def overrides(train_m, val_m, bd, accum):
    """NeMo 설정 덮어쓰기. 공식 설정에 없는 키는 ++ (있으면 덮어쓰기, 없으면 추가).
    바꾸면 이 PC에서 먼저 검사한다: python colab/check_train_config.py"""
    return [
        # .nemo 파일에서 이어 학습할 때는 init_from_nemo_model (ft_v2는 ft_v1에서 시작, 11-16)
        f"+init_from_nemo_model={BASE}" if BASE.endswith(".nemo") else f"+init_from_pretrained_model={BASE}",
        # 데이터
        f"model.train_ds.manifest_filepath={train_m}", "model.train_ds.is_tarred=false",
        "model.train_ds.tarred_audio_filepaths=null", f"model.train_ds.batch_duration={bd}",
        "++model.train_ds.use_bucketing=true", "model.train_ds.num_buckets=10", "model.train_ds.bucket_buffer_size=5000",
        f"model.train_ds.max_duration={MAX_DURATION}", "model.train_ds.min_duration=0.5", "model.train_ds.shuffle=true",
        "++model.train_ds.perturb_speed=true", "model.train_ds.num_workers=4",
        # 언어: 항상 ko-KR을 준다. 공식 기본값 unified는 절반을 '언어 스스로 맞히기'(auto)로 학습한다 (11-15 시험 학습).
        # 마리음은 항상 한국어라 필요 없다
        "model.train_ds.default_prompt_mode=langID", "++model.validation_ds.default_prompt_mode=langID",
        f"model.validation_ds.manifest_filepath={val_m}", "model.validation_ds.batch_size=8",
        "model.validation_ds.num_workers=2",
        "~model.test_ds",
        # 검증 중 정답·예측 문장을 로그에 찍지 않는다 (AI Hub 전사가 로그와 에이전트 화면에 남지 않게, 11-16)
        "model.log_prediction=false",
        # 최적화: 작은 학습률 + 코사인 (Noam의 lr은 배율이라 헷갈려서 바꾼다)
        "model.optim.name=adamw", f"model.optim.lr={LR}", "model.optim.weight_decay=1e-3",
        "model.optim.sched.name=CosineAnnealing", f"model.optim.sched.warmup_steps={WARMUP}",
        "model.optim.sched.min_lr=1e-6", "~model.optim.sched.d_model",
        # 학습기. strategy는 공식 설정에서 DDPStrategy 묶음이라 지우고 새로 넣는다 (GPU 1장)
        "trainer.devices=1", "trainer.accelerator=gpu", "~trainer.strategy", "+trainer.strategy=auto",
        "trainer.precision=bf16-mixed", f"trainer.max_steps={MAX_STEPS}",
        f"trainer.limit_train_batches={EVAL_EVERY}", "trainer.val_check_interval=1.0",
        f"trainer.accumulate_grad_batches={accum}", "trainer.log_every_n_steps=25", "trainer.gradient_clip_val=0.5",
        # 체크포인트: Drive, 이어하기
        f"exp_manager.exp_dir={DRIVE / 'ckpt'}", f"exp_manager.name={RUN}", "++exp_manager.version=v1",
        "++exp_manager.use_datetime_version=false", "exp_manager.resume_if_exists=true",
        "exp_manager.resume_ignore_no_checkpoint=true",
        "exp_manager.checkpoint_callback_params.save_top_k=1",
        "exp_manager.checkpoint_callback_params.always_save_nemo=true",
        "exp_manager.create_tensorboard_logger=true",
    ]


def main():
    t0 = time.time()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    gpu, mem = gpu_mem_gb()
    # 메모리별 한 배치 음성 길이(초). 96GB에서도 400초는 메모리 부족이 났다는 사례가 있어 보수적으로 잡는다
    bd = float(os.environ.get("BATCH_DURATION", 40 if mem < 30 else 90 if mem < 50 else 160))
    accum = max(1, math.ceil(EFFECTIVE_BATCH_S / bd))
    print(f"GPU {gpu} {mem:.0f}GB, batch_duration {bd}s x accum {accum}, max_steps {MAX_STEPS}, lr {LR}", flush=True)

    unpack()
    install()
    train_m, val_m = DRIVE / "data/manifests" / TRAIN_MANIFEST, DRIVE / "data/manifests" / VAL_MANIFEST
    for m in (train_m, val_m):
        if not m.exists():
            raise SystemExit(f"{m}가 없다. 이 PC의 ~/marieum-private/manifests/를 Drive data/manifests/에 올린다")

    args = [sys.executable, str(NEMO / "examples/asr/speech_to_text_finetune.py"),
            f"--config-path={NEMO / CONF_DIR}", f"--config-name={CONF_NAME}"] + overrides(train_m, val_m, bd, accum)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {gpu} bd={bd} accum={accum} steps={MAX_STEPS} lr={LR}\n")
        f.flush()
        p = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
    print(f"exit {p.returncode}, {(time.time() - t0) / 60:.0f}분. 로그: {LOG}", flush=True)
    if p.returncode != 0:
        print(open(LOG, encoding="utf-8").read()[-3000:])
        raise SystemExit(p.returncode)
    nemos = sorted((DRIVE / "ckpt" / RUN / "v1" / "checkpoints").glob("*.nemo"))
    print("저장된 .nemo:", [n.name for n in nemos])


if __name__ == "__main__":
    main()
