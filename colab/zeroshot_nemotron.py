"""Colab GPU VM에서 Nemotron 3.5 ASR Streaming 0.6B 제로샷 추론을 돌린다.

    colab --auth=oauth2 exec -s nemo -f colab/zeroshot_nemotron.py

전제 (docs/colab_runbook.md)
  /content/upload/test_v2.tar.part-*   이 PC에서 올린 테스트 음성 묶음 (manifest.jsonl + audio/*.flac, 정답 없음)
출력
  /content/out/nemotron_zs_<지연>/hyps.jsonl   {"speaker", "text"}  -> 이 PC로 내려받아 eval/score.py로 채점
  /content/out/nemotron_zs_<지연>/run.log       NeMo 로그, 처리 시간

설정은 환경변수로 바꾼다 (colab exec는 인자를 받지 않는다).
처음 실행하면 NeMo main 브랜치를 설치한다 (5~10분). 같은 세션에서 다시 돌리면 건너뛴다.

Colab 밖 Linux GPU 서버(NCP 등)에서는 WORKDIR로 작업 폴더를 바꾼다 (docs/colab_runbook.md "NCP 서버에서 평가")
    WORKDIR=~/asr-eval MODEL=~/asr-eval/ft_v1.nemo OUT_NAME=ft_v1_320ms python zeroshot_nemotron.py
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# 파인튜닝한 모델을 평가할 때: MODEL=/content/drive/.../checkpoints/xxx.nemo  OUT_NAME=ft_v1_320ms
MODEL = os.path.expanduser(os.environ.get("MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b"))
ATT_CONTEXT = os.environ.get("ATT_CONTEXT", "[56,3]")  # 320ms. [56,0]=80ms, [56,13]=1.12s
BATCH = 4
WORK = Path(os.path.expanduser(os.environ.get("WORKDIR", "/content")))
UPLOAD = WORK / "upload"
DATA = WORK / "test_v2"
NEMO = WORK / "NeMo"
TAG = {"[56,0]": "80ms", "[56,1]": "160ms", "[56,3]": "320ms", "[56,6]": "560ms", "[56,13]": "1120ms"}
OUT = WORK / "out" / os.environ.get("OUT_NAME", f"nemotron_zs_{TAG.get(ATT_CONTEXT, 'ctx')}")


def sh(cmd, log=None, **kw):
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    if log is None:
        subprocess.run(cmd, check=True, shell=isinstance(cmd, str), **kw)
        return
    with open(log, "a", encoding="utf-8") as f:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), stdout=f, stderr=subprocess.STDOUT, **kw)
    if p.returncode != 0:
        print(open(log, encoding="utf-8").read()[-3000:])
        raise SystemExit(f"실패 (exit {p.returncode}). 로그: {log}")


def unpack():
    if (DATA / "manifest.jsonl").exists():
        return
    parts = sorted(glob.glob(str(UPLOAD / "test_v2.tar.part-*")))
    if not parts:
        raise SystemExit(f"{UPLOAD}에 test_v2.tar.part-* 가 없다. 런북 2단계를 먼저 한다")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    sh(f"cat {' '.join(parts)} | tar -x -C {DATA.parent}")
    if (DATA / "refs.jsonl").exists():  # 정답은 VM에 두지 않는다
        (DATA / "refs.jsonl").unlink()


def install():
    try:
        import nemo.collections.asr  # noqa: F401
        if (NEMO / "examples").exists():
            return
    except ImportError:
        pass
    if not shutil.which("ffmpeg"):  # Colab은 root. 서버에서는 미리 sudo apt-get install libsndfile1 ffmpeg
        sh(["apt-get", "-qq", "install", "-y", "libsndfile1", "ffmpeg"])
    if not NEMO.exists():
        sh(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/NeMo.git", str(NEMO)])
        ref = os.environ.get("NEMO_REF")  # 정해진 커밋에 고정할 때. 40자 전체 해시를 쓴다 (짧은 해시는 fetch가 못 찾는다). 없으면 main 최신
        if ref:
            sh(["git", "-C", str(NEMO), "fetch", "--depth", "1", "origin", ref])
            sh(["git", "-C", str(NEMO), "checkout", "-q", "FETCH_HEAD"])
    sh([sys.executable, "-m", "pip", "install", "-q", "Cython", "packaging"])
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", f"{NEMO}[asr]"])


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "run.log"
    sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    unpack()
    install()

    # NeMo 스크립트는 manifest의 text 개수가 맞아야 결과 파일을 쓴다. 채점은 이 PC에서 하므로 자리표시만 넣는다
    rows = [json.loads(l) for l in open(DATA / "manifest.jsonl", encoding="utf-8")]
    man = OUT / "manifest_abs.jsonl"
    with open(man, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, "audio_filepath": str(DATA / r["audio_filepath"]), "text": "가"},
                               ensure_ascii=False) + "\n")
    audio_min = sum(r["duration"] for r in rows) / 60

    # T4(무료)는 bf16을 지원하지 않는다 (11-16 CU가 바닥나 A100·L4를 못 받음). 이전 평가(A100)는 bf16
    import torch
    amp_dtype = os.environ.get("AMP_DTYPE") or ("bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16")
    script = NEMO / "examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"
    t1 = time.time()
    sh([sys.executable, str(script),
        f"model_path={MODEL}" if MODEL.endswith(".nemo") else f"pretrained_name={MODEL}",
        f"dataset_manifest={man}", f"output_path={OUT}",
        f"att_context_size={ATT_CONTEXT}", f"batch_size={BATCH}", "decoder_type=rnnt",
        "pad_and_drop_preencoded=true", "target_lang=ko-KR", "strip_lang_tags=true",
        "amp=true", f"amp_dtype={amp_dtype}", "cuda=0"], log=log)
    infer_s = time.time() - t1

    outs = sorted(OUT.glob("streaming_out_*.json"))
    if not outs:
        raise SystemExit(f"결과 파일이 없다. 로그: {log}")
    preds = [json.loads(l)["pred_text"] for l in open(outs[-1], encoding="utf-8")]
    if len(preds) != len(rows):
        raise SystemExit(f"결과 {len(preds)}개 != 입력 {len(rows)}개")
    with open(OUT / "hyps.jsonl", "w", encoding="utf-8") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"speaker": r["speaker"], "text": p}, ensure_ascii=False) + "\n")
    outs[-1].unlink()  # 자리표시 정답이 섞인 파일은 지운다

    info = {"model": MODEL, "att_context_size": ATT_CONTEXT, "amp_dtype": amp_dtype, "files": len(rows), "audio_min": round(audio_min, 1),
            "infer_s": round(infer_s), "rtf": round(infer_s / (audio_min * 60), 4), "total_s": round(time.time() - t0)}
    json.dump(info, open(OUT / "run_info.json", "w"), indent=1)
    print(json.dumps(info, ensure_ascii=False))
    # 11-16 평가 끝무렵 VM이 회수돼 결과를 잃었다. Drive가 연결돼 있으면 바로 복사해 둔다
    drive = Path("/content/drive/MyDrive/cdy-hack-ag5/marieum-asr-ft")
    if drive.exists():
        keep = drive / "runs/eval" / OUT.name
        keep.mkdir(parents=True, exist_ok=True)
        for name in ("hyps.jsonl", "run_info.json"):
            shutil.copy(OUT / name, keep / name)
        print(f"Drive에도 저장: {keep}")
    print(f"완료. 내려받기: colab download -s nemo {OUT}/hyps.jsonl ./work/eval/{OUT.name}/hyps.jsonl")


if __name__ == "__main__":  # eval_segmented.py가 unpack·install을 가져다 쓴다
    main()
