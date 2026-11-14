"""Colab GPU VM에서 Nemotron 3.5 ASR Streaming 0.6B 제로샷 추론을 돌린다.

    colab --auth=oauth2 exec -s nemo -f colab/zeroshot_nemotron.py

전제 (docs/colab_runbook.md)
  /content/upload/test_v2.tar.part-*   이 PC에서 올린 테스트 음성 묶음 (manifest.jsonl + audio/*.flac, 정답 없음)
출력
  /content/out/nemotron_zs_<지연>/hyps.jsonl   {"speaker", "text"}  -> 이 PC로 내려받아 eval/score.py로 채점
  /content/out/nemotron_zs_<지연>/run.log       NeMo 로그, 처리 시간

설정은 아래 상수로 바꾼다 (colab exec는 인자를 받지 않는다).
처음 실행하면 NeMo main 브랜치를 설치한다 (5~10분). 같은 세션에서 다시 돌리면 건너뛴다.
"""

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# 파인튜닝한 모델을 평가할 때: MODEL=/content/drive/.../checkpoints/xxx.nemo  OUT_NAME=ft_v1_320ms
MODEL = os.environ.get("MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b")
ATT_CONTEXT = os.environ.get("ATT_CONTEXT", "[56,3]")  # 320ms. [56,0]=80ms, [56,13]=1.12s
BATCH = 4
UPLOAD = Path("/content/upload")
DATA = Path("/content/test_v2")
NEMO = Path("/content/NeMo")
TAG = {"[56,0]": "80ms", "[56,1]": "160ms", "[56,3]": "320ms", "[56,6]": "560ms", "[56,13]": "1120ms"}
OUT = Path("/content/out") / os.environ.get("OUT_NAME", f"nemotron_zs_{TAG.get(ATT_CONTEXT, 'ctx')}")


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
    sh(["apt-get", "-qq", "install", "-y", "libsndfile1", "ffmpeg"])
    if not NEMO.exists():
        sh(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/NeMo.git", str(NEMO)])
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

    script = NEMO / "examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"
    t1 = time.time()
    sh([sys.executable, str(script),
        f"model_path={MODEL}" if MODEL.endswith(".nemo") else f"pretrained_name={MODEL}",
        f"dataset_manifest={man}", f"output_path={OUT}",
        f"att_context_size={ATT_CONTEXT}", f"batch_size={BATCH}", "decoder_type=rnnt",
        "pad_and_drop_preencoded=true", "target_lang=ko-KR", "strip_lang_tags=true",
        "amp=true", "amp_dtype=bfloat16", "cuda=0"], log=log)
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

    info = {"model": MODEL, "att_context_size": ATT_CONTEXT, "files": len(rows), "audio_min": round(audio_min, 1),
            "infer_s": round(infer_s), "rtf": round(infer_s / (audio_min * 60), 4), "total_s": round(time.time() - t0)}
    json.dump(info, open(OUT / "run_info.json", "w"), indent=1)
    print(json.dumps(info, ensure_ascii=False))
    print(f"완료. 내려받기: colab download -s nemo {OUT}/hyps.jsonl ./work/eval/{OUT.name}/hyps.jsonl")


main()
