"""Colab GPU VM: 시험 학습 모델로 중등도·중증 화자의 말소리 구간을 다시 받아쓴다 (중증 구간 다시 건지기 1단계).

yoona-J 받아쓰기가 엉망이라 정답 위치를 못 찾고 버려진 구간이 중증에서 32%였다.
시험 학습 모델(pilot_v1)은 이 화자들을 더 잘 알아들으므로, 이 받아쓰기를 두 번째 길잡이로 쓴다.
정렬은 이 PC의 data_prep/align_segments.py --segs2 가 한다 (정답은 VM에 올리지 않는다).

    colab --auth=oauth2 upload -s retx colab/retranscribe_segments.py /content/retranscribe_segments.py
    colab --auth=oauth2 upload -s retx work/upload/retranscribe_speakers.txt /content/retranscribe_speakers.txt
    echo 'import subprocess; p=subprocess.run("python /content/retranscribe_segments.py", shell=True); print("exit", p.returncode)' | colab --auth=oauth2 exec -s retx

입력
  Drive data/train_audio.tar.part-*              학습 녹음 (없으면 풀어 둔다)
  Drive runs/segments/<speaker>.jsonl            1차(yoona-J) 구간 경계
  Drive ckpt/pilot_v1/v1/checkpoints/pilot_v1.nemo
  /content/retranscribe_speakers.txt             대상 화자 (중등도·중증, 72명)
출력
  Drive runs/segments_pilot/<speaker>.jsonl      같은 구간, hyp만 시험 학습 모델 것 (speaker, stem, start, end, hyp)
  Drive runs/segments_pilot/DONE
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/cdy-hack-ag5/marieum-asr-ft")
AUDIO = Path("/content/train_audio")
CLIPS = Path("/content/seg_clips")
NEMO = Path("/content/NeMo")
MODEL = os.environ.get("MODEL", str(DRIVE / "ckpt/pilot_v1/v1/checkpoints/pilot_v1.nemo"))
ATT_CONTEXT = os.environ.get("ATT_CONTEXT", "[56,13]")  # 정렬용이라 정확도가 높은 1.12s 지연을 쓴다
SEGS = DRIVE / "runs/segments"
OUT = DRIVE / "runs/segments_pilot"
WORK = Path("/content/retx")
SR = 16000


def sh(cmd, **kw):
    print("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, shell=isinstance(cmd, str), **kw)


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
    if not DRIVE.exists():
        raise SystemExit(f"{DRIVE}가 없다. colab drivemount를 먼저 한다 (사용자)")
    if not (AUDIO.exists() and any(AUDIO.iterdir())):
        sh(f"cat {DRIVE}/data/train_audio.tar.part-* | tar -x -C /content")
    install()
    import soundfile as sf

    speakers = [l.strip() for l in open("/content/retranscribe_speakers.txt", encoding="utf-8") if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    todo = [s for s in speakers if not (OUT / f"{s}.jsonl").exists()]
    print(f"대상 {len(speakers)}명, 남은 {len(todo)}명", flush=True)

    # 1) 구간을 짧은 wav로 자른다 (NeMo 스트리밍 추론 스크립트가 offset을 받는지 확인하지 못해서 직접 자른다)
    rows = []
    for spk in todo:
        segs = [json.loads(l) for l in open(SEGS / f"{spk}.jsonl", encoding="utf-8")]
        cache = {}
        for k, s in enumerate(segs):
            src = AUDIO / spk / f"{s['stem']}.flac"
            if s["stem"] not in cache:
                cache = {s["stem"]: sf.read(src, dtype="int16")[0]}
            wav = cache[s["stem"]][int(s["start"] * SR):int(s["end"] * SR)]
            dst = CLIPS / spk / f"{s['stem']}_{k:05d}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dst, wav, SR)
            rows.append({"audio_filepath": str(dst), "duration": round(len(wav) / SR, 3), "text": "가",
                         "target_lang": "ko-KR", "_seg": s})
    man = WORK / "manifest.jsonl"
    with open(man, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_seg"}, ensure_ascii=False) + "\n")
    print(f"구간 {len(rows)}개, {sum(r['duration'] for r in rows) / 3600:.1f}h 잘랐다 ({(time.time() - t0) / 60:.0f}분)", flush=True)

    # 2) 시험 학습 모델로 받아쓴다 (제로샷 평가와 같은 공식 스트리밍 추론 스크립트)
    script = NEMO / "examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"
    log = OUT / "run.log"
    with open(log, "a", encoding="utf-8") as f:
        p = subprocess.run([sys.executable, str(script), f"model_path={MODEL}", f"dataset_manifest={man}",
                            f"output_path={WORK}", f"att_context_size={ATT_CONTEXT}", "batch_size=32",
                            "decoder_type=rnnt", "pad_and_drop_preencoded=true", "target_lang=ko-KR",
                            "strip_lang_tags=true", "amp=true", "amp_dtype=bfloat16", "cuda=0"],
                           stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(open(log, encoding="utf-8").read()[-3000:])
        raise SystemExit(p.returncode)
    out = sorted(WORK.glob("streaming_out_*.json"))[-1]
    preds = [json.loads(l)["pred_text"] for l in open(out, encoding="utf-8")]
    if len(preds) != len(rows):
        raise SystemExit(f"결과 {len(preds)}개 != 입력 {len(rows)}개")
    out.unlink()  # 자리표시 정답이 섞인 파일

    # 3) 화자별로 1차와 같은 형식으로 저장
    by = {}
    for r, h in zip(rows, preds):
        s = r["_seg"]
        by.setdefault(s["speaker"], []).append({"speaker": s["speaker"], "stem": s["stem"], "start": s["start"],
                                                "end": s["end"], "hyp": h.strip()})
    for spk, rs in by.items():
        tmp = OUT / f"{spk}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.rename(OUT / f"{spk}.jsonl")
    (OUT / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    print(f"완료: 화자 {len(by)}명, {(time.time() - t0) / 60:.0f}분", flush=True)


if __name__ == "__main__":
    main()
