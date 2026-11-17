"""Colab GPU VM: NeMo 공식 스트리밍 파이프라인(CacheAwareRNNTPipeline)으로 test_v2를 돌린다 (11-17).

docs/engine_handoff.md 6절 1·2단계:
  1. 예전 경로(EXP-010, CER 28.4)와 같은 답이 나오는가. 발화 끝 판정을 끄고(stop_history_eou 아주 크게) 녹음 전체를 돌린다.
     계산 방식 두 가지: bf16 계산(파이프라인 기본) / fp32 가중치 + autocast(예전 경로와 같다)
  2. 모델의 발화 끝 판정(stop_history_eou 800 / 1200 / 1600 / 2000ms)이 구음장애 화자의 문장을 어디서 끊는가.
     발화가 확정된 시각과 글자 수만 남긴다. 운영 VAD(쉼 1.75초) 구간과의 비교는 이 PC에서 eval/eou_compare.py로 한다

    python /content/job.py start pipeline_eval.py   (그다음 python /content/job.py wait)

입력  /content/upload/test_v2.tar.part-*, Drive ckpt/ft_v3/v1/checkpoints/ft_v3.nemo
출력  /content/out/pipe_<이름>/hyps.jsonl     {"speaker", "text"}  (이 PC에서 eval/score.py로 채점)
      /content/out/pipe_<이름>/eou.json       {화자: [[확정 시각, 글자 수], ...]}  숫자만
      /content/out/pipe_<이름>/run_info.json
      /content/out/pipe_vad/pipe_vad.json    {화자: [[말 시작 ms, 말 끝 ms, 구간 닫힘 ms, 이유], ...]}  운영 Segmenter 구간
      Drive runs/eval/ 에도 같은 이름으로 복사
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("CLIP_S", "100000")  # streaming_metrics가 녹음 전체를 읽게
import zeroshot_nemotron as Z  # noqa: E402  unpack(), install(), 경로
import streaming_metrics as M  # noqa: E402  운영 Segmenter로 VAD 구간 만들기

DRIVE = Path("/content/drive/MyDrive/marieum-asr-ft")
MODEL = os.path.expanduser(os.environ.get("MODEL", str(DRIVE / "ckpt/ft_v3/v1/checkpoints/ft_v3.nemo")))
BATCH = int(os.environ.get("BATCH", "32"))
NO_EOU = 100_000_000  # 발화 끝 판정을 사실상 끈다 (녹음 끝에서만 확정)
# (이름, 계산 방식, stop_history_eou ms)
RUNS = [("pipe_bf16_noeou", "bf16", NO_EOU), ("pipe_amp_noeou", "amp", NO_EOU),
        ("pipe_eou800", "bf16", 800), ("pipe_eou1200", "bf16", 1200),
        ("pipe_eou1600", "bf16", 1600), ("pipe_eou2000", "bf16", 2000)]
ONLY = [x for x in os.environ.get("ONLY", "").split(",") if x]


def ensure_nemo():
    try:
        from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder  # noqa: F401
    except ImportError:
        Z.install()  # 설치한 프로세스에서는 바로 import가 안 될 수 있어 새로 띄운다
        os.execv(sys.executable, [sys.executable] + sys.argv)


def make_cfg(dtype, eou):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(Z.NEMO / "examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml")
    over = {
        "asr": {"model_name": MODEL, "strip_lang_tags": True,
                "compute_dtype": "bfloat16" if dtype == "bf16" else "float32", "use_amp": dtype == "amp"},
        "lang": "ko-KR",
        "streaming": {"att_context_size": [56, 3], "batch_size": BATCH, "num_slots": max(256, BATCH)},
        "endpointing": {"stop_history_eou": eou},
        "enable_itn": False, "enable_nmt": False, "asr_output_granularity": "segment",
        "return_tail_result": True,  # 예전 경로의 마지막 조각 keep_all_outputs=True와 맞춘다
        "calculate_wer": False, "calculate_bleu": False, "log_level": 30,
    }
    return OmegaConf.merge(cfg, OmegaConf.create(over))


def vad_segments(rows):
    """운영 서버와 같은 구간 (streaming_metrics.make_segments). 확률은 버리고 구간만 남긴다."""
    out = Z.WORK / "out" / "pipe_vad.json"
    if out.exists():
        return
    M.SEGS = Z.WORK / "out" / "pipe_vad_raw.json"
    segs = M.make_segments(M.load_clips())
    keep = {spk: [[s["start_ms"], s["end_ms"], s["close_ms"], s["reason"]] for s in v["segments"]]
            for spk, v in segs.items()}
    json.dump(keep, open(out, "w"))
    M.SEGS.unlink(missing_ok=True)
    print(f"VAD 구간 {sum(len(v) for v in keep.values())}개", flush=True)


def keep_on_drive(d, names):
    if DRIVE.exists():
        dst = DRIVE / "runs/eval" / d.name
        dst.mkdir(parents=True, exist_ok=True)
        for n in names:
            shutil.copy(d / n, dst / n)


def run_one(name, dtype, eou, rows):
    import torch
    from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
    out = Z.WORK / "out" / name
    out.mkdir(parents=True, exist_ok=True)
    paths = [str(Z.DATA / r["audio_filepath"]) for r in rows]
    spk = {Path(p).name: r["speaker"] for p, r in zip(paths, rows)}  # 파이프라인이 경로를 바꿔 돌려줄 수 있어 파일 이름으로 맞춘다
    t0 = time.time()
    pipe = PipelineBuilder.build_pipeline(make_cfg(dtype, eou))
    t1 = time.time()
    res = pipe.run(paths)
    infer_s = time.time() - t1
    by = {}
    for v in res.values():
        by[spk[Path(str(v["audio_filepath"])).name]] = v
    with open(out / "hyps.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"speaker": r["speaker"], "text": by.get(r["speaker"], {}).get("text", "").strip()},
                               ensure_ascii=False) + "\n")
    # 발화가 확정된 시각(그때까지 들어간 오디오 길이, NeMo get_elapsed_duration 그대로)과 글자 수. 문장은 남기지 않는다
    eou_times = {s: [[round(float(d), 3), len(t.strip())] for t, d in v.get("asr_segments", [])] for s, v in by.items()}
    json.dump(eou_times, open(out / "eou.json", "w"))
    audio_min = sum(r["duration"] for r in rows) / 60
    info = {"name": name, "model": MODEL, "dtype": dtype, "stop_history_eou": eou, "att_context_size": "[56,3]",
            "batch": BATCH, "files": len(rows), "audio_min": round(audio_min, 1), "load_s": round(t1 - t0),
            "infer_s": round(infer_s), "rtf": round(infer_s / (audio_min * 60), 4),
            "finals": sum(len(v) for v in eou_times.values()), "gpu": torch.cuda.get_device_name(0)}
    json.dump(info, open(out / "run_info.json", "w"), indent=1)
    print(json.dumps(info, ensure_ascii=False), flush=True)
    keep_on_drive(out, ["hyps.jsonl", "eou.json", "run_info.json"])
    del pipe
    torch.cuda.empty_cache()


def main():
    ensure_nemo()
    if not os.path.exists(MODEL):
        raise SystemExit(f"{MODEL}가 없다. colab drivemount를 먼저 한다 (사용자)")
    Z.unpack()
    rows = [json.loads(l) for l in open(Z.DATA / "manifest.jsonl", encoding="utf-8")]
    vad_segments(rows)
    vad_dir = Z.WORK / "out" / "pipe_vad"
    vad_dir.mkdir(exist_ok=True)
    shutil.copy(Z.WORK / "out" / "pipe_vad.json", vad_dir / "pipe_vad.json")
    keep_on_drive(vad_dir, ["pipe_vad.json"])
    for name, dtype, eou in RUNS:
        if ONLY and name not in ONLY:
            continue
        t = time.time()
        try:
            run_one(name, dtype, eou, rows)
            print(f"[{name}] 끝, {(time.time() - t) / 60:.0f}분", flush=True)
        except Exception as e:  # noqa: BLE001  하나가 실패해도 다음 것은 돈다
            print(f"[{name}] 실패: {e!r}", flush=True)


if __name__ == "__main__":
    main()
