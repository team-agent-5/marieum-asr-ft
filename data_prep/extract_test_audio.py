"""test_v2 화자의 평가 파일을 16kHz 모노 FLAC으로 꺼내고, 평가용 manifest와 정답 파일을 만든다.

출력 (work/test_v2/, gitignore):
  audio/<speaker>.flac   eval_note=first600s면 앞 600초만
  manifest.jsonl         Colab에 올리는 목록. 전사가 없다 (audio_filepath, duration, speaker, target_lang)
  정답 전사는 레포 밖 ~/marieum-private/test_v2_refs.jsonl에 쓴다 (--refs-out).
  이 PC에서 채점할 때만 쓰고, Colab·git에 올리지 않는다. 레포에서 일하는 다른 에이전트가 읽지 못하게 밖에 둔다

    python data_prep/extract_test_audio.py --root "E:/Dataset_Backup/구음장애_음성인식_데이터"
"""

import argparse
import csv
import json
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from inventory import iter_labels
from make_listening_set import source_index

SR = 16000
CHUNK = 1 << 22  # 4MB씩 ffmpeg에 흘려 보낸다 (큰 wav를 메모리에 다 올리지 않게)


def extract(job):
    out = Path(job["out"])
    if out.exists():
        return job["speaker"], "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", "-"]
    if job["first600s"]:
        cmd += ["-t", "600"]
    cmd += ["-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    src = job["src"]
    fh = open(src[1], "rb") if src[0] == "disk" else zipfile.ZipFile(src[1]).open(src[2])
    try:
        with fh:
            while chunk := fh.read(CHUNK):
                p.stdin.write(chunk)
    except BrokenPipeError:  # -t 600이면 ffmpeg가 먼저 끝난다
        pass
    finally:
        try:
            p.stdin.close()
        except BrokenPipeError:
            pass
    err = p.wait()
    if err != 0 or not out.exists():
        return job["speaker"], f"error {p.stderr.read().decode(errors='ignore')[:200]}"
    return job["speaker"], "ok"


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return round(float(r.stdout.strip()), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="splits/test_v2.csv")
    ap.add_argument("--out", default="work/test_v2")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--refs-out", default=str(Path.home() / "marieum-private" / "test_v2_refs.jsonl"))
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)

    rows = list(csv.DictReader(open(args.split, encoding="utf-8-sig")))
    stems = {r["eval_stem"]: r for r in rows}

    print("원천 zip 목록 읽는 중...")
    zidx = source_index(root)
    disk = {p.stem: str(p) for p in root.rglob("*.wav")}
    jobs = []
    for r in rows:
        s = r["eval_stem"]
        src = ("disk", disk[s]) if s in disk else ("zip", *zidx[s]) if s in zidx else None
        if src is None:
            raise SystemExit(f"원천 없음: {s}")
        jobs.append({"speaker": r["speaker"], "src": src, "first600s": r["eval_note"] == "first600s",
                     "out": str(out / "audio" / f"{r['speaker']}.flac")})

    print(f"{len(jobs)}개 추출")
    with ProcessPoolExecutor(args.workers) as ex:
        for spk, status in ex.map(extract, jobs):
            if status != "skip":
                print(f"  {spk}: {status}")

    print("정답 전사 읽는 중 (출력하지 않는다)...")
    refs = {}
    for _, _, _, stem, d in iter_labels(root):
        if stem in stems:
            refs[stem] = d.get("Transcript") or ""
    missing = [s for s in stems if s not in refs]
    if missing:
        raise SystemExit(f"라벨 없음: {missing}")

    Path(args.refs_out).parent.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as fm, \
         open(args.refs_out, "w", encoding="utf-8") as fr:
        for r in rows:
            audio = out / "audio" / f"{r['speaker']}.flac"
            fm.write(json.dumps({"audio_filepath": f"audio/{audio.name}", "duration": duration(audio),
                                 "speaker": r["speaker"], "target_lang": "ko-KR", "text": ""},
                                ensure_ascii=False) + "\n")
            fr.write(json.dumps({"speaker": r["speaker"], "stem": r["eval_stem"],
                                 "partial_audio": r["eval_note"] == "first600s",
                                 "text": refs[r["eval_stem"]]}, ensure_ascii=False) + "\n")
    total = sum(json.loads(l)["duration"] for l in open(out / "manifest.jsonl", encoding="utf-8"))
    print(f"완료: {len(rows)}개, 합계 {total / 60:.0f}분 -> {out}")


if __name__ == "__main__":
    main()
