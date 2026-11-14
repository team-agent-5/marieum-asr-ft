"""학습용 오디오를 화자마다 목표 시간만큼 골라 16kHz 모노 FLAC으로 꺼낸다.

목표: candidate/seen_test/val 화자 30분, anchor 15분. seen_test는 holdout_stem(평가 파일)을 쓰지 않는다.
파일 순서: 말하기(03,07,05)와 문장 읽기 먼저, 단어 목록(01)은 마지막. 같은 과제 안에서는 짧은 파일부터.
다음 파일이 남은 시간보다 많이 길면 앞부분만 쓴다 (최소 2분, 한 파일에서 최대 30분).

출력 (work/train_audio/, gitignore):
  <speaker>/<stem>.flac   selection.csv
  정답 전사는 레포 밖 ~/marieum-private/train_refs.jsonl (출력·커밋 금지)

    python data_prep/select_train_audio.py --root "E:/Dataset_Backup/구음장애_음성인식_데이터" --dry-run
    python data_prep/select_train_audio.py --root "E:/Dataset_Backup/구음장애_음성인식_데이터"
"""

import argparse
import csv
import json
import subprocess
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from extract_test_audio import duration
from inventory import iter_labels
from make_listening_set import source_index

SR = 16000
CHUNK = 1 << 22  # 4MB씩 ffmpeg에 흘려 보낸다
ORDER = ["03", "07", "05", "02", "06", "04", "53", "52", "01"]
MIN_CHUNK, MAX_FILE, SLACK = 2.0, 30.0, 5.0  # 분. SLACK: 남은 시간보다 이만큼 넘게 길면 앞부분만 쓴다
COLS = ["speaker", "split", "role", "tier", "folder", "stem", "test_code", "file_minutes", "used_minutes", "truncated"]


def read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def rank(r):
    c = r["test_code"]
    return (ORDER.index(c) if c in ORDER else len(ORDER), float(r["minutes"]), r["stem"])


def select(spk, files):
    target = 15.0 if spk["role"] == "anchor" else 30.0
    total, picked = 0.0, []
    for f in sorted(files, key=rank):
        remaining = target - total
        if remaining < MIN_CHUNK:
            break
        m = float(f["minutes"])
        if m <= MAX_FILE and m <= remaining + SLACK:
            used = m
        else:
            used = min(remaining, MAX_FILE, m)
        picked.append((f, round(used, 3), used < m - 1e-6))
        total += used
    return picked


def extract(job):
    out = Path(job["out"])
    if out.exists():
        return job["speaker"], "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.flac")  # 중간에 끊겨도 반쪽 파일이 남지 않게
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", "-"]
    if job["seconds"]:
        cmd += ["-t", str(job["seconds"])]
    cmd += ["-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", str(tmp)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    src = job["src"]
    fh = open(src[1], "rb") if src[0] == "disk" else zipfile.ZipFile(src[1]).open(src[2])
    try:
        with fh:
            while chunk := fh.read(CHUNK):
                p.stdin.write(chunk)
    except OSError:  # -t면 (Windows는 EINVAL로 올 때도 있다) ffmpeg가 먼저 끝난다
        pass
    finally:
        try:
            p.stdin.close()
        except OSError:
            pass
    err = p.wait()
    if err != 0 or not tmp.exists():
        return job["speaker"], f"error {p.stderr.read().decode(errors='ignore')[:200]}"
    tmp.replace(out)
    return job["speaker"], "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="work/train_audio")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit-speakers", type=int, default=0)
    ap.add_argument("--refs-out", default=str(Path.home() / "marieum-private" / "train_refs.jsonl"))
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)

    spks = [dict(r, split="train") for r in read_csv("splits/train_v1.csv")]
    spks += [dict(r, split="val", role="val", holdout_stem="") for r in read_csv("splits/val_v1.csv")]
    if args.limit_speakers:  # 시험용: train과 val에서 절반씩
        tr = [s for s in spks if s["split"] == "train"]
        va = [s for s in spks if s["split"] == "val"]
        spks = tr[:(args.limit_speakers + 1) // 2] + va[:args.limit_speakers // 2]
    by_spk = defaultdict(list)
    for r in read_csv("work/inventory/files.csv"):
        by_spk[r["speaker"]].append(r)

    rows = []
    for s in spks:
        files = by_spk[s["speaker"]]
        if s["role"] == "seen_test":
            assert any(f["stem"] == s["holdout_stem"] for f in files), f"holdout 없음: {s['speaker']}"
            files = [f for f in files if f["stem"] != s["holdout_stem"]]
        for f, used, trunc in select(s, files):
            if s["role"] == "seen_test":
                assert f["stem"] != s["holdout_stem"]
            rows.append(dict(speaker=s["speaker"], split=s["split"], role=s["role"], tier=s["tier"],
                             folder=f["folder"], stem=f["stem"], test_code=f["test_code"],
                             file_minutes=f["minutes"], used_minutes=used, truncated=int(trunc)))

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "selection.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, COLS)
        w.writeheader()
        w.writerows(rows)

    def fmt(agg):
        return ", ".join(f"{k} {v:.1f}h" for k, v in sorted(agg.items()))

    agg_s, agg_t = defaultdict(float), defaultdict(float)
    for r in rows:
        agg_s[r["split"]] += r["used_minutes"] / 60
        agg_t[r["tier"]] += r["used_minutes"] / 60
    got = defaultdict(float)
    for r in rows:
        got[r["speaker"]] += r["used_minutes"]
    short = [s["speaker"] for s in spks if got[s["speaker"]] < (15 if s["role"] == "anchor" else 30) - 1]
    print(f"화자 {len(spks)}명, 파일 {len(rows)}개, 잘라 쓴 파일 {sum(r['truncated'] for r in rows)}개, "
          f"합계 {sum(agg_s.values()):.1f}h")
    print("split별: " + fmt(agg_s))
    print("tier별: " + fmt(agg_t))
    print(f"목표에 못 미친 화자 {len(short)}명")
    if args.dry_run:
        return

    print("원천 zip 목록 읽는 중...")
    zidx = source_index(root)
    disk = {p.stem: str(p) for p in root.rglob("*.wav")}
    jobs = []
    for r in rows:
        s = r["stem"]
        src = ("disk", disk[s]) if s in disk else ("zip", *zidx[s]) if s in zidx else None
        if src is None:
            raise SystemExit(f"원천 없음: {s}")
        jobs.append({"speaker": r["speaker"], "src": src, "out": str(out / r["speaker"] / f"{s}.flac"),
                     "seconds": round(r["used_minutes"] * 60, 1) if r["truncated"] else 0})

    print("정답 전사 읽는 중 (출력하지 않는다)...")
    want = {r["stem"] for r in rows}
    refs = {}
    for _, _, _, stem, d in iter_labels(root):
        if stem in want:
            refs[stem] = d.get("Transcript") or ""
    missing = [s for s in want if s not in refs]
    if missing:
        raise SystemExit(f"라벨 없음 {len(missing)}개: {missing[:5]}")
    Path(args.refs_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.refs_out, "w", encoding="utf-8") as fr:
        for r in rows:
            fr.write(json.dumps({"stem": r["stem"], "speaker": r["speaker"], "split": r["split"],
                                 "truncated": bool(r["truncated"]),
                                 "used_seconds": round(r["used_minutes"] * 60, 1),
                                 "text": refs[r["stem"]]}, ensure_ascii=False) + "\n")
    print(f"전사 {len(rows)}줄 -> {args.refs_out}")

    print(f"{len(jobs)}개 추출")
    cnt = defaultdict(int)
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (spk, status) in enumerate(ex.map(extract, jobs), 1):
            kind = status.split()[0]
            cnt[kind] += 1
            if kind == "error":
                print(f"  실패 {spk}: {status}", flush=True)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} (ok {cnt['ok']}, skip {cnt['skip']}, 실패 {cnt['error']})", flush=True)
    print(f"추출 {cnt['ok']}, 건너뜀 {cnt['skip']}, 실패 {cnt['error']}")

    paths = [j["out"] for j in jobs if Path(j["out"]).exists()]
    with ProcessPoolExecutor(args.workers) as ex:
        durs = dict(zip(paths, ex.map(duration, paths, chunksize=20)))
    agg_s, agg_t = defaultdict(float), defaultdict(float)
    for j, r in zip(jobs, rows):
        d = durs.get(j["out"], 0) / 3600
        agg_s[r["split"]] += d
        agg_t[r["tier"]] += d
    size = sum(p.stat().st_size for p in out.rglob("*.flac"))
    print(f"실제 오디오 {sum(durs.values()) / 3600:.1f}h, 디스크 {size / 1e9:.2f}GB")
    print("split별: " + fmt(agg_s))
    print("tier별: " + fmt(agg_t))


if __name__ == "__main__":
    main()
