"""문장 나누기 결과로 학습·검증 manifest를 만든다 (화자당 상한, 묶음별 반복).

입력  ~/marieum-private/segments_{train,val}.jsonl   (align_segments.py 출력)
출력  ~/marieum-private/manifests/train_v1.jsonl, val_v1.jsonl   (전사가 들어 있다. 레포 밖, Drive에만 올린다)
      work/manifests/train_v1_stats.csv                       (숫자만)

- 화자당 상한: 학습 20분, 기준점 10분, 검증 10분. 상한 안에서는 무작위로 고른다 (파일·과제가 섞이게)
- 묶음별 반복: 중증 3번, 중등도 2번, 경증 1번, 기준점 2번 (--repeat로 바꾼다).
  NeMo Lhotse 로더는 manifest별 가중치 설정이 불확실해서 줄을 반복해 비율을 맞춘다. 셔플 버퍼가 섞는다

    python data_prep/build_train_manifests.py
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

PRIVATE = Path.home() / "marieum-private"
KEEP = ["audio_filepath", "offset", "duration", "text", "target_lang", "lang", "speaker", "tier"]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def cap_per_speaker(rows, minutes, rng):
    by = defaultdict(list)
    for r in rows:
        by[r["speaker"]].append(r)
    out = []
    for spk, rs in sorted(by.items()):
        rng.shuffle(rs)
        lim = minutes(rs[0]["tier"]) * 60
        tot = 0.0
        for r in rs:
            if tot + r["duration"] > lim:
                continue
            out.append(r)
            tot += r["duration"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segs", default=str(PRIVATE))
    ap.add_argument("--out", default=str(PRIVATE / "manifests"))
    ap.add_argument("--stats", default="work/manifests/train_v1_stats.csv")
    ap.add_argument("--cap-min", type=float, default=20)
    ap.add_argument("--anchor-cap-min", type=float, default=10)
    ap.add_argument("--val-cap-min", type=float, default=10)
    ap.add_argument("--repeat", default="severe=3,moderate=2,mild=1,anchor=2")
    ap.add_argument("--seed", type=int, default=20261115)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rep = {k: int(v) for k, v in (x.split("=") for x in args.repeat.split(","))}

    train = cap_per_speaker(load(Path(args.segs) / "segments_train.jsonl"),
                            lambda t: args.anchor_cap_min if t == "anchor" else args.cap_min, rng)
    val = cap_per_speaker(load(Path(args.segs) / "segments_val.jsonl"), lambda t: args.val_cap_min, rng)

    train_rep = [r for r in train for _ in range(rep.get(r["tier"], 1))]
    rng.shuffle(train_rep)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train_v1.jsonl", train_rep), ("val_v1.jsonl", val)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in KEEP if k in r}, ensure_ascii=False) + "\n")

    stats = defaultdict(lambda: {"speakers": set(), "segs": 0, "hours": 0.0, "hours_rep": 0.0})
    for r in train:
        s = stats[("train", r["tier"])]
        s["speakers"].add(r["speaker"])
        s["segs"] += 1
        s["hours"] += r["duration"] / 3600
        s["hours_rep"] += rep.get(r["tier"], 1) * r["duration"] / 3600
    for r in val:
        s = stats[("val", r["tier"])]
        s["speakers"].add(r["speaker"])
        s["segs"] += 1
        s["hours"] += r["duration"] / 3600
        s["hours_rep"] += r["duration"] / 3600
    total_rep = sum(s["hours_rep"] for (sp, _), s in stats.items() if sp == "train")
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["split", "tier", "speakers", "segments", "hours", "repeat", "hours_after_repeat", "share"])
        print("split 묶음      화자  구간    시간  반복  반복후시간  비율")
        for (sp, t), s in sorted(stats.items()):
            share = s["hours_rep"] / total_rep if sp == "train" else ""
            row = [sp, t, len(s["speakers"]), s["segs"], round(s["hours"], 2), rep.get(t, 1) if sp == "train" else 1,
                   round(s["hours_rep"], 2), round(share, 3) if share != "" else ""]
            w.writerow(row)
            print(f"{sp:5s} {t:9s} {row[2]:4d} {row[3]:6d} {row[4]:6.1f} {row[5]:4d} {row[6]:9.1f}  {row[7]}")
    print(f"-> {out}/train_v1.jsonl ({len(train_rep)}줄), val_v1.jsonl ({len(val)}줄)")


if __name__ == "__main__":
    main()
