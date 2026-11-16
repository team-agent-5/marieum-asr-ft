"""문장 나누기 결과로 학습·검증 manifest를 만든다 (화자당 상한, 묶음별 반복).

입력  ~/marieum-private/segments_{train,val}.jsonl   (align_segments.py 출력)
출력  ~/marieum-private/manifests/train_v1.jsonl, val_v1.jsonl   (전사가 들어 있다. 레포 밖, Drive에만 올린다)
      work/manifests/train_v1_stats.csv                       (숫자만)

- 화자당 상한: 학습 20분, 기준점 10분, 검증 10분. 상한 안에서는 무작위로 고른다 (파일·과제가 섞이게)
- 앞뒤 여유 (--context-pad, v4부터): 구간 앞뒤에 실제 녹음을 0.2~최대초 더 붙인다. 이웃 말소리 구간(VAD)까지의
  쉼 안에서만 붙여서 정답에 없는 말이 들어가지 않게 한다. "조용하다가 말이 시작되는" 소리를 배우게 한다 (11-16 EXP-004:
  쉬었다 다시 말하는 문장 첫머리를 1.8배 더 빠뜨림)
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
KEEP = ["audio_filepath", "offset", "duration", "text", "target_lang", "lang", "speaker", "tier", "merged"]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def merge_neighbors(rows, rng, max_gap, max_len, prob):
    """같은 녹음에서 바로 이어지는 구간(seg_idx가 연속, 쉼 max_gap초 이하)을 max_len초까지 이어 붙인다.
    이어 붙일 수 있는 묶음마다 prob 확률로 붙인 것을, 아니면 원래 짧은 구간들을 쓴다 (짧은 것과 긴 것이 섞이게).
    붙인 구간에는 사이의 쉼도 들어간다: 쉬었다가 이어 말할 때 글자를 놓치지 않게 배우라는 뜻이다."""
    by = defaultdict(list)
    for r in rows:
        by[r["stem"]].append(r)
    out, merged = [], 0
    for stem, rs in sorted(by.items()):
        rs.sort(key=lambda r: r["seg_idx"])
        run = [rs[0]]

        def flush(run):
            nonlocal merged
            # run 안에서 앞에서부터 max_len을 넘지 않게 묶는다
            i = 0
            while i < len(run):
                j = i + 1
                while j < len(run) and run[j]["offset"] + run[j]["duration"] - run[i]["offset"] <= max_len:
                    j += 1
                group = run[i:j]
                if len(group) > 1 and rng.random() < prob:
                    first, last = group[0], group[-1]
                    out.append({**first, "duration": round(last["offset"] + last["duration"] - first["offset"], 2),
                                "text": " ".join(g["text"] for g in group), "merged": len(group)})
                    merged += 1
                else:
                    out.extend(group)
                i = j

        for prev, cur in zip(rs, rs[1:]):
            gap = cur["offset"] - (prev["offset"] + prev["duration"])
            if cur["seg_idx"] == prev["seg_idx"] + 1 and 0 <= gap <= max_gap:
                run.append(cur)
            else:
                flush(run)
                run = [cur]
        flush(run)
    return out, merged


def add_context(rows, vad_dir, rng, pad_max, margin=0.1):
    """구간 앞뒤로 이웃 VAD 구간까지의 쉼 안에서 0.2~pad_max초를 더 붙인다. 녹음 끝 구간의 뒤는 붙이지 않는다
    (파일 길이를 모르므로). 붙인 양은 pad_before/pad_after로 남긴다 (숫자 확인용, manifest에는 안 쓴다)."""
    bounds = defaultdict(list)
    for f in Path(vad_dir).glob("*.jsonl"):
        for line in open(f, encoding="utf-8"):
            v = json.loads(line)
            bounds[v["stem"]].append((v["start"], v["end"]))
    for b in bounds.values():
        b.sort()
    out, mismatch = [], 0
    for r in rows:
        b = bounds[r["stem"]]
        k = r["seg_idx"]
        if k >= len(b) or abs(b[k][0] - r["offset"]) > 0.02:
            mismatch += 1
            out.append(r)
            continue
        room_before = r["offset"] - (b[k - 1][1] if k > 0 else 0.0) - margin
        room_after = (b[k + 1][0] - b[k][1] - margin) if k + 1 < len(b) else 0.0
        pb = max(0.0, min(rng.uniform(0.2, pad_max), room_before, r["offset"]))
        pa = max(0.0, min(rng.uniform(0.2, pad_max), room_after))
        out.append({**r, "offset": round(r["offset"] - pb, 3), "duration": round(r["duration"] + pb + pa, 3),
                    "pad_before": pb, "pad_after": pa})
    if mismatch:
        print(f"주의: VAD 경계와 안 맞는 구간 {mismatch}개는 여유 없이 쓴다")
    return out


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
    ap.add_argument("--stats", default="work/manifests/train_v2_stats.csv")
    ap.add_argument("--cap-min", type=float, default=20)
    ap.add_argument("--anchor-cap-min", type=float, default=10)
    ap.add_argument("--val-cap-min", type=float, default=10)
    ap.add_argument("--repeat", default="severe=3,moderate=2,mild=1,anchor=2")
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--merge-gap", type=float, default=3.0, help="이어 붙일 최대 쉼(초). 0이면 이어 붙이지 않는다")
    ap.add_argument("--merge-max", type=float, default=18.0, help="이어 붙인 구간 최대 길이(초). 속도 변형 1.1배 후 22초 안")
    ap.add_argument("--merge-prob", type=float, default=0.5)
    ap.add_argument("--name", default="v2", help="출력 이름: train_<name>.jsonl, val_<name>.jsonl")
    ap.add_argument("--context-pad", type=float, default=0.0, help="구간 앞뒤 여유 최대(초). 0이면 붙이지 않는다 (v4: 0.8)")
    ap.add_argument("--vad-dir", default="work/segments/vm", help="문장 나누기 1단계 VAD 구간 (여유를 붙일 때 이웃 경계)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rep = {k: int(v) for k, v in (x.split("=") for x in args.repeat.split(","))}

    train_rows = load(Path(args.segs) / "segments_train.jsonl")
    val_rows = load(Path(args.segs) / "segments_val.jsonl")
    if args.context_pad > 0:
        train_rows = add_context(train_rows, args.vad_dir, rng, args.context_pad)
        val_rows = add_context(val_rows, args.vad_dir, rng, args.context_pad)
        pads = [r.get("pad_before", 0) + r.get("pad_after", 0) for r in train_rows]
        print(f"앞뒤 여유: 평균 {sum(pads) / len(pads):.2f}초, 여유 0인 구간 {sum(p == 0 for p in pads)}개")
    if args.merge_gap > 0:
        train_rows, n_merged = merge_neighbors(train_rows, rng, args.merge_gap, args.merge_max, args.merge_prob)
        print(f"이어 붙인 구간 {n_merged}개")
    train = cap_per_speaker(train_rows, lambda t: args.anchor_cap_min if t == "anchor" else args.cap_min, rng)
    val = cap_per_speaker(val_rows, lambda t: args.val_cap_min, rng)

    train_rep = [r for r in train for _ in range(rep.get(r["tier"], 1))]
    rng.shuffle(train_rep)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in ((f"train_{args.name}.jsonl", train_rep), (f"val_{args.name}.jsonl", val)):
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
    print(f"-> {out}/train_{args.name}.jsonl ({len(train_rep)}줄), val_{args.name}.jsonl ({len(val)}줄)")


if __name__ == "__main__":
    main()
