"""직접 매긴 난이도 점수로 train / val / test_v2 화자 분할을 만든다.

- test_v1 22명 중 7명(SEEN)은 테스트 파일만 남기고 나머지 녹음을 학습에 넣는다 ("들어 본 사람")
- 나머지 15명은 처음 듣는 사람으로 테스트에 그대로 둔다
- test_v2에 11·13·15 그룹 화자를 묶음별로 더 넣는다 (test_v1에 이 그룹이 없어서)
- val은 묶음(경증·중등도·중증)마다 고르게 뽑는다
- 이니셜·성별·지역이 같고 나이가 2살 이내인 ID는 같은 사람일 수 있어 한 묶음으로 같은 쪽에 둔다
- 기준점(잘 말하는 사람)은 채점에서 0~1점인 사람만 학습에 넣는다

    python data_prep/make_splits.py
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

SEEN = ["21-KMH-M26-GS", "21-LJG-M44-KK", "28-KJM-M99-SU", "26-BDS-M25-KK",
        "26-BJS-F32-JJ", "26-BHL-F25-SU", "25-OTK-M30-SU"]
EXCLUDE = {"15-KSW-M53-JL"}  # Training·Validation 양쪽에 나오는 화자
FIRST_600S = {"21-HHS-M30-SU", "26-BHN-F25-SU"}  # test_v1에서 앞 600초만 쓰는 긴 녹음
TEST_ADD_FOLDERS = ("11", "13", "15")
TEST_ADD = {"mild": 2, "moderate": 2, "severe": 2}
VAL_N = {"mild": 3, "moderate": 3, "severe": 3}
NOISY = {"녹음문제", "진행자음성"}  # val·test에 넣지 않는다
EVAL_CODES = ["03", "07", "02", "06", "04", "01"]  # 새 테스트 화자의 평가 파일 우선순위 (말하기 먼저)


def tier(score):
    s = int(score)
    return "mild" if s <= 1 else "moderate" if s <= 3 else "severe"


def parse_id(spk):
    _, ini, sa, reg = spk.split("-")
    return ini, sa[0], int(sa[1:]), reg


def clusters_of(speakers):
    """같은 사람일 수 있는 ID 묶음. 나이 99는 미상이라 묶지 않는다."""
    parent = {s: s for s in speakers}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_key = defaultdict(list)
    for s in speakers:
        ini, sex, age, reg = parse_id(s)
        by_key[(ini, sex, reg)].append((age, s))
    for items in by_key.values():
        for i, (a1, s1) in enumerate(items):
            for a2, s2 in items[i + 1:]:
                if 99 not in (a1, a2) and abs(a1 - a2) <= 2:
                    parent[find(s1)] = find(s2)
    groups = defaultdict(list)
    for s in speakers:
        groups[find(s)].append(s)
    return {s: sorted(g) for g in groups.values() for s in g}


def pick_eval_file(files):
    """새 테스트 화자의 평가용 파일 하나: 말하기 과제 우선, 3~20분 사이를 선호."""
    for code in EVAL_CODES:
        cands = [f for f in files if f["test_code"] == code]
        good = [f for f in cands if 3 <= float(f["minutes"]) <= 20]
        if good or cands:
            return sorted(good or cands, key=lambda f: float(f["minutes"]))[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="splits/listen_scores_v1.csv")
    ap.add_argument("--test", default="splits/test_v1.csv")
    ap.add_argument("--inventory", default="work/inventory/files.csv")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--seed", type=int, default=20261114)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    scores = {r["speaker"]: r for r in csv.DictReader(open(args.scores, encoding="utf-8-sig"))}
    test_v1 = {r["speaker"]: r for r in csv.DictReader(open(args.test, encoding="utf-8-sig"))}
    files = defaultdict(list)
    for r in csv.DictReader(open(args.inventory, encoding="utf-8-sig")):
        files[r["speaker"]].append(r)
    minutes = {s: sum(float(f["minutes"]) for f in fs) for s, fs in files.items()}
    tags = lambda s: set(t for t in scores[s]["tags"].split("|") if t)

    assert all(s in test_v1 for s in SEEN), "SEEN은 test_v1 화자여야 한다"
    cands = sorted(s for s, r in scores.items() if r["group"] == "candidate" and s not in EXCLUDE)
    cluster = clusters_of(cands + list(test_v1))
    for s in test_v1:  # 테스트 화자와 같은 사람일 수 있는 후보가 있으면 멈춘다
        assert cluster[s] == [s], f"테스트 화자 {s}와 묶이는 ID: {cluster[s]}"

    taken = set()

    def free_clusters(pool):
        seen, out = set(), []
        for s in pool:
            c = tuple(cluster[s])
            if c not in seen and not taken & set(c):
                seen.add(c)
                out.append(c)
        return out

    def ok_for_eval(c):
        return all(not (tags(s) & NOISY) for s in c)

    def draw(pool, n_by_tier, spread_folders):
        """묶음별로 n명씩 뽑는다. 클러스터 대표 점수는 구성원 최댓값."""
        chosen = []
        for t, n in n_by_tier.items():
            cs = [c for c in free_clusters(pool) if ok_for_eval(c)
                  and tier(max(int(scores[s]["score"]) for s in c)) == t]
            rng.shuffle(cs)
            if spread_folders:  # 질환 폴더가 겹치지 않게 돌아가며 고른다
                by_f = defaultdict(list)
                for c in cs:
                    by_f[c[0][:2]].append(c)
                order = []
                while any(by_f.values()):
                    for f in sorted(by_f):
                        if by_f[f]:
                            order.append(by_f[f].pop())
                cs = order
            for c in cs[:n]:
                chosen.append((t, c))
                taken.update(c)
            if len(cs) < n:
                print(f"주의: {t} 후보가 {len(cs)}개뿐 (요청 {n})")
        return chosen

    # 1) test_v2
    test_rows = []
    for s, r in sorted(test_v1.items()):
        test_rows.append({
            "speaker": s, "track": "seen" if s in SEEN else "unseen", "source": "test_v1",
            "folder": scores[s]["folder"], "score": scores[s]["score"], "test_v1_score": r["score"],
            "tier": tier(scores[s]["score"]), "tags": scores[s]["tags"], "eval_stem": r["stem"],
            "eval_note": "first600s" if s in FIRST_600S else ""})
    add_pool = [s for s in cands if s[:2] in TEST_ADD_FOLDERS]
    for t, c in draw(add_pool, TEST_ADD, spread_folders=True):
        for s in c:
            f = pick_eval_file(files[s])
            long = f is not None and float(f["minutes"]) > 20  # 긴 녹음은 test_v1처럼 앞 600초만
            test_rows.append({
                "speaker": s, "track": "unseen", "source": "new", "folder": scores[s]["folder"],
                "score": scores[s]["score"], "test_v1_score": "", "tier": tier(scores[s]["score"]),
                "tags": scores[s]["tags"], "eval_stem": f["stem"] if f else "",
                "eval_note": "first600s" if long else ""})

    # 2) val
    val_rows = []
    for t, c in draw(cands, VAL_N, spread_folders=True):
        for s in c:
            val_rows.append({"speaker": s, "folder": scores[s]["folder"], "score": scores[s]["score"],
                             "tier": tier(scores[s]["score"]), "tags": scores[s]["tags"],
                             "minutes": round(minutes[s], 1)})

    # 3) train = 남은 후보 + 들어 본 테스트 화자(테스트 파일 제외) + 기준점
    train_rows = []
    for s in cands:
        if s in taken:
            continue
        train_rows.append({"speaker": s, "role": "candidate", "folder": scores[s]["folder"],
                           "score": scores[s]["score"], "tier": tier(scores[s]["score"]),
                           "tags": scores[s]["tags"], "holdout_stem": "", "minutes": round(minutes[s], 1)})
    for s in SEEN:
        held = test_v1[s]["stem"]
        rest = sum(float(f["minutes"]) for f in files[s] if f["stem"] != held)
        train_rows.append({"speaker": s, "role": "seen_test", "folder": scores[s]["folder"],
                           "score": scores[s]["score"], "tier": tier(scores[s]["score"]),
                           "tags": scores[s]["tags"], "holdout_stem": held, "minutes": round(rest, 1)})
    for s, r in sorted(scores.items()):
        if r["group"] == "anchor" and r["score"] != "" and int(r["score"]) <= 1:
            train_rows.append({"speaker": s, "role": "anchor", "folder": r["folder"], "score": r["score"],
                               "tier": "anchor", "tags": r["tags"], "holdout_stem": "",
                               "minutes": round(minutes[s], 1)})

    # 겹침 검사: 화자·클러스터가 두 곳에 들어가면 안 된다 (seen_test는 의도된 예외)
    tr = {r["speaker"] for r in train_rows if r["role"] != "seen_test"}
    va = {r["speaker"] for r in val_rows}
    te = {r["speaker"] for r in test_rows if r["track"] == "unseen"}
    assert not (tr & va) and not (tr & te) and not (va & te), "분할이 겹친다"

    out = Path(args.out)
    for name, rows in (("train_v1.csv", train_rows), ("val_v1.csv", val_rows), ("test_v2.csv", test_rows)):
        with open(out / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def summary(rows, key="tier"):
        d = defaultdict(int)
        for r in rows:
            d[r[key]] += 1
        return dict(sorted(d.items()))

    print("train", len(train_rows), summary(train_rows), summary(train_rows, "role"))
    print("val  ", len(val_rows), summary(val_rows))
    print("test ", len(test_rows), summary(test_rows), summary(test_rows, "track"))
    multi = sorted({tuple(c) for c in cluster.values() if len(c) > 1})
    print(f"같은 사람 의심 묶음 {len(multi)}개:", "; ".join("/".join(c) for c in multi))


if __name__ == "__main__":
    main()
