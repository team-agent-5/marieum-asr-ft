"""모델 출력(hyp)을 정답과 비교해 화자별·묶음별 CER을 낸다. 전사 문장은 출력하지 않고 숫자만 남긴다.

입력
  --hyps   모델 출력 jsonl. 줄마다 {"speaker": ..., "text": ...} (Colab 제로샷 스크립트 출력)
  --refs   ~/marieum-private/test_v2_refs.jsonl (extract_test_audio.py가 만든 정답. 레포 밖, 이 PC에만 있다)
  --split  splits/test_v2.csv (묶음·track·태그)
출력 (--out 폴더)
  per_speaker.csv   화자별 CER, 치환·삭제·삽입, 반복 루프 글자 수
  summary.csv       전체·묶음별·track별·test_v1 쉬움/어려움 평균

    python eval/score.py --hyps work/eval/nemotron_zs/hyps.jsonl --out work/eval/nemotron_zs

- CER은 공백 없는 글자 단위 (textnorm.normalize)
- partial_audio(앞 600초만 쓴 긴 녹음)는 정답의 뒤쪽을 무료로 건너뛰는 정렬을 쓴다.
  정답이 녹음 전체 전사라서, 들은 부분까지만 맞춰 채점한다
- 평균은 화자별 CER의 평균(macro)이다. 말을 많이 한 화자가 결과를 독차지하지 않게 한다
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from textnorm import normalize

# 정답은 레포 밖에 둔다. 레포 폴더에서 일하는 다른 에이전트가 읽거나 올리지 못하게
PRIVATE_REFS = Path.home() / "marieum-private" / "test_v2_refs.jsonl"
DIAG, UP, LEFT = 0, 1, 2  # UP = 출력에만 있는 글자(삽입), LEFT = 정답에만 있는 글자(삭제)


def align(hyp: str, ref: str, free_ref_tail: bool = False):
    """글자 단위 편집 거리와 (치환, 삭제, 삽입, 쓴 정답 길이).

    행 = 출력 글자, 열 = 정답 글자. 왼쪽 이동 비용은 누적 최솟값 트릭으로 한 줄씩 벡터 계산한다.
    free_ref_tail이면 마지막 줄에서 가장 작은 열을 고른다 (정답 뒷부분 무료)."""
    h = np.frombuffer(hyp.encode("utf-32-le"), dtype=np.uint32)
    r = np.frombuffer(ref.encode("utf-32-le"), dtype=np.uint32)
    n, m = len(h), len(r)
    if n == 0 or m == 0:
        return {"sub": 0, "del": m, "ins": n, "ref_used": m}
    idx = np.arange(m + 1, dtype=np.int64)
    prev = idx.copy()  # 0번째 줄: 정답 j글자를 모두 삭제
    dirs = np.empty((n + 1, m + 1), dtype=np.uint8)
    dirs[0, :] = LEFT
    for i in range(1, n + 1):
        diag = prev[:-1] + (r != h[i - 1])
        up = prev + 1
        tmp = up.copy()
        tmp[1:] = np.minimum(diag, up[1:])
        cur = np.minimum.accumulate(tmp - idx) + idx
        d = np.where(cur < tmp, LEFT, UP).astype(np.uint8)
        from_diag = np.zeros(m + 1, dtype=bool)
        from_diag[1:] = (cur[1:] == tmp[1:]) & (diag <= up[1:])
        d[from_diag] = DIAG
        d[0] = UP
        dirs[i] = d
        prev = cur
    j = int(prev.argmin()) if free_ref_tail else m
    ref_used = j
    i, s, dl, ins = n, 0, 0, 0
    while i > 0 or j > 0:
        k = dirs[i, j] if i > 0 else LEFT
        if k == DIAG:
            s += int(h[i - 1] != r[j - 1])
            i, j = i - 1, j - 1
        elif k == UP:
            ins += 1
            i -= 1
        else:
            dl += 1
            j -= 1
    return {"sub": s, "del": dl, "ins": ins, "ref_used": ref_used}


_LOOP = re.compile(r"(.{2,10}?)\1{2,}")


def loop_chars(hyp: str, ref: str) -> int:
    """같은 2~10글자가 3번 이상 연달아 나오는데 정답에는 그런 반복이 없는 글자 수."""
    return sum(len(m.group(0)) for m in _LOOP.finditer(hyp) if m.group(0) not in ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--refs", default=str(PRIVATE_REFS))
    ap.add_argument("--split", default="splits/test_v2.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dual", choices=["spoken", "written"], default="spoken")
    args = ap.parse_args()

    refs = {j["speaker"]: j for j in map(json.loads, open(args.refs, encoding="utf-8"))}
    hyps = {j["speaker"]: j.get("text", "") for j in map(json.loads, open(args.hyps, encoding="utf-8"))}
    meta = {r["speaker"]: r for r in csv.DictReader(open(args.split, encoding="utf-8-sig"))}

    rows = []
    for spk, ref in refs.items():
        if spk not in hyps:
            print(f"출력 없음: {spk}")
            continue
        rt, ht = normalize(ref["text"], args.dual), normalize(hyps[spk], args.dual)
        partial = ref["partial_audio"]
        if partial:  # 긴 정답 전체로 정렬하면 느리다. 출력 길이의 3배 + 여유까지만 본다
            rt = rt[: len(ht) * 3 + 500]
        a = align(ht, rt, free_ref_tail=partial)
        n_ref = a["ref_used"]
        err = a["sub"] + a["del"] + a["ins"]
        m = meta.get(spk, {})
        rows.append({
            "speaker": spk, "track": m.get("track", ""), "source": m.get("source", ""),
            "tier": m.get("tier", ""), "score": m.get("score", ""), "test_v1_score": m.get("test_v1_score", ""),
            "tags": m.get("tags", ""), "partial_audio": partial, "ref_chars": n_ref, "hyp_chars": len(ht),
            "sub": a["sub"], "del": a["del"], "ins": a["ins"],
            "cer": round(100 * err / max(n_ref, 1), 2),
            "ins_rate": round(100 * a["ins"] / max(n_ref, 1), 2),
            "loop_chars": loop_chars(ht, rt),
        })

    def mean(rs, key="cer"):
        return round(sum(float(r[key]) for r in rs) / len(rs), 2) if rs else ""

    groups = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[f"tier={r['tier']}"].append(r)
        groups[f"track={r['track']}"].append(r)
        if r["track"] == "unseen":
            groups[f"unseen,tier={r['tier']}"].append(r)
        if r["source"] == "test_v1":
            v1 = "hard" if int(r["test_v1_score"]) >= 3 else "easy"
            groups["test_v1"].append(r)
            groups[f"test_v1,{v1}"].append(r)
        for t in filter(None, r["tags"].split("|")):
            groups[f"tag={t}"].append(r)
    summary = [{"group": g, "n": len(rs), "cer": mean(rs), "ins_rate": mean(rs, "ins_rate"),
                "loop_chars": sum(r["loop_chars"] for r in rs)} for g, rs in sorted(groups.items())]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("per_speaker.csv", rows), ("summary.csv", summary)):
        with open(out / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    for s in summary:
        print(f"{s['group']:28s} n={s['n']:2d}  CER {s['cer']:6}  삽입 {s['ins_rate']:6}  반복 {s['loop_chars']}")


if __name__ == "__main__":
    main()
