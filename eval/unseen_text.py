"""학습 라벨에 없는 글(처음 보는 문장)에서만 CER을 잰다. 숫자만 출력한다.

test_v2 30개 중 22개는 글의 80% 이상이 학습 라벨에도 있다 (같은 낭독 대본을 다른 사람이 읽었다).
그런 부분은 모델이 글을 외워서 맞힐 수 있어서, 시연(처음 보는 말)의 성능을 과장한다.
정답의 글자마다 "학습 라벨의 6글자 조각에 들어 있나"를 표시하고, 들어 있지 않은 글자에서만 오류를 센다.

    python eval/unseen_text.py --runs yoonaj_vad ft_v1_320ms ft_v2_320ms

- 정렬은 score.py와 같다 (partial_audio는 정답 뒤쪽 무료). 삽입은 바로 뒤 정답 글자 몫으로 센다
- 처음 보는 글자가 MIN_CHARS 이상인 화자만 화자별 평균에 넣는다. 전체 글자 합(pooled)도 낸다
- yoona-J의 학습 글은 모른다. yoona-J에게는 "처음 보는 글"이 아닐 수 있다
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import DIAG, LEFT, PRIVATE_REFS, UP  # noqa: E402
from textnorm import normalize  # noqa: E402

N = 6
MIN_CHARS = 50
TRAIN_TEXT = Path.home() / "marieum-private" / "segments_train.jsonl"


def ops(hyp, ref, free):
    """정답 글자마다 오류 수 (치환·삭제 1, 삽입은 뒤 글자에 더한다)와 쓴 정답 길이."""
    h = np.frombuffer(hyp.encode("utf-32-le"), dtype=np.uint32)
    r = np.frombuffer(ref.encode("utf-32-le"), dtype=np.uint32)
    n, m = len(h), len(r)
    err = np.zeros(m + 1, dtype=np.int64)
    if n == 0 or m == 0:
        err[:m] = 1
        err[max(m - 1, 0)] += n
        return err[:m], m
    idx = np.arange(m + 1, dtype=np.int64)
    prev = idx.copy()
    dirs = np.empty((n + 1, m + 1), dtype=np.uint8)
    dirs[0, :] = LEFT
    for i in range(1, n + 1):
        diag = prev[:-1] + (r != h[i - 1])
        up = prev + 1
        tmp = up.copy()
        tmp[1:] = np.minimum(diag, up[1:])
        cur = np.minimum.accumulate(tmp - idx) + idx
        d = np.where(cur < tmp, LEFT, UP).astype(np.uint8)
        fd = np.zeros(m + 1, dtype=bool)
        fd[1:] = (cur[1:] == tmp[1:]) & (diag <= up[1:])
        d[fd] = DIAG
        d[0] = UP
        dirs[i] = d
        prev = cur
    j = int(prev.argmin()) if free else m
    used, i = j, n
    while i > 0 or j > 0:
        k = dirs[i, j] if i > 0 else LEFT
        if k == DIAG:
            err[j - 1] += int(h[i - 1] != r[j - 1])
            i, j = i - 1, j - 1
        elif k == UP:
            err[min(j, used - 1) if used else 0] += 1
            i -= 1
        else:
            err[j - 1] += 1
            j -= 1
    return err[:used], used


def seen_mask(ref, grams):
    seen = np.zeros(len(ref), dtype=bool)
    for i in range(len(ref) - N + 1):
        if ref[i:i + N] in grams:
            seen[i:i + N] = True
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="work/eval/<run>/hyps.jsonl")
    ap.add_argument("--train-text", default=str(TRAIN_TEXT))
    ap.add_argument("--split", default="splits/test_v2.csv")
    args = ap.parse_args()

    grams = set()
    for line in open(args.train_text, encoding="utf-8"):
        t = normalize(json.loads(line)["text"])
        grams.update(t[i:i + N] for i in range(len(t) - N + 1))
    refs = {j["speaker"]: j for j in map(json.loads, open(PRIVATE_REFS, encoding="utf-8"))}
    tier = {r["speaker"]: r["tier"] for r in csv.DictReader(open(args.split, encoding="utf-8-sig"))}

    for run in args.runs:
        hyps = {j["speaker"]: j["text"] for j in map(json.loads, open(f"work/eval/{run}/hyps.jsonl", encoding="utf-8"))}
        per = []
        for spk, ref in refs.items():
            rt, ht = normalize(ref["text"]), normalize(hyps.get(spk, ""))
            if ref["partial_audio"]:
                rt = rt[: len(ht) * 3 + 500]
            err, used = ops(ht, rt, ref["partial_audio"])
            new = ~seen_mask(rt[:used], grams)
            per.append((spk, tier.get(spk, ""), int(new.sum()), int(err[new].sum()), used, int(err.sum())))
        ok = [p for p in per if p[2] >= MIN_CHARS]
        pooled = sum(p[3] for p in per) / max(1, sum(p[2] for p in per))
        line = f"{run:16s} 처음 보는 글자 {sum(p[2] for p in per):6d}자 (전체의 {sum(p[2] for p in per) / sum(p[4] for p in per):.0%})" \
               f"  CER 합산 {100 * pooled:5.1f}  화자 평균(n={len(ok)}) {100 * np.mean([p[3] / p[2] for p in ok]):5.1f}"
        for t in ("mild", "moderate", "severe"):
            ts = [p for p in ok if p[1] == t]
            if ts:
                line += f"  {t} {100 * np.mean([p[3] / p[2] for p in ts]):5.1f}(n={len(ts)})"
        print(line)


if __name__ == "__main__":
    main()
