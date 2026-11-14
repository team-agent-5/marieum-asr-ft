"""채점 페이지에서 저장한 CSV를 레포용 splits/listen_scores_v1.csv로 바꾼다.

메모(자유 글)는 레포에 넣지 않는다. 메모의 키워드만 태그로 옮긴다.
평가불가(X 또는 메모의 '평가불가')는 usable=False로 표시한다. val·test에 쓰지 않는다.

    python data_prep/import_listen_scores.py work/listen/scores/listen_scores_2026-11-14_final.csv
"""

import argparse
import csv
from pathlib import Path

# 메모 키워드 -> 태그
MEMO_TAGS = [
    ("평가불가", "평가불가"),
    ("안내원", "진행자음성"),
    ("진행자", "진행자음성"),
    ("노이즈", "녹음문제"),
    ("에코", "녹음문제"),
    ("울림", "녹음문제"),
    ("힘이 없", "힘없음"),
    ("느리", "느림"),
    ("느려", "느림"),
    ("느림", "느림"),
    ("늘려", "느림"),
    ("떨림", "떨림"),
    ("사투리", "사투리"),
    ("혀가 짧", "ㅅ→th"),
    ("높낮이", "억양이상"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", help="채점 페이지에서 저장한 CSV")
    ap.add_argument("--test", default="splits/test_v1.csv")
    ap.add_argument("--out", default="splits/listen_scores_v1.csv")
    args = ap.parse_args()

    test = {r["speaker"]: r for r in csv.DictReader(open(args.test, encoding="utf-8-sig"))}
    rows = []
    for r in csv.DictReader(open(args.scores, encoding="utf-8-sig")):
        tags = [t for t in r["tags"].split("|") if t]
        for key, tag in MEMO_TAGS:
            if key in r["memo"] and tag not in tags:
                tags.append(tag)
        unusable = r["score"] in ("", "x") or "평가불가" in tags
        rows.append({
            "speaker": r["speaker"],
            "group": r["group"],
            "folder": r["folder"],
            "score": "" if r["score"] == "x" else r["score"],
            "tags": "|".join(t for t in tags if t != "평가불가"),
            "usable": not unusable,
            "test_v1_score": test.get(r["speaker"], {}).get("score", ""),
        })
    rows.sort(key=lambda r: (r["group"] != "reference", r["group"], r["speaker"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)}명 -> {args.out} (usable=False {sum(not r['usable'] for r in rows)}명)")


if __name__ == "__main__":
    main()
