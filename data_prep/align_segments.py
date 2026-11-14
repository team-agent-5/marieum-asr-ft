"""문장 나누기 2단계 (이 PC): VM이 만든 말소리 구간 받아쓰기를 녹음 전체 정답과 정렬해, 구간마다 정답 문장을 붙인다.

입력
  --segs       VM 출력 폴더 (colab/segment_transcribe.py, 화자별 jsonl: speaker, stem, start, end, hyp)
  --refs       ~/marieum-private/train_refs.jsonl (녹음 전체 정답, 레포 밖)
  --selection  work/train_audio/selection.csv (split, tier, used_minutes, file_minutes)
출력
  ~/marieum-private/segments_{train,val}.jsonl   학습 manifest (NeMo 형식, offset/duration으로 원본 FLAC을 가리킨다)
  work/segments/align_stats.csv                  화자별 숫자 (구간 수, 남긴 구간·분량, 구간 CER 중앙값)

방법
  1. 파일 하나의 구간 받아쓰기를 시간 순서로 이어 붙여, 정답과 글자 단위로 정렬한다 (정답 뒤쪽은 무료로 건너뜀:
     30분만 잘라 쓴 녹음은 정답이 더 길다)
  2. 구간 i의 정답 범위 = 구간 i에서 처음 맞은 정답 글자 ~ 다음 구간에서 처음 맞은 정답 글자 앞까지
     (마지막 구간은 마지막으로 맞은 글자까지). 하나도 안 맞은 구간은 버린다 (진행자 말, 잡음)
  3. 구간마다 받아쓰기와 붙인 정답의 CER을 재서, 묶음별 기준보다 나쁘면 버린다.
     중증은 받아쓰기(yoona-J) 자체가 부정확해서 기준을 느슨하게 둔다

    python data_prep/align_segments.py --segs work/segments/vm
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from score import align  # noqa: E402
from textnorm import clean_for_training, norm_with_map, normalize  # noqa: E402

PRIVATE = Path.home() / "marieum-private"
MAX_CER = {"anchor": 0.2, "mild": 0.3, "moderate": 0.4, "severe": 0.6}  # 구간 CER 기준 (받아쓰기 vs 붙인 정답)
MIN_REF_CHARS = 2
CPS = (0.3, 15.0)  # 초당 글자 수 허용 범위. 구음장애는 느려서 하한을 낮게 둔다

DIAG, UP, LEFT = 0, 1, 2


def align_path(hyp: str, ref: str):
    """출력 글자마다 짝지어진 정답 위치(삽입이면 -1). 정답 뒤쪽은 무료. score.align과 같은 계산에 경로만 더한다."""
    h = np.frombuffer(hyp.encode("utf-32-le"), dtype=np.uint32)
    r = np.frombuffer(ref.encode("utf-32-le"), dtype=np.uint32)
    n, m = len(h), len(r)
    match = np.full(n, -1, dtype=np.int64)
    if n == 0 or m == 0:
        return match
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
        from_diag = np.zeros(m + 1, dtype=bool)
        from_diag[1:] = (cur[1:] == tmp[1:]) & (diag <= up[1:])
        d[from_diag] = DIAG
        d[0] = UP
        dirs[i] = d
        prev = cur
    i, j = n, int(prev.argmin())
    while i > 0 and j >= 0:
        k = dirs[i, j] if j > 0 else UP
        if k == DIAG:
            if h[i - 1] == r[j - 1]:
                match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif k == UP:
            i -= 1
        else:
            j -= 1
    return match


def assign(segs, ref_raw, ref_cap=None):
    """한 파일의 구간들에 정답 문장을 붙인다. [(seg, text, seg_cer)] (버린 구간은 text=None)."""
    clean = clean_for_training(ref_raw)
    rnorm, rpos = norm_with_map(clean)
    if ref_cap is not None:
        rnorm, rpos = rnorm[:ref_cap], rpos[:ref_cap]
    hyps = [normalize(s["hyp"]) for s in segs]
    owner = np.concatenate([np.full(len(h), k) for k, h in enumerate(hyps)]) if hyps else np.array([])
    match = align_path("".join(hyps), rnorm)
    first = {}
    last = {}
    for pos, k in enumerate(owner):
        j = match[pos]
        if j >= 0:
            first.setdefault(int(k), int(j))
            last[int(k)] = int(j)
    starts = sorted(first.items())
    out = []
    for k, s in enumerate(segs):
        if k not in first:
            out.append((s, None, None))
            continue
        nxt = [st for kk, st in starts if kk > k]
        a = first[k]
        b = nxt[0] if nxt else last[k] + 1
        b = max(b, last[k] + 1)
        span = rnorm[a:b]
        al = align(hyps[k], span)
        cer = (al["sub"] + al["del"] + al["ins"]) / max(len(span), 1)
        text = clean[rpos[a]: rpos[b - 1] + 1].strip(" ,")
        out.append((s, text if len(span) >= MIN_REF_CHARS else None, round(cer, 3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segs", required=True)
    ap.add_argument("--refs", default=str(PRIVATE / "train_refs.jsonl"))
    ap.add_argument("--selection", default="work/train_audio/selection.csv")
    ap.add_argument("--audio-root", default="/content/train_audio", help="manifest에 쓸 VM 쪽 오디오 경로")
    ap.add_argument("--out", default=str(PRIVATE))
    ap.add_argument("--stats", default="work/segments/align_stats.csv")
    args = ap.parse_args()

    refs = {j["stem"]: j for j in map(json.loads, open(args.refs, encoding="utf-8"))}
    sel = {r["stem"]: r for r in csv.DictReader(open(args.selection, encoding="utf-8-sig"))}
    by_file = defaultdict(list)
    for f in sorted(Path(args.segs).glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            s = json.loads(line)
            by_file[s["stem"]].append(s)

    manifests = defaultdict(list)
    stats = defaultdict(lambda: defaultdict(float))
    seg_cers = defaultdict(list)
    for stem, segs in sorted(by_file.items()):
        meta, ref = sel[stem], refs[stem]
        segs.sort(key=lambda s: s["start"])
        cap = None
        if meta["truncated"] in ("1", "True", "true"):  # 잘라 쓴 녹음: 들은 비율의 1.5배 + 여유까지만 정답을 본다
            frac = float(meta["used_minutes"]) / max(float(meta["file_minutes"]), 1e-6)
            cap = int(len(normalize(ref["text"])) * min(1.0, frac * 1.5)) + 300
        tier = meta["tier"]
        st = stats[meta["speaker"]]
        st["split"], st["tier"] = meta["split"], tier
        for s, text, cer in assign(segs, ref["text"], cap):
            dur = s["end"] - s["start"]
            st["segs"] += 1
            st["speech_min"] += dur / 60
            if text is None:
                st["no_match"] += 1
                continue
            seg_cers[meta["speaker"]].append(cer)
            cps = len(normalize(text)) / max(dur, 1e-6)
            if cer > MAX_CER.get(tier, 0.4) or not CPS[0] <= cps <= CPS[1]:
                st["dropped"] += 1
                continue
            st["kept"] += 1
            st["kept_min"] += dur / 60
            manifests[meta["split"]].append({
                "audio_filepath": f"{args.audio_root}/{meta['speaker']}/{stem}.flac",
                "offset": s["start"], "duration": round(dur, 2), "text": text,
                "target_lang": "ko-KR", "lang": "ko-KR",
                "speaker": meta["speaker"], "tier": tier, "seg_cer": cer})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in manifests.items():
        with open(out / f"segments_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    fields = ["speaker", "split", "tier", "segs", "no_match", "dropped", "kept", "speech_min", "kept_min", "median_seg_cer"]
    with open(args.stats, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for spk, st in sorted(stats.items()):
            c = sorted(seg_cers[spk])
            w.writerow({"speaker": spk, **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in st.items()},
                        "median_seg_cer": c[len(c) // 2] if c else ""})

    tot = defaultdict(lambda: [0, 0.0, 0.0, 0])
    for spk, st in stats.items():
        t = tot[(st["split"], st["tier"])]
        t[0] += 1
        t[1] += st["speech_min"] / 60
        t[2] += st["kept_min"] / 60
        t[3] += int(st["kept"])
    print("split·묶음  화자  말소리h  남긴h  남긴구간")
    for (sp, ti), (n, sh, kh, ks) in sorted(tot.items()):
        print(f"{sp:5s} {ti:8s} {n:4d} {sh:7.1f} {kh:6.1f} {ks:7d}")
    print("manifest:", ", ".join(f"{k} {len(v)}개" for k, v in manifests.items()), "->", out)


if __name__ == "__main__":
    main()
