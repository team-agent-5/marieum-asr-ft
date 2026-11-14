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
    """한 파일의 구간들에 정답 문장을 붙인다.

    반환: (items, ctx). items[k] = {s, text, cer, a, b, first, last} (정답과 하나도 안 맞으면 text=None)
    a~b: 붙인 정답 범위 (CER용 글자 위치). first/last: 이 구간 받아쓰기가 실제로 맞힌 첫·마지막 정답 위치"""
    clean = clean_for_training(ref_raw)
    rnorm, rpos = norm_with_map(clean)
    if ref_cap is not None:
        rnorm, rpos = rnorm[:ref_cap], rpos[:ref_cap]
    hyps = [normalize(s["hyp"]) for s in segs]
    ctx = {"clean": clean, "rnorm": rnorm, "rpos": rpos, "hyps": hyps}
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
    items = []
    for k, s in enumerate(segs):
        if k not in first:
            items.append({"s": s, "text": None, "cer": None, "a": None, "b": None, "first": None, "last": None})
            continue
        nxt = [st for kk, st in starts if kk > k]
        a = first[k]
        b = max(nxt[0] if nxt else last[k] + 1, last[k] + 1)
        text, cer = span_text(ctx, k, a, b)
        items.append({"s": s, "text": text, "cer": cer, "a": a, "b": b, "first": first[k], "last": last[k]})
    return items, ctx


def span_text(ctx, k, a, b):
    """정답 범위 a~b의 학습용 문장과, 구간 k 받아쓰기와의 CER. 너무 짧으면 (None, None)"""
    if b - a < MIN_REF_CHARS:
        return None, None
    span = ctx["rnorm"][a:b]
    al = align(ctx["hyps"][k], span)
    cer = (al["sub"] + al["del"] + al["ins"]) / max(len(span), 1)
    return ctx["clean"][ctx["rpos"][a]: ctx["rpos"][b - 1] + 1].strip(" ,"), round(cer, 3)


SANDWICH_TIERS = {"moderate", "severe"}
CPS_SANDWICH = (1.0, 10.0)  # 순서로 살릴 때는 더 엄격하게: 진행자 말 사이 짧은 틈이 들어오지 않게


def judge(assigned, tier, sandwich=True):
    """assign 결과에 남길지 판단을 붙인다. [{text, cer, keep, how, dur}] (how: match | sandwich)

    sandwich: 받아쓰기가 너무 엉망이라 버려질 구간이라도, 바로 앞뒤 구간이 확실히 맞았다면
    "앞 구간이 마지막으로 맞힌 글자 다음 ~ 뒤 구간이 처음 맞힌 글자 앞"이 이 구간의 정답이다.
    이 구간 받아쓰기는 위치 계산에 쓰지 않는다 (엉망인 받아쓰기가 우연히 맞힌 글자로 범위가 어긋나지 않게,
    11-15 합성 점검에서 이 문제로 0% 정확했다). 앞 구간 범위도 마지막으로 맞힌 글자까지로 줄인다."""
    items, ctx = assigned
    out = []
    for it in items:
        dur = it["s"]["end"] - it["s"]["start"]
        ok = False
        if it["text"] is not None:
            cps = len(normalize(it["text"])) / max(dur, 1e-6)
            ok = it["cer"] <= MAX_CER.get(tier, 0.4) and CPS[0] <= cps <= CPS[1]
        out.append({"text": it["text"], "cer": it["cer"], "keep": ok, "how": "match", "dur": dur})
    # 중등도·중증에만 쓴다. 합성 점검에서 살린 구간의 약 90%가 정확했지만 진행자 구간의 약 1.2%가 잘못 들어왔다.
    # 경증은 이미 84%가 남아서 이 위험을 감수할 이유가 없다
    if sandwich and tier in SANDWICH_TIERS:
        # 같은 녹음에서 확실히 맞은 구간들의 말 속도. 살릴 구간도 이와 비슷해야 한다
        # (진행자 말 구간에는 앞 구간이 빠뜨린 2~3글자 틈만 붙어서 말 속도가 터무니없이 느리게 나온다)
        rates = sorted(len(normalize(r["text"])) / max(r["dur"], 1e-6) for r in out if r["keep"])
        med = rates[len(rates) // 2] if rates else None
        for k in range(1, len(out) - 1):
            if out[k]["keep"] or med is None:
                continue
            prev, nxt = out[k - 1], out[k + 1]
            if not (prev["keep"] and prev["how"] == "match" and nxt["keep"] and nxt["how"] == "match"):
                continue
            a, b = items[k - 1]["last"] + 1, items[k + 1]["first"]
            text, cer = span_text(ctx, k, a, b)
            if text is None:
                continue
            cps = len(normalize(text)) / max(out[k]["dur"], 1e-6)
            if not (CPS_SANDWICH[0] <= cps <= CPS_SANDWICH[1] and 0.4 * med <= cps <= 2.5 * med):
                continue
            out[k].update(text=text, cer=cer, keep=True, how="sandwich")
            ptext, pcer = span_text(ctx, k - 1, items[k - 1]["a"], a)  # 앞 구간이 이 구간 정답을 먹지 않게
            if ptext is not None:
                prev.update(text=ptext, cer=pcer)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segs", required=True)
    ap.add_argument("--segs2", help="2차 길잡이 폴더 (colab/retranscribe_segments.py 출력). 같은 구간을 다른 모델로 받아쓴 것")
    ap.add_argument("--no-sandwich", dest="sandwich", action="store_false",
                    help="앞뒤 구간이 남았을 때 순서로 위치를 정해 살리는 규칙을 끈다")
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
    by_file2 = defaultdict(dict)  # 2차 길잡이 (시험 학습 모델 받아쓰기). 같은 구간 경계, hyp만 다르다
    if args.segs2:
        for f in sorted(Path(args.segs2).glob("*.jsonl")):
            for line in open(f, encoding="utf-8"):
                s = json.loads(line)
                by_file2[s["stem"]][(round(s["start"], 2), round(s["end"], 2))] = s

    src_count = defaultdict(lambda: defaultdict(float))
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

        passes = [("p1", judge(assign(segs, ref["text"], cap), tier, args.sandwich))]
        alt = by_file2.get(stem)
        if alt:
            segs2 = [alt.get((round(s["start"], 2), round(s["end"], 2)), {**s, "hyp": ""}) for s in segs]
            passes.append(("p2", judge(assign(segs2, ref["text"], cap), tier, args.sandwich)))

        for k, s in enumerate(segs):
            dur = s["end"] - s["start"]
            st["segs"] += 1
            st["speech_min"] += dur / 60
            # 길잡이 중 하나라도 위치를 확실히 찾았으면 남긴다. 둘 다면 받아쓰기와 더 잘 맞은 쪽
            cands = [(name, r[k]) for name, r in passes if r[k]["keep"]]
            if not cands:
                st["no_match" if all(r[k]["text"] is None for _, r in passes) else "dropped"] += 1
                continue
            name, r = min(cands, key=lambda c: (c[1]["how"] != "match", c[1]["cer"]))
            src = name if r["how"] == "match" else f"{name}-sandwich"
            seg_cers[meta["speaker"]].append(r["cer"])
            st["kept"] += 1
            st["kept_min"] += dur / 60
            src_count[tier][src] += dur / 3600
            manifests[meta["split"]].append({
                "audio_filepath": f"{args.audio_root}/{meta['speaker']}/{stem}.flac",
                "offset": s["start"], "duration": round(dur, 2), "text": r["text"],
                "target_lang": "ko-KR", "lang": "ko-KR",
                "speaker": meta["speaker"], "tier": tier, "seg_cer": r["cer"], "src": src,
                "stem": stem, "seg_idx": k})  # 이어 붙이기용: 같은 파일에서 바로 다음 구간인지 안다

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
    print("남긴 구간의 출처 (시간 h): p1=yoona-J 길잡이, p2=시험 학습 모델 길잡이, -sandwich=앞뒤 순서로 위치를 정함")
    for t in sorted(src_count):
        print(f"  {t:8s} " + "  ".join(f"{k} {v:.2f}" for k, v in sorted(src_count[t].items())))


if __name__ == "__main__":
    main()
