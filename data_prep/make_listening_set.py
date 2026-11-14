"""직접 들어 난이도(0~5)를 매길 짧은 클립을 화자마다 뽑는다.

화자당 2개: 읽기(02 짧은 문장 > 06 긴 문장 > 04 > 01 단어) 1개, 말하기(03 자기소개 > 07 > 05) 1개.
말하기 과제가 없으면 다른 읽기 과제로 채운다.

녹음은 대부분 무음이고 말소리가 드문드문 있다. 그래서 앞 10분을 풀어 0.8초보다 긴 무음을 0.8초로
줄인 뒤, 말소리가 가장 많은 20초를 고른다. 말소리 비율이 낮으면 같은 역할의 다른 파일을 시도한다.
음량은 말소리 구간 기준으로 맞추고 16kHz 모노 FLAC으로 저장한다. 전사는 읽지 않는다.

출력은 work/listen/ (gitignore): clips/*.flac, listen.csv, index.html (ffmpeg 필요)

    python data_prep/make_listening_set.py --root "E:/Dataset_Backup/구음장애_음성인식_데이터"
    python data_prep/make_listening_set.py --root ... --force   # 클립을 모두 다시 만든다
"""

import argparse
import csv
import json
import random
import struct
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from inventory import zip_name

# 학습 후보 질환 폴더 앞 두 자리. 후두(3x)·전음성(27)은 학습 대상이 아니라 기준점(잘 말하는 사람)으로만 일부 듣는다
CANDIDATE = {"11", "12", "13", "15", "21", "25", "26", "28"}  # + 32.감음신경성(코드 오기로 보임)
ANCHOR = {"27": 20, "31": 10}  # 폴더 앞 두 자리: 들을 인원
EXCLUDE = {"15-KSW-M53-JL"}  # Training·Validation 양쪽에 나오는 화자
READ = ["02", "06", "04", "01"]
TALK = ["03", "07", "05"]
CLIP_S, SCAN_S, SR = 20, 600, 16000
HOP = SR // 20  # 50ms 프레임
MAX_GAP = 16  # 무음은 0.8초(16프레임)까지만 남긴다
GOOD_RATIO = 0.4  # 이보다 말소리가 적으면 다른 파일을 시도
MAX_TRIES = 4


def group_of(folder):
    if folder.startswith("32.감"):
        return "candidate"
    head = folder[:2]
    if head in CANDIDATE:
        return "candidate"
    if head in ANCHOR:
        return "anchor"
    return None


def pick_files(files):
    """역할(read/talk)마다 시도할 파일 목록 (선호 순서)."""
    rs = sorted(files, key=lambda r: r["stem"])
    order = lambda codes: [r for c in codes for r in rs if r["test_code"] == c]
    read, talk = order(READ), order(TALK)
    other = [r for r in rs if r["test_code"] not in READ + TALK]
    if not talk:  # 말하기 과제가 없으면 읽기 과제 중 앞의 것과 다른 과제부터
        talk = [r for r in read if not read or r["test_code"] != read[0]["test_code"]] + read
    read, talk = (read or other or talk), (talk or other)  # 말하기 과제만 있으면 읽기 칸도 그 파일의 다른 구간
    return {role: lst[:MAX_TRIES] for role, lst in (("read", read), ("talk", talk)) if lst}


def source_index(root):
    """원천 zip 안의 wav stem -> (zip 경로, member 이름)."""
    idx = {}
    for zp in sorted(root.rglob("*.zip")):
        if not zp.stem[:2] in ("TS", "VS"):
            continue
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                name = zip_name(info)
                if name.lower().endswith(".wav"):
                    idx.setdefault(Path(name).stem, (str(zp), info.filename))
    return idx


def wav_head_bytes(fh):
    """앞 SCAN_S초를 담을 바이트 수. 형식(16·24·32비트, μ-law, 스테레오)이 섞여 있어 헤더에서 읽는다."""
    head = fh.read(1 << 16)
    byte_rate, data_at = 48000 * 8, 44
    i = 12
    while i + 8 <= len(head):
        cid, size = head[i:i + 4], struct.unpack("<I", head[i + 4:i + 8])[0]
        if cid == b"fmt " and i + 20 <= len(head):
            byte_rate = struct.unpack("<I", head[i + 16:i + 20])[0]
        if cid == b"data":
            data_at = i + 8
            break
        i += 8 + size + (size & 1)
    return head, data_at + byte_rate * SCAN_S


def decode(src):
    fh = open(src[1], "rb") if src[0] == "disk" else zipfile.ZipFile(src[1]).open(src[2])
    with fh:
        head, need = wav_head_bytes(fh)
        data = head + fh.read(max(need - len(head), 0))
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "-", "-t", str(SCAN_S), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
        input=data, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float64)


def speech_clip(x, avoid=None):
    """무음을 줄인 뒤 말소리가 가장 많은 CLIP_S초. (오디오, 말소리 비율, 원본 시작 초, 쓴 프레임).

    avoid: 이미 다른 클립에 쓴 프레임. 같은 파일에서 두 번째 클립을 뽑을 때 겹치지 않게 한다."""
    n = len(x) // HOP
    if n < 20:
        return None, 0.0, 0.0, None
    fr = x[: n * HOP].reshape(n, HOP)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    speech = db > np.percentile(db, 10) + 15  # 잡음 바닥보다 15dB 큰 프레임
    if avoid is not None:
        speech[avoid[avoid < n]] = False
    near = np.convolve(speech, np.ones(9), "same") > 0  # 말소리 앞뒤 0.2초
    keep, gap = np.zeros(n, bool), 0
    for i in range(n):
        gap = 0 if near[i] else gap + 1
        keep[i] = gap <= MAX_GAP // 2
    # 긴 무음의 끝쪽 0.4초도 남겨 다음 말소리 앞이 자연스럽게
    for i in range(n - 2, -1, -1):
        if not keep[i] and keep[i + 1] and near[i + 1]:
            keep[max(0, i - MAX_GAP // 2 + 1): i + 1] = True
    idx = np.flatnonzero(keep)
    w = CLIP_S * SR // HOP
    if len(idx) <= w:
        start = 0
    else:
        start = int(np.convolve(speech[idx].astype(float), np.ones(w), "valid").argmax())
    sel = idx[start: start + w]
    ratio = float(speech[sel].mean()) if len(sel) else 0.0
    y = fr[sel].reshape(-1)
    loud = fr[sel][speech[sel]].reshape(-1)
    rms = np.sqrt((loud ** 2).mean()) if len(loud) else np.sqrt((y ** 2).mean() + 1e-9)
    gain = min(3276.0 / (rms + 1e-9), 32000.0 / (np.abs(y).max() + 1e-9), 300.0)  # 약 -20 dBFS
    return y * gain, ratio, (sel[0] * HOP / SR if len(sel) else 0.0), sel


def extract(job):
    """화자 한 명의 클립들 (read 먼저). 말하기 클립은 읽기와 다른 파일을 먼저 시도하고,
    같은 파일밖에 없으면 겹치지 않는 구간을 쓴다."""
    results, used, cache = [], {}, {}
    for out, tries in job["roles"]:
        try:
            tries = sorted(tries, key=lambda t: t[0] in used)  # 이미 쓴 파일은 뒤로
            best = None
            for stem, test_code, src in tries:
                if stem not in cache:
                    cache.clear()  # 메모리 절약: 직전 파일 하나만 둔다
                    cache[stem] = decode(src)
                y, ratio, t0, sel = speech_clip(cache[stem], used.get(stem))
                if y is not None and (best is None or ratio > best[1]):
                    best = (y, ratio, t0, stem, test_code, sel)
                if best and best[1] >= GOOD_RATIO:
                    break
            if best is None:
                results.append((out, None, "error decode"))
                continue
            y, ratio, t0, stem, test_code, sel = best
            used[stem] = sel if stem not in used else np.concatenate([used[stem], sel])
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-", out],
                input=np.clip(y, -32768, 32767).astype(np.int16).tobytes(), check=True)
            results.append((out, {"stem": stem, "test_code": test_code, "speech_ratio": round(ratio, 2),
                                  "start_s": round(t0)}, "ok"))
        except Exception as e:  # 한 파일 실패가 전체를 멈추지 않게
            results.append((out, None, f"error {type(e).__name__}: {e}"))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--inventory", default="work/inventory/files.csv")
    ap.add_argument("--test", default="splits/test_v1.csv")
    ap.add_argument("--out", default="work/listen")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="이미 있는 클립도 다시 만든다")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)

    files = list(csv.DictReader(open(args.inventory, encoding="utf-8-sig")))
    test = {r["speaker"]: r for r in csv.DictReader(open(args.test, encoding="utf-8-sig"))}
    by_spk = {}
    for r in files:
        by_spk.setdefault(r["speaker"], []).append(r)

    rng = random.Random(args.seed)
    speakers = []  # (speaker, group)
    anchors = {k: [] for k in ANCHOR}
    for spk, rs in sorted(by_spk.items()):
        if spk in EXCLUDE:
            continue
        if spk in test:
            speakers.append((spk, "reference"))
            continue
        g = group_of(rs[0]["folder"])
        if g == "candidate":
            speakers.append((spk, g))
        elif g == "anchor":
            anchors[rs[0]["folder"][:2]].append(spk)
    for k, n in ANCHOR.items():
        speakers += [(s, "anchor") for s in rng.sample(sorted(anchors[k]), min(n, len(anchors[k])))]

    print("원천 zip 목록 읽는 중...")
    zidx = source_index(root)
    disk = {p.stem: str(p) for p in root.rglob("*.wav")}
    src_of = lambda s: ("disk", disk[s]) if s in disk else ("zip", *zidx[s]) if s in zidx else None

    old = {}
    if (out / "listen.csv").exists() and not args.force:
        old = {r["clip"]: r for r in csv.DictReader(open(out / "listen.csv", encoding="utf-8-sig"))}

    rows, jobs = [], []
    for spk, g in speakers:
        first = by_spk[spk][0]
        t = test.get(spk, {})
        roles, spk_rows = [], []
        for role, cands in pick_files(by_spk[spk]).items():
            tries = [(r["stem"], r["test_code"], src_of(r["stem"])) for r in cands if src_of(r["stem"])]
            if not tries:
                print(f"원천 없음: {spk} {role}")
                continue
            clip = f"clips/{spk}_{role}.flac"
            spk_rows.append({"speaker": spk, "group": g, "folder": first["folder"], "role": role, "clip": clip,
                             "ref_score": t.get("score", ""), "ref_tags": t.get("tags", ""),
                             "yoonaj_train": first["yoonaj_train"]})
            roles.append((str(out / clip), tries))
        done = all(r["clip"] in old and "speech_ratio" in old[r["clip"]] and (out / r["clip"]).exists()
                   for r in spk_rows)
        if done:  # 화자 단위로 다시 만든다 (읽기·말하기가 겹치지 않게)
            for r in spk_rows:
                r.update({k: old[r["clip"]][k] for k in ("stem", "test_code", "speech_ratio", "start_s")})
        elif roles:
            jobs.append({"roles": roles})
        rows += spk_rows

    print(f"화자 {len(speakers)}명, 클립 {len(rows)}개, 다시 만들 화자 {len(jobs)}명")
    info, bad = {}, 0
    with ProcessPoolExecutor(args.workers) as ex:
        for i, results in enumerate(ex.map(extract, jobs), 1):
            for path, meta, status in results:
                if meta is None:
                    bad += 1
                    print(f"  실패 {Path(path).name}: {status}")
                else:
                    info[Path(path).name] = meta
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}")
    for r in rows:
        r.update(info.get(Path(r["clip"]).name, {}))
    rows = [r for r in rows if "speech_ratio" in r]

    out.mkdir(parents=True, exist_ok=True)
    fields = ["speaker", "group", "folder", "role", "test_code", "stem", "clip", "speech_ratio", "start_s",
              "ref_score", "ref_tags", "yoonaj_train"]
    with open(out / "listen.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tpl = (Path(__file__).parent / "listen_template.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(tpl.replace("/*__DATA__*/[]", json.dumps(rows, ensure_ascii=False)),
                                    encoding="utf-8")
    low = sum(float(r["speech_ratio"]) < 0.3 for r in rows)
    print(f"완료: 클립 {len(rows)}개, 실패 {bad}개, 말소리 30% 미만 {low}개 -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
