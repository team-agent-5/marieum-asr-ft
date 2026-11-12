"""AI Hub 구음장애 음성인식 데이터(608) 라벨만 읽어 파일·화자 목록을 만든다.

음성은 읽지 않는다. 전사 문장은 저장·출력하지 않고 글자 수만 남긴다 (AI Hub 재배포 금지).
결과는 work/inventory/ (gitignore) 에 쓴다.

    python data_prep/inventory.py --root "E:/Dataset_Backup/구음장애_음성인식_데이터"
"""

import argparse
import csv
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

# 파일명: ID-01-15-N-KSW-04-M-53-JL, ID-02-25-N-KSM-02-01-M-45-JL, ...-KK2 (끝 숫자는 같은 화자의 추가 파일)
NAME = re.compile(r"^ID-(\d+)-(\d+)-N-([A-Za-z]+)-([\d-]+)-([MF])-(\d+)-([A-Za-z]+)(-?\d*)$")
# 기존 모델(yoona-J General) 학습에 쓰인 폴더 (CALIBRATION.md §4 학습 데이터)
YOONAJ = {("Training", "26"), ("Validation", "13"), ("Validation", "11")}


def zip_name(info):
    """AI Hub zip은 파일명이 cp949인데 UTF-8 플래그가 없어 cp437로 잘못 풀린다."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except UnicodeError:
        return info.filename


def iter_labels(root):
    """(split, 라벨 묶음 이름, 질환 폴더, 파일 stem, json dict) — 라벨 zip 기준."""
    for zp in sorted(root.rglob("*.zip")):
        group = zp.stem  # TL01_뇌신경장애 ...
        if not re.match(r"^[TV]L\d\d", group):
            continue
        split = "Training" if group.startswith("TL") else "Validation"
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                name = zip_name(info)
                if info.is_dir() or not name.lower().endswith(".json"):
                    continue
                parts = Path(name).parts
                folder = parts[-2] if len(parts) > 1 else ""
                yield split, group, folder, Path(name).stem, json.loads(z.read(info).decode("utf-8-sig"))


def audio_index(root):
    """압축을 푼 wav가 디스크에 있는지 (stem -> 경로)."""
    idx = {}
    for p in root.rglob("*.wav"):
        idx.setdefault(p.stem, str(p))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="work/inventory")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wavs = audio_index(root)
    rows, bad = [], []
    for split, group, folder, stem, d in iter_labels(root):
        m = NAME.match(stem)
        if not m:
            bad.append(stem)
            continue
        typ, sub, ini, test, sex, age, area, _ = m.groups()
        di, pi, ti = d.get("Disease_info", {}), d.get("Patient_info", {}), d.get("Test_info", {})
        rows.append({
            "split": split, "group": group, "folder": folder, "stem": stem,
            "speaker": f"{sub}-{ini.upper()}-{sex}{age}-{area.upper()}",
            "type": di.get("Type", typ), "sub": di.get("Subcategory1", sub),
            "test_code": test.split("-")[0], "test_method": ti.get("TestMethod", ""),
            "sex": sex, "age": int(age), "area": area.upper(),
            "minutes": round(float(d.get("playTime") or 0) / 60, 3),
            "sr": (d.get("Meta_info") or {}).get("SamplingRate", ""),
            "ref_chars": len((d.get("Transcript") or "").replace(" ", "")),
            "yoonaj_train": (split, sub) in YOONAJ,
            "wav_on_disk": stem in wavs,
        })

    with open(out / "files.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    spk = defaultdict(lambda: {"files": 0, "minutes": 0.0, "splits": set(), "methods": defaultdict(float)})
    for r in rows:
        s = spk[r["speaker"]]
        s["files"] += 1
        s["minutes"] += r["minutes"]
        s["splits"].add(r["split"])
        s["methods"][r["test_code"]] += r["minutes"]
        for k in ("folder", "sex", "age", "area", "yoonaj_train"):
            s[k] = r[k]
    codes = sorted({r["test_code"] for r in rows})
    with open(out / "speakers.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "folder", "sex", "age", "area", "splits", "yoonaj_train", "files", "minutes"]
                   + [f"min_{c}" for c in codes])
        for k, s in sorted(spk.items()):
            w.writerow([k, s["folder"], s["sex"], s["age"], s["area"], "+".join(sorted(s["splits"])),
                        s["yoonaj_train"], s["files"], round(s["minutes"], 1)]
                       + [round(s["methods"].get(c, 0), 1) for c in codes])

    # 요약 (숫자만)
    print(f"label files {len(rows)}, unparsed names {len(bad)}, speakers {len(spk)}")
    print(f"total hours {sum(r['minutes'] for r in rows) / 60:.1f}, wav on disk "
          f"{sum(r['minutes'] for r in rows if r['wav_on_disk']) / 60:.1f} h")
    print("\n[test_code -> TestMethod]")
    meth = defaultdict(lambda: defaultdict(float))
    for r in rows:
        meth[r["test_code"]][r["test_method"]] += r["minutes"] / 60
    for c in codes:
        print(f"  {c}: " + ", ".join(f"{m} {h:.0f}h" for m, h in meth[c].items()))
    print("\n[group / folder]  speakers  hours  (wav on disk h)")
    agg = defaultdict(lambda: [set(), 0.0, 0.0])
    for r in rows:
        a = agg[(r["group"], r["folder"])]
        a[0].add(r["speaker"])
        a[1] += r["minutes"] / 60
        a[2] += r["minutes"] / 60 if r["wav_on_disk"] else 0
    for (g, fo), (s, h, hd) in sorted(agg.items()):
        print(f"  {g:<16} {fo:<24} {len(s):>4} {h:>7.1f} ({hd:.1f})")
    both = [k for k, s in spk.items() if len(s["splits"]) > 1]
    print(f"\nspeakers in both Training and Validation: {len(both)}")
    if bad:
        print("unparsed examples:", bad[:5])


if __name__ == "__main__":
    main()
