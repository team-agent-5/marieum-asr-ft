"""다른 에이전트가 Colab 실행을 맡은 뒤, 시킨 대로만 했는지 이 PC에서 검사한다.

    python eval/audit_colab_run.py --expect-head 7e53c02

검사
  1. 레포: HEAD가 맡기기 전과 같고, 추적 파일 수정·새 파일이 없다 (work/는 제외)
  2. Colab 기록 (~/.config/colab-cli/history/<세션>.jsonl, CLI가 남기는 실행 기록)
     - 올린 파일이 work/upload/ 아래 것뿐이고 정답(refs)이 없다
     - VM에서 실행한 코드가 레포의 colab/*.py와 같다 (다르면 앞부분을 보여 준다)
  3. 세션: smoke·nemo VM이 남아 있지 않다
  4. 결과 파일: 30줄, 화자가 manifest와 같고, 빈 출력이 적다
  5. 그럴듯함: 이 PC에서 채점한 CER이 비정상적으로 낮지 않다 (정답 유출 의심)
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HIST = Path.home() / ".config" / "colab-cli" / "history"
SESSIONS = ["smoke", "nemo"]
RESULTS = ["work/eval/nemotron_zs_320ms/hyps.jsonl", "work/eval/whisper_yoonaj/hyps.jsonl"]
ok = True


def report(passed, msg):
    global ok
    ok &= passed
    print(("  OK   " if passed else "  FAIL ") + msg)


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="ignore").stdout


def norm(code):
    return hashlib.sha1("\n".join(l.rstrip() for l in code.strip().splitlines()).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-head", required=True, help="맡기기 전 git HEAD (짧은 sha)")
    ap.add_argument("--min-cer", type=float, default=3.0, help="전체 CER이 이보다 낮으면 정답 유출 의심")
    args = ap.parse_args()

    print("1. 레포")
    head = run(["git", "rev-parse", "--short", "HEAD"]).strip()
    report(head.startswith(args.expect_head) or args.expect_head.startswith(head), f"HEAD {head} (기대 {args.expect_head})")
    dirty = [l for l in run(["git", "status", "--porcelain"]).splitlines() if l.strip()]
    report(not dirty, "추적 파일 변경·새 파일 없음" + ("" if not dirty else ": " + ", ".join(dirty[:10])))

    print("2. Colab 실행 기록")
    known = {norm(p.read_text(encoding="utf-8")): p.name for p in (ROOT / "colab").glob("*.py")}
    for s in SESSIONS:
        f = HIST / f"{s}.jsonl"
        if not f.exists():
            print(f"  -    {s}: 기록 없음")
            continue
        for line in open(f, encoding="utf-8"):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "op" in e:  # 파일 올리기·내려받기
                local = str(e.get("local", "")).replace("\\", "/")
                if e["op"] == "upload":
                    good = "/work/upload/" in "/" + local or local.endswith("README.md")
                    report(good and "refs" not in local, f"{s} 업로드 {local} -> {e.get('remote')}")
            elif "code" in e:
                code = e["code"]
                name = known.get(norm(code))
                short = code.strip().splitlines()[0][:80] if code.strip() else ""
                if name:
                    print(f"  OK   {s} 실행 colab/{name}")
                elif len(code) < 400 and ("print" in code or "open(" in code) and "refs" not in code:
                    print(f"  -    {s} 짧은 명령: {short}")
                else:
                    report(False, f"{s} 레포에 없는 코드 실행: {short} ({len(code)}자)")
                if "refs.jsonl" in code:
                    report(False, f"{s} 실행 코드에 refs.jsonl이 나온다")

    print("3. 남은 VM")
    live = run(["colab", "--auth=oauth2", "sessions"])
    report(not any(f"[{s}]" in live for s in SESSIONS), "smoke·nemo 세션 반납됨")

    print("4. 결과 파일")
    manifest = [json.loads(l)["speaker"] for l in open(ROOT / "work/test_v2/manifest.jsonl", encoding="utf-8")]
    for rel in RESULTS:
        p = ROOT / rel
        if not p.exists():
            print(f"  -    {rel} 없음")
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        spk = [r["speaker"] for r in rows]
        empty = sum(not r.get("text", "").strip() for r in rows)
        report(sorted(spk) == sorted(manifest), f"{rel}: {len(rows)}줄, 화자 일치")
        report(empty <= 3, f"{rel}: 빈 출력 {empty}개")
        out = p.parent
        subprocess.run([sys.executable, str(ROOT / "eval/score.py"), "--hyps", str(p), "--out", str(out)],
                       cwd=ROOT, capture_output=True)
        summ = {r["group"]: r for r in __import__("csv").DictReader(open(out / "summary.csv", encoding="utf-8-sig"))}
        cer = float(summ["all"]["cer"])
        report(cer >= args.min_cer, f"{rel}: 전체 CER {cer} (이보다 낮으면 유출 의심: {args.min_cer})")
        info = out / "run_info.json"
        if info.exists():
            print("  -    run_info:", json.load(open(info, encoding="utf-8")))

    print("\n결과:", "통과" if ok else "확인 필요 (FAIL 항목)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
