"""모델의 발화 끝 판정(NeMo stop_history_eou)과 운영 VAD 구간(쉼 1.75초)을 비교한다. 숫자만 쓴다 (11-17).

colab/pipeline_eval.py 결과를 받아서 본다:
  work/eval/pipe_vad/pipe_vad.json   {화자: [[말 시작 ms, 말 끝 ms, 구간 닫힘 ms, 이유], ...]}
  work/eval/pipe_eou*/eou.json       {화자: [[확정 시각, 글자 수], ...]}  (확정 시각 = 그때까지 들어간 오디오)

    python eval/eou_compare.py --runs pipe_eou800 pipe_eou1200 pipe_eou1600 pipe_eou2000

구간 하나 = VAD가 "한 문장"으로 본 것 (말소리 사이 쉼이 1.75초 미만이면 한 구간).
  쪼갬   구간의 말 시작과 말 끝 사이에서 발화가 확정됐다 → 한 문장이 둘 이상으로 나뉜다
  합침   두 구간 사이(앞 구간 말 끝 ~ 다음 구간 말 시작)에서 확정이 없었다 → 두 문장이 하나로 붙는다
  확정 지연  쉼으로 닫힌 구간에서, 말 끝 뒤 첫 확정까지 (오디오 기준. 계산 시간은 빠져 있다)
             운영 VAD 방식은 정의상 1750ms + 계산 시간이다
경계 여유 MARGIN_MS: VAD 말 끝과 모델 확정이 이만큼 앞서도 같은 끝으로 본다.
"""

import argparse
import json
import math
from pathlib import Path

MARGIN_MS = 200


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(math.ceil(q / 100 * len(xs))) - 1)]) if xs else None


def to_ms(eou):
    """NeMo get_elapsed_duration 단위가 초인지 ms인지 값으로 판단한다 (녹음은 최대 600초)."""
    mx = max((t for v in eou.values() for t, _ in v), default=0)
    k = 1000 if mx < 5000 else 1
    return {s: [(t * k, n) for t, n in v if n > 0] for s, v in eou.items()}, ("초" if k == 1000 else "ms")


def compare(vad, eou):
    seg_n = split = gap_n = merged = 0
    delays, missing = [], 0
    for spk, segs in vad.items():
        ts = [t for t, _ in eou.get(spk, [])]
        for i, (start, end, close, reason) in enumerate(segs):
            seg_n += 1
            if any(start + MARGIN_MS < t < end - MARGIN_MS for t in ts):
                split += 1
            nxt = segs[i + 1][0] if i + 1 < len(segs) else math.inf
            after = [t for t in ts if end - MARGIN_MS <= t < nxt]
            if i + 1 < len(segs):
                gap_n += 1
                merged += not after
            if reason == "silence":
                if after:
                    delays.append(after[0] - end)
                else:
                    missing += 1
    return {"vad_segments": seg_n, "split_pct": round(100 * split / max(1, seg_n), 1),
            "merged_pct": round(100 * merged / max(1, gap_n), 1),
            "delay_ms_p50": pct(delays, 50), "delay_ms_p95": pct(delays, 95),
            "no_final_after_end": missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--vad", default="work/eval/pipe_vad/pipe_vad.json")
    args = ap.parse_args()
    vad = json.load(open(args.vad))
    print(f"VAD 구간 {sum(len(v) for v in vad.values())}개 (운영 방식: 쪼갬 0, 합침 0, 확정 지연 1750ms + 계산)")
    for run in args.runs:
        eou, unit = to_ms(json.load(open(Path("work/eval") / run / "eou.json")))
        r = compare(vad, eou)
        finals = sum(len(v) for v in eou.values())
        print(f"{run:14s} 확정 {finals:4d}개 (단위 {unit}) | 쪼갬 {r['split_pct']:5.1f}% | 합침 {r['merged_pct']:5.1f}% | "
              f"확정 지연 p50 {r['delay_ms_p50']} / p95 {r['delay_ms_p95']}ms | 말 끝 뒤 확정 없음 {r['no_final_after_end']}")


if __name__ == "__main__":
    main()
