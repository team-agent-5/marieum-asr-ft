#!/usr/bin/env bash
# test_v2 음성과 manifest를 Colab 업로드용 150MB 조각으로 묶는다. 정답(refs.jsonl)은 넣지 않는다.
#   bash colab/pack_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
out=work/upload
mkdir -p "$out"
rm -f "$out"/test_v2.tar.part-*
tar -cf - -C work test_v2/manifest.jsonl test_v2/audio | split -b 150M -d - "$out/test_v2.tar.part-"
ls -la "$out"
