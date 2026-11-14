#!/usr/bin/env bash
# test_v2 음성과 manifest를 Colab 업로드용 50MB 조각으로 묶는다. 정답은 넣지 않는다.
# 150MB 조각은 업로드 중 연결이 끊겼다 (11-14). 50MB도 실패하면 20M으로 줄인다.
#   bash colab/pack_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
out=work/upload
mkdir -p "$out"
rm -f "$out"/test_v2.tar.part-*
tar -cf - -C work test_v2/manifest.jsonl test_v2/audio | split -b 50M -d - "$out/test_v2.tar.part-"
ls -la "$out"
