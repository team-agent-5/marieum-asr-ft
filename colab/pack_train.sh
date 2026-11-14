#!/usr/bin/env bash
# 학습 녹음(work/train_audio, 약 4.5GB)을 브라우저로 Drive에 올릴 1GB 조각으로 묶는다. 정답은 넣지 않는다.
#   bash colab/pack_train.sh
# 올릴 곳: Google Drive  marieum-asr-ft/data/   (VM에서는 /content/drive/MyDrive/marieum-asr-ft/data)
set -euo pipefail
cd "$(dirname "$0")/.."
out=work/upload
mkdir -p "$out"
rm -f "$out"/train_audio.tar.part-*
tar -cf - -C work --exclude='*.part.flac' --exclude='*.log' --exclude='*.csv' train_audio \
  | split -b 1000M -d - "$out/train_audio.tar.part-"
ls -la "$out"/train_audio.tar.part-*
