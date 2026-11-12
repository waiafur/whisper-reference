#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEV_DIR=/data/local/tmp/ov-whisper

if ! adb shell test -x "$DEV_DIR/reference_runner"; then
    echo "ERROR: device is not prepared; run scripts/run_on_device.sh first" >&2
    exit 1
fi

set +e
output="$(adb shell "cd $DEV_DIR && LD_LIBRARY_PATH=$DEV_DIR ./reference_runner \
    --model-dir $DEV_DIR/models/does-not-exist \
    --wav $DEV_DIR/audio/russian.wav \
    --language ru --language-token '<|ru|>' \
    --audio-id ru-001 --out $DEV_DIR/should-not-exist.json" 2>&1)"
status=$?
set -e

echo "$output"
if [ "$status" -eq 0 ]; then
    echo "FAIL: nonexistent model directory exited with 0" >&2
    exit 1
fi
if ! grep -q "model directory not found" <<<"$output"; then
    echo "FAIL: diagnostic message missing" >&2
    exit 1
fi
echo "OK: nonexistent model directory exited $status with a diagnostic"
