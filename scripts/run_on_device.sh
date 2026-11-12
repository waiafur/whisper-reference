#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${REPO}/src/build/stage"
DEV_DIR=/data/local/tmp/ov-whisper
MODELS_HOST="${HOME}/whisper-export/ov"
OUT_HOST="${REPO}/reference_outputs.json"

echo "[1/4] push binaries + libs"
adb shell mkdir -p "$DEV_DIR"
adb push "$STAGE"/* "$DEV_DIR/"

echo "[2/4] push WAVs"
adb shell mkdir -p "$DEV_DIR/audio"
adb push "${REPO}/audio/prepared/russian.wav" "$DEV_DIR/audio/"
adb push "${REPO}/audio/prepared/english.wav" "$DEV_DIR/audio/"

echo "[3/4] push models (once, ~150 MB)"
adb shell mkdir -p "$DEV_DIR/models"
for m in whisper-tiny-fp16 whisper-small-fp16; do
  adb push "${MODELS_HOST}/${m}" "$DEV_DIR/models/"
done

echo "[4/4] run 4 references"
RESULTS=()
i=0
for spec in \
  "whisper-tiny-fp16 russian.wav ru <|ru|> ru-001" \
  "whisper-tiny-fp16 english.wav en <|en|> en-001" \
  "whisper-small-fp16 russian.wav ru <|ru|> ru-001" \
  "whisper-small-fp16 english.wav en <|en|> en-001" ; do
  i=$((i+1))
  set -- $spec
  model=$1; wav=$2; lang=$3; token=$4; id=$5
  echo "  -> $model / $id / $lang"
  adb shell "cd $DEV_DIR && LD_LIBRARY_PATH=$DEV_DIR ./reference_runner \
    --model-dir $DEV_DIR/models/$model --wav $DEV_DIR/audio/$wav \
    --language $lang --language-token '$token' \
    --audio-id $id --out $DEV_DIR/one.json"
  adb pull "$DEV_DIR/one.json" "${REPO}/.one_${i}.json" >/dev/null
  RESULTS+=("${REPO}/.one_${i}.json")
done

python3 - <<'PY' "${RESULTS[@]}"
import json, sys
recs = []
for p in sys.argv[1:]:
    recs.extend(json.load(open(p, encoding="utf-8")))
json.dump(recs, open("reference_outputs.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"wrote {len(recs)} records -> reference_outputs.json")
PY

rm -f "${REPO}"/.one_*.json
echo "Done: ${OUT_HOST}"