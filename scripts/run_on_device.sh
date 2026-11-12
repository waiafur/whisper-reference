#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${REPO}/src/build/stage"
DEV_DIR=/data/local/tmp/ov-whisper
MODELS_HOST="${WHISPER_MODELS_DIR:-${HOME}/whisper-export/ov}"
OUT_HOST="${REPO}/reference_outputs.json"
INPUT_MANIFEST="${REPO}/manifests/input_manifest.json"
ENV_MANIFEST="${REPO}/manifests/environment_manifest.json"

[ -x "${STAGE}/reference_runner" ] || {
    echo "ERROR: ${STAGE}/reference_runner not found; build src/ first" >&2
    exit 1
}

model_field() { # model_field <model-dir-name> <key>
    python3 - "$ENV_MANIFEST" "$1" "$2" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))["models"][sys.argv[2]]
print(m[sys.argv[3]])
PY
}

push_if_changed() { # push_if_changed <local-file> <remote-dir>
    local src="$1" dst="$2" name local_size remote_size
    name="$(basename "$src")"
    local_size="$(stat -c %s "$src")"
    remote_size="$(adb shell stat -c %s "$dst/$name" 2>/dev/null | tr -d '\r' || true)"
    if [ "$remote_size" = "$local_size" ]; then
        echo "  -> $name up to date"
    else
        adb push "$src" "$dst/"
    fi
}

input_sha256() { # input_sha256 <audio-id> <local-wav>
    local id="$1" file="$2" expected actual
    expected="$(python3 - "$INPUT_MANIFEST" "$id" <<'PY'
import json, sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["inputs"]:
    if item["id"] == sys.argv[2]:
        print(item["prepared_wav_sha256"])
        break
else:
    sys.exit(f"no input manifest entry for {sys.argv[2]}")
PY
)"
    actual="$(sha256sum "$file" | cut -d' ' -f1)"
    if [ "$actual" != "$expected" ]; then
        echo "ERROR: $file sha256 $actual != manifest $expected" >&2
        exit 1
    fi
    echo "$actual"
}

echo "[1/4] push binaries + libs"
adb shell mkdir -p "$DEV_DIR"
for f in "$STAGE"/*; do
    push_if_changed "$f" "$DEV_DIR"
done

echo "[2/4] push WAVs"
adb shell mkdir -p "$DEV_DIR/audio"
push_if_changed "${REPO}/audio/prepared/russian.wav" "$DEV_DIR/audio"
push_if_changed "${REPO}/audio/prepared/english.wav" "$DEV_DIR/audio"

echo "[3/4] push models (only if missing)"
adb shell mkdir -p "$DEV_DIR/models"
for m in whisper-tiny-fp16 whisper-small-fp16; do
    if adb shell test -f "$DEV_DIR/models/$m/openvino_decoder_model.xml"; then
        echo "  -> $m already present"
    else
        [ -d "${MODELS_HOST}/${m}" ] || {
            echo "ERROR: model not found: ${MODELS_HOST}/${m}" >&2
            echo "       prepare it with scripts/prepare_models.sh or set WHISPER_MODELS_DIR" >&2
            exit 1
        }
        echo "  -> pushing $m"
        adb push "${MODELS_HOST}/${m}" "$DEV_DIR/models/"
    fi
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
    model_id="$(model_field "$model" model_id)"
    model_revision="$(model_field "$model" hf_revision)"
    sha="$(input_sha256 "$id" "${REPO}/audio/prepared/$wav")"
    echo "  -> $model / $id / $lang"
    adb shell "cd $DEV_DIR && LD_LIBRARY_PATH=$DEV_DIR ./reference_runner \
        --model-dir $DEV_DIR/models/$model \
        --model-id '$model_id' --model-revision '$model_revision' \
        --wav $DEV_DIR/audio/$wav --input-sha256 '$sha' \
        --language $lang --language-token '$token' \
        --audio-id $id --out $DEV_DIR/one.json"
    adb pull "$DEV_DIR/one.json" "${REPO}/.one_${i}.json" >/dev/null
    RESULTS+=("${REPO}/.one_${i}.json")
done

python3 - "$OUT_HOST" "${RESULTS[@]}" <<'PY'
import json, sys
out = sys.argv[1]
recs = []
for p in sys.argv[2:]:
    recs.extend(json.load(open(p, encoding="utf-8")))
json.dump(recs, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"wrote {len(recs)} records -> {out}")
PY

rm -f "${REPO}"/.one_*.json
echo "Done: ${OUT_HOST}"
