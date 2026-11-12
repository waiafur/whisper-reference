#!/usr/bin/env bash
# Download the pinned Whisper checkpoints, export them with the exact settings
# recorded in manifests/environment_manifest.json, and verify the result.
#
# Defaults:
#   export root : $HOME/whisper-export        (WHISPER_EXPORT_DIR)
#   models      : $HOME/whisper-export/ov     (WHISPER_MODELS_DIR for the runner)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_MANIFEST="${REPO}/manifests/environment_manifest.json"
EXPORT_DIR="${WHISPER_EXPORT_DIR:-${HOME}/whisper-export}"
MODELS_DIR="${WHISPER_MODELS_DIR:-${EXPORT_DIR}/ov}"
VENV="${EXPORT_DIR}/.venv"

mkdir -p "$EXPORT_DIR"

if [ ! -d "$VENV" ]; then
    echo "creating venv: $VENV"
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "installing pinned export toolchain"
pip install --upgrade pip
pip install "openvino==2026.4.0" "openvino-tokenizers==2026.4.0.0"
printf "openvino==2026.4.0\nopenvino-tokenizers==2026.4.0.0\n" > "${EXPORT_DIR}/constraints.txt"
pip install "torch==2.15.0+cpu" --index-url https://download.pytorch.org/whl/cpu
pip install -c "${EXPORT_DIR}/constraints.txt" \
    "optimum==2.3.0" \
    "optimum-intel==2.2.0" \
    "transformers==5.5.4" \
    "nncf==3.4.0" \
    "huggingface_hub==1.21.0"
pip uninstall -y torchvision torchaudio || true
pip freeze > "${EXPORT_DIR}/pip-freeze.txt"

python3 - "$ENV_MANIFEST" <<'PY'
import json
import sys
from importlib.metadata import PackageNotFoundError, version

tools = json.load(open(sys.argv[1], encoding="utf-8"))["export"]["tools"]
for name, expected in tools.items():
    try:
        actual = version(name)
    except PackageNotFoundError:
        sys.exit(f"ERROR: {name} is not installed")
    if actual != expected:
        sys.exit(f"ERROR: {name} {actual} != pinned {expected}")
print("export toolchain matches the manifest")
PY

python3 - "$ENV_MANIFEST" "$EXPORT_DIR" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

from huggingface_hub import snapshot_download

manifest_path, export_dir = sys.argv[1:3]
export_dir = pathlib.Path(export_dir)
manifest = json.load(open(manifest_path, encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


for directory, info in manifest["models"].items():
    name = directory[: -len("-fp16")]
    hf_dir = export_dir / "hf" / name
    out_dir = export_dir / "ov" / directory

    if (out_dir / "openvino_decoder_model.xml").exists():
        print(f"[{directory}] already exported, verifying")
    else:
        print(f"[{directory}] downloading {info['model_id']}@{info['hf_revision']}")
        snapshot_download(
            info["model_id"],
            revision=info["hf_revision"],
            local_dir=str(hf_dir),
            allow_patterns=["*.json", "*.txt", "*.safetensors"],
        )
        print(f"[{directory}] exporting with: {info['export_command']}")
        subprocess.run(info["export_command"], shell=True, check=True, cwd=export_dir)

    revisions_path = export_dir / "hf_revisions.json"
    revisions = {}
    if revisions_path.exists():
        revisions = json.loads(revisions_path.read_text(encoding="utf-8"))
    revisions[info["model_id"]] = {
        "revision": info["hf_revision"],
        "local_dir": str(hf_dir),
    }
    revisions_path.write_text(json.dumps(revisions, indent=2) + "\n", encoding="utf-8")

    for filename, meta in info["files"].items():
        path = out_dir / filename
        if not path.exists():
            sys.exit(f"ERROR: missing {path}")
        if path.stat().st_size != meta["bytes"]:
            sys.exit(f"ERROR: size mismatch for {path}")
        if sha256(path) != meta["sha256"]:
            sys.exit(f"ERROR: sha256 mismatch for {path}")

    config = json.load(open(out_dir / "config.json", encoding="utf-8"))
    generation = json.load(open(out_dir / "generation_config.json", encoding="utf-8"))
    expected = info["multilingual"]
    if generation.get("is_multilingual") is not True:
        sys.exit(f"ERROR: {directory} is not multilingual")
    if config["vocab_size"] != expected["vocab_size"]:
        sys.exit(f"ERROR: unexpected vocab_size for {directory}")
    if len(generation.get("lang_to_id", {})) != expected["lang_to_id_count"]:
        sys.exit(f"ERROR: unexpected lang_to_id count for {directory}")

    print(f"[{directory}] OK: {len(info['files'])} files verified, multilingual=true")

print(f"models ready: {export_dir / 'ov'}")
PY

if [ "$MODELS_DIR" != "${EXPORT_DIR}/ov" ]; then
    echo "NOTE: models were exported to ${EXPORT_DIR}/ov; the runner uses WHISPER_MODELS_DIR=$MODELS_DIR"
fi
