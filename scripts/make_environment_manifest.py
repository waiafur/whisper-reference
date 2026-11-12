#!/usr/bin/env python3
"""Build manifests/environment_manifest.json from the export manifest and local builds."""
import argparse
import json
import pathlib
import platform
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
HOME = pathlib.Path.home()

DEVICE_COMMANDS = {
    "manufacturer": "adb shell getprop ro.product.manufacturer",
    "model": "adb shell getprop ro.product.model",
    "device": "adb shell getprop ro.product.device",
    "android_release": "adb shell getprop ro.build.version.release",
    "android_sdk": "adb shell getprop ro.build.version.sdk",
    "security_patch": "adb shell getprop ro.build.version.security_patch",
    "abi": "adb shell getprop ro.product.cpu.abi",
    "abilist": "adb shell getprop ro.product.cpu.abilist",
    "soc": "adb shell getprop ro.soc.model",
    "build_fingerprint": "adb shell getprop ro.build.fingerprint",
}


def read(path, default=None):
    try:
        return pathlib.Path(path).read_text(encoding="utf-8")
    except OSError:
        return default


def git_rev(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_define(text, name):
    match = re.search(rf"#define\s+{name}\s+(\S+)", text or "")
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default=str(HOME / "whisper-export"))
    parser.add_argument("--openvino-dir", default=str(HOME / "openvino-android"))
    parser.add_argument("--device-json")
    parser.add_argument("--out", default=str(ROOT / "manifests/environment_manifest.json"))
    args = parser.parse_args()

    export_dir = pathlib.Path(args.export_dir)
    ov_dir = pathlib.Path(args.openvino_dir)
    export = json.loads((export_dir / "export_manifest.json").read_text(encoding="utf-8"))

    version_header = read(
        ov_dir / "openvino-install/runtime/include/openvino/core/version.hpp", ""
    )
    ov_version = ".".join(
        parse_define(version_header, f"OPENVINO_VERSION_{part}")
        for part in ("MAJOR", "MINOR", "PATCH")
    )
    genai_version = re.search(
        r'set\(PACKAGE_VERSION "([^"]+)"\)',
        read(ov_dir / "openvino-build/OpenVINOGenAIConfigVersion.cmake", ""),
    )
    ndk_version = re.search(
        r"Pkg\.Revision\s*=\s*(\S+)",
        read(ov_dir / "android-ndk/source.properties", ""),
    )

    cache = read(ROOT / "src/build/CMakeCache.txt", "")
    def cache_value(name, default):
        match = re.search(rf"^{name}:[^=]*=(.*)$", cache, re.MULTILINE)
        return match.group(1).strip() if match else default

    device = {key: None for key in DEVICE_COMMANDS}
    device.update({"status": "pending", "capture_commands": DEVICE_COMMANDS})
    if pathlib.Path(args.out).exists():
        existing = json.loads(pathlib.Path(args.out).read_text(encoding="utf-8"))
        device.update(existing.get("device", {}))
    if args.device_json:
        device.update(json.loads(pathlib.Path(args.device_json).read_text(encoding="utf-8")))
        device["status"] = "recorded"

    models = {}
    for export_name, info in export["models"].items():
        directory = f"{export_name}-fp16"
        models[directory] = {
            "model_id": info["model_id"],
            "hf_revision": info["hf_revision"],
            "export_command": info["export_command"],
            "weight_format": info["weight_format"],
            "multilingual": info["multilingual"],
            "files": info["files"],
        }

    manifest = {
        "generated_by": "scripts/make_environment_manifest.py",
        "device": device,
        "runtime": {
            "openvino_version": ov_version,
            "openvino_git_revision": git_rev(ov_dir / "openvino"),
            "openvino_genai_version": genai_version.group(1) if genai_version else None,
            "openvino_genai_git_revision": git_rev(ov_dir / "openvino.genai"),
            "tbb_git_revision": git_rev(ov_dir / "one-tbb"),
            "android_ndk": ndk_version.group(1) if ndk_version else None,
            "android_abi": cache_value("ANDROID_ABI", "arm64-v8a"),
            "android_platform": cache_value("ANDROID_PLATFORM", "android-34"),
            "android_stl": cache_value("ANDROID_STL", "c++_shared"),
            "cmake": subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0],
            "host_os": platform.platform(),
            "host_python": platform.python_version(),
        },
        "export": {
            "python": export["python"],
            "tools": export["tools"],
        },
        "models": models,
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
