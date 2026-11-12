# OpenVINO GenAI Whisper reference outputs

Reproducible native (C++) speech-recognition references for the multilingual
Whisper `tiny` and `small` OpenVINO models on one Android CPU device. The
records are meant to be used as golden data when checking Java/JNI and Kotlin
call sites.

## Layout

| Path | Contents |
| --- | --- |
| `src/reference_runner.cpp` | Native runner: reads a 16 kHz mono WAV, calls `WhisperPipeline::generate`, writes one JSON record. |
| `src/CMakeLists.txt` | Android (arm64-v8a) build and staging of the runner + runtime libraries. |
| `scripts/prepare_audio.py` | m4a -> WAV 16 kHz mono PCM16 -> float32 dump, plus `manifests/input_manifest.json`. |
| `scripts/prepare_models.sh` | Downloads the pinned checkpoints, exports and verifies the two OpenVINO models. |
| `scripts/run_on_device.sh` | The single command that runs tiny/small x Russian/English and writes `reference_outputs.json`. |
| `scripts/check_determinism.sh` | Runs the reference command twice and verifies identical text/timestamps. |
| `scripts/check_errors.sh` | Verifies the nonexistent-model-directory diagnostic and nonzero exit. |
| `scripts/make_environment_manifest.py` | Regenerates `manifests/environment_manifest.json`. |
| `manifests/input_manifest.json` | Input IDs, format, ranges, counts, checksums, conversion command. |
| `manifests/environment_manifest.json` | Model IDs/revisions, export settings, OpenVINO/GenAI revisions, build and device info. |
| `audio/source/` | Short original recordings (`.m4a`). |
| `audio/prepared/` | Exact prepared samples (`.wav`) and float32 dumps (`.f32`). |
| `transcripts/` | Manually checked transcripts. |
| `reference_outputs.json` | Four reference records (tiny/small x ru/en). |
| `reference_outputs.run1.json` | Previous run, kept as determinism evidence. |

Model binaries are intentionally not in Git (`models/`, `*.xml`, `*.bin` are
ignored) and must be pushed from the export directory by the run script.

## Inputs and provenance

Both recordings were made by the repository author and
are used with the author's permission. Durations are 16.021 s (Russian) and
17.536 s (English), within the 5-30 s requirement. Transcripts were written by
hand and checked against the audio; see `manifests/input_manifest.json` for
source/prepared checksums and the exact `ffmpeg` conversion command (paths are
relative to the repository root).

The native input format is the example's format: 16 kHz, mono, PCM16 WAV
samples converted to `float32` as `int16 / 32768`, range roughly `[-1, 1]`.

## Environment

All model revisions, export tool versions, runtime revisions and build settings
are recorded in `manifests/environment_manifest.json`. The reference device is
recorded there as a Xiaomi/POCO `22111317PG` (`moonstone`, SoC SM6375) running
Android 14 / SDK 34, ABI `arm64-v8a`. Build values include OpenVINO 2026.5.0,
OpenVINO GenAI 2026.5.0.0, NDK r28c, Android platform 34 and `c++_shared`.

## Models

Models are not in Git. By default the run command expects:

```text
$HOME/whisper-export/ov/whisper-tiny-fp16/
$HOME/whisper-export/ov/whisper-small-fp16/
```

Override the location with `WHISPER_MODELS_DIR=/path/to/ov`. Each directory
must contain the `openvino_*_model.xml/bin`, tokenizer/detokenizer, and config
JSON files produced by the export below.

To prepare them from scratch:

```sh
bash scripts/prepare_models.sh
```

The script creates `$HOME/whisper-export/.venv` (override with
`WHISPER_EXPORT_DIR`), installs the pinned export toolchain from
`manifests/environment_manifest.json` (OpenVINO 2026.4.0, optimum-intel 2.2.0,
OpenVINO GenAI runtime stays whatever the native build uses), downloads the
checkpoints at their pinned Hugging Face revisions, runs the exact
`export_command` recorded in the manifest, and verifies every exported file by
size and sha256 plus the multilingual metadata (`vocab_size=51865`, 99 language
tokens, `is_multilingual=true`).

| Model | HF revision | Output directory |
| --- | --- | --- |
| `openai/whisper-tiny` | `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` | `whisper-tiny-fp16` |
| `openai/whisper-small` | `973afd24965f72e36ca33b3055d56a652f456b4d` | `whisper-small-fp16` |

Re-running the script is safe: existing exports are verified, not re-exported.
The pinned values live in `manifests/environment_manifest.json`, so they stay in
sync with the reference records.

## Build

Prerequisites: OpenVINO + OpenVINO GenAI built for Android aarch64 with the CPU
plugin, oneTBB, Android NDK r28c, `dr_wav.h`, and `adb`. The native OpenVINO
build is assumed to follow the existing build guide (it is not scripted here);
the exact runtime revisions are recorded in the environment manifest. Paths
default to `$HOME/openvino-android` (see `src/CMakeLists.txt`).

```sh
cmake -S src -B src/build \
  -DCMAKE_TOOLCHAIN_FILE="$HOME/openvino-android/android-ndk/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DANDROID_STL=c++_shared \
  -DCMAKE_BUILD_TYPE=Release
cmake --build src/build -j
```

The build stages the runner, `libc++_shared.so`, TBB, `libopenvino.so`,
`libopenvino_genai.so`, the CPU plugin, the IR/GGUF frontends and the
tokenizers in `src/build/stage/`, so the bundle is complete on a clean machine.

## Run (one command)

```sh
bash scripts/run_on_device.sh
```

It pushes the staged binary/libraries and WAVs, pushes each model only if it is
not already on the device, then runs the four combinations and merges the
records into `reference_outputs.json`. Model binaries are expected in
`$WHISPER_MODELS_DIR` (default `$HOME/whisper-export/ov`).

Each record contains the model path/ID/revision, audio ID, exact input checksum
and format, effective decoding parameters, resolved language, text and
timestamps (`segments`; timestamps are supported by this pinned build).

## Determinism

```sh
bash scripts/check_determinism.sh
```

It saves the current output as `reference_outputs.run1.json`, reruns all four
references on the same device, and fails unless text and timestamps are
identical. The runner pins `inference_num_threads=1`: with the default thread
count, timestamp logits occasionally flipped between runs even though the text
stayed the same. The current `run1`/`reference_outputs` pair is identical.

## Error behavior

```sh
bash scripts/check_errors.sh
```

A nonexistent model directory must print
`ERROR: model directory not found: ...` and exit nonzero (the runner returns 2).

## Audio ownership and interruption contract

- **Sample ownership.** The recordings and prepared samples are owned by the
  repository author and included solely as test fixtures, with permission.
- **Last native input access.** The runner builds a `std::vector<float>` and
  passes it to `WhisperPipeline::generate`, which accepts `RawSpeechInput` by
  `const&`. The runner does not touch the buffer after `generate()` returns;
  it is freed immediately afterwards (`reference_runner.cpp`).
- **Interruption mechanism.** The pinned API exposes only the optional
  `StreamerVariant` argument of `WhisperPipeline::generate`
  (`openvino/genai/whisper_pipeline.hpp`). A streamer's `write()` may return
  `ov::genai::StreamingStatus::STOP` or `CANCEL`
  (`openvino/genai/streamer_base.hpp`) to stop generation. There is no separate
  cancel token or timeout. This task does not use interruption; long
  recordings, microphone input and cancellation are out of scope.

## Known limitations

- The `tiny` Russian transcription contains ASR errors and intentionally does
  not match the hand-written transcript; the `small` Russian result matches
  apart from `её`/`ее`. English matches the transcript for both models.
- Timestamps: `segments` are present; if a future build reports no
  chunks, the runner writes `segments: null`.
- The `small` English model may predict a final segment end (18.68 s) beyond
  the 17.536 s clip; timestamps come directly from the model.
