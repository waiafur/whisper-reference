#!/usr/bin/env python3
"""m4a -> WAV 16 kHz mono PCM16 -> float32 dump (int16/32768, как read_wav) + манифест."""
import array, hashlib, json, pathlib, subprocess, sys, tempfile, wave

ROOT = pathlib.Path(__file__).resolve().parent.parent
INPUTS = [
    {"id": "ru-001", "name": "russian", "language": "ru", "language_token": "<|ru|>"},
    {"id": "en-001", "name": "english", "language": "en", "language_token": "<|en|>"},
]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def convert(src, dst):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map_metadata", "-1",
           "-fflags", "+bitexact", "-flags:a", "+bitexact",
           "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", str(dst)]
    subprocess.run(cmd, check=True)
    return " ".join(cmd)

ffv = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
items = []
for it in INPUTS:
    n = it["name"]
    src = ROOT / "audio/source" / f"{n}.m4a"
    wav = ROOT / "audio/prepared" / f"{n}.wav"
    f32 = ROOT / "audio/prepared" / f"{n}.f32"
    txt = ROOT / "transcripts" / f"{n}.txt"
    assert src.exists(), src
    assert txt.exists() and txt.read_text(encoding="utf-8").strip(), txt

    cmd = convert(src, wav)
    with tempfile.TemporaryDirectory() as td:   # проверка детерминизма конвертации
        again = pathlib.Path(td) / "again.wav"
        convert(src, again)
        assert sha256(again) == sha256(wav), "конвертация недетерминирована"

    with wave.open(str(wav), "rb") as w:
        ch, sw, sr, nf = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(nf)
    assert (ch, sw, sr) == (1, 2, 16000), (ch, sw, sr)
    pcm = array.array("h"); pcm.frombytes(raw)
    if sys.byteorder == "big": pcm.byteswap()
    flt = array.array("f", (s / 32768.0 for s in pcm))
    if sys.byteorder == "big": flt.byteswap()
    f32.write_bytes(flt.tobytes())

    items.append({
        "id": it["id"], "language": it["language"], "language_token": it["language_token"],
        "task": "transcribe",
        "origin": "own recording by the repository author (phone voice recorder)",
        "source_file": f"audio/source/{n}.m4a", "source_sha256": sha256(src),
        "prepared_wav": f"audio/prepared/{n}.wav", "prepared_wav_sha256": sha256(wav),
        "float32_dump": f"audio/prepared/{n}.f32", "float32_dump_sha256": sha256(f32),
        "transcript_file": f"transcripts/{n}.txt", "transcript_sha256": sha256(txt),
        "sample_rate_hz": sr, "channels": ch, "sample_count": len(pcm),
        "duration_s": round(len(pcm) / sr, 3),
        "wav_sample_dtype": "int16", "wav_sample_min": min(pcm), "wav_sample_max": max(pcm),
        "native_sample_dtype": "float32 (int16 / 32768)",
        "native_sample_min": min(flt), "native_sample_max": max(flt),
        "conversion": {"tool": ffv, "command": cmd},
    })
    print(it["id"], "samples:", len(pcm), "dur:", round(len(pcm) / sr, 3), "s")

out = ROOT / "manifests/input_manifest.json"
out.write_text(json.dumps({"inputs": items}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("записано", out)
