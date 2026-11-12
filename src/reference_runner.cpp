// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Reference runner for OpenVINO GenAI Whisper on Android (aarch64).
// Reads a prepared 16 kHz mono WAV, runs WhisperPipeline with an explicit
// language + transcribe task, and writes a JSON record to --out.
//
// Audio ownership: we build std::vector<float> from the WAV and pass it to
// WhisperPipeline::generate, which accepts RawSpeechInput as const&. The
// pipeline does not retain the buffer beyond the call, so this is the last
// native access; the vector is freed when generate() returns.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <openvino/runtime/properties.hpp>

#include "openvino/genai/whisper_pipeline.hpp"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

namespace fs = std::filesystem;

static std::string json_escape(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b";  break;
            case '\f': o << "\\f";  break;
            case '\n': o << "\\n";  break;
            case '\r': o << "\\r";  break;
            case '\t': o << "\\t";  break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o << b; }
                else o << c;
        }
    }
    return o.str();
}

struct WavData {
    std::vector<float> samples;  // normalized float32 in [-1, 1]
    uint32_t sample_rate = 0;
    uint16_t channels = 0;
    uint64_t sample_count = 0;
};

static bool load_wav_16k_mono(const fs::path& path, WavData& out, std::string& err) {
    drwav wav;
    if (!drwav_init_file(&wav, path.string().c_str(), nullptr)) {
        err = "dr_wav failed to open: " + path.string();
        return false;
    }
    out.sample_rate = wav.sampleRate;
    out.channels = wav.channels;
    out.sample_count = wav.totalPCMFrameCount;

    if (wav.sampleRate != 16000 || wav.channels != 1) {
        drwav_uninit(&wav);
        std::ostringstream m;
        m << "expected 16 kHz mono, got " << wav.sampleRate << " Hz / "
          << wav.channels << " ch: " << path;
        err = m.str();
        return false;
    }

    out.samples.resize(wav.totalPCMFrameCount);
    drwav_uint64 got = drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, out.samples.data());
    drwav_uninit(&wav);
    if (got != out.samples.size()) {
        err = "short read from WAV";
        return false;
    }
    return true;
}

static std::string json_string_or_null(const std::string& s) {
    return s.empty() ? std::string("null") : "\"" + json_escape(s) + "\"";
}

static const char* json_bool(bool b) {
    return b ? "true" : "false";
}

static void write_record(std::ostream& os, bool first,
                         const std::string& model_dir,
                         const std::string& model_id,
                         const std::string& model_revision,
                         const std::string& audio_id,
                         const std::string& language,
                         const std::string& language_token,
                         const std::string& wav_path,
                         const std::string& input_sha256,
                         const WavData& wav,
                         const ov::genai::WhisperGenerationConfig& cfg,
                         const ov::genai::WhisperDecodedResults& res) {
    if (!first) os << ",\n";
    os << "  {\n";
    os << "    \"model\": \"" << json_escape(model_dir) << "\",\n";
    os << "    \"model_id\": " << json_string_or_null(model_id) << ",\n";
    os << "    \"model_revision\": " << json_string_or_null(model_revision) << ",\n";
    os << "    \"audio_id\": \"" << json_escape(audio_id) << "\",\n";
    os << "    \"language\": \"" << json_escape(language) << "\",\n";
    os << "    \"language_token\": \"" << json_escape(language_token) << "\",\n";
    os << "    \"resolved_language\": \"" << json_escape(res.language) << "\",\n";
    os << "    \"task\": \"" << json_escape(cfg.task.value_or("")) << "\",\n";

    os << "    \"input\": {\n";
    os << "      \"wav\": \"" << json_escape(wav_path) << "\",\n";
    os << "      \"sha256\": " << json_string_or_null(input_sha256) << ",\n";
    os << "      \"sample_rate_hz\": " << wav.sample_rate << ",\n";
    os << "      \"channels\": " << wav.channels << ",\n";
    os << "      \"sample_count\": " << wav.sample_count << ",\n";
    os << "      \"sample_dtype\": \"float32 (int16/32768)\"\n";
    os << "    },\n";

    os << "    \"effective_parameters\": {\n";
    os << "      \"device\": \"CPU\",\n";
    os << "      \"language\": " << json_string_or_null(cfg.language.value_or("")) << ",\n";
    os << "      \"task\": " << json_string_or_null(cfg.task.value_or("")) << ",\n";
    os << "      \"return_timestamps\": " << json_bool(cfg.return_timestamps) << ",\n";
    os << "      \"word_timestamps\": " << json_bool(cfg.word_timestamps) << ",\n";
    os << "      \"num_beams\": " << cfg.num_beams << ",\n";
    os << "      \"do_sample\": " << json_bool(cfg.do_sample) << ",\n";
    os << "      \"max_new_tokens\": " << cfg.max_new_tokens << ",\n";
    os << "      \"inference_num_threads\": 1\n";
    os << "    },\n";

    std::string text;
    if (!res.texts.empty()) text = res.texts.front();
    os << "    \"text\": \"" << json_escape(text) << "\",\n";

    os << "    \"segments\": ";
    if (res.chunks.has_value() && !res.chunks->empty()) {
        os << "[\n";
        bool f = true;
        for (const auto& c : *res.chunks) {
            if (!f) os << ",\n";
            f = false;
            os << "      {\"start\": " << c.start_ts
               << ", \"end\": " << c.end_ts
               << ", \"text\": \"" << json_escape(c.text) << "\"}";
        }
        os << "\n    ]\n";
    } else {
        os << "null\n";
    }
    os << "  }";
}

int main(int argc, char** argv) {
    std::string model_dir, model_id, model_revision, wav_path, input_sha256;
    std::string language, language_token, audio_id, out_path;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << name << "\n"; std::exit(2); }
            return argv[++i];
        };
        if      (a == "--model-dir")      model_dir      = need("--model-dir");
        else if (a == "--model-id")       model_id       = need("--model-id");
        else if (a == "--model-revision") model_revision = need("--model-revision");
        else if (a == "--wav")            wav_path       = need("--wav");
        else if (a == "--input-sha256")   input_sha256   = need("--input-sha256");
        else if (a == "--language")       language       = need("--language");
        else if (a == "--language-token") language_token = need("--language-token");
        else if (a == "--audio-id")       audio_id       = need("--audio-id");
        else if (a == "--out")            out_path       = need("--out");
        else { std::cerr << "unknown arg: " << a << "\n"; return 2; }
    }
    if (model_dir.empty() || wav_path.empty() || language.empty() ||
        language_token.empty() || audio_id.empty() || out_path.empty()) {
        std::cerr << "usage: reference_runner --model-dir DIR --wav FILE "
                     "--language ru --language-token '<|ru|>' "
                     "--audio-id ru-001 --out OUT.json "
                     "[--model-id ID --model-revision SHA --input-sha256 SHA]\n";
        return 2;
    }

    // AC: nonexistent model directory -> diagnostic + nonzero exit
    if (!fs::is_directory(model_dir)) {
        std::cerr << "ERROR: model directory not found: " << model_dir << "\n";
        return 2;
    }

    WavData wav;
    std::string err;
    if (!load_wav_16k_mono(wav_path, wav, err)) {
        std::cerr << "ERROR: " << err << "\n";
        return 3;
    }

    try {
        // Single-threaded inference keeps greedy decoding bit-for-bit repeatable:
        // timestamp logits can otherwise flip on rounding differences between runs.
        ov::genai::WhisperPipeline pipe(model_dir, "CPU", ov::inference_num_threads(1));

        ov::genai::WhisperGenerationConfig cfg = pipe.get_generation_config();
        cfg.language          = language_token;   // e.g. "<|ru|>"
        cfg.task              = "transcribe";      // NOT translate
        cfg.return_timestamps = true;
        cfg.word_timestamps   = false;
        cfg.num_beams         = 1;
        cfg.do_sample         = false;
        cfg.max_new_tokens    = 448;
        pipe.set_generation_config(cfg);

        ov::genai::WhisperDecodedResults res = pipe.generate(wav.samples);
        // Per audio ownership contract: wav.samples no longer read after this.

        if (res.texts.empty() || res.texts.front().empty()) {
            std::cerr << "ERROR: empty transcription\n";
            return 4;
        }

        std::ofstream out(out_path);
        if (!out) { std::cerr << "ERROR: cannot write " << out_path << "\n"; return 5; }
        out << "[\n";
        write_record(out, true, model_dir, model_id, model_revision, audio_id,
                     language, language_token, wav_path, input_sha256, wav, cfg, res);
        out << "\n]\n";
        std::cout << "OK: " << audio_id << " / " << language << " -> " << out_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 6;
    }
}