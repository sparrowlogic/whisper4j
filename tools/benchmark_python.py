#!/usr/bin/env python3
"""Benchmark Python whisper. Outputs parseable JSON matching Java Benchmark format.
Requires: pip install openai-whisper (commit cba3768 for reproducibility)."""
import json, os, sys, time

import whisper, torch

AUDIO = os.path.join(os.path.dirname(__file__), "..", "src", "test", "resources", "data", "physicsworks.wav")
MODELS = ["tiny.en", "base.en", "small.en", "medium.en", "large-v3-turbo"]

audio = whisper.load_audio(AUDIO)
duration = len(audio) / 16000.0

results = {
    "runtime": "python-whisper",
    "whisper_version": whisper.__version__,
    "torch_version": torch.__version__,
    "device": "cpu",
    "threads": torch.get_num_threads(),
    "audio": AUDIO,
    "audio_duration_s": round(duration, 1),
    "beam_size": 1,
    "models": [],
}

for name in MODELS:
    entry = {"model": name}
    try:
        t0 = time.time()
        model = whisper.load_model(name, device="cpu")
        entry["load_ms"] = int((time.time() - t0) * 1000)

        # Warmup
        whisper.transcribe(model, audio[:16000 * 5], language="en", beam_size=1, temperature=0.0)

        t0 = time.time()
        result = whisper.transcribe(model, audio, language="en", beam_size=1, temperature=0.0)
        total_s = time.time() - t0

        entry["total_ms"] = int(total_s * 1000)
        entry["rtf"] = round(duration / total_s, 1)
        entry["segments"] = len(result.get("segments", []))
        entry["text"] = result.get("text", "").strip()
        del model
    except Exception as e:
        entry["error"] = str(e)

    results["models"].append(entry)

print(json.dumps(results))
