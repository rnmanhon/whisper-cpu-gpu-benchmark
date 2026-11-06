"""Standalone Whisper benchmark. Arguments: runtime, device, precision, language.

Example: python compare-runtimes.py faster cuda float16 yue
Use WHISPER_BENCH_AUDIO_DIR for WAV files (default /audio), and
WHISPER_BENCH_FASTER_MODEL / WHISPER_BENCH_OPENAI_MODEL for model locations.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

backend, device, compute, language = sys.argv[1:]
torch.set_num_threads(8)


def emit(kind, **record):
    print(json.dumps(dict(kind=kind, backend=backend, device=device,
                          compute=compute, language=language, **record),
                     ensure_ascii=False), flush=True)


versions = {name: importlib.metadata.version(name) for name in
            ("openai-whisper", "faster-whisper", "ctranslate2", "torch")}
emit("environment", versions=versions, architecture=platform.machine(), threads=8)
clips = []
audio_directory = os.environ.get("WHISPER_BENCH_AUDIO_DIR", "/audio")
paths = sorted(Path(audio_directory).glob("*.wav"))
if not paths:
    raise SystemExit("No WAV files found in the audio directory")
for path in paths:
    name = path.name
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    speech = torch.from_numpy(data.T).mean(dim=0, keepdim=True)
    if sr != 16000:
        speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)
    audio = np.ascontiguousarray(speech[0].numpy())
    clips.append((name, audio))
    emit("clip", name=name, duration_s=len(audio)/16000,
         sha256=hashlib.sha256(path.read_bytes()).hexdigest())

start = time.perf_counter()
if backend == "faster":
    import ctranslate2
    from faster_whisper import WhisperModel

    emit("supported", compute_types=sorted(ctranslate2.get_supported_compute_types(device)))
    model = WhisperModel(os.environ.get("WHISPER_BENCH_FASTER_MODEL", "/models/faster"), device=device,
                         compute_type=compute, cpu_threads=8, num_workers=1)
    emit("load", seconds=time.perf_counter()-start,
         actual_compute_type=model.model.compute_type)

    def transcribe(audio):
        segments, _ = model.transcribe(
            audio, language=language, task="transcribe", beam_size=1, best_of=1,
            temperature=0, condition_on_previous_text=False, vad_filter=False,
            word_timestamps=False,
        )
        # faster-whisper is lazy: consuming segments belongs INSIDE the timer.
        return "".join(segment.text for segment in segments)
else:
    import whisper

    model = whisper.load_model(os.environ.get("WHISPER_BENCH_OPENAI_MODEL", "/models/openai/large-v3.pt"), device="cpu").to(device)
    emit("load", seconds=time.perf_counter()-start)

    def transcribe(audio):
        return model.transcribe(
            audio, language=language, temperature=0,
            fp16=(compute == "float16"), condition_on_previous_text=False,
        )["text"]

for run in range(3):
    for name, audio in clips:
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        text = transcribe(audio)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        emit("transcription", name=name, run=run, warmup=(run == 0),
             seconds=elapsed, rtf=elapsed/(len(audio)/16000), text=text)
