# Standalone GB10 benchmark container

This recipe contains only the benchmark worker and its inference dependencies.
It needs no application, web service, database, voice library or private settings.
Use it on an ARM64 NVIDIA GB10 host with Docker GPU access already configured.
It targets compute capability 12.1; it is not a general x86/Jetson recipe.

The published table was measured in an existing container. This separate recipe
was added afterwards to make the experiment reproducible without that environment.
It matches the primary runtime versions, model revision and decoding settings;
its smaller dependency set is not a byte-identical copy of the measured image.
Keep results from a new run separate from the original measurements.

## Build

From the directory containing this README, run:

```sh
docker build -f Dockerfile -t whisper-benchmark:gb10 ..
```

The build installs Python 3.11, Torch/torchaudio 2.9.1 from the CUDA 13 index,
OpenAI Whisper 20231117, faster-whisper 1.2.1, and CTranslate2 4.8.5. It compiles
CTranslate2's pinned source commit with CUDA 13, cuDNN 9 and GB10 `sm_121` support.
The build script includes the CMake architecture-parser correction. No model
weights or audio are baked into the image. Builds need internet access and ample
disk space for the CUDA packages, compiler and Torch; compilation is untimed.

## Prepare models and audio

Create two directories relative to your current working directory:

```sh
mkdir -p models audio
```

Place your WAV recordings in `audio/`. The worker uses all `*.wav` files in
filename order, averages channels and resamples to 16 kHz before timing.
Your own recordings will produce different timings and transcripts; exact
repetition of the original experiment requires its audio checksums to match.

Download both model formats once, outside any benchmark:

```sh
docker run --rm --entrypoint python \
  -v "$PWD/models:/models" \
  whisper-benchmark:gb10 /download-models.py
```

This creates `models/openai/large-v3.pt` and the pinned converted checkpoint in
`models/faster/`. It needs network access and several GB of disk space. Existing
checkpoints can instead be placed at these paths. Model preparation does not
need GPU access.

## Check GPU access

```sh
docker run --rm --gpus all --network none --entrypoint python \
  whisper-benchmark:gb10 -c 'import torch, ctranslate2; print("Torch CUDA:", torch.cuda.is_available()); print("CT2 devices:", ctranslate2.get_cuda_device_count()); print(ctranslate2.get_supported_compute_types("cuda"))'
```

Expect Torch CUDA to be true, at least one CTranslate2 CUDA device, and support
for `float16` and `int8_float16`. A driver/container GPU setup failure must be
resolved before running the GPU rows.

## Run the matrix

Run configurations sequentially on an otherwise idle machine:

```sh
docker run --rm --gpus all --network none \
  -e HF_HUB_OFFLINE=1 \
  -v "$PWD/models:/models:ro" -v "$PWD/audio:/audio:ro" \
  whisper-benchmark:gb10 openai cpu float32 yue > openai-cpu.jsonl

docker run --rm --gpus all --network none \
  -e HF_HUB_OFFLINE=1 \
  -v "$PWD/models:/models:ro" -v "$PWD/audio:/audio:ro" \
  whisper-benchmark:gb10 openai cuda float16 yue > openai-gpu.jsonl

docker run --rm --gpus all --network none \
  -e HF_HUB_OFFLINE=1 \
  -v "$PWD/models:/models:ro" -v "$PWD/audio:/audio:ro" \
  whisper-benchmark:gb10 faster cpu int8 yue > faster-cpu.jsonl

docker run --rm --gpus all --network none \
  -e HF_HUB_OFFLINE=1 \
  -v "$PWD/models:/models:ro" -v "$PWD/audio:/audio:ro" \
  whisper-benchmark:gb10 faster cuda float16 yue > faster-gpu.jsonl

docker run --rm --gpus all --network none \
  -e HF_HUB_OFFLINE=1 \
  -v "$PWD/models:/models:ro" -v "$PWD/audio:/audio:ro" \
  whisper-benchmark:gb10 faster cuda int8_float16 yue > faster-gpu-int8.jsonl
```

The CPU configurations can omit `--gpus all`. Every configuration loads its
model, performs one warm-up over all recordings, then two measured passes.
Mean only records with `kind="transcription"` and `warmup=false`, grouped by
clip and configuration. JSONL output includes versions, audio checksums,
transcripts, loading time and actual CTranslate2 computation type.
Audio decoding, resampling and model loading are outside the transcription
timer; GPU calls are synchronized and lazy segments consumed inside it.

To compare language hints, rerun a configuration with `zh` in place of `yue`,
keeping those results separate. Do not run CPU and GPU configurations in parallel.

## Recipe verification (2026-11-06)

Built this independent image on a GB10 and confirmed CUDA detection plus real
OpenAI GPU FP16, faster-whisper GPU FP16 and faster-whisper CPU INT8
transcription on one 6.04-second WAV. Each smoke test completed a warm-up and
two measured passes with network access disabled and read-only model/audio
mounts. This checks the recipe, rather than rerunning the full published matrix.
The model download helper uses the pinned Whisper package's checksum-verifying
downloader and Hugging Face snapshot download; the smoke tests reused existing
checkpoints instead of downloading them again.
