# Whisper CPU/GPU Benchmark

Compare OpenAI Whisper and faster-whisper large-v3 on NVIDIA GB10, including
CPU INT8, GPU FP16, GPU INT8/FP16, and Cantonese (`yue`) transcription.

- [Setup, Docker build, model preparation and all five configurations](standalone/README.md)
- [Original measurements and limitations](RESULTS.md)

## Quick start

Clone this repository, then build from its `standalone` directory:

```sh
git clone https://github.com/rnmanhon/whisper-cpu-gpu-benchmark.git
cd whisper-cpu-gpu-benchmark/standalone
docker build -f Dockerfile -t whisper-benchmark:gb10 ..
```

Follow the setup guide to download the models once, supply WAV files, check GPU
access and run each configuration sequentially. This Docker recipe targets
ARM64 NVIDIA GB10 (`sm_121`), with GPU access already configured on the host.

## Contents

| File | Purpose |
| --- | --- |
| `compare-runtimes.py` | Shared timing worker, warm-up, transcripts and JSONL output |
| `standalone/Dockerfile` | Independent Python/Torch/Whisper benchmark image |
| `standalone/build-ctranslate2.sh` | Pinned CUDA-enabled ARM CTranslate2 source build |
| `standalone/download-models.py` | One-time model preparation |
| `standalone/README.md` | Complete setup and benchmark commands |
| `RESULTS.md` | Original results and measurement scope |

No audio recordings or model weights are included. Supply your own WAV files.
The original results used two short human-recorded Cantonese clips; your
recordings will produce different timings and transcripts. Output JSONL can
contain spoken text and filenames, so review results before uploading them.

The independent image was prepared after the original measurements. Its primary
runtime versions and decoding settings match, but its dependency set is not a
byte-identical copy of the original environment. Treat new runs as new results.

## Upstream projects

- [OpenAI Whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- [Converted large-v3 checkpoint](https://huggingface.co/Systran/faster-whisper-large-v3)

Dependencies and downloaded models retain their respective licenses.
