# Original measurements — 6 November 2026

NVIDIA GB10, 20 ARM CPU cores, eight inference threads. Two fixed human-recorded
Cantonese clips, 6.04 and 9.06 seconds. Values are mean seconds per transcription
from two measured passes after one warm-up; each row below uses `yue`.

| Runtime | Device / computation | 6.04 s clip | 9.06 s clip |
| --- | --- | ---: | ---: |
| OpenAI Whisper | CPU FP32 | 20.94 | 33.81 |
| OpenAI Whisper | GPU FP16 decoding | 1.15 | 1.94 |
| faster-whisper | CPU INT8/FP32 | 5.09 | 6.65 |
| faster-whisper | GPU FP16 | 0.61 | 0.99 |
| faster-whisper | GPU INT8/FP16 | 0.49 | 0.80 |

OpenAI Whisper 20231117; Torch 2.9.1+cu130; faster-whisper 1.2.1;
CTranslate2 4.8.5 with an ARM CUDA source build. Both use large-v3;
the faster-whisper checkpoint is stored in FP16, while OpenAI's GPU path uses
FP16 decoding with model weights stored in FP32.

Greedy decoding, temperature zero, no previous-text conditioning. faster-whisper
uses beam size one, best-of one, VAD disabled and no word timestamps. Its lazy
segment iterator is consumed inside the timer. GPU calls are synchronized.
Audio decoding, resampling, loading, installation and compilation are excluded.

The earlier `zh` experiment measured OpenAI CPU at 19.03 / 25.50 seconds
and GPU at 1.04 / 1.43 seconds. Those values must not be mixed with `yue` rows.
`yue` preserved more expected Cantonese wording in these clips, while name
errors remained. INT8 introduced punctuation and character-form differences.

## Limits

Two short clean recordings do not establish general transcription accuracy,
long-recording throughput, concurrency, energy use or cost. Configuration order
was fixed; other services remained running with speech queues empty. CPU thread
counts were not tuned. Runtime comparisons include precision/storage differences.

The public standalone recipe was added afterwards and checked with real OpenAI
GPU FP16, faster-whisper GPU FP16 and CPU INT8 transcription on one 6.04-second
WAV. These checks validate the recipe; they do not replace the original table.
The original recordings and raw transcripts are not included in this repository.
