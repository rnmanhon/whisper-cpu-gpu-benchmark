# Build context: parent directory, containing compare-runtimes.py.
# GB10/aarch64 only. No application code, services or private configuration.
FROM python:3.11-slim@sha256:a2bc8c35469b6fe37735f7c4dae39049470b2ce068e73f799c02452de31d24c6
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.9.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu130
RUN pip install --no-cache-dir setuptools==70.0.0 wheel \
    && pip install --no-cache-dir --no-build-isolation openai-whisper==20231117 \
    && pip install --no-cache-dir numpy==1.26.4 soundfile==0.12.1 \
       faster-whisper==1.2.1 ctranslate2==4.8.5 av==18.1.0
COPY standalone/build-ctranslate2.sh /opt/build-ctranslate2.sh
RUN bash /opt/build-ctranslate2.sh
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:/usr/local/lib/python3.11/site-packages/nvidia/cu13/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib
ENV WHISPER_BENCH_AUDIO_DIR=/audio \
    WHISPER_BENCH_FASTER_MODEL=/models/faster \
    WHISPER_BENCH_OPENAI_MODEL=/models/openai/large-v3.pt
COPY compare-runtimes.py /benchmark.py
COPY standalone/download-models.py /download-models.py
WORKDIR /work
ENTRYPOINT ["python", "-u", "/benchmark.py"]
CMD ["faster", "cuda", "float16", "yue"]
