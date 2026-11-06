#!/bin/bash
# Benchmark-only CUDA build for NVIDIA GB10 (aarch64, sm_121).
set -euo pipefail
test "$(uname -m)" = aarch64
pip install --no-cache-dir cmake==4.4.4 pybind11==3.1.0 \
    nvidia-cuda-runtime==13.0.48 nvidia-cublas==13.0.0.19 nvidia-cudnn-cu13==9.13.0.50 \
    nvidia-cuda-nvcc==13.0.88 nvidia-nvvm==13.0.88 nvidia-cuda-crt==13.0.88
cuda_root=$(python -c 'import site; print(site.getsitepackages()[0] + "/nvidia/cu13")')
cudnn_root=$(python -c 'import site; print(site.getsitepackages()[0] + "/nvidia/cudnn")')
for lib in cudart cublas cublasLt; do
    ln -sf "lib${lib}.so.13" "$cuda_root/lib/lib${lib}.so"
done
git clone --depth 1 --branch v4.8.5 --recurse-submodules --shallow-submodules \
    https://github.com/OpenNMT/CTranslate2.git /tmp/asr-ctranslate2
test "$(git -C /tmp/asr-ctranslate2 rev-parse HEAD)" = 7f59ae114cba0f7d5fb1fe6099142e4c3b0037c8
# Legacy FindCUDA assumes one digit before the decimal, so it rejects 12.1.
python - <<'PY'
from pathlib import Path
import cmake
p = Path(cmake.CMAKE_DATA) / 'share/cmake-4.4/Modules/FindCUDA/select_compute_arch.cmake'
p.write_text(p.read_text().replace('[0-9]', '[0-9]+').replace('[0-9]++', '[0-9]+'))
PY
cmake -S /tmp/asr-ctranslate2 -B /tmp/asr-ctranslate2/build \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_POLICY_DEFAULT_CMP0146=OLD \
    -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF -DWITH_RUY=ON \
    -DOPENMP_RUNTIME=COMP -DCUDA_ARCH_LIST=12.1 \
    -DCUDA_TOOLKIT_ROOT_DIR="$cuda_root" -DCUDNN_INCLUDE_DIR="$cudnn_root/include" \
    -DCUDNN_LIBRARIES="$cudnn_root/lib/libcudnn.so.9" -DCMAKE_BUILD_TYPE=Release -DBUILD_CLI=OFF
cmake --build /tmp/asr-ctranslate2/build -j 8
cmake --install /tmp/asr-ctranslate2/build
CTRANSLATE2_ROOT=/usr/local CMAKE_BUILD_PARALLEL_LEVEL=4 \
    pip install --no-cache-dir --no-deps --no-build-isolation /tmp/asr-ctranslate2/python
rm -rf /tmp/asr-ctranslate2
