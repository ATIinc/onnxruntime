#!/usr/bin/env bash
# Build ATI's onnxruntime-gpu wheel: upstream v1.26.0 plus CUDA kernels for every GPU generation ATI robots
# and dev machines use, including Blackwell (sm_120), which the upstream 1.26 wheel only reaches via PTX JIT
# (about 96 s of first-inference stall per process on an RTX 5060 Ti).
#
# Host requirements: Linux x86_64, Python 3.12, CUDA toolkit 12.8 (nvcc), cuDNN 9 headers + libs, cmake >= 3.28,
# gcc 13, uv (https://docs.astral.sh/uv/). Produces build/Linux/Release/dist/onnxruntime_gpu-1.26.0+ati1-*.whl
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ATI_LOCAL_VERSION:=+ati1}"
: "${CUDA_HOME:=/usr/local/cuda-12.8}"
: "${CUDNN_HOME:=/usr}"
# Turing (RTX 20xx), Ampere (RTX 30xx), Ada (RTX 40xx), Blackwell (RTX 50xx) as SASS only: no PTX on purpose, so an
# unsupported GPU fails fast instead of JIT-compiling for minutes. Datacenter parts (sm_80/90/100) are omitted.
: "${CUDA_ARCHS:=75-real;86-real;89-real;120-real}"
: "${PARALLEL:=6}"
: "${NVCC_THREADS:=1}"
# Fused-attention contrib kernels (flash / memory-efficient attention) are for transformer LLM graphs; our exported
# graphs use plain MatMul/Softmax. Each of their kernel files needs several GB of RAM to compile per architecture
# and they OOM-killed an 8-way build on a 30 GB host, so they are off.
: "${EXTRA_DEFINES:=onnxruntime_USE_FLASH_ATTENTION=OFF onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF}"

if [ ! -x .venv/bin/python ]; then
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python numpy packaging setuptools wheel ninja
fi
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export ORT_PYTHON_LOCAL_VERSION="$ATI_LOCAL_VERSION"
export ORT_CUDA_ARCHITECTURES="$CUDA_ARCHS"

.venv/bin/python tools/ci_build/build.py \
	--build_dir build/Linux --config Release \
	--build_wheel --wheel_name_suffix=gpu \
	--use_cuda --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" \
	--cmake_generator Ninja --parallel "$PARALLEL" --nvcc_threads "$NVCC_THREADS" \
	--skip_submodule_sync --skip_tests --update --build \
	--cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF "CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHS" $EXTRA_DEFINES

ls -la build/Linux/Release/dist/*.whl
