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
# Turing (RTX 20xx), Ampere (RTX 30xx), Ada (RTX 40xx), Blackwell (RTX 50xx) as SASS, plus Blackwell PTX for
# newer parts. Datacenter parts (sm_80/90/100) are deliberately omitted to keep the build time and wheel size down.
: "${CUDA_ARCHS:=75-real;86-real;89-real;120-real;120-virtual}"
: "${PARALLEL:=8}"
: "${NVCC_THREADS:=2}"

if [ ! -x .venv/bin/python ]; then
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python numpy packaging setuptools wheel ninja
fi
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export ORT_PYTHON_LOCAL_VERSION="$ATI_LOCAL_VERSION"

.venv/bin/python tools/ci_build/build.py \
	--build_dir build/Linux --config Release \
	--build_wheel --wheel_name_suffix=gpu \
	--use_cuda --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" \
	--cmake_generator Ninja --parallel "$PARALLEL" --nvcc_threads "$NVCC_THREADS" \
	--skip_submodule_sync --skip_tests --update --build \
	--cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF "CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHS"

ls -la build/Linux/Release/dist/*.whl
