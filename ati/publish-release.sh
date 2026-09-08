#!/usr/bin/env bash
# Publish the wheel built by build-wheel.sh as a GitHub release on ATIinc/onnxruntime.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${RELEASE_TAG:=v1.26.0-ati1}"
: "${RELEASE_BRANCH:=ati/v1.26.0-ati1}"
wheel=$(ls build/Linux/Release/dist/onnxruntime_gpu-*.whl)
gh release create "$RELEASE_TAG" --repo ATIinc/onnxruntime --target "$RELEASE_BRANCH" \
	--title "onnxruntime-gpu 1.26.0+ati1 (CUDA 12.8, sm_75/86/89/120)" \
	--notes "Upstream v1.26.0 rebuilt with CUDA kernels for sm_75, sm_86, sm_89 and sm_120 (SASS only, no PTX) so Blackwell GPUs such as the RTX 5060 Ti do not JIT every kernel on first inference; onnxruntime.build_and_package_info.cuda_architectures lists them so consumers can fail fast on other GPUs. Built with ati/build-wheel.sh (CUDA 12.8, cuDNN 9, Python 3.12, x86_64 Linux)." \
	"$wheel"
