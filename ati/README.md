# ATI onnxruntime builds

Branch `ati/v1.26.0-ati1` = upstream `v1.26.0` + `setup.py` honoring `ORT_PYTHON_LOCAL_VERSION` + these scripts.

1. `ati/build-wheel.sh` builds `onnxruntime_gpu-1.26.0+ati1-cp312-cp312-linux_x86_64.whl` (hours; CUDA 12.8 host).
2. `ati/publish-release.sh` uploads it to the GitHub release `v1.26.0-ati1`.

Consumers pin the release asset URL in their `pyproject.toml`. Bump `ati1` -> `ati2` for any rebuild.
