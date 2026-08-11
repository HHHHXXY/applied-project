#!/usr/bin/env bash
# Environment for the GPU TRAINING box.
#
# This is deliberately not env/create_env.sh. That one is pinned around
# signatory, which only builds against torch 1.9 — whose newest CUDA build is
# cu111 and will not run on an sm_89/sm_90 card (4090, L40S, H100) at all.
#
# The training box does not need signatory. `build-cache` computes the
# signatures on CPU and writes plain per-date files; `train` reads those and
# never calls the signature code. So here we install a current torch, and the
# pure-PyTorch backend in gsm_alpha/signature/torch_backend.py stands in for
# signatory anywhere it is still referenced (imports, channel counts, the
# benchmark). No pins, no build step.
#
#   Usage:  bash env/create_env_gpu.sh [env_name]
#
# Verified: the full test suite passes on python 3.13 / torch 2.13 /
# pytorch-lightning 2.6.5 with no signatory present.

set -euo pipefail

ENV_NAME="${1:-gsm-gpu}"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "==> creating conda env '${ENV_NAME}' (python 3.11)"
conda create -y -n "${ENV_NAME}" python=3.11
conda activate "${ENV_NAME}"

echo "==> installing a current torch with CUDA"
# Let pip pick the default CUDA build; override the index URL if your driver
# needs a specific one, e.g. .../whl/cu121.
python -m pip install --no-cache-dir torch

echo "==> installing the training stack"
python -m pip install --no-cache-dir \
  pytorch-lightning numpy pandas pyarrow pyyaml scipy tqdm pytest

echo "==> verifying"
python - <<'PY'
import torch, pytorch_lightning as pl
print(f"torch      {torch.__version__}")
print(f"lightning  {pl.__version__}")
print(f"cuda       {torch.cuda.is_available()}", end="")
print(f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "  <-- no GPU visible")

from gsm_alpha.signature import backend_report
print(backend_report("auto"))
PY

cat <<EOF

done. next steps on this box:

  conda activate ${ENV_NAME}

  # 1. sanity: does the GPU actually help, and by how much?
  python -m gsm_alpha.cli --config configs/paper.yaml benchmark \
      --device cuda --precision 16 --stocks 1500 4300

  # 2. if the cache was built elsewhere, just point at it and train:
  python -m gsm_alpha.cli --config configs/paper.yaml train

  # 3. otherwise build it here first (CPU-bound, no GPU use):
  python -m gsm_alpha.cli --config configs/paper.yaml build-cache --threads \$(nproc)

EOF
