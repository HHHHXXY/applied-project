#!/usr/bin/env bash
# Build the conda environment this project runs in.
#
# The binding constraint is signatory: it is a C++ extension compiled against a
# specific PyTorch, and the last release (1.2.6.1.9.0) targets torch 1.9.0, so
# the whole stack is pinned around that. Everything else follows from it:
#
#   python 3.9      torch 1.9 publishes no wheels for 3.10+
#   pip < 24.1      pytorch-lightning 1.6.5 ships legacy metadata newer pip rejects
#   setuptools 59.5 torch 1.9's tensorboard shim reads distutils.version, removed in 60
#   torchmetrics<1  1.x requires torch >= 1.10
#   numpy < 2       torch 1.9 was built against the numpy 1.x C API
#
# Order matters twice over: signatory needs torch importable *before* it builds
# (hence --no-build-isolation), and installing pytorch-lightning after torch
# would drag in a modern torch, so torch is reinstalled and pinned last.
#
# Usage:  bash env/create_env.sh [env_name]
#
# The project also runs on a plain modern PyTorch with no signatory at all — the
# pure-PyTorch signature backend in gsm_alpha/signature/torch_backend.py takes
# over automatically. It is slower, and tests/test_signature_backend.py is what
# pins it to signatory's output.

set -euo pipefail

ENV_NAME="${1:-gsm}"
TORCH_VERSION="1.9.0"
SIGNATORY_VERSION="1.2.6.1.9.0"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "==> creating conda env '${ENV_NAME}' (python 3.9)"
conda create -y -n "${ENV_NAME}" python=3.9
conda activate "${ENV_NAME}"

echo "==> pinning pip and setuptools"
python -m pip install -q "pip<24.1" "setuptools==59.5.0" wheel

echo "==> installing torch ${TORCH_VERSION} (cpu)"
python -m pip install --no-cache-dir "torch==${TORCH_VERSION}+cpu" \
  -f https://download.pytorch.org/whl/torch_stable.html

echo "==> building signatory ${SIGNATORY_VERSION} against the installed torch"
python -m pip install --no-cache-dir --no-build-isolation "signatory==${SIGNATORY_VERSION}"

echo "==> installing the training and data stack"
python -m pip install --no-cache-dir \
  "pytorch-lightning==1.6.5" \
  "torchmetrics==0.9.3" \
  "numpy<2" \
  "pandas<2.1" \
  "pyarrow" \
  "protobuf<4" \
  "pyyaml" \
  "scipy" \
  "tqdm" \
  "pytest"

echo "==> re-pinning torch (the lightning install can pull a newer one)"
python -m pip install --no-cache-dir --force-reinstall --no-deps \
  "torch==${TORCH_VERSION}+cpu" -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -q "setuptools==59.5.0"

echo "==> verifying"
python - <<'PY'
import warnings
warnings.filterwarnings("ignore")
import torch, signatory, pytorch_lightning as pl, numpy, pandas
print(f"torch       {torch.__version__}")
print(f"signatory   {signatory.__version__}")
print(f"lightning   {pl.__version__}")
print(f"numpy       {numpy.__version__}")
print(f"pandas      {pandas.__version__}")
x = torch.randn(2, 16, 3)
assert signatory.logsignature(x, 5, mode="words").shape == (2, 80)
print("signatory logsignature: OK")
PY

echo
echo "done. activate with:  conda activate ${ENV_NAME}"
