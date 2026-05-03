#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements-lightning.txt

python - <<'PY'
import torch
import transformers
import datasets

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"datasets={datasets.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu_count={torch.cuda.device_count()}")
    print(f"gpu_name={torch.cuda.get_device_name(0)}")
PY

