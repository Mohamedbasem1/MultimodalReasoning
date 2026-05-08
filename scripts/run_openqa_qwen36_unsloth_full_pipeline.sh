#!/usr/bin/env bash
set -euo pipefail

DATASET="SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual"
ADAPTER_DIR="outputs/qwen36-35b-a3b-openqa-unsloth-lora-300-lr3e5"
LEGACY_OUTPUT="outputs/visual_openqa_qwen36_35b_a3b_unsloth_lora_300_lr3e5_test_legacy.json"
OFFICIAL_OUTPUT="outputs/visual_openqa_qwen36_35b_a3b_unsloth_lora_300_lr3e5_test_official.json"
LOG_DIR="outputs/logs"
LOG_FILE="${LOG_DIR}/qwen36_unsloth_openqa_full_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}" outputs

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Starting Qwen3.6-35B-A3B Unsloth OpenQA pipeline"
echo "Log: ${LOG_FILE}"
date

echo
echo "Installing Unsloth/Qwen3.6 dependencies"
pip install -U -r requirements-unsloth-qwen36.txt
pip install -U git+https://github.com/huggingface/transformers.git
pip install --force-reinstall --no-deps "trl==0.13.0"

echo
echo "Dependency versions"
python - <<'PY'
import builtins
import transformers
import trl
from trl.trainer.utils import ConstantLengthDataset
from transformers import AutoConfig

try:
    from transformers.utils import auto_docstring
except Exception:
    def auto_docstring(obj=None, *args, **kwargs):
        if callable(obj):
            return obj
        def decorator(inner):
            return inner
        return decorator
builtins.auto_docstring = auto_docstring
import unsloth

print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("unsloth import ok")
cfg = AutoConfig.from_pretrained("unsloth/Qwen3.6-35B-A3B")
print("model_type", cfg.model_type)
print("ConstantLengthDataset", ConstantLengthDataset.__name__)
PY

echo
echo "Training Qwen3.6-35B-A3B OpenQA LoRA with Unsloth"
python scripts/train_visual_openqa_lora_qwen36_unsloth.py \
  --dataset "${DATASET}" \
  --train-split train \
  --load-in-4bit \
  --gradient-checkpointing \
  --max-steps 300 \
  --learning-rate 3e-5 \
  --save-steps 25 \
  --image-variant enhanced \
  --enhance-longest-side 1000 \
  --output-dir "${ADAPTER_DIR}"

echo
echo "Running blinded test prediction"
python scripts/run_visual_openqa_qwen36_unsloth.py \
  --dataset "${DATASET}" \
  --split test \
  --adapter "${ADAPTER_DIR}" \
  --load-in-4bit \
  --image-variant enhanced \
  --enhance-longest-side 1000 \
  --output "${LEGACY_OUTPUT}"

echo
echo "Converting to official OpenQA submission format"
python scripts/convert_openqa_submission.py \
  "${LEGACY_OUTPUT}" \
  --dataset "${DATASET}" \
  --split test \
  --split-answers \
  --output "${OFFICIAL_OUTPUT}"

echo
echo "Validating official submission"
python scripts/validate_openqa_submission.py \
  "${OFFICIAL_OUTPUT}" \
  --dataset "${DATASET}" \
  --split test \
  --official-format

echo
echo "Done"
echo "Official submission: ${OFFICIAL_OUTPUT}"
echo "Log: ${LOG_FILE}"
date
