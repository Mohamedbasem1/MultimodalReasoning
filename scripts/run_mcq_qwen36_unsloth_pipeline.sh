#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual}"
SPLIT="${SPLIT:-test}"
ADAPTER_DIR="${ADAPTER_DIR:-outputs/qwen36-35b-a3b-openqa-unsloth-lora-300-lr3e5}"
OUTPUT="${OUTPUT:-outputs/visual_mcq_qwen36_unsloth_openqa_lora_enhanced_test.json}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_FILE="${LOG_DIR}/mcq_qwen36_unsloth_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}" outputs
exec > >(tee -a "${LOG_FILE}") 2>&1
export UNSLOTH_MOE_BACKEND="${UNSLOTH_MOE_BACKEND:-native_torch}"

echo "Starting Visual MCQ Qwen3.6 Unsloth pipeline"
echo "Log: ${LOG_FILE}"
date

echo
echo "Installing Qwen3.6 Unsloth dependencies"
pip install -r requirements-unsloth-qwen36.txt
pip install --force-reinstall --no-deps \
  "transformers==5.5.0" \
  "trl==0.24.0" \
  "unsloth==2026.5.2" \
  unsloth_zoo \
  "tokenizers==0.22.2" \
  "huggingface_hub==1.14.0"

if [ ! -f "${ADAPTER_DIR}/adapter_config.json" ]; then
  echo
  echo "ERROR: Adapter not found at ${ADAPTER_DIR}"
  echo "Download/upload the Qwen3.6 adapter first, or set ADAPTER_DIR to a valid folder."
  exit 1
fi

echo
echo "Running Visual MCQ prediction with Qwen3.6"
UNSLOTH_MOE_BACKEND="${UNSLOTH_MOE_BACKEND}" python scripts/run_visual_mcq_qwen36_unsloth.py \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --adapter "${ADAPTER_DIR}" \
  --load-in-4bit \
  --image-variant enhanced \
  --enhance-longest-side "${ENHANCE_LONGEST_SIDE:-768}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-8}" \
  --output "${OUTPUT}"

echo
echo "Validating MCQ submission"
python scripts/validate_mcq_submission.py \
  "${OUTPUT}" \
  --dataset "${DATASET}" \
  --split "${SPLIT}"

echo
echo "Done"
echo "Submission: ${OUTPUT}"
echo "Log: ${LOG_FILE}"
date
