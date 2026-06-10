#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual}"
SPLIT="${SPLIT:-test}"
TRAIN_DATASET="${TRAIN_DATASET:-MBZUAI/EXAMS-V}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-validation}"
ADAPTER_DIR="${ADAPTER_DIR:-outputs/qwen3vl8b-thinking-examsv-lora-600-lr5e5}"
OUTPUT="${OUTPUT:-outputs/visual_mcq_qwen3vl8b_thinking_lora_600_lr5e5_enhanced_test.json}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_FILE="${LOG_DIR}/mcq_qwen3vl_full_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}" outputs
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Starting Visual MCQ Qwen3-VL pipeline"
echo "Log: ${LOG_FILE}"
date

echo
echo "Installing Qwen3-VL dependencies"
pip install -U -r requirements-qwen3vl.txt

if [ "${TRAIN_ADAPTER:-0}" = "1" ]; then
  if [ -f "${ADAPTER_DIR}/adapter_config.json" ]; then
    echo
    echo "Adapter already exists: ${ADAPTER_DIR}"
  else
    echo
    echo "Training adapter: ${ADAPTER_DIR}"
    python scripts/train_visual_mcq_lora_qwen3vl.py \
      --dataset "${TRAIN_DATASET}" \
      --train-split "${TRAIN_SPLIT}" \
      --eval-split "${EVAL_SPLIT}" \
      --load-in-4bit \
      --gradient-checkpointing \
      --max-steps "${MAX_STEPS:-600}" \
      --learning-rate "${LEARNING_RATE:-5e-5}" \
      --eval-limit "${EVAL_LIMIT:-300}" \
      --eval-steps "${EVAL_STEPS:-100}" \
      --save-steps "${SAVE_STEPS:-100}" \
      --output-dir "${ADAPTER_DIR}"
  fi
fi

if [ ! -f "${ADAPTER_DIR}/adapter_config.json" ]; then
  echo
  echo "ERROR: Adapter not found at ${ADAPTER_DIR}"
  echo "Either download it there or rerun with TRAIN_ADAPTER=1."
  exit 1
fi

echo
echo "Running Visual MCQ prediction"
python scripts/run_visual_mcq_qwen3vl.py \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --adapter "${ADAPTER_DIR}" \
  --load-in-4bit \
  --image-variant enhanced \
  --enhance-longest-side "${ENHANCE_LONGEST_SIDE:-1600}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-16}" \
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
