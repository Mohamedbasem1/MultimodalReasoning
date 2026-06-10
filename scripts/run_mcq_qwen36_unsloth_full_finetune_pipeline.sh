#!/usr/bin/env bash
set -euo pipefail

TRAIN_DATASET="${TRAIN_DATASET:-MBZUAI/EXAMS-V}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-validation}"
EXAMSV_TEST_SPLIT="${EXAMSV_TEST_SPLIT:-test}"
BLIND_DATASET="${BLIND_DATASET:-SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual}"
BLIND_SPLIT="${BLIND_SPLIT:-test}"
ADAPTER_DIR="${ADAPTER_DIR:-outputs/qwen36-35b-a3b-mcq-examsv-lora-epoch1}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_FILE="${LOG_DIR}/mcq_qwen36_full_finetune_$(date +%Y%m%d_%H%M%S).log"

VALIDATION_OUTPUT="${VALIDATION_OUTPUT:-outputs/mcq_examsv_validation_qwen36_lora_epoch1.json}"
EXAMSV_TEST_OUTPUT="${EXAMSV_TEST_OUTPUT:-outputs/mcq_examsv_test_qwen36_lora_epoch1.json}"
BLIND_OUTPUT="${BLIND_OUTPUT:-outputs/visual_mcq_qwen36_lora_epoch1_blind_test.json}"

mkdir -p "${LOG_DIR}" outputs
exec > >(tee -a "${LOG_FILE}") 2>&1
export UNSLOTH_MOE_BACKEND="${UNSLOTH_MOE_BACKEND:-native_torch}"

echo "Starting Qwen3.6 Unsloth Visual MCQ full fine-tune pipeline"
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

if [ -f "${ADAPTER_DIR}/adapter_config.json" ]; then
  echo
  echo "Adapter already exists, skipping training: ${ADAPTER_DIR}"
else
  TRAIN_LIMIT_ARGS=()
  if [ -n "${TRAIN_LIMIT:-}" ]; then
    TRAIN_LIMIT_ARGS=(--train-limit "${TRAIN_LIMIT}")
  fi

  echo
  echo "Training on ${TRAIN_DATASET} [${TRAIN_SPLIT}]"
  echo "This uses the full train split unless TRAIN_LIMIT is set."
  python scripts/train_visual_mcq_lora_qwen36_unsloth.py \
    --dataset "${TRAIN_DATASET}" \
    --train-split "${TRAIN_SPLIT}" \
    --load-in-4bit \
    --gradient-checkpointing \
    --num-epochs "${NUM_EPOCHS:-1.0}" \
    --max-steps "${MAX_STEPS:--1}" \
    --learning-rate "${LEARNING_RATE:-3e-5}" \
    --save-steps "${SAVE_STEPS:-100}" \
    --image-variant enhanced \
    --enhance-longest-side "${TRAIN_ENHANCE_LONGEST_SIDE:-768}" \
    --output-dir "${ADAPTER_DIR}" \
    "${TRAIN_LIMIT_ARGS[@]}"
fi

echo
echo "Scoring EXAMS-V validation split"
UNSLOTH_MOE_BACKEND="${UNSLOTH_MOE_BACKEND}" python scripts/run_visual_mcq_qwen36_unsloth.py \
  --dataset "${TRAIN_DATASET}" \
  --split "${VALIDATION_SPLIT}" \
  --adapter "${ADAPTER_DIR}" \
  --load-in-4bit \
  --image-variant enhanced \
  --enhance-longest-side "${INFER_ENHANCE_LONGEST_SIDE:-768}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-8}" \
  --output "${VALIDATION_OUTPUT}"

python scripts/validate_mcq_submission.py \
  "${VALIDATION_OUTPUT}" \
  --dataset "${TRAIN_DATASET}" \
  --split "${VALIDATION_SPLIT}" \
  --id-column sample_id

echo
echo "Scoring EXAMS-V test split"
UNSLOTH_MOE_BACKEND="${UNSLOTH_MOE_BACKEND}" python scripts/run_visual_mcq_qwen36_unsloth.py \
  --dataset "${TRAIN_DATASET}" \
  --split "${EXAMSV_TEST_SPLIT}" \
  --adapter "${ADAPTER_DIR}" \
  --load-in-4bit \
  --image-variant enhanced \
  --enhance-longest-side "${INFER_ENHANCE_LONGEST_SIDE:-768}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-8}" \
  --output "${EXAMSV_TEST_OUTPUT}"

python scripts/validate_mcq_submission.py \
  "${EXAMSV_TEST_OUTPUT}" \
  --dataset "${TRAIN_DATASET}" \
  --split "${EXAMSV_TEST_SPLIT}" \
  --id-column sample_id

echo
echo "Running blind Visual MCQ test"
UNSLOTH_MOE_BACKEND="${UNSLOTH_MOE_BACKEND}" python scripts/run_visual_mcq_qwen36_unsloth.py \
  --dataset "${BLIND_DATASET}" \
  --split "${BLIND_SPLIT}" \
  --adapter "${ADAPTER_DIR}" \
  --load-in-4bit \
  --image-variant enhanced \
  --enhance-longest-side "${INFER_ENHANCE_LONGEST_SIDE:-768}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-8}" \
  --output "${BLIND_OUTPUT}"

python scripts/validate_mcq_submission.py \
  "${BLIND_OUTPUT}" \
  --dataset "${BLIND_DATASET}" \
  --split "${BLIND_SPLIT}"

echo
echo "Done"
echo "Validation predictions: ${VALIDATION_OUTPUT}"
echo "EXAMS-V test predictions: ${EXAMSV_TEST_OUTPUT}"
echo "Blind submission: ${BLIND_OUTPUT}"
echo "Adapter: ${ADAPTER_DIR}"
echo "Log: ${LOG_FILE}"
date
