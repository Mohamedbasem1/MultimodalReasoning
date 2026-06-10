#!/usr/bin/env bash
set -euo pipefail

DATASET="SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual"
ADAPTER_DIR="outputs/qwen25vl7b-instruct-openqa-lora-600-lr3e5"
LEGACY_OUTPUT="outputs/visual_openqa_qwen25vl_lora_600_lr3e5_test_legacy.json"
OFFICIAL_OUTPUT="outputs/visual_openqa_qwen25vl_lora_600_lr3e5_test_official.json"
LOG_DIR="outputs/logs"
LOG_FILE="${LOG_DIR}/qwen25_openqa_full_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}" outputs

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Starting Qwen2.5-VL OpenQA full pipeline"
echo "Log: ${LOG_FILE}"
date

echo
echo "Installing dependencies"
pip install -r requirements.txt

echo
echo "Training Qwen2.5-VL OpenQA LoRA"
python scripts/train_visual_openqa_lora_qwen25vl.py \
  --dataset "${DATASET}" \
  --train-split train \
  --validation-from-train 100 \
  --load-in-4bit \
  --gradient-checkpointing \
  --max-steps 600 \
  --learning-rate 3e-5 \
  --eval-limit 100 \
  --eval-steps 100 \
  --save-steps 100 \
  --image-variant enhanced \
  --output-dir "${ADAPTER_DIR}"

echo
echo "Running blinded test prediction"
python scripts/run_visual_openqa_qwen25vl.py \
  --dataset "${DATASET}" \
  --split test \
  --adapter "${ADAPTER_DIR}" \
  --image-variant enhanced \
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
