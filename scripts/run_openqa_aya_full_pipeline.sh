#!/usr/bin/env bash
set -euo pipefail

DATASET="SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual"
ADAPTER_DIR="outputs/aya-vision-8b-openqa-lora-600-lr5e5"
LEGACY_OUTPUT="outputs/visual_openqa_aya_vision_8b_lora_600_lr5e5_test_legacy.json"
OFFICIAL_OUTPUT="outputs/visual_openqa_aya_vision_8b_lora_600_lr5e5_test_official.json"
LOG_DIR="outputs/logs"
LOG_FILE="${LOG_DIR}/aya_openqa_full_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}" outputs

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Starting Aya Vision 8B OpenQA full pipeline"
echo "Log: ${LOG_FILE}"
date

echo
echo "Installing Aya Vision dependencies"
pip install -r requirements-aya-vision.txt

echo
echo "Training Aya Vision 8B OpenQA LoRA"
python scripts/train_visual_openqa_lora_aya.py \
  --dataset "${DATASET}" \
  --train-split train \
  --load-in-4bit \
  --gradient-checkpointing \
  --max-steps 600 \
  --learning-rate 5e-5 \
  --eval-steps 0 \
  --save-steps 50 \
  --image-variant enhanced \
  --output-dir "${ADAPTER_DIR}"

echo
echo "Running blinded test prediction"
python scripts/run_visual_openqa_aya.py \
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
