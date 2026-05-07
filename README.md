# ImageCLEF 2026 Visual MCQ Starter

This repo starts a Visual MCQ baseline with `Qwen/Qwen2.5-VL-7B-Instruct`.

The output format is the competition JSON:

```json
[
  {"question_id": "example_id", "answer_key": "A"}
]
```

## Setup

Use a GPU machine if possible. The official task environment mentions an A40 40GB GPU; Qwen2.5-VL-7B can also run on smaller GPUs with careful settings, but CPU inference is not practical.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you see `KeyError: 'qwen2_5_vl'`, update Transformers from source:

```powershell
pip install -U git+https://github.com/huggingface/transformers accelerate
```

## Lightning AI Studio

Recommended Studio GPU: use **L40S, A100, H100, or another GPU with at least 24GB VRAM**. If an A40 40GB option is available, that is closest to the official task environment. An L4 24GB may work for inference, but reduce `--max-pixels` if you hit out-of-memory errors.

In a Lightning AI Studio terminal:

```bash
git clone <your-repo-url> imageclef-mr2026
cd imageclef-mr2026
bash scripts/setup_lightning.sh
```

If you uploaded this folder manually instead of cloning, just `cd` into the folder and run:

```bash
bash scripts/setup_lightning.sh
```

Confirm the Studio sees the GPU:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Run a tiny smoke test first:

```bash
python scripts/run_visual_mcq_qwen25.py --limit 3 --output outputs/lightning_smoke.json
python scripts/validate_mcq_submission.py outputs/lightning_smoke.json --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual --split test --allow-subset
```

## Fine-Tuning On Lightning AI

The competition Visual MCQ test split has no labels, so do **not** fine-tune on it. Fine-tune on the labeled EXAMS-V MCQ train split, validate on EXAMS-V validation, then run prediction on the competition Visual MCQ test split.

Start with a very small training smoke test:

```bash
python scripts/train_visual_mcq_lora.py \
  --load-in-4bit \
  --gradient-checkpointing \
  --train-limit 200 \
  --eval-limit 50 \
  --max-steps 20 \
  --eval-steps 10 \
  --save-steps 10 \
  --output-dir outputs/qwen25vl7b-examsv-lora-smoke
```

Then run a real LoRA/QLoRA fine-tune:

```bash
python scripts/train_visual_mcq_lora.py \
  --load-in-4bit \
  --gradient-checkpointing \
  --num-epochs 1 \
  --eval-limit 300 \
  --eval-steps 250 \
  --save-steps 250 \
  --output-dir outputs/qwen25vl7b-examsv-lora
```

If you want to focus only on visually grounded training rows, add:

```bash
--filter-type image_text
```

After fine-tuning, run the competition Visual MCQ prediction with the adapter:

```bash
python scripts/run_visual_mcq_qwen25.py \
  --adapter outputs/qwen25vl7b-examsv-lora \
  --output outputs/visual_mcq_qwen25vl7b_lora.json

python scripts/validate_mcq_submission.py \
  outputs/visual_mcq_qwen25vl7b_lora.json \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test
```

Submit `outputs/visual_mcq_qwen25vl7b_lora.json`.

## Visual OpenQA With Qwen3-VL

Visual OpenQA is a generative task: the model must produce a free-form answer instead of an `A/B/C/D/E` choice. The submission JSON uses:

```json
[
  {"question_id": "example_id", "answer": "short answer text"}
]
```

Install the Qwen3-VL dependencies:

```bash
pip install -r requirements-qwen3vl.txt
```

Run a tiny prediction smoke test on the competition Visual OpenQA test split:

```bash
python scripts/run_visual_openqa_qwen3vl.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual \
  --split test \
  --limit 5 \
  --image-variant enhanced \
  --output outputs/visual_openqa_qwen3vl_smoke.json

python scripts/validate_openqa_submission.py \
  outputs/visual_openqa_qwen3vl_smoke.json \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual \
  --split test \
  --allow-subset
```

Start with a small OpenQA QLoRA smoke run:

```bash
python scripts/train_visual_openqa_lora_qwen3vl.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual \
  --train-split train \
  --validation-from-train 100 \
  --internal-test-split dev \
  --load-in-4bit \
  --gradient-checkpointing \
  --train-limit 200 \
  --eval-limit 100 \
  --internal-test-limit 50 \
  --max-steps 20 \
  --eval-steps 10 \
  --save-steps 10 \
  --image-variant enhanced \
  --output-dir outputs/qwen3vl8b-thinking-openqa-lora-smoke
```

The OpenQA prompt explicitly asks the model to learn the reference-answer structure: concise wording, same language when possible, correct units, exact numbers, and no extra sentence framing.

Run a longer OpenQA LoRA:

```bash
python scripts/train_visual_openqa_lora_qwen3vl.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual \
  --train-split train \
  --validation-from-train 100 \
  --internal-test-split dev \
  --load-in-4bit \
  --gradient-checkpointing \
  --max-steps 600 \
  --learning-rate 3e-5 \
  --eval-limit 100 \
  --eval-steps 100 \
  --save-steps 100 \
  --image-variant enhanced \
  --output-dir outputs/qwen3vl8b-thinking-openqa-lora-600-lr3e5
```

With `--validation-from-train 100`, the trainer shuffles the filtered train split using `--seed`, holds out 100 labeled train rows for validation, and trains on the rest. `--internal-test-split dev` is scored only after training, so it stays separate from checkpoint selection. Use the blinded `test` split only for the final prediction file.

Predict the competition Visual OpenQA test split:

```bash
python scripts/run_visual_openqa_qwen3vl.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual \
  --split test \
  --adapter outputs/qwen3vl8b-thinking-openqa-lora-600-lr3e5 \
  --image-variant enhanced \
  --output outputs/visual_openqa_qwen3vl_lora.json

python scripts/validate_openqa_submission.py \
  outputs/visual_openqa_qwen3vl_lora.json \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual \
  --split test
```

## Qwen3-VL-8B-Thinking Fine-Tuning

`Qwen/Qwen3-VL-8B-Thinking` is a stronger Normal-category experiment than the Tiny Qwen2.5-VL-7B baseline. It needs the newer Qwen3-VL Transformers code, so use a fresh Lightning Studio or reinstall Transformers before running it.

Install the Qwen3-VL dependencies:

```bash
pip install -r requirements-qwen3vl.txt
```

Run a small QLoRA smoke test first:

```bash
python scripts/train_visual_mcq_lora_qwen3vl.py \
  --dataset MBZUAI/EXAMS-V \
  --train-split train \
  --eval-split validation \
  --load-in-4bit \
  --gradient-checkpointing \
  --prompt-file prompts/visual_mcq_final_only.txt \
  --train-limit 200 \
  --eval-limit 50 \
  --max-steps 20 \
  --eval-steps 10 \
  --save-steps 10 \
  --output-dir outputs/qwen3vl8b-thinking-examsv-lora-smoke
```

If the smoke run works, start with a 300-step run and compare against the current Qwen2.5-VL best score:

```bash
python scripts/train_visual_mcq_lora_qwen3vl.py \
  --dataset MBZUAI/EXAMS-V \
  --train-split train \
  --eval-split validation \
  --load-in-4bit \
  --gradient-checkpointing \
  --prompt-file prompts/visual_mcq_final_only.txt \
  --max-steps 300 \
  --learning-rate 1e-4 \
  --eval-limit 300 \
  --eval-steps 100 \
  --save-steps 100 \
  --output-dir outputs/qwen3vl8b-thinking-examsv-lora-300
```

Evaluate it on the labeled EXAMS-V test split using the same enhanced-image setting that gave the best Qwen2.5-VL result:

```bash
python scripts/run_visual_mcq_qwen3vl.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --adapter outputs/qwen3vl8b-thinking-examsv-lora-300 \
  --image-variant enhanced \
  --output outputs/examsv_test_qwen3vl8b_thinking_lora_300_enhanced_full.json
```

Only run the competition test if it beats the current best EXAMS-V test score:

```bash
python scripts/run_visual_mcq_qwen3vl.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --adapter outputs/qwen3vl8b-thinking-examsv-lora-300 \
  --image-variant enhanced \
  --output outputs/imageclef_visual_mcq_qwen3vl8b_thinking_lora_enhanced.json

python scripts/validate_mcq_submission.py \
  outputs/imageclef_visual_mcq_qwen3vl8b_thinking_lora_enhanced.json \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test
```

### Qwen3 LoRA Checkpoint Soup

After a longer Qwen3 run, average the saved LoRA checkpoints into one smoother adapter. This costs no extra training and can sometimes beat the final checkpoint:

```bash
python scripts/soup_lora_adapters.py \
  --adapters \
    outputs/qwen3vl8b-thinking-examsv-lora-600-lr5e5/checkpoint-300 \
    outputs/qwen3vl8b-thinking-examsv-lora-600-lr5e5/checkpoint-400 \
    outputs/qwen3vl8b-thinking-examsv-lora-600-lr5e5/checkpoint-500 \
    outputs/qwen3vl8b-thinking-examsv-lora-600-lr5e5/checkpoint-600 \
  --output-dir outputs/qwen3vl8b-thinking-examsv-lora-600-lr5e5-soup-300-600
```

Evaluate the soup on EXAMS-V test:

```bash
python scripts/run_visual_mcq_qwen3vl.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --adapter outputs/qwen3vl8b-thinking-examsv-lora-600-lr5e5-soup-300-600 \
  --image-variant enhanced \
  --output outputs/examsv_test_qwen3vl8b_thinking_lora_600_lr5e5_soup_300_600_enhanced_full.json
```

If it beats the current best score, run the ImageCLEF test with the soup adapter:

```bash
python scripts/run_visual_mcq_qwen3vl.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --adapter outputs/qwen3vl8b-thinking-examsv-lora-600-lr5e5-soup-300-600 \
  --image-variant enhanced \
  --output outputs/imageclef_visual_mcq_qwen3vl8b_thinking_lora_600_lr5e5_soup_300_600_enhanced.json
```

## Aya Vision 8B Fine-Tuning

`CohereLabs/aya-vision-8b` is a gated multilingual VLM. Before using it, accept the model terms on Hugging Face, log in from Lightning with a token from the same account, and confirm that the competition allows `CC-BY-NC-4.0` models.

Use a separate Lightning Studio if possible because Aya Vision requires a specific Transformers branch:

```bash
pip install -r requirements-aya-vision.txt
hf auth login
hf download CohereLabs/aya-vision-8b config.json --repo-type model --local-dir /tmp/aya-test
```

Run a small QLoRA smoke test:

```bash
python scripts/train_visual_mcq_lora_aya.py \
  --dataset MBZUAI/EXAMS-V \
  --train-split train \
  --eval-split validation \
  --load-in-4bit \
  --gradient-checkpointing \
  --train-limit 200 \
  --eval-limit 50 \
  --max-steps 20 \
  --eval-steps 0 \
  --save-steps 10 \
  --output-dir outputs/aya-vision-8b-examsv-lora-smoke
```

If the smoke run works, try a 300-step adapter:

```bash
python scripts/train_visual_mcq_lora_aya.py \
  --dataset MBZUAI/EXAMS-V \
  --train-split train \
  --eval-split validation \
  --load-in-4bit \
  --gradient-checkpointing \
  --max-steps 300 \
  --learning-rate 1e-4 \
  --eval-limit 300 \
  --eval-steps 0 \
  --save-steps 100 \
  --output-dir outputs/aya-vision-8b-examsv-lora-300
```

Evaluate it on EXAMS-V test:

```bash
python scripts/run_visual_mcq_aya.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --adapter outputs/aya-vision-8b-examsv-lora-300 \
  --image-variant enhanced \
  --output outputs/examsv_test_aya_vision_8b_lora_300_enhanced_full.json
```

Only run the ImageCLEF test if it beats the current Qwen3 score:

```bash
python scripts/run_visual_mcq_aya.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --adapter outputs/aya-vision-8b-examsv-lora-300 \
  --image-variant enhanced \
  --output outputs/imageclef_visual_mcq_aya_vision_8b_lora_300_enhanced.json
```

## MiniCPM-V 4.5 Inference

`openbmb/MiniCPM-V-4_5` is an Apache-2.0 8.7B VLM built on Qwen3-8B with strong OCR, document parsing, and 30+ language support. Start with inference before spending GPU on fine-tuning.

Install:

```bash
pip install -r requirements-minicpm-v.txt
```

Run a 500-example EXAMS-V test:

```bash
python scripts/run_visual_mcq_minicpm.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --limit 500 \
  --image-variant enhanced \
  --max-new-tokens 32 \
  --output outputs/examsv_test_minicpm_v45_enhanced_500.json
```

If MiniCPM loading fails with `all_tied_weights_keys`, pull the latest repo version. The runner disables Transformers low-memory/meta loading by default to avoid that compatibility path.

If it is close to the current best, run the full EXAMS-V test:

```bash
python scripts/run_visual_mcq_minicpm.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --image-variant enhanced \
  --max-new-tokens 32 \
  --output outputs/examsv_test_minicpm_v45_enhanced_full.json
```

Optional slower deep-thinking mode:

```bash
python scripts/run_visual_mcq_minicpm.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --limit 500 \
  --image-variant enhanced \
  --enable-thinking \
  --max-new-tokens 64 \
  --output outputs/examsv_test_minicpm_v45_thinking_enhanced_500.json
```

### MiniCPM-V 4.5 Fine-Tuning

MiniCPM-V scored poorly in zero-shot mode on EXAMS-V, so only run a short QLoRA smoke test before spending serious GPU time:

```bash
python scripts/train_visual_mcq_lora_minicpm.py \
  --dataset MBZUAI/EXAMS-V \
  --train-split train \
  --eval-split validation \
  --load-in-4bit \
  --gradient-checkpointing \
  --train-limit 200 \
  --eval-limit 50 \
  --max-steps 20 \
  --eval-steps 10 \
  --save-steps 10 \
  --output-dir outputs/minicpm-v45-examsv-lora-smoke
```

If the smoke run finishes, evaluate the adapter quickly:

```bash
python scripts/run_visual_mcq_minicpm.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --adapter outputs/minicpm-v45-examsv-lora-smoke \
  --limit 500 \
  --image-variant enhanced \
  --max-new-tokens 32 \
  --output outputs/examsv_test_minicpm_v45_lora_smoke_500.json
```

## Phi-4-Reasoning-Vision-15B Inference

`microsoft/Phi-4-reasoning-vision-15B` is a 15B MIT-licensed VLM focused on visual reasoning, charts, OCR, documents, and STEM-style questions. Run inference first before considering LoRA.

Install in a fresh Lightning Studio if possible because the model card requires newer Torch/Transformers:

```bash
pip install -r requirements-phi4vision.txt
```

Run a 500-example EXAMS-V test:

```bash
python scripts/run_visual_mcq_phi4vision.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --limit 500 \
  --image-variant enhanced \
  --reasoning-mode nothink \
  --max-new-tokens 48 \
  --output outputs/examsv_test_phi4_reasoning_vision_15b_nothink_enhanced_500.json
```

If it is close to the current Qwen3 best, try automatic reasoning mode:

```bash
python scripts/run_visual_mcq_phi4vision.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --limit 500 \
  --image-variant enhanced \
  --reasoning-mode auto \
  --max-new-tokens 128 \
  --output outputs/examsv_test_phi4_reasoning_vision_15b_auto_enhanced_500.json
```

If either 500-example run beats the current Qwen3 score trend, run the full EXAMS-V test by removing `--limit`.

### Phi-4 Fine-Tuning

Phi-4's custom loader currently conflicts with bitsandbytes 4-bit casting, so train this on a larger GPU without `--load-in-4bit`. Start with a 20-step smoke run:

```bash
python scripts/train_visual_mcq_lora_phi4vision.py \
  --dataset MBZUAI/EXAMS-V \
  --train-split train \
  --eval-split validation \
  --gradient-checkpointing \
  --train-limit 200 \
  --eval-limit 50 \
  --max-steps 20 \
  --eval-steps 10 \
  --save-steps 10 \
  --image-variant enhanced \
  --reasoning-mode nothink \
  --output-dir outputs/phi4-reasoning-vision-15b-examsv-lora-smoke
```

Evaluate the smoke adapter on 500 labeled test rows:

```bash
python scripts/run_visual_mcq_phi4vision.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --adapter outputs/phi4-reasoning-vision-15b-examsv-lora-smoke \
  --limit 500 \
  --image-variant enhanced \
  --reasoning-mode nothink \
  --max-new-tokens 48 \
  --output outputs/examsv_test_phi4_reasoning_vision_15b_lora_smoke_500.json
```

If the smoke run improves over the zero-shot 28.2% sample, run a longer LoRA:

```bash
python scripts/train_visual_mcq_lora_phi4vision.py \
  --dataset MBZUAI/EXAMS-V \
  --train-split train \
  --eval-split validation \
  --gradient-checkpointing \
  --max-steps 300 \
  --learning-rate 5e-5 \
  --eval-limit 300 \
  --eval-steps 100 \
  --save-steps 100 \
  --image-variant enhanced \
  --reasoning-mode nothink \
  --output-dir outputs/phi4-reasoning-vision-15b-examsv-lora-300-lr5e5
```

Then evaluate the full labeled test:

```bash
python scripts/run_visual_mcq_phi4vision.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --adapter outputs/phi4-reasoning-vision-15b-examsv-lora-300-lr5e5 \
  --image-variant enhanced \
  --reasoning-mode nothink \
  --max-new-tokens 48 \
  --output outputs/examsv_test_phi4_reasoning_vision_15b_lora_300_lr5e5_enhanced_full.json
```

## Second-Stage Weak-Case Fine-Tuning

After error analysis, the weakest groups were Arabic/Urdu, `image_text`, graphs, tables, and lower grades. Continue training from the current best adapter instead of starting from scratch:

```bash
python scripts/train_visual_mcq_lora.py \
  --dataset MBZUAI/EXAMS-V \
  --train-split train \
  --eval-split validation \
  --init-adapter outputs/qwen25vl7b-examsv-lora-4k \
  --load-in-4bit \
  --gradient-checkpointing \
  --prompt-file prompts/visual_mcq_weakcase_prompt.txt \
  --include-language Arabic Urdu \
  --include-grade 9 10 11 \
  --include-binary-columns graph table figure \
  --weak-filter-mode or \
  --max-steps 300 \
  --learning-rate 5e-5 \
  --eval-limit 300 \
  --eval-steps 100 \
  --save-steps 100 \
  --output-dir outputs/qwen25vl7b-examsv-lora-weakstage
```

Optional narrower image-text-only variant:

```bash
python scripts/train_visual_mcq_lora.py \
  --dataset MBZUAI/EXAMS-V \
  --train-split train \
  --eval-split validation \
  --init-adapter outputs/qwen25vl7b-examsv-lora-4k \
  --load-in-4bit \
  --gradient-checkpointing \
  --prompt-file prompts/visual_mcq_weakcase_prompt.txt \
  --filter-type image_text \
  --max-steps 200 \
  --learning-rate 5e-5 \
  --eval-limit 300 \
  --eval-steps 100 \
  --save-steps 100 \
  --output-dir outputs/qwen25vl7b-examsv-lora-imagetext-stage
```

Evaluate the weak-stage adapter with the current best enhanced direct pipeline:

```bash
python scripts/run_visual_mcq_voting.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-weakstage \
  --num-prompts 1 \
  --image-variants enhanced \
  --output outputs/examsv_test_qwen25vl7b_weakstage_enhanced_full.json
```

If it beats `52.34%`, generate competition output:

```bash
python scripts/run_visual_mcq_voting.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-weakstage \
  --num-prompts 1 \
  --image-variants enhanced \
  --output outputs/imageclef_visual_mcq_qwen25vl7b_lora_weakstage_enhanced.json
```

## Voting Inference Enhancement

The strongest confirmed baseline so far is Qwen2.5-VL-7B with the EXAMS-V LoRA adapter. To improve it without more training, use multi-prompt voting. This runs three prompt variants per image:

- direct full-image solving
- OCR/detail-focused solving
- option verification/elimination

The voting script can also run an enhanced image variant and inject external OCR text into the prompt. The safest optional OCR dependency on Lightning is EasyOCR:

```bash
pip install -r requirements-ocr.txt
```

PaddleOCR is stronger in many document-style OCR settings and supports very broad multilingual recognition, but it has a heavier install stack. Use it only if the Lightning image supports it cleanly.

DeepSeek-OCR is also supported as a stronger VLM-style OCR extractor. It is a 3B MIT-licensed Hugging Face model, so it is heavier than EasyOCR but can produce richer document Markdown:

```bash
pip install -r requirements-deepseek-ocr.txt
```

If your environment has FlashAttention installed, you can use `--deepseek-ocr-attn-implementation flash_attention_2`; otherwise keep the default `sdpa`.

Try it first on a labeled EXAMS-V subset:

```bash
python scripts/run_visual_mcq_voting.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-4k \
  --limit 500 \
  --output outputs/examsv_test_qwen25vl7b_voting_500.json

python scripts/validate_mcq_submission.py \
  outputs/examsv_test_qwen25vl7b_voting_500.json \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --allow-subset
```

Stronger OCR + image enhancement version:

```bash
python scripts/run_visual_mcq_voting.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-4k \
  --limit 500 \
  --image-variants original enhanced \
  --ocr-engine easyocr \
  --ocr-langs en \
  --ocr-on-enhanced \
  --output outputs/examsv_test_qwen25vl7b_voting_ocr_500.json
```

For multilingual OCR experiments, add language codes supported by EasyOCR. Start small because every added language can download extra OCR weights:

```bash
--ocr-langs en ar de es fr it pl hr hu ru
```

If voting beats the single-prompt result, run it on the competition test split:

```bash
python scripts/run_visual_mcq_voting.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-4k \
  --image-variants original enhanced \
  --output outputs/imageclef_visual_mcq_qwen25vl7b_lora_voting.json

python scripts/validate_mcq_submission.py \
  outputs/imageclef_visual_mcq_qwen25vl7b_lora_voting.json \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test
```

For a faster version, use only two prompts:

```bash
python scripts/run_visual_mcq_voting.py \
  --num-prompts 2 \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-4k \
  --output outputs/imageclef_visual_mcq_qwen25vl7b_lora_voting2.json
```

OCR-enhanced competition run:

```bash
python scripts/run_visual_mcq_voting.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-4k \
  --image-variants original enhanced \
  --ocr-engine easyocr \
  --ocr-langs en \
  --ocr-on-enhanced \
  --output outputs/imageclef_visual_mcq_qwen25vl7b_lora_voting_ocr.json
```

DeepSeek-OCR enhanced run:

```bash
python scripts/run_visual_mcq_voting.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-4k \
  --image-variants original enhanced \
  --ocr-engine deepseek \
  --ocr-on-enhanced \
  --ocr-max-chars 2200 \
  --output outputs/imageclef_visual_mcq_qwen25vl7b_lora_voting_deepseek_ocr.json
```

If memory gets tight, add `--load-in-4bit` for Qwen or `--ocr-cpu` for the OCR model. DeepSeek-OCR on CPU is much slower, so prefer GPU when memory allows.

## Error Analysis

Use this after any labeled EXAMS-V run to see where the model fails. It reports answer bias, confusion matrix, metadata group accuracy, image-size buckets, and a sample of errors.

```bash
python scripts/analyze_mcq_errors.py \
  outputs/examsv_test_qwen25vl7b_enhanced_full.json \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --raw-output outputs/examsv_test_qwen25vl7b_enhanced_full.raw.jsonl \
  --output-dir outputs/error_analysis_qwen25vl7b_enhanced
```

Open:

```bash
outputs/error_analysis_qwen25vl7b_enhanced/summary.json
outputs/error_analysis_qwen25vl7b_enhanced/errors.jsonl
```

## Conditional Self-Consistency

This is a smarter version of voting for the current best model. It runs direct/OCR/verify prompts on the enhanced image. If all prompts agree, it keeps the answer. If they disagree, it runs one verifier prompt and uses that answer.

```bash
python scripts/run_visual_mcq_consistency.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-4k \
  --limit 500 \
  --image-variant enhanced \
  --agreement-policy unanimous_then_verifier \
  --output outputs/examsv_test_qwen25vl7b_consistency_500.json
```

If it beats enhanced direct on 500, run the full EXAMS-V test:

```bash
python scripts/run_visual_mcq_consistency.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-4k \
  --image-variant enhanced \
  --agreement-policy unanimous_then_verifier \
  --output outputs/examsv_test_qwen25vl7b_consistency_full.json
```

Competition consistency submission:

```bash
python scripts/run_visual_mcq_consistency.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-4k \
  --image-variant enhanced \
  --agreement-policy unanimous_then_verifier \
  --output outputs/imageclef_visual_mcq_qwen25vl7b_lora_consistency.json
```

## Metadata Routing Ensemble

The error analysis showed weaker performance on `image_text`, Arabic/Urdu, graphs, and tables. This router keeps the strongest direct enhanced predictions by default, but switches selected weak-case metadata rows to another prediction file, such as the multilingual prompt output.

Run it on EXAMS-V first:

```bash
python scripts/route_mcq_predictions.py \
  --primary outputs/examsv_test_qwen25vl7b_enhanced_full.json \
  --secondary outputs/examsv_test_qwen25vl7b_multilingual_full.json \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --output outputs/examsv_test_qwen25vl7b_routed_multilingual.json \
  --route-languages Arabic Urdu \
  --route-types image_text \
  --route-binary-columns graph table
```

If this beats the direct enhanced full score, generate the routed competition file after producing both competition prediction files:

```bash
python scripts/route_mcq_predictions.py \
  --primary outputs/imageclef_visual_mcq_qwen25vl7b_lora_enhanced.json \
  --secondary outputs/imageclef_visual_mcq_qwen25vl7b_lora_multilingual_enhanced.json \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --output outputs/imageclef_visual_mcq_qwen25vl7b_lora_routed_multilingual.json \
  --route-languages Arabic Urdu \
  --route-types image_text \
  --route-binary-columns graph table
```

## Candidate Scoring Enhancement

For MCQ, a stronger alternative to generation is candidate scoring: compute the log probability of each answer letter (`A`-`E`) and choose the highest. This avoids parsing failures and can be more stable than asking the model to generate one token.

Test on the same first 500 EXAMS-V test examples:

```bash
python scripts/run_visual_mcq_candidate_scoring.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-4k \
  --limit 500 \
  --num-prompts 1 \
  --image-variants original \
  --output outputs/examsv_test_qwen25vl7b_scored_500.json
```

If this beats the direct generation baseline, run the full EXAMS-V test:

```bash
python scripts/run_visual_mcq_candidate_scoring.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-4k \
  --num-prompts 1 \
  --image-variants original \
  --output outputs/examsv_test_qwen25vl7b_scored_full.json
```

Candidate scoring can also consume precomputed DeepSeek-OCR JSONL:

```bash
python scripts/run_visual_mcq_candidate_scoring.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-4k \
  --limit 100 \
  --num-prompts 1 \
  --image-variants original \
  --ocr-json outputs/examsv_test_deepseek_ocr_100.jsonl \
  --output outputs/examsv_test_qwen25vl7b_scored_deepseek_ocr_100.json
```

Competition candidate-scored submission:

```bash
python scripts/run_visual_mcq_candidate_scoring.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-4k \
  --num-prompts 1 \
  --image-variants original \
  --output outputs/imageclef_visual_mcq_qwen25vl7b_lora_scored.json
```

## Vero-Qwen25-7B Model Experiment

`zlab-princeton/Vero-Qwen25-7B` is a Qwen2.5-VL-7B based visual reasoning model trained with RL across charts, OCR, STEM, spatial reasoning, grounding, and counting. Because it keeps the Qwen2.5-VL architecture, it can be tested with the same inference scripts.

Run a quick 500-example test without the EXAMS-V LoRA adapter:

```bash
python scripts/run_visual_mcq_voting.py \
  --model zlab-princeton/Vero-Qwen25-7B \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --limit 500 \
  --num-prompts 1 \
  --prompt-files prompts/visual_mcq_final_only.txt \
  --image-variants enhanced \
  --max-new-tokens 64 \
  --output outputs/examsv_test_vero_qwen25_7b_enhanced_500.json
```

If that beats the enhanced Qwen2.5-VL LoRA result on the same subset, run full EXAMS-V test:

```bash
python scripts/run_visual_mcq_voting.py \
  --model zlab-princeton/Vero-Qwen25-7B \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --num-prompts 1 \
  --prompt-files prompts/visual_mcq_final_only.txt \
  --image-variants enhanced \
  --max-new-tokens 64 \
  --output outputs/examsv_test_vero_qwen25_7b_enhanced_full.json
```

Optional risky test: apply the EXAMS-V LoRA adapter on top of Vero. This may help or hurt because the adapter was trained on the base Qwen weights:

```bash
python scripts/run_visual_mcq_voting.py \
  --model zlab-princeton/Vero-Qwen25-7B \
  --adapter outputs/qwen25vl7b-examsv-lora-4k \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --limit 500 \
  --num-prompts 1 \
  --prompt-files prompts/visual_mcq_final_only.txt \
  --image-variants enhanced \
  --max-new-tokens 64 \
  --output outputs/examsv_test_vero_qwen25_7b_lora_enhanced_500.json
```

Competition Vero submission:

```bash
python scripts/run_visual_mcq_voting.py \
  --model zlab-princeton/Vero-Qwen25-7B \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --num-prompts 1 \
  --prompt-files prompts/visual_mcq_final_only.txt \
  --image-variants enhanced \
  --max-new-tokens 64 \
  --output outputs/imageclef_visual_mcq_vero_qwen25_7b_enhanced.json
```

## InternVL3-8B Model Experiment

`OpenGVLab/InternVL3-8B-hf` is the Hugging Face Transformers implementation of InternVL3-8B. It uses a Qwen2.5-7B language component with InternViT vision encoder and is Apache-2.0 licensed.

Test on 500 EXAMS-V examples first:

```bash
python scripts/run_visual_mcq_internvl3.py \
  --model OpenGVLab/InternVL3-8B-hf \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --limit 500 \
  --image-variant enhanced \
  --output outputs/examsv_test_internvl3_8b_enhanced_500.json
```

If memory is tight, add:

```bash
--load-in-4bit
```

If it beats the Qwen2.5-VL-7B LoRA enhanced result, run full EXAMS-V test:

```bash
python scripts/run_visual_mcq_internvl3.py \
  --model OpenGVLab/InternVL3-8B-hf \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --image-variant enhanced \
  --output outputs/examsv_test_internvl3_8b_enhanced_full.json
```

Competition InternVL3 submission:

```bash
python scripts/run_visual_mcq_internvl3.py \
  --model OpenGVLab/InternVL3-8B-hf \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --image-variant enhanced \
  --output outputs/imageclef_visual_mcq_internvl3_8b_enhanced.json
```

### Two-Account DeepSeek-OCR Workflow

Use this when DeepSeek-OCR needs a different Transformers version than Qwen2.5-VL.

In the **DeepSeek-OCR-only Lightning account**:

```bash
git clone -b main https://github.com/Mohamedbasem1/MultimodalReasoning.git imageclef-mr2026
cd imageclef-mr2026
pip install -r requirements-deepseek-ocr.txt

python scripts/extract_deepseek_ocr.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --limit 100 \
  --image-variant enhanced \
  --output outputs/examsv_test_deepseek_ocr_100.jsonl
```

Download or copy `outputs/examsv_test_deepseek_ocr_100.jsonl` into the **Qwen Lightning account**, then run:

```bash
python scripts/run_visual_mcq_voting.py \
  --dataset MBZUAI/EXAMS-V \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-4k \
  --limit 100 \
  --num-prompts 1 \
  --image-variants original \
  --ocr-json outputs/examsv_test_deepseek_ocr_100.jsonl \
  --output outputs/examsv_test_qwen25vl7b_deepseek_ocr_100.json
```

For the competition test set, extract OCR in the OCR account:

```bash
python scripts/extract_deepseek_ocr.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --image-variant enhanced \
  --output outputs/imageclef_visual_mcq_deepseek_ocr.jsonl
```

Then use that OCR JSONL in the Qwen account:

```bash
python scripts/run_visual_mcq_voting.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --adapter outputs/qwen25vl7b-examsv-lora-4k \
  --num-prompts 1 \
  --image-variants original \
  --ocr-json outputs/imageclef_visual_mcq_deepseek_ocr.jsonl \
  --output outputs/imageclef_visual_mcq_qwen25vl7b_lora_deepseek_ocr.json
```

## Zero-Shot Prediction

If you only want zero-shot prediction without fine-tuning:

```bash
python scripts/run_visual_mcq_qwen25.py --output outputs/visual_mcq_qwen25vl7b.json
python scripts/validate_mcq_submission.py outputs/visual_mcq_qwen25vl7b.json --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual --split test
```

If memory is tight:

```bash
python scripts/run_visual_mcq_qwen25.py --max-pixels 1003520 --output outputs/visual_mcq_qwen25vl7b_lowres.json
```

## Smoke Test

Run five Visual MCQ test rows:

```powershell
python scripts/run_visual_mcq_qwen25.py --limit 5 --output outputs/visual_mcq_smoke.json
python scripts/validate_mcq_submission.py outputs/visual_mcq_smoke.json --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual --split test --allow-subset
```

## Full Visual MCQ Run

```powershell
python scripts/run_visual_mcq_qwen25.py --output outputs/visual_mcq_qwen25vl7b.json
python scripts/validate_mcq_submission.py outputs/visual_mcq_qwen25vl7b.json --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual --split test
```

Submit `outputs/visual_mcq_qwen25vl7b.json` to the Visual MCQ leaderboard.

## Local Development With Labels

The official Visual MCQ test split has no labels. To estimate accuracy locally, use EXAMS-V validation:

```powershell
python scripts/run_visual_mcq_qwen25.py --dataset MBZUAI/EXAMS-V --split validation --filter-type image_text --limit 100 --output outputs/examsv_val_100.json
python scripts/validate_mcq_submission.py outputs/examsv_val_100.json --dataset MBZUAI/EXAMS-V --split validation --filter-type image_text --allow-subset
```

## Useful Options

- `--prompt-file prompts/visual_mcq_prompt.txt` changes the prompt without editing code.
- `--max-pixels 1003520` controls visual resolution. Increase it if OCR misses small text; decrease it if memory is tight.
- `--attn-implementation flash_attention_2` can speed up Linux GPU runs if FlashAttention is installed.
- `--load-in-4bit` can reduce memory on Linux with bitsandbytes installed.
