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
