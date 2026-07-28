# FAU at ImageCLEF 2026 Multimodal Reasoning

This repository contains the FAU team system for the **ImageCLEF 2026 Task on Multimodal Reasoning**. We participated in both visual subtasks:

- **Visual MCQ**: choose one answer option from `A` to `E`.
- **Visual OpenQA**: generate concise free-form answers from multilingual visual exam questions.

Our final system is intentionally inference-focused. The strongest results came from controlling model outputs, scoring candidates directly, and combining complementary runs, rather than relying on raw generation or small task-specific fine-tuning.

## Official Results

### Visual MCQ

FAU ranked **3rd** on the official Visual MCQ leaderboard.

| Rank | Team | Overall | EN | BG | ZH | HR | IT | SR |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | spirosbax | 0.8406 | 0.8560 | 0.8909 | 0.7807 | 0.8772 | 0.9074 | 0.8704 |
| 2 | DS@GT | 0.7986 | 0.8520 | 0.8636 | 0.6725 | 0.8596 | 0.8704 | 0.8333 |
| **3** | **FAU** | **0.7108** | **0.7480** | **0.6909** | **0.6345** | **0.7368** | **0.8148** | **0.7593** |

### Visual OpenQA

FAU ranked **1st** on the official Visual OpenQA leaderboard by COMET.

| Rank | Team | COMET | BG | ZH | HR | EN | IT | SR | BLEU | ROUGE-L | METEOR |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | **FAU** | **0.6488** | **0.6493** | **0.7287** | **0.6588** | **0.6499** | **0.6132** | **0.5838** | **0.1391** | **0.2762** | **0.2383** |
| 2 | wangshou66 | 0.6366 | 0.6156 | 0.7012 | 0.6597 | 0.6519 | 0.5920 | 0.5929 | 0.1308 | 0.2717 | 0.2388 |
| 3 | uned-martinez | 0.5938 | 0.5721 | 0.5985 | 0.6390 | 0.5942 | 0.5767 | 0.5804 | 0.0980 | 0.2452 | 0.1842 |

## System Summary

### MCQ: Candidate Label Scoring

For multiple choice questions, free-form generation is fragile. A model may know the answer but still output a sentence, explanation, chain-of-thought, or malformed label. Our MCQ pipeline therefore scores the answer labels directly:

1. Load the visual question image.
2. Convert to RGB and resize/enhance the image.
3. Apply a strict prompt with answer prefill: `ANSWER:`.
4. Read next-token logits for labels `A`, `B`, `C`, `D`, and `E`.
5. Store raw label scores.
6. Fuse complementary score dictionaries or vote over final labels.
7. Validate the official JSON format.

The final submitted MCQ run was an ensemble over strong Qwen-family visual models. The best post-release local merge used weighted score fusion between the strongest two score-producing runs.

### OpenQA: Concise Answer Generation and Answer-Level Ensembling

OpenQA has no fixed answer set, so candidate label scoring is not possible. We instead used controlled generation:

1. Load the visual question image and metadata.
2. Enhance the image for readability.
3. Use a concise no-reasoning prompt.
4. Decode deterministically with a short maximum answer length.
5. Remove reasoning traces, answer prefixes, XML-like tags, and duplicate whitespace.
6. Combine cleaned answer lists from multiple models using COMET-weighted answer-level ensembling.

The best OpenQA submission combined strong Qwen-family answer generators with position-weighted clustering, top-2 union, and weighted union variants.

## Key Lessons

- **MCQ should be scored, not generated.** Direct A-E logit scoring avoids brittle regex extraction from long explanations.
- **OpenQA needs strict answer cleanup.** BLEU, ROUGE-L, METEOR, and COMET all suffer when outputs contain reasoning traces or extra text.
- **OCR was not automatically helpful.** OCR.space recovered text for the datasets, but raw OCR often contained broken formulas, duplicated fragments, and incorrect reading order. Injecting it into prompts degraded both MCQ and OpenQA results.
- **Small-data LoRA/QLoRA was not reliable.** Fine-tuned adapters generally underperformed the strongest direct inference runs.
- **Ensembling helped when models were complementary.** Adding weak models blindly was less useful than merging strong, partially different predictors.

## Repository Layout

```text
.
|-- paper/                         # CEUR-WS system paper source and figures
|-- prompts/                       # MCQ/OpenQA prompt templates
|-- scripts/                       # Inference, evaluation, OCR, and ensemble scripts
|-- Result/
|   |-- MCQ/                       # Local MCQ predictions, scores, OCR, and eval summaries
|   `-- OpenQA/                    # Local OpenQA predictions, OCR, and metric files
|-- requirements.txt               # Base Qwen2.5/utility environment
|-- requirements-qwen3vl.txt       # Qwen3-VL environment
|-- requirements-unsloth-qwen36.txt# Unsloth/Qwen3.6 environment
`-- requirements-huihui-mcq-ocr.txt
```

`Result/` and `paper/` contain local artifacts and are intentionally not required for basic inference.

## Datasets

The project uses the following Hugging Face datasets:

- `SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual`
- `SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual`
- `MBZUAI/EXAMS-V` for development and diagnostic fine-tuning

Some ImageCLEF datasets are gated. Authenticate before running:

```bash
hf auth login
```

## Environment Setup

Use a GPU machine. The final heavy runs were mainly executed on large-memory NVIDIA GPUs such as RTX PRO 6000 Blackwell or A100/L40S-class machines.

```bash
git clone https://github.com/Mohamedbasem1/MultimodalReasoning.git imageclef-mr2026
cd imageclef-mr2026
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Confirm CUDA:

```bash
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

On Windows PowerShell:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

### Important Environment Note

The Qwen3-VL and Unsloth/Qwen3.6 stacks can require incompatible Torch/Transformers versions. Use separate environments when possible:

- Base / Qwen2.5 / utilities: `requirements.txt`
- Qwen3-VL: `requirements-qwen3vl.txt`
- Unsloth Qwen3.6: `requirements-unsloth-qwen36.txt`

This avoids the common problem where upgrading Torch/Transformers for Qwen3-VL breaks Unsloth, or installing Unsloth downgrades packages needed by Qwen3-VL.

## Prompt Files

| Prompt | Use |
|---|---|
| `prompts/visual_mcq_final_only.txt` | Strict final-letter MCQ prompt |
| `prompts/visual_mcq_huihui_ocr_strict.txt` | MCQ OCR ablation prompt |
| `prompts/visual_mcq_prompt.txt` | Earlier MCQ baseline prompt |
| `prompts/visual_mcq_prompt_ocr.txt` | Earlier OCR prompt |
| `prompts/visual_mcq_prompt_verify.txt` | Verification/elimination prompt |
| `prompts/visual_mcq_weakcase_prompt.txt` | Weak-case fine-tuning prompt |
| `prompts/visual_openqa_prompt.txt` | Main concise OpenQA prompt |
| `prompts/visual_openqa_ocr_strict.txt` | OpenQA OCR ablation prompt |

## Output Formats

### Visual MCQ

```json
[
  {"question_id": "example_id", "answer_key": "A"}
]
```

### Visual OpenQA

```json
[
  {"question_id": "example_id", "answers": ["short answer text"]}
]
```

Some intermediate OpenQA scripts may write `answer` instead of `answers`; use `scripts/convert_openqa_submission.py` to convert to the official format.

## Quick Smoke Tests

### MCQ

```bash
python scripts/run_visual_mcq_qwen25.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --limit 5 \
  --image-variant enhanced \
  --output outputs/mcq_smoke.json

python scripts/validate_mcq_submission.py \
  outputs/mcq_smoke.json \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --allow-subset
```

### OpenQA

```bash
pip install -r requirements-qwen3vl.txt

python scripts/run_visual_openqa_qwen3vl.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual \
  --split test \
  --limit 5 \
  --image-variant enhanced \
  --output outputs/openqa_smoke.json

python scripts/validate_openqa_submission.py \
  outputs/openqa_smoke.json \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual \
  --split test \
  --allow-subset
```

## Reproducing MCQ Runs

### Qwen3.6-35B-A3B Base Candidate Scoring

```bash
pip install -r requirements-unsloth-qwen36.txt

HF_HOME=$PWD/.hf_cache \
HF_DATASETS_CACHE=$PWD/.hf_cache/datasets \
HF_HUB_DISABLE_XET=1 \
UNSLOTH_MOE_BACKEND=native_torch \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python scripts/run_visual_mcq_qwen36_unsloth.py \
  --model unsloth/Qwen3.6-35B-A3B \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --load-in-4bit \
  --selection-method logits \
  --answer-prefill "ANSWER: " \
  --prompt-file prompts/visual_mcq_final_only.txt \
  --image-variant enhanced \
  --enhance-longest-side 768 \
  --output outputs/visual_mcq_qwen36_base_blind_test_logits.json
```

### Huihui-Qwen3.6-27B Candidate Scoring

```bash
pip install -r requirements-huihui-mcq-ocr.txt

HF_HOME=$PWD/.hf_cache \
HF_DATASETS_CACHE=$PWD/.hf_cache/datasets \
HF_HUB_DISABLE_XET=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python scripts/run_visual_mcq_gemma4.py \
  --model sakamakismile/Huihui-Qwen3.6-27B-abliterated \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --load-in-4bit \
  --selection-method logits \
  --answer-prefill "ANSWER: " \
  --prompt-file prompts/visual_mcq_final_only.txt \
  --image-variant enhanced \
  --enhance-longest-side 768 \
  --output outputs/visual_mcq_huihui_qwen36_27b_abliterated_blind_test.json
```

### Merge Raw MCQ Scores

Weighted score fusion was the strongest local MCQ merge. The best local merge used Huihui with weight `2.0` and Qwen3.6 base with weight `1.0`.

```bash
python scripts/merge_mcq_raw_scores.py \
  --inputs \
    outputs/visual_mcq_huihui_qwen36_27b_abliterated_blind_test.raw.jsonl \
    outputs/visual_mcq_qwen36_base_blind_test_logits.raw.jsonl \
  --weights 2.0 1.0 \
  --output outputs/visual_mcq_huihui2_qwen36base1_scoremerge_blind_test.json

python scripts/validate_mcq_submission.py \
  outputs/visual_mcq_huihui2_qwen36base1_scoremerge_blind_test.json \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test
```

### Evaluate MCQ Result Folder

After the gold labels were released, use:

```bash
python scripts/evaluate_mcq_result_folder.py \
  --result-dir Result/MCQ \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --json-output Result/MCQ/mcq_top_level_eval_summary.json \
  --csv-output Result/MCQ/mcq_top_level_eval_summary.csv
```

The evaluator implements the leaderboard-compatible interpretation for multi-answer gold labels: a single predicted letter is counted correct if it is one of the accepted gold letters.

## Reproducing OpenQA Runs

### Qwen3-VL OpenQA

```bash
pip install -r requirements-qwen3vl.txt

HF_HOME=$PWD/.hf_cache \
HF_DATASETS_CACHE=$PWD/.hf_cache/datasets \
HF_HUB_DISABLE_XET=1 \
python scripts/run_visual_openqa_qwen3vl.py \
  --model Qwen/Qwen3-VL-32B-Thinking \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual \
  --split train \
  --torch-dtype auto \
  --max-new-tokens 192 \
  --image-variant enhanced \
  --output outputs/openqa_qwen3vl32b_thinking_train_legacy.json \
  --gold-output outputs/openqa_qwen3vl32b_thinking_train_gold.json
```

Evaluate on the labeled train split:

```bash
python scripts/evaluate_openqa_predictions.py \
  outputs/openqa_qwen3vl32b_thinking_train_legacy.json \
  --gold-file outputs/openqa_qwen3vl32b_thinking_train_gold.json \
  --output outputs/openqa_qwen3vl32b_thinking_train_metrics.json \
  --verbose
```

### Qwen2.5-VL-32B OpenQA

```bash
python scripts/run_visual_openqa_qwen25vl.py \
  --model Qwen/Qwen2.5-VL-32B-Instruct \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual \
  --split train \
  --torch-dtype auto \
  --max-new-tokens 192 \
  --image-variant enhanced \
  --output outputs/openqa_qwen25vl32b_instruct_train_legacy.json \
  --gold-output outputs/openqa_qwen25vl32b_instruct_train_gold.json

python scripts/evaluate_openqa_predictions.py \
  outputs/openqa_qwen25vl32b_instruct_train_legacy.json \
  --gold-file outputs/openqa_qwen25vl32b_instruct_train_gold.json \
  --output outputs/openqa_qwen25vl32b_instruct_train_metrics.json \
  --verbose
```

### Convert OpenQA Predictions to Official Format

```bash
python scripts/convert_openqa_submission.py \
  outputs/openqa_qwen3vl32b_thinking_test_legacy.json \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual \
  --split test \
  --split-answers \
  --output outputs/openqa_qwen3vl32b_thinking_test_official.json

python scripts/validate_openqa_submission.py \
  outputs/openqa_qwen3vl32b_thinking_test_official.json \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual \
  --split test \
  --official-format
```

### OpenQA Answer-Level Ensemble

The OpenQA ensemble combines official-format answer files after cleanup.

```bash
python scripts/ensemble_openqa.py \
  --out outputs/ensemble_position_weighted.json \
  --strategy position_weighted
```

Available strategies:

- `position_weighted`: position-wise clustering with model weights.
- `top2_union`: conservative replacement using top-model agreement.
- `weighted_union`: recall-oriented pooling of unique answers.

## OCR Experiments

OCR was useful for diagnostics but not for the final submitted systems.

Run OCR.space on a dataset:

```bash
python scripts/run_ocr_space_dataset.py \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --engine 2 \
  --api-key "$OCR_SPACE_API_KEY" \
  --output Result/MCQ/ocr_space_mcq_visual_test_engine2_merged_success.jsonl \
  --summary-output Result/MCQ/ocr_space_mcq_visual_test_engine2_merged_success.summary.json
```

Use OCR in an MCQ ablation:

```bash
python scripts/run_visual_mcq_gemma4.py \
  --model sakamakismile/Huihui-Qwen3.6-27B-abliterated \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual \
  --split test \
  --load-in-4bit \
  --selection-method logits \
  --answer-prefill "ANSWER: " \
  --prompt-file prompts/visual_mcq_huihui_ocr_strict.txt \
  --ocr-json Result/MCQ/ocr_space_mcq_visual_test_engine2_merged_success.jsonl \
  --ocr-max-chars 2200 \
  --image-variant enhanced \
  --enhance-longest-side 768 \
  --output outputs/visual_mcq_huihui_qwen36_27b_abliterated_ocr_prompt_test.json
```

Use OCR in an OpenQA ablation:

```bash
python scripts/run_visual_openqa_qwen3vl.py \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --dataset SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual \
  --split train \
  --prompt-file prompts/visual_openqa_ocr_strict.txt \
  --ocr-json Result/OpenQA/ocr_space_openqa_visual_train_engine2_merged_success_v3.jsonl \
  --ocr-max-chars 2600 \
  --max-new-tokens 192 \
  --image-variant enhanced \
  --output outputs/openqa_qwen3vl8b_thinking_ocr_train_legacy.json \
  --gold-output outputs/openqa_qwen3vl8b_thinking_ocr_train_gold.json
```

## Development Results

### MCQ Single Models

| Model | Size | Accuracy |
|---|---:|---:|
| Qwen3.6-35B-A3B base | 36.1B/A3B | 0.6849 |
| Huihui-Qwen3.6-27B-abliterated | 27B | 0.6822 |
| Qwen2.5-VL-32B | 32B | 0.6052 |
| Qwen3.6-35B-A3B LoRA epoch 1 | 36.1B/A3B | 0.5801 |
| Qwen3-VL-8B-Thinking direct | 8B | 0.5774 |
| Qwen3-VL-8B-Thinking LoRA-300 | 8B | 0.5649 |
| Huihui-Qwen3.6-27B with OCR prompt | 27B | 0.5542 |
| Qwen3-VL-8B-Thinking with OCR prompt | 8B | 0.5103 |

### MCQ Ensembles and Merges

| Run | Method | Accuracy |
|---|---|---:|
| Huihui-Qwen3.6-27B + Qwen3.6-35B-A3B base | Weighted score fusion, weights 2.0/1.0 | 0.7126 |
| Official FAU MCQ ensemble | Weighted voting / score fusion | 0.7108 |
| Qwen3.6 base + Qwen3.6 LoRA | Weighted score fusion, weights 3.0/1.0 | 0.6885 |
| Qwen3.6 base + Qwen3.6 LoRA | Confidence router | 0.6777 |
| Three Qwen2.5-VL-7B variants | Majority vote | 0.5452 |

### OpenQA Single Models on Train

| Model | Size | BLEU | ROUGE-L | METEOR | COMET |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-32B-Thinking | 32B | 0.1076 | 0.1694 | 0.1514 | 0.6270 |
| Qwen2.5-VL-32B-Instruct | 32B | 0.0973 | 0.1689 | 0.1488 | 0.6125 |
| Qwen3-VL-8B-Thinking | 8B | 0.0848 | 0.1598 | 0.1274 | 0.6093 |
| Qwen3.6-35B-A3B base | 36.1B/A3B | 0.0657 | 0.1418 | 0.1456 | 0.5663 |
| InternVL3-8B-hf | 8B | 0.0326 | 0.1126 | 0.1030 | 0.4681 |
| Aya Vision 32B | 32B | 0.0194 | 0.0664 | 0.0576 | 0.4667 |
| Phi-4 Reasoning Vision | 15B | 0.0268 | 0.0815 | 0.0554 | 0.4344 |
| Mistral Small FP8 | 24B | 0.0040 | 0.0200 | 0.0064 | 0.4248 |
| InternVL3.5 38B | 38B | 0.0076 | 0.0369 | 0.0401 | 0.4150 |

### OpenQA OCR Ablation on Train

| Model | OCR | BLEU | ROUGE-L | METEOR | COMET |
|---|---|---:|---:|---:|---:|
| Qwen3-VL-32B-Thinking | No | 0.1076 | 0.1694 | 0.1514 | 0.6270 |
| Qwen3-VL-32B-Thinking | Yes | 0.1040 | 0.1698 | 0.1264 | 0.6078 |
| Qwen3-VL-8B-Thinking | No | 0.0848 | 0.1598 | 0.1274 | 0.6093 |
| Qwen3-VL-8B-Thinking | Yes | 0.0857 | 0.1496 | 0.1110 | 0.6043 |

## Paper

The CEUR-WS system paper is in:

```text
paper/imageclef2026_mr_system.tex
```

Figures are exported as vector PDFs:

```text
paper/MCQ_Figure.pdf
paper/OpenQA_Figure.pdf
```

## Troubleshooting

### Gated dataset or model

```bash
hf auth login
```

Then confirm access:

```bash
python - <<'PY'
from datasets import load_dataset
print(load_dataset("SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual", split="test"))
PY
```

### Qwen3-VL import errors

Install the Qwen3-VL environment in a fresh virtual environment:

```bash
pip install -r requirements-qwen3vl.txt
```

### Unsloth conflicts after installing Qwen3-VL

Create a separate environment and install:

```bash
pip install -r requirements-unsloth-qwen36.txt
```

### CUDA out of memory

Use one or more of:

```bash
--load-in-4bit
--image-variant enhanced
--enhance-longest-side 768
```

Also set:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

### Long jobs

Run with `nohup` and tail the log:

```bash
nohup python <script.py> <args> > run.log 2>&1 &
tail -f run.log
```

## Citation

If you use this repository, please cite the ImageCLEF 2026 overview paper, the ImageCLEF 2026 Multimodal Reasoning task overview paper, and the FAU working-notes system paper.

```bibtex
@inproceedings{ImageCLEFMultimodalReasoningTaskOverview2026,
  title = {{O}verview of the {I}mage{CLEF} 2026 {T}ask on {M}ultimodal {R}easoning},
  author = {Dimitrov, Dimitar and Hee, Ming Shan and Ahsan, Momina and Ahmad, Sarfraz and Zlatkova, Dimitrina and Pachov, Georgi and Xie, Zhuohan and Nakov, Preslav and Koychev, Ivan},
  booktitle = {CLEF 2026 Working Notes},
  series = {CEUR Workshop Proceedings},
  year = {2026},
  month = {September 21--24},
  address = {Jena, Germany},
  publisher = {CEUR-WS.org}
}
```
