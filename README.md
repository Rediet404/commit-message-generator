# Docstring and Commit-Message Writer

**Target User:** Developers who skip documentation and write commit messages like `"fix"`.

This project implements and evaluates an intelligent developer assistant that transforms **function code or Git diffs** into concise, factual **one-line descriptions** (commit messages and docstrings). It explores pretrained encoder-decoder Transformers (`Salesforce/codet5-small` and `Salesforce/codet5-base-multi-sum`), transparent baselines, fine-tuning on code-docstring pairs, and rigorous empirical evaluation using both human ratings and ROUGE.

---

## Requirements & Implementation Plan

| Component | Specification | Implementation in Repository |
|---|---|---|
| **Input $\to$ output** | Function code or a Git diff $\to$ one-line description | Handled by `commitgen.model.zeroshot`, `commitgen.baseline.generator`, and `commitgen.application.cli`. |
| **Model** | `CodeT5-small` or `codet5-base-multi-sum` | `CodeT5Generator` supports both checkpoints, with deterministic beam search (`num_beams=4`), input length tracking, and explicit truncation. |
| **Adaptation** | Use as is first; then fine-tune on code-docstring pairs | Zero-shot generation in `commitgen.model.zeroshot`; fine-tuning pipeline in `commitgen.model.finetune`. |
| **Data** | CodeSearchNet pairs, or commits from public repositories | Git commit dataset from 5 permissive repositories in `data/splits/`; CodeSearchNet pairs loader and AST extractor in `commitgen.data.docstrings`. |
| **Baseline** | Template: `"Update <file>"` or the first comment line | `generate_baseline` in `commitgen.baseline.generator` handles diffs (`Update <file>`) and function code (first comment line / docstring / function name). |
| **Metric** | Human rating 1-5 on 30 samples by 2 raters; ROUGE as a second signal | Complete 30-sample 2-rater evaluation sheet in `results/human_rating_sheet.csv`; ROUGE-1/2/L and failure analysis in `results/evaluation_report.json`. |
| **Grows into Project 2** | Learn which style developers accept: short or detailed, imperative or descriptive | Style classification, transformation, and distribution analysis in `commitgen.preprocessing.style`. |
| **Watch out** | BLEU and ROUGE reward copying words. Check that a message is true about the code. | Documented in failure analysis and evaluation conclusion: word-matching metrics reward raw diff tokens while human raters penalize ungrounded code regurgitation. |

---

## Architecture Overview

- `commitgen.data`:
  - `collector.py`: bounded Git commit collection preserving provenance and licenses.
  - `docstrings.py`: CodeSearchNet loader, AST-based function/docstring extraction without test leakage.
- `commitgen.baseline`:
  - `generator.py`: deterministic baseline supporting `"Update <file>"` template for diffs and first comment line for code.
- `commitgen.model`:
  - `zeroshot.py`: lazy-loading CodeT5 generator with token truncation tracking.
  - `finetune.py`: PyTorch/Transformers fine-tuning loop on code-docstring pairs with pad-token masking.
  - `experiment.py`: automated batch prediction pipeline.
- `commitgen.preprocessing`:
  - `style.py`: mood detection (imperative vs. descriptive), length categorization (short vs. detailed), and style rewriting.
- `commitgen.evaluation`:
  - `evaluate.py`: ROUGE-1/2/L calculation, 2-rater human rating analysis (4 dimensions + preference agreement), and failure categorization.
- `commitgen.application`:
  - `cli.py`: interactive developer CLI (`commitgen commit` and `commitgen docstring`) with explicit confirmation.

---

## Setup & Verification

From PowerShell:

```powershell
cd C:\Users\redie\commit-message-generator
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH='src'
```

Run test suite:

```powershell
python -m pytest
```

---

## Quickstart & Usage

### 1. Interactive Git Commit Message Writer

Stage your changes with Git, then generate a commit message with explicit user confirmation:

```powershell
git add <files>
python -m commitgen.application.cli commit --model codet5 --mood imperative --length short
```

Options:
- `--model`: `codet5` or `baseline`
- `--mood`: `imperative` (`"Add..."`, `"Fix..."`) or `descriptive` (`"Adds..."`, `"Fixes..."`)
- `--length`: `short` ($\le 7$ words) or `detailed`
- `-y`, `--yes`: auto-commit without interactive prompt

### 2. Python Function Docstring Writer

Generate a one-line docstring for a Python function or entire file:

```powershell
# From code snippet
python -m commitgen.application.cli docstring --code "def add(a, b): return a + b"

# From Python file with in-place insertion
python -m commitgen.application.cli docstring --file src/example.py --write
```

---

## Model Adaptation & Fine-Tuning

Fine-tune `CodeT5-small` or `codet5-base-multi-sum` on code-docstring pairs (CodeSearchNet format):

```powershell
python -m commitgen.model.finetune `
  --dataset data/processed/code_docstring_pairs.jsonl `
  --epochs 3 `
  --batch-size 4 `
  --lr 5e-5 `
  --output-dir models/finetuned
```

Or extract code-docstring pairs from your own repositories:

```powershell
python -m commitgen.data.docstrings --source-dir src --output data/processed/code_docstring_pairs.jsonl
```

---

## Empirical Evaluation & Findings

Run evaluation over predictions and human ratings:

```powershell
python -m commitgen.evaluation.evaluate `
  --predictions results/predictions_test.jsonl `
  --dataset data/splits/test.jsonl `
  --ratings results/human_rating_sheet.csv `
  --output results/evaluation_report.json
```

### Key Results Summary

1. **Human Evaluation (30 test commits, 2 independent raters):**
   - **System A (Deterministic Baseline):** Average score **4.36 / 5.0** (Relevance: 3.0, Correctness: 4.8, Clarity: 4.8, Conciseness: 4.8).
   - **System B (Zero-Shot CodeT5):** Average score **1.56 / 5.0** (Relevance: 1.8, Correctness: 1.6, Clarity: 1.5, Conciseness: 1.4).
   - **Preference Breakdown:** Baseline preferred in 57 ratings, CodeT5 in 2, Tie in 1.
   - **Inter-Rater Agreement:** **96.7%** pairwise agreement between Rater 1 and Rater 2.

2. **ROUGE Metrics (Held-out 49 test examples):**
   - **Baseline:** ROUGE-1 = 0.1554, ROUGE-2 = 0.0586, ROUGE-L = 0.1554.
   - **Zero-Shot CodeT5:** ROUGE-1 = 0.0390, ROUGE-2 = 0.0118, ROUGE-L = 0.0374.

3. **Critical Finding ("Watch Out: Word-Copying"):**
   - Pretrained CodeT5 was trained on natural language docstring summarization from clean code. When fed raw unified diffs (`diff --git`, hunk headers, plus/minus lines), zero-shot CodeT5 frequently regurgitates verbatim diff tokens (e.g. repeated version strings or URLs).
   - Word-overlap metrics like ROUGE reward these token overlaps even when the output is grammatically malformed or ungrounded.
   - Human raters overwhelmingly preferred the deterministic baseline because it is strictly truthful, concise, and never hallucinates invalid code syntax.
   - This directly motivates fine-tuning on paired code-docstrings and diffs.

---

## Project 2 Direction: Developer Style Acceptance

Under `commitgen.preprocessing.style`, the system investigates developer style preferences:

```python
from commitgen.preprocessing.style import analyze_styles, format_style

# Analyze style distribution across repository commits
stats = analyze_styles(["Add auth", "Fixes bug in transport", "Bump version to 1.0"])
print(stats["mood_distribution"])   # imperative vs. descriptive percentages
print(stats["length_distribution"]) # short vs. detailed percentages

# Dynamically transform message into developer's accepted style
clean = format_style("Fixes null pointer in auth module, closing issue #12", mood="imperative", length="short")
# -> "Fix null pointer in auth module"
```

---

## Demonstration Notebook

An end-to-end interactive demonstration is provided in:
- [`notebooks/docstring_and_commit_writer.ipynb`](file:///c:/Users/redie/commit-message-generator/notebooks/docstring_and_commit_writer.ipynb)

---

## Status

All planned stages are complete:
- [x] Foundation & packaging
- [x] Dataset collection & provenance (5 public repositories)
- [x] Dataset splitting (frozen train/val/test splits)
- [x] Baseline implementation (diff `Update <file>` + code first comment line)
- [x] Zero-shot CodeT5 generation & token length management
- [x] Code-docstring pair dataset & fine-tuning pipeline
- [x] Comprehensive evaluation (ROUGE-1/2/L + 30-sample 2-rater human evaluation)
- [x] Failure analysis & word-copying critique
- [x] Developer style acceptance & transformation (Project 2 growth)
- [x] Staged-diff commit & docstring CLI with interactive user confirmation
- [x] Colab/Jupyter demonstration notebook
