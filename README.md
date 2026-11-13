# Group Members:

## Rediet Mesfin     ATE/502014
## Yohannes Siyum    ATE/5195/14 
## Esrom  Basaznew   ATE/5227/14
## Kalkidan Tadesse  ATE/0652/14
## Eisha Gieder      ATE/1084/14

# Docstring and Commit-Message Writer

**Target User:** Developers who skip documentation and write commit messages like `"fix"`.

This project implements and evaluates an intelligent developer assistant that transforms **function code or Git diffs** into concise, factual **one-line descriptions** (commit messages and docstrings). It supports configurable pretrained encoder-decoder Transformers, including `Salesforce/codet5-small` and `Salesforce/codet5-base-multi-sum`, transparent baselines, fine-tuning on code-docstring or commit-pair data, and empirical evaluation using human ratings and ROUGE.

---

## How to run

Open PowerShell at the project folder and install the package in the virtual
environment:

```powershell
cd <project-root>
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

Check that the CLI is available:

```powershell
commitgen --help
```

Generate a commit message from staged changes:

```powershell
git add <files>
commitgen commit --model baseline
```

Use CodeT5 instead of the baseline:

```powershell
commitgen commit --model codet5
```

Use the locally fine-tuned checkpoint:

```powershell
commitgen commit `
  --model codet5 `
  --model-path models/finetuned-commits
```

The CLI displays the proposed message and asks for confirmation before
committing. Do not use `--yes` during a demonstration unless you intentionally
want to create a Git commit.

Generate from a diff file without staging changes:

```powershell
commitgen commit --model baseline --diff .\path\to\your.diff
```

Generate a one-line docstring from Python code:

```powershell
commitgen docstring `
  --model baseline `
  --code "def add(a, b): return a + b"
```

Run the tests and compilation check:

```powershell
python -m pytest
python -m compileall -q src tests
```

If `commitgen` is not recognized, run `python -m pip install -e .` again. If
you use the source-tree module form instead, set:

```powershell
$env:PYTHONPATH='src'
python -m commitgen.application.cli --help
```

For all CLI options, fine-tuning commands, evaluation commands, and
troubleshooting, see the detailed [CLI guide](#cli-guide) below.

---

## Requirements & Implementation Plan

| Component | Specification | Implementation in Repository |
|---|---|---|
| **Input $\to$ output** | Function code or a Git diff $\to$ one-line description | Handled by `commitgen.model.zeroshot`, `commitgen.baseline.generator`, and `commitgen.application.cli`. |
| **Model** | `CodeT5-small` or `codet5-base-multi-sum` | `CodeT5Generator` supports both checkpoints, with deterministic beam search (`num_beams=4`), input length tracking, and explicit truncation. |
| **Adaptation** | Use as is first; then fine-tune on paired examples | Zero-shot generation and a one-epoch commit diff/subject fine-tuning run are implemented; the fine-tuning pipeline also accepts CodeSearchNet-format code/docstring records. |
| **Data** | CodeSearchNet pairs, or commits from public repositories | Primary experiment: 1,000 commits from 5 permissive repositories in `data/splits/`; `commitgen.data.docstrings` implements CodeSearchNet-format loading and AST extraction, but no CodeSearchNet experiment or full download is included. |
| **Baseline** | Template: `"Update <file>"` or the first comment line | `generate_baseline` in `commitgen.baseline.generator` handles diffs (`Update <file>`) and function code (first comment line / docstring / function name). |
| **Metric** | Human rating 1-5 on 30 samples by 2 raters; ROUGE as a second signal | Complete 30-sample 2-rater evaluation sheet in `results/human_rating_sheet.csv`; ROUGE-1/2/L and failure analysis in `results/evaluation_report.json`. |
| **Grows into Project 2** | Learn which style developers accept: short or detailed, imperative or descriptive | Style classification, transformation, and distribution analysis in `commitgen.preprocessing.style`. |
| **Watch out** | BLEU and ROUGE reward copying words. Check that a message is supported by the code. | Documented in failure analysis and evaluation conclusion: word-matching metrics reward raw diff tokens while human raters penalize ungrounded code regurgitation. |

---

## Architecture Overview

- `commitgen.data`:
  - `collector.py`: bounded Git commit collection preserving provenance and licenses.
  - `docstrings.py`: CodeSearchNet loader, AST-based function/docstring extraction without test leakage.
- `commitgen.baseline`:
  - `generator.py`: deterministic baseline supporting `"Update <file>"` template for diffs and first comment line for code.
- `commitgen.model`:
  - `zeroshot.py`: lazy-loading CodeT5 generator with token truncation tracking.
  - `finetune.py`: PyTorch/Transformers fine-tuning loop on code-docstring or commit diff/subject pairs with pad-token masking.
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
cd <project-root>
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
$env:PYTHONPATH='src'
```

The editable install provides the `commitgen` command. The `PYTHONPATH` line is
also supported for running the project directly from the source tree.

Run test suite:

```powershell
python -m pytest
python -m compileall -q src tests
```

The frozen repository-aware split contains 724 training examples, 227
validation examples, and 49 test examples. The complete test split comes from
`encode/httpx`, while training uses `psf/requests`, `pallets/click`, and
`pallets/flask`, and validation uses `pydantic/pydantic`. This prevents
repository overlap between partitions, but means the 49-example test result
measures generalization to HTTPX specifically rather than broad
cross-repository generalization.

---

## CLI guide

The project is primarily run through a command-line interface. Use either the
installed `commitgen` command or the equivalent Python module command:

```powershell
commitgen --help
python -m commitgen.application.cli --help
```

The CLI has two commands:

| Command | Purpose |
|---|---|
| `commit` | Generate a commit subject from staged Git changes or a supplied diff |
| `docstring` | Generate a one-line description for Python function code |

### Generate a message from staged changes

First stage changes normally:

```powershell
git add <files>
commitgen commit --model baseline
```

The CLI displays the proposed message and asks whether to commit, edit, or
abort. It does not create a commit unless you confirm it. To use CodeT5:

```powershell
commitgen commit --model codet5
```

CodeT5 runs locally on CPU by default. The first CodeT5 run may download
`Salesforce/codet5-small`; later runs use the local model cache.

For a safe demonstration without staged changes, pass a diff file:

```powershell
commitgen commit --model baseline --diff .\path\to\your.diff
```

The `--diff` option accepts either a path to a text diff file or a raw diff
string. The proposed message is shown, but the command should be run without
`--yes` during demonstrations so no unintended Git commit is created.

To bypass confirmation for an intentional commit:

```powershell
commitgen commit --model baseline --yes
```

Use `--yes` only when the staged changes and generated message have already
been reviewed.

### Generate a function docstring

For a short Python function supplied on the command line:

```powershell
commitgen docstring `
  --model baseline `
  --code "def add(a, b): return a + b"
```

For a Python file:

```powershell
commitgen docstring `
  --model baseline `
  --file .\src\example.py
```

To insert the generated docstring into the first function without a docstring:

```powershell
commitgen docstring `
  --model baseline `
  --file .\src\example.py `
  --write
```

Review the file after using `--write`. The CLI modifies the selected file only
when this option is explicitly provided.

### Common options

Both commands support:

```text
--model baseline|codet5
--model-path <local CodeT5 checkpoint directory>
--mood imperative|descriptive
--length short|detailed
```

`baseline` is deterministic and does not download a model. `codet5` uses the
pretrained model and may take longer on CPU. `--model-path` selects a local
fine-tuned CodeT5 checkpoint instead of the default pretrained checkpoint.
`short` limits the final formatted message to a short subject; `detailed` does
not apply that shortening step.

### CLI troubleshooting

- If `commitgen` is not recognized, activate the virtual environment and run
  `python -m pip install -e .` again.
- If imports fail when using the module form, set
  `$env:PYTHONPATH='src'` in the current PowerShell session.
- If CodeT5 cannot load, use `--model baseline` to verify the CLI and inspect
  the reported model-cache or network error. Do not silently substitute another
  model.
- If the commit command reports no staged changes, run `git add <files>` or
  pass a diff file with `--diff`.

## Model Adaptation & Fine-Tuning

Fine-tune `CodeT5-small` on the collected commit pairs using the frozen
training and validation splits. The test split is not used during training:

```powershell
python -m commitgen.model.finetune `
  --dataset data/splits/train.jsonl `
  --validation-dataset data/splits/validation.jsonl `
  --epochs 1 `
  --batch-size 2 `
  --lr 5e-5 `
  --output-dir models/finetuned-commits
```

For a bounded CPU smoke test:

```powershell
python -m commitgen.model.finetune `
  --dataset data/splits/train.jsonl `
  --validation-dataset data/splits/validation.jsonl `
  --smoke 4 `
  --validation-smoke 2 `
  --epochs 1 `
  --batch-size 2 `
  --output-dir models/finetuned-commits-smoke
```

The training report records training/validation sizes and how many inputs
exceeded the 512-token training limit. Do not evaluate or tune on
`data/splits/test.jsonl` until training choices are frozen.

The fine-tuning loop also accepts code-docstring JSONL records with `code` and
`docstring` fields. To extract those pairs from a repository:

```powershell
python -m commitgen.data.docstrings --source-dir src --output data/processed/code_docstring_pairs.jsonl
```

After training, generate predictions on the unchanged held-out test split:

```powershell
python -m commitgen.model.experiment `
  --split data/splits/test.jsonl `
  --output-dir results/finetuned `
  --model-name models/finetuned-commits `
  --output-name test
```

The fine-tuned checkpoint from the one-epoch CPU run produced ROUGE-1
`0.1478`, ROUGE-2 `0.0371`, and ROUGE-L `0.1446` on the 49 test examples.
This is below the filename baseline's ROUGE-1 `0.1554` and ROUGE-L `0.1554`,
and no new human ratings were collected for the fine-tuned system. These
results are therefore evidence of partial improvement over zero-shot CodeT5,
not evidence that fine-tuning has surpassed the baseline.

The separate fine-tuned report is
`results/finetuned_evaluation.json`. The original human-rated zero-shot
comparison remains in `results/evaluation_report.json`.

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
   - **Preference Breakdown:** The 30 examples were rated by 2 raters, producing 60 preference decisions in total: the baseline was preferred 57 times, CodeT5 2 times, and the systems were tied once.
   - **Inter-Rater Agreement:** **96.7%** descriptive pairwise agreement, meaning that the two raters selected the same preference on 29 of the 30 examples. This is separate from the aggregate 57/2/1 count. It is a small-sample descriptive measure, not a statistical reliability estimate, significance test, or evidence that the ratings are unbiased.

2. **ROUGE Metrics (Held-out 49 test examples):**
   - **Baseline:** ROUGE-1 = 0.1554, ROUGE-2 = 0.0586, ROUGE-L = 0.1554.
   - **Zero-Shot CodeT5:** ROUGE-1 = 0.0390, ROUGE-2 = 0.0118, ROUGE-L = 0.0374.
   - **One-epoch fine-tuned CodeT5:** ROUGE-1 = 0.1478, ROUGE-2 = 0.0371, ROUGE-L = 0.1446.
   - The fine-tuned result was evaluated on the same frozen test split, but it has not received a new human-rating round.

3. **Critical Finding ("Watch Out: Word-Copying"):**
   - Pretrained CodeT5 was trained on natural language docstring summarization from clean code. When fed raw unified diffs (`diff --git`, hunk headers, plus/minus lines), zero-shot CodeT5 frequently regurgitates verbatim diff tokens (e.g. repeated version strings or URLs).
   - Word-overlap metrics like ROUGE reward these token overlaps even when the output is grammatically malformed or ungrounded.
   - Human raters overwhelmingly preferred the deterministic baseline because it is conservative and concise; its filename-based output can still be incomplete or fail to describe the actual purpose of a change.
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
- [`notebooks/docstring_and_commit_writer.ipynb`](./notebooks/docstring_and_commit_writer.ipynb)

The notebook has been executed with the current source tree and includes
captured outputs for:

- baseline generation from a diff, a commented function, and a function-name
  fallback (`Update auth`, `Authenticate user session with token verification`,
  and `Execute calculate total tax`);
- CodeT5 generator initialization with the default CPU settings;
- loading 8 starter code-docstring pairs;
- the completed ROUGE and human-rating report (30 samples, 2 raters, 96.7%
  agreement);
- developer-style analysis (75% imperative, 87.5% short messages in the
  eight-message demonstration sample); and
- the programmatic CLI helpers (`Update config` and
  `Return hex digest of SHA-256 hash`).

The CodeT5 and fine-tuning cells intentionally demonstrate configuration and
lazy initialization. They do not download a model or start a training run.
Use the CLI commands and fine-tuning commands above when you want to run those
operations explicitly.

---

## Status

All planned stages are complete:
- [x] Foundation & packaging
- [x] Dataset collection & provenance (5 public repositories)
- [x] Dataset splitting (frozen train/val/test splits)
- [x] Baseline implementation (diff `Update <file>` + code first comment line)
- [x] Zero-shot CodeT5 generation & token length management
- [x] Commit diff/subject fine-tuning pipeline and one-epoch checkpoint
- [x] Comprehensive evaluation (ROUGE-1/2/L + 30-sample 2-rater human evaluation)
- [x] Failure analysis & word-copying critique
- [x] Developer style acceptance & transformation (Project 2 growth)
- [x] Staged-diff commit & docstring CLI with interactive user confirmation
- [x] Executed Colab/Jupyter demonstration notebook with captured outputs
