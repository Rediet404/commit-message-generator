# Transformer-Based Git Commit Message Generator

This project explores whether a pretrained encoder-decoder Transformer can
generate concise, meaningful commit messages from Git diffs. It is an academic
project for Addis Ababa University's Selected Topics in Software Engineering
course.

## Problem and objectives

Developers sometimes commit changes with vague messages such as `fix`. The
project targets developers who want a factual, consistent suggestion based on
their staged changes.

The project objectives are to:

- compare a simple filename-based baseline with `Salesforce/codet5-small`;
- evaluate both systems on a frozen held-out test set;
- measure text quality and efficiency without fabricating results;
- document observed failures and limitations; and
- provide a reproducible Python project that can run locally or in Google
  Colab.

## Architecture overview

The planned system is organized into separate layers:

- `commitgen.data`: dataset loading and provenance;
- `commitgen.preprocessing`: filtering, diff preparation, and splitting;
- `commitgen.baseline`: a transparent template baseline;
- `commitgen.model`: tokenizer, model loading, generation, and optional
  fine-tuning;
- `commitgen.evaluation`: automatic metrics, human-rating templates, and
  failure analysis; and
- `commitgen.application`: the later Git-facing CLI.

The dataset collector is implemented as the first Stage 3 component. It uses
bounded Git clones and writes local JSONL artifacts; generated data remains
ignored by Git.

## Setup

From Windows PowerShell:

```powershell
cd C:\Users\redie\commit-message-generator
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The foundation currently uses only the Python standard library at runtime.
The requirements file contains the test dependency needed for local
development. ML dependencies will be added only when their implementation
stage begins.

## Verify the project

```powershell
python -c "import commitgen; print(commitgen.__version__)"
python -m pytest
```

## Collect the real commit dataset

The collector uses five public repositories whose checked-in license files are
verified after cloning: Requests (Apache-2.0), Flask (BSD-3-Clause), Click
(BSD-3-Clause), Pydantic (MIT), and HTTPX (BSD-3-Clause). It performs one
complete parent-to-commit diff per non-merge commit and records the source
repository, commit SHA, date, and license.

From Windows PowerShell:

```powershell
cd C:\Users\redie\commit-message-generator
.\.venv\Scripts\Activate.ps1
python -m commitgen.data.collector --output-dir data --target-examples 2000
```

The bounded default is 800 commits inspected per repository and a 100,000-byte
maximum diff. The command writes ignored artifacts to:

- `data/raw/commits.jsonl`: collected records before the final target cap;
- `data/processed/commits.jsonl`: accepted records;
- `data/processed/quality_report.json` and `quality_report.md`;
- `data/splits/*.jsonl` and `data/splits/manifest.json` when a repository-aware
  split is valid.

Records with missing parents, merge commits, binary changes, generated/vendor
or lock files, empty or vague subjects, empty diffs, oversized diffs, and
duplicates are excluded. Oversized diffs are skipped rather than truncated.

The collector preserves upstream provenance and does not relicense repository
content. Each source repository and commit SHA remains in every record. A
repository-aware split is emitted only when at least five repositories are
available; otherwise the report explicitly marks the split as invalid.

## Stage 4: baseline and CodeT5 zero-shot generation

The deterministic baseline extracts changed paths from each diff and emits a
short filename-based subject such as `Add auth` or `Update 3 py files`. It does
not inspect the reference subject.

The zero-shot experiment uses the pretrained
[`Salesforce/codet5-small`](https://huggingface.co/Salesforce/codet5-small)
checkpoint with `AutoTokenizer` and `AutoModelForSeq2SeqLM`. The input is:

```text
summarize: <complete Git diff>
```

Inputs are measured before generation. Inputs over 512 tokens are explicitly
truncated to 512 tokens and the prediction records retain the original token
count and truncation flag. Generation is deterministic: CPU execution,
`num_beams=4`, `do_sample=False`, and `max_new_tokens=32`.

Install the runtime dependencies and run the ten-example smoke test:

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH='src'
python -m commitgen.model.experiment `
  --split data\splits\test.jsonl --output-dir results --smoke 10
```

After a successful smoke test, run the frozen 49-example test split:

```powershell
python -m commitgen.model.experiment `
  --split data\splits\test.jsonl --output-dir results
```

Prediction JSONL files and the anonymous 30-example human-rating sheet are
written under `results/`. This stage performs no fine-tuning or training. The
current test set contains only 49 examples from the `encode/httpx` repository,
so results are not repository-general.

## Stage 5: evaluation and failure analysis

Run the evaluation after predictions exist:

```powershell
cd C:\Users\redie\commit-message-generator
$env:PYTHONPATH='src'
python -m commitgen.evaluation.evaluate `
  --predictions results\predictions_test.jsonl `
  --dataset data\splits\test.jsonl `
  --ratings results\human_rating_sheet.csv `
  --output results\evaluation_report.json `
  --expand-rating-sheet
```

This preserves the 30 existing examples and prediction columns while adding
relevance, correctness, clarity, conciseness, and preference fields for each
of two independent raters. Read `results\human_rating_guide.md` before
rating. Scores are integers from 1 to 5, anchored as Very poor, Poor,
Acceptable, Good, and Excellent.

The report calculates ROUGE-1, ROUGE-2, and ROUGE-L using
`rouge-score==0.1.2`. It reports mean per-example F1 for all 49 examples,
including empty predictions in the denominator. ROUGE measures textual
overlap with the reference subject; it does not establish whether a message
understands the code change.

Human results remain `pending` until both raters complete every required
dimension and preference field. The report lists missing or invalid ratings,
descriptive preference agreement, truncation counts, diff-size groups, and
representative failure-analysis examples. No statistical significance claim
is made.

## Planned experiment stages

1. **Foundation:** packaging, configuration, logging, and tests.
2. **Dataset collection:** collect bounded, real commit/diff pairs from
   permissively licensed repositories and preserve provenance.
3. **Dataset preparation:** clean records, remove leakage and duplicates, and
   freeze the repository-separated split manifest.
4. **Baseline:** implement and evaluate the deterministic filename baseline.
5. **CodeT5 zero-shot:** load the pretrained encoder-decoder model, generate
   deterministic predictions, and record resource usage.
6. **Optional fine-tuning:** run a small reproducible Colab experiment only if
   the verified data and available resources justify it.
7. **Evaluation:** collect human ratings for 30 held-out examples, calculate
   secondary ROUGE metrics, and save predictions.
8. **Failure analysis:** categorize actual errors and report real examples.
9. **Application:** add the staged-diff CLI with explicit user confirmation
   before any commit.
10. **Submission:** complete documentation, reproducibility checks, and the
    demonstration outline.

## Reproducibility and data policy

Raw datasets, model checkpoints, caches, secrets, and generated temporary
files must not be committed. Dataset records must retain provenance and
license information. The collector uses one record per complete commit and
does not modify or override upstream repository license terms.

## Status

Stage 4 (baseline and zero-shot generation) is implemented. No fine-tuning or
training has been performed.