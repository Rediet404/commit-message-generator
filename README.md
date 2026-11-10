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

The current stage contains only project foundation code. Data collection,
preprocessing, model generation, evaluation, and the CLI will be added and
validated incrementally.

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

## Verify the foundation

```powershell
python -c "import commitgen; print(commitgen.__version__)"
python -m pytest
```

## Planned experiment stages

1. **Foundation:** packaging, configuration, logging, and tests.
2. **Dataset research:** verify CommitPack's file-level schema, group records
   by commit ID, reconstruct one complete diff per commit, and preserve
   repository and license metadata.
3. **Dataset preparation:** clean records, remove leakage and duplicates, and
   freeze repository-separated or chronological train/validation/test splits.
4. **Baseline:** implement and evaluate `Update <file>` and optional comment
   extraction.
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
license information. CommitPack records are file-level, so future processing
must group modifications by commit ID before creating a single training
example for that commit.

## Status

Stage 1 (project foundation) is in progress. No model or dataset results are
claimed yet.