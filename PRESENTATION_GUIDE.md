# Video Presentation Guide

## Recommended length

Aim for 8-10 minutes. The goal is to demonstrate the complete research process,
not to show every source file.

## Before recording

Open these windows or tabs:

1. The project README.
2. `src/commitgen/baseline/generator.py`.
3. `src/commitgen/model/zeroshot.py`.
4. `data/splits/manifest.json`.
5. `results/evaluation_report.json`.
6. A PowerShell terminal at the repository root.

Make sure the model cache is already available if you plan to run CodeT5 live.
If model loading is slow or unreliable, use the saved prediction artifact and
state that the live demonstration is replaying a reproducible completed run.
Do not recollect data or fine-tune during the recording.

## Suggested recording sequence

### 1. Problem and objective — 45 seconds

Explain the problem in one sentence:

> Developers often write vague commit messages such as “fix” or “update”.
> This project investigates whether a model can turn a Git diff into a concise,
> factual one-line commit subject.

State the research comparison:

- a transparent filename-based baseline;
- zero-shot `Salesforce/codet5-small`;
- human evaluation as the primary measure;
- ROUGE as a secondary measure.

### 2. Dataset and leakage control — 60 seconds

Show `data/splits/manifest.json` and explain:

- 1,000 accepted real commits;
- five public repositories with documented permissive licenses;
- one complete parent-to-commit diff per record;
- 724 training, 227 validation, and 49 test examples;
- the test repository is `encode/httpx`;
- repositories are not shared across the three splits.

Say explicitly:

> The repository-aware split reduces repository leakage, but it also means the
> test set is small and comes from one repository. This is an important
> limitation, not something I am hiding.

Do not display the full contents of the rating CSV on screen. Show the rating
guide or the evaluation summary instead.

### 3. Show the baseline — 60 seconds

Open `src/commitgen/baseline/generator.py`. Point out that it:

- extracts changed paths from the diff;
- detects added and deleted files;
- produces a deterministic short message;
- uses a safe fallback such as `Update <file>` or `Update N files`;
- never uses the reference commit subject.

Run a small baseline example:

```powershell
$env:PYTHONPATH='src'
python -c "from commitgen.baseline.generator import generate_diff_baseline; print(generate_diff_baseline('diff --git a/auth.py b/auth.py'))"
```

Explain that this baseline is intentionally simple. Its strength is that it is
readable and conservative; its weakness is that it usually misses the semantic
reason for the change.

### 4. Show CodeT5 zero-shot generation — 90 seconds

Open `src/commitgen/model/zeroshot.py` and point out:

- `Salesforce/codet5-small`;
- `AutoTokenizer` and `AutoModelForSeq2SeqLM`;
- CPU execution;
- input format: `summarize: <Git diff>`;
- maximum input length of 512 tokens;
- explicit token measurement and truncation reporting;
- deterministic generation with four beams and sampling disabled;
- no fine-tuning in this experiment.

If running a live smoke test, use the existing experiment command documented in
the README. Otherwise show `results/predictions_smoke.jsonl` and explain that
the smoke test produced 10/10 successful model outputs.

Important wording:

> This is a zero-shot experiment. The model was not trained on this project's
> commit-message dataset, so this is a test of direct transfer rather than a
> claim that the model has been adapted to Git diffs.

### 5. Show a prediction comparison — 75 seconds

Open `results/predictions_test.jsonl` and show one or two records. Explain that
both systems received the same 49 test examples and that each record preserves:

- repository and commit SHA;
- reference subject;
- baseline prediction;
- CodeT5 prediction;
- input token count;
- truncation flag;
- timing and generation status.

Use the following observed example categories:

- the baseline often produces a generic but safe filename description;
- CodeT5 sometimes copies diff headers, URLs, assertions, or code fragments;
- CodeT5 was successful on 49/49 examples, but successful generation does not
  mean semantically correct generation;
- 20/49 inputs were truncated at the 512-token limit.

Avoid saying truncation caused every failure. Say it is a plausible
contributing factor that requires a controlled follow-up experiment.

### 6. Explain human evaluation — 75 seconds

Show `results/human_rating_guide.md` and the evaluation report. Explain:

- 30 held-out examples;
- two independent raters;
- anonymous System A and System B labels;
- relevance, correctness, clarity, and conciseness;
- 1-5 scale;
- overall preference;
- the reference subject remained visible;
- no missing or invalid ratings.

Clarify that the ratings in the completed sheet are user-provided human
ratings, and the software validates and aggregates them without inventing
values.

### 7. Show results — 90 seconds

Present this table:

| Measure | Baseline | CodeT5 zero-shot |
|---|---:|---:|
| Human overall average | 4.362 / 5 | 1.558 / 5 |
| Preference count | 57 | 2 |
| ROUGE-1 | 0.1554 | 0.0390 |
| ROUGE-2 | 0.0586 | 0.0118 |
| ROUGE-L | 0.1554 | 0.0374 |

Mention the one tie and the descriptive preference agreement of 29/30, or
96.7%.

Then explain the interpretation:

> The two evaluation signals agree in this experiment: the baseline performs
> better. However, ROUGE only measures textual overlap. It does not prove that
> a message correctly understands the code change.

### 8. Failure analysis and conclusion — 90 seconds

Show a representative failure from the evaluation report. Describe what is
observed, without claiming an unsupported cause:

- 16 CodeT5 outputs were categorized as code-like or malformed;
- 30 baseline outputs were categorized as filename-only or generic;
- large and truncated diffs appear frequently among difficult cases;
- the baseline is safe but shallow;
- CodeT5 can be more specific in isolated cases, but is unreliable overall.

End with:

> The project demonstrates that zero-shot CodeT5 is not sufficient for this
> raw-diff task. The result justifies future fine-tuning on paired diffs and
> commit subjects, with a larger and more diverse repository-level test set.

## Commands worth showing

From PowerShell:

```powershell
cd C:\Users\redie\commit-message-generator
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH='src'
python -m pytest
python -m compileall -q src tests
```

Evaluation replay:

```powershell
python -m commitgen.evaluation.evaluate `
  --predictions results/predictions_test.jsonl `
  --dataset data/splits/test.jsonl `
  --ratings results/human_rating_sheet.csv `
  --output results/evaluation_report.json
```

CLI demonstration:

```powershell
python -m commitgen.application.cli commit --model baseline --diff "diff --git a/auth.py b/auth.py"
python -m commitgen.application.cli docstring --model baseline --code "def add(a, b): return a + b"
```

Do not use `--yes` during recording unless you intentionally want to create a
real Git commit. The safe demonstration is to show the proposed message and
choose abort.

## Delivery tips

- Zoom the terminal and editor to at least 125%.
- Keep the results table visible long enough to read.
- Do not read every metric mechanically; explain what it means.
- Do not call the baseline “intelligent” or claim CodeT5 was fine-tuned.
- Say “the experiment found” rather than “the model can never”.
- Mention the single-repository test limitation before presenting the result.
- If a live model download fails, show the saved artifact and report the exact
  limitation rather than switching models or hiding the failure.
