# Video Presentation Script

## Slide 1 — Title

Hello. This presentation is about a Transformer-based Git commit message
generator.

The problem is that developers often write vague commit messages such as
“fix”, “update”, or “changes”. These messages make project history harder to
understand. My project investigates whether a Git diff can be converted into a
concise, factual, one-line commit subject.

The experiment compares two approaches: a transparent rule-based baseline and
the pretrained `Salesforce/codet5-small` model used without fine-tuning.

## Slide 2 — Requirements and research question

The project uses a pretrained Transformer, includes a simple baseline, evaluates
on held-out data, and performs failure analysis. Following the recommended
evaluation approach, human ratings are the primary measure and ROUGE is
secondary.

The main research question is:

> Does zero-shot CodeT5 produce useful commit subjects from raw Git diffs, and
> does it improve on a simple filename-based baseline?

## Slide 3 — Data collection

The dataset contains 1,000 real commits collected from five public repositories:
Requests, Flask, Click, Pydantic, and HTTPX.

Each record contains one complete parent-to-commit diff, the commit subject,
the commit hash, repository provenance, date, and license information. Merge
commits, empty changes, binary-only changes, uninformative subjects, and
generated or vendor files were filtered where identifiable.

The data is divided into 724 training examples, 227 validation examples, and
49 test examples. The split is repository-aware: a repository appears in only
one partition. This reduces leakage, although it creates an important
limitation because all 49 test examples come from HTTPX.

## Slide 4 — System design

The input is a unified Git diff and the output is a one-line commit subject.

The baseline extracts changed file paths and creates a short deterministic
message such as “Update auth” or “Update 3 files”. It can identify added and
deleted files, but it does not try to infer the full semantic purpose of a
change.

The Transformer uses the input format:

```text
summarize: <Git diff>
```

It runs on the CPU with a 512-token input limit, four-beam deterministic
generation, and a maximum of 32 new tokens. Input length is measured before
generation, and truncation is recorded rather than hidden.

Most importantly, this is zero-shot generation. No fine-tuning was performed
for this experiment.

## Slide 5 — Live demonstration

First I will demonstrate the baseline. I provide a small diff involving an
authentication file. The baseline identifies the changed path and returns a
short update message.

This shows the advantage of a transparent baseline: it is fast, deterministic,
and unlikely to make an unsupported semantic claim.

Next I will show the saved CodeT5 prediction artifact. Both systems were run on
the same 49 test examples. The artifact records the reference subject, both
predictions, token count, truncation status, timing, and generation status.

The CodeT5 model generated non-empty outputs for all 49 examples. However,
some outputs copied diff headers, URLs, assertions, or code fragments instead
of producing a natural-language commit subject. Twenty of the 49 inputs were
truncated because they exceeded the 512-token input limit.

This does not prove that truncation caused every failure. It is an observed
condition and a possible contributor that should be tested separately in future
work.

## Slide 6 — Human evaluation

For the primary evaluation, two independent raters assessed the same 30
examples. The systems were shown anonymously as System A and System B.

Each system was rated from 1 to 5 on relevance, correctness, clarity, and
conciseness. The raters also selected System A, System B, or Tie as their
overall preference.

The reference commit subject remained visible, but the model identities were
not shown. The completed sheet contains 30 rows, with no missing or invalid
ratings.

The agreement on paired overall preference was 29 out of 30 comparisons, or
96.7 percent. This is descriptive agreement; no claim of statistical
significance is being made.

## Slide 7 — Results

The baseline received an overall human score of 4.362 out of 5. CodeT5
received 1.558 out of 5.

Across the two raters and 30 examples, the baseline was preferred 57 times,
CodeT5 was preferred twice, and there was one tie.

The ROUGE results show the same ordering:

- Baseline ROUGE-1: 0.1554; ROUGE-2: 0.0586; ROUGE-L: 0.1554.
- CodeT5 ROUGE-1: 0.0390; ROUGE-2: 0.0118; ROUGE-L: 0.0374.

ROUGE was calculated over all 49 test examples, including any empty
predictions. It measures textual overlap with the reference subject, not
whether the output is factually supported by the diff. Therefore human
judgment remains the primary result.

## Slide 8 — Failure analysis

The baseline had 30 outputs categorized as filename-only or generic. This is
its main weakness: it is safe and readable, but often does not explain why the
change was made.

Sixteen CodeT5 outputs were categorized as code-like or malformed. Examples
include copied diff headers, repeated URLs, assertions, line offsets, and
source-code fragments. These outputs demonstrate that a model trained for
code-oriented summarization does not automatically understand the structure of
a raw unified diff.

Large diffs and truncated inputs were common in difficult examples. One
important limitation is that this analysis is based on observed categories and
representative examples; it does not establish a causal relationship between
truncation and model failure.

## Slide 9 — Conclusion

For this experiment, the transparent baseline performed substantially better
than zero-shot CodeT5 in both human evaluation and ROUGE.

The baseline succeeded because it produced conservative, concise messages. It
was not semantically deep, but it avoided unsupported claims. CodeT5 showed
that a pretrained model can generate text, but zero-shot transfer from clean
code summarization to raw Git diffs was not reliable.

The project is therefore complete as a zero-shot baseline study. Fine-tuning
is justified as future work, not as a result already achieved. The next
experiment should train on paired Git diffs and commit subjects, compare
against the same frozen test set, and use a larger and more diverse
repository-level evaluation.

Thank you.
