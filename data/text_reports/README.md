# Text severity dataset (docs/08 §4, docs/15 D2) 

Tag: **SIM** (LLM-drafted, team-reviewed, not clinician-labelled).

## File: `reports.csv` (UTF-8)

| column | type | meaning |
|---|---|---|
| `report_id` | str, unique | e.g. `en-0001` |
| `language` | `en` / `hi` / `kn` | Hindi/Kannada in native script or romanised, as a bystander would speak |
| `text` | str | one short bystander report (1–3 sentences, may be panicked/incomplete) |
| `label` | 0–3 | rubric below |
| `reviewed` | 0 / 1 | 1 only after a human read and fixed it; only reviewed rows are used |
| `split` | `train` / `test` | leave empty; filled once by `--assign-split` and never changed |
| `notes` | str, optional | reviewer notes |

## Rubric (docs/08 §4)
- **3 critical**: unconscious / not breathing / heavy uncontrolled bleeding / major trauma
- **2 severe**: conscious but serious injury, or heavy bleeding that is controllable
- **1 moderate**: minor–moderate injury, stable
- **0 stable**: no injury / low concern

## Steps
1. Draft ~150 reports per language from templates (injury type × severity × style × noise).
2. Hand-review every row, set `reviewed=1`. Hindi/Kannada: native speaker review.
3. `python -m backend.ml.text_severity --assign-split` (once; fixes the 20% held-out test set).
4. `make train-text` (or `python -m backend.ml.text_severity --save-model`).
   Do not change prompts or the rubric after looking at test results.

## Draft file (generated)
`reports.csv` is first produced by `python -m scripts.draft_text_reports` (450 rows, 150 per language,
balanced over 5 injury types x 4 labels; hi/kn include 40 romanised "typing style" rows each; seed 42).
All rows are `reviewed=0` drafts: Hindi/Kannada wording is machine-drafted [TODO-VERIFY native review].
Re-running needs `--force` and **destroys review work**, so generate once. The file has a UTF-8 BOM so
Excel shows Devanagari/Kannada; the loader accepts it. Openers/suffixes are shared across labels, so
the label signal is in the core phrase: reviewers should also rewrite ~30% of rows in their own words
and add 20-30 hand-typed reports, otherwise the test score mostly measures template recovery.
