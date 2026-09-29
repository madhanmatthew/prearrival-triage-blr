# Component 2 — Ambulance Vitals & Predictive Triage Agent (Core ML Contribution)

**Owner responsibility:** the project's central machine learning work. This is
the component the paper's contribution section should center on.

**Reads first:** `00_MASTER_OVERVIEW.md` for system context and citations
(DeepTriager, VitalML, NEWS2).

---

## 1. Purpose

This is a single, unified model that operates on **sequential (time-series)**
patient vitals captured in the ambulance, and produces two connected outputs
from one architecture:

1. **Current triage severity** — how serious is the patient right now.
2. **Deterioration trend** — is the patient likely to worsen significantly in
   the next 10-15 minutes, based on how vitals are trending over the sequence of
   readings, not just the latest one. This is the project's central novelty
   claim, and it is built in from the start — not a separate later add-on.

Both outputs come from the same trained system: a sequence encoder over
time-stamped vitals readings, whose representation feeds both the severity
classification and the trend prediction.

Output feeds the fused Emergency Risk Score (see
`05_coordinator_fusion_evaluation.md`) and Component 4 (Hospital Agent).

## 2. Current State (MVP) — honest baseline

**Does not exist.** No vitals-based ML of any kind is present in the current
prototype. Severity is currently inferred only from text. This is entirely new
work, not an upgrade of an existing model.

## 3. Clinical Baseline — NEWS2 (implement first, before the ML model)

NEWS2 (National Early Warning Score 2, Royal College of Physicians UK) is the
real clinical standard this project must benchmark against. Do not skip this —
without it, "our model achieves 89% accuracy" is a meaningless number with
nothing to compare against.

**NEWS2 features and scoring bands (implement exactly as below):**

| Parameter | Scoring bands (0-3) |
|---|---|
| Respiratory rate | ≤8 or ≥25 → 3; 21-24 → 2; 9-11 → 1; else → 0 |
| SpO2 | ≤91 → 3; 92-93 → 2; 94-95 → 1; ≥96 → 0 |
| Supplemental O2 | Yes → 2; No → 0 |
| Systolic BP | ≤90 or ≥220 → 3; 91-100 → 2; 101-110 → 1; else → 0 |
| Heart rate | ≤40 or ≥131 → 3; 111-130 → 2; 41-50 or 91-110 → 1; else → 0 |
| Consciousness (AVPU) | Alert → 0; Voice/Pain/Unresponsive → 3 |
| Temperature | ≤35.0 → 3; 35.1-36.0 or 38.1-39.0 → 1; ≥39.1 → 2; else → 0 |

**Risk banding from total score:** ≥7 → critical(3); 5-6 → severe(2); 1-4 →
moderate(1); 0 → stable(0).

A working Python implementation of this scoring logic already exists — see
`vitals_model.py` (Section 9 below) — reuse it directly rather than
reimplementing.

## 4. Features (6-10 total, matches NEWS2 exactly plus optional additions)

**Core (required, matches NEWS2), captured at each timestamp in the sequence:**
`heart_rate`, `resp_rate`, `spo2`, `systolic_bp`, `temperature`,
`consciousness` (AVPU), `supplemental_o2`

**Optional additions (only if they genuinely improve the model — test with
ablation, don't add blindly):**
`age`, `injury_type` (from Component 1's severity signal), `time_since_incident`
(from ETA model), `bleeding_present` (binary)

**Derived time-series features (used by the sequence encoder, not just raw
values):** rate-of-change between consecutive readings (`hr_delta`,
`spo2_delta`, etc.) — these are what let the model detect a *trend*, not just a
current value.

## 5. Datasets

| Option | What it is | Access |
|---|---|---|
| **MIMIC-III Clinical Database Demo** (preferred) | Real de-identified ICU vitals, 100 patients, hourly time-stamped readings via CHARTEVENTS | physionet.org/content/mimiciii-demo/1.4/ — no credentialing required (unlike full MIMIC-III) |
| **Kaggle "Human Vital Signs Dataset"** (supplement/fallback) | Simulated but clinically structured (HR/SpO2/BP/temp/risk category) | kaggle.com/datasets/nasirayub2/human-vital-sign-dataset — note: single-reading rows, not sequential, so this alone cannot support the trend-prediction output; use it only to supplement the snapshot severity task |

**Sequential data is a hard requirement, not optional.** Because the model
predicts trend, not just current state, it needs 2+ time-stamped readings per
patient. MIMIC-III's CHARTEVENTS table (hourly readings) is the dataset that
actually supports this — plan around it as the primary source, not an
afterthought.

**Critical requirement — read this carefully:** whichever dataset is used, the
model's target label (`severity_label`) must come from a **real outcome**
present in the data (ICU admission, mortality flag, documented risk category) —
**not** derived from the NEWS2 score itself. If you compute
`severity_label = news2_risk_band(...)` and then benchmark your ML model against
NEWS2 using that same derived label, the comparison is circular and
meaningless (NEWS2 will trivially "predict" its own formula perfectly). This
exact mistake was caught during a placeholder-data test run — do not repeat it
with real data.

- MIMIC-III demo: use ICU transfer / mortality / length-of-stay flags from the
  outcome tables as the real label.
- Kaggle dataset: use its `Risk Category` column directly (already independent
  of NEWS2).

## 6. Model Architecture — One Unified Sequence Model

**Encoder:** BiLSTM or GRU over the time-stamped vitals sequence (plus derived
delta features from Section 4). This is the time-series backbone of the whole
component — everything else attaches to it.

**Two output heads from the same encoder:**
1. **Severity head** — classifies current severity band from the encoder's
   representation at the latest timestep.
2. **Trend head** — binary/probability output: will the patient's severity band
   worsen within the next N minutes/readings, based on the full sequence the
   encoder has seen.

**Comparison baselines (train and report all of these, not just the sequence
model in isolation):**
1. Majority-class baseline (dumbest possible baseline)
2. NEWS2 rule-based score (the real clinical standard)
3. Logistic Regression on single-snapshot features (simple tabular baseline)
4. RandomForest / Gradient Boosting on single-snapshot features (stronger
   tabular baseline)
5. The BiLSTM/GRU sequence model (the actual deliverable)

Report accuracy, F1 (macro, since severity classes are imbalanced), and
confusion matrix for all five, on the severity task. Report AUC for the trend
task separately (majority-class and NEWS2 don't have a native way to predict
trend, so the trend task's comparison is primarily sequence-model-vs-tabular-
baseline-with-deltas, and against literature precedent numbers like VitalML's).

This structure exists precisely so the sequence model has to *earn* its place
against simpler baselines — cite VitalML (predicts decompensation up to 90
minutes ahead) as literature precedent for the trend-prediction concept, don't
claim to have invented it.

### Fusion with Component 1 (text severity)
- Concatenation-based fusion (not attention/transformer — DeepTriager shows
  concatenation fusion alone reaches AUC 0.956, which is more than sufficient
  and dramatically simpler to implement and defend in a 2-month timeline).
- Text severity representation (from Component 1's dialogue output) + the
  sequence encoder's hidden representation → concatenated → dense layer(s) →
  final fused severity/risk output.
- Full fusion design and ablation study (proving fusion beats either modality
  alone) documented in `05_coordinator_fusion_evaluation.md`.

## 7. Explainability (SHAP + sequence-appropriate methods)

- Use `shap.TreeExplainer` on the tabular baseline models (RandomForest/
  Gradient Boosting) — works natively with tree ensembles, fast and exact.
- **For the BiLSTM/GRU sequence model, SHAP's TreeExplainer does not apply.**
  Use Integrated Gradients or attention-weight visualization instead. Do not
  force SHAP onto a model type it wasn't designed for — an evaluator familiar
  with XAI will notice the mismatch, and it undermines the credibility of the
  explainability claim rather than supporting it.
- Deliverable: a model card documenting dataset, features, all five baseline
  metrics, explainability outputs, and known limitations.

## 8. Evaluation Checklist

- [ ] NEWS2 baseline implemented and validated against known clinical examples
- [ ] Real (non-NEWS2-derived) severity labels sourced from chosen dataset
- [ ] Sequential (multi-timestamp) data confirmed available per patient — not
  just single-snapshot rows
- [ ] All five baselines (majority-class, NEWS2, Logistic Regression,
  RandomForest/GB, BiLSTM/GRU) benchmarked on the same held-out test set
- [ ] Cross-validation used, not a single train/test split
- [ ] Confusion matrix + F1-macro reported for the severity task across all
  baselines
- [ ] AUC reported for the trend-prediction task
- [ ] SHAP summary generated for tabular baselines; Integrated Gradients or
  attention visualization generated for the sequence model
- [ ] Model card completed with dataset, features, full metrics table, and
  limitations

## 9. Existing Scaffolded Code

A working starting module already exists: `backend/ml/vitals_model.py`. It
currently implements the NEWS2 baseline and a tabular (RandomForest/GB)
snapshot classifier — this is the starting point for the tabular baselines in
Section 6, not the final deliverable. It contains:
- Full NEWS2 scoring implementation (`news2_component_scores`,
  `news2_total_score`, `news2_risk_band`)
- Data loading with real-CSV support (`load_vitals_data`) and a clearly-labeled
  synthetic placeholder generator for pipeline testing
  (`generate_placeholder_vitals_data` — **do not use its output for reported
  results**, it exists only so the pipeline is runnable before real data is
  wired in)
- `train_vitals_severity_model()` — trains RF/GB and returns model + metrics
- `print_comparison()` — ML vs NEWS2 side-by-side reporting
- `explain_with_shap()` — SHAP hook

**Still to be built on top of this:** the BiLSTM/GRU sequence encoder with the
two output heads (severity + trend), trained on sequential MIMIC-III
CHARTEVENTS data, and the fusion layer connecting to Component 1's text
severity output.

**Next concrete step:** replace `generate_placeholder_vitals_data()`'s usage
with a real CSV loaded via `load_vitals_data(csv_path)`, ensure
`severity_label` comes from a real outcome column (not NEWS2-derived), and
re-run to get the first real tabular-baseline numbers — then build the sequence
model on top using sequential CHARTEVENTS data.

## 10. Interface Contract With Other Components

**Input from Component 1:** text severity signal (format TBD, see
`01_reporting_agent.md` Section 10).

**Output to Component 4 (Hospital Agent):** severity score/trend, consumed as a
new input feature replacing the current synthetic severity value in the
existing hospital scoring model.

**Output to Coordinator / fused Emergency Risk Score:** raw model output
(probability distribution over severity classes, plus trend probability) —
exact format to be finalized in `05_coordinator_fusion_evaluation.md`.
