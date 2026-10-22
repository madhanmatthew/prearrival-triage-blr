# Model card — vitals models (docs/05 §6.4)

Research prototype for decision support in a final-year project. **Not a medical device; not clinically
validated.** All numbers are traceable to `experiments/log.csv` (see `docs/RESULTS_SUMMARY.md` §4).

## 1. Models

| Model | Task | Input | Output | Deployed? |
|---|---|---|---|---|
| BiGRU two-head (`backend/ml/seq_model.py`) | 4-class severity + trend (new deterioration within the next 6 readings ≈ 6 h) | last 6 hourly readings × 12 features (7 vitals/flags + 5 deltas) + mask | class probabilities, trend probability | yes, inside the fusion network |
| Fusion network (`backend/ml/fusion_model.py`) | 4-class severity from vitals + bystander text | BiGRU encoding + 19-d text vector (`07` §3) | `FusedRisk` (probs, class, risk score) | yes (`/incident/{id}/vitals`) |
| VitalDB HGB, isotonic-calibrated (`backend/ml/vitaldb_model.py`) | new sustained SBP < 90 or SpO2 < 90 within 10 min (D19) | last 5 one-minute readings (HR, RR, SpO2, SBP, temp) → 25 summaries | calibrated probability + top-3 SHAP reasons | yes (`deterioration_10min_prob`) |

## 2. Data

* **MIMIC-IV Clinical Database Demo v2.2** (PhysioNet, 100 ICU patients, open access): hourly windows, labels
  from real outcomes (death within 24 h / vasopressor or ventilation within 6 h / remaining ICU stay; `08` §2) —
  **never from NEWS2**. 12,299 windows.
* **VitalDB open dataset** (vitaldb.net, intra-operative): 5,877 cases, 1-minute medians, 203,294 windows,
  5,643 subjects, positive rate 13.8 % (`08` §2b).
* **Fusion text: SYNTHETIC.** MIMIC has no bystander reports; text features are drawn from a class-conditional
  generator (`data/text_feature_generator.yaml`) at agreement 0.5 / 0.65 / 0.8.

## 3. Evaluation protocol

Nested patient-grouped cross-validation (D5): outer `StratifiedGroupKFold(5)` for reporting, inner 3-fold for
tuning, grouped by patient (MIMIC) or subject (VitalDB) so no person is in both train and test. All baselines
(majority, NEWS2, logistic regression, random forest, gradient boosting) use the identical outer folds.
Mean ± std over outer folds.

## 4. Results (summary)

* MIMIC severity macro-F1: BiGRU **0.386 ± 0.037** vs NEWS2 0.278, GB 0.368. Trend PR-AUC: BiGRU 0.162 vs NEWS2
  0.061 (AUROC 0.708 vs 0.622).
* VitalDB 10-min deterioration: HGB PR-AUC **0.512 ± 0.008**, AUROC 0.833; BiGRU 0.497 / 0.826; NEWS2 (partial)
  0.242 / 0.660. Calibration (in-sample check): mean predicted 0.137 vs prevalence 0.135.
* Fusion (synthetic text): fused F1 0.473 / 0.575 / 0.721 at agreement 0.5 / 0.65 / 0.8 vs vitals-only 0.386;
  fused does not beat text-only at higher agreement.

## 5. Explainability

* Tree models: SHAP TreeExplainer (D13). MIMIC severity is driven by consciousness (AVPU) and supplemental
  oxygen; trend by SBP and heart rate. VitalDB: low or falling SBP and rising HR raise risk.
* BiGRU: Integrated Gradients (Captum), never SHAP (rule 10).
* Live: the VitalDB model shows its top-3 reasons per reading; temperature is excluded from live reasons because
  its importance in VitalDB is a surgical-workflow artefact (probe not yet placed early in surgery).

## 6. Intended use and limits

* Intended: demonstrate how a live ambulance vitals stream could update a risk estimate for the receiving
  hospital, alongside the bystander report. Decision support only; a clinician decides.
* **Domain gap:** MIMIC is in-hospital ICU data, VitalDB is anaesthetised surgical patients — neither is
  pre-hospital. Numbers do not transfer to ambulance patients without new data.
* MIMIC Demo has only 100 patients: high variance across folds.
* The MIMIC trend horizon is "next 6 readings (≈ 6 h)" (D12); only the VitalDB model may claim 10 minutes, and
  only "on intra-operative data".
* Fusion gain depends on the synthetic text generator; it demonstrates the pipeline, not clinical value.
* Population: US ICU (MIMIC) and Korean surgical (VitalDB) patients; not representative of Indian pre-hospital
  patients, age or sex subgroups (`10`).
* Planned: MIMIC-IV-ED (credentialed PhysioNet access applied 2026-10-20) for an emergency-department severity
  head and a joint model with VitalDB (D19).
