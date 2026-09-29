# 15 — Final Decisions (resolves conflicts between docs)

**Precedence (highest first):** this file (`15`) > `12_required_edits_to_existing_files.md` >
`07` / `08` / `09` > component files `01`–`05` > `00`.

**Non-normative files (presentation/viva only, never a build spec):** `11`, `13`, `14`, `16`,
`17_PROJECT_EXPLAINER_HOD_FACULTY.md`, `18_PRESENTATION_SCRIPT_7_SLIDES.md`. If they disagree
with a spec file, the spec file is right and the presentation file must be corrected.

---

## 1. Team ownership

| Component | Owner |
|---|---|
| 1 Reporting (Whisper, dialogue state machine, text classifier, RAG) | Sankalp |
| 2 Vitals + fusion (MIMIC, NEWS2, baselines, BiGRU, fusion, XAI) | Chetan |
| 3 Traffic/Police (SUMO, DQN, baselines, police alerts, traffic RF) | Madhan |
| 4 Hospital + integration (hospitals, blood factor, coordinator, API, frontend, benchmark) | Ragavendra |

**Workflow:** all code is written by Madhan with Claude Code. Owners pull the repo, run
training/evaluation on their own machines with one-command Makefile targets, and push results
(rows in `experiments/log.csv`, figures in `reports/`, notes in `docs/progress/<name>.md`).
Each owner must be able to explain their component's code and results in the viva.

## 2. Resolved conflicts

| # | Topic | FINAL decision | Overrides |
|---|---|---|---|
| D1 | Severity scale | 4 classes `0 stable, 1 moderate, 2 severe, 3 critical` everywhere. Fused `risk_score` in [0,1] per `07` §2.3. No 1–5 scales anywhere. | `17` cheat sheet |
| D2 | Text severity classifier | TF-IDF **character n-grams (2–5)** + Logistic Regression, 4 classes. Dataset 400–600 reports (EN/HI/KN), LLM-drafted from templates, every sample hand-reviewed, 20% held out. Multilingual sentence embeddings = optional comparison only. | `17` |
| D3 | Vitals labels | Exactly `08` §2 (death_24h / intervention_6h / remaining ICU LOS → 4 classes; trend = new deterioration event within next 6 readings). | `17` ("sustained low BP/SpO2") |
| D4 | Vitals dataset | MIMIC-III Clinical Database Demo only. MIMIC-IV is future work (needs credentialing); do not plan around it. | `17` |
| D5 | Evaluation protocol (vitals + fusion) | **Nested patient-grouped CV is the primary result.** Outer `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped by `patient_id` for reporting (each patient tested exactly once); inner 3-fold grouped CV for tuning. **No separate held-out test set** (100 patients is too few). Report mean ± std over outer folds. "Test touched once" means: an outer test fold is never used for any tuning decision. All baselines use the identical outer folds. | `06` §2, `AGENTS.md` rule 4 wording |
| D6 | Vitals input features | Exactly `07` §4.1: T=6, F=12 (7 vitals/flags + 5 deltas) + mask. No missing-value indicator features in the main model (optional ablation only). | `18` |
| D7 | Fusion input | 19-d text vector from `07` §3 (includes 4 text-severity probabilities). Fusion is one joint network (`07` §4.2); trend probability is an output, not a fusion input. | `17` ("5 probs + trend prob") |
| D8 | RL baselines | Required: (a) SUMO fixed-time, (b) always-green-for-ambulance. Optional third if time: SUMO actuated. | `17` ("fixed-time + actuated") |
| D9 | Coordinator | Target = 6-row conflict table in `12` §05. The MVP `coordinator.py` implements only some of these; the rest are new work. Code wins on details of already-implemented conflicts. | `17` ("two conflict types") |
| D10 | Hospital scoring | Target formula `08` §7.2. First step when the MVP code is imported: document what `hospital_agent.py` currently does (weights, load factor, hard penalties), then reconcile, and update `08` §7.2 to match the final code. | — |
| D11 | Hospital/ambulance data | Reuse the MVP's hospital and ambulance-base lists (reported: 20 hospitals, 16 bases, verify on import). Extend columns to `08` §7.1. | `08` "15–30" range |
| D12 | Trend horizon | "Next 6 readings (≈6 h in MIMIC)". Never claim 10–15 minutes. | `02` §1 |
| D13 | XAI | SHAP TreeExplainer on RF/GB only; Integrated Gradients (Captum) on BiGRU. | `00` §7 |

## 3. Wording corrections (apply in `00` and in every slide/script)

- Hospitals **already receive advance alerts** from 108. The gap is: no live vitals trend,
  no AI risk score, no capability-aware hospital ranking or corridor clearing.
- Karnataka "**plans to introduce**" connected ambulances with real-time vitals transmission
  (as publicly reported, ETV Bharat, 25 May 2026). Not "explicitly stated".
- SIGMA: evaluated **in SUMO on four Kolkata intersection models** calibrated to municipal
  counts (arXiv preprint). Not "real intersections".
- DeepTriager AUC 0.956 / 0.9594: `[TODO-VERIFY in original paper]` before any slide.
- 2026 multimodal triage paper (arXiv 2607.16662): TabNet + self-attention fusion, not BERT
  `[TODO-VERIFY]`.
- Add DeepTriage-CN (Sci Rep 2026) and the prehospital AI review (Cureus 2025) to related work.

## 4. Engineering rules for the MVP import

- The ORS API key was hard-coded in two MVP files: **rotate the key**, read it from `.env`
  (`ORS_API_KEY`), never commit `.env`.
- The trained ETA model is not used by the ambulance agent in the MVP: wire it in or state it.
- Never commit MIMIC data or model binaries. Models go to the shared Drive folder;
  `experiments/log.csv` and `reports/` figures are committed.
