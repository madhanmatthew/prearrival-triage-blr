# AGENTS.md — Read This First (for any coding agent or new team member)

## 1. What this project is (5 lines)

A **pre-arrival multimodal AI triage and emergency coordination system for Bengaluru**.
A bystander reports an emergency by voice/text → the system asks guided follow-ups and gives
grounded first-aid tips → ambulance vitals stream is modelled by a sequence model that
predicts current severity and deterioration → RL controls traffic signals on a real ORR corridor
(simulated in SUMO) → a hospital agent picks the best receiving hospital (trauma capability,
beds, blood stock) → a rule-based coordinator resolves conflicts and issues the dispatch.
Novelty claim = **context + integration** (pre-arrival, multilingual, low-resource, Bengaluru),
NOT a new architecture. Never write "first-ever" or "novel architecture" anywhere.

## 2. Which file to read for which task

All docs live in `docs/`. Read only what the task needs.

| Task | Read |
|---|---|
| **Conflicts between docs, final decisions, team owners** | `docs/15_DECISIONS_FINAL.md` (**highest precedence**) |
| Context, novelty, status | `docs/00_MASTER_OVERVIEW.md` |
| Voice intake, dialogue, RAG first-aid | `docs/01_reporting_agent.md` + `docs/07_interfaces_and_schemas.md` |
| Vitals model, NEWS2, sequence model | `docs/02_ambulance_vitals_agent.md` + `docs/08_data_and_labels.md` |
| RL traffic, SUMO, police alerts | `docs/03_traffic_police_agent.md` + `docs/09_setup_and_runbook.md` §4 |
| Hospital scoring, blood stock | `docs/04_hospital_agent.md` + `docs/08_data_and_labels.md` §7 |
| Fusion, coordinator, ablation, benchmark | `docs/05_coordinator_fusion_evaluation.md` + `docs/07_interfaces_and_schemas.md` |
| Training methodology, CV, seeds, tuning | `docs/06_ml_engineering_practices.md` (CV protocol overridden by `15` D5) |
| Message formats, API, WebSocket, tensor shapes | `docs/07_interfaces_and_schemas.md` |
| Datasets, labels, synthetic data rules | `docs/08_data_and_labels.md` |
| Environment, run commands, repo layout, demo | `docs/09_setup_and_runbook.md` |
| Risks, fallbacks, ethics, priorities | `docs/10_risks_ethics_scope.md` |
| Corrections to 00–06 | `docs/12_required_edits_to_existing_files.md` |
| Presentation / viva only (**never a build spec**) | `docs/11`, `13`, `14`, `16`, `17`, `18` |

**Precedence when documents conflict:** `15` > `12` > `07`/`08`/`09` > component files (`01`–`05`) > `00`.

## 3. Hard rules — never break these

1. **Never derive `severity_label` from NEWS2.** Labels come from real outcomes (see `08`). Otherwise the NEWS2 benchmark is circular.
2. **Never report results from placeholder/synthetic-generator output** (`generate_placeholder_vitals_data`) as real results. Synthetic data must be labelled "synthetic" in every table and figure.
3. **Split by patient, never by row.** Use `StratifiedGroupKFold` grouped by `patient_id`. Rows/windows from one patient must never appear in both train and validation/test.
4. **Nested grouped CV is the evaluation protocol** (`15` D5): outer 5-fold for reporting, inner 3-fold for tuning. An outer test fold is never used for any tuning decision. All baselines share the same outer folds.
5. **Location comes from device GPS only**, never parsed from speech.
6. **First-aid guidance must come from retrieved corpus text only** (RAG). If the corpus does not cover it, say so and advise waiting for professionals. Log the source chunk for every guidance item. Never give medication or dosage advice.
7. **The dialogue flow is a bounded state machine**; the LLM only phrases questions. Do not let LangChain/LLM decide dialogue logic. LangChain is used only for the RAG pipeline.
8. **The coordinator stays rule-based.** Do not replace with RL. Document it honestly as rule-based.
9. **Do not build a standalone blood-bank app.** Blood stock is one scoring factor inside the hospital agent.
10. **Do not use SHAP TreeExplainer on the BiGRU.** Use Integrated Gradients (Captum). SHAP only on tree models.
11. **Keep ORS routing as-is.** No custom route-prediction model.
12. **Fix seeds everywhere** (numpy, random, sklearn, torch, SB3, SUMO `--seed`), pin library versions, save models with version tags.
13. **Do not invent facts, citations, numbers, MIMIC item IDs or dataset columns.** If something is unknown, mark it `[ASSUMPTION]` or `[TODO-VERIFY]` and continue.
14. **This is a research prototype, not a medical device.** No claim of clinical validity anywhere in UI, code comments, or reports.
15. **Never commit secrets, MIMIC data or model binaries.** Keys go in `.env` (gitignored).

## 4. Current status (update as work progresses)

This is a **fresh build** (`15` D14). The 6th-sem MVP is not imported; anything the older docs
call "existing" or "in MVP" must be built new here.

| Item | State | Path |
|---|---|---|
| Docs, decisions, agent rules | Done | `docs/`, `AGENTS.md`, `CLAUDE.md` |
| Pydantic schemas (`07` contract) | Done, with tests | `backend/schemas.py`, `tests/test_schemas.py` |
| Repo skeleton, Makefile stubs, env template, experiment log | Done | root |
| NEWS2 scoring (`02` §3 bands) | Done, with tests | `backend/ml/news2.py`, `tests/test_news2.py` |
| MIMIC windowing + outcome labels | Not started | `backend/data/mimic_windows.py` |
| Tabular baselines (majority, LR, RF, GB) + nested grouped CV | Not started | `backend/ml/baselines.py` |
| BiGRU two-head model | Not started | `backend/ml/seq_model.py` |
| Fusion model + ablation | Not started | `backend/ml/fusion_model.py` |
| XAI (SHAP trees, IG BiGRU) | Not started | `backend/ml/explain.py` |
| SUMO corridor (nodes, edges, routes, cfg) | Done, with tests (fixed-time baseline runs) | `sumo/`, `tests/test_sumo_corridor.py` |
| RL env wrapper, DQN train/eval, baselines | Not started | `backend/rl/` |
| Police alert sizing (rule-based) | Not started | `backend/agents/police_agent.py` |
| Whisper intake + WER eval | Not started | `backend/asr/` |
| Dialogue state machine (6 questions) | Not started | `backend/dialogue/` |
| Text severity classifier (TF-IDF char n-gram + LR) | Not started | `backend/ml/text_severity.py` |
| RAG pipeline + groundedness eval | Not started | `backend/rag/` |
| Hospital table + ambulance bases | Not started | `data/hospitals.csv`, `data/ambulance_bases.csv` |
| Hospital agent + blood factor + RF surrogate | Not started | `backend/agents/hospital_agent.py` |
| Coordinator (6 conflicts, rule-based) | Not started | `backend/agents/coordinator.py` |
| FastAPI app + WebSockets | Not started | `backend/main.py` |
| Frontend (Leaflet map, police + hospital views) | Not started | `frontend/` |
| Benchmark (8 + 2 scenarios, fair baseline) | Not started | `backend/benchmarks/benchmark.py` |
| Traffic/ETA ML models | Optional (`15` D16) | `backend/ml/eta_model.py` |

## 5. Working conventions

- Prefer small, testable modules. Every model module exposes `train()`, `evaluate()`, `predict()`, `save()`, `load()`.
- Every model is runnable by one Makefile target so teammates can train without editing code. Config via CLI flags, YAML, or `.env`.
- Every experiment appends a row to `experiments/log.csv` (see header). Every reported number in a paper/slide must be traceable to a row in that log.
- Trained models saved as `models/<name>_v<N>_<YYYYMMDD>.(joblib|pt|zip)` with normalisation stats; shared via Drive, not git.
- Schemas in `docs/07_interfaces_and_schemas.md` are the contract, implemented in `backend/schemas.py`. Change a schema → update `07` first, then `schemas.py`, then code.
- End every work session by updating `docs/progress/<owner>.md` and the status table above.

## 6. Where the honest limits are (say these, never hide them)

- Fusion training pairs are **synthetic** (MIMIC has no bystander text). Ablation result measures the pairing generator's assumptions as well as the model.
- MIMIC is ICU data (in-hospital), not pre-hospital. Domain gap = stated limitation.
- MIMIC-III Demo has ~100 patients: small; expect high CV variance.
- RL results are from simulation on an approximated corridor.
- Blood stock and bed occupancy are simulated unless real data is obtained.
