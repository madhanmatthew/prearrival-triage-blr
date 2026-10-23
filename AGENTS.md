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
| RL traffic, SUMO, police alerts | `docs/03_traffic_police_agent.md` + `docs/09_setup_and_runbook.md` §4 + **`docs/19_RL_DATA_CALIBRATION.md`** (network + demand + calibration, overrides `03` §5 / `08` §6) |
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
16. **Free-tier APIs only** (`15` §4). No service that needs a paid billing account or prepayment.

## 4. Current status (update as work progresses)

This is a **fresh build** (`15` D14). The 6th-sem MVP is not imported; anything the older docs
call "existing" or "in MVP" must be built new here.

| Item | State | Path |
|---|---|---|
| Docs, decisions, agent rules | Done | `docs/`, `AGENTS.md`, `CLAUDE.md` |
| Pydantic schemas (`07` contract) | Done, with tests | `backend/schemas.py`, `tests/test_schemas.py` |
| Repo skeleton, Makefile stubs, env template, experiment log | Done | root |
| NEWS2 scoring (`02` §3 bands) | Done, with tests | `backend/ml/news2.py`, `tests/test_news2.py` |
| MIMIC windowing + outcome labels | Done, with tests (MIMIC-IV Demo) | `backend/data/mimic_windows.py`, `tests/test_mimic_windows.py` |
| Tabular baselines (majority, NEWS2, LR, RF, GB) + nested grouped CV | Done, full run logged | `backend/ml/baselines.py`, `tests/test_baselines.py` |
| BiGRU two-head model | Done, with tests; full nested-CV run logged | `backend/ml/seq_model.py`, `tests/test_seq_model.py` |
| VitalDB 10-min deterioration (`15` D19, labels `08` §2b) | Done, with tests; full nested-CV run logged (203k windows, 5,643 subjects): HGB PR-AUC 0.512 / AUROC 0.833, BiGRU 0.497 / 0.826, NEWS2 0.242 / 0.660; deployed: isotonic-calibrated HGB `models/vitaldb_hgb_v1_20261021.joblib` (`--train-final`), `FusedRisk.deterioration_10min_prob` on every vitals reading, shown on dispatcher + hospital pages labelled intra-operative | `backend/data/vitaldb_windows.py`, `backend/ml/vitaldb_model.py`, `scripts/download_vitaldb.py` |
| Fusion model + 3-way ablation + agreement sensitivity (synthetic text) | Done, full run logged (fused < text_only at 0.65/0.8); re-run for gain-vs-text table | `backend/ml/fusion_model.py`, `data/text_feature_generator.yaml`, `tests/test_fusion_model.py` |
| XAI (SHAP trees, IG BiGRU) | Code done, with tests; IG run on saved BiGRU (figures in `reports/`); tree SHAP run done 2026-10-22 (MIMIC: severity top avpu/O2/temp, trend top sbp/hr; `reports/xai_shap_*`). VitalDB HGB: SHAP global summary + live top-3 reasons in `FusedRisk.deterioration_reasons` (temperature kept out of live text: surgical artefact) | `backend/ml/explain.py`, `backend/ml/vitaldb_explain.py`, `tests/test_explain.py`, `tests/test_vitaldb_explain.py` |
| SUMO schematic corridor (nodes, edges, routes, cfg) | Done, with tests (fixed-time baseline runs); kept as `19` fallback / compressed-spacing base | `sumo/`, `tests/test_sumo_corridor.py` |
| SUMO corridor from OSM (`19` §1, `15` D21) | Done, with tests: OSM extract -> net (signals kept), ORR eastbound route 17.3 km on the main carriageway (TomTom 16.8 km), 4 RL signals on the Silk Board -> Iblur stretch (3 junctions + 1 pedestrian crossing, D21 rev. 2026-10-22, `tls_map.json`); ORR signal-free Bellandur -> KR Puram | `scripts/build_osm_corridor.py`, `sumo/osm/`, `data/raw/osm/` |
| RL env on the OSM corridor | Done, with tests (2-phase states per junction, same 16-action / 24-d spec, rl/fixed/preempt); needs fitted demand `sumo/demand/<slot>.rou.xml` | `backend/rl/osm_env.py`, `tests/test_osm_env.py` |
| Calibration targets per slot (TomTom typical + live, published peaks) | Done: `data/traffic/calibration_targets.csv` | `scripts/build_calibration_targets.py` |
| Traffic data collection: TomTom typical (Routing, historic) + TomTom live (Flow) (`19` §2–3) | Collected: live flow 7.4 days (4 points); typical eastbound complete (672 rows); reverse direction missing (routes left the ORR) | `scripts/collect_tomtom_typical.py`, `scripts/collect_tomtom.py` |
| YOLO vehicle counts (`19` §4) | Optional since `15` D20 (published counts are the primary source) | `backend/vision/count_vehicles.py` |
| Published traffic figures (`15` D20) | Collected: BTP ORR study Oct 2025 (Iblur–Bellandur ~10,657 PCU peak vs 4,800), Silk Board Jul 2024 (17,816 veh 09:30–10:30, signal cycle 285 s + greens), KRDCL April 2016 TVC (mix/daily profile only) | `data/traffic/published_counts.csv`, `data/raw/traffic_src/` |
| Demand fitting + calibration table (`19` §5–6) | Done (fallback): demand-only fits failed at peak (v1 Silk Board entry queue; v2 needed 14x local demand + gridlock, `reports/rl_calibration_demand_v2_failed.md`); final = SPEED-ONLY calibration: moderate demand, per-leg per-slot lane limits (<= 60 km/h) + lower local demand on the signal-limited Silk Board leg, Silk Board on the reported 2024 program. 12/12 rows within 15 % of TomTom typical (worst -10.3 %; corridor low +0.3 %, medium +4.4 %, peak +3.5 %, 5 seeds). Stated limit: slot speeds imposed, not emergent from demand | `scripts/calibrate_osm_demand.py`, `scripts/calibrate_osm_speed.py`, `sumo/demand/`, `sumo/osm/silkboard_2024.tll.xml`, `reports/rl_calibration.md` |
| RL env wrapper + fixed-time / always-green baselines (schematic corridor) | Done, with tests; full baseline run logged | `backend/rl/env.py`, `backend/rl/baselines.py`, `tests/test_rl_env.py` |
| RL DQN train/eval (schematic corridor, 3 demand levels) | Done, 3 seeds logged; `reports/rl_comparison.csv` | `backend/rl/train.py` |
| RL on the OSM corridor (randomized demand, baselines, DQN) | Randomized demand done (slot, demand x U(0.8,1.2), 10 % incidents; mix jitter not applied); baselines done (6 paired episodes/slot: fixed 520/1463/1515 s, always-green 625/1081/1244 s); DQN 3 seeds (~60k steps each, deadline-limited): seed 2 beats both baselines at medium/peak (892 / 1007 s, -39 % / -34 % vs fixed), seeds 1 and 3 unstable (gridlock, ambulance not finishing in some slots) | `backend/rl/demand.py`, `backend/rl/osm_rl.py`, `reports/rl_osm_*` |
| Police alert sizing (rule-based) | Done, with tests; wired into dispatch (units required + PoliceAlert on `/ws/police`); junction posts with timing on the 4 ORR corridor junctions (corridor-only) + lifecycle (ack, GPS position updates, auto-release, close); constants are assumptions | `backend/agents/police_agent.py`, `tests/test_police_agent.py` |
| Whisper intake + WER eval + WER-vs-SNR noise eval (faster-whisper, local) | Clean WER run on 90 real clips (large-v3: en 3.6%, hi 30.8%, kn 79.2% WER / 23.7% CER); GPU session run on RTX 3050 (turbo ~1 s/clip, large-v3 1-4 s/clip), noise WER-vs-SNR logged (`15` D18); per-language fine-tunes deployed via `LanguageRouter` (vasista22 hi/kn medium, VAD off): dev hi 25.7 % / kn 47.8 % WER, held-out hi 24.3 % / kn 58.0 % | `backend/asr/`, `scripts/convert_hf_whisper.py`, `tests/test_asr.py`, `data/asr_test/` |
| Dialogue state machine (6 questions) | Done, with tests; code-mixed hi/kn cues + "breathing normally?" fix; ASR->cue robustness eval logged: dev en/hi/kn 1.00/0.83/0.83, HELD-OUT 1.00/0.60/0.60 (fine-tunes); hi/kn wording native-reviewed 2026-10-22 + native prompt audio | `backend/dialogue/`, `backend/asr/eval_triage.py`, `tests/test_dialogue.py`, `tests/test_eval_triage.py` |
| Text severity classifier (TF-IDF char n-gram + LR) | Done: 450 template drafts human-reviewed (1 relabel, 0 edits), fixed 20 % test split; test macro-F1 0.978 (en 1.00 / hi 1.00 / kn 0.93) = mostly template recovery (SIM); on the 30 real held-out recordings predictions are inconsistent (mean confidence 0.5) — stated limit; `models/text_severity_v1_20261023.joblib` loaded by the app | `backend/ml/text_severity.py`, `tests/test_text_severity.py`, `data/text_reports/README.md` |
| RAG pipeline + groundedness eval | v2 done, with tests (guard, dialogue routing, bge-small, keywords); IFRC 2020 corpus (18 topics); v1 + v2 test runs logged (v2: hit@4 0.96, answer 1.00, refusal 0.67) | `backend/rag/`, `scripts/extract_ifrc_corpus.py`, `data/first_aid_corpus/` |
| Hospital table + ambulance bases | Done: 17 hospitals (OSM coordinates + 6th-sem capability flags, `[TODO-VERIFY]` flag sources), 16 approximated ambulance bases; beds/ICU/blood SIMULATED; dispatch runs end to end | `data/hospitals.csv`, `data/ambulance_bases.csv`, `scripts/build_hospitals.py`, `data/raw/hospitals_src/` |
| Hospital agent + blood factor + RF surrogate | Done, with tests; runs on the real `data/hospitals.csv` (RF surrogate not trained: rule-only scores in the benchmark) | `backend/agents/hospital_agent.py`, `tests/test_hospital_agent.py` |
| Coordinator (6 conflicts, rule-based) | Done, with tests; wired into `POST /incident/{id}/dispatch` and the benchmark | `backend/agents/coordinator.py`, `tests/test_coordinator.py` |
| FastAPI app + WebSockets | Done, with tests; real app runs end to end with `data/hospitals.csv` + `data/ambulance_bases.csv` | `backend/main.py`, `backend/services.py`, `backend/routing.py`, `tests/test_api.py` |
| Frontend (Leaflet map, police + hospital views, bystander guidance text + browser speech `01` §10) | Done (static HTML/JS served at `/ui/`, no build step): reporter (lang, GPS, voice/text, tap answers, guidance steps + speech), dispatcher (map, risk, dispatch, route, demo vitals replay, simulated drive), police (live alerts, junction posts, ack/close), hospital (live risk); smoke-tested in browser with fixture services; native prompt recordings not yet wired | `frontend/`, `GET /incidents`, `GET /resources` |
| Benchmark (8 + 2 scenarios, fair baseline) | Done, with tests; run logged: appropriate hospital AI 9/10 vs naive 5/10, mean total time 27.5 vs 25.9 min (straight-line ETAs, SIMULATED beds/blood); `POST /benchmark/run` | `backend/benchmarks/benchmark.py`, `data/benchmark_scenarios.csv`, `reports/benchmark_*.csv` |
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
- VitalDB is intra-operative (anaesthetised surgical patients), not pre-hospital; its 10-min label is vital-sign-defined (`15` D19).
- MIMIC-IV Demo has 100 patients (`15` D4): small; expect high CV variance.
- RL results are from simulation on an approximated corridor.
- RL calibration targets (typical + live travel times) both come from TomTom; published counts (D20) are the only independent check (`19` §9).
- OSM corridor calibration is speed-only: per-slot leg speeds are imposed with variable speed signs, not produced by demand (demand-only fits failed at peak).
- Blood stock and bed occupancy are simulated unless real data is obtained.