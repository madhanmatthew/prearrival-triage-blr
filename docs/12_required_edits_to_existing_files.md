# 12 — Required Edits to Existing Files (00–06)

**Precedence:** where this file conflicts with 00–06, this file wins. Apply the edits below
to the original files when convenient; until then agents must follow this list.

---

## 00_MASTER_OVERVIEW.md
- Add repo layout (see `09` §1), glossary (below), and priority tiers (`10` §1).
- Update section 8 file table to include `AGENTS.md`, `07`–`12`.
- Status table (§6): add file paths (see `AGENTS.md` §4).
- §7: the LangChain RAG row stays; add "Integrated Gradients" for the sequence model in the XAI row.

**Glossary:** NEWS2 (National Early Warning Score 2) · AVPU (Alert/Voice/Pain/Unresponsive) · GCS (Glasgow Coma Scale) · MIMIC (ICU database) · CHARTEVENTS (MIMIC table of charted readings) · ASR (speech-to-text) · WER (word error rate) · RAG (retrieval-augmented generation) · LCEL (LangChain Expression Language) · ORR (Outer Ring Road) · MDT (mobile data terminal in 108 ambulances) · SUMO (traffic simulator) · DQN/PPO (RL algorithms) · IG (Integrated Gradients) · ORS (OpenRouteService) · eRaktKosh (national blood-stock system).

## 01_reporting_agent.md
- §6.3 groundedness: replace lexical check with an NLI or LLM-judge entailment check per answer sentence; the function must return either `bool` (all sentences entailed) or a `float` ratio, and the docstring must match. Report % fully grounded answers.
- §6.2 imports: use `langchain_text_splitters` and `langchain_huggingface`; pin versions.
- §10 Open decisions: resolve as follows `[DECISION]` unless the team changes it — LLM via API with template fallback; Whisper `small` local with API fallback; languages en/hi/kn; interface = 19-dim vector (`07` §3).
- §9 interface: replaced by `07` §2.1 and §3.
- Add: similarity threshold for "not covered" response.

## 02_ambulance_vitals_agent.md
- §1 & §6 trend target: "worsens within next 6 readings (≈6 h in MIMIC)", not 10–15 minutes (`08` §2).
- §5 labels: use `08` §2 rules (death_24h / intervention_6h / remaining ICU LOS → 4 classes).
- §3 add: new confusion also scores 3 in NEWS2 (not available in MIMIC); 4-class banding is a project convention.
- §4 add: GCS→AVPU mapping (`08` §2); MIMIC records GCS, not AVPU.
- §6 add: sequence spec (T=6, F=12, masks, imputation, normalisation) and architecture (`07` §4).
- §10 interface: replaced by `07` §2.3 (`FusedRisk`).
- §8 checklist: CV must be `StratifiedGroupKFold` by `patient_id`.

## 03_traffic_police_agent.md
- §6: replace with the exact RL spec in `09` §4 (single joint agent, 16 actions, custom observation/reward, weights, budget).
- §5: add demand spec (`08` §6).
- §7: add second baseline "always-green-for-ambulance" and report ≥20 eval episodes per demand level.
- Add risk note: 5–6 km spacing → modest coupling; add optional compressed-spacing experiment.

## 04_hospital_agent.md
- §6: add explicit formula and RF target definition (`08` §7.2–7.3); note RF is a surrogate.
- Add hospitals table schema (`08` §7.1).
- §3 interface: input is `FusedRisk` (`07` §2.3); output `HospitalRecommendation` (`07` §2.4).

## 05_coordinator_fusion_evaluation.md
- §3: add fusion architecture, loss, modality dropout (`07` §4.2–4.4).
- §4: add mandatory caveat that text modality is synthetic and the generator rules (`08` §5); add sensitivity table.
- §5 add conflict table below.
- §6.4: SHAP applies to tree models; Integrated Gradients/attention for the sequence model.

**Coordinator conflict table** `[DECISION — verify against existing coordinator.py; code wins on details]`

| Conflict | Detection | Resolution (rule-based) |
|---|---|---|
| Route crosses police cordon | Route polyline intersects cordon polygon | Re-request ORS route avoiding polygon; if ETA increase > threshold, flag to dispatcher |
| Hospital bed shortfall | `beds_available` = 0 for required ward | Switch to next-ranked hospital; notify original |
| Two incidents want the same ambulance | Same `ambulance_id` bid | Priority bid = weighted (`risk_score`, ETA, age of incident); higher wins |
| Police units insufficient | Requested > available | Allocate by severity priority; escalate remainder |
| Two ambulances need preemption at same junction | Overlapping ETAs at junction | Preempt for higher `risk_score`; delay other |
| Blood shortage at top hospital | `blood_flag = short` with transfusion risk | Re-rank with blood weight; pick best trade-off |

## 06_ml_engineering_practices.md
- §3: replace `StratifiedKFold` with `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` and pass `groups=patient_id`. Never split by row.
- §2: add "split by patient" and same split reused for all baselines.
- §5: add that BiGRU CV must follow the same folds as tabular baselines.
- §9: models saved as `models/<name>_v<N>_<YYYYMMDD>`; save normalisation stats with each model.
