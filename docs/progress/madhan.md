# Progress — Madhan (Component 3: Traffic RL + police) · also: all Claude Code work

## Role
- Owns: SUMO ORR corridor, RL env wrapper, DQN training, fixed-time + always-green baselines,
  police alert sizing (docs/03, docs/09 §4, docs/08 §6).
- Also writes all code for every component with Claude Code; teammates run training.
- Fresh build: no MVP code is reused (docs/15 D14). Build order is docs/15 §5.

## Done
- 2026-09-29: Repo created. Docs imported, `docs/15_DECISIONS_FINAL.md` written, AGENTS.md /
  CLAUDE.md set up, `docs/00` wording corrected, skeleton + Makefile stubs, `backend/schemas.py`
  (docs/07 contract) with 10 passing tests. Decided: fresh build, no MVP import.
- 2026-09-29: `backend/ml/news2.py`: 7 component scores (docs/02 §3), total, 4-class banding
  (docs/08 §3, marked as project convention), GCS→AVPU (docs/08 §2), `score_reading()` for
  `VitalsReading`. `tests/test_news2.py`: 136 tests (every band edge on both sides, every GCS
  value, class boundaries, hand-checked worked examples, missing/invalid inputs). 146/146 pass.
  Decisions (in the module docstring):
  - Missing (`None`/NaN) → component `None`; total `None` if any parameter is missing, never
    scored as 0. Gap-filling stays upstream (docs/07 §4.3).
  - Opt-in `missing_o2_as_air=True` scores a missing O2 flag as room air `[ASSUMPTION]`;
    the result carries `o2_assumed_air` so any table using it can say so.
  - Non-integer values use the band's upper edge (RR 24.5 → 3, temp 35.05 → 1).
  - Negative values, SpO2 > 100, unknown AVPU, GCS outside 3–15 raise `ValueError`.
  - Not implemented: SpO2 Scale 2, "new confusion", official single-parameter red escalation.
- 2026-09-30: SUMO corridor. `eclipse-sumo==1.27.1` installed via pip (pinned in requirements).
  `sumo/corridor.{nod,edg,typ}.xml` → `corridor.net.xml` (4 TL junctions at 0/6000/11000/16000 m,
  ORR 3+3 lanes, cross 2+2 lanes). `sumo/gen_routes.py` → `routes_{low,medium,peak}.rou.xml`
  (main 900/1200/1500, cross 300/450/600 veh/h, Poisson, 60/30/10 mix, 1 ambulance).
  `corridor.sumocfg` = fixed-time baseline (a). Make targets `sumo-build`, `sumo-routes`,
  `sumo-run DEMAND=… SEED=…`. `tests/test_sumo_corridor.py` (incl. a real SUMO run). 156/156 pass.
  - TL phases (netconvert default, 90 s cycle): 0 = ORR main green 41 s, 1 = yellow 4 s,
    2 = cross green 41 s, 3 = yellow 4 s. RL `main_green`/`cross_green` map to phases 0/2.
  - Smoke run, seed 42, ambulance depart 300 s, fixed-time: transit 910 / 1183 / 1001 s
    (low / medium / peak). Single seed, smoke check only — not a result.
  - [ASSUMPTION] Straight corridor; 300 m entry/cross stubs; ORR 60 km/h, cross 40 km/h;
    background traffic goes straight; ambulance speedFactor 1.3, impatience 1.0.
  - Ambulance has no SUMO bluelight device (it cannot run reds or force a rescue lane), so
    signal control is what changes its transit time. State this in the report.

- 2026-09-30: `backend/data/mimic_windows.py` + `make data-mimic` on MIMIC-IV Demo v2.2 (`15` D4):
  hourly bins, <=2-bin ffill, labels per docs/08 §2, `make_windows()` [N,6,7]+mask,
  `add_deltas()` -> [N,6,12] (call after train-fold median fill). Item IDs looked up in the
  demo's `icu/d_items` (in `ITEMS`). `tests/test_mimic_windows.py`; 180/180 tests pass.
  Output: 12,299 windows, 140 stays, 100 patients. Severity balance 0/1/2/3 =
  22.7 / 18.1 / 10.3 / 48.9 %. Trend: 6,403 eligible windows, 2.8 % positive.
  NEWS2 complete in 55.9 % of windows (GCS/temp gaps).
  - [ASSUMPTION]s are listed in the module docstring (NIBP preferred, O2 not charted = room
    air, intubated verbal = 1, vasopressor list, interval-overlap definition of intervention).
  - Class 3 is large because `intervention_6h` counts ongoing vasopressor/ventilation, not
    only new starts (MIMIC-IV splits infusions into rate-change rows). Team to confirm.
  - Trend positives 2.8 % (< 5 %): expect unstable trend metrics; report PR-AUC, not accuracy.

- 2026-09-30: `backend/ml/baselines.py` + `make train-baselines`: majority, NEWS2, LR, RF, GB;
  nested grouped CV (outer 5 / inner 3, seed 42; same patient folds for every model and
  both tasks); severity scored by macro-F1 + summed confusion matrix, trend by AUROC +
  PR-AUC (tuned on PR-AUC). Features = last reading's 12 features, train-fold median impute.
  NEWS2 scored from the same imputed inputs. Appends to `experiments/log.csv`, writes
  `reports/baselines_results.csv` + `baselines_confusion.csv`. `tests/test_baselines.py`;
  185/185 tests pass. `--quick` smoke run worked (~2 min); full-grid run not done yet.

- 2026-09-30: Baselines full run logged (`experiments/log.csv`). Severity macro-F1:
  majority 0.164, NEWS2 0.278, LR 0.352, RF 0.368, GB 0.368. Trend AUROC / PR-AUC:
  NEWS2 0.622 / 0.061, LR 0.750 / 0.158, RF 0.712 / 0.112, GB 0.694 / 0.094 (prevalence 0.028).
  ML > NEWS2 clearly; ML models tied within fold std.
- 2026-09-30: `backend/ml/seq_model.py` + `make train-seq`: BiGRU two-head (docs/07 §4.2
  vitals-only), loss CE + lambda*BCE (docs/07 §4.4), lambda in {0.25, 0.5, 1.0} and epoch
  count tuned by inner grouped 3-fold CV, same outer folds as baselines, train-fold-only
  preprocessing. `--save-model` fits on all data for XAI/demo. [ASSUMPTION] trend BCE
  pos_weight = neg/pos. `tests/test_seq_model.py`; 192/192 pass. `--quick` smoke worked.

- 2026-09-30: **Decision change (`15` D17, `19` §2):** Google Routes dropped. Google Cloud India
  billing required a ₹1,000 prepayment; project is free-tier only. Typical travel times now
  from TomTom Routing (`historicTrafficTravelTimeInSeconds`, future `departAt`, 24h×7),
  `scripts/collect_tomtom_typical.py` → `tomtom_typical.csv`, Make target `collect-typical`.
  Limit recorded: typical + live are same provider; YOLO counts are the independent check.
  `TOMTOM_API_KEY` placed in `.env`. No Google key needed.

- 2026-09-30: BiGRU full nested-CV run. Severity macro-F1 0.385 ± 0.037 vs best tabular 0.368
  (RF/GB); trend AUROC 0.708 ± 0.105 (LR 0.750), PR-AUC 0.162 ± 0.074 (LR 0.158); NEWS2
  0.278 / 0.622 / 0.061. Differences to tabular are within fold spread: a tie. Tuning runs
  early-stopped at 3-6 epochs. Claim: "BiGRU matches but does not significantly outperform
  tabular baselines on the 100-patient MIMIC-IV Demo; learned models clearly beat NEWS2."
  Full MIMIC-IV = future work; do not tune further on 100 patients.
- 2026-09-30: RL env + baselines (schematic corridor). `backend/rl/env.py` (`CorridorEnv`,
  Gymnasium over traci, custom, not sumo-rl): 16 actions, 24-d obs, reward per docs/09 §4,
  5 s steps, min_green 10, yellow 3. `control=` "rl" | "fixed" (SUMO program untouched =
  baseline a) | "preempt" (baseline b: default 41/4 cycle, ORR green while ambulance <= 1 km
  upstream). Ambulance added via traci at a random second in [200, 600].
  `backend/rl/baselines.py` + `make rl-baseline [EPISODES=20 SEED=42]` ->
  `reports/rl_baselines_{episodes,summary}.csv` + log rows; all policies share seeds.
  `tests/test_rl_env.py` 5 pass. Smoke (2 episodes, medium, NOT a result): fixed 996 s vs
  always-green 914 s transit; general wait ~1.1 s both. ~25 s per episode.
  [ASSUMPTION]s in the env.py docstring (queue cap 40, depart range, 2400 s cap).
- 2026-09-30: `scripts/collect_tomtom.py` (Flow polling, 30 min, `--once` for a test round),
  `scripts/traffic_common.py`, config `scripts/traffic_points.json`,
  `tests/test_collect_traffic.py`. No real API call made yet.
  [TODO-VERIFY] junction coordinates in `traffic_points.json` are approximate; collectors
  refuse to run until `"verified": true`.
- 2026-09-30: `scripts/collect_tomtom_typical.py` (TomTom Routing `calculateRoute`, historic
  typical times, 1,344 slots, resumable, `--max-requests 600` per run, `--dry-run`) ->
  `data/raw/traffic/tomtom_typical.csv`; `make collect-typical`. `collect_google.py` deleted.
  Shared `next_departure` / `segment_jobs` live in `scripts/traffic_common.py`. 9 offline tests
  pass. [TODO-VERIFY] request/response field names are from memory of TomTom's docs: smoke-test
  with `--max-requests 2` and check `length_m` (~6/5/5 km) before the full run.

- 2026-09-30: First `collect_tomtom_typical` batches. APIs smoke-tested OK (Flow 4/4 rows;
  Routing Silk Board->Bellandur 6,308 m). Two runs overlapped and duplicated rows: deduped, and
  a lock file now blocks concurrent runs. Reverse (east->west) routes came back 8.9 km and
  9.6 km vs ~6 / ~5 km nominal, i.e. TomTom left the ORR: reverse rows deleted, forward kept
  (336 rows). Added `via_points_reverse` and a length guard (`expected_length_m`, tol 30 %,
  [ASSUMPTION] nominal lengths from docs/19; aborts after 5 rejects in a row). Flow poller running
  since 2026-09-30 (check snapped coords: bellandur point snapped to a local FRC4 road).
  TODO: add reverse via points, then re-collect reverse slots on later days.

- 2026-09-30: `backend/rl/train.py` (SB3 DQN, MlpPolicy 64x64, eval callback every 1000 steps,
  best-by-eval checkpoint) + `make rl-train SEED=1 STEPS=100000` / `make rl-eval MODEL=...`.
  Trains on randomized demand (episode samples low/medium/peak + random ambulance time), then
  evaluates deterministically on each level with the baselines' episode seeds (paired).
  Writes `reports/rl_dqn_{episodes,summary}_s<seed>.csv` + log rows (model `dqn_s<seed>`).
  Smoke test added to `tests/test_rl_env.py`. Not run for real yet. [ASSUMPTION] hyper-parameters
  untuned. Docs/19 §7 jitter (demand x U(0.8,1.2), mix, incidents) waits for the OSM network.

- 2026-09-30: RL work paused by choice. Dialogue state machine (`backend/dialogue/state_machine.py`,
  `lexicon.py`, `tests/test_dialogue.py`, 37 pass): 6 bounded questions, one per turn, fixed
  order, skips questions the initial report clearly answered (plain "bleeding" still asks the
  severity follow-up), outputs the 19-d vector (docs/07 §3) and a validated `IncidentReport`.
  LLM is only an optional `phraser` hook; templates are the fallback. GPS passed in, never parsed.
  [ASSUMPTION]s in the module docstring (order, bleeding level from raw wording, unknown count = 1,
  unconscious -> can't speak). [TODO-VERIFY] hi/kn wording + lexicons need native review (Sankalp).
  Text-severity slots 15-18 are a uniform placeholder until the text classifier exists.

- 2026-09-30: Whisper intake (`docs/01` §10 decided: **faster-whisper, local**, CPU int8, no API).
  `backend/asr/transcribe.py` (`Transcriber`, lazy model load into `models/whisper/`, config from
  `WHISPER_MODEL`/`WHISPER_MODE`), `intake.py` (audio -> `DialogueManager`), `eval_wer.py`
  (corpus WER + CER per language, own edit-distance, NFC/punctuation normalisation [ASSUMPTION]),
  `make eval-asr`, `data/asr_test/README.md` (format). `tests/test_asr.py` 10 pass with a fake model;
  no real audio or model download done yet. `faster-whisper==1.2.1` pinned in requirements.
  Language is forced to the true language during WER so it measures recognition only.
  TODO (Sankalp): record >= 30 utterances/language + `reference.csv`; run `make eval-asr`.

- 2026-09-30: Hospital agent (`backend/agents/hospital_agent.py`, 10 tests pass, fixture hospitals only).
  docs/08 §7.2 formula, blood weight redistributes to proximity when no transfusion risk, unknown
  group -> O- proxy, RF surrogate on simulated episodes (`simulate_episodes`, reward per §7.3),
  `use_rf=False` for the rule-only ablation. ETA is supplied by caller (ORS). [ASSUMPTION]
  trauma_level 0-3 higher=better, required level = severity_class; injury->specialty map
  [TODO-VERIFY] against real `data/hospitals.csv` vocabulary. No real hospital data built yet (D11).

- 2026-09-30: Fusion (`backend/ml/fusion_model.py`, `make train-fusion`, 10 tests pass). One `FusionNet`
  with `variant` = vitals_only | text_only | fused (docs/07 §4.2, modality dropout 0.2 text / 0.1 vitals,
  never both). Synthetic text from `data/text_feature_generator.yaml` (label-only, all probabilities
  [ASSUMPTION] fixed before results). `evaluate()` runs the 3-way ablation at agreement 0.5/0.65/0.8 on the
  shared outer folds and writes `reports/fusion_sensitivity.csv` (paired per-fold gain mean/std), confusion CSVs, and log rows
  tagged text=SYNTHETIC. vitals_only is the same net + seed as the BiGRU (test-verified), so it
  must reproduce 0.385; if not, something is wrong. Full run not done yet. Mandatory caveat is printed and must go on every table.

- 2026-09-30: Text severity classifier (`backend/ml/text_severity.py`, `make train-text`, 10 tests pass on a toy
  fixture). TF-IDF char_wb 2-5 + LR (balanced), C tuned by 5-fold stratified CV on train split only; fixed 20%
  held-out split stored in the CSV (`--assign-split`, stratified language x label, seed 42, never reassigned).
  Reports macro-F1 overall + per language, confusion CSV, log row tagged SIM. `predict_proba` -> dialogue slots
  15-18 via `compose_text(transcript, answers)`. [ASSUMPTION] trained on single reports, answers appended only
  at inference (input gap). Dataset format/rubric in `data/text_reports/README.md`; no data yet.

- 2026-09-30: XAI (`backend/ml/explain.py`, `make explain`, 8 tests pass). SHAP TreeExplainer on RF/GB
  pipelines (imputed 12 tabular features); multiclass sklearn GB is unsupported by shap, so per-tree TreeSHAP
  is summed (exact, raw log-odds space, test-verified vs decision_function). Integrated Gradients (Captum)
  on the BiGRU per head, baseline = zero z-scored input [ASSUMPTION], padded steps get 0; convergence delta
  saved. `shap==0.52.0` installed + pinned. IG run done on `bigru_v1_20260930.pt` (2000 windows, 64 steps,
  |delta| mean 0.005 sev / 0.009 trend): severity top = avpu_bin, temp, supplemental_o2, sbp, hr; t0
  dominates. Outputs `reports/xai_ig_*`, `reports/figures/xai_ig_*`. Tree SHAP not run yet (needs RF/GB
  grid search on all windows). Explanations of fit-on-all models, not results.

- 2026-10-05: Results in. **Fusion** full run (log rows 2026-10-01, commit 953d88e): vitals_only 0.385 (reproduces
  BiGRU), text_only 0.472/0.600/0.753, fused 0.473/0.575/0.721 at agreement 0.5/0.65/0.8. **Fused does NOT beat
  text_only** (from console per-fold F1: gain vs text approx +0.002, -0.024, -0.032; 3/5, 2/5, 2/5 folds better).
  `gain_table` now reports gain vs vitals AND vs text (was vs vitals only, which overstated fusion);
  `fusion_fold_f1.csv` added for traceability. Re-run `make train-fusion` for the traceable table (seeded,
  should reproduce). Do NOT change architecture/generator to make fusion win (outer folds already seen).
  **RL** DQN seeds 1-3 done: ambulance transit 809/830/872 s (low/med/peak) vs always-green 843/871/932 vs
  fixed 947/962/1018; general wait 0.10-0.14 s vs ~1.1 s; seed spread small (803-814 s at low).
  Found: episodes end at ambulance arrival, so raw throughput_veh depends on episode length -> added
  `throughput_veh_per_h` (DQN 2516/3717/4968 veh/h, between always-green and fixed). Summaries regenerated
  from episode CSVs; combined table `reports/rl_comparison.csv`. veh/h still mildly favours longer episodes
  (more steady-state); a fixed-horizon eval would remove that. Schematic corridor, not calibrated (docs/19).

- 2026-10-05: Draft text reports generated: `scripts/draft_text_reports.py` -> `data/text_reports/reports.csv`
  (450 rows, 150/lang, 20 type x label cells, hi/kn 40 romanised each, all reviewed=0, seed 42; test added,
  11 pass). TODO (you): review every row in Excel, set reviewed=1, then `--assign-split` and `make train-text`.
  Templated -> high score expected; add own-words rows. Do not re-run the generator with --force after reviewing.

- 2026-10-05: RAG first-aid guidance (`backend/rag/pipeline.py`, `eval_rag.py`, `make build-rag` / `eval-rag`,
  `tests/test_rag.py` 14 pass; 310 total). LangChain splitter + MiniLM (normalised) + FAISS inner product;
  top-4 with `min_similarity` 0.35 [ASSUMPTION] -> "not covered" below it (docs/07 §6). Modes: extractive
  (default, no LLM, sentences copied verbatim) or `LLM_PROVIDER=openai_compat` (free tier, LCEL chain;
  failure -> "Wait for professional help; call 108."). Medication/dosage keyword filter (rule 6).
  Groundedness per sentence (docs/12 §01): verbatim match, else NLI `cross-encoder/nli-deberta-v3-small`
  P(entail) >= 0.5 [ASSUMPTION]; float ratio + fully_grounded; ungrounded sentences dropped at serve time,
  every kept sentence carries chunk ids. `queries_from_report()` maps dialogue answers -> ordered queries.
  Eval: hit@k, answer/refusal rate, % fully grounded; threshold swept on dev split only. Real embedder +
  NLI smoke-checked (entail 0.995 / contradiction 0.0). English guidance only. Deps pinned.
  No corpus or query set yet (must come from real public guidelines; formats in the two READMEs).

- 2026-10-06: First-aid corpus. Downloaded IFRC International first aid, resuscitation, and education guidelines
  2020 (official PDF, non-commercial reuse with acknowledgement; `PROVENANCE.md`). WHO-ICRC BEC rejected
  (clinician-level, drug steps). `scripts/extract_ifrc_corpus.py` -> 18 topic files in `data/first_aid_corpus/`
  (Key action / Early recognition / First aid steps / NOTE, verbatim; medication sentences, dispatcher/trainer
  text and repeated layout blocks dropped). `make build-rag`: 136 chunks. Retrieval sanity check OK (bleeding,
  burns, fracture, chest pain -> right file; snake bite 0.25 < 0.35). "what tablet should I give" scores 0.39
  (fractures): threshold must be tuned on dev; medication filter still blocks drug text. Hand review pending.

- 2026-10-06: RAG threshold sweep on dev rows (15 in-scope, 6 out-of-scope, `data/rag_eval/queries.csv`, 61 queries
  drafted by Claude, user-reviewed): 0.30 -> answer 1.00 / refuse 0.67; 0.35 -> 0.93 / 0.83; **0.40 -> 0.87 / 1.00
  (balanced 0.93, best)**; 0.45 -> 0.53 / 1.00. Chosen `MIN_SIMILARITY = 0.40` (pipeline default). Only 6 dev
  out-of-scope rows, so the estimate is coarse. Test split not run yet.
- 2026-10-06: RAG test split, scored once (log row, extractive, min_sim 0.40, 30 in / 10 out): hit@4 0.90, answer
  rate 0.93, refusal 0.60, % fully grounded 1.00 (extractive = verbatim by construction; NLI not exercised).
  Failures: "not breathing and not responding" -> breathing_difficulty + recovery_position (CPR missed: safety-
  relevant); glass in thigh -> chest_abdomen; amputation + seizure refused (0.37/0.38); answered out-of-scope:
  diabetic, fever, painkiller injection, ambulance ETA. These test queries are now seen: any v2 fix needs a
  fresh test set.
- 2026-10-06: RAG v2 (after v1 test failures). (1) Query guard `query_refusal_reason`: medication, non-medical,
  and IFRC topics excluded from the corpus (list from the IFRC ToC) -> "not covered" before retrieval.
  (2) Dialogue routing: `queries_from_report` -> (problem, query, topic_file); unresponsive + breathing
  difficulty/unknown -> CPR (corpus rule "if in doubt, assume breathing is not normal"), unresponsive + no
  difficulty -> recovery position; routed retrieval filters to that file, no threshold; `guide_report()`.
  (3) Chunks embedded with topic name + header `keywords` (hand-written, extractor KEYWORDS); embed model
  configurable, index records it. Dev comparison (61 queries = v1 dev + v1 test, now seen): hit@4 MiniLM 0.956,
  bge-small-en-v1.5 1.000 -> bge chosen. Threshold 0.55: dev in-scope min top score 0.559, only one unguarded
  out-of-scope dev query (0.557), so no data-driven cut exists; chose safety-first (no in-scope dev refusal,
  e.g. "collapsed, no breathing" 0.600). Guard catches 15/16 dev out-of-scope. Fresh v2 test set: 28 in /
  12 out (7 without any guard keyword), written before any v2 run, not yet scored. 27 RAG tests, 329 total.
  Limits: guard + keywords are hand rules written after seeing v1 failures; test queries also by Claude.
- 2026-10-06: RAG v2 test split scored once (log row, bge-small, min_sim 0.55, 28 in / 12 out): hit@4 0.964,
  answer rate 1.00, refusal 0.67, % fully grounded 1.00 (extractive), guard refusals of in-scope 0. Guard caught
  all 4 medication/excluded-topic queries; threshold refused floor cleaner + all 3 non-medical. Failures:
  uncovered medical complaints answered with nearby topics (headache -> stroke, hiccups -> breathing, earache ->
  cpr_child, grief -> shock; scores 0.57-0.64 > in-scope minimum), and free-text "dropped down, not breathing" ->
  breathing_difficulty (CPR missed in free text; dialogue routing sends this case to CPR). Conclusion: cosine
  threshold cannot separate "uncovered medical" from "covered"; use dialogue-routed guidance in the API and
  free-text retrieval only as fallback. No further tuning on this test set.
- 2026-10-06: Coordinator (`backend/agents/coordinator.py`, `tests/test_coordinator.py` 21 pass; 350 total). Rule-based
  (rule 8), all six docs/12 §05 conflicts: cordon_on_route (own segment/polygon geometry, re-route via injected
  `Router`, dispatcher flag if detour > 5 min), hospital_bed_shortfall (next-ranked with beds, notify original),
  blood_shortage (non-short hospital within +10 min, else keep + request transfer), ambulance_contention (greedy by
  bid 0.6 risk + 0.25 ETA + 0.15 age; loser takes next ambulance or is queued), police_units_short (severity
  priority, partial allocation, remainder escalated), junction_preemption (ETAs within 30 s -> higher risk keeps
  preemption). `coordinate()` -> DispatchDecisions with conflicts_resolved + plain-language explanation and a
  `needs_dispatcher` list. All thresholds/weights [ASSUMPTION], fixed before any benchmark. Deterministic
  (tie-breaks by report time, incident_id). Junction rule ignores approach direction (SignalState has none).
- 2026-10-06: FastAPI app (`backend/main.py`, `services.py`, `routing.py`; `tests/test_api.py` 11 pass; 361 total). All
  docs/07 §5 endpoints + /health; WS /ws/police, /ws/map, /ws/hospital/{id}; in-memory store. `load_services()` loads
  what exists and lists gaps in /health (docs/07 §6 degraded modes). Fusion `predict(..., vitals=False)` added for
  text-only risk before vitals (m_vit=0). Dispatch runs the coordinator over all pending incidents (contention),
  ETAs via ORS REST (`OrsRouter`, [TODO-VERIFY] fields) else straight line 25 km/h x 1.4 [ASSUMPTION].
  Real-model smoke run OK: dialogue -> text-only risk -> routed guidance -> vitals (text+vitals, NEWS2 14) ->
  dispatch 503 (no hospitals/ambulance files). Police units requested = 0 until police agent exists.
  Safety note: Q_BREATHING asks "difficulty breathing?"; for an unresponsive person "no" is ambiguous (could
  mean not breathing at all) but routes to recovery position. Needs a decision (see chat).
## Next (in order)
0b. Hand-review `data/first_aid_corpus/*_ifrc.md`; write `data/rag_eval/queries.csv`, then `make build-rag`, `eval-rag --sweep`, `eval-rag --min-sim <t>`.
0a. `make train-fusion` (full run: 7 variants x nested CV, CPU, expect a long run), commit reports + log rows.
0. Run `make rl-train SEED=1` (100k steps, roughly 1-2 h on this env), then seeds 2 and 3.
1. Verify coordinates, smoke-test the TomTom key, start `collect-traffic` (>= 7 days).
2. Smoke-test `collect-typical ARGS="--max-requests 2"`, then run it on 3 separate days.
3. Run `make rl-baseline` (120 episodes, ~50 min) and commit reports + log rows.
4. `python -m backend.ml.seq_model --save-model` (Chetan's XAI/demo model).
5. DQN: `backend/rl/train.py`, after the OSM network / randomized demand (`19` §1, §7).

## RL data plan (docs/19, decision D17) — start collection early, it needs days
- [x] TomTom key (Freemium) → `.env` as `TOMTOM_API_KEY` (Google dropped, see 2026-09-30 entry)
- [ ] Smoke-test key: Flow Segment Data + Routing `calculateRoute` (both return 200)
- [ ] `scripts/collect_tomtom.py` and start it running ≥ 7 days in background
- [ ] `scripts/collect_tomtom_typical.py` (24h × 7 × 8 = 1,344 calls, resumable, spread ≥ 2 days)
- [ ] Record 10–15 min video at 2–3 ORR junction approaches (peak + off-peak)
- [ ] OSM network + `tls_map.json` → YOLO counts → routeSampler → validation table ≤ 15%
- [ ] Then RL env wrapper with randomized demand, baselines, DQN seeds

- 2026-10-06: ASR noise robustness: `backend/asr/noise.py` (SNR mixing), `eval_noise.py` (WER vs SNR, VAD on/off, optional denoise), `make eval-asr-noise`, tests pass (16). TODO (Sankalp): street-noise wavs in `data/asr_test/noise/`, ~10 noisy clips/language.

- 2026-10-07: First real WER run (90 team clips, 30/lang, read from scripts; logged). Added
  `medium`/`large-v3` + repetition guard (`repetition_penalty`, `no_repeat_ngram_size`; env
  `WHISPER_REPETITION_PENALTY`/`WHISPER_NO_REPEAT_NGRAM`; 1.1 set a priori, not tuned). Pinned
  `av>=14,<17` (av 19 breaks faster-whisper decode). Clean WER / CER:
  small no-guard en .036/.019 hi .517/.316 kn 1.059/1.306 (15 kn loops, 19/30 kn in Devanagari);
  small rp1.1 en .049/.030 hi .501/.231 kn 1.118/1.180; medium rp1.1 en .041/.018 hi .405/.201
  kn 1.031/1.331 (23 kn loops); large-v3 rp1.1 en .036/.017 hi .308/.161 kn .792/.237 (no loops,
  all Kannada script). Remaining kn WER = colloquial/loanword spelling (CER is the fairer metric).
  CPU int8 time for 90 clips: small 14 min, medium 55 min, large-v3 85 min (~57 s per ~9 s clip:
  too slow for live intake on CPU). Noise eval not run yet (large-v3 x 900 runs ~= 14 h CPU).
- 2026-10-07: large-v3-turbo rp1.1: en .036/.018 hi .332/.168 kn .842/.262 (no loops, all
  Kannada script); 65 min for 90 clips on CPU (~44 s/clip, incl. model download) vs large-v3
  85 min: only ~1.3x faster on CPU (encoder dominates). Live intake needs a GPU either way.

- 2026-10-07: GPU support for ASR: `WHISPER_DEVICE`/`WHISPER_COMPUTE` (env, empty = default),
  Windows DLL path for pip CUDA wheels (`requirements-gpu.txt`), `backend/asr/latency.py`
  (median RTF per language, logged), `scripts/asr_gpu_day.py` (latency + clean WER + noise WER,
  resumable). Smoke-tested on CPU with `tiny` (not logged); GPU path untested (no NVIDIA here).
  Decision D18 recorded (en+hi+kn live on local GPU; English-live fallback).
  NEXT: dialogue keyword matching per language (hi/kn spellings from the Whisper output) +
  tap buttons; ASR->triage robustness eval; recorded prompts list for native speakers.

- 2026-10-08: Dialogue robust to real hi/kn ASR output + breathing-question safety fix.
  (1) Q_BREATHING now asks "breathing normally?" (IFRC check) in en/hi/kn; reply inverted so stored
  "yes" still = breathing problem (schema/RAG/fusion unchanged); unresponsive + "no"/"don't know" -> CPR.
  (2) Report cues are regexes on `fold()`ed text (nukta, chandrabindu, ZWNJ dropped): Devanagari/Kannada
  loanwords (एक्सीडेंट, ब्लीडिंग, अनकॉन्शियस, ಆಕ್ಸಿಡೆಂಟ್, ಮೂರ್ಛೆ), kn negation suffix (ಉಸಿರಾಡ್ತಿಲ್ಲ -> no),
  explicit accident words before "fall", bare vehicle words after it. Removed over-broad cues (bare
  ಉಸಿರಾಟ / "सांस लेने में" set breathing problem even for "breathing slowly"). Tap buttons = English
  yes/no/don't know in every language. (3) `backend/asr/eval_triage.py`: cue extraction from script vs
  Whisper transcript (large-v3) and vs the English script of the same scenario. All-field agreement
  hyp vs en-script: en 1.00, hi 0.83, kn 0.83 (ASR-only: hi 0.87, kn 0.83; lexicon-only: hi 0.97,
  kn 1.00); logged. **DEV figure**: variants written from these 90 clips; a clean number needs new
  recordings. Remaining misses = ASR garbles (चस्पेन, तक्रा, ಕುಷಿದು). 400 tests pass.
  NEXT: native review of hi/kn wording (incl. new Q_BREATHING); fresh held-out clips for eval_triage;
  recorded-prompt list for native speakers.

- 2026-10-17: GPU session run on Harish's RTX 3050 (4 GB), `scripts/asr_gpu_day.py`, rp1.1, 90 clips.
  Latency (median s/clip, RTF): turbo fp16 en .9 s/.09, hi 1.3 s/.14, kn 1.4 s/.17; large-v3 int8_fp16
  en 1.3 s/.14, hi 3.0 s/.30, kn 4.1 s/.51. Live intake on local GPU is feasible (turbo). Clean WER/CER:
  turbo en .036/.017 hi .332/.165 kn .823/.246; large-v3 en .036/.016 hi .305/.159 kn .804/.239.
  Noise WER (turbo, VAD off): en .036 clean -> .039 @10 dB -> .043 @5 dB; hi .340 -> .419 -> .470;
  kn .848 -> .857 -> .988. Kannada WER stays high (spelling variants; CER is the fairer metric).
  Full rows in experiments/log.csv, reports/asr_*_cuda*.csv. Not yet done: native review, fresh clips.

- 2026-10-21: Police alert sizing (`backend/agents/police_agent.py`, rule-based, constants [ASSUMPTION]):
  units from severity + injured count + road accident, perimeter from severity/injury type, caps 6 units
  / 400 m, medical-only = escort without cordon. Wired in `backend/main.py`: units requested feed the
  coordinator; after dispatch a `PoliceAlert` (allocated units, route polyline) goes to `/ws/police`.
  11 new tests; full suite 410 passed. Also: `scripts/download_vitaldb.py` (open VitalDB, 1-min vitals)
  for the short-window vitals work; MIMIC-IV-ED credentialing submitted 2026-10-20.

- 2026-10-21: Police alerts v2 (rule-based, no ML): `PoliceAlert.junction_posts` (docs/07 §2.6 updated
  first) = ORR corridor junctions within 150 m of the scene->hospital route (coords from
  scripts/traffic_points.json), in travel order, ambulance ETA = depart (now + ETA to scene + 10 min
  on scene) + route duration x distance share, officer in place 2 min earlier, `manual_clearing` where
  SignalState lost preemption (not yet fed by the API: no live SignalState there). `AlertBoard`
  lifecycle: ack alert/post, `POST /incident/{id}/position` re-times posts and releases passed ones,
  auto-close when all passed, `POST /police/{id}/close`, `GET /police/{id}`. Posts only when units are
  allocated. All constants [ASSUMPTION]. 422 tests pass.
  NEXT (optional): traffic-aware post priority once TomTom Flow is collected; SignalState -> manual
  clearing in the live API; SUMO delay model (SIM labels) only if time allows.

- 2026-10-21: Frontend v1 in `frontend/` (plain HTML/JS + Leaflet 1.9.4 from unpkg, served by FastAPI at
  `/ui/`, `/` redirects there). Pages: report (en/hi/kn, device GPS + labelled demo override, hold-to-speak
  voice upload, yes/no/don't-know taps, guidance one step at a time with chunk ids + browser speech,
  English only for guidance), dispatch (incident list + map, fused risk cards, dispatch, route polyline,
  vitals replay of hand-written DEMO values in `frontend/demo_vitals.json` (not patient data), simulated
  drive posting GPS so police posts release), police (`/ws/police`, posts, ack/close, perimeter), hospital
  (`/ws/hospital/{id}`, risk history). New endpoints `GET /incidents`, `GET /resources` (docs/07 §5).
  Browser smoke test with fixture services: all pages work, no console errors. With the uniform
  placeholder text model severity is always 0, so police posts only show with the real fusion model.
  NEXT: native-speaker prompt recordings (D18), SUMO replay panel, replace demo vitals with a held-out
  real window.

- 2026-10-21: `data/hospitals.csv` (17) + `data/ambulance_bases.csv` (16) from `scripts/build_hospitals.py`.
  Coordinates from the OSM hospital extract (ODbL), joined by hand-checked OSM id; capability flags
  from the 6th-sem curated list (sources [TODO-VERIFY]); dropped Rajarajeswari MC, PES IMSR, Narayana
  Health City (not in the extract). trauma_level = polytrauma + neurosurgery + (emergency or accident
  victims). Beds/ICU/blood SIMULATED (seed 42), beds_total empty. Ambulance bases = approximated zone
  points from the 6th-sem project, not official 108 bases. Dispatch now runs end to end (straight-line
  ETAs without an ORS key). No burns/cardiology flags in the source, so specialty match is 0 for
  burn/cardiac incidents (stated limit).

- 2026-10-21: VitalDB short-horizon model (D19). Download: 5,877 cases (scripts/download_vitaldb.py,
  1-min medians, ~26 MB). Windows (docs/08 §2b): 203,294 windows, 5,875 cases, 5,643 subjects,
  positive rate 0.138. Nested grouped CV 5x3 by subjectid, PR-AUC / AUROC (mean +/- std over outer folds):
  majority .138/.500, partial NEWS2 .242/.660, logreg .346/.749, HGB .512+/-.008/.833, BiGRU
  .497+/-.006/.826 (CPU, 24 min). Logged in experiments/log.csv (rows carry commit d46a692, the code
  was committed right after). BiGRU usually stopped at 14-15 of max 15 epochs: a longer cap may help
  (not tuned on outer folds). Trees beat the BiGRU here; report as is.
  NEXT: train final model (`--train-final`) and add P(deterioration in 10 min) to the vitals endpoint +
  UI; SHAP on HGB; joint model when MIMIC-IV-ED access arrives.

- 2026-10-21: System benchmark (D15). `data/benchmark_scenarios.csv`: 8 Bengaluru incidents (4 at the
  verified ORR junctions, 4 approximate [TODO-VERIFY on map]) + 2 blood cases (scene 40% of the way between
  two hospitals; the nearer one has no O+, set via `hospital_overrides`). Fair baseline: same straight-line
  ETA model, 2 min dispatch overhead, 10 min on scene, same ambulances, same scenario severity; naive =
  nearest ambulance + nearest hospital; AI = coordinator + rule-only hospital agent (RF not trained).
  Result: appropriate hospital (trauma_level >= severity, beds > 0, blood when needed) AI 9/10 vs naive
  5/10; mean total 27.47 vs 25.88 min (AI +1.6 min for capable hospitals). Honest miss: S5 (Hebbal,
  severity 3) AI picked Aster CMI (level 2) because bed availability outweighed the trauma gap in the
  rule score; not tuned to win. Burn/cardiac: no hospital lists the specialty, so only trauma level is
  checked. `POST /benchmark/run` returns the table. Logged (component=system).

- 2026-10-21: VitalDB model deployed. `python -m backend.ml.vitaldb_model --train-final` fits the CV winner
  (HGB; inner-CV grid on all windows picked lr 0.1, 31 leaves, inner PR-AUC 0.512) wrapped in isotonic
  calibration on grouped folds (class_weight=balanced inflates raw scores). Saved
  `models/vitaldb_hgb_v1_20261021.joblib` (5.8 MB). Sanity check on a 20k-window sample (in-sample,
  indicative only): mean p 0.137 vs prevalence 0.135; bins 0-.1 -> 0.036, .5-1 -> 0.707.
  `FusedRisk` gains `deterioration_10min_prob` + `deterioration_model` (docs/07 §2.3 first); services load
  the latest `vitaldb_hgb_v*.joblib`; `/incident/{id}/vitals` computes it from the last 5 readings treated
  as 1-minute steps [ASSUMPTION: ambulance readings ~1/min]. Dispatcher + hospital pages show it with the
  "intra-operative model" label. Browser check: demo replay of falling vitals -> 78%.

- 2026-10-21: SHAP for the deployed VitalDB HGB (`python -m backend.ml.vitaldb_explain`): TreeExplainer on each of
  the 3 calibrated HGB fold models, averaged (log-odds before isotonic calibration; direction/ranking only).
  Global, 5,000 sampled windows (`reports/vitaldb_shap_*.csv`, `reports/figures/vitaldb_shap_{bar,beeswarm}.png`):
  by vital SBP 1.03 > temp 0.55 > HR 0.33 > RR 0.18 > SpO2 0.14 (mean |SHAP| summed); top features last_sbp,
  min_sbp, change_temp, last_temp, change_hr. Low/falling SBP and rising HR raise risk (plausible). Temperature
  ranks high mostly when MISSING (grey in beeswarm): probe not yet placed early in surgery, around induction
  hypotension = surgical workflow artefact, stated limit; temp is excluded from the live reasons. Live: top-3
  reasons per vitals reading in `FusedRisk.deterioration_reasons` (docs/07 §2.3), ~1 ms, shown as "Why" on
  dispatcher + hospital pages. Example (demo replay): "SBP latest 88 mmHg (raises risk) · SBP 5-min lowest
  88 mmHg (raises risk) · HR 5-min highest 131 bpm (raises risk)".

- 2026-10-21: Traffic volumes from published sources (D20) instead of own video. Current: BTP ORR study (DH,
  25 Oct 2025; read in browser): Iblur Jn - Bellandur Main Rd Jn ~10,657 PCU at peak vs 4,800 design, ORR V/C 2.22;
  ~3,360 BMTC buses/day on ORR (quoted). Silk Board (DH, 18 Jul 2024): 17,816 vehicles 09:30-10:30, >=8,300 from
  BTM; cycle 285 s at 09:30, green ~150 s BTM / ~90 s HSR (ORR) / 45 s each Hosur Rd side. OLD: KRDCL Elevated
  Corridors DFR (Feb 2019) Table E-1, April 2016 7-day TVC near Silk Board / KR Puram / Marathahalli -> vehicle
  mix and daily profile only. CMP 2020 PDF link 404. NEXT (needs TomTom data): OSM corridor, demand fitted to the
  2025/2024 peaks, Silk Board signal program from the 2024 timings, TomTom times as the speed check.

- 2026-10-21: OSM corridor (D21). TomTom typical eastbound completed with the key on this laptop (672 rows, 0 rejected; reverse direction still missing). `scripts/build_calibration_targets.py` -> `data/traffic/calibration_targets.csv` (weekday slots low 02-04 / medium 12-14 / peak 17-19 IST; SB->KRP typical 1533 / 2312 / 2937 s). OSM extract via overpass.kumi.systems (1,540 ways, main roads, bbox 12.905-13.010 N, 77.610-77.712 E). netconvert keeps OSM signal tags (219 TLS); `--tls.discard-simple` had dropped them. ORR route found leg by leg (plain shortest path cut through Koramangala): 17.3 km incl. 0.9 km before Silk Board; legs 6.65 / 5.17 / 6.00 km vs TomTom 6.27 / 4.62 / 5.92. Finding: no ORR signals from Bellandur to KR Puram (underpass / flyover / unsignalised). RL junctions = the 4 junction signals in the first 6 km (D21, `sumo/osm/tls_map.json`, main/cross by approach direction). NEXT: OSM env wrapper (2-phase states), demand fitting to the slot targets, calibration table.

- 2026-10-22: Indic ASR trial (90 dev clips, forced language, rp 1.1, VAD OFF for every row below, large-v3 int8_float16,
  fine-tunes float16, GPU). Kannada: vasista22/whisper-kannada-medium WER .478 / CER .154 vs large-v3 .801 / .242.
  Hindi: vasista22/whisper-hindi-medium .257 / .117 vs large-v3 .308 / .157. Both Apache-2.0, converted with
  `scripts/convert_hf_whisper.py` to models/whisper/ct2/ (gitignored). VAD cut sentence endings with the kn fine-tune
  (WER .624 / CER .357 with VAD on): VAD now configurable (WHISPER_VAD, WHISPER_VAD_<LANG>, eval_wer --no-vad).
  `LanguageRouter` (backend/asr/transcribe.py) picks WHISPER_MODEL_<LANG> per language; services use it. Local .env
  set to kn/hi fine-tunes + large-v3-turbo default. Caveat: dev clips only; the held-out clips (docs/NATIVE_SPEAKER_TASKS.md
  Part C) will give the clean number. Candidate not yet tried: ARTPARK-IISc/whisper-small-vaani-kannada.
- 2026-10-22: OSM demand calibration v1 FAILED (stopped during validation). Speed factor fit OK (0.806 -> night corridor
  39.2 vs TomTom 39.5 km/h) but corridor speed stayed 34-36 km/h for scale 1.5-3.0 (targets 26.2 medium, 20.6 peak).
  Diagnosis: eastbound through traffic enters only ~0.9 km before Silk Board and queues at the Silk Board signal
  (218 halted on the approach edge, mean depart delay 218 s, only 233 through vehicles finished in 1.5 h); the route
  uses a 1-lane 'Sarjapur Road' edge at J3 instead of the ORR carriageway; westbound gridlocks at the Mahadevapura
  flyover merge (1 vehicle finished). So downstream legs run nearly empty (45-47 km/h). NEXT: network cleanup at those
  junctions + demand entering along the corridor (per-leg on-ramps), then re-fit.

- 2026-10-22: Network cleanup (1-day timebox). Route finder now uses a weighted Dijkstra preferring the ORR main
  carriageway (>= 2 lanes, ORR-named; others x4 cost): eb/wb routes all >= 3 lanes, legs 6.70/5.17/6.00 km eb.
  The old J3 (Sarjapur Rd signal) sits on a 1-lane at-grade side road; the ORR main carriageway is grade-separated
  there -> D21 revised: controlled = Silk Board (1.0 km), ped crossing (2.0 km), 19th Main jn (3.6 km), jn at 6.0 km.
  Demand v2: through share 0.4 + per-leg local flows (on-ramps) both directions + per-leg multipliers fitted per slot.
  Result v2 (reports/rl_calibration_demand_v2_failed.md): speed factor 0.75 (night 39.5 vs 39.5 km/h); medium fit
  reached 9 % on the fit seed with multipliers 0.85/3.6/4.2, but Marathahalli->KR Puram probes did not finish in
  validation; peak needed 14.4x local demand on Bellandur->Marathahalli and gridlocked (full corridor +51 %).
  -> Fallback as agreed: SPEED-ONLY calibration (`scripts/calibrate_osm_speed.py`): moderate fixed demand, per-leg
  per-slot lane speed limits (variable speed signs) fitted to TomTom typical leg speeds; RL env loads
  sumo/demand/<slot>.vss.add.xml. Stated limit: slot speeds imposed, not emergent from demand.

- 2026-10-22: Speed-only calibration, final (run 3). Silk Board on the reported 2024 program
  (`sumo/osm/silkboard_2024.tll.xml`, built by build_osm_corridor from link geometry: route-direction 147+3 s,
  opposite 87+3 s, Hosur 42+3 s). Lane limits capped at 60 km/h; signal-limited legs get lower local demand
  (Silk Board leg x0.54 medium, x0.62 peak). Validation 5 seeds: all 12 rows within 15 % of TomTom typical, worst
  -10.3 % (peak Silk Board leg); full corridor low +0.3 %, medium +4.4 %, peak +3.5 %. Live (point speed) errors up
  to -46 %: indicative only. experiments/log.csv: the LAST three `osm_speed_*` rows (2026-10-22) are final; earlier
  `osm_speed_*` rows (run 1 without the Silk Board program; run 2 with uncapped limits, 2 legs missed) and the
  `osm_demand_*` rows are superseded. RL env: episodes end 1 km past the last controlled signal (7.7 km), cap 3600 s,
  `amb_progress_m` metric; fixed-time peak ambulance waits ~10 min at Silk Board + long jam before the 6.0 km jn.
  NEXT: backend/rl/demand.py (randomized demand, docs/19 §7), baselines + DQN 3 seeds on OSM (overnight).

## Blockers
- Rotate the old ORS key from the 6th-sem repo before using ORS here.
- `make` is not installed on this machine; tests run with `python -m pytest -q` (same as
  `make test`). Install make (`choco install make`) before targets with real recipes are needed.