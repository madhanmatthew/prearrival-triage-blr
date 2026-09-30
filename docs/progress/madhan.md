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
  at inference (input gap). Dataset format/rubric in `data/text_reports/README.md`; no data yet (Sankalp).

## Next (in order)
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

## Blockers
- Rotate the old ORS key from the 6th-sem repo before using ORS here.
- `make` is not installed on this machine; tests run with `python -m pytest -q` (same as
  `make test`). Install make (`choco install make`) before targets with real recipes are needed.