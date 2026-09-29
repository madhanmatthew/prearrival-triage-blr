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

## Next (in order)
1. **Week 2:** tabular baselines (majority, NEWS2, LR, RF, GB) + nested grouped CV (`15` D5).
2. **Week 2:** always-green-for-ambulance baseline (TraCI) + `make rl-baseline` logging to
   `experiments/log.csv`; then RL env wrapper (docs/09 §4: 16 actions, 24-d obs, custom reward).

## Blockers
- Rotate the old ORS key from the 6th-sem repo before using ORS here.
- `make` is not installed on this machine; tests run with `python -m pytest -q` (same as
  `make test`). Install make (`choco install make`) before targets with real recipes are needed.
