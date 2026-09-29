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

## Next (in order)
1. **Week 1, second session:** `backend/data/mimic_windows.py` + `make data-mimic` (docs/08 §2),
   once Chetan has the MIMIC demo downloaded and the item-ID list checked.
2. **Week 2:** pin SUMO / sumo-rl / SB3 versions that actually install; write
   `sumo/corridor.nod.xml` + `.edg.xml` (junctions at 0 / 6000 / 11000 / 16000 m, cross
   streets, 3+3 main lanes, 2+2 cross lanes), routes with 3 demand levels (docs/08 §6),
   fixed-time + always-green baselines.

## Blockers
- Rotate the old ORS key from the 6th-sem repo before using ORS here.
- `make` is not installed on this machine; tests run with `python -m pytest -q` (same as
  `make test`). Install make (`choco install make`) before targets with real recipes are needed.
