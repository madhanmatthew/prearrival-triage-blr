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

## Next (in order)
1. **Week 1, first session:** `backend/ml/news2.py` (exact bands in docs/02 §3, GCS→AVPU in
   docs/08 §2) with unit tests on known clinical examples. No data needed.
2. **Week 1, second session:** `backend/data/mimic_windows.py` + `make data-mimic` (docs/08 §2),
   once Chetan has the MIMIC demo downloaded and the item-ID list checked.
3. **Week 2:** pin SUMO / sumo-rl / SB3 versions that actually install; write
   `sumo/corridor.nod.xml` + `.edg.xml` (junctions at 0 / 6000 / 11000 / 16000 m, cross
   streets, 3+3 main lanes, 2+2 cross lanes), routes with 3 demand levels (docs/08 §6),
   fixed-time + always-green baselines.

## Blockers
- Rotate the old ORS key from the 6th-sem repo before using ORS here.
