# Progress — Madhan (Component 3: Traffic RL + police) · also: all Claude Code work

## Role
- Owns: SUMO ORR corridor, RL env wrapper, DQN training, fixed-time + always-green baselines,
  police alerts, traffic RF retrain (docs/03, docs/09 §4, docs/08 §6).
- Also writes all code for every component with Claude Code; teammates run training.

## Done
- 2026-09-29: Repo created. Docs imported, `docs/15_DECISIONS_FINAL.md` written, AGENTS.md /
  CLAUDE.md set up, `docs/00` wording corrected, skeleton + Makefile stubs, `backend/schemas.py`
  (docs/07 contract) with 10 passing tests.

## Next (in order)
1. **Import the 6th-sem MVP** from the laptop into this repo (one Claude Code session):
   copy agents, ML models, benchmark, frontend, SUMO node/edge files into the layout in
   docs/09 §1; move the ORS key to `.env` and **rotate it**; run the MVP once to confirm it works;
   document current `hospital_agent.py` weights and implemented coordinator conflicts in
   `docs/progress/ragavendra.md` (docs/15 D9, D10).
2. Pin SUMO / sumo-rl / SB3 versions that actually install; rebuild `corridor.net.xml`.
3. Route file + 3 demand levels (docs/08 §6); fixed-time + always-green baselines.
4. Then Chetan's MIMIC pipeline (week 1 priority, so he can start running).

## Blockers
- None yet.
