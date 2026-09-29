# CLAUDE.md

@AGENTS.md
@docs/15_DECISIONS_FINAL.md

## How to work in this repo

- `docs/15_DECISIONS_FINAL.md` overrides every other doc where they conflict.
- Do NOT read all of `docs/` at the start of a session. Read only the files the
  AGENTS.md routing table lists for the current task.
- Never treat `docs/11`, `13`, `14`, `16`, `17`, `18` as a build spec.
- One task per session. Before coding, state a short plan and the files you will touch.
- Start by reading `docs/progress/<owner>.md` for the component you are working on.
- Every model module exposes `train()`, `evaluate()`, `predict()`, `save()`, `load()` and
  is runnable by one Makefile target, so teammates can train without editing code.
- Config via CLI flags / YAML / `.env`, never hard-coded paths or keys.
- Write or update tests in `tests/` for every scoring, label, metric and schema function.
  Run `make test` before proposing a commit.
- At the end of a session: update `docs/progress/<owner>.md` (done / next / blockers)
  and the status table in `AGENTS.md`, then suggest a commit message.
- If a spec is ambiguous, mark `[TODO-VERIFY]` and ask; never invent dataset columns,
  MIMIC item IDs, citations or numbers.
