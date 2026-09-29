# Pre-Arrival Multimodal AI Triage & Emergency Coordination — Bengaluru

Final-year B.E. (AI & ML) project, CMR Institute of Technology, Bengaluru.

> **Research prototype, not a medical device.** Not clinically validated. Must not be used for
> real patient care or real emergency dispatch.

A pre-arrival pipeline that connects four stages of emergency response:

1. **Reporting** — multilingual (Kannada / Hindi / English) voice or text intake with Whisper,
   a bounded triage dialogue, and first-aid guidance grounded only in curated protocol text (RAG).
2. **Vitals** — a two-head BiGRU over ambulance vital-sign sequences predicting current severity
   and deterioration, benchmarked against NEWS2 and tabular baselines; fused with the report
   into a live-updating Emergency Risk Score.
3. **Traffic** — DQN signal control on a SUMO model of the Outer Ring Road corridor
   (Silk Board → Bellandur → Marathahalli → KR Puram), benchmarked against fixed-time signals.
4. **Hospital** — capability-aware hospital ranking (trauma level, beds, specialty, blood stock),
   with a rule-based coordinator resolving conflicts.

The contribution is **context and integration** of established methods, evaluated honestly
against clinical and rule-based baselines — not a new architecture.

## Team

| Component | Owner |
|---|---|
| 1 Reporting (Whisper, dialogue, RAG) | Sankalp |
| 2 Vitals ML + fusion | Chetan |
| 3 Traffic RL + police | Madhan |
| 4 Hospital + integration | Ragavendra |

## Quick start

```bash
git clone https://github.com/madhanmatthew/prearrival-triage-blr.git
cd prearrival-triage-blr
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # fill in keys; never commit .env
make test
```

Windows users: run `make` from Git Bash or WSL.

## Where to read

- **Start here:** [`AGENTS.md`](AGENTS.md) (rules + which doc to read for which task)
- **Final decisions:** [`docs/15_DECISIONS_FINAL.md`](docs/15_DECISIONS_FINAL.md)
- **Your tasks:** `docs/progress/<your-name>.md`
- **Contracts between components:** [`docs/07_interfaces_and_schemas.md`](docs/07_interfaces_and_schemas.md), implemented in [`backend/schemas.py`](backend/schemas.py)

## Team workflow

- Code is written on Madhan's machine with Claude Code; everyone else pulls and runs.
- Work on your own branch (`train/vitals`, `train/rl`, …) and open a PR to `main`.
- Push results, not artifacts: rows in `experiments/log.csv`, figures in `reports/figures/`,
  notes in `docs/progress/<name>.md`. Models and data go to the shared Drive, never git.
- Every reported number must trace back to a row in `experiments/log.csv`.

## Data

MIMIC-III Clinical Database Demo (PhysioNet) is used under its terms and is **never committed**.
Each teammate downloads it separately. See [`data/PROVENANCE.md`](data/PROVENANCE.md).
