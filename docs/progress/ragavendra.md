# Progress — Ragavendra (Component 4: Hospital + integration)

## Role
- Owns: hospital table, scoring formula reconciliation, blood-stock factor, coordinator
  conflict table, FastAPI endpoints + WebSockets, frontend panels, benchmark (8+2 scenarios)
  (docs/04, 07 §2.4–2.7 + §5, 08 §7, 12 §05, 15 D9–D11).

## Do now (no code needed)
1. Once the MVP is imported: list the existing hospitals and ambulance bases and check each
   hospital's real attributes from public sources: trauma capability (level 0–3),
   specialties, 24×7 status. Target columns are in docs/08 §7.1. Record sources in
   `data/PROVENANCE.md`.
2. Check eRaktKosh (https://eraktkosh.mohfw.gov.in) for any open API or export. If none, write
   down that you checked (date) — then blood stock is simulated and labelled.
3. Design the 2 extra benchmark scenarios (docs/08 §7.4): two equally close hospitals, only one
   with matching blood stock.

## MVP audit (to be filled during the import session)
- Current `hospital_agent.py` weights / penalties: [TODO]
- Coordinator conflict types already implemented: [TODO]
- Gap vs docs/12 §05 six-row table: [TODO]

## Done
- (nothing yet)

## Blockers
- MVP code not imported yet.
