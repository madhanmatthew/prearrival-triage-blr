# Progress — Ragavendra (Component 4: Hospital + integration)

## Role
- Owns: hospital + ambulance-base tables, hospital scoring (docs/08 §7.2 formula is the spec),
  blood-stock factor, RF surrogate (docs/08 §7.3), coordinator with all 6 conflicts
  (docs/12 §05), FastAPI endpoints + WebSockets, frontend panels, benchmark 8+2 scenarios
  with a fair baseline (docs/04, 07 §2.4–2.7 + §5, 15 D9–D11, D15).
- Fresh build: nothing from the 6th-sem MVP is reused (docs/15 D14).

## Do now (no code needed)
1. **Build `data/hospitals.csv`** from public sources: 15–30 Bengaluru hospitals with the
   docs/08 §7.1 columns: name, lat/lon, trauma level (0–3), beds_total, specialties
   (semicolon list), open_24x7. Leave beds_available and blood columns empty (they are
   simulated later). Record every source in `data/PROVENANCE.md`.
2. **Build `data/ambulance_bases.csv`**: base_id, name, lat, lon (public 108/ambulance
   station locations if available; otherwise mark as approximated).
3. Check eRaktKosh (https://eraktkosh.mohfw.gov.in) for any open API or export. If none,
   write down that you checked and the date; blood stock is then simulated and labelled.
4. Write the **10 benchmark scenarios** (docs/08 §7.4 columns): 8 Bengaluru incidents
   (location, injury type, severity class, blood need) + 2 where two hospitals are equally
   close and only one has matching blood.

## Done
- 2026-10-27 (Madhan + Claude Code): Hospital page (`web/src/pages/Hospital.tsx`) now shows the
  10-min deterioration probability with its top-3 SHAP reasons; live-mode risk-score history is
  built client-side per incident (`appendRiskHistory`, `web/src/store/world.test.ts`). Checked in
  the browser with the demo vitals (7 points: text-only risk + 6 readings).
- 2026-10-27: "Demo vitals" now has a picker: 2 REAL VitalDB windows (10 one-minute readings each,
  picked by data rules, not by the model: case 1 deteriorating, real event ~7 min later; case 2
  stable) + the old hand-written example. `make demo-vitals` -> `frontend/demo_vitals_vitaldb.json`
  (`scripts/build_demo_vitals.py`, `tests/test_build_demo_vitals.py`); `npm run build` picks it up.
  Live check: case 1 -> 10-min risk 26 %, SHAP "SBP 5-min lowest 102", "HR fell 8 bpm". NEWS2 shows
  "–" for VitalDB replays (no AVPU/O2). The deployed model saw these cases in training: illustration only.

## Blockers
- None.
