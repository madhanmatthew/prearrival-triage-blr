# Component 4 — Hospital Selection Agent

**Owner responsibility:** hospital scoring/selection logic, plus blood-bank
availability integration.

**Reads first:** `00_MASTER_OVERVIEW.md` for system context.

---

## 1. Purpose

Given an incident's severity/trend and location, select the best receiving
hospital based on distance, trauma capability, bed availability, specialty
match, and (new) blood-type stock availability.

## 2. Current State (MVP) — reasonably solid, needs real inputs, not a rebuild

- `hospital_agent.py`: RandomForestRegressor Q-value scoring model, combining
  distance, trauma capability, bed availability (synthetic), and specialty
  bonus — roughly 80% hand-tuned policy score + 20% RF Q-value blend.
- This component is **the least in need of architectural change** among the
  four — the main gaps are (a) synthetic severity input, (b) synthetic bed
  occupancy data, (c) no blood-type awareness.

## 3. Upgrade 1 — Real Severity/Trend Input

- Currently receives a synthetic severity value.
- Replace with the real output from Component 2 (vitals/trend model) — see
  `02_ambulance_vitals_agent.md` Section 10 for the interface contract once
  finalized.
- This single change is what makes hospital selection reflect how critical the
  patient actually is, rather than a placeholder number.

## 4. Upgrade 2 — Blood-Type Availability Scoring Factor

**Important framing — do not misrepresent this in the report:** India already
has **eRaktKosh**, a government-run centralized, real-time blood-stock system,
mandatory for government and private blood banks, searchable by
state/district/blood group/component. **Do not build a standalone blood-bank
finder app** — that duplicates existing government infrastructure and will read
as either naive or overclaiming in a viva.

**Correct scope:** add blood-type availability as **one more scoring factor
inside the existing hospital agent**, alongside distance/beds/trauma/specialty —
not a separate 5th component.

- When severity/injury type suggests transfusion risk (e.g., major trauma,
  significant blood loss signal from Component 2), nudge hospital selection
  toward hospitals with matching blood stock.
- **Data access:** attempt to source live eRaktKosh-style data if an accessible
  API/interface exists; if not, use a realistic simulated blood-stock dataset for
  demo purposes — same pattern already used for hospital bed occupancy in the
  current MVP. Document clearly in the report which was used.

## 5. Upgrade 3 — Real Hospital Data

- Where obtainable, replace synthetic hospital capacity/occupancy data with real
  or realistically-sourced data (public hospital directories, trauma center
  listings for Bengaluru).

## 6. Model / Scoring Structure (retained from MVP, upgraded inputs only)

Keep the existing hybrid scoring approach:
- Rule-based weighted score component (distance, trauma capability, specialty
  match) — this part can stay rule-based; it does not need to become "more ML"
  just for its own sake. Say this plainly in the report rather than disguising
  it as learned.
- RandomForestRegressor Q-value component — retrain on real severity input (from
  Component 2) once available, replacing the synthetic training signal it
  currently uses.

## 7. Evaluation Checklist

- [ ] Severity input switched from synthetic to real (Component 2 output)
- [ ] Blood-type scoring factor added and tested against scenarios where it
  should change the hospital pick (e.g., two equally close hospitals, one with
  matching blood stock)
- [ ] Real or realistic hospital capacity data sourced and documented
- [ ] Benchmark: does hospital selection quality improve (trauma-appropriate +
  blood-matched picks) compared to the MVP's synthetic-input version, across the
  same 8 benchmark scenarios already used in `benchmark.py`?

## 8. Interface Contract With Other Components

**Input from Component 2:** severity/trend score (replaces current synthetic
value).

**Input from Component 3:** current traffic/route conditions (already
integrated in the MVP via distance/ETA).

**Output to Coordinator:** selected hospital + score breakdown (for conflict
resolution and the fused Emergency Risk Score's downstream action).
