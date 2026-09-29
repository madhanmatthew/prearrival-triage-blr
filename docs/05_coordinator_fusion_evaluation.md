# Coordinator, Multimodal Fusion & Evaluation Methodology

**Owner responsibility:** system-level integration — the fused Emergency Risk
Score, coordinator conflict resolution, and the overall evaluation/benchmarking
strategy that ties all 4 components together.

**Reads first:** `00_MASTER_OVERVIEW.md`, and at least skim
`02_ambulance_vitals_agent.md` (fusion depends on it directly).

---

## 1. Purpose

This is the system-level layer that makes the project more than "4 separate
models stapled together." Two responsibilities:

1. **Fused Emergency Risk Score** — combine Component 1's text severity and
   Component 2's vitals-based severity/trend into one live-updating decision
   variable.
2. **Coordinator** — resolve conflicts between agents' outputs (e.g., ambulance's
   fastest route crossing a police-cordoned perimeter) and produce the final
   dispatch decision.

## 2. Current State (MVP) — honest baseline

- Coordinator exists and works: detects conflicts (e.g., route vs. perimeter,
  hospital bed shortfall) and resolves them via a **weighted priority-bidding
  scheme** — hand-tuned formulas, not learned. This is fine to keep as rule-based
  for this project (an RL/learned coordinator was explicitly scoped OUT — see
  `00_MASTER_OVERVIEW.md` Section 12); just document it honestly as rule-based.
- **No fusion of Component 1 and Component 2 outputs exists** — the MVP's 4
  agents run largely independently. Building the fused Emergency Risk Score is
  new work.

## 3. Fused Emergency Risk Score — Design

```
Text Severity (Component 1)  ──┐
                                 ├──► Concatenation ──► Dense layer(s) ──► Fused Risk Score
Vitals Encoder Output (Component 2) ──┘                                    (updates as new
                                                                              vitals stream in)
```

- **Fusion method: concatenation, not attention/transformer.** Justification:
  DeepTriager demonstrates concatenation-based fusion alone reaches AUC
  0.956-0.959 — attention/transformer fusion is not necessary to achieve strong
  results, and is significantly more complex to implement and debug within a
  2-month timeline. Do not over-engineer this.
- **Live-updating property:** the score should refine as new vitals readings
  arrive during transit — mirrors how a clinician continuously updates their
  assessment rather than making one static call. This is the system-level
  novelty claim; make sure the implementation actually re-runs/updates the score
  on new data, not just computes it once at intake.
- Output consumed by: Component 4 (hospital selection input) and the
  Coordinator (dispatch decision priority).

## 4. Ablation Study — Required, Not Optional

**This is what proves the fusion is doing real work, not just adding
complexity for its own sake.** Compare three model variants on the same test
set:

1. **Text-only** severity prediction (Component 1's signal alone)
2. **Vitals-only** severity prediction (Component 2's model alone)
3. **Fused** (both, concatenated)

Report accuracy/F1/AUC for all three. The expected and defensible result: fused
> either alone. If it isn't, that's an important, honestly-reportable finding —
document it rather than hiding it; the pediatric modality-dropout paper (cited
in `00_MASTER_OVERVIEW.md`) specifically addresses cases where fusion models
over-rely on one modality, which is a legitimate, citable failure mode to
discuss if it occurs.

## 5. Coordinator — Conflict Resolution (retained rule-based logic)

- Keep existing weighted priority-bidding formulas.
- Document explicitly in the report: this is intentionally rule-based, not
  learned — an RL-based coordinator was considered and deliberately deferred to
  future work (see `00_MASTER_OVERVIEW.md` Section 12) because it was assessed
  as too ambitious to build and validate credibly within the 2-month timeline.
  This is a scoping decision, not an oversight — say so if asked in viva.

## 6. Full System Evaluation Strategy

### 6.1 Per-Component Baselines (see each component file for detail)

| Component | Baseline(s) to beat |
|---|---|
| Vitals/Triage (Component 2) | Majority-class, NEWS2 clinical score |
| Traffic/Signal Control (Component 3) | Fixed-time signal control |
| Hospital Selection (Component 4) | Distance-only naive selection |
| Overall dispatch | Naive nearest-unit dispatch (existing `benchmark.py`) |

### 6.2 System-Level Benchmark (existing asset — extend, don't rebuild)

- `backend/benchmarks/benchmark.py` already compares full AI-coordinated
  dispatch against a naive nearest-base/straight-line-distance baseline across 8
  real Bengaluru emergency scenarios.
- **Extend this** once real components are in place: re-run the same 8
  scenarios with (a) real vitals-based severity instead of synthetic, (b)
  blood-matched hospital selection, (c) RL-controlled traffic where applicable,
  and report before/after numbers.

### 6.3 Fusion Ablation (Section 4 above)

### 6.4 Explainability Deliverables

- SHAP summary for Component 2's tree-based severity classifier.
- Model card documenting dataset, features, metrics, and limitations for the
  vitals model specifically (this is the component most likely to be scrutinized
  in viva given its clinical framing).

## 7. What "Done" Looks Like for This Document's Scope

- [ ] Fused Emergency Risk Score implemented (concatenation-based)
- [ ] Live-update behavior verified (score changes as new vitals data arrives,
  not computed once statically)
- [ ] Ablation study run and reported (text-only vs. vitals-only vs. fused)
- [ ] `benchmark.py` re-run with real component outputs, before/after numbers
  documented
- [ ] Coordinator's rule-based nature explicitly documented (not disguised as
  learned)

## 8. Report/Paper Framing — Use This Language

> "We build on established multimodal triage fusion architectures (DeepTriager,
> VitalML), applying them to the pre-arrival ambulance-to-hospital context in
> low-resource, multilingual, connectivity-constrained settings — a gap not
> addressed in existing literature, which focuses on in-hospital,
> EHR-scale deployments. Our reinforcement-learning-based traffic signal
> component follows the same real-world grounding, extending prior work (SIGMA,
> validated on Indian intersections) to a named Bengaluru Outer Ring Road
> corridor."

This framing is honest, defensible, and correctly positions novelty as
*context and integration*, not invented architecture — do not deviate from this
framing toward "first-ever" or "novel architecture" claims anywhere in the
report or viva.
