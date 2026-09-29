# Component 3 — Traffic & Police Agent (Reinforcement Learning Signal Control)

**Owner responsibility:** RL-based traffic signal control for ambulance
green-corridor priority, plus police alerting logic.

**Reads first:** `00_MASTER_OVERVIEW.md` for system context and citations
(SIGMA, EMVLight).

---

## 1. Purpose

Replace fixed-time traffic signal control with a **learned** policy that
prioritizes ambulance transit through a real Bengaluru corridor, without
completely blocking cross-traffic (a naive "always green for ambulance" policy
is trivial and unrealistic — the reward function must penalize general traffic
disruption too).

Also owns: police alert dashboard (incident, severity, probable route) — this
part stays as existing WebSocket-based infrastructure, not new ML.

## 2. Current State (MVP) — honest baseline

- `traffic_agent.py` / `police_agent.py`: RandomForest congestion/delay models
  trained on **synthetic** data; rule-based police deployment sizing
  (units/perimeter scaled by severity).
- Route selection: OpenRouteService (ORS) API shortest-path — **keep this as-is**,
  do not build a custom route-prediction model, it adds complexity without real
  value over a working routing API.
- **No RL-based signal control exists in the MVP at all.** This is entirely new
  work.

## 3. Why RL, and Why This Is a Legitimate (Not Invented-For-Novelty) Choice

- **SIGMA** (2026): RL-based emergency vehicle prioritization, validated on real
  intersection models from **Kolkata, India** — directly precedented for an
  Indian city context, cite this explicitly.
- **EMVLight**: multi-agent RL framework for joint emergency vehicle routing +
  signal control.
- Published RL traffic-signal work generally: delay reductions up to ~73% over
  fixed-time baselines in general traffic scenarios.
- This is the project's third distinct ML paradigm (alongside supervised
  sequential/fusion learning in Component 2, and NLP/RAG in Component 1) — real
  methodological breadth, not decoration.

## 4. The Corridor — Real Bengaluru Geometry

**Full 10-junction list** (use this for the report's motivation/grounding —
sourced from BBMP/traffic police congestion data): Silk Board Junction, KR Puram
Junction, Marathahalli Junction, Bellandur Junction, Hebbal/Mekhri Circle,
Goraguntepalya Junction, Veerannapalya Junction (near BEL), Gokaldas Images
Junction, Banashankari Signal, Tin Factory Junction.

**Actual corridor built/simulated (must be a connected route, not scattered
points):** **Silk Board → Bellandur → Marathahalli → KR Puram** — all on the
Outer Ring Road (ORR), a real continuous corridor, 4 signalized intersections.

**Real approximate inter-junction distances (used to scale the SUMO network):**
- Silk Board → Bellandur: ~6 km
- Bellandur → Marathahalli: ~5 km
- Marathahalli → KR Puram: ~5 km

## 5. Simulation Environment — SUMO + sumo-rl

**Toolchain (already verified working):**
- `pip install eclipse-sumo sumo-rl` — installs SUMO 1.27+ and the RL-environment
  wrapper. (apt-based SUMO install can hit broken-mirror issues in some sandboxed
  environments — the pip package `eclipse-sumo` is the more reliable path and was
  confirmed working.)
- `pip install stable-baselines3` — for DQN/PPO training.
- Set `SUMO_HOME` environment variable to the installed package path (e.g.
  `/usr/local/lib/python3.12/dist-packages/sumo` when installed via pip into that
  environment).

**Network structure (already scaffolded as a starting point — rebuild/extend as
needed):**
- 4 signalized junction nodes (`silkboard`, `bellandur`, `marathahalli`,
  `krpuram`) positioned along a line, spaced at the real distances above
  (scaled in meters: 0m, 6000m, 11000m, 16000m).
- Entry/exit nodes at each end of the corridor (`entry_west`, `exit_east`).
- Each junction also has a **perpendicular cross-street** (north/south approach
  nodes) — this is what makes the signal-control decision non-trivial; without
  cross-traffic, "always green" would be the trivially optimal policy, which
  defeats the point of the RL formulation.
- Main ORR line: 3 lanes each direction (matches real ORR being multi-lane).
- Cross streets: 2 lanes each direction.
- Files: `corridor.nod.xml` (nodes), `corridor.edg.xml` (edges) — built with
  `netconvert` to produce the final `.net.xml` SUMO network file. A route file
  (`.rou.xml`) is needed next, containing background through-traffic + cross
  traffic + at least one explicit ambulance vehicle with elevated priority/
  impatience parameters.

## 6. RL Formulation

| Element | Definition |
|---|---|
| **State** | Per-intersection: incoming lane queue lengths/density, current signal phase, ambulance presence + distance/ETA to that intersection |
| **Action** | Signal phase selection per intersection (which approach gets green) |
| **Reward** | Primary: minimize ambulance transit time / waiting time through the corridor. Secondary (penalty term): general vehicle delay/queue length, to prevent the trivial "always green for ambulance, ignore everyone else" degenerate policy |
| **Algorithm** | DQN or PPO (stable-baselines3) — DQN is the simpler starting point given `sumo-rl`'s Gymnasium-compatible environment wrapper |

## 7. Benchmark — What "Proof of Worth" Looks Like Here

- **Baseline:** SUMO's default fixed-time signal program (no RL).
- **Metric:** ambulance transit time through the full corridor, RL-controlled vs.
  fixed-time, on the same traffic demand pattern.
- **Secondary metric:** general traffic delay/throughput, RL vs. fixed-time — to
  demonstrate the policy isn't just blocking all cross-traffic to help the
  ambulance (report this number honestly even if it shows some general-traffic
  cost — that trade-off is expected and should be discussed, not hidden).
- Report as: "RL-controlled corridor reduced ambulance transit time by X% vs.
  fixed-time signals, with a Y% change in general traffic delay."

## 8. Police Alerting (existing, not new ML)

- WebSocket push to a police dashboard: incident location, severity, and
  probable ambulance route (reused from ORS routing, not separately predicted).
- Deployment sizing (unit count, perimeter radius) stays as existing rule-based
  logic scaled by severity/injured count — this is legitimately simple and does
  not need to be replaced with ML; document it as rule-based, don't disguise it.

## 9. Retraining Existing Traffic Models on Real Data

- The existing `traffic_agent.py` RandomForest delay/congestion models should be
  retrained on real Bengaluru traffic/accident data (sourced from BBMP/Karnataka
  open data or traffic police reports) instead of synthetic data — this is a
  more modest but still necessary upgrade, separate from the new RL work above.

## 10. Evaluation Checklist

- [ ] SUMO network built and validated (vehicles can traverse the full corridor
  without errors)
- [ ] Fixed-time baseline run and ambulance transit time logged
- [ ] RL agent trained (DQN or PPO) on the same network
- [ ] RL-controlled run's ambulance transit time logged and compared to baseline
- [ ] General traffic delay reported for both conditions (not just ambulance
  time — this is what makes the comparison honest)
- [ ] Existing traffic/congestion RandomForest models retrained on real data
  (separate from the RL work, but same component owner)

## 11. Interface Contract With Other Components

**Input from Coordinator:** incident location + severity (to determine which
corridor/intersections need prioritization and how aggressively).

**Output to Coordinator:** live signal-state updates, route status, and any
conflict signals (e.g., a police-cordoned area intersecting the planned route) —
this feeds the Coordinator's conflict-resolution logic, documented in
`05_coordinator_fusion_evaluation.md`.
