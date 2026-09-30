# 19 — RL Corridor: Real Network, Real-Data Calibration, Randomized Training

**Status:** `[DECISION]` (see `15` D17). This file **replaces** the schematic straight-line
network in `03` §5 and the fixed demand ranges in `08` §6. The RL agent spec itself
(`09` §4: 16 actions, 24-d observation, reward, DQN, baselines, seeds) is unchanged.

**Change 2026-09-30:** Google Routes replaced by TomTom Routing (historic traffic, future
`departAt`). Reason: Google Cloud India billing needs a ₹1,000 prepayment; project is
free-tier only (`15` §4). One key (`TOMTOM_API_KEY`) now covers §2 and §3.

**Goal:** the SUMO corridor and its traffic must be defensibly close to real Bengaluru ORR
traffic across the day, not guessed. Claim in the report: *"real OSM geometry; demand fitted
to our own vehicle counts; validated against TomTom typical (historic) and live travel times
per time slot."*

---

## Pipeline overview

```
[1] OSM network ──► sumo/osm/corridor.net.xml (real lanes, junctions, 4 signals)
[2] TomTom Routing (historic typical, 24h×7) ─┐
[3] TomTom Flow (live, polled)  ──────────────┼──► data/raw/traffic/*.csv ──► speed/time targets per slot
[4] YOLO counts from own video  ──────────────┴──► counts + vehicle mix per junction
[5] routeSampler / calibrators ──► sumo/demand/<slot>.rou.xml (fitted demand)
[6] Validation table (sim vs real travel time per slot, target ±15%)
[7] RL training on randomized demand ──► evaluation on calibrated low / medium / peak slots
```

## 1. Real network from OpenStreetMap

- Extract the ORR bounding box covering Silk Board → Bellandur → Marathahalli → KR Puram
  (OSM Overpass export or `osmWebWizard.py`). Save the raw `.osm` to `data/raw/osm/`
  (OSM is ODbL: record in `PROVENANCE.md`, attribution required).
- `netconvert --osm-files ... --geometry.remove --roundabouts.guess --ramps.guess
  --junctions.join --tls.guess-signals --tls.discard-simple --tls.join
  --output.street-names -o sumo/osm/corridor.net.xml`.
- Keep only the ORR main carriageway plus the cross streets at the 4 target junctions
  (`netconvert --keep-edges.in-boundary` or edge filtering). Document removed parts.
- Identify the 4 traffic-light IDs (`tlLogic`) matching Silk Board, Bellandur, Marathahalli,
  KR Puram → `sumo/osm/tls_map.json` (`{"silkboard": "<tls_id>", ...}`). Verify visually in
  `sumo-gui`. `[TODO-VERIFY: Silk Board is a complex multi-level junction; flyovers bypass the
  signal — model only the signalised at-grade movement and state it]`.
- The RL env wrapper must work with the phase structure of these real TLS programs: map
  `main_green` / `cross_green` to the real phase indices per junction in `tls_map.json`.
- The existing schematic corridor in `sumo/` (straight line, real spacing, `gen_routes.py`)
  stays as-is: it is the working fallback if the OSM import stalls, and the base for the
  optional compressed-spacing experiment. New OSM work lives in `sumo/osm/`.

## 2. TomTom Routing API — typical travel times (historic, full week)

- `scripts/collect_tomtom_typical.py`: for each corridor segment (SB→Bellandur,
  Bellandur→Marathahalli, Marathahalli→KRP, and full SB→KRP) in **both directions**, call
  `calculateRoute` with `traffic=true`, `computeTravelTimeFor=all`, `travelMode=car` and a
  **future** `departAt` (IST offset) for every hour of a representative week (24 × 7 × 8 =
  1,344 requests).
- Store `historicTrafficTravelTimeInSeconds` (typical, primary target),
  `noTrafficTravelTimeInSeconds` (free-flow), `travelTimeInSeconds`, `lengthInMeters`.
- Key `TOMTOM_API_KEY` in `.env`. Cache every response; never re-query the same slot. Spread the
  run over ≥ 2 days so that, together with §3 polling, daily usage stays within the free tier
  (`[TODO-VERIFY]` current TomTom free daily limit before running). Script must be resumable.
- Output: `data/raw/traffic/tomtom_typical.csv`
  `(collected_at, segment, direction, dow, hour, historic_s, no_traffic_s, travel_time_s, length_m)`.
- Note: these are TomTom's historic-pattern estimates; label them "typical (TomTom historic)".

## 3. TomTom Traffic Flow API — live observed speeds

- `scripts/collect_tomtom.py`: every 30 min, query Flow Segment Data for 2–3 points per
  segment; store `currentSpeed, freeFlowSpeed, currentTravelTime, freeFlowTravelTime,
  confidence, roadClosure`. Key `TOMTOM_API_KEY`. Run for ≥ 7 days in the background.
- Output: `data/raw/traffic/tomtom_flow.csv`.
- Use: live check of the historic typical profile from §2 (correlation per hour reported).
  Same provider as §2, so this is a consistency check, not an independent source (§9).

## 4. YOLO vehicle counts from own video

- Record 10–15 min at 2–3 junction approaches (footbridge, fixed phone, daylight), at least one
  peak and one off-peak clip. Record location, time, direction. Consent/privacy: no faces or
  plates stored in outputs; raw video stays off git.
- `backend/vision/count_vehicles.py`: YOLOv8 (pretrained COCO: car, motorcycle, bus, truck;
  autos map to `[TODO-VERIFY]` class, likely car/truck confusion → spot-check manually)
  + ByteTrack tracker + a virtual counting line per direction. Output vehicles per 5 min by
  class: `data/processed/traffic/yolo_counts.csv`.
- **Validate the counter:** hand-count 2 × 2-minute segments; report count error per class.
- Convert to SUMO vehicle types: car, motorcycle (two-wheeler, `vClass=motorcycle`),
  bus, truck, auto (`[ASSUMPTION]` length ~2.6 m, max speed ~50 km/h). Vehicle mix % goes into
  every demand file.
- These counts are the **independent** real-world check (own measurement, not TomTom).

## 5. Fit demand with routeSampler

- Build detector/edge count targets from YOLO counts (scaled per hour by the TomTom
  time-of-day profile where no direct counts exist; state this).
- `randomTrips.py` → candidate routes; `routeSampler.py --edgedata-files <counts> --routes
  <candidates>` → `sumo/demand/<slot>.rou.xml` per time slot.
- Optionally SUMO calibrators for speed matching.

## 6. Validation (the proof table)

- For each slot, simulate ≥ 5 seeds without RL (default TLS), measure mean travel time of
  ordinary cars per segment and SB→KRP.
- Table in `reports/rl_calibration.md`: `slot | TomTom typical | TomTom live | SUMO |
  error % vs typical | error % vs live`. **Target: |error| ≤ 15%** for low/medium/peak slots.
  Also GEH < 5 for count-matched edges (YOLO counts = independent check).
- Tune demand scaling per slot until targets are met; log every iteration in
  `experiments/log.csv` (`component=rl_calibration`).
- If a slot cannot be matched (e.g. Silk Board spillback), report it honestly.

## 7. Time slots and randomized training

- **Calibrated evaluation slots:** `low` = late night (≈ 02:00–04:00), `medium` = midday
  (≈ 12:00–14:00), `peak` = evening rush (≈ 18:00–20:00, from the TomTom typical/live peak
  hour). Final slot hours are chosen from the collected data and recorded in `docs/progress`.
- **Training uses domain randomization:** each episode samples a slot and then perturbs it:
  demand × U(0.8, 1.2), vehicle-mix jitter ±5 pp, random ambulance entry time, and with
  prob. 0.1 a random lane-blocking incident on one edge. Seeds fixed per episode index.
- **Evaluation** (unchanged from `09` §4): ≥ 20 deterministic episodes per calibrated slot,
  ≥ 3 training seeds, baselines fixed-time (real TLS programs from OSM/netconvert) and
  always-green; metrics = ambulance transit time, mean waiting of others, throughput, max
  cross queue.

## 8. Files and targets

| Path | Purpose |
|---|---|
| `scripts/collect_tomtom_typical.py` | typical travel times, TomTom historic (resumable, ≥ 2 days) |
| `scripts/collect_tomtom.py` | live flow polling (background ≥ 7 days) |
| `backend/vision/count_vehicles.py` | YOLOv8 + ByteTrack counts |
| `sumo/osm/` | OSM-derived net, `tls_map.json`, vehicle types |
| `sumo/demand/` | fitted `<slot>.rou.xml` |
| `backend/rl/demand.py` | slot loading + randomization |
| `reports/rl_calibration.md` | validation table |

Makefile: `collect-typical` (TomTom Routing), `collect-traffic` (TomTom Flow),
`count-vehicles VIDEO=...`, `sumo-osm`, `sumo-demand`, `sumo-validate`, then existing
`rl-baseline`, `rl-train`, `rl-eval`.

## 9. Honest limits (state them)

- Typical and live travel times both come from TomTom: not two independent sources. YOLO
  counts + GEH are the only independent real-world check.
- TomTom typical values are historic-pattern estimates; TomTom live covers only the collection
  week.
- YOLO counts come from short clips at a few approaches; other edges are scaled estimates.
- Signal timings in OSM/netconvert are guessed, not the real Bengaluru programs.
- Still a simulation; results are not field measurements.