# RL corridor calibration (OSM ORR) — speed-only fallback

Generated 2026-10-22 by `python -m scripts.calibrate_osm_speed`. Simulation, not field data.

**Method.** Demand-only fitting failed (peak needed ~14x local demand on one leg, then gridlocked; see `rl_calibration_demand_v2_failed.md`). Here demand is fixed and moderate per slot and each leg's lane speed limit is fitted per slot so probe speeds match TomTom typical speeds. **Slot speeds are imposed, not emergent from demand.** Queues at the 4 RL signals still come from the simulated traffic.

* Driver speed factor 0.75 (fitted on the night slot); demand scale per slot: low 0.15, medium 1.0, peak 1.0
* Lane limits capped at 60 km/h (OSM ORR limit); where a leg was still too slow at the cap (signal-limited), its local demand was lowered instead. Local-demand multipliers: low: silkboard_bellandur 1.0, bellandur_marathahalli 0.8961, marathahalli_krpuram 0.7824; medium: silkboard_bellandur 0.5443, bellandur_marathahalli 1.0, marathahalli_krpuram 1.0; peak: silkboard_bellandur 0.6221, bellandur_marathahalli 1.0, marathahalli_krpuram 1.0
* Fitted lane speed limits (km/h): low: silkboard_bellandur 55.4, bellandur_marathahalli 60.0, marathahalli_krpuram 60.0; medium: silkboard_bellandur 60.0, bellandur_marathahalli 45.1, marathahalli_krpuram 52.9; peak: silkboard_bellandur 60.0, bellandur_marathahalli 35.6, marathahalli_krpuram 33.6
* Silk Board runs the reported 2024 peak program (285 s cycle: BTM/ORR-west 150 s, HSR/ORR-east 90 s, Hosur Rd 45 s; `sumo/osm/silkboard_2024.tll.xml`); other signals use OSM/netconvert programs.
* Leg lengths in SUMO: silkboard_bellandur 6.70 km, bellandur_marathahalli 5.17 km, marathahalli_krpuram 6.00 km, silkboard_krpuram 16.29 km

## Validation (5 seeds, target |error| ≤ 15 % vs TomTom typical)

| slot | segment | TomTom typical km/h | TomTom live km/h | SUMO km/h (mean ± sd) | err vs typical | ≤15 % |
|---|---|---|---|---|---|---|
| low | silkboard_bellandur | 35.7 | 42.0 | 34.7 ± 0.7 | -2.9 % | yes |
| low | bellandur_marathahalli | 42.9 | 39.0 | 42.1 ± 1.1 | -1.7 % | yes |
| low | marathahalli_krpuram | 41.5 | 40.4 | 41.0 ± 1.3 | -1.1 % | yes |
| low | silkboard_krpuram | 39.5 | 42.0 | 39.6 ± 0.9 | +0.3 % | yes |
| medium | silkboard_bellandur | 21.2 | 40.0 | 22.7 ± 1.5 | +7.0 % | yes |
| medium | bellandur_marathahalli | 29.1 | 24.5 | 27.0 ± 0.3 | -7.3 % | yes |
| medium | marathahalli_krpuram | 30.9 | 37.6 | 30.5 ± 0.8 | -1.4 % | yes |
| medium | silkboard_krpuram | 26.2 | 40.0 | 27.3 ± 0.6 | +4.4 % | yes |
| peak | silkboard_bellandur | 21.8 | 36.1 | 19.6 ± 0.9 | -10.3 % | yes |
| peak | bellandur_marathahalli | 20.8 | 26.6 | 19.9 ± 0.6 | -4.0 % | yes |
| peak | marathahalli_krpuram | 18.9 | 24.8 | 19.2 ± 0.3 | +1.1 % | yes |
| peak | silkboard_krpuram | 20.6 | 36.1 | 21.3 ± 1.0 | +3.5 % | yes |

## Limits

* Speeds per leg and slot are imposed with variable speed signs; congestion is not reproduced by demand.
* Demand level, through share and background trips are assumptions (scripts/calibrate_osm_demand.py).
* Vehicle mix from a 2016 model table (D20); TomTom typical and live are one provider (docs/19 §9).
* Westbound uses the eastbound leg limits (TomTom westbound typical times were not collected).
