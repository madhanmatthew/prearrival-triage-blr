# RL corridor calibration (OSM ORR, docs/19 §6, D20, D21)

Generated 2026-10-22 by `python -m scripts.calibrate_osm_demand`. Simulation, not field data.

* Fitted desired-speed factor (LOW slot, night): **0.75**
* Fitted demand scale per slot: low **0.15**, medium **1.0**, peak **1.0** (per direction: through 1200 veh/h x scale; local per leg 1800 veh/h x scale x leg multiplier; background 3000 veh/h x scale)
* Per-leg multipliers (fitted to each leg's TomTom typical speed, same both directions): low: silkboard_bellandur 1.0, bellandur_marathahalli 1.0, marathahalli_krpuram 1.0; medium: silkboard_bellandur 0.8537, bellandur_marathahalli 3.613, marathahalli_krpuram 4.2108; peak: silkboard_bellandur 0.9222, bellandur_marathahalli 14.3887, marathahalli_krpuram 4.0
* Vehicle mix: motorcycle 49%, car 38%, autorickshaw 11%, bus 2% (KRDCL DFR 2019, 2016 model PCUs converted with IRC 106 factors; not a 2026 count)
* Speeds compared (SUMO leg lengths: silkboard_bellandur 6.70 km, bellandur_marathahalli 5.17 km, marathahalli_krpuram 6.00 km, silkboard_krpuram 16.29 km); target |error| <= 15 % vs TomTom typical. Live = point speed at leg start (indicative only).

## Validation

| slot | segment | TomTom typical km/h | TomTom live km/h | SUMO km/h (mean ± sd) | err vs typical | err vs live | ≤15 % |
|---|---|---|---|---|---|---|---|
| low | silkboard_bellandur | 35.7 | 42.0 | 41.2 ± 0.8 | +15.3 % | -1.9 % | NO |
| low | bellandur_marathahalli | 42.9 | 39.0 | 40.1 ± 0.8 | -6.4 % | +3.0 % | yes |
| low | marathahalli_krpuram | 41.5 | 40.4 | 44.7 ± 1.6 | +7.8 % | +10.6 % | yes |
| low | silkboard_krpuram | 39.5 | 42.0 | 40.9 ± 0.8 | +3.6 % | -2.6 % | yes |
| medium | silkboard_bellandur | 21.2 | 40.0 | 20.4 ± 1.9 | -4.0 % | -49.0 % | yes |
| medium | bellandur_marathahalli | 29.1 | 24.5 | 32.8 ± 1.0 | +12.8 % | +33.8 % | yes |
| medium | marathahalli_krpuram | 30.9 | 37.6 | nan ± nan | +nan % |  | NO |
| medium | silkboard_krpuram | 26.2 | 40.0 | 29.9 ± 1.1 | +14.3 % | -25.2 % | yes |
| peak | silkboard_bellandur | 21.8 | 36.1 | 18.9 ± 2.2 | -13.3 % | -47.6 % | yes |
| peak | bellandur_marathahalli | 20.8 | 26.6 | nan ± nan | +nan % |  | NO |
| peak | marathahalli_krpuram | 18.9 | 24.8 | nan ± nan | +nan % |  | NO |
| peak | silkboard_krpuram | 20.6 | 36.1 | 31.2 ± 0.3 | +51.2 % | -13.7 % | NO |

## Volume check (not fitted)

SUMO eastbound flow just past the 5.9 km junction (toward Bellandur), peak slot: **1412.0 PCU/h**. BTP 2025 (Deccan Herald): Iblur–Bellandur ~10,657 PCU at peak vs 4,800 design capacity; direction and exact hour not stated, so this is an order-of-magnitude check only.

## Limits

* One speed factor for all vehicle types and slots; demand scale fitted on the full corridor only.
* Background trips are random OD pairs on the extract, not observed OD flows.
* Signal programs other than the 4 RL junctions are OSM/netconvert defaults.
* TomTom typical and live are one provider (docs/19 §9); live is a point speed.
