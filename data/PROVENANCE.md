# Dataset Provenance Log

One entry per dataset (docs/08 §8). Never commit restricted data; record where it came from.

| Dataset | Source URL | Access date | Licence / terms | Preprocessing | Real/Sim | Used by |
|---|---|---|---|---|---|---|
| MIMIC-IV Clinical Database Demo v2.2 | https://physionet.org/content/mimic-iv-demo/2.2/ | [TODO] | PhysioNet terms `[TODO-VERIFY]`; do not redistribute | Hourly windows + outcome labels (docs/08 §2) | REAL (ICU) | Chetan |
| VitalDB open dataset (5,877 cases with HR, SpO2, RR, temp, SBP) | https://vitaldb.net/dataset (API https://api.vitaldb.net) | 2026-10-21 | Open dataset; licence/citation terms `[TODO-VERIFY on vitaldb.net]` | 1-min medians per track (`scripts/download_vitaldb.py`), windows + labels (docs/08 §2b) | REAL (intra-operative) | Madhan |
| KRDCL Elevated Corridors DFR Vol. I (AECOM Asia, Deloitte, ISEC; Feb 2019) | https://data.opencity.in/dataset/bc084f6b-c338-4eb0-bf4a-b67fa4e4aa52/resource/56ec46a8-aebe-40a6-906f-72421c07e6ea/download/2eb72abf-ac7b-4077-b87c-bb7ad321e805.pdf (copy: `data/raw/traffic_src/krdcl_elevated_corridors_dfr_vol1_2019.pdf`) | 2026-10-21 | Government consultancy report shared on OpenCity; terms `[TODO-VERIFY]` | Table E-1 (7-day TVC, April 2016) -> `data/traffic/published_counts.csv` with page refs | REAL (survey 2016) | Madhan |
| BTP ORR capacity study (news-reported) | https://www.deccanherald.com/amp/story/india/karnataka/bengaluru/study-by-bengaluru-traffic-police-reveals-vehicle-movement-exceeds-road-capacity-on-orr-sarjapur-road-3774552 | 2026-10-21 | News article; study itself not obtained `[TODO-VERIFY]` | Iblur–Bellandur PCU vs capacity, peak PCU/hr/lane -> `published_counts.csv` | REAL (reported) | Madhan |
| First-aid corpus | IFRC / Red Cross public guidelines `[TODO: exact URLs]` | [TODO] | [TODO] | Converted to .txt, chunked | REAL docs | Sankalp |
| Text severity reports (EN/HI/KN) | Team-built | [TODO] | Team-owned | LLM-drafted, hand-reviewed, rubric-labelled | SIM | Sankalp |
| Whisper WER recordings | Team-recorded, consented | [TODO] | Team-owned; not committed | Reference transcripts | REAL | Sankalp |
| Hospital locations (17) | OpenStreetMap via overpass-turbo, `data/raw/hospitals_src/bengaluru_hospitals.geojson` (1,061 amenity=hospital points) | OSM snapshot 2026-05-09 | ODbL (© OpenStreetMap contributors; attribution required) | Joined to the curated list by hand-checked OSM id (`scripts/build_hospitals.py` MATCH); 3 of 20 dropped (no confident match) | REAL | Madhan |
| Hospital capability flags (emergency, polytrauma, accident victims, neurosurgery, orthopaedics) | 6th-sem project, `data/raw/hospitals_src/curated_emergency_hospitals.csv` `[TODO-VERIFY: original source per flag]` | [TODO] | Team-compiled | trauma_level / specialties / open_24x7 derived by rule (`scripts/build_hospitals.py`) | REAL attrs `[TODO-VERIFY]` | Madhan |
| Ambulance bases (16) | 6th-sem project, `data/raw/hospitals_src/ambulance_bases_mvp.csv` | [TODO] | Team-compiled | Renamed columns; `location_quality=approximated` (zone points, NOT official 108 base locations) | APPROX | Madhan |
| Beds available, ICU beds, blood stock per hospital | Simulated, seed 42 (`scripts/build_hospitals.py`) | 2026-10-21 | — | beds 2–25, ICU 0–6, blood 0–30 units per group | SIM | Madhan |
| Blood stock | Simulated (eRaktKosh has no open API found) `[TODO-VERIFY]` | [TODO] | — | Per hospital × 8 groups | SIM | Ragavendra |
| SUMO ORR corridor | Real inter-junction distances, generated demand | [TODO] | — | docs/08 §6 demand spec | SIM | Madhan |
| OSM ORR extract | https://www.openstreetmap.org (Overpass export) | [TODO] | ODbL, attribution required | netconvert -> sumo/osm/ | REAL geometry | Madhan |
| Google typical travel times | Google Maps Routes API | [TODO] | Google Maps Platform terms (store only derived stats as allowed `[TODO-VERIFY]`) | per segment × hour × weekday | REAL (predicted typical) | Madhan |
| TomTom live flow | TomTom Traffic Flow API | [TODO] | TomTom terms `[TODO-VERIFY]` | polled every 30 min ≥ 7 days | REAL (observed) | Madhan |
| Junction video counts | Own recordings | [TODO] | Own; raw video not committed, no faces/plates in outputs | YOLOv8 + ByteTrack counts per 5 min | REAL | Madhan |
