# Dataset Provenance Log

One entry per dataset (docs/08 §8). Never commit restricted data; record where it came from.

| Dataset | Source URL | Access date | Licence / terms | Preprocessing | Real/Sim | Used by |
|---|---|---|---|---|---|---|
| MIMIC-IV Clinical Database Demo v2.2 | https://physionet.org/content/mimic-iv-demo/2.2/ | [TODO] | PhysioNet terms `[TODO-VERIFY]`; do not redistribute | Hourly windows + outcome labels (docs/08 §2) | REAL (ICU) | Chetan |
| First-aid corpus | IFRC / Red Cross public guidelines `[TODO: exact URLs]` | [TODO] | [TODO] | Converted to .txt, chunked | REAL docs | Sankalp |
| Text severity reports (EN/HI/KN) | Team-built | [TODO] | Team-owned | LLM-drafted, hand-reviewed, rubric-labelled | SIM | Sankalp |
| Whisper WER recordings | Team-recorded, consented | [TODO] | Team-owned; not committed | Reference transcripts | REAL | Sankalp |
| Hospitals + ambulance bases | From MVP `[TODO: original sources]` | [TODO] | [TODO] | Extended to docs/08 §7.1 columns | REAL attrs + SIM beds/blood | Ragavendra |
| Blood stock | Simulated (eRaktKosh has no open API found) `[TODO-VERIFY]` | [TODO] | — | Per hospital × 8 groups | SIM | Ragavendra |
| SUMO ORR corridor | Real inter-junction distances, generated demand | [TODO] | — | docs/08 §6 demand spec | SIM | Madhan |
| OSM ORR extract | https://www.openstreetmap.org (Overpass export) | [TODO] | ODbL, attribution required | netconvert -> sumo/osm/ | REAL geometry | Madhan |
| Google typical travel times | Google Maps Routes API | [TODO] | Google Maps Platform terms (store only derived stats as allowed `[TODO-VERIFY]`) | per segment × hour × weekday | REAL (predicted typical) | Madhan |
| TomTom live flow | TomTom Traffic Flow API | [TODO] | TomTom terms `[TODO-VERIFY]` | polled every 30 min ≥ 7 days | REAL (observed) | Madhan |
| Junction video counts | Own recordings | [TODO] | Own; raw video not committed, no faces/plates in outputs | YOLOv8 + ByteTrack counts per 5 min | REAL | Madhan |
