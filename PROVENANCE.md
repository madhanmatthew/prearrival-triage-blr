# Data provenance

Every external dataset or document used by the project: where it came from, licence, and how
it was processed. Raw files live in `data/raw/` (not committed).

## First-aid corpus (RAG, `data/first_aid_corpus/`)

| Field | Value |
|---|---|
| Source | IFRC, *International first aid, resuscitation, and education guidelines 2020* |
| URL | https://www.ifrc.org/sites/default/files/2022-02/EN_GFARC_GUIDELINES_2020.pdf |
| Landing page | https://www.ifrc.org/document/international-first-aid-resuscitation-and-education-guidelines |
| Raw file | `data/raw/first_aid/EN_GFARC_GUIDELINES_2020.pdf` (4,432,256 bytes, 476 pages) |
| Downloaded | 2026-10-06 |
| Licence | (c) IFRC 2020. "Copies of all or part of this study may be made for non-commercial use, providing the source is acknowledged." (PDF p.2) |
| Processing | `python scripts/extract_ifrc_corpus.py`: 18 topics (pages in the script's `TOPICS`); keeps only Key action / Early recognition / First aid steps / NOTE text, verbatim; drops medication/dosage sentences (AGENTS rule 6), dispatcher- and trainer-facing sentences, and repeated layout blocks. Hand review pending. |
| Not used | WHO-ICRC Basic Emergency Care (clinician-level, includes drug steps). |
