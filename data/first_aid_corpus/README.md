# First-aid corpus (docs/01 §6.4, §7; AGENTS rule 6)

Guidance is copied or generated **only** from these files. Nothing here may be written from
memory or by an LLM: every file is text taken from a public guideline document.

## Scope (docs/01 §6.4)
Bleeding control, unresponsive person / CPR basics, choking, fractures / immobilisation,
shock, burns, road-accident scene safety, chest pain / collapse. Nothing else.

## Sources
WHO, Red Cross / Red Crescent (incl. Indian Red Cross), IFRC First Aid Guidelines, St John
Ambulance, Indian government first-aid material. Only documents that are publicly published
and allow academic reuse; record the licence. Do not scrape paywalled or unclear-rights content.

## File format: one `.md` or `.txt` per topic per source (UTF-8)
```
---
title: <document title>
publisher: <organisation>
url: <link to the exact document>
license: <licence or "public guideline, academic use">
topic: bleeding            # bleeding|cpr|choking|fracture|shock|burns|scene_safety|chest_pain
accessed: 2026-10-05
---
Short steps, one instruction per line, copied from the source.
```
Rules:
- One instruction per line (chunks then hold whole steps).
- **Leave out** any medication or dosage step (rule 6); the pipeline also filters them.
- File names: `<topic>_<publisher>.md`, e.g. `bleeding_ifrc.md`.
- Add each file to `PROVENANCE.md`.

## Build
`python -m backend.rag.pipeline build` (`make build-rag`) -> `models/rag/faiss_index/` (not committed).
Try one query: `python -m backend.rag.pipeline ask "heavy bleeding from the leg"`.
