# Progress — Sankalp (Component 1: Reporting — Whisper, dialogue, text classifier, RAG)

## Role
- Owns: Whisper ASR + WER eval, dialogue state machine (6 bounded questions), text severity
  classifier (TF-IDF char n-grams + LR), RAG first-aid pipeline + groundedness eval
  (docs/01, 07 §2.1 + §3, 08 §4, 15 D2).

## Do now (no code needed)
1. **Record the WER test set:** ≥30 short emergency-style utterances per language (en/hi/kn),
   consenting team voices only, with exact reference transcripts in a CSV
   (`file, language, reference`). Keep audio on Drive, not git.
2. **Collect the first-aid corpus:** public IFRC / Red Cross first-aid guideline documents,
   scoped to bleeding, CPR, fractures, shock, burns (20–60 pages). Convert to `.txt` into
   `data/first_aid_corpus/`. Record every source URL + licence in `data/PROVENANCE.md`.
3. **Write the text-severity rubric** (docs/08 §4) as a one-page doc, then start drafting
   report templates (injury type × severity × phrasing). Hindi/Kannada need native review.
4. Write 30–50 first-aid test queries with the expected source document for each (RAG eval set).

## Done
- 2026-10-27 (Madhan + Claude Code): RAG v3. 15 more IFRC 2020 topics extracted (`scripts/extract_ifrc_corpus.py`,
  per-topic layout fixes in `TOPIC_RULES`), query guard now refuses only topics with no corpus file (electric shock,
  nosebleed, drowning, croup, eye injury, frostbite, overdose, jellyfish). 12 old out-of-scope eval queries relabelled
  in scope, 36 new v3 test queries written before the run. Threshold kept at 0.55 (dev sweep). v3 test: hit@4 0.951,
  answer 0.984, refusal 0.600, grounded 1.00. Misses: 'barking cough' and 'blood from her nose' not refused; headache /
  hiccups / earache / grief still answered (as in v2). TO REVIEW before the viva: asthma file keeps 'help the person to
  use their inhaler' and diabetic file keeps '15-20 grams of glucose' (rule 6 judgement).

## Next
- Week 2: run `make eval-asr` once the Whisper wrapper exists.

## Blockers
- LLM provider for dialogue phrasing / RAG generation not chosen yet (docs/14 D4).
