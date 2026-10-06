# RAG evaluation queries (docs/01 §8)

## File: `queries.csv` (UTF-8)
| column | meaning |
|---|---|
| `query_id` | unique |
| `query` | a bystander-style situation, e.g. "my friend fell off the bike and his arm is bent" |
| `expected_file` | corpus file that should answer it; **empty = out of scope** (must get "not covered") |
| `split` | `dev` (threshold tuning) or `test` (reported once) |
| `notes` | optional |

Aim: >= 40 in-scope queries (every topic, varied wording, some from the dialogue's
`queries_from_report`) + >= 15 out-of-scope ones (snake bite, poisoning, medication
questions, non-medical). ~1/3 dev, 2/3 test. Write the queries **before** looking at results.

## Run
1. `python -m backend.rag.eval_rag --sweep` -> pick `min_similarity` from the dev table.
2. `python -m backend.rag.eval_rag --min-sim <t>` (`make eval-rag ARGS="--min-sim <t>"`)
   -> `reports/rag_eval_queries.csv` + log row. Do not change the threshold after this.
3. Optional LLM mode: set `LLM_PROVIDER=openai_compat`, `LLM_BASE_URL`, `LLM_MODEL`,
   `LLM_API_KEY` in `.env` (free tier only) and re-run step 2. Report both modes.
Also do a manual check of ~20 answers (groundedness by a human reader) for the report.
