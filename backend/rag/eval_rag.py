"""RAG evaluation (docs/01 §8, docs/12 §01): retrieval hit@k, refusal, groundedness.

Query set: `data/rag_eval/queries.csv` (format in `data/rag_eval/README.md`).
  * in-scope rows name the corpus file that should answer them (`expected_file`);
  * out-of-scope rows leave it empty: the system must answer "not covered".
  * `split` = dev | test. The similarity threshold is chosen on dev only (`--sweep`);
    the reported numbers come from test rows with the chosen threshold, scored once.

Metrics (test split):
  hit@k              in-scope: expected file among the top-k retrieved chunks
  answer_rate        in-scope: status "ok" (1 - false refusal rate)
  refusal_rate       out-of-scope: status "not_covered"
  pct_fully_grounded answered in-scope: every generated sentence entailed (before dropping)
  mean_groundedness  mean per-answer ratio
  medication_removed sentences removed by the medication rule
Writes `reports/rag_eval_queries.csv` + one row in `experiments/log.csv`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.ml.baselines import _git_commit
from backend.rag import pipeline as rp

QUERIES_PATH = Path("data/rag_eval/queries.csv")
REPORT_PATH = Path("reports/rag_eval_queries.csv")
LOG_PATH = Path("experiments/log.csv")
REQUIRED_COLS = ("query_id", "query", "expected_file", "split")
SWEEP_GRID = [round(x, 2) for x in np.arange(0.20, 0.61, 0.05)]


def load_queries(path: Path = QUERIES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if not df["split"].isin(["dev", "test"]).all():
        raise ValueError("split must be dev or test")
    if df["query_id"].duplicated().any():
        raise ValueError("duplicate query_id")
    df["in_scope"] = df["expected_file"].str.strip() != ""
    return df


def sweep_threshold(rag: rp.FirstAidRAG, dev: pd.DataFrame, grid=SWEEP_GRID) -> pd.DataFrame:
    """Retrieval-only sweep on dev rows: balanced accuracy of answer vs refuse per threshold."""
    top = {r.query_id: (-1.0 if rp.query_refusal_reason(r.query)       # guard refuses first
                        else max((s for _, s in rag.retrieve(r.query)), default=-1.0))
           for r in dev.itertuples()}
    ins_hit = [r.expected_file in {d.metadata["file"] for d, _ in rag.retrieve(r.query)}
               for r in dev[dev.in_scope].itertuples()]
    print(f"dev hit@{rag.k} (in-scope, threshold-free): {np.mean(ins_hit):.3f}")
    rows = []
    for t in grid:
        ins = [top[q] >= t for q in dev.loc[dev.in_scope, "query_id"]]
        outs = [top[q] < t for q in dev.loc[~dev.in_scope, "query_id"]]
        ar = float(np.mean(ins)) if ins else np.nan
        rr = float(np.mean(outs)) if outs else np.nan
        rows.append({"min_similarity": t, "answer_rate": ar, "refusal_rate": rr,
                     "balanced_acc": np.nanmean([ar, rr])})
    return pd.DataFrame(rows)


def evaluate(rag: rp.FirstAidRAG, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for r in df.itertuples():
        g = rag.guide(r.query)
        files = [x["file"] for x in g.retrieved]
        rows.append({
            "query_id": r.query_id, "split": r.split, "in_scope": r.in_scope,
            "expected_file": r.expected_file, "status": g.status,
            "refusal_reason": g.refusal_reason,
            "hit_at_k": (r.expected_file in files) if r.in_scope else np.nan,
            "top_score": g.retrieved[0]["score"] if g.retrieved else np.nan,
            "groundedness": g.groundedness, "fully_grounded": g.fully_grounded,
            "n_sentences": len(g.sentences), "n_dropped": len(g.dropped_ungrounded),
            "n_medication_removed": len(g.removed_medication),
            "chunk_ids": json.dumps(sorted({c for s in g.sentences for c in s.chunk_ids})),
            "message": g.message,
        })
    out = pd.DataFrame(rows)
    ins, oos = out[out.in_scope], out[~out.in_scope]
    answered = ins[ins.status == "ok"]
    summary = {
        "n_in_scope": int(len(ins)), "n_out_of_scope": int(len(oos)),
        "hit_at_k": float(ins.hit_at_k.astype(float).mean()) if len(ins) else np.nan,
        "answer_rate": float((ins.status == "ok").mean()) if len(ins) else np.nan,
        "refusal_rate": float((oos.status == "not_covered").mean()) if len(oos) else np.nan,
        "pct_fully_grounded": float(answered.fully_grounded.astype(float).mean())
        if len(answered) else np.nan,
        "mean_groundedness": float(answered.groundedness.astype(float).mean())
        if len(answered) else np.nan,
        "medication_removed": int(out.n_medication_removed.sum()),
        "llm_unavailable": int((out.status == "llm_unavailable").sum()),
        "guard_refusals_in_scope": int(ins.refusal_reason.isin(
            ["medication", "non_medical", "excluded_topic"]).sum()),
    }
    return out, summary


def _log(rag: rp.FirstAidRAG, summary: dict, n_queries: int) -> None:
    params = {"mode": rag.mode, "k": rag.k, "min_similarity": rag.min_similarity,
              "embed": rag.embed_model, "version": "v3", "nli": rp.NLI_MODEL if rag.entail else None,
              "entail_threshold": rp.ENTAIL_THRESHOLD}
    notes = (f"task=rag_groundedness split=test queries={n_queries} "
             + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in summary.items()))
    row = {"date": dt.date.today().isoformat(), "git_commit": _git_commit(),
           "component": "reporting_rag", "model": f"rag_{rag.mode}",
           "params_json": json.dumps(params), "cv_mean": "", "cv_std": "",
           "test_score": round(summary["pct_fully_grounded"], 4), "data_version": "first-aid-corpus",
           "seed": "", "notes": notes}
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=False, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate RAG retrieval + groundedness")
    p.add_argument("--queries", type=Path, default=QUERIES_PATH)
    p.add_argument("--index", type=Path, default=rp.INDEX_DIR)
    p.add_argument("--sweep", action="store_true", help="threshold sweep on dev rows, then exit")
    p.add_argument("--min-sim", type=float, default=rp.MIN_SIMILARITY)
    p.add_argument("--no-nli", action="store_true", help="verbatim check only (extractive mode)")
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()

    df = load_queries(a.queries)
    rag = rp.load(a.index, nli=not a.no_nli and not a.sweep, min_similarity=a.min_sim)
    if a.sweep:
        print(sweep_threshold(rag, df[df.split == "dev"]).to_string(index=False))
        print("Pick min_similarity from dev only, then run without --sweep using --min-sim.")
        return
    if rag.mode == "llm" and rag.entail is None:
        raise SystemExit("LLM mode needs the NLI judge (drop --no-nli)")
    test = df[df.split == "test"]
    per_q, summary = evaluate(rag, test)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    per_q.to_csv(REPORT_PATH, index=False)
    print(json.dumps(summary, indent=2))
    if not a.no_log:
        _log(rag, summary, len(test))


if __name__ == "__main__":
    main()
