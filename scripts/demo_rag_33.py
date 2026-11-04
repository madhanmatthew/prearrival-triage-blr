"""Demo helper: show that every one of the 33 first-aid corpus topics is retrieved (docs/01 §8).

One query per corpus file lives in `data/rag_eval/demo33.csv` (columns topic_file, lang, query,
source). Each query is run through the same `FirstAidRAG.guide()` the API uses for free text
(no dialogue routing, similarity threshold on). The table shows, per topic:

  top1   the best-scoring chunk comes from the expected corpus file
  hit@4  the expected file is among the top-4 chunks (the metric used in `eval_rag.py`)
  answer the system answered (status "ok") instead of saying "not covered"
  score  similarity of the best chunk (threshold = `--min-sim`, default from the pipeline)

This is a demonstration, not an evaluation: the English queries come from the dev/test rows of
`data/rag_eval/queries.csv` (so they are not fresh test queries), and the Hindi / Kannada rows are
draft translations [TODO-VERIFY native review]. Retrieval is an English model over an English
corpus (BAAI/bge-small-en-v1.5), so hi / kn free text is expected to retrieve poorly; in the
live pipeline Hindi and Kannada reports are routed from the dialogue answers instead (Chapter 8.4).
Quote only the English hit@4 from `eval_rag.py` (experiments/log.csv) in slides.

Run (from the repo root, index built with `make build-rag`):
  python -m scripts.demo_rag_33                       # English, all 33 topics
  python -m scripts.demo_rag_33 --lang all            # en + hi + kn tables
  python -m scripts.demo_rag_33 --only snake_bites --show-steps 4     # one topic, with the steps
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from backend.rag import pipeline as rp

DEMO_PATH = Path("data/rag_eval/demo33.csv")
REPORT_PATH = Path("reports/demo_rag_33.csv")
LANGS = ("en", "hi", "kn")
REQUIRED_COLS = ("topic_file", "lang", "query")


def load_rows(path: Path = DEMO_PATH, langs: tuple[str, ...] = ("en",),
              only: str | None = None) -> list[dict]:
    """Demo queries for the chosen languages (optionally one topic, matched by name prefix)."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(c not in rows[0] for c in REQUIRED_COLS):
        raise ValueError(f"{path} needs columns {REQUIRED_COLS}")
    rows = [r for r in rows if r["lang"] in langs]
    if only:
        rows = [r for r in rows if r["topic_file"].startswith(only)]
    if not rows:
        raise ValueError("no demo rows match the chosen --lang / --only")
    return rows


def run_query(rag: rp.FirstAidRAG, row: dict) -> dict:
    """Free-text guidance for one demo row and the retrieval facts the table needs."""
    g = rag.guide(row["query"])
    files = [r["file"] for r in g.retrieved]
    expected = row["topic_file"]
    return {
        "lang": row["lang"], "topic_file": expected, "query": row["query"],
        "top1_file": files[0] if files else "",
        "top1": bool(files) and files[0] == expected,
        "hit4": expected in files[:4],
        "status": g.status, "refusal_reason": g.refusal_reason or "",
        "score": g.retrieved[0]["score"] if g.retrieved else "",
        "fully_grounded": g.fully_grounded, "n_steps": len(g.sentences),
        "steps": [(s.text, s.chunk_ids) for s in g.sentences],
    }


def summarise(results: list[dict]) -> dict:
    n = len(results)
    ans = [r for r in results if r["status"] == "ok"]
    return {"n": n, "top1": sum(r["top1"] for r in results), "hit4": sum(r["hit4"] for r in results),
            "answered": len(ans), "grounded": sum(bool(r["fully_grounded"]) for r in ans),
            "n_answered": len(ans)}


def _mark(ok: bool) -> str:
    return "yes" if ok else "NO"


def print_table(results: list[dict], show_steps: int = 0) -> None:
    print(f"{'#':>2}  {'topic':<22} {'top1':<5} {'hit@4':<5} {'answer':<7} {'score':<6} "
          f"{'grounded':<8} retrieved-from (top-1)")
    for i, r in enumerate(results, 1):
        topic = r["topic_file"].replace("_ifrc.md", "")
        score = f"{r['score']:.2f}" if r["score"] != "" else "-"
        grounded = "-" if r["status"] != "ok" else _mark(bool(r["fully_grounded"]))
        print(f"{i:>2}  {topic:<22} {_mark(r['top1']):<5} {_mark(r['hit4']):<5} "
              f"{_mark(r['status'] == 'ok'):<7} {score:<6} {grounded:<8} "
              f"{r['top1_file'].replace('_ifrc.md', '') or '-'}"
              + (f"  [{r['refusal_reason']}]" if r["refusal_reason"] else ""))
        if show_steps:
            print(f"      query: {r['query']}")
            for text, ids in r["steps"][:show_steps]:
                print(f"      - {text}  ({', '.join(ids)})")
    s = summarise(results)
    print(f"\n{results[0]['lang']}: top-1 {s['top1']}/{s['n']}, hit@4 {s['hit4']}/{s['n']}, "
          f"answered {s['answered']}/{s['n']}, fully grounded {s['grounded']}/{s['n_answered']}")


def write_report(results: list[dict], path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ("lang", "topic_file", "query", "top1_file", "top1", "hit4", "status",
            "refusal_reason", "score", "fully_grounded", "n_steps")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in results:
            w.writerow([r[c] for c in cols])


def main() -> None:
    p = argparse.ArgumentParser(description="Demo: retrieval over all 33 first-aid topics")
    p.add_argument("--lang", choices=(*LANGS, "all"), default="en")
    p.add_argument("--csv", type=Path, default=DEMO_PATH)
    p.add_argument("--index", type=Path, default=rp.INDEX_DIR)
    p.add_argument("--min-sim", type=float, default=rp.MIN_SIMILARITY)
    p.add_argument("--only", help="one topic (name prefix, e.g. snake_bites)")
    p.add_argument("--show-steps", type=int, default=0, metavar="N",
                   help="print the first N guidance sentences (with chunk ids) per topic")
    p.add_argument("--out", type=Path, default=REPORT_PATH)
    a = p.parse_args()

    langs = LANGS if a.lang == "all" else (a.lang,)
    rows = load_rows(a.csv, langs, a.only)
    # extractive mode (no LLM) and no NLI model: every sentence is copied from a retrieved chunk
    rag = rp.load(a.index, nli=False, min_similarity=a.min_sim)
    results = [run_query(rag, r) for r in rows]
    for lang in langs:
        part = [r for r in results if r["lang"] == lang]
        if part:
            print(f"\n=== {lang} ({rag.embed_model}, min similarity {a.min_sim}) ===")
            print_table(part, a.show_steps)
    write_report(results, a.out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
