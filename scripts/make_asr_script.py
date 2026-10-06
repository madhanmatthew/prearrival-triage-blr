"""Pick sentences to read aloud for the ASR test set -> data/asr_test/reference.csv + recording_sheet.md.

    python -m scripts.make_asr_script [--per-language 30]

Source: data/text_reports/reports.csv. Only native-script rows are used (romanised rows are
skipped: they have no unambiguous spelling to score against), 40-140 characters (about 5-15 s
spoken), spread across the 4 severity labels. Reviewed rows are preferred; drafts are used only if
there are not enough reviewed ones. Read each sentence exactly as printed, save it as the file
named in the sheet, and the reference is correct by construction. Read-aloud speech is easier than
spontaneous speech: state this limitation with the WER numbers.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

SRC = Path("data/text_reports/reports.csv")
OUT_DIR = Path("data/asr_test")
LANGS = ("en", "hi", "kn")
SEED = 42


def pick(df: pd.DataFrame, lang: str, n: int, seed: int = SEED) -> pd.DataFrame:
    d = df[(df.language == lang) & ~df.notes.fillna("").str.contains("script=roman")]
    d = d[d.text.str.len().between(40, 140)]
    rng = random.Random(seed)
    out = []
    for tier in (d[d.reviewed == 1], d[d.reviewed != 1]):     # reviewed first, then drafts
        pool = {k: g.sample(frac=1, random_state=seed) for k, g in tier.groupby("label")}
        while len(out) < n and any(len(g) for g in pool.values()):
            for k in sorted(pool, key=lambda _: rng.random()):
                if len(out) < n and len(pool[k]):
                    out.append(pool[k].iloc[0])
                    pool[k] = pool[k].iloc[1:]
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Make the ASR read-aloud script")
    p.add_argument("--src", type=Path, default=SRC)
    p.add_argument("--per-language", type=int, default=30)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    ref_path = OUT_DIR / "reference.csv"
    if ref_path.exists() and not a.force:
        raise SystemExit(f"{ref_path} exists; use --force to overwrite")
    df = pd.read_csv(a.src, dtype={"report_id": str}, encoding="utf-8-sig")
    rows = []
    for lang in LANGS:
        sel = pick(df, lang, a.per_language)
        if len(sel) < a.per_language:
            print(f"WARNING: only {len(sel)} usable sentences for {lang}")
        for i, r in enumerate(sel.itertuples(), 1):
            rows.append({"file": f"{lang}_{i:03d}.wav", "language": lang, "reference": r.text,
                         "reviewed": int(r.reviewed)})
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out[["file", "language", "reference"]].to_csv(ref_path, index=False, encoding="utf-8")
    lines = ["# ASR recording sheet", "",
             "Record each line exactly as written (mono, quiet room; some with street noise).",
             "Save as the file name shown, into `data/asr_test/`. Get consent from any other speaker.", ""]
    for lang in LANGS:
        lines += [f"## {lang}", ""]
        lines += [f"- [ ] `{r.file}` {'(draft text)' if not r.reviewed else ''}  \n  {r.reference}"
                  for r in out[out.language == lang].itertuples()]
        lines.append("")
    (OUT_DIR / "recording_sheet.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(out)} rows to {ref_path} and recording_sheet.md "
          f"({int((out.reviewed == 0).sum())} from unreviewed drafts)")


if __name__ == "__main__":
    main()
