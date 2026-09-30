"""Word / character error rate of the Whisper wrapper on team-recorded utterances (docs/01 §8).

Test set layout (recorded by the team, >= 30 per language, docs/08 §1):
    data/asr_test/reference.csv     columns: file,language,reference   (file relative to the CSV)
    data/asr_test/<file>.wav

    python -m backend.asr.eval_wer [--data data/asr_test] [--model small] [--quick]

Reports corpus-level WER and CER per language (sum of edit errors / sum of reference length,
not a mean of per-utterance rates) to reports/asr_wer.csv, per-utterance rows to
reports/asr_wer_utterances.csv, and one row per language to experiments/log.csv (test_score =
WER). The data are real recordings; nothing here is synthetic. Kannada/Hindi WER also depends
on spelling conventions in the reference, so write references the way Whisper would spell
common words or report CER alongside WER.

[ASSUMPTION] Text normalisation before scoring: Unicode NFC, lowercase, punctuation removed,
whitespace collapsed. Numerals are not normalised (write references with the same convention).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import unicodedata
from pathlib import Path

import pandas as pd

from backend.asr.transcribe import SUPPORTED_LANGUAGES, Transcriber

LOG_PATH = Path("experiments/log.csv")
REPORT_DIR = Path("reports")


def normalize(text: str) -> str:
    """NFC, lowercase, drop punctuation; keep letters, digits and Indic combining marks."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(ch if (ch.isalnum() or unicodedata.category(ch).startswith("M")) else " "
                   for ch in text)
    return " ".join(text.split())


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance (substitution, insertion, deletion each cost 1)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_counts(reference: str, hypothesis: str) -> dict:
    ref, hyp = normalize(reference), normalize(hypothesis)
    rw, hw = ref.split(), hyp.split()
    return {"word_errors": edit_distance(rw, hw), "ref_words": len(rw),
            "char_errors": edit_distance(list(ref), list(hyp)), "ref_chars": len(ref)}


def corpus_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-language corpus WER/CER from per-utterance error counts."""
    g = df.groupby("language").agg(utterances=("file", "count"),
                                   word_errors=("word_errors", "sum"), ref_words=("ref_words", "sum"),
                                   char_errors=("char_errors", "sum"), ref_chars=("ref_chars", "sum"))
    g["wer"] = g.word_errors / g.ref_words
    g["cer"] = g.char_errors / g.ref_chars
    return g.reset_index()


def load_reference(data_dir: Path) -> pd.DataFrame:
    ref = pd.read_csv(data_dir / "reference.csv", encoding="utf-8", dtype=str, keep_default_na=False)
    missing = {"file", "language", "reference"} - set(ref.columns)
    if missing:
        raise ValueError(f"reference.csv is missing columns {sorted(missing)}")
    bad = sorted(set(ref.language) - set(SUPPORTED_LANGUAGES))
    if bad:
        raise ValueError(f"unsupported languages in reference.csv: {bad}")
    if (ref.reference.str.strip() == "").any():
        raise ValueError("reference.csv has empty reference transcripts")
    return ref


def evaluate(data_dir: Path, transcriber: Transcriber, limit: int | None = None) -> pd.DataFrame:
    ref = load_reference(data_dir)
    if limit:
        ref = ref.groupby("language").head(limit)
    rows = []
    for r in ref.itertuples():
        # language is forced to the recording's true language so WER measures recognition only
        res = transcriber.transcribe(data_dir / r.file, language=r.language)
        rows.append({"file": r.file, "language": r.language, "reference": r.reference,
                     "hypothesis": res.text, **error_counts(r.reference, res.text)})
    return pd.DataFrame(rows)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_results(rates: pd.DataFrame, model_size: str) -> None:
    today, commit = dt.date.today().isoformat(), _git_commit()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rates.itertuples():
            w.writerow([today, commit, "asr", f"faster-whisper-{model_size}",
                        json.dumps({"language": r.language, "beam_size": 5, "compute": "int8"}),
                        "", "", f"{r.wer:.4f}", "team-recorded", 42,
                        f"task=asr metric=WER cer={r.cer:.4f} utterances={r.utterances} "
                        "language forced to true language, real recordings"])


def main() -> None:
    p = argparse.ArgumentParser(description="Whisper WER per language")
    p.add_argument("--data", type=Path, default=Path("data/asr_test"))
    p.add_argument("--model", default=None, help="tiny|base|small (default: WHISPER_MODEL)")
    p.add_argument("--quick", action="store_true", help="3 utterances per language; not logged")
    a = p.parse_args()
    t = Transcriber(model_size=a.model)
    utt = evaluate(a.data, t, limit=3 if a.quick else None)
    rates = corpus_rates(utt)
    print(rates.round(4).to_string(index=False))
    REPORT_DIR.mkdir(exist_ok=True)
    suffix = "_quick" if a.quick else ""
    utt.to_csv(REPORT_DIR / f"asr_wer_utterances{suffix}.csv", index=False, encoding="utf-8")
    rates.to_csv(REPORT_DIR / f"asr_wer{suffix}.csv", index=False)
    if not a.quick:
        log_results(rates, t.model_size)
        print(f"appended {len(rates)} rows to {LOG_PATH}")
        if (rates.utterances < 30).any():
            print("WARNING: fewer than 30 utterances for some language (docs/08 §1 asks for >= 30)")


if __name__ == "__main__":
    main()
