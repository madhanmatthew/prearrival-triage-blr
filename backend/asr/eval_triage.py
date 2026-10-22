"""Task-level ASR robustness: do Whisper errors change what the dialogue reads from a report?

Input: a per-utterance WER file (`reports/asr_wer_utterances*.csv`: file, language, reference,
hypothesis). For every clip the dialogue manager reads the initial report twice, from the
script (reference) and from the Whisper transcript (hypothesis), and we compare the fields it
extracts: injury type, and the cues that skip questions (unconscious, breathing problem, heavy
bleeding, high-energy mechanism, number injured).

Two agreement rates per language and field:
  * asr   : hypothesis vs the same language's reference (pure ASR effect);
  * vs_en : hypothesis vs the English reference of the same scenario number (en_001 = hi_001 =
            kn_001 are translations of one script), i.e. ASR + language/lexicon gap together.
"ref_vs_en" (reference vs English reference) isolates the lexicon gap alone.

These are agreements with the rule-based extraction from a script, NOT accuracy against
clinical labels. [ASSUMPTION] the hi/kn cue variants were written after looking at these same
90 clips, so the numbers are dev figures. The held-out set (data/asr_test/heldout/, h01-h10,
new unscripted recordings, references human-verified) gives the clean figure: pass --heldout.

Usage: python -m backend.asr.eval_triage --utterances reports/asr_wer_utterances_large-v3_rp1.1.csv
       python -m backend.asr.eval_triage --utterances reports/asr_wer_utterances_heldout.csv            --heldout --out-tag _heldout
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd

from backend.asr.eval_wer import LOG_PATH, _git_commit
from backend.dialogue.state_machine import DialogueManager

FIELDS = ("injury_type", "unconscious", "breathing_problem", "heavy_bleeding",
          "high_energy", "num_injured")
REPORT_DIR = Path("reports")


def extract(text: str, language: str) -> dict:
    """Fields the dialogue manager reads from an initial report (no questions answered)."""
    dm = DialogueManager(language=language)
    dm.start(text if isinstance(text, str) else "")
    a = dm.answers
    return {"injury_type": dm.injury_type,
            "unconscious": "Q_CONSCIOUS" in a,
            "breathing_problem": "Q_BREATHING" in a,
            "heavy_bleeding": "Q_BLEEDING" in a,
            "high_energy": "Q_MECHANISM" in a,
            "num_injured": a["Q_NUM_INJURED"].count if "Q_NUM_INJURED" in a else None}


def scenario_id(file: str) -> str:
    m = re.search(r"_(h?\d+)\.\w+$", file)
    return m.group(1) if m else file


def compare(utt: pd.DataFrame) -> pd.DataFrame:
    """One row per clip and field: ref, hyp and English-reference values + agreement flags."""
    en_ref = {scenario_id(r.file): extract(r.reference, "en")
              for r in utt.itertuples() if r.language == "en"}
    rows = []
    for r in utt.itertuples():
        sid = scenario_id(r.file)
        ref, hyp, en = extract(r.reference, r.language), extract(r.hypothesis, r.language), \
            en_ref.get(sid)
        for f in FIELDS:
            rows.append({"file": r.file, "language": r.language, "scenario": sid, "field": f,
                         "ref": ref[f], "hyp": hyp[f], "en_ref": None if en is None else en[f],
                         "asr": ref[f] == hyp[f],
                         "ref_vs_en": None if en is None else ref[f] == en[f],
                         "vs_en": None if en is None else hyp[f] == en[f]})
    return pd.DataFrame(rows)


def summarize(cmp: pd.DataFrame) -> pd.DataFrame:
    """Agreement rate per language and field, plus an 'all_fields' row (every field agrees)."""
    out = []
    for lang, g in cmp.groupby("language", sort=False):
        for f, gf in g.groupby("field", sort=False):
            out.append({"language": lang, "field": f, "clips": len(gf),
                        **{k: gf[k].astype(float).mean() for k in ("asr", "ref_vs_en", "vs_en")}})
        per_clip = g.groupby("file")[["asr", "ref_vs_en", "vs_en"]].agg(
            lambda s: bool(s.astype(bool).all()))
        out.append({"language": lang, "field": "all_fields", "clips": len(per_clip),
                    **{k: per_clip[k].astype(float).mean() for k in ("asr", "ref_vs_en", "vs_en")}})
    return pd.DataFrame(out)


def log_results(summary: pd.DataFrame, source: str, heldout: bool = False) -> None:
    split = ("HELD-OUT: new recordings, lexicon not tuned on them" if heldout
             else "DEV: lexicon written on these clips")
    today, commit = dt.date.today().isoformat(), _git_commit()
    allf = summary[summary.field == "all_fields"]
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in allf.itertuples():
            w.writerow([today, commit, "asr_triage", "dialogue_cues",
                        json.dumps({"language": r.language, "utterances": Path(source).name}),
                        "", "", f"{r.vs_en:.4f}", "team-recorded", 42,
                        f"task=asr_triage metric=all-field agreement hyp vs en-reference; "
                        f"asr={r.asr:.4f} ref_vs_en={r.ref_vs_en:.4f} clips={r.clips} "
                        f"{split}"])


def main() -> None:
    p = argparse.ArgumentParser(description="ASR -> dialogue cue robustness")
    p.add_argument("--utterances", type=Path,
                   default=REPORT_DIR / "asr_wer_utterances_large-v3_rp1.1.csv")
    p.add_argument("--out-tag", default="", help="suffix for report files")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--heldout", action="store_true", help="label the log row as held-out, not dev")
    a = p.parse_args()
    utt = pd.read_csv(a.utterances, encoding="utf-8")
    cmp = compare(utt)
    summary = summarize(cmp)
    print(summary.round(3).to_string(index=False))
    REPORT_DIR.mkdir(exist_ok=True)
    cmp.to_csv(REPORT_DIR / f"asr_triage_fields{a.out_tag}.csv", index=False, encoding="utf-8")
    summary.to_csv(REPORT_DIR / f"asr_triage_summary{a.out_tag}.csv", index=False)
    if not a.no_log:
        log_results(summary, str(a.utterances), a.heldout)
        print(f"appended {summary.field.eq('all_fields').sum()} rows to {LOG_PATH}")


if __name__ == "__main__":
    main()
