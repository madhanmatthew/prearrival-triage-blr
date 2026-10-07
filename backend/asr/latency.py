"""Whisper latency per clip on this machine (decides live intake vs offline-only, docs/15 D18).

    python -m backend.asr.latency [--model large-v3-turbo] [--per-language 10] [--out-tag _cuda]

Device and precision come from WHISPER_DEVICE / WHISPER_COMPUTE (or --device / --compute).
Uses the first N clips per language from data/asr_test/reference.csv, one warm-up call first
(model load + CUDA init are not timed), language forced as in eval_wer. Reports seconds per
clip and real-time factor (RTF = processing time / audio duration; < 1 = faster than real time)
to reports/asr_latency<tag>.csv and one row per language to experiments/log.csv
(test_score = median RTF). Timing is wall-clock on whatever else the machine is doing: close
other heavy programs first.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import time
from pathlib import Path

import pandas as pd

from backend.asr.eval_wer import LOG_PATH, REPORT_DIR, _git_commit, load_reference
from backend.asr.transcribe import Transcriber


def measure(data_dir: Path, t: Transcriber, per_language: int = 10) -> pd.DataFrame:
    ref = load_reference(data_dir).groupby("language").head(per_language)
    first = ref.iloc[0]
    t.transcribe(data_dir / first.file, language=first.language)   # warm-up, not timed
    rows = []
    for r in ref.itertuples():
        start = time.perf_counter()
        res = t.transcribe(data_dir / r.file, language=r.language)
        sec = time.perf_counter() - start
        rows.append({"file": r.file, "language": r.language, "audio_s": res.duration_s,
                     "process_s": sec, "rtf": sec / res.duration_s if res.duration_s else float("nan")})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("language").agg(clips=("file", "count"), audio_s_mean=("audio_s", "mean"),
                                      process_s_median=("process_s", "median"),
                                      process_s_max=("process_s", "max"),
                                      rtf_median=("rtf", "median")).reset_index()


def log_results(s: pd.DataFrame, t: Transcriber) -> None:
    today, commit = dt.date.today().isoformat(), _git_commit()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in s.itertuples():
            w.writerow([today, commit, "asr", f"faster-whisper-{t.model_size}",
                        json.dumps({"language": r.language, "device": t.device, "compute": t.compute_type,
                                    "repetition_penalty": t.repetition_penalty}),
                        "", "", f"{r.rtf_median:.4f}", "team-recorded", "",
                        f"task=asr_latency metric=median_RTF process_s_median={r.process_s_median:.2f} "
                        f"process_s_max={r.process_s_max:.2f} clips={r.clips} host={platform.node()}"])


def main() -> None:
    p = argparse.ArgumentParser(description="Whisper latency per clip")
    p.add_argument("--data", type=Path, default=Path("data/asr_test"))
    p.add_argument("--model", default=None)
    p.add_argument("--device", default=None, help="cpu|cuda (default: WHISPER_DEVICE)")
    p.add_argument("--compute", default=None, help="default: WHISPER_COMPUTE")
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--per-language", type=int, default=10)
    p.add_argument("--out-tag", default="")
    p.add_argument("--no-log", action="store_true", help="smoke test: do not append to experiments/log.csv")
    a = p.parse_args()
    t = Transcriber(model_size=a.model, device=a.device, compute_type=a.compute,
                    repetition_penalty=a.repetition_penalty)
    df = measure(a.data, t, a.per_language)
    s = summarize(df)
    print(f"{t.model_size} on {t.device}/{t.compute_type}")
    print(s.round(3).to_string(index=False))
    REPORT_DIR.mkdir(exist_ok=True)
    df.to_csv(REPORT_DIR / f"asr_latency_clips{a.out_tag}.csv", index=False)
    s.to_csv(REPORT_DIR / f"asr_latency{a.out_tag}.csv", index=False)
    if not a.no_log:
        log_results(s, t)
        print(f"appended {len(s)} rows to {LOG_PATH}")


if __name__ == "__main__":
    main()
