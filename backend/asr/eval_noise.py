"""WER vs SNR: how much background noise hurts Whisper, with and without VAD (docs/01 §8).

    python -m backend.asr.eval_noise [--data data/asr_test] [--noise-dir data/asr_test/noise]
        [--snr 20 10 5 0] [--model small] [--denoise] [--quick]

Takes the clean recordings from `reference.csv`, mixes recorded street noise (`*.wav` in
--noise-dir) at each SNR with a fixed seed, transcribes with VAD on and off, and reports
corpus WER/CER per language x SNR x condition to reports/asr_wer_snr.csv. The clean row
(snr = "clean") uses the unmixed audio. Rows are appended to experiments/log.csv.

Mixed clips are SYNTHETIC noise on real speech: label them "synthetic noise" wherever shown.
If --noise-dir has no wav files, generated pink noise is used and the noise_source column says
"generated-pink" (a weak stand-in; collect real street noise for the reported result).
Nothing is trained: Whisper is used as released.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from backend.asr import noise as N
from backend.asr.eval_wer import LOG_PATH, REPORT_DIR, _git_commit, corpus_rates, error_counts, load_reference
from backend.asr.transcribe import Transcriber

SEED = 42


def load_noises(noise_dir: Path, rng: np.random.Generator) -> tuple[list[np.ndarray], str]:
    files = sorted(noise_dir.glob("*.wav")) if noise_dir.is_dir() else []
    if files:
        return [N.load_audio(f) for f in files], "recorded"
    return [N.pink_noise(N.SAMPLE_RATE * 30, rng)], "generated-pink"


def evaluate(data_dir: Path, snrs: list[float], noises: list[np.ndarray], noise_source: str,
             transcribers: dict[str, Transcriber], use_denoise: bool = False,
             limit: int | None = None) -> pd.DataFrame:
    ref = load_reference(data_dir)
    if limit:
        ref = ref.groupby("language").head(limit)
    rng = np.random.default_rng(SEED)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, r in enumerate(ref.itertuples()):
            speech = N.load_audio(data_dir / r.file)
            conditions = [("clean", speech)]
            for snr in snrs:
                noise = noises[int(rng.integers(len(noises)))]
                conditions.append((str(snr), N.mix_at_snr(speech, noise, snr, rng)))
            for snr, audio in conditions:
                if use_denoise and snr != "clean":
                    audio = N.denoise(audio)
                path = Path(tmp) / f"{i}_{snr}.wav"
                N.write_wav(path, audio)
                for cond, t in transcribers.items():
                    res = t.transcribe(path, language=r.language)
                    rows.append({"file": r.file, "language": r.language, "snr_db": snr,
                                 "condition": cond, "noise_source": noise_source,
                                 "hypothesis": res.text, **error_counts(r.reference, res.text)})
    return pd.DataFrame(rows)


def summarize(utt: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (snr, cond, src), g in utt.groupby(["snr_db", "condition", "noise_source"]):
        rates = corpus_rates(g)
        rates.insert(0, "snr_db", snr)
        rates.insert(1, "condition", cond)
        rates.insert(2, "noise_source", src)
        out.append(rates)
    return pd.concat(out, ignore_index=True)


def log_results(rates: pd.DataFrame, t: Transcriber, denoised: bool) -> None:
    today, commit = dt.date.today().isoformat(), _git_commit()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rates.itertuples():
            w.writerow([today, commit, "asr", f"faster-whisper-{t.model_size}",
                        json.dumps({"language": r.language, "snr_db": r.snr_db,
                                    "condition": r.condition, "denoise": denoised,
                                    "device": t.device, "compute": t.compute_type,
                                    "repetition_penalty": t.repetition_penalty,
                                    "no_repeat_ngram_size": t.no_repeat_ngram_size}),
                        "", "", f"{r.wer:.4f}", f"team-recorded+synthetic-noise({r.noise_source})",
                        SEED, f"task=asr_noise metric=WER cer={r.cer:.4f} utterances={r.utterances} "
                        "noise mixed into real speech, synthetic; no training"])


def main() -> None:
    p = argparse.ArgumentParser(description="Whisper WER vs SNR")
    p.add_argument("--data", type=Path, default=Path("data/asr_test"))
    p.add_argument("--noise-dir", type=Path, default=Path("data/asr_test/noise"))
    p.add_argument("--snr", type=float, nargs="+", default=[20, 10, 5, 0])
    p.add_argument("--model", default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--no-repeat-ngram", type=int, default=None)
    p.add_argument("--denoise", action="store_true", help="spectral gating before ASR (noisereduce)")
    p.add_argument("--quick", action="store_true", help="3 utterances per language; not logged")
    p.add_argument("--out-tag", default="", help="suffix for report files, e.g. _large-v3-turbo")
    a = p.parse_args()
    noises, src = load_noises(a.noise_dir, np.random.default_rng(SEED))
    if src != "recorded":
        print(f"WARNING: no wav in {a.noise_dir}; using generated pink noise (weak stand-in)")
    guard = dict(repetition_penalty=a.repetition_penalty, no_repeat_ngram_size=a.no_repeat_ngram)
    ts = {"vad_on": Transcriber(model_size=a.model, vad_filter=True, **guard),
          "vad_off": Transcriber(model_size=a.model, vad_filter=False, **guard)}
    utt = evaluate(a.data, a.snr, noises, src, ts, a.denoise, limit=3 if a.quick else None)
    rates = summarize(utt)
    print(rates.round(4).to_string(index=False))
    REPORT_DIR.mkdir(exist_ok=True)
    suffix = a.out_tag + ("_quick" if a.quick else "")
    utt.to_csv(REPORT_DIR / f"asr_wer_snr_utterances{suffix}.csv", index=False, encoding="utf-8")
    rates.to_csv(REPORT_DIR / f"asr_wer_snr{suffix}.csv", index=False)
    if not a.quick:
        log_results(rates, ts["vad_on"], a.denoise)
        print(f"appended {len(rates)} rows to {LOG_PATH}")


if __name__ == "__main__":
    main()
