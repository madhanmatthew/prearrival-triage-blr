"""One command for the GPU-laptop session (docs/15 D18): latency, clean WER, noise WER.

    python scripts/asr_gpu_day.py [--skip-large] [--noise-large] [--allow-cpu]

Steps (each a separate process, so one model's GPU memory is freed before the next):
  1. latency      large-v3-turbo (float16), large-v3 (int8_float16; fits a 4 GB GPU)
  2. clean WER    both models, all 90 clips
  3. noise WER    large-v3-turbo (all SNRs, VAD on/off); large-v3 too with --noise-large
All use repetition_penalty 1.1 (set a priori, docs/progress/madhan.md 2026-10-07). Every step
appends rows to experiments/log.csv and writes reports/asr_*<tag>.csv. Stops at the first
failing step and prints how to resume (re-run with the finished steps skipped via --from).
Estimated total on an RTX 3050 4 GB: 1-3 h [ASSUMPTION, not measured yet].
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.asr.transcribe import cuda_device_count  # noqa: E402

RP = "1.1"
MODELS = {"large-v3-turbo": "float16", "large-v3": "int8_float16"}


def steps(skip_large: bool, noise_large: bool) -> list[tuple[str, str, list[str]]]:
    models = [m for m in MODELS if not (skip_large and m == "large-v3")]
    out = []
    for m in models:
        out.append((f"latency {m}", m, ["-m", "backend.asr.latency", "--model", m,
                                        "--repetition-penalty", RP, "--out-tag", f"_{m}_cuda"]))
    for m in models:
        out.append((f"clean WER {m}", m, ["-m", "backend.asr.eval_wer", "--model", m,
                                          "--repetition-penalty", RP, "--out-tag", f"_{m}_rp{RP}_cuda"]))
    for m in models:
        if m == "large-v3" and not noise_large:
            continue
        out.append((f"noise WER {m}", m, ["-m", "backend.asr.eval_noise", "--model", m,
                                          "--repetition-penalty", RP, "--out-tag", f"_{m}_rp{RP}_cuda"]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="ASR GPU session: latency + clean WER + noise WER")
    p.add_argument("--skip-large", action="store_true", help="turbo only")
    p.add_argument("--noise-large", action="store_true", help="also run the noise eval on large-v3")
    p.add_argument("--from", dest="start", type=int, default=1, help="resume from step N")
    p.add_argument("--allow-cpu", action="store_true", help="run on CPU if no GPU (very slow)")
    a = p.parse_args()

    n_gpu = cuda_device_count()
    device = "cuda" if n_gpu else "cpu"
    if not n_gpu and not a.allow_cpu:
        sys.exit("No CUDA GPU visible. Check `nvidia-smi`, then `pip install -r requirements-gpu.txt`.\n"
                 "Use --allow-cpu to run anyway (hours).")
    plan = steps(a.skip_large, a.noise_large)
    for i, (name, _, _) in enumerate(plan, 1):
        print(f"  {i}. {name}" + ("   (skipped)" if i < a.start else ""))
    for i, (name, model, args) in enumerate(plan, 1):
        if i < a.start:
            continue
        env = dict(os.environ, WHISPER_DEVICE=device, PYTHONIOENCODING="utf-8",
                   HF_HUB_DISABLE_SYMLINKS_WARNING="1",
                   WHISPER_COMPUTE=MODELS[model] if device == "cuda" else "int8")
        print(f"\n=== step {i}/{len(plan)}: {name} ({device}/{env['WHISPER_COMPUTE']}) {time.strftime('%H:%M:%S')}",
              flush=True)
        t0 = time.time()
        rc = subprocess.run([sys.executable, *args], env=env).returncode
        print(f"--- {name}: {'ok' if rc == 0 else f'FAILED (exit {rc})'} in {(time.time() - t0) / 60:.1f} min",
              flush=True)
        if rc != 0:
            sys.exit(f"Stopped. Fix the error above, then resume: python scripts/asr_gpu_day.py --from {i}")
    print("\nAll steps done. Commit experiments/log.csv and reports/asr_*_cuda*.csv (not audio, not models).")


if __name__ == "__main__":
    main()
