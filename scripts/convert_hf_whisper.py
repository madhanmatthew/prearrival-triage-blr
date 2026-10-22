"""Convert a Hugging Face Whisper fine-tune to CTranslate2 for faster-whisper (Indic ASR trial).

    python -m scripts.convert_hf_whisper vasista22/whisper-kannada-medium [--quantization float16]

Writes models/whisper/ct2/<repo name>/ (gitignored, like the other Whisper weights). The HF repo is
first re-saved with transformers so a fast `tokenizer.json` exists (faster-whisper needs it), then
converted with CTranslate2's transformers converter. Candidate models and their licences are listed
in docs/progress/madhan.md (all Apache-2.0, not gated).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

OUT_ROOT = Path("models/whisper/ct2")
HF_CACHE = Path("models/whisper/hf_src")


def convert(repo: str, quantization: str = "float16", force: bool = False) -> Path:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    name = repo.split("/")[-1]
    out = OUT_ROOT / name
    if out.exists() and (out / "model.bin").exists() and not force:
        return out
    src = HF_CACHE / name
    if not (src / "tokenizer.json").exists():
        # download the repo files explicitly: newer transformers does not fetch old *.bin weights
        from huggingface_hub import snapshot_download
        raw = Path(snapshot_download(repo, local_dir=HF_CACHE / f"{name}_raw",
                                     allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors"]))
        proc = WhisperProcessor.from_pretrained(raw)
        model = WhisperForConditionalGeneration.from_pretrained(raw, use_safetensors=any(raw.glob("*.safetensors")))
        src.mkdir(parents=True, exist_ok=True)
        proc.save_pretrained(src)
        model.save_pretrained(src)
        # newer transformers writes processor_config.json; CTranslate2 wants the original file
        if (raw / "preprocessor_config.json").exists():
            shutil.copy(raw / "preprocessor_config.json", src / "preprocessor_config.json")
    raw = HF_CACHE / f"{name}_raw"
    if not (src / "preprocessor_config.json").exists() and (raw / "preprocessor_config.json").exists():
        shutil.copy(raw / "preprocessor_config.json", src / "preprocessor_config.json")
    if out.exists():
        shutil.rmtree(out)
    exe = Path(sys.executable).with_name("ct2-transformers-converter.exe")
    cmd = [str(exe) if exe.exists() else "ct2-transformers-converter", "--model", str(src),
           "--output_dir", str(out), "--quantization", quantization,
           "--copy_files", "tokenizer.json", "preprocessor_config.json"]
    subprocess.run(cmd, check=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="HF Whisper fine-tune -> CTranslate2")
    ap.add_argument("repo")
    ap.add_argument("--quantization", default="float16")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(convert(a.repo, a.quantization, a.force))


if __name__ == "__main__":
    main()
