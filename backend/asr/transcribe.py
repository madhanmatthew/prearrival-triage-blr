"""Local Whisper speech-to-text for the reporting agent (docs/01 §4, docs/15 §4).

faster-whisper (CTranslate2), CPU int8, fully local: no API key, no billing. Config comes
from the environment / `.env` (WHISPER_MODEL = tiny|base|small|medium|large-v3|large-v3-turbo, WHISPER_MODE = local,
WHISPER_REPETITION_PENALTY, WHISPER_NO_REPEAT_NGRAM, WHISPER_DEVICE = cpu|cuda,
WHISPER_COMPUTE = int8|float16|int8_float16),
never hard-coded. The model is downloaded by faster-whisper on first use into
`models/whisper/` (gitignored, not committed).

Decoding is greedy-deterministic (temperature 0, beam 5), so the same audio gives the same
text. Location is never taken from speech (AGENTS rule 5): the transcript only feeds the
dialogue state machine as the initial report.

Demo scope is en / hi / kn; the architecture is language-agnostic. If Whisper detects a
language outside that set the result is flagged `supported=False` and `intake()` raises so the
caller can fall back to text input.

Research prototype, not a medical device.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

SUPPORTED_LANGUAGES = ("en", "hi", "kn")
MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")
MODEL_DIR = Path("models/whisper")


@dataclass
class TranscriptResult:
    text: str
    language: str
    language_probability: float
    duration_s: float

    @property
    def supported(self) -> bool:
        return self.language in SUPPORTED_LANGUAGES


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:  # python-dotenv is optional; plain environment variables still work
        pass


def _add_pip_cuda_dlls() -> None:
    """Windows: make cuBLAS/cuDNN from the pip wheels (nvidia-cublas-cu12, nvidia-cudnn-cu12,
    see requirements-gpu.txt) findable by CTranslate2. No-op elsewhere or if not installed."""
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    for root in getattr(nvidia, "__path__", []):
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def cuda_device_count() -> int:
    _add_pip_cuda_dlls()
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # no CUDA runtime / driver
        return 0


def _default_factory(size: str, device: str, compute_type: str) -> Any:
    if device == "cuda":
        _add_pip_cuda_dlls()
    from faster_whisper import WhisperModel
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return WhisperModel(size, device=device, compute_type=compute_type,
                        download_root=str(MODEL_DIR))


class Transcriber:
    """Loads the model once; `transcribe()` can be called many times.

    `model_factory(size, device, compute_type)` is injectable so tests run without the
    real model or any audio.
    """

    def __init__(self, model_size: Optional[str] = None, mode: Optional[str] = None,
                 device: Optional[str] = None, compute_type: Optional[str] = None, vad_filter: bool = True,
                 repetition_penalty: Optional[float] = None, no_repeat_ngram_size: Optional[int] = None,
                 model_factory: Optional[Callable[[str, str, str], Any]] = None):
        _load_env()
        self.model_size = model_size or os.environ.get("WHISPER_MODEL", "small")
        self.mode = mode or os.environ.get("WHISPER_MODE", "local")
        if self.mode != "local":
            raise NotImplementedError("only WHISPER_MODE=local is implemented (free-tier, docs/15 §4)")
        if self.model_size not in MODEL_SIZES:
            raise ValueError(f"WHISPER_MODEL must be one of {MODEL_SIZES}, got {self.model_size!r}")
        # empty env values (e.g. `WHISPER_COMPUTE=` in .env) fall back to the defaults
        self.device = device or os.environ.get("WHISPER_DEVICE") or "cpu"
        self.compute_type = (compute_type or os.environ.get("WHISPER_COMPUTE")
                             or ("float16" if self.device == "cuda" else "int8"))
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"WHISPER_DEVICE must be cpu or cuda, got {self.device!r}")
        self.vad_filter = vad_filter
        # Repetition guard (deterministic): small Whisper loops on hi/kn ("ಸಾರಿ ಸಾರಿ ...").
        # 1.0 / 0 = faster-whisper defaults (no guard).
        self.repetition_penalty = float(repetition_penalty if repetition_penalty is not None
                                        else os.environ.get("WHISPER_REPETITION_PENALTY") or 1.0)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size if no_repeat_ngram_size is not None
                                        else os.environ.get("WHISPER_NO_REPEAT_NGRAM") or 0)
        self._factory = model_factory or _default_factory
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._factory(self.model_size, self.device, self.compute_type)
        return self._model

    def transcribe(self, audio_path: str | Path, language: Optional[str] = None) -> TranscriptResult:
        """Transcribe one audio file. `language` = en|hi|kn forces the language; None = auto-detect."""
        if language is not None and language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"language must be one of {SUPPORTED_LANGUAGES} or None, got {language!r}")
        if not Path(audio_path).is_file():
            raise FileNotFoundError(audio_path)
        segments, info = self.model.transcribe(
            str(audio_path), language=language, beam_size=5, temperature=0.0,
            vad_filter=self.vad_filter, condition_on_previous_text=False,
            repetition_penalty=self.repetition_penalty, no_repeat_ngram_size=self.no_repeat_ngram_size)
        text = " ".join(s.text.strip() for s in segments).strip()
        return TranscriptResult(text=text, language=info.language,
                                language_probability=float(info.language_probability),
                                duration_s=float(info.duration))
