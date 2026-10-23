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

import gc
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
                 device: Optional[str] = None, compute_type: Optional[str] = None, vad_filter: Optional[bool] = None,
                 repetition_penalty: Optional[float] = None, no_repeat_ngram_size: Optional[int] = None,
                 model_factory: Optional[Callable[[str, str, str], Any]] = None):
        _load_env()
        self.model_size = model_size or os.environ.get("WHISPER_MODEL", "small")
        self.mode = mode or os.environ.get("WHISPER_MODE", "local")
        if self.mode != "local":
            raise NotImplementedError("only WHISPER_MODE=local is implemented (free-tier, docs/15 §4)")
        # a local CTranslate2 folder (e.g. models/whisper/ct2/whisper-kannada-medium, made by
        # scripts/convert_hf_whisper.py) is accepted as well as the stock sizes
        if self.model_size not in MODEL_SIZES and not (Path(self.model_size) / "model.bin").exists():
            raise ValueError(f"WHISPER_MODEL must be one of {MODEL_SIZES} or a CTranslate2 model folder, "
                             f"got {self.model_size!r}")
        # empty env values (e.g. `WHISPER_COMPUTE=` in .env) fall back to the defaults
        self.device = device or os.environ.get("WHISPER_DEVICE") or "cpu"
        self.compute_type = (compute_type or os.environ.get("WHISPER_COMPUTE")
                             or ("float16" if self.device == "cuda" else "int8"))
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"WHISPER_DEVICE must be cpu or cuda, got {self.device!r}")
        # VAD trims silence; it cut the ends of Kannada sentences with the vasista22 fine-tune
        # (2026-10-22), so it is configurable: WHISPER_VAD=0 turns it off. Default stays on.
        self.vad_filter = (vad_filter if vad_filter is not None
                           else os.environ.get("WHISPER_VAD", "1").strip().lower() not in ("0", "false", "no"))
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


class LanguageRouter:
    """Per-language ASR models behind the Transcriber interface (Indic ASR trial, 2026-10-22).

    `WHISPER_MODEL_<LANG>` (e.g. WHISPER_MODEL_KN=models/whisper/ct2/whisper-kannada-medium) picks a
    model for that language; `WHISPER_VAD_<LANG>` overrides the VAD setting for it (the vasista22
    Kannada fine-tune needs WHISPER_VAD_KN=0, its sentence endings were cut otherwise). Languages
    without an override, and auto-detect (language=None), use the default Transcriber. Models load
    lazily, so an override costs GPU memory only once that language is used.
    """

    def __init__(self, default: "Transcriber | None" = None,
                 overrides: Optional[dict[str, "Transcriber"]] = None):
        self.default = default or Transcriber()
        self.overrides = dict(overrides or {})

    @classmethod
    def from_env(cls, model_factory: Optional[Callable[[str, str, str], Any]] = None) -> "LanguageRouter":
        _load_env()
        default = Transcriber(model_factory=model_factory)
        overrides = {}
        for lang in SUPPORTED_LANGUAGES:
            path = (os.environ.get(f"WHISPER_MODEL_{lang.upper()}") or "").strip()
            if not path:
                continue
            vad_env = os.environ.get(f"WHISPER_VAD_{lang.upper()}")
            vad = None if vad_env is None else vad_env.strip().lower() not in ("0", "false", "no")
            overrides[lang] = Transcriber(model_size=path, device=default.device, vad_filter=vad,
                                          repetition_penalty=default.repetition_penalty,
                                          no_repeat_ngram_size=default.no_repeat_ngram_size,
                                          compute_type=os.environ.get(f"WHISPER_COMPUTE_{lang.upper()}") or None,
                                          model_factory=model_factory)
        return cls(default, overrides)

    @property
    def model_size(self) -> str:
        extra = ", ".join(f"{k}={Path(v.model_size).name}" for k, v in self.overrides.items())
        return self.default.model_size + (f" ({extra})" if extra else "")

    def for_language(self, language: Optional[str]) -> "Transcriber":
        return self.overrides.get(language, self.default) if language else self.default

    def transcribe(self, audio_path: str | Path, language: Optional[str] = None) -> TranscriptResult:
        target = self.for_language(language)
        # A 4 GB GPU holds one model (turbo + hi + kn mediums ran out of memory, 2026-10-23):
        # on cuda, unload the other models before switching language (reload costs a few seconds).
        if target.device == "cuda":
            for t in (self.default, *self.overrides.values()):
                if t is not target:
                    t._model = None
            gc.collect()
        return target.transcribe(audio_path, language=language)
