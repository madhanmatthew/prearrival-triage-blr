"""Shared test setup: tests never read the developer's real `.env` ASR settings.

Without this, a local `.env` (e.g. WHISPER_DEVICE=cuda, WHISPER_MODEL_KN=...) leaks into the ASR tests
through `backend.asr.transcribe._load_env()` and the expected CPU defaults no longer hold.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_whisper_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("WHISPER_"):
            monkeypatch.delenv(k, raising=False)
    try:
        from backend.asr import transcribe
    except ImportError:
        return
    monkeypatch.setattr(transcribe, "_load_env", lambda: None)
