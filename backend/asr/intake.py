"""Voice -> dialogue hand-off (docs/01 §3): transcript becomes the initial report.

    dm, result = intake("report.wav", transcriber)     # then dm.next_question() ...

GPS is not read from the audio or the transcript (AGENTS rule 5); the caller passes the device
GPS to `DialogueManager.to_incident_report()` later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from backend.asr.transcribe import SUPPORTED_LANGUAGES, Transcriber, TranscriptResult
from backend.dialogue.state_machine import DialogueManager


def intake(audio_path: str | Path, transcriber: Transcriber,
           language: Optional[str] = None) -> tuple[DialogueManager, TranscriptResult]:
    result = transcriber.transcribe(audio_path, language=language)
    if not result.supported:
        raise ValueError(f"detected language {result.language!r} is outside {SUPPORTED_LANGUAGES}; "
                         "ask the reporter to type or pick a language")
    if not result.text:
        raise ValueError("no speech detected in the audio")
    dm = DialogueManager(language=result.language)
    dm.start(result.text)
    return dm, result
