"""Audio helpers for the noise-robustness eval (docs/01 §8): load, mix at a target SNR, write WAV.

Mixing: noise is looped/cropped to the speech length (random offset from a seeded rng) and
scaled so that 10*log10(P_speech / P_noise) = snr_db, with powers measured over the whole clip.
Output is clipped to [-1, 1]. Audio is mono float32 at 16 kHz.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def load_audio(path: str | Path) -> np.ndarray:
    """Any ffmpeg/PyAV-readable file -> mono float32 at 16 kHz (via faster-whisper's decoder)."""
    from faster_whisper.audio import decode_audio
    return np.asarray(decode_audio(str(path), sampling_rate=SAMPLE_RATE), dtype=np.float32)


def write_wav(path: str | Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generated 1/f noise; a stand-in only when no recorded street noise is available."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec))
    f[0] = 1
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return (x / (np.std(x) + 1e-12)).astype(np.float32)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float,
               rng: np.random.Generator) -> np.ndarray:
    if len(noise) == 0 or len(speech) == 0:
        raise ValueError("speech and noise must be non-empty")
    reps = int(np.ceil(len(speech) / len(noise))) + 1
    noise = np.tile(noise, reps)
    start = int(rng.integers(0, len(noise) - len(speech) + 1))
    noise = noise[start:start + len(speech)]
    p_s, p_n = float(np.mean(speech ** 2)), float(np.mean(noise ** 2))
    if p_s == 0 or p_n == 0:
        raise ValueError("speech or noise is silent; SNR undefined")
    scale = np.sqrt(p_s / (p_n * 10 ** (snr_db / 10)))
    return np.clip(speech + scale * noise, -1.0, 1.0).astype(np.float32)


def measured_snr_db(speech: np.ndarray, mixed: np.ndarray) -> float:
    resid = mixed - speech
    return float(10 * np.log10(np.mean(speech ** 2) / np.mean(resid ** 2)))


def denoise(audio: np.ndarray) -> np.ndarray:
    """Optional spectral gating (noisereduce). Compared, not assumed to help."""
    try:
        import noisereduce as nr
    except ImportError as e:
        raise ImportError("pip install noisereduce to use --denoise") from e
    return nr.reduce_noise(y=audio, sr=SAMPLE_RATE).astype(np.float32)
