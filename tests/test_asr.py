"""Tests for the Whisper wrapper, hand-off and WER scoring (no model download, no real audio)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.asr import eval_wer as E
from backend.asr.intake import intake
from backend.asr.transcribe import Transcriber


class FakeModel:
    def __init__(self, text="hello there", lang="en", prob=0.9):
        self.text, self.lang, self.prob, self.calls = text, lang, prob, []

    def transcribe(self, path, **kw):
        self.calls.append(kw)
        segs = [SimpleNamespace(text=f" {w} ") for w in self.text.split(", ")]
        info = SimpleNamespace(language=kw.get("language") or self.lang,
                               language_probability=self.prob, duration=2.5)
        return iter(segs), info


def make(tmp_path, **kw):
    fake = FakeModel(**kw)
    made = []
    t = Transcriber(model_size="tiny", mode="local",
                    model_factory=lambda *a: made.append(a) or fake)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    return t, fake, made, wav


def test_edit_distance_and_wer_basics():
    assert E.edit_distance(list("kitten"), list("sitting")) == 3
    assert E.edit_distance([], ["a"]) == 1 and E.edit_distance(["a"], ["a"]) == 0
    c = E.error_counts("the cat sat", "the cat sat")
    assert c["word_errors"] == 0 and c["ref_words"] == 3
    c = E.error_counts("the cat sat", "the dog sat down")
    assert c["word_errors"] == 2 and c["ref_words"] == 3   # 1 substitution + 1 insertion


def test_normalize_keeps_indic_marks_and_drops_punctuation():
    assert E.normalize("  Hello,  WORLD! ") == "hello world"
    assert E.normalize("ಹೌದು, ಇಲ್ಲ.") == "ಹೌದು ಇಲ್ಲ"          # virama/vowel marks survive
    assert E.normalize("नहीं!") == "नहीं"
    assert E.error_counts("नहीं", "नही")["word_errors"] == 1


def test_corpus_rate_is_pooled_not_mean_of_rates():
    df = pd.DataFrame([
        {"file": "a", "language": "en", "word_errors": 1, "ref_words": 2, "char_errors": 1, "ref_chars": 10},
        {"file": "b", "language": "en", "word_errors": 0, "ref_words": 8, "char_errors": 0, "ref_chars": 30},
    ])
    r = E.corpus_rates(df).iloc[0]
    assert r.wer == pytest.approx(0.1) and r.utterances == 2   # not (0.5 + 0) / 2


def test_transcriber_lazy_load_once_and_result(tmp_path):
    t, fake, made, wav = make(tmp_path, text="a, b")
    assert made == []                                   # nothing loaded at construction
    r = t.transcribe(wav, language="hi")
    t.transcribe(wav)
    assert len(made) == 1 and made[0] == ("tiny", "cpu", "int8")
    assert r.text == "a b" and r.language == "hi" and r.supported and r.duration_s == 2.5
    assert fake.calls[0]["temperature"] == 0.0 and fake.calls[0]["language"] == "hi"


def test_transcriber_validation(tmp_path):
    t, _, _, wav = make(tmp_path)
    with pytest.raises(ValueError):
        t.transcribe(wav, language="fr")
    with pytest.raises(FileNotFoundError):
        t.transcribe(tmp_path / "missing.wav")
    with pytest.raises(ValueError):
        Transcriber(model_size="huge")
    with pytest.raises(NotImplementedError):
        Transcriber(model_size="tiny", mode="api")


def test_env_config(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "base")
    monkeypatch.setenv("WHISPER_MODE", "local")
    assert Transcriber(model_factory=lambda *a: None).model_size == "base"


def test_intake_feeds_dialogue(tmp_path):
    t, _, _, wav = make(tmp_path, text="car accident, two injured, he is unconscious")
    dm, res = intake(wav, t)
    assert res.language == "en" and dm.injury_type == "road_accident"
    assert dm.answers["Q_CONSCIOUS"].value == "no" and dm.num_injured == 2


def test_intake_rejects_unsupported_language_and_silence(tmp_path):
    t, _, _, wav = make(tmp_path, text="bonjour", lang="fr")
    with pytest.raises(ValueError, match="outside"):
        intake(wav, t)
    t, _, _, wav = make(tmp_path, text="")
    with pytest.raises(ValueError, match="no speech"):
        intake(wav, t)


def test_evaluate_forces_language_and_scores(tmp_path):
    t, fake, _, _ = make(tmp_path, text="hello there")
    (tmp_path / "u1.wav").write_bytes(b"x")
    (tmp_path / "u2.wav").write_bytes(b"x")
    pd.DataFrame({"file": ["u1.wav", "u2.wav"], "language": ["en", "kn"],
                  "reference": ["hello there", "hello world"]}).to_csv(
        tmp_path / "reference.csv", index=False, encoding="utf-8")
    utt = E.evaluate(tmp_path, t)
    assert [c["language"] for c in fake.calls] == ["en", "kn"]
    rates = E.corpus_rates(utt).set_index("language")
    assert rates.loc["en", "wer"] == 0 and rates.loc["kn", "wer"] == 0.5


def test_reference_validation(tmp_path):
    p = tmp_path / "reference.csv"
    pd.DataFrame({"file": ["a"], "language": ["fr"], "reference": ["x"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="unsupported"):
        E.load_reference(tmp_path)
    pd.DataFrame({"file": ["a"], "language": ["en"], "reference": [" "]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="empty"):
        E.load_reference(tmp_path)
    pd.DataFrame({"file": ["a"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing"):
        E.load_reference(tmp_path)


# ---- noise robustness (backend/asr/noise.py, eval_noise.py) ----
import numpy as np  # noqa: E402

from backend.asr import eval_noise as EN  # noqa: E402
from backend.asr import noise as NZ  # noqa: E402


def _tone(n=16000):
    return (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / 16000)).astype(np.float32)


@pytest.mark.parametrize("snr", [20, 10, 0])
def test_mix_hits_target_snr(snr):
    rng = np.random.default_rng(0)
    s = _tone()
    m = NZ.mix_at_snr(s, NZ.pink_noise(5000, rng), snr, rng)   # shorter noise is tiled
    assert len(m) == len(s)
    assert NZ.measured_snr_db(s, m) == pytest.approx(snr, abs=0.3)


def test_mix_is_seeded_and_rejects_silence():
    s, n = _tone(), NZ.pink_noise(32000, np.random.default_rng(1))
    a = NZ.mix_at_snr(s, n, 5, np.random.default_rng(7))
    b = NZ.mix_at_snr(s, n, 5, np.random.default_rng(7))
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        NZ.mix_at_snr(np.zeros(100, np.float32), n, 5, np.random.default_rng(0))


def test_transcriber_vad_flag_is_passed(tmp_path):
    for flag in (True, False):
        fake = FakeModel()
        t = Transcriber(model_size="tiny", mode="local", vad_filter=flag, model_factory=lambda *a: fake)
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"x")
        t.transcribe(wav, language="en")
        assert fake.calls[0]["vad_filter"] is flag


def test_eval_noise_end_to_end_with_fake_model(tmp_path, monkeypatch):
    data = tmp_path / "d"
    data.mkdir()
    NZ.write_wav(data / "a.wav", _tone())
    pd.DataFrame({"file": ["a.wav"], "language": ["en"], "reference": ["hello there"]}) \
        .to_csv(data / "reference.csv", index=False)
    monkeypatch.setattr(NZ, "load_audio", lambda p: _tone())
    fake = FakeModel()
    ts = {c: Transcriber(model_size="tiny", mode="local", vad_filter=(c == "vad_on"),
                         model_factory=lambda *a: fake) for c in ("vad_on", "vad_off")}
    noises, src = EN.load_noises(tmp_path / "none", np.random.default_rng(0))
    assert src == "generated-pink"
    utt = EN.evaluate(data, [10, 0], noises, src, ts)
    assert set(utt.snr_db) == {"clean", "10", "0"} and set(utt.condition) == {"vad_on", "vad_off"}
    s = EN.summarize(utt)
    assert len(s) == 6 and (s.wer == 0).all() and (s.noise_source == "generated-pink").all()


def test_repetition_guard_and_large_models(tmp_path, monkeypatch):
    fake = FakeModel()
    t = Transcriber(model_size="large-v3", mode="local", repetition_penalty=1.2,
                    no_repeat_ngram_size=4, model_factory=lambda *a: fake)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    t.transcribe(wav, language="kn")
    assert fake.calls[0]["repetition_penalty"] == 1.2 and fake.calls[0]["no_repeat_ngram_size"] == 4
    monkeypatch.setenv("WHISPER_REPETITION_PENALTY", "1.1")
    monkeypatch.setenv("WHISPER_NO_REPEAT_NGRAM", "3")
    t2 = Transcriber(model_size="medium", mode="local", model_factory=lambda *a: fake)
    assert (t2.repetition_penalty, t2.no_repeat_ngram_size) == (1.1, 3)
    monkeypatch.delenv("WHISPER_REPETITION_PENALTY")
    monkeypatch.delenv("WHISPER_NO_REPEAT_NGRAM")
    t3 = Transcriber(model_size="small", mode="local", model_factory=lambda *a: fake)
    assert (t3.repetition_penalty, t3.no_repeat_ngram_size) == (1.0, 0)   # default = no guard


# ---- GPU config, latency, GPU-day runner ----
def test_device_and_compute_from_env(monkeypatch):
    for k in ("WHISPER_DEVICE", "WHISPER_COMPUTE"):
        monkeypatch.delenv(k, raising=False)
    t = Transcriber(model_size="tiny", mode="local", model_factory=lambda *a: None)
    assert (t.device, t.compute_type) == ("cpu", "int8")
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    monkeypatch.setenv("WHISPER_COMPUTE", "")                 # empty .env value -> default
    t = Transcriber(model_size="tiny", mode="local", model_factory=lambda *a: None)
    assert (t.device, t.compute_type) == ("cuda", "float16")
    monkeypatch.setenv("WHISPER_COMPUTE", "int8_float16")
    assert Transcriber(model_size="large-v3", mode="local").compute_type == "int8_float16"
    monkeypatch.setenv("WHISPER_DEVICE", "tpu")
    with pytest.raises(ValueError):
        Transcriber(model_size="tiny", mode="local")


def test_latency_measure_and_summary(tmp_path):
    from backend.asr import latency as L
    data = tmp_path / "d"
    data.mkdir()
    rows = []
    for lang in ("en", "hi"):
        for i in range(3):
            (data / f"{lang}{i}.wav").write_bytes(b"x")
            rows.append({"file": f"{lang}{i}.wav", "language": lang, "reference": "a b"})
    pd.DataFrame(rows).to_csv(data / "reference.csv", index=False)
    fake = FakeModel(text="a b")
    t = Transcriber(model_size="tiny", mode="local", model_factory=lambda *a: fake)
    df = L.measure(data, t, per_language=2)
    assert len(df) == 4 and len(fake.calls) == 5              # 1 untimed warm-up + 4 timed
    assert (df.audio_s == 2.5).all() and (df.rtf >= 0).all()
    s = L.summarize(df)
    assert list(s.language) == ["en", "hi"] and (s.clips == 2).all()


def test_gpu_day_plan():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("gpu_day", Path("scripts/asr_gpu_day.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    names = [n for n, _, _ in g.steps(skip_large=False, noise_large=False)]
    assert names == ["latency large-v3-turbo", "latency large-v3", "clean WER large-v3-turbo",
                     "clean WER large-v3", "noise WER large-v3-turbo"]
    assert len(g.steps(skip_large=True, noise_large=True)) == 3
    assert g.MODELS["large-v3"] == "int8_float16"             # fits a 4 GB GPU
