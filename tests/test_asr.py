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
