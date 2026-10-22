"""Tests for the ASR -> dialogue cue robustness eval (backend/asr/eval_triage.py)."""
import pandas as pd
import pytest

from backend.asr import eval_triage as et


def _utt():
    return pd.DataFrame([
        {"file": "eng rec/en_001.wav", "language": "en",
         "reference": "Accident, a man is not breathing.", "hypothesis": "Accident, a man is not breathing."},
        {"file": "hin rec/hi_001.m4a", "language": "hi",
         "reference": "एक्सीडेंट हुआ है, वो सांस नहीं ले रहा", "hypothesis": "अक्सिडेंट हुआ है वो सास नहीं ले रहा"},
        {"file": "eng rec/en_002.wav", "language": "en",
         "reference": "He fell from the roof.", "hypothesis": "He fell from the roof."},
        {"file": "hin rec/hi_002.m4a", "language": "hi",
         "reference": "वो छत से गिर गया", "hypothesis": "वो छत से"},          # ASR dropped the verb
    ])


def test_scenario_id():
    assert et.scenario_id("kan rec/kn_017.m4a") == "017" == et.scenario_id("eng rec/en_017.wav")
    # held-out clips (h01-h10) pair across languages the same way
    assert et.scenario_id("heldout/hi_h03.wav") == "h03" == et.scenario_id("heldout/en_h03.wav")


def test_extract_fields():
    f = et.extract("Accident, 2 injured, unconscious", "en")
    assert set(f) == set(et.FIELDS)
    assert f["injury_type"] == "road_accident" and f["unconscious"] and f["num_injured"] == 2
    assert et.extract(float("nan"), "en")["injury_type"] == "other"   # empty hypothesis


def test_compare_and_summarize():
    cmp = et.compare(_utt())
    assert len(cmp) == 4 * len(et.FIELDS)
    s = et.summarize(cmp).set_index(["language", "field"])
    assert s.loc[("en", "all_fields"), "asr"] == 1.0
    assert s.loc[("hi", "injury_type"), "asr"] == pytest.approx(0.5)      # hi_002 lost "fall"
    assert s.loc[("hi", "injury_type"), "ref_vs_en"] == 1.0
    assert s.loc[("hi", "breathing_problem"), "vs_en"] == 1.0
    assert s.loc[("hi", "all_fields"), "clips"] == 2
