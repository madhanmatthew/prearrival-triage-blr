import pandas as pd
import pytest
from fastapi.testclient import TestClient

from scripts import review_text_reports as rv


def make_csv(tmp_path):
    p = tmp_path / "r.csv"
    pd.DataFrame({"report_id": ["en-1", "hi-1", "kn-1"], "language": ["en", "hi", "kn"],
                  "text": ["man fell, awake", "आदमी गिर गया", "ವ್ಯಕ್ತಿ ಬಿದ್ದಿದ್ದಾರೆ"], "label": [1, 2, 3],
                  "reviewed": [0, 0, 0], "split": [None] * 3, "notes": [None] * 3}).to_csv(p, index=False, encoding="utf-8-sig")
    return p


def test_apply_review_relabel_and_edit(tmp_path):
    df = rv.load(make_csv(tmp_path))
    df = rv.apply_review(df, "en-1", 2, "man fell from a ladder, awake", "madhan")
    r = df.set_index("report_id").loc["en-1"]
    assert r.reviewed == 1 and r.label == 2 and r.text.startswith("man fell from a ladder")
    assert "relabel 1->2" in r.notes and "text edited" in r.notes and "by madhan" in r.notes
    with pytest.raises(ValueError):
        rv.apply_review(df, "hi-1", 7, None, "")
    with pytest.raises(KeyError):
        rv.apply_review(df, "nope", 1, None, "")


def test_api_queue_review_and_language_filter(tmp_path):
    p = make_csv(tmp_path)
    c = TestClient(rv.create_app(p, language="hi"))
    r = c.get("/next").json()
    assert r["row"]["report_id"] == "hi-1" and r["total"] == 1 and r["done"] == 0
    assert c.post("/review", json={"report_id": "hi-1", "label": 2, "reviewer": "x"}).json()["ok"]
    assert c.get("/next").json()["row"] is None
    back = rv.load(p)
    assert int(back.set_index("report_id").loc["hi-1", "reviewed"]) == 1
    assert "हि" not in "" and back.set_index("report_id").loc["kn-1", "text"] == "ವ್ಯಕ್ತಿ ಬಿದ್ದಿದ್ದಾರೆ"   # UTF-8 kept
    assert "Rubric" in c.get("/").text
