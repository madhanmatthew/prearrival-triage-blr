"""Tests for the text severity classifier (docs/15 D2). Toy fixture text only, not the dataset."""
import numpy as np
import pandas as pd
import pytest

from backend.dialogue.state_machine import DialogueManager
from backend.ml import text_severity as ts

TOY = {  # label -> phrases (toy test fixture, not training data)
    0: ["no one is hurt just a small scratch", "minor fall he is fine and walking",
        "car dent nobody injured", "slipped but okay no pain"],
    1: ["arm hurts maybe fracture but talking", "cut on leg small bleeding stopped",
        "twisted ankle cannot walk properly", "bruises and pain in the back"],
    2: ["heavy bleeding from leg we pressed cloth", "head injury conscious but confused",
        "broken leg bone visible awake", "deep cut bleeding a lot still talking"],
    3: ["not breathing unconscious after crash", "unconscious lot of blood not stopping",
        "no pulse not responding", "fell from building not moving not breathing"],
}


def toy_df(n_rep=3, langs=("en", "hi", "kn")):
    rows = []
    for lang in langs:
        for label, phrases in TOY.items():
            for r in range(n_rep):
                for i, p in enumerate(phrases):
                    rows.append({"report_id": f"{lang}-{label}-{r}-{i}", "language": lang,
                                 "text": f"{p} {'please come' * r}", "label": label,
                                 "reviewed": 1})
    return pd.DataFrame(rows)


def test_pipeline_is_char_ngram_2_5_lr():
    pipe = ts.build_pipeline()
    tf = pipe.named_steps["tfidf"]
    assert tf.analyzer == "char_wb" and tf.ngram_range == (2, 5)
    assert type(pipe.named_steps["lr"]).__name__ == "LogisticRegression"


def test_load_reports_validates_and_filters(tmp_path):
    df = toy_df(1, ("en",))
    df.loc[0, "reviewed"] = 0
    df.loc[1, "text"] = "   "
    p = tmp_path / "r.csv"
    df.to_csv(p, index=False)
    out = ts.load_reports(p)
    assert len(out) == len(df) - 2
    df.loc[2, "label"] = 5
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="label"):
        ts.load_reports(p)


@pytest.mark.parametrize("col,val,msg", [("language", "ta", "languages"),
                                         ("report_id", "en-0-0-0", "duplicate")])
def test_load_reports_rejects_bad_rows(tmp_path, col, val, msg):
    df = toy_df(1, ("en",))
    df.loc[3, col] = val
    df.to_csv(tmp_path / "r.csv", index=False)
    with pytest.raises(ValueError, match=msg):
        ts.load_reports(tmp_path / "r.csv")


def test_assign_split_is_20pct_stratified_deterministic_and_stable():
    df = toy_df(5)
    a, b = ts.assign_split(df), ts.assign_split(df)
    assert (a["split"] == b["split"]).all()
    assert abs((a["split"] == "test").mean() - 0.2) < 0.02
    for (_, _), g in a.groupby(["language", "label"]):
        assert (g["split"] == "test").sum() >= 1
    # existing split is never reassigned when new rows arrive
    more = pd.concat([a, toy_df(1).assign(report_id=lambda d: "new-" + d["report_id"])])
    c = ts.assign_split(more)
    assert (c.iloc[:len(a)]["split"].to_numpy() == a["split"].to_numpy()).all()
    assert c["split"].notna().all()


def test_train_predict_proba_shape_and_sum():
    df = ts.assign_split(toy_df(3))
    model, meta = ts.train(df[df["split"] == "train"], c_grid=[1.0, 10.0], n_splits=3)
    assert meta["C"] in (1.0, 10.0) and meta["cv_f1_macro"] is not None
    p = ts.predict_proba(model, ["unconscious not breathing", "just a scratch"])
    assert p.shape == (2, 4)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert p[0].argmax() == 3 and p[1].argmax() == 0
    assert ts.predict_proba(model, "no one hurt").shape == (4,)


def test_predict_proba_pads_missing_class():
    df = toy_df(2, ("en",))
    df = df[df["label"] != 2]
    model, _ = ts.train(df, c_grid=[1.0])
    p = ts.predict_proba(model, ["heavy bleeding"])
    assert p.shape == (1, 4) and p[0, 2] == 0.0
    np.testing.assert_allclose(p.sum(), 1.0)


def test_evaluate_reports_overall_and_per_language():
    df = ts.assign_split(toy_df(3))
    model, _ = ts.train(df[df["split"] == "train"], c_grid=[10.0])
    res = ts.evaluate(model, df[df["split"] == "test"])
    assert set(res["per_language"]) == {"en", "hi", "kn"}
    assert np.array(res["confusion"]).sum() == res["n_test"]
    assert res["f1_macro"] > 0.5          # toy phrases are trivially separable


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "MODEL_DIR", tmp_path)
    model, meta = ts.train(toy_df(1, ("en",)), c_grid=[1.0])
    path = ts.save(model, meta)
    assert path.name.startswith("text_severity_v1_")
    np.testing.assert_allclose(ts.predict_proba(ts.load(path), "not breathing"),
                               ts.predict_proba(model, "not breathing"))


def test_compose_text_and_dialogue_slots():
    assert ts.compose_text("crash near signal", ["no", "", "yes a lot"]) == \
        "crash near signal . no . yes a lot"
    assert ts.compose_text("", None) == ""
    model, _ = ts.train(toy_df(1, ("en",)), c_grid=[1.0])
    dm = DialogueManager(language="en")
    dm.start("accident, man unconscious and not breathing")
    probs = ts.predict_proba(model, ts.compose_text(dm.transcript)).tolist()
    f = dm.text_features(probs)
    assert f[15:19] == pytest.approx(probs)
