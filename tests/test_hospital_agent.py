import pandas as pd
import pytest

from backend.agents import hospital_agent as ha
from backend.schemas import FusedRisk


def _hosp(hid, trauma=3, beds=10, spec="trauma", **blood):
    row = {"hospital_id": hid, "name": hid, "trauma_level": trauma, "beds_available": beds,
           "specialties": spec, **{c: 10 for c in ha.BLOOD_COLS}}
    row.update({f"blood_{k}": v for k, v in blood.items()})
    return row


def _risk(cls):
    p = [0.0] * 4
    p[cls] = 1.0
    return FusedRisk.from_probs(p, incident_id="I", patient_id="P", timestamp="t",
                                modalities_used=["text"], model_version="x")


def _agent(rows):
    return ha.HospitalAgent(pd.DataFrame(rows))


def test_factor_functions():
    assert ha.proximity(0) == 1 and ha.proximity(45) == 0 and ha.proximity(15) == 0.5
    assert ha.trauma_match(3, 3) == 1 and ha.trauma_match(2, 3) == 0.5 and ha.trauma_match(1, 3) == 0
    assert ha.bed_avail(20) == 1 and ha.bed_avail(5) == 0.5
    assert ha.specialty_match("burn", "Burns; cardiology") == 1
    assert ha.specialty_match("burn", "cardiology") == 0
    assert ha.specialty_match("other", "x") == 0.5


def test_blood_needed_rule():
    assert ha.blood_needed(0, "burn", True)
    assert ha.blood_needed(2, "road_accident", False)
    assert not ha.blood_needed(2, "cardiac_medical", False)
    assert not ha.blood_needed(1, "road_accident", False)


def test_weights_sum_to_one_and_redistribute():
    assert sum(ha.policy_weights(True).values()) == pytest.approx(1)
    w = ha.policy_weights(False)
    assert w["blood"] == 0 and w["proximity"] == pytest.approx(0.45)


def test_blood_score_unknown_uses_o_neg_and_flags():
    r = pd.Series(_hosp("a", **{"O-": 2}))
    assert ha.blood_score(r, None) == (0.5, "unknown")
    assert ha.blood_score(pd.Series(_hosp("a", **{"A+": 0})), "A+") == (0.0, "short")
    assert ha.blood_score(pd.Series(_hosp("a")), "A+") == (1.0, "matched")


def test_blood_breaks_tie_between_equally_close_hospitals():  # docs/08 §7.4
    ag = _agent([_hosp("stock", **{"B-": 6}), _hosp("empty", **{"B-": 0})])
    rec = ag.predict("I", _risk(3), "road_accident", {"stock": 10, "empty": 10},
                     blood_group="B-", use_rf=False)
    assert rec.ranked[0].hospital_id == "stock"
    assert rec.ranked[1].blood_flag == "short"


def test_blood_ignored_when_no_transfusion_risk():
    ag = _agent([_hosp("stock", **{"B-": 6}), _hosp("empty", **{"B-": 0})])
    rec = ag.predict("I", _risk(1), "cardiac_medical", {"stock": 10, "empty": 10},
                     blood_group="B-", use_rf=False)
    assert rec.ranked[0].total_score == rec.ranked[1].total_score
    assert all(h.breakdown.blood_match == 0 for h in rec.ranked)


def test_trauma_capability_beats_small_eta_gain_for_critical():
    ag = _agent([_hosp("near_low", trauma=0), _hosp("far_high", trauma=3)])
    rec = ag.predict("I", _risk(3), "road_accident", {"near_low": 8, "far_high": 14},
                     use_rf=False)
    assert rec.ranked[0].hospital_id == "far_high"


def test_only_hospitals_with_eta_are_ranked_and_top_k():
    ag = _agent([_hosp("a"), _hosp("b"), _hosp("c")])
    rec = ag.predict("I", _risk(0), "other", {"a": 5, "b": 9}, use_rf=False, top_k=1)
    assert [h.hospital_id for h in rec.ranked] == ["a"]
    with pytest.raises(ValueError):
        ag.predict("I", _risk(0), "other", {}, use_rf=False)


def test_rf_surrogate_trains_saves_loads_and_blends(tmp_path):
    ag = _agent([_hosp("a", beds=0, trauma=0), _hosp("b")])
    m = ag.train(n_episodes=800, seed=1)
    assert m["mae_reward"] < 8 and m["data"] == "simulated episodes"
    ag.save(tmp_path / "rf.joblib")
    ag2 = _agent([_hosp("a", beds=0, trauma=0), _hosp("b")])
    ag2.load(tmp_path / "rf.joblib")
    rec = ag2.predict("I", _risk(3), "road_accident", {"a": 10, "b": 10})
    assert rec.ranked[0].hospital_id == "b"
    assert 0 <= rec.ranked[0].breakdown.rf_qvalue <= 1


def test_simulated_episodes_are_seeded():
    X1, y1 = ha.simulate_episodes(50, 3)
    X2, y2 = ha.simulate_episodes(50, 3)
    assert (X1 == X2).all() and (y1 == y2).all()
