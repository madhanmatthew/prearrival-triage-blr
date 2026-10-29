"""Window picking rules for the live demo replay (scripts/build_demo_vitals.py)."""
import numpy as np
import pandas as pd

from scripts.build_demo_vitals import N_READINGS, _readings, find_windows


def _case(sbp: list[float]) -> pd.DataFrame:
    n = len(sbp)
    return pd.DataFrame({"hr": np.full(n, 80.0), "rr": np.full(n, 12.0), "spo2": np.full(n, 98.0),
                         "sbp": sbp, "temp": np.full(n, 36.5)}, index=pd.RangeIndex(n, name="minute"))


def test_stable_window_needs_30_quiet_minutes():
    found = find_windows(_case([120.0] * 45))
    assert "stable" in found and "deteriorating" not in found
    assert found["stable"][0] == N_READINGS - 1          # first eligible end minute


def test_deteriorating_window_ends_before_a_new_event():
    sbp = [130.0] * 10 + list(np.linspace(128, 100, 10)) + [95, 92, 91, 85, 84, 84, 84, 84, 84, 84]
    t, outcome = find_windows(_case(sbp))["deteriorating"]
    win = slice(t - N_READINGS + 1, t + 1)
    assert min(sbp[win]) >= 90                             # no event inside the replayed window
    assert sbp[t - N_READINGS + 1] - sbp[t] >= 15
    assert "Real outcome" in outcome


def test_missing_vital_excludes_window_and_readings_are_json_ready():
    c = _case([120.0] * 45)
    c.loc[3, "spo2"] = np.nan
    t, _ = find_windows(c)["stable"]
    assert t - N_READINGS + 1 > 3
    r = _readings(c, t)
    assert len(r) == N_READINGS and r[0]["avpu"] is None and isinstance(r[0]["hr"], int)
