import numpy as np
import pandas as pd
import pytest

from backend.data import vitaldb_windows as vw


def raw(n, **cols):
    d = {"minute": np.arange(n), "hr": [80.0] * n, "spo2": [98.0] * n, "rr": [14.0] * n,
         "temp": [36.5] * n, "sbp_nibp": [np.nan] * n, "sbp_art": [120.0] * n}
    d.update({k: list(v) for k, v in cols.items()})
    return pd.DataFrame(d)


def test_cleaning_ranges_and_sbp_source():
    r = raw(6, hr=[80, 300, 80, 80, 80, 80], sbp_art=[120, -5, np.nan, np.nan, np.nan, np.nan],
            sbp_nibp=[np.nan, np.nan, np.nan, np.nan, 110, np.nan], temp=[20, 36, 36, 36, 36, 36])
    c = vw.clean_case(r)
    assert np.isnan(c.temp.iloc[0])                       # probe off -> missing
    assert c.hr.iloc[1] == 80                             # 300 dropped, forward-filled
    # art invalid at 1 -> ffill 120 for <= 2 minutes (1, 2), then nibp; nibp 110 from minute 4
    assert list(c.sbp.iloc[:3]) == [120, 120, 120]
    assert np.isnan(c.sbp.iloc[3]) and c.sbp.iloc[4] == 110 and c.sbp.iloc[5] == 110


def test_minute_grid_filled_and_duplicates_dropped():
    r = raw(3)
    r = pd.concat([r, r.iloc[[1]]]).reset_index(drop=True)
    r.loc[2, "minute"] = 5                                # gap 3,4
    c = vw.clean_case(r)
    assert list(c.index) == [0, 1, 2, 3, 4, 5]


def test_event_needs_two_consecutive_low_minutes():
    sbp = [120, 85, 120, 85, 85, 120]
    c = vw.clean_case(raw(6, sbp_art=sbp))
    assert vw.event_minutes(c).tolist() == [False, False, False, False, True, False]
    c2 = vw.clean_case(raw(3, spo2=[88, 88, 98]))
    assert vw.event_minutes(c2).tolist() == [False, True, False]


def test_windows_labels_and_eligibility():
    # 30 minutes normal, then a sustained drop at minutes 20-21
    sbp = [120.0] * 30
    sbp[20] = sbp[21] = 80
    c = vw.clean_case(raw(30, sbp_art=sbp))
    X, M, Y, E = vw.case_windows(c)
    ends = dict(zip(E.tolist(), Y.tolist()))
    assert ends[4] == 0                                   # (4, 14]: nothing
    assert ends[14] == 1                                  # (14, 24] contains event minute 21
    assert 24 not in ends or ends[24] == 0
    assert X.shape[1:] == (vw.T, len(vw.VITALS)) and M.all()
    assert all((e - 4) % vw.STRIDE == 0 for e in E)


def test_windows_skip_ongoing_episode():
    sbp = [120.0] * 30
    for k in range(8, 16):
        sbp[k] = 80                                       # event minutes 9..15
    c = vw.clean_case(raw(30, sbp_art=sbp))
    _, _, _, E = vw.case_windows(c)
    assert 9 not in E and 14 not in E                     # inside the episode


def test_windows_need_known_label():
    c = vw.clean_case(raw(20, sbp_art=[120.0] * 10 + [np.nan] * 10,
                          spo2=[98.0] * 10 + [np.nan] * 10))
    _, _, _, E = vw.case_windows(c)
    assert E.max() <= 9


def test_features_deltas_and_gap():
    X = np.array([[[np.nan] * 5, [80, 14, 98, 120, 36.5], [90, 15, 97, 110, 36.6]]], np.float32)
    M = np.array([[False, True, True]])
    F = vw.add_features(X, M)
    assert F.shape == (1, 3, vw.N_FEATURES)
    assert F[0, 1, 5:10].tolist() == [0] * 5 and F[0, 1, 10] == 0     # after padding
    assert F[0, 2, 5:10] == pytest.approx([10, 1, -1, -10, 0.1], abs=1e-5)
    assert F[0, 2, 10] == 1


def test_build_and_load(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame({"caseid": [1, 2, 3], "subjectid": [7, 7, 9]}).to_csv(raw_dir / "cases.csv", index=False)
    sbp = [120.0] * 30
    sbp[20] = sbp[21] = 80
    for cid in (1, 2):
        raw(30, sbp_art=sbp).to_csv(raw_dir / f"case_{cid}.csv.gz", index=False)
    out = tmp_path / "w.npz"
    d = vw.build(raw_dir, out)                            # case 3 file missing: skipped
    assert set(d["caseid"]) == {1, 2} and set(d["groups"]) == {7}
    back = vw.load(out)
    assert np.array_equal(back["y"], d["y"]) and back["X"].shape[1:] == (vw.T, 5)
