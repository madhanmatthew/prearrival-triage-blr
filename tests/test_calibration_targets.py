import pandas as pd

from scripts import build_calibration_targets as bc


def typical():
    rows = []
    for dow in range(7):
        for h in range(24):
            for seg, base in (("silkboard_bellandur", 600), ("silkboard_krpuram", 1500)):
                rows.append({"segment": seg, "direction": "forward", "dow": float(dow), "hour": float(h),
                             "historic_s": base * (2 if 17 <= h < 19 else 1) + (1000 if dow >= 5 else 0),
                             "length_m": 6000.0 if seg == "silkboard_bellandur" else 16800.0})
            rows.append({"segment": "silkboard_bellandur", "direction": "reverse", "dow": 0.0,
                         "hour": float(h), "historic_s": 99999, "length_m": 9000.0})
    return pd.DataFrame(rows)


def flow():
    # 2026-10-05 is a Monday; UTC 12:00 = 17:30 IST
    return pd.DataFrame({"collected_at": ["2026-10-05T12:00:00+00:00", "2026-10-04T12:00:00+00:00",
                                          "2026-10-05T21:00:00+00:00"],
                         "point": ["silkboard"] * 3, "current_speed_kmph": [20, 99, 40],
                         "free_flow_speed_kmph": [42, 42, 42]})


def counts():
    return pd.DataFrame({"source_id": ["BTP2025_DH"], "metric": ["peak_volume"], "value": [10657],
                         "unit": ["PCU (peak hours)"]})


def test_slots_use_weekday_forward_and_ist_hours():
    t = bc.build(typical(), flow(), counts()).set_index(["slot", "segment"])
    assert t.loc[("peak", "silkboard_bellandur"), "typical_s"] == 1200      # x2 in 17-19, weekdays only
    assert t.loc[("low", "silkboard_bellandur"), "typical_s"] == 600
    assert t.loc[("peak", "silkboard_bellandur"), "live_speed_kmh"] == 20   # Sunday 99 excluded
    assert t.loc[("low", "silkboard_bellandur"), "live_speed_kmh"] == 40    # 21:00 UTC = 02:30 IST
    assert t.loc[("peak", "silkboard_bellandur"), "peak_volume"] == 10657
    assert pd.isna(t.loc[("medium", "silkboard_bellandur"), "peak_volume"])
    assert t.loc[("low", "silkboard_bellandur"), "typical_speed_kmh"] == 36.0


def test_real_targets_ordered_low_medium_peak():
    t = bc.build(pd.read_csv(bc.TYPICAL), pd.read_csv(bc.FLOW), pd.read_csv(bc.COUNTS))
    full = t[t.segment == "silkboard_krpuram"].set_index("slot").typical_s
    assert full["low"] < full["medium"] < full["peak"]
    assert (t.n_typical == 10).all()                                         # 2 hours x 5 weekdays
