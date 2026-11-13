import numpy as np
import pandas as pd
import pytest

from stormwater_intel import panel as P

COMMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def _toy():
    idx = pd.date_range("2020-01-01", "2020-12-31")
    y = pd.DataFrame(0.0, index=idx, columns=COMMS)
    y.loc["2020-03-01", "AAA"] = 1                                 # before any event
    y.loc["2020-06-02", "AAA"] = 3
    y.loc["2020-06-03", "AAA"] = 2                                 # event 1 window: 2-4 Jun
    y.loc["2020-07-11", "AAA"] = 4
    y.loc["2020-07-11", "BBB"] = 1                                 # event 2 window: 10-12 Jul
    rain = pd.DataFrame(5.0, index=pd.date_range("2020-05-01", "2020-09-30"), columns=COMMS)
    statics = pd.DataFrame(0.0, index=COMMS, columns=P.STATIC_FEATURES)
    statics["lat"] = 51.0 + 0.01 * np.arange(len(COMMS))
    statics["lon"] = -114.0 + 0.01 * np.arange(len(COMMS))
    statics["dwell_cnt"] = 50.0
    events = pd.DataFrame({
        "event_id": ["2020-06-02", "2020-07-10"],
        "start": pd.to_datetime(["2020-06-02", "2020-07-10"]),
        "window_end": pd.to_datetime(["2020-06-04", "2020-07-12"]),
        "n_storm_days": [1, 1], "peak_gauge_mm": [20.0, 30.0], "n_gauges_hit": [3, 5],
        "city_mean_mm": [15.0, 25.0],
    })
    return events, y, rain, statics


def test_panel_labels_and_leakage_free_history():
    events, y, rain, statics = _toy()
    panel = P.build_event_panel(events, y, rain, rain, statics)
    assert len(panel) == 2 * len(COMMS)
    e1 = panel[panel["event_id"] == "2020-06-02"].set_index("comm_code")
    e2 = panel[panel["event_id"] == "2020-07-10"].set_index("comm_code")

    assert e1.loc["AAA", "y"] == 5 and e2.loc["AAA", "y"] == 4 and e2.loc["BBB", "y"] == 1
    # first event: no previous events, only the March complaint precedes it
    assert e1["hist_ev_mean"].isna().all() and e1["days_since_prev_event"].isna().all()
    assert e1.loc["AAA", "h365"] == 1
    # second event sees the first window (5) but never its own complaints (4)
    assert e2.loc["AAA", "h365"] == 1 + 5
    assert e2.loc["AAA", "hist_ev_mean"] == 5 and e2.loc["BBB", "hist_ev_mean"] == 0
    assert e2.loc["AAA", "hist_ev_any"] == 1 and (e2["hist_ev_n"] == 1).all()
    assert e2.loc["AAA", "hist_ev_share"] == pytest.approx(1.0)
    assert e2["days_since_prev_event"].iloc[0] == 36               # 4 Jun -> 10 Jul


def test_panel_rain_features():
    events, y, rain, statics = _toy()
    e1 = P.build_event_panel(events, y, rain, rain, statics).query("event_id == '2020-06-02'")
    assert (e1["rain_win_mm"] == 15).all()                         # 3 days x 5 mm
    assert (e1["rain_prev7_mm"] == 35).all()
    assert (e1["rain_prev30_mm"] == 150).all()                     # 3 May .. 1 Jun, all inside the season


def test_history_tracker_ewm_and_heavy_share():
    t = P.HistoryTracker(2, alpha=0.5, heavy_mm=25.0)
    t.update(np.array([4.0, 0.0]), peak_gauge_mm=10)               # light
    t.update(np.array([0.0, 2.0]), peak_gauge_mm=40)               # heavy
    f = t.features()
    assert f["hist_ev_mean"].tolist() == [2.0, 1.0]
    assert f["hist_ev_ewm"].tolist() == [2.0, 1.0]                 # 0.5*4 + 0.5*0, 0.5*0 + 0.5*2
    assert f["hist_heavy_share"].tolist() == [0.0, 1.0]            # only the heavy event counts


def test_knn_weights_rows_sum_to_one_and_exclude_self():
    w = P.knn_weights(51.0 + 0.01 * np.arange(8), -114.0 + 0.01 * np.arange(8), k=3)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.allclose(np.diag(w), 0.0)
    assert (w > 0).sum(axis=1).tolist() == [3] * 8
