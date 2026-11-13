import numpy as np
import pandas as pd
import pytest

from stormwater_intel import rainfall as R


def test_to_xy_km_one_degree_latitude():
    xy = R.to_xy_km([51.05, 52.05], [-114.07, -114.07])
    assert xy[1, 1] - xy[0, 1] == pytest.approx(111.19, abs=0.1)
    assert xy[1, 0] == pytest.approx(0.0, abs=1e-9)


def test_gauge_daily_totals_and_rolling_maxima():
    ts = pd.to_datetime(["2025-06-01 10:00", "2025-06-01 10:05", "2025-06-01 10:10", "2025-06-01 15:00"])
    tips = pd.DataFrame({"channel": 7, "ts": ts, "mm": [1.0, 1.0, 1.0, 0.4]})
    out = R.gauge_daily(tips).iloc[0]
    assert out["total_mm"] == pytest.approx(3.4)
    assert out["max_5min"] == pytest.approx(1.0)
    assert out["max_15min"] == pytest.approx(3.0)
    assert out["max_60min"] == pytest.approx(3.0)


def test_daily_matrix_zero_when_online_nan_when_offline():
    tips = pd.DataFrame({
        "channel": [1] * 25 + [2] * 3,
        "ts": pd.date_range("2025-06-01", periods=28, freq="h"),
        "mm": 0.2,
    })
    active = R.active_gauge_seasons(tips, min_tips=20)
    assert active == {(1, 2025)}              # gauge 2 recorded too little to trust
    gd = R.gauge_daily(tips[tips["channel"] == 1])
    mat = R.daily_matrix(gd, active)
    assert list(mat.columns) == [1]
    assert mat.loc["2025-06-01", 1] == pytest.approx(4.8)      # 24 tips of 0.2 mm
    assert mat.loc["2025-07-15", 1] == 0.0                      # online, dry
    assert len(mat) == 153                                       # May 1 - Sep 30


def test_idw_equidistant_gauges_average_and_nan_is_ignored():
    d = np.array([[2.0, 2.0]])
    assert R.idw(np.array([[4.0, 8.0]]), d, k=None)[0, 0] == pytest.approx(6.0)
    assert R.idw(np.array([[4.0, np.nan]]), d, k=None)[0, 0] == pytest.approx(4.0)


def test_idw_nearest_k_and_all_offline():
    d = np.array([[1.0, 10.0]])
    assert R.idw(np.array([[0.0, 100.0]]), d, k=1)[0, 0] == pytest.approx(0.0)
    assert np.isnan(R.idw(np.array([[np.nan, np.nan]]), d)[0, 0])


def _matrix(storm_days: dict[str, float]) -> pd.DataFrame:
    idx = pd.date_range("2025-06-01", "2025-06-30")
    mat = pd.DataFrame(0.0, index=idx, columns=[1, 2, 3])
    for day, mm in storm_days.items():
        mat.loc[day, [1, 2]] = mm
    return mat


def test_storm_events_tile_fixed_windows_and_absorb_storm_days_inside():
    mat = _matrix({"2025-06-03": 20, "2025-06-04": 12, "2025-06-06": 11, "2025-06-20": 30})
    ev = R.detect_storm_events(mat, trigger_mm=10, min_gauges=2, window_days=3)
    assert list(ev["event_id"]) == ["2025-06-03", "2025-06-06", "2025-06-20"]
    assert list(ev["n_storm_days"]) == [2, 1, 1]                 # 4 Jun is absorbed by the 3-5 Jun window
    assert ((ev["window_end"] - ev["start"]).dt.days == 2).all()  # every window is exactly 3 days
    assert (ev["start"].iloc[1:].to_numpy() > ev["window_end"].iloc[:-1].to_numpy()).all()  # disjoint
    first = ev.iloc[0]
    assert first["peak_gauge_mm"] == pytest.approx(20)
    assert first["n_gauges_hit"] == 2
    assert first["city_mean_mm"] == pytest.approx(64 / 3)         # gauges 1-2: 32 mm each, gauge 3 dry


def test_long_wet_spell_becomes_consecutive_events_not_one_cluster():
    days = {f"2025-06-{d:02d}": 15 for d in range(1, 16)}         # 15 stormy days in a row
    ev = R.detect_storm_events(_matrix(days), trigger_mm=10, min_gauges=2, window_days=3)
    assert len(ev) == 5 and (ev["n_storm_days"] == 3).all()


def test_active_gauge_seasons_drops_dead_but_noisy_gauges():
    n = {1: 100, 2: 100, 3: 22}                                   # gauge 3: 4.4 mm vs median 20 mm
    ts = pd.concat([pd.Series(pd.date_range("2025-06-01", periods=k, freq="h")) for k in n.values()],
                   ignore_index=True)
    tips = pd.DataFrame({"channel": np.repeat(list(n), list(n.values())), "ts": ts, "mm": 0.2})
    assert R.active_gauge_seasons(tips, min_tips=20, min_rel_total=0.25) == {(1, 2025), (2, 2025)}


def test_load_tips_file_priority_clip_and_min_year(tmp_path):
    hist, cur = tmp_path / "h.csv", tmp_path / "c.csv"
    hist.write_text("channel,name,year,timestamp,rainfall\n"
                    "1,A,2021,2021-06-01T10:00:00.000,0.4\n"
                    "1,A,2011,2011-06-01T10:00:00.000,0.2\n"
                    "1,A,2021,2021-06-01T10:05:00.000,43.2\n")
    cur.write_text("channel,name,year,timestamp,rainfall\n01,A,2021,2021-06-01T10:00:00,9.9\n")
    tips = R.load_tips([hist, cur], min_year=2012, max_tip_mm=10.0)
    assert len(tips) == 2                                          # 2011 dropped, duplicate collapsed
    assert tips.loc[tips["ts"] == pd.Timestamp("2021-06-01 10:00"), "mm"].iloc[0] == pytest.approx(0.4)
    assert tips["mm"].max() == pytest.approx(10.0)                 # 43.2 mm clipped


def test_single_gauge_hit_is_not_a_storm():
    mat = _matrix({})
    mat.loc["2025-06-10", 1] = 40.0
    assert R.detect_storm_events(mat, trigger_mm=10, min_gauges=2).empty
