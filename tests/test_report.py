import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from stormwater_intel import report as RP


def _scored(n_events: int = 3, n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for e in range(n_events):
        d = {"event_id": f"e{e}", "y": rng.poisson(1.0, n)}
        for m in RP.CURVE_MODELS:
            d[f"score_{m}"] = np.zeros(n) if m == "random" else rng.random(n)
        frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True)


def _universe():
    return gpd.GeoDataFrame({"comm_code": ["AAA", "BBB"], "name": ["ALPHA", "BETA"]},
                            geometry=[box(-114.1, 51.0, -114.0, 51.1), box(-114.0, 51.0, -113.9, 51.1)],
                            crs="EPSG:4326")


def _gauges():
    return pd.DataFrame({"name": ["g1"], "lat": [51.05], "lon": [-114.0]}, index=pd.Index([1], name="channel"))


def test_gain_curve_is_monotone_and_ends_at_one():
    x, y = RP.gain_curve(_scored(), "score_xgb")
    assert x[-1] == pytest.approx(1.0) and y[-1] == pytest.approx(1.0)
    assert np.all(np.diff(y) >= -1e-12)


def test_gain_curve_of_a_perfect_ranking_beats_random():
    s = _scored()
    s["score_xgb"] = s["y"] + np.random.default_rng(1).random(len(s)) * 0.01
    _, perfect = RP.gain_curve(s, "score_xgb")
    _, rnd = RP.gain_curve(s, "score_random")
    assert perfect[3] > rnd[3] + 0.1


def test_plot_gain_curves_writes_a_png(tmp_path):
    out = tmp_path / "gain.png"
    RP.plot_gain_curves({"a": _scored(), "b": _scored(seed=2)}, out)
    assert out.stat().st_size > 5_000


def test_build_map_contains_every_layer_and_the_disclaimer(tmp_path):
    layers = {
        "Risk layer": {"values": pd.Series({"AAA": 90.0, "BBB": 10.0}), "kind": "percentile",
                       "label": "Risk percentile", "show": True},
        "Actual layer": {"values": pd.Series({"AAA": 3.0, "BBB": 0.0}), "kind": "count", "label": "Complaints"},
    }
    out = tmp_path / "map.html"
    RP.build_map(_universe(), _gauges(), layers, out)
    html = out.read_text(encoding="utf-8")
    for text in ("Risk layer", "Actual layer", "City rain gauges", "Complaint risk, not flood risk"):
        assert text in html


def test_plot_static_map_writes_a_png(tmp_path):
    out = tmp_path / "maps.png"
    values = pd.Series({"AAA": 90.0, "BBB": 10.0})
    RP.plot_static_map(_universe(), _gauges(), [
        {"title": "risk", "values": values, "cmap": "YlOrRd", "vmin": 0, "vmax": 100, "label": "pct", "annotate": True},
        {"title": "count", "values": values / 30, "cmap": "Blues", "vmin": 0, "vmax": 3, "label": "n"},
    ], out)
    assert out.stat().st_size > 5_000
