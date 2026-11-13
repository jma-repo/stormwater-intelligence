import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from stormwater_intel import pipeline as PL


def test_render_maps_rebuilds_from_saved_outputs_without_a_model_run(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"largest_2025_event": {"event_id": "2025-06-20"}}))
    pd.DataFrame({"comm_code": ["AAA", "BBB"], "risk_percentile": [100.0, 50.0],
                  "expected_per_1000_dwellings": [3.0, float("nan")]}).to_csv(
        tmp_path / "risk_ranking_design_storm_30mm.csv", index=False)
    pd.DataFrame({"comm_code": ["AAA", "BBB"], "name": ["A", "B"], "y": [4, 0], "score_xgb": [2.0, 0.1],
                  "pred_percentile": [100.0, 50.0]}).to_csv(
        tmp_path / "backtest_2025-06-20_predicted_vs_actual.csv", index=False)
    universe = gpd.GeoDataFrame({"comm_code": ["AAA", "BBB"], "name": ["ALPHA", "BETA"]},
                                geometry=[box(-114.1, 51.0, -114.0, 51.1), box(-114.0, 51.0, -113.9, 51.1)],
                                crs="EPSG:4326")
    gauges = pd.DataFrame({"name": ["g1"], "lat": [51.05], "lon": [-114.0]}, index=pd.Index([1], name="channel"))

    PL.render_maps(tmp_path, universe, gauges)

    html = (tmp_path / "risk_map.html").read_text(encoding="utf-8")
    for layer in ("Design storm (30 mm uniform): risk percentile", "risk per 1,000 dwellings",
                  "Backtest 2025-06-20: predicted risk percentile", "Backtest 2025-06-20: actual complaints"):
        assert layer in html
    assert (tmp_path / "figures" / "risk_maps.png").stat().st_size > 5_000
