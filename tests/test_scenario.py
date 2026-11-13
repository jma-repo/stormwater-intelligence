from types import SimpleNamespace

import pandas as pd
import pytest

from stormwater_intel import scenario as S
from tests.test_panel import COMMS, _toy


def _inputs():
    events, y, rain, statics = _toy()
    return SimpleNamespace(y_daily=y, events=events, statics=statics, aux_daily={},
                           fields={"total": rain, "m60": rain, "k3": rain})


def test_design_storm_frame_uses_scenario_rain_and_all_real_history():
    row = S.design_storm_frame(_inputs(), rain_mm=40.0, peak_60min_mm=20.0).set_index("comm_code")
    assert list(row.index) == COMMS
    assert (row["rain_win_mm"] == 40).all() and (row["rain_60min_mm"] == 20).all()
    assert row["rain_prev7_mm"].tolist() == pytest.approx([10.0] * len(COMMS))
    assert row["rain_prev30_mm"].tolist() == pytest.approx([40.0] * len(COMMS))
    # history comes from every real event and the whole previous year of complaints
    assert row.loc["AAA", "h365"] == 1 + 3 + 2 + 4 and row.loc["BBB", "h365"] == 1
    assert row.loc["AAA", "hist_ev_mean"] == pytest.approx(4.5) and row.loc["BBB", "hist_ev_mean"] == pytest.approx(0.5)
    assert (row["hist_ev_n"] == 2).all()


class _DummyModel:
    def predict(self, frame):                      # riskier where more complaints were filed last year
        return frame["h365"].to_numpy() + 1.0


def test_score_design_storm_ranks_and_normalises_per_dwelling():
    inp = _inputs()
    inp.universe = pd.DataFrame({"comm_code": COMMS, "name": [c.title() for c in COMMS],
                                 "sector": "X", "comm_structure": "Y"})
    inp.statics = inp.statics.assign(dwell_cnt=[800.0] * 5 + [100.0])     # last community is below the floor
    ranking = S.score_design_storm(inp, _DummyModel(), features=["h365"])
    assert ranking.iloc[0]["comm_code"] == "AAA" and ranking.iloc[0]["risk_rank"] == 1
    assert ranking["risk_percentile"].max() == 100 and ranking["risk_percentile"].min() > 0
    assert ranking["expected_share_pct"].sum() == pytest.approx(100.0)
    top = ranking.iloc[0]
    assert top["expected_per_1000_dwellings"] == pytest.approx(1000 * top["expected_complaints"] / 800)
    assert pd.isna(ranking.set_index("comm_code").loc["FFF", "expected_per_1000_dwellings"])


def test_design_storm_overrides_seasonal_context_not_the_data_end_date():
    row = S.design_storm_frame(_inputs(), doy=200, days_since_prev_event=10.0)
    assert (row["doy"] == 200).all() and (row["days_since_prev_event"] == 10.0).all()
    assert row["start"].iloc[0] == pd.Timestamp("2021-01-01")      # window opens the day after the data ends
