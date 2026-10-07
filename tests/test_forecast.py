import numpy as np
import pandas as pd
from kr_capflow_public.features import build_daily_flow_frame
from kr_capflow_public.forecast import walk_forward_flow_forecast, evaluate_forecasts


def test_walk_forward_produces_oos_rows():
    n = 260
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=n)
    f = np.zeros(n)
    for i in range(1, n):
        f[i] = 0.7 * f[i-1] + rng.normal(0, 5)
    inst = rng.normal(0, 5, n)
    retail = -(f + inst) + rng.normal(0, 1, n)
    market = pd.DataFrame({
        "DATE": dates,
        "INDEX_RETURN": rng.normal(0, 0.01, n),
        "TRADING_VALUE_MN": rng.uniform(900, 1300, n),
        "FOREIGN_NET_MN": f,
        "INSTITUTION_NET_MN": inst,
        "RETAIL_NET_MN": retail,
    })
    frame, features = build_daily_flow_frame(market, horizons=(1,))
    d = walk_forward_flow_forecast(frame, features, min_train=80, stride=10)
    m = evaluate_forecasts(d)
    assert len(d) > 5
    assert m.loc[0, "N_OOS"] == len(d)
    assert np.isfinite(m.loc[0, "MAE"])
