import numpy as np
import pandas as pd
import pytest

from kr_capflow_public.features import build_daily_flow_frame


def sample(n=40):
    d = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({
        "DATE": d,
        "INDEX_RETURN": np.linspace(-0.01, 0.01, n),
        "TRADING_VALUE_MN": np.linspace(1000, 1200, n),
        "FOREIGN_NET_MN": np.sin(np.arange(n)) * 10,
        "INSTITUTION_NET_MN": np.cos(np.arange(n)) * 8,
        "RETAIL_NET_MN": -(np.sin(np.arange(n)) * 10 + np.cos(np.arange(n)) * 8),
    })


def test_build_daily_frame_has_targets_and_lags():
    frame, features = build_daily_flow_frame(sample(), horizons=(1,))
    assert "FOREIGN_NORM_LAG1" in features
    assert "TARGET_H1_FOREIGN_NORM_FLOW" in frame.columns
    assert frame["ADTV20_MN"].notna().sum() == 21


def test_out_of_order_rows_fail_closed():
    x = sample()
    x = pd.concat([x.iloc[:10], x.iloc[11:12], x.iloc[10:11], x.iloc[12:]], ignore_index=True)
    with pytest.raises(ValueError, match="chronological"):
        build_daily_flow_frame(x, horizons=(1,))
