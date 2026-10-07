from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd

INVESTORS = ("FOREIGN", "INSTITUTION", "RETAIL")
FLOW_COLUMNS = {k: f"{k}_NET_MN" for k in INVESTORS}


def _require_columns(df: pd.DataFrame, cols: Iterable[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing columns {missing}")


def assert_chronological(df: pd.DataFrame, date_col: str = "DATE") -> None:
    d = pd.to_datetime(df[date_col], errors="coerce")
    if d.isna().any():
        raise ValueError("invalid dates")
    if d.duplicated().any():
        raise ValueError("duplicate dates")
    if not d.is_monotonic_increasing:
        raise ValueError("rows must be strictly chronological; implicit sorting is prohibited")


def future_sum(s: pd.Series, horizon: int) -> pd.Series:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    x = pd.to_numeric(s, errors="coerce")
    parts = [x.shift(-j) for j in range(1, horizon + 1)]
    return pd.concat(parts, axis=1).sum(axis=1, min_count=horizon)


def build_daily_flow_frame(
    market: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 5),
    liquidity_window: int = 20,
) -> tuple[pd.DataFrame, list[str]]:
    """Create a compact PIT-style feature frame.

    The public edition expects already-observed daily values and does not attempt
    to refresh external data providers. Rows must arrive in chronological order.
    """
    required = ["DATE", "INDEX_RETURN", "TRADING_VALUE_MN", *FLOW_COLUMNS.values()]
    _require_columns(market, required, "daily market panel")
    x = market.copy()
    x["DATE"] = pd.to_datetime(x["DATE"], errors="coerce").dt.normalize()
    assert_chronological(x)

    tv = pd.to_numeric(x["TRADING_VALUE_MN"], errors="coerce")
    x["ADTV20_MN"] = tv.rolling(liquidity_window, min_periods=liquidity_window).mean()
    x["INDEX_RETURN"] = pd.to_numeric(x["INDEX_RETURN"], errors="coerce")
    x["TV_LOG_CHG_1"] = np.log(tv.where(tv > 0)).diff()
    x["ORIGIN_WEEKDAY"] = x["DATE"].dt.weekday.astype(float)

    features = ["INDEX_RETURN", "TV_LOG_CHG_1", "ORIGIN_WEEKDAY"]
    for inv in INVESTORS:
        net = pd.to_numeric(x[FLOW_COLUMNS[inv]], errors="coerce")
        norm = net / x["ADTV20_MN"]
        x[f"{inv}_NORM_CUR"] = norm
        for lag in (1, 2, 3):
            x[f"{inv}_NORM_LAG{lag}"] = norm.shift(lag)
            features.append(f"{inv}_NORM_LAG{lag}")
        x[f"{inv}_NORM_ROLL3"] = norm.rolling(3, min_periods=3).mean()
        x[f"{inv}_NORM_ROLL5"] = norm.rolling(5, min_periods=5).mean()
        x[f"{inv}_POS_SHARE5"] = norm.gt(0).where(norm.notna()).rolling(5, min_periods=5).mean()
        features.extend([f"{inv}_NORM_CUR", f"{inv}_NORM_ROLL3", f"{inv}_NORM_ROLL5", f"{inv}_POS_SHARE5"])

        for h in horizons:
            denom = float(h) * x["ADTV20_MN"]
            x[f"TARGET_H{h}_{inv}_NORM_FLOW"] = future_sum(net, h) / denom
            x[f"PERSIST_H{h}_{inv}_NORM_FLOW"] = net.rolling(h, min_periods=h).sum() / denom
            x[f"TARGET_H{h}_MATURITY_DATE"] = x["DATE"].shift(-h)

    return x, features


def build_weekly_participation(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate absolute normalized-flow magnitudes to weekly participation shares."""
    _require_columns(frame, ["DATE", *[f"{i}_NORM_CUR" for i in INVESTORS]], "weekly participation")
    x = frame[["DATE", *[f"{i}_NORM_CUR" for i in INVESTORS]]].copy()
    x["WEEK_END"] = pd.to_datetime(x["DATE"]).dt.to_period("W-FRI").dt.end_time.dt.normalize()
    for inv in INVESTORS:
        x[f"{inv}_ABS"] = pd.to_numeric(x[f"{inv}_NORM_CUR"], errors="coerce").abs()
    cols = [f"{i}_ABS" for i in INVESTORS]
    weekly = x.groupby("WEEK_END", as_index=False)[cols].sum(min_count=1)
    denom = weekly[cols].sum(axis=1, min_count=1)
    for inv in INVESTORS:
        weekly[f"{inv}_SHARE"] = weekly[f"{inv}_ABS"] / denom.replace(0, np.nan)
    return weekly[["WEEK_END", *[f"{i}_SHARE" for i in INVESTORS]]]
