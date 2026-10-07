from __future__ import annotations

import pandas as pd


def build_sector_excess_return(
    sector: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    date_col: str = "DATE",
    sector_col: str = "SECTOR",
    sector_return_col: str = "SECTOR_RETURN",
    benchmark_return_col: str = "BENCHMARK_RETURN",
) -> pd.DataFrame:
    required_s = [date_col, sector_col, sector_return_col]
    required_b = [date_col, benchmark_return_col]
    if any(c not in sector.columns for c in required_s) or any(c not in benchmark.columns for c in required_b):
        raise ValueError("sector/benchmark input schema mismatch")
    s = sector.copy(); b = benchmark.copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    b[date_col] = pd.to_datetime(b[date_col], errors="coerce")
    if b[date_col].duplicated().any():
        raise ValueError("benchmark dates must be unique")
    out = s.merge(b[[date_col, benchmark_return_col]], on=date_col, how="left", validate="many_to_one")
    out["SECTOR_EXCESS_RETURN"] = pd.to_numeric(out[sector_return_col], errors="coerce") - pd.to_numeric(out[benchmark_return_col], errors="coerce")
    return out


def aggregate_sector_flow(
    panel: pd.DataFrame,
    *,
    date_col: str = "DATE",
    sector_col: str = "SECTOR",
    investor_col: str = "INVESTOR",
    flow_col: str = "NET_FLOW",
) -> pd.DataFrame:
    required = [date_col, sector_col, investor_col, flow_col]
    if any(c not in panel.columns for c in required):
        raise ValueError("sector-flow input schema mismatch")
    x = panel.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x[flow_col] = pd.to_numeric(x[flow_col], errors="coerce")
    return x.groupby([date_col, sector_col, investor_col], as_index=False)[flow_col].sum(min_count=1)
