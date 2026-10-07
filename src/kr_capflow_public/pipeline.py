from __future__ import annotations

import pandas as pd

from .features import INVESTORS, build_daily_flow_frame, build_weekly_participation
from .regimes import StructuralRegimeSpec, classify_structural_regime
from .forecast import walk_forward_flow_forecast, evaluate_forecasts


def run_public_pipeline(
    market: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 5),
    min_train: int = 120,
    stride: int = 5,
) -> dict[str, pd.DataFrame]:
    frame, features = build_daily_flow_frame(market, horizons=horizons)
    weekly = build_weekly_participation(frame)
    regimes = classify_structural_regime(weekly, spec=StructuralRegimeSpec())

    details = []
    for inv in INVESTORS:
        for h in horizons:
            d = walk_forward_flow_forecast(
                frame, features, investor=inv, horizon=h,
                min_train=min_train, stride=stride,
            )
            if not d.empty:
                details.append(d)
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    metrics = evaluate_forecasts(detail)
    return {
        "daily_frame": frame,
        "weekly_regimes": regimes,
        "forecast_detail": detail,
        "forecast_metrics": metrics,
    }
