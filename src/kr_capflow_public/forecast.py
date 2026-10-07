from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .features import INVESTORS


@dataclass(frozen=True)
class RidgeState:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    beta: np.ndarray

    def predict(self, row: np.ndarray) -> float:
        z = (np.asarray(row, dtype=float) - self.x_mean) / self.x_scale
        return float(self.y_mean + z @ self.beta)


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[RidgeState, np.ndarray]:
    x_mean = X.mean(axis=0)
    x_scale = X.std(axis=0, ddof=1)
    x_scale = np.where(np.isfinite(x_scale) & (x_scale > 1e-12), x_scale, 1.0)
    y_mean = float(y.mean())
    Z = (X - x_mean) / x_scale
    yc = y - y_mean
    penalty = alpha * np.eye(Z.shape[1])
    beta = np.linalg.solve(Z.T @ Z + penalty, Z.T @ yc)
    fitted = y_mean + Z @ beta
    return RidgeState(x_mean, x_scale, y_mean, beta), y - fitted


def walk_forward_flow_forecast(
    frame: pd.DataFrame,
    features: list[str],
    *,
    investor: str = "FOREIGN",
    horizon: int = 1,
    min_train: int = 120,
    alpha: float = 5.0,
    stride: int = 5,
) -> pd.DataFrame:
    """Expanding-window forecast using only rows available before each origin.

    The alpha/min_train/stride defaults are public demonstration parameters and
    are intentionally not represented as a private frozen model specification.
    """
    investor = investor.upper()
    if investor not in INVESTORS:
        raise ValueError(f"investor must be one of {INVESTORS}")
    target = f"TARGET_H{horizon}_{investor}_NORM_FLOW"
    baseline = f"PERSIST_H{horizon}_{investor}_NORM_FLOW"
    maturity = f"TARGET_H{horizon}_MATURITY_DATE"
    required = ["DATE", target, baseline, maturity, *features]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    x = frame.copy()
    rows: list[dict] = []
    valid_origins = x.index[x[features].notna().all(axis=1) & x[target].notna() & x[maturity].notna()].tolist()
    candidates = valid_origins[min_train::max(1, stride)]

    for idx in candidates:
        origin = pd.Timestamp(x.at[idx, "DATE"])
        train = x.loc[: idx - 1].copy()
        # Target labels are usable only after their economic horizon has matured.
        train = train[pd.to_datetime(train[maturity], errors="coerce").le(origin)]
        train = train.dropna(subset=[target, *features])
        if len(train) < min_train:
            continue
        X = train[features].to_numpy(float)
        y = train[target].to_numpy(float)
        state, residuals = _fit_ridge(X, y, alpha)
        pred = state.predict(x.loc[idx, features].to_numpy(float))
        actual = float(x.at[idx, target])
        persistence = float(x.at[idx, baseline]) if pd.notna(x.at[idx, baseline]) else np.nan
        base_rate = float(np.mean(y > 0))
        if len(residuals):
            prob_buy = float(np.mean((pred + residuals) > 0))
        else:
            prob_buy = float(pred > 0)
        rows.append({
            "ORIGIN": origin,
            "MATURITY": pd.Timestamp(x.at[idx, maturity]),
            "INVESTOR": investor,
            "HORIZON": horizon,
            "PREDICTION": pred,
            "ACTUAL": actual,
            "PERSISTENCE": persistence,
            "PROB_NET_BUY": prob_buy,
            "BASE_RATE_PROB": base_rate,
        })
    return pd.DataFrame(rows)


def evaluate_forecasts(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=[
            "INVESTOR", "HORIZON", "N_OOS", "MAE", "PERSISTENCE_MAE",
            "MAE_IMPROVEMENT_VS_PERSISTENCE", "DIRECTION_ACCURACY",
            "BRIER", "BASE_RATE_BRIER", "BRIER_IMPROVEMENT_VS_BASE_RATE",
        ])
    rows = []
    for (investor, horizon), g in detail.groupby(["INVESTOR", "HORIZON"], sort=True):
        err = (g["ACTUAL"] - g["PREDICTION"]).abs()
        p_err = (g["ACTUAL"] - g["PERSISTENCE"]).abs()
        actual_dir = (g["ACTUAL"] > 0).astype(float)
        pred_dir = (g["PREDICTION"] > 0).astype(float)
        brier = (g["PROB_NET_BUY"] - actual_dir).pow(2)
        base_brier = (g["BASE_RATE_PROB"] - actual_dir).pow(2)
        mae = float(err.mean())
        pmae = float(p_err.mean())
        b = float(brier.mean())
        bb = float(base_brier.mean())
        rows.append({
            "INVESTOR": investor,
            "HORIZON": int(horizon),
            "N_OOS": int(len(g)),
            "MAE": mae,
            "PERSISTENCE_MAE": pmae,
            "MAE_IMPROVEMENT_VS_PERSISTENCE": (pmae - mae) / pmae if pmae > 0 else np.nan,
            "DIRECTION_ACCURACY": float((pred_dir == actual_dir).mean()),
            "BRIER": b,
            "BASE_RATE_BRIER": bb,
            "BRIER_IMPROVEMENT_VS_BASE_RATE": (bb - b) / bb if bb > 0 else np.nan,
        })
    return pd.DataFrame(rows)
