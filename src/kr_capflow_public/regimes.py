from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StructuralRegimeSpec:
    """Illustrative public parameters; not the private frozen specification."""

    dominance_min_share: float = 0.50
    dominance_gap: float = 0.10
    persistence_weeks: int = 2
    share_sum_tolerance: float = 0.02

    def validate(self) -> "StructuralRegimeSpec":
        if not 0 <= self.dominance_min_share <= 1:
            raise ValueError("dominance_min_share must be in [0, 1]")
        if not 0 <= self.dominance_gap <= 1:
            raise ValueError("dominance_gap must be in [0, 1]")
        if self.persistence_weeks <= 0:
            raise ValueError("persistence_weeks must be positive")
        return self


def classify_structural_regime(
    weekly: pd.DataFrame,
    *,
    spec: StructuralRegimeSpec | None = None,
) -> pd.DataFrame:
    spec = (spec or StructuralRegimeSpec()).validate()
    required = ["WEEK_END", "FOREIGN_SHARE", "INSTITUTION_SHARE", "RETAIL_SHARE"]
    missing = [c for c in required if c not in weekly.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    x = weekly.copy()
    x["WEEK_END"] = pd.to_datetime(x["WEEK_END"], errors="coerce")
    if x["WEEK_END"].isna().any() or x["WEEK_END"].duplicated().any():
        raise ValueError("invalid/duplicate WEEK_END")
    if not x["WEEK_END"].is_monotonic_increasing:
        raise ValueError("weekly rows must be chronological")

    raw: list[str] = []
    for _, row in x.iterrows():
        shares = {
            "FOREIGN": float(row["FOREIGN_SHARE"]) if pd.notna(row["FOREIGN_SHARE"]) else np.nan,
            "INSTITUTION": float(row["INSTITUTION_SHARE"]) if pd.notna(row["INSTITUTION_SHARE"]) else np.nan,
            "RETAIL": float(row["RETAIL_SHARE"]) if pd.notna(row["RETAIL_SHARE"]) else np.nan,
        }
        vals = list(shares.values())
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in vals) or abs(sum(vals) - 1.0) > spec.share_sum_tolerance:
            raw.append("INSUFFICIENT_DATA")
            continue
        ordered = sorted(shares.items(), key=lambda kv: (-kv[1], kv[0]))
        lead, top = ordered[0]
        second = ordered[1][1]
        raw.append(f"{lead}_LED" if top >= spec.dominance_min_share and top - second >= spec.dominance_gap else "NO_CLEAR_DOMINANT")

    confirmed: list[str] = []
    status: list[str] = []
    previous: str | None = None
    candidate: str | None = None
    run = 0
    for label in raw:
        if label == "INSUFFICIENT_DATA":
            confirmed.append("INSUFFICIENT_DATA")
            status.append("INSUFFICIENT_DATA")
            candidate, run = None, 0
            continue
        if label == previous:
            confirmed.append(label)
            status.append("STABLE")
            candidate, run = None, 0
            continue
        if label == candidate:
            run += 1
        else:
            candidate, run = label, 1
        if run >= spec.persistence_weeks:
            previous = label
            confirmed.append(label)
            status.append("CONFIRMED_SWITCH")
            candidate, run = None, 0
        elif previous is None:
            confirmed.append("MIXED_TRANSITION")
            status.append(f"UNCONFIRMED_{label}")
        else:
            confirmed.append(previous)
            status.append(f"TRANSITION_TO_{label}")

    x["STRUCTURAL_REGIME_RAW"] = raw
    x["STRUCTURAL_REGIME"] = confirmed
    x["REGIME_STATUS"] = status
    return x
