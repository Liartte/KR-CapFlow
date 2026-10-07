import pandas as pd
from kr_capflow_public.regimes import classify_structural_regime, StructuralRegimeSpec


def test_persistence_prevents_single_week_flip():
    x = pd.DataFrame({
        "WEEK_END": pd.date_range("2025-01-03", periods=5, freq="7D"),
        "FOREIGN_SHARE": [0.60, 0.61, 0.20, 0.62, 0.63],
        "INSTITUTION_SHARE": [0.20, 0.19, 0.60, 0.18, 0.17],
        "RETAIL_SHARE": [0.20, 0.20, 0.20, 0.20, 0.20],
    })
    out = classify_structural_regime(x, spec=StructuralRegimeSpec(persistence_weeks=2))
    assert out.loc[1, "STRUCTURAL_REGIME"] == "FOREIGN_LED"
    assert out.loc[2, "STRUCTURAL_REGIME"] == "FOREIGN_LED"
