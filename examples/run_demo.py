from pathlib import Path
import pandas as pd

from kr_capflow_public.pipeline import run_public_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_kr_capflow.csv"
OUT = ROOT / "artifacts"

market = pd.read_csv(DATA, parse_dates=["DATE"])
result = run_public_pipeline(market)
OUT.mkdir(exist_ok=True)
result["forecast_metrics"].to_csv(OUT / "demo_metrics.csv", index=False)
result["weekly_regimes"].tail(12).to_csv(OUT / "demo_regime_snapshot.csv", index=False)

print("KR-CapFlow public demo")
print("Synthetic rows:", len(market))
print("\nForecast metrics (synthetic data only):")
print(result["forecast_metrics"].to_string(index=False))
print("\nLatest structural regimes:")
print(result["weekly_regimes"].tail(5)[["WEEK_END", "STRUCTURAL_REGIME", "REGIME_STATUS"]].to_string(index=False))
