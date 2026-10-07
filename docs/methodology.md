# Methodology

## 1. Research object

The public edition treats future investor flow as the primary forecast object. For investor `i`, normalized flow is:

`normalized_flow(i,t) = net_flow(i,t) / ADTV20(t)`

where `ADTV20(t)` is a trailing 20-session average trading-value proxy known at the forecast origin.

## 2. Point-in-time feature discipline

The feature layer enforces chronological ordering and constructs only backward-looking transformations. The public implementation includes:

- current normalized flow;
- 1/2/3-session lags;
- trailing 3/5-session means;
- trailing 5-session positive-flow share;
- index return, trading-value change and weekday controls.

Future targets are kept separate from predictor construction.

## 3. Forecast evaluation

The demo uses a fixed-alpha ridge model to keep the implementation transparent. For each origin:

1. training rows end before the origin;
2. rows whose target would not yet have matured are excluded;
3. the model predicts future normalized flow;
4. the model is compared with a persistence benchmark based on recent observed flow;
5. magnitude error is measured with MAE and direction probability with Brier score.

The public demo is intentionally simple. It is a method demonstration, not the complete private model stack.

## 4. Structural regime

Daily absolute normalized-flow magnitudes are aggregated to weekly participation shares. A participant can become the structural leader only when its share exceeds both a minimum dominance level and a gap over the second-largest participant. A persistence rule prevents one-week flips from immediately changing the confirmed regime.

The thresholds shipped here are illustrative public defaults and are not represented as the exact private research thresholds.

## 5. Sector layer

The sector utilities expose two reusable operations:

- benchmark-relative sector return;
- investor-flow aggregation by date and sector.

No licensed constituent history or actual sector data are included.

## 6. Claim boundary

This repository demonstrates research engineering and validation structure. It does not establish causality, trading profitability or production eligibility. Any live empirical conclusion requires licensed/official data, frozen specifications and independent validation outside this public demo.
