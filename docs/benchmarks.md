# Benchmarks

The full LightGBM model against simpler and classical methods, all scored the same way on the test block
(2023-03-01 to 2023-04-15, 145,116 rows, 69 segments). The other methods all get the same 11 inputs: segment
keys, hour and weekday, installed capacity, temperature and direct radiation, and the target 2 days ago,
7 days ago and its 7-day mean. MAE is in kWh per segment-hour; lower is better. Bias is the share of energy
over- (+) or under- (-) forecast.

| Model | MAE | Production MAE | Consumption MAE | Production bias | Consumption bias |
|---|---:|---:|---:|---:|---:|
| LightGBM, full model (v6) | 50.70 | 41.95 | 59.46 | -15.7% | +2.4% |
| LightGBM, 11 inputs, one model per side | 67.31 | 59.85 | 74.78 | -38.2% | +3.7% |
| Prophet | 81.36 | 63.20 | 99.51 | -31.3% | +0.7% |
| SARIMAX | 90.63 | 72.40 | 108.87 | -43.9% | -6.1% |
| TSB | 96.82 | 75.15 | 118.49 | -21.3% | +4.7% |
| Croston | 96.83 | 75.14 | 118.52 | -21.3% | +4.7% |
| DeepAR | 97.80 | 98.12 | 97.48 | -83.1% | +5.1% |
| Naive: same hour a week earlier | 100.22 | 85.91 | 114.54 | 0.0% | 0.0% |

The first time I ran this, the classical models (SARIMAX, Prophet, DeepAR) were fitted once and then
forecast all 46 days ahead without seeing any new actuals, while the naive baseline used last week's real
values every day. They all came out worse than the naive forecast, mostly because they missed how fast
solar production grows in spring. Once every method got the same recent actuals as inputs, all of them beat
the naive forecast, and the full LightGBM model still had about a quarter less error than a plain LightGBM
on the same 11 inputs.

Run them with `python -m enefit.benchmarks.runner --model all` (summary table and chart:
`python -m enefit.benchmarks.runner --summary-only`).
