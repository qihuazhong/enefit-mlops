# Milestone variants: evidence-based diff of the Enefit LightGBM training notebook

Milestones are numbered v1-v6 in order; the Kaggle notebook versions they come from: v1 = Kaggle version 84, v2 = 125, v3 = 158, v4 = 190, v5 = 217, v6 = 238
(notebook files `reference/kaggle_notebooks/v1.ipynb` ... `v6.ipynb`). "Version NNN" quotes below are Kaggle version-log
entries, in Kaggle's own numbering.

Scope: `reference/kaggle_notebooks/v1.ipynb` (2023-11-08), `v2` (2023-11-15), `v3` (2023-12-31), `v4` (2024-01-05), `v5` (2024-01-09), `v6` (2024-01-15), and `loaded-trained-models.ipynb` (column **final**; inference + optional online training around the pickled models `0127e_custom_3fcv`, trained 2024-01-27).

Full cell-by-cell evidence (code quotes per row, full feature lists, model counts, open questions): git history, commit 1aa2d24.

## 1. Summary table

Side 0 is production (`is_consumption=0`) and side 1 is consumption. Written "P / C" when the two sides differ.

| Row | v1 | v2 | v3 | v4 | v5 | v6 | final (0127e) |
|---|---|---|---|---|---|---|---|
| Headline change | single tweedie model per side | 7 targets/side (target + 3 pos/neg diff pairs), weighted-grid weather | + data augmentation (200k rows) | + weather bias correction, 6 folds, kurt/skew | CV by **day**, counterpart-lag bug fixed, bias correction off, 400k aug | float16 data, reg_sqrt, ff 0.7 for diff targets, LightGBM 4.2.0 pin | 11 targets × {raw, scaled}, 3-fold "custom" CV, `day_category` |
| CV | GroupKFold(5), G=`month` | GroupKFold(5), G=`month` | GroupKFold(5), G=`month` | GroupKFold(6), G=`month` | GroupKFold(6), G=`day` | GroupKFold(6), G=`day` | 3 folds per model (artifact); grouping **not determined** |
| objective | tweedie | tweedie | tweedie | tweedie | tweedie | tweedie | tweedie |
| tweedie_variance_power P / C | not set (LightGBM default) | 1.4 / 1.3 | 1.4 / 1.3 | 1.4 / 1.3 | 1.4 / 1.3 | 1.4 / 1.25 | 1.4 / 1.25 |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 (the 0.02 = `0.1 / 5` exists only in disabled online fine-tuning) |
| num_leaves / max_depth | 127 / 7 | 127 / 7 | 127 / 7 | 127 / 7 | 127 / 7 | 127 / 7 | 127 / 7 |
| feature_fraction | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 for `target`, 0.7 for diff targets | 0.9 for `target`/`scaled_target`, 0.7 otherwise |
| bagging | not set | not set | not set | not set | not set | not set (commented) | bagging_fraction 1, bagging_freq 0 |
| lambda_l1 / lambda_l2 (reg_alpha / reg_lambda) | not set | 0.2 / 0.2 | 0.2 / 0.2 | 0.2 / 0.2 | P 0.2/0.2, C 1.2/1.2 | P 0.2/0.2, C 1.2/1.2 | P 0.2/0.2, C 1.2/1.2 |
| cat_l2 | 0.1 | 0.1 | 0.1 | 0.1 | P 0.1, C 1.1 | P 0.1, C 1.1 | P 0.1, C 1.1 |
| reg_sqrt | not set (commented) | not set (commented) | not set (commented) | not set | not set | True | 1 |
| max_bin / tree_learner | not set | not set | not set | not set | not set | not set (both commented) | 255 / serial |
| num_boost_round (max) | 3000 | 1000 | 1000 | 1000 | 1400 | 1600 | 2000 |
| early-stopping rounds | 50 | 100 | 100 | 100 | 100 | 100 | not in params. Inferred from trees − CV-history length: 200 in 34 models, 100 in 8 (2 hit the 2000 cap) |
| categorical features | county, product_type, daytype, dayofweek | same | same, + `hour` for C | same as v3 | same as v3 | same as v3 | product_type, day_category, county (+ hour for C). **Exception:** 8 production scaled-diff models use product_type, county, dayofweek, daytype |
| monotone constraints | P: euros_per_mwh +1, installed_capacity +1, direct_solar_radiation +1, cloudcover_total −1. C: euros_per_mwh −1, eic_count +1 | P: installed_capacity +1, direct_solar_radiation +1, cloudcover_total/mid/low −1. C: eic_count +1. The intended `_neg` sign flip is a no-op (section 7) | same as v2 | same | same | same | same values in the artifacts, `_neg` models included |
| modeled targets per side | `target` | `target`, `lag_7_day_mean_diff(_neg)`, `lag_2_day_diff(_neg)`, `lag_7_day_diff(_neg)` (7) | same 7 | same 7 | same 7 | same 7 | 11 targets: + `lag_7_day_1h_mean_diff(_neg)`, `lag_3_day_diff(_neg)`; each raw and `scaled_` (22) |
| raw vs scaled | raw | raw | raw (scaled path exists but is off) | raw (off) | raw (off) | raw (off) | raw + scaled |
| scaling formula | — | — | `installed_capacity/1000` in `get_prediciton` but never used | same, unused | same, unused | same, unused | P: `y/(installed_capacity.fillna(0)+1000)*1000`. C: `y/(eic_count.fillna(0)+100)*100` |
| augmentation | none | none | last 200k rows **in index order** (not by date), factor ~N(1.2, 0.1), no seed, 13 cols | last 200k rows by datetime, U[1, 1.6), seed 42, 27 cols | 400k rows, U[1, 1.6), seed 42, 30 cols | 400k, U[1, 1.6), seed 42, 32 cols | final-notebook code: 400k, U[1, 1.6), seed 42, 40 cols. Whether 0127e used it: not determined |
| ensemble / reconstruction weights | mean over fold boosters | target 2/7, (lag_7_day_mean + Δ) 2/7, (lag_2_day + Δ) 2/7, (lag_7_day + Δ) 1/7, both sides | same | same | same | same | 12 reconstructions × [2,2,1,2,1,1]/18 per prefix, both sides |
| weather bias correction | no (function absent) | no (absent) | no (absent) | **ON** (train and inference) | off (call removed, function body inert) | off | off (call commented) |
| training filters | `datetime < 2023-05-28` + `dropna(how='any')` | `datetime < 2023-05-28` + target not NaN | target not NaN | target not NaN | target not NaN | target not NaN; `filter_hours_ahead=False` | final-notebook code: target not NaN, `filter_hours_ahead=False`. For 0127e: not determined |
| sample weights | — | month weights computed, **not passed** | month weights computed, not passed | recency weights computed, not passed | same | same (`weight_column` commented) | none computed |
| float precision of train frame | default float64 | default | default | default | default | **float16** cast before lags | float32 (final notebook) |
| feature count, `target` model, P / C | 31 / 31 | 37 / 28 | 39 / 29 | 40 / 33 | 42 / 41 | 43 / 42 | 46 / 47 (artifacts) |
| feature-count range over all models, P / C | 31 / 31 | 37–43 / 28–34 | 39–48 / 29–37 | 40–50 / 33–42 | 42–54 / 38–48 | 43–56 / 39–49 | 46–59 / 46–56 |
| LightGBM version | not determined | not determined | not determined | not determined | not determined | 4.2.0 wheel installed | 4.2.0 wheel installed |
| Kaggle wall-clock (given) | 0.2 h | 1.8 h | 1.9 h | 2.6 h | 4.6 h | 4.1 h | not provided |

## 2. Feature-set diff between consecutive milestones

The diff is mechanical. For v1–v6, lists come from executing each notebook's own `select_features`. For "final", they come from artifact `feature_name()` of the unscaled models. Each transition shows the `target` model, which has the same "base" set as `lag_7_day_mean_diff`, plus the union over all target models of that version. `_neg` models always have exactly the same features as their positive twin (asserted during extraction). Per-model counts, per-target deltas and the full `target`-model lists follow below.


### Side 0: production (is_consumption=0)

| transition | model compared | +added (n) | −removed (n) |
|---|---|---|---|
| v1→v2 | `target` model | 12: `cloudcover_low_2_day_diff`, `cloudcover_low_7_day_mean_diff`, `direct_solar_radiation_2_day_diff`, `direct_solar_radiation_7_day_mean_diff`, `snowfall_1_day_rolling_sum`, `surface_solar_radiation_downwards_2_day_diff`, `surface_solar_radiation_downwards_7_day_mean_diff`, `temp_minus_dewpoint`, `temperature`, `total_precipitation`, `total_precipitation_2_day_diff`, `total_precipitation_7_day_mean_diff` | 6: `dewpoint`, `euros_per_mwh`, `lag_4_day`, `lag_5_day`, `lag_6_day`, `lag_7_day_consumption_counterpart` |
| v2→v3 | `target` model | 2: `eic_count_7_day_mean_diff`, `installed_capacity_7_day_mean_diff` | 0: — |
| v2→v3 | union over all target models | 6: `eic_count_7_day_diff`, `eic_count_7_day_mean_diff`, `euros_per_mwh_lag_2_day_diff`, `euros_per_mwh_lag_7_day_diff`, `installed_capacity_7_day_diff`, `installed_capacity_7_day_mean_diff` | 0: — |
| v3→v4 | `target` model | 1: `cloudcover_total_7_day_mean_diff` | 0: — |
| v3→v4 | union over all target models | 3: `cloudcover_total_2_day_diff`, `cloudcover_total_7_day_diff`, `cloudcover_total_7_day_mean_diff` | 0: — |
| v4→v5 | `target` model | 2: `daytype_lag_3_day`, `snowfall_1_day_rolling_sum_lag_2_day_diff` | 0: — |
| v4→v5 | union over all target models | 4: `daytype_lag_3_day`, `lag_7_day_1h`, `lag_7_day_m1h`, `snowfall_1_day_rolling_sum_lag_2_day_diff` | 0: — |
| v5→v6 | `target` model | 1: `direct_solar_radiation_2_day_1h_diff` | 0: — |
| v5→v6 | union over all target models | 3: `diffuse_radiation_lag_2_day_h`, `diffuse_radiation_lag_7_day_h`, `direct_solar_radiation_2_day_1h_diff` | 0: — |
| v6→final | `target` model | 6: `consumption_counterpart_all_lag_7_day`, `cos_month`, `day_category`, `day_category_lag_3_day`, `lag_8_14_day_mean`, `sin_month` | 3: `dayofweek`, `daytype`, `daytype_lag_3_day` |
| v6→final | union over all target models | 21: `cloudcover_total_3_day_diff`, `consumption_counterpart_all_lag_2_day`, `consumption_counterpart_all_lag_3_day`, `consumption_counterpart_all_lag_7_day`, `cos_month`, `day_category`, `day_category_lag_2_day`, `day_category_lag_3_day`, `day_category_lag_7_day`, `diffuse_radiation_lag_3_day_h`, `direct_solar_radiation_3_day_1h_diff`, `direct_solar_radiation_3_day_diff`, `lag_3_day_1h`, `lag_7_day_1h_mean`, `lag_8_14_day_mean`, `ratio_lag_14_7_day`, `sin_month`, `surface_solar_radiation_downwards_3_day_diff`, `temperature_3_day_1h_diff`, `temperature_3_day_diff`, `total_precipitation_3_day_diff` | 2: `euros_per_mwh_lag_2_day_diff`, `euros_per_mwh_lag_7_day_diff` |

### Side 1: consumption (is_consumption=1)

| transition | model compared | +added (n) | −removed (n) |
|---|---|---|---|
| v1→v2 | `target` model | 5: `euros_per_mwh_lag_7_day_mean_diff`, `lag_8_14_day_mean`, `snowfall_1_day_rolling_sum`, `temperature`, `total_precipitation` | 8: `10_metre_u_wind_component`, `10_metre_v_wind_component`, `cloudcover_mid`, `cloudcover_total`, `dewpoint`, `lag_5_day`, `lag_6_day`, `lag_7_day_consumption_counterpart` |
| v2→v3 | `target` model | 3: `eic_count_7_day_mean_diff`, `hour`, `installed_capacity_7_day_mean_diff` | 2: `cos_hour`, `sin_hour` |
| v2→v3 | union over all target models | 7: `eic_count_7_day_diff`, `eic_count_7_day_mean_diff`, `euros_per_mwh_lag_2_day_diff`, `euros_per_mwh_lag_7_day_diff`, `hour`, `installed_capacity_7_day_diff`, `installed_capacity_7_day_mean_diff` | 2: `cos_hour`, `sin_hour` |
| v3→v4 | `target` model | 4: `cloudcover_total_7_day_mean_diff`, `lag_3_day_consumption_counterpart`, `rolling_kurt_24`, `rolling_skew_24` | 0: — |
| v3→v4 | union over all target models | 6: `cloudcover_total_2_day_diff`, `cloudcover_total_7_day_diff`, `cloudcover_total_7_day_mean_diff`, `lag_3_day_consumption_counterpart`, `rolling_kurt_24`, `rolling_skew_24` | 0: — |
| v4→v5 | `target` model | 8: `daytype_lag_3_day`, `direct_solar_radiation_7_day_mean_diff`, `rolling_mean_24_lag_7_day`, `temperature_2_day_1h_diff`, `temperature_7_day_diff`, `temperature_7_day_mean_diff`, `temperature_8_14_day_mean_diff`, `temperature_lag_1h` | 0: — |
| v4→v5 | union over all target models | 11: `daytype_lag_3_day`, `direct_solar_radiation_7_day_1h_diff`, `direct_solar_radiation_7_day_mean_diff`, `lag_7_day_1h`, `lag_7_day_m1h`, `rolling_mean_24_lag_7_day`, `temperature_2_day_1h_diff`, `temperature_7_day_1h_diff`, `temperature_7_day_mean_diff`, `temperature_8_14_day_mean_diff`, `temperature_lag_1h` | 0: — |
| v5→v6 | `target` model | 1: `direct_solar_radiation_2_day_1h_diff` | 0: — |
| v5→v6 | union over all target models | 1: `direct_solar_radiation_2_day_1h_diff` | 0: — |
| v6→final | `target` model | 10: `cos_hour`, `cos_month`, `day_category`, `day_category_lag_3_day`, `ratio_lag_21_7_day`, `ratio_lag_9_2_day`, `sin_hour`, `sin_month`, `target_all_lag_2_day`, `target_all_lag_7_day` | 5: `dayofweek`, `daytype`, `daytype_lag_3_day`, `euros_per_mwh`, `euros_per_mwh_lag_7_day_mean_diff` |
| v6→final | union over all target models | 29: `cloudcover_total_3_day_diff`, `cos_hour`, `cos_month`, `day_category`, `day_category_lag_2_day`, `day_category_lag_3_day`, `day_category_lag_7_day`, `direct_solar_radiation_3_day_1h_diff`, `direct_solar_radiation_3_day_diff`, `lag_3_day_1h`, `lag_5_day`, `lag_6_day`, `lag_7_day_1h_mean`, `ratio_lag_14_7_day`, `ratio_lag_21_7_day`, `ratio_lag_9_2_day`, `ratio_lag_9_7_day`, `ratio_rolling_mean_lag_9_7_day`, `sin_hour`, `sin_month`, `surface_solar_radiation_downwards_3_day_diff`, `surface_solar_radiation_downwards_7_day_mean_diff`, `target_all_lag_2_day`, `target_all_lag_7_day`, `temperature_14_7_day_diff`, `temperature_3_day_1h_diff`, `temperature_3_day_diff`, `temperature_7_day_1h_mean_diff`, `total_precipitation_3_day_diff` | 9: `dayofweek`, `daytype`, `daytype_lag_2_day`, `daytype_lag_3_day`, `daytype_lag_7_day`, `euros_per_mwh`, `euros_per_mwh_lag_2_day_diff`, `euros_per_mwh_lag_7_day_diff`, `euros_per_mwh_lag_7_day_mean_diff` |

## 3. Feature-definition changes (same name, different computation)

These are changes that a feature-*list* diff cannot see. They decide the subset question in section 4.

**D1. Forecast-weather aggregation to county.** Affects every raw forecast column (`temperature`, `dewpoint`, `cloudcover_*`, `10_metre_*`, `direct_solar_radiation`, `surface_solar_radiation_downwards`, …).
- Before (v1), the nearest grid point to a hard-coded county coordinate:
  - `v1 c9: nearest_point = ((coordinates - available_coordinates) ** 2).sum(axis=1).argmin()`
  - `v1 c9: forecast_weather_df = forecast_weather_df.groupby(['county', 'datetime'])[weather_features].mean()`
  - with `v1 c9: forecast_weather_df['datetime'] = pd.to_datetime(forecast_weather_df.forecast_datetime).dt.tz_convert(None)`
- After (v2+), a pixel-weighted average:
  - `v2 c11: weather_values_arr = (np.swapaxes(weather_condition_arr, 1, 2)  @ county_coordinate_weights.set_index('county_id').values.T)`
  - The same matrix product is in v6 c27 and final c11.

**D2. Weather timestamp zone and hour alignment.** Affects all weather features.
- v2:
  - `v2 c8: forecast_weather_train['forecast_datetime'] = pd.to_datetime(forecast_weather_train['forecast_datetime']).dt.tz_convert("EET").dt.tz_localize(None)`
  - `v2 c7: historical_weather_train['datetime'] = pd.to_datetime(historical_weather_train['datetime'], utc=True).dt.tz_convert("EET").dt.tz_localize(None)`
  - Forward shift: `v2 c11: weather_by_county[cumulative_features] = weather_by_county[cumulative_features].shift(-1)` and `v2 c11: weather_by_county[point_features] = (weather_by_county[point_features].shift(-1) + weather_by_county[point_features]) / 2`
- v3 and later:
  - `v3 c9: forecast_weather_train['forecast_datetime'] = pd.to_datetime(forecast_weather_train['forecast_datetime'])#.dt.tz_convert("EET").dt.tz_localize(None)`
  - Backward average, with no shift of cumulative features: `v3 c11: weather_by_county[point_features] = (weather_by_county[point_features].shift(1) + weather_by_county[point_features]) / 2`
  - v6 c27 and final c11 have the same line.

**D3. Forecast bias correction (v4 only).**
- `v4 c14: forecast_weather_train = correct_weather_bias(forecast_weather_train)`, which adds per-(month, hour, lat, lon) offsets: `v4 c13: forecast_df['direct_solar_radiation'] = (`, plus the analogous `temperature` and `dewpoint` lines.
- It is off from v5 on. See row "weather bias correction" in section 1.1.
- Changes `direct_solar_radiation`, `temperature`, `dewpoint` and every feature derived from them, such as `*_2_day_diff` and `temp_minus_dewpoint`.

**D4. Consumption-counterpart lags.** This is a bug in v2–v4 (log: "Version 192 | bug fix: consumption_counterpart features", "Version 193 | bug fix for grouped_consumption_counterpart features").
- v1, correct (counterpart series): `v1 c18: train['lag_2_day_consumption_counterpart'] = train.groupby(['county', 'is_business', 'product_type', 'is_consumption'])['consumption_counterpart'].shift(24*2)`
- v2, v3 and v4 use the unit's **own** target:
  - `v2 c17: train[f'lag_{i}_day_consumption_counterpart'] = grouped_target.shift(24*i)`
  - `v3 c17: train[f'lag_{i}_day_consumption_counterpart'] = grouped_target.shift(24*i)`
  - `v4 c31: train[f'lag_{i}_day_consumption_counterpart'] = grouped_target.shift(24*i)`
  - Mean variants: `v4 c31: train['lag_7_day_mean_consumption_counterpart'] = np.mean([grouped_target.shift(24*i) for i in range(2, 9)], axis=0)`. v2 and v3 use the mean of the buggy columns: `v2 c17: train['lag_7_day_mean_consumption_counterpart'] = np.mean([train[f'lag_{i}_day_consumption_counterpart'] for i in range(2,9)], axis=0)`.
  - The inference copies have the same bug: `v2 c53`, `v3 c57`, `v4 c73: new_df[f'lag_{i}_day_consumption_counterpart'] = grouped_target.shift(24*i)`.
- v5 and later, fixed:
  - `v5 c30: train[f'lag_{i}_day_consumption_counterpart'] = grouped_consumption_counterpart.shift(24*i)`
  - `v6 c35: train[f'lag_{i}_day_consumption_counterpart'] = grouped_consumption_counterpart.shift(24*i)`
  - `final c20: train[f'lag_{i}_day_consumption_counterpart'] = grouped_consumption_counterpart.shift(24*i)`
- Effect, an inference from the code: in v2–v4, `lag_2_day_consumption_counterpart` equals `lag_2_day` before the latter's `.fillna(0)`. The model therefore never saw the counterpart series.

**D5. Electricity-price day lags.**
- Before: `v2 c12: new_df[f'euros_per_mwh_lag_{i}_day'] = new_df['euros_per_mwh'].shift(28*i)`
- After: `v3 c12: new_df[f'euros_per_mwh_lag_{i}_day'] = new_df['euros_per_mwh'].shift(24*i)`
- Changes `euros_per_mwh_lag_7_day_mean_diff`, which v2 side 1 uses (log: "Version 150 | ...fix a typo in getting lag elec price (24 hours per day not 28)").

**D6. Client join key.** Changes `eic_count` and `installed_capacity` and their diffs.
- Before: `v1 c15: on=["county", "is_business", "product_type", "data_block_id"],` and `v2 c14: on=["county", "is_business", "product_type", "data_block_id"],`
- After: `v3 c14: left_on=["county", "is_business", "product_type", "date"],` and `right_on=["county", "is_business", "product_type", "available_date"],` with `v3 c5: client_train['available_date'] = client_train['date'] + datetime.timedelta(days=2)`.
- Whether the two joins give identical rows: not determined.

**D7. Target-lag missing values.**
- v1 leaves NaN: `v1 c18: train['lag_2_day'] = train.groupby(['county', 'is_business', 'product_type', 'is_consumption'])['target'].shift(24*2)`. It then drops any row with any NaN: `v1 c19: train.dropna(axis=0, how='any', inplace=True)`.
- v2 and later fill with 0: `v2 c17: train[f'lag_{i}_day'] = grouped_target.shift(24*i).fillna(0)`. v6 c35 and final c20 are the same.
- This also changes `lag_7_day_mean`, the mean of the filled lags.

**D8. `snowfall_1_day_rolling_sum`.**
- Before (per county): `v4 c25: weather_by_county[['snowfall_1_day_rolling_sum']] = weather_by_county[['snowfall']].reset_index().groupby(['county'])[['snowfall']].rolling(24*1).sum().values`
- After (one flat rolling window over the (county, datetime)-sorted frame): `v5 c24: weather_by_county['snowfall_1_day_rolling_sum'] = weather_by_county['snowfall'].rolling(24*1).sum().values`. v6 c27 and final c11 are the same.
- Inference from the code: the two differ only in the first 23 hours of each county block.

**D9. Numeric precision.**
- v6 quantizes the whole frame, target included, before any lag is built: `v6 c33: train[col] = train[col].astype(np.float32).astype(np.float16)`.
- v5 has no cast. Final uses `final c18: train[col] = train[col].astype(np.float32)`.
- In this repo, v1–v5 also run with v6's float16 cast (`config._FP16`, in their `notebook_fidelity` tag): a full-period
  float64 feature table needs about 16 GiB to build (`docs/aws_sizing.md`), float16 about 6 GiB.

**D10. `lag_7_day_1h` in final inference.** See section 5. The final inference code computes it with `.fillna(0)`, while the final notebook's training code does not.

There are also non-definition renames and additions, such as v2 adding `daytype_lag_2/7_day` and v3 adding `daytype_lag_3_day`. They add new names and do not change existing ones.

---

## 4. Surprising findings

1. **The monotone `_neg` sign flip never fires (v2 → final).**
   - The code is `v2 c37: 'monotone_constraints': [value*((key[-3:]!='neg')*2-1) for key, value in get_monotone_constraints_map(features, is_consumption).items()],` and the same line is in v6 c65 and final c47.
   - Inside the comprehension, `key` is the feature name, and none ends in "neg". Even the outer `key` is the tuple `(is_consumption, target)`, whose `[-3:]` is a tuple.
   - Artifact proof: `0_lag_2_day_diff_neg` has direct_solar_radiation +1 and cloudcover_* −1, identical to `0_lag_2_day_diff`. The `_neg` targets (decrease vs lag) are constrained in the same direction as the increases.
2. **The counterpart-lag features were the unit's own target lags in v2–v4** (D4). Three milestones trained without real counterpart information, while the v1 implementation was correct.
3. **v6 casts the target and all floats to float16 before building lags** (D9). The final notebook reverts to float32.
4. **The final notebook cannot reproduce 0127e.** Its `select_features` lists, `k_fold = 6` and `lr 0.1/5` do not match the artifacts (lr 0.1, 3 folds, `day_category`, `sin/cos_month`).
5. **The 44 artifacts come from at least two training configurations.** 8 production scaled-diff models use `dayofweek`/`daytype` categoricals and ES 100. The other 36 use `day_category` and ES 200 (where not capped).
6. **v3 augmented the last 200k rows in series order, not the latest dates, with no seed.** Fixed in v4 (`sort_values(['datetime'])`, `np.random.seed(42)`).
7. **Augmentation scales the target but not some scale-dependent inputs** (mechanical list of used features missing from the augmentation list):
   - v3: `lag_2_day`, `lag_7_day`, `lag_7_day_mean`, `rolling_mean_24`, `rolling_std_24`, `lag_8_14_day_mean`, …
   - v5/v6: `rolling_mean_24_lag_7_day`, `eic_count_7_day_diff`, `installed_capacity_7_day_diff`.
   - final code: `lag_7_day_1h_mean`, also the reconstruction base of `lag_7_day_1h_mean_diff`, plus `rolling_mean_24_lag_7_day`.
8. **Sample weights were computed in v2–v6 but never passed to LightGBM.**
9. **v5's out-of-fold analysis used different folds than training.** `v5 c67: G = train_subset['month']` vs training `v5 c60: G = train_subset['day']`. Fixed in `v6 c71: G = train_subset['day']`.
10. **Several log descriptions are not reflected in the milestone code.**
    - "Version 216 | More data augmentation (800000)" vs v5's 400000.
    - "Version 227 | ...filter_hours_ahead=True for training" vs `v6 c30: filter_hours_ahead=False`.
    - "Version 175 | Group k fold by days" vs v4's month grouping.
    - "Version 230 | ...coef_0, coef_1, coef_2, coef_3 = 1/3, 1/3, 0, 1/3 for consumption" vs v6 `2/7, 2/7, 2/7, 1/7` (the 1/3 line is commented in `v6 c70`).

---
