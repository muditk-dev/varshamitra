# VarshaMitra Model Card
## AI for a Resilient Monsoon India (SIH 26080)

---

### 1. Model Details & Purpose
- **Model Name**: VarshaMitra Regime-Aware Rainfall Post-Processing Engine
- **Version**: `production-v1.0.0`
- **Feature Schema**: `production-19-v1`
- **Model Architecture**:
  1. *Synoptic Regime Gating*: 6-Class XGBoost Classifier (`v2.1-lightweight-oof`)
  2. *Bias Post-Processing*: Soft-Gated Mixture of Experts (`v3.0-lightweight-moe-histgbdt`)
  3. *Heavy Rainfall Exceedance*: Focal Loss XGBoost Binary Classifiers (`v1.2-focal-calibrated`)
  4. *Uncertainty Quantification*: Pinball Quantile Regressors + Monotonic Rearrangement + Split Conformal Calibrator (`v1.0-conformal-p10-p50-p90`)
  5. *Explainability*: Native C++ TreeSHAP & scikit-learn TreeExplainer
- **Primary Purpose**: Correct systematic NWP rainfall forecast biases over complex terrain, quantify predictive uncertainty via conformal prediction intervals, and estimate extreme rainfall exceedance risks across 36 Maharashtra districts.

---

### 2. Intended Use & Boundaries
- **Intended Users**: State disaster management authorities, district collectors, agricultural extension planners, meteorological researchers.
- **Intended Application**: Operational 24-hour lead post-processing of daily gridded monsoon precipitation forecasts.
- **Out-of-Scope Use Cases**:
  - Sub-daily / convective storm tracking (time resolution is daily 24h accumulation).
  - Geographic domains outside Maharashtra without local calibration.
  - Non-monsoon winter cyclogenesis in the Indian Ocean without seasonal retraining.

---

### 3. Geographic & Temporal Scope
- **Geographic Domain**: Maharashtra State, India ($15.5^\circ\text{N} - 22.0^\circ\text{N}$, $72.5^\circ\text{E} - 81.0^\circ\text{E}$), gridded at $0.25^\circ \times 0.25^\circ$ resolution (36 administrative districts).
- **Temporal Scope**: Southwest Monsoon Season (JJAS: June 1 to September 30, 2024).
- **Chronological Split**:
  - *Training Period*: June 1, 2024 to July 31, 2024 ($N = 58,377$ grid samples)
  - *Validation Period (OOF Calibration)*: August 1, 2024 to August 31, 2024 ($N = 29,667$ grid samples)
  - *Locked Test Period*: September 1, 2024 to September 30, 2024 ($N = 28,710$ grid samples)

---

### 4. Six Canonical Monsoon Regimes
1. **Active Monsoon (0)**: Continuous onshore low-level westerly jet ($u_{850} > 12$ m/s), high tropospheric moisture ($\text{RH}_{850} > 80\%$), strong Western Ghats orographic uplift.
2. **Break Monsoon (1)**: Monsoon trough shifts northward; weak low-level westerly winds ($u_{850} < 6$ m/s), elevated MSLP anomaly, suppressed convective activity over peninsular India.
3. **Monsoon Depression / Low (2)**: Low-pressure synoptic vortices traversing westward from the Bay of Bengal; marked negative MSLP anomaly ($<-3.5$ hPa) and elevated cyclonic vorticity ($>1.5 \times 10^{-5}\text{ s}^{-1}$).
4. **Orographic (3)**: Forced mechanical ascent along the steep Western Ghats escarpment; characterized by high upslope flow ($>0.35$ m/s) and windward terrain gradient.
5. **Coastal (4)**: Low-elevation maritime Konkan corridor with pronounced land-sea thermal contrasts, high specific humidity ($q_{850}$), and offshore moisture convergence.
6. **Western Disturbance (5)**: Subtropical upper-tropospheric westerly trough affecting north-northwestern Maharashtra, characterized by mid-latitude shear anomalies.

---

### 5. Input Features (Production 19-Feature Registry)
All features are authorized forecast-side predictors satisfying the fail-closed Leakage Guard (`src/feature_registry.py`):
1. `precip_raw` — Raw numerical weather prediction precipitation (mm)
2. `elevation` — Surface elevation above sea level (m)
3. `slope_lon` — Zonal topographic gradient ($\partial z / \partial x$)
4. `slope_lat` — Meridional topographic gradient ($\partial z / \partial y$)
5. `u850` — 850 hPa zonal wind component (m/s)
6. `v850` — 850 hPa meridional wind component (m/s)
7. `u200` — 200 hPa zonal wind component (m/s)
8. `v200` — 200 hPa meridional wind component (m/s)
9. `mslp` — Mean sea level pressure (hPa)
10. `rh850` — 850 hPa relative humidity (%)
11. `wind_shear` — Deep-layer vertical wind shear $|V_{850} - V_{200}|$ (m/s)
12. `wind_speed_850` — Low-level wind speed magnitude (m/s)
13. `upslope_flow` — Mechanical vertical wind component $V_{850} \cdot \nabla z$ (m/s)
14. `vorticity` — 850 hPa relative vorticity ($10^{-5}\text{ s}^{-1}$)
15. `mslp_anomaly` — Deviation of MSLP from 30-day running spatial mean (hPa)
16. `q_850` — 850 hPa specific humidity (g/kg)
17. `mfc` — Moisture flux convergence $-\nabla \cdot (q V_{850})$ ($10^{-7}\text{ kg}\cdot\text{m}^{-2}\text{s}^{-1}$)
18. `precip_roll3` — 3-day antecedent forecast accumulation index (mm)
19. `coastal_proximity` — Distance-to-coast marine boundary index $[0, 1]$

---

### 6. Validation Strategy & Benchmark Results (B0–B4)
Evaluated strictly on the locked September 2024 test split ($N = 28,710$ samples):

| Tier | Model | RMSE (mm) | MAE (mm) | MBE (mm) | CSI (10mm) | Heavy POD ($\ge 64.5$mm) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **B0** | Raw NWP (NOAA GFS 0.25°) | 24.71 | 9.18 | +7.15 | 0.544 | 0.525 |
| **B1** | Global Statistical Scaling | 17.23 | 6.84 | +0.82 | 0.612 | 0.580 |
| **B2** | Global ML (HistGBDT) | **8.37** | 4.12 | -0.15 | 0.765 | 0.780 |
| **B3** | Hard Regime Routing | 9.43 | 3.94 | -0.08 | 0.782 | 0.815 |
| **B4** | VarshaMitra Soft MoE | 9.40 | **3.88** | **+0.02** | **0.790** | **0.833** |

---

### 7. Uncertainty Quantification Methodology
- **Predictive Quantiles ($P_{10}, P_{50}, P_{90}$)**: Independent pinball-loss gradient boosted regressors with Chernozhukov monotonic rearrangement enforcing $P_{10} \le P_{50} \le P_{90}$ across all samples ($0$ crossing violations).
- **Split-Conformal Prediction**: Non-conformity scores calibrated on the August 2024 validation split:
  $$\hat{q} = \text{Quantile}_{1 - \alpha}\left(|y_i - \hat{y}_i|; \frac{\lceil(N+1)(1-\alpha)\rceil}{N}\right) = 9.1667\text{ mm}$$
  Empirical coverage on locked September test set: **89.40%** (target: 90.00%).
- **Regime Entropy**: Normalized Shannon entropy over the 6 regime gating probabilities measuring circulation classification ambiguity:
  $$H_{\text{norm}}(x) = -\frac{1}{\ln(6)}\sum_{r=0}^{5} P_r(x)\ln(P_r(x)) \in [0, 1]$$

---

### 8. Explainability Methodology
- **TreeSHAP Implementation**:
  - XGBoost Multiclass Regime Classifier: Native C++ TreeSHAP (`pred_contribs=True`), exact log-odds additivity ($\text{error} \le 3.34 \times 10^{-6}$).
  - Active Monsoon Residual Corrector: `shap.TreeExplainer` on HistGBDT, exact mm additivity ($\text{error} \le 1.49 \times 10^{-13}\text{ mm}$).
  - Heavy Rainfall Exceedance: Native C++ TreeSHAP, exact log-odds risk additivity ($\text{error} \le 1.43 \times 10^{-6}$).
- **Mixture Decomposition**: Isolates expert mixture components ($C_r = P_r \times \text{Expert}_r$) from internal tree feature attributions ($\phi_j$).

---

### 9. Heavy Rainfall Alert Thresholds
Standardized against India Meteorological Department (IMD) operational definitions:
- **Heavy Rainfall**: $\ge 64.5$ mm / 24h (Yellow / Orange alert trigger)
- **Very Heavy Rainfall**: $\ge 115.5$ mm / 24h (Orange / Red alert trigger)
- **Extremely Heavy Rainfall**: $\ge 204.5$ mm / 24h (Red alert trigger)

---

### 10. Operational Limitations & Scientific Safeguards
1. **Descriptive Comparison Only**: Metric differences between B2 and B4 are descriptive; formal paired block bootstrap significance has not yet been established.
2. **Global RMSE vs. Tail Performance**: Global ML (B2) minimizes quadratic continuous error across light-to-moderate rain. The Soft MoE (B4) provides targeted advantages in absolute error (MAE), operational CSI threat score, and extreme precipitation detection (POD $\ge 64.5$ mm).
3. **No Physical Causality**: TreeSHAP values represent statistical attribution within tree ensembles and must not be interpreted as physical or atmospheric causality.
4. **Western Disturbance Non-Parametric Modeling**: Western Disturbance uses 1D Empirical Quantile Mapping; TreeSHAP is mathematically non-applicable and is not faked.
5. **Sample Sparsity**: Break Monsoon occurrences are sparse in the test set ($N=34$ grid-cell hours).
6. **No Extreme-Tail Guarantee**: Conformal prediction guarantees marginal 90% coverage over the bulk distribution, not conditional extreme-tail coverage for catastrophic 1-in-100-year events.
