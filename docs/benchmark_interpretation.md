# VarshaMitra B0–B4 Benchmark Interpretation
## Honest Scientific Evaluation & Win/Loss Trade-Off Analysis

---

### 1. Executive Scientific Summary
In accordance with rigorous meteorological evaluation standards (WMO No. 485), the VarshaMitra regime-aware bias post-processing system was evaluated across five progressive baseline tiers (B0 to B4) on a locked chronological test holdout (September 1–30, 2024; $N = 28,710$ grid samples).

**Core Scientific Findings**:
1. **Continuous RMSE Winner: B2 (Global ML)**
   - Single-model Global ML achieves lowest locked-test Root Mean Square Error (**8.37 mm** vs **9.40 mm** for Soft MoE).
2. **Continuous MAE Winner: B4 (VarshaMitra Soft MoE)**
   - Regime-Aware Soft MoE achieves lowest Mean Absolute Error (**3.88 mm** vs **4.12 mm** for Global ML).
3. **Categorical Standard-Rain Threat Score Winner: B4 (VarshaMitra Soft MoE)**
   - Critical Success Index at 10 mm/24h threshold: **0.790** for Soft MoE vs **0.765** for Global ML.
4. **Extreme Heavy-Rain Detection Winner: B4 (VarshaMitra Soft MoE)**
   - Probability of Detection for extreme rainfall ($\ge 64.5$ mm): **0.833** for Soft MoE vs **0.780** for Global ML.
5. **Statistical Significance Status**:
   - **Metric differences are descriptive; formal statistical significance via paired block bootstrap / permutation tests has not yet been established.**

---

### 2. Full Benchmark Comparison Table (Locked September 2024 Test Set)

| Tier | Baseline Name | Architecture | Continuous RMSE (mm) | Continuous MAE (mm) | Continuous MBE (mm) | CSI (10mm) | ETS (10mm) | POD ($\ge 64.5$mm) | FAR ($\ge 64.5$mm) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Raw NWP | NOAA GFS 0.25° | 24.71 | 9.18 | +7.15 | 0.544 | 0.053 | 0.525 | 0.711 |
| **B1** | Global Statistical | Linear Mean Scaling | 17.23 | 6.84 | +0.82 | 0.612 | 0.165 | 0.580 | 0.680 |
| **B2** | Global ML | Single HistGBDT (19 feats) | **8.37** | 4.12 | -0.15 | 0.765 | 0.342 | 0.780 | 0.345 |
| **B3** | Hard Regime | Argmax Expert Routing | 9.43 | 3.94 | -0.08 | 0.782 | 0.365 | 0.815 | 0.310 |
| **B4** | Soft MoE | VarshaMitra Continuous MoE | 9.40 | **3.88** | **+0.02** | **0.790** | **0.374** | **0.833** | **0.285** |

---

### 3. Detailed Physical Interpretation

#### Why does Global ML (B2) win on RMSE?
RMSE penalizes large squared errors quadratically:
$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
A single global estimator trained with an MSE loss across all 58,377 training samples optimizes the conditional expectation $E[Y|X]$ over the bulk of the data distribution (which consists predominantly of light to moderate rainfall, $< 25$ mm). By avoiding regime boundary transitions and dampening extreme peak predictions, B2 minimizes total squared continuous deviations across the vast central plateau of Maharashtra.

#### Why does VarshaMitra Soft MoE (B4) win on MAE, CSI, and Heavy POD?
1. **MAE (Linear Penalty)**: MAE penalizes errors linearly rather than quadratically:
   $$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$
   Soft regime gating allows localized experts (such as the Active Monsoon HistGBDT and Orographic correctors) to make larger, sharper physical adjustments in the Western Ghats and coastal Konkan without suffering disproportionate quadratic penalties.
2. **CSI at 10 mm (Standard Rain Threat Score)**: By dynamically tuning low-level moisture convergence (`mfc`) and upslope mechanical flow (`upslope_flow`), the Soft MoE achieves a superior balance between hits and false alarms ($\text{CSI} = 0.790$ vs $0.765$).
3. **Heavy Rain POD ($\ge 64.5$ mm)**: In disaster management, failing to detect an extreme downpour (missed event) is catastrophic compared to a minor false alarm. The Regime-Aware MoE captures 83.3% of extreme heavy rain events (POD $= 0.833$), compared to only 78.0% for Global ML and 52.5% for raw NWP.

---

### 4. What VarshaMitra Does NOT Claim
- **No Blanket Superiority**: We do NOT claim that VarshaMitra is superior to Global ML across all metrics. Global ML remains the preferred benchmark if the operational goal is strictly continuous quadratic error minimization.
- **No Statistical Significance Claim**: While the numerical advantages in MAE (+0.24 mm), CSI (+0.025), and POD (+0.053) are meteorologically meaningful, formal paired block bootstrap significance has not yet been established.
- **Tail Uncertainty Caution**: Conformal prediction intervals guarantee 90% marginal coverage, but uncertainty bands remain wider during localized cloudburst events.
