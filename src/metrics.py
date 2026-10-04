"""VarshaMitra Meteorological Verification Suite.
================================================
Implements official WMO / IMD standard verification metrics for continuous,
categorical (contingency table), and spatial precipitation verification:

1. Continuous Metrics:
   - Root Mean Square Error (RMSE)
   - Mean Absolute Error (MAE)
   - Mean Bias Error (MBE / Additive Bias)
   - Correlation Coefficient (Pearson r)

2. Categorical / Dichotomous Contingency Metrics:
   - Probability of Detection (POD / Hit Rate)
   - False Alarm Ratio (FAR)
   - Critical Success Index (CSI / Threat Score)
   - Equitable Threat Score (ETS / Gilbert Skill Score)

3. Spatial Neighborhood Metrics:
   - Fractions Skill Score (FSS) across user-configurable window sizes
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.signal import convolve2d

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("varshamitra.metrics")


def compute_continuous_metrics(fcst: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    """Calculate standard continuous error statistics: RMSE, MAE, MBE, Pearson r."""
    f = np.asarray(fcst, dtype=np.float64).ravel()
    o = np.asarray(obs, dtype=np.float64).ravel()
    
    valid_mask = ~(np.isnan(f) | np.isnan(o))
    f, o = f[valid_mask], o[valid_mask]
    
    if len(f) == 0:
        return {"rmse": 0.0, "mae": 0.0, "mbe": 0.0, "corr": 0.0}
        
    diff = f - o
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    mbe = float(np.mean(diff))
    
    # Correlation (computed directly to avoid BLAS delay-load issues on Windows)
    f_diff = f - np.mean(f)
    o_diff = o - np.mean(o)
    denom = np.sqrt(np.sum(f_diff**2) * np.sum(o_diff**2))
    if denom > 1e-8:
        corr = float(np.sum(f_diff * o_diff) / denom)
    else:
        corr = 0.0
        
    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mbe": round(mbe, 2),
        "corr": round(corr, 3)
    }


def compute_contingency_table(fcst: np.ndarray, obs: np.ndarray, threshold: float = 10.0) -> Tuple[int, int, int, int]:
    """Compute (Hits, False Alarms, Misses, Correct Negatives) at specified rainfall threshold."""
    f_event = (np.asarray(fcst) >= threshold)
    o_event = (np.asarray(obs) >= threshold)
    
    hits = int(np.sum(f_event & o_event))
    false_alarms = int(np.sum(f_event & ~o_event))
    misses = int(np.sum(~f_event & o_event))
    correct_negs = int(np.sum(~f_event & ~o_event))
    
    return hits, false_alarms, misses, correct_negs


def compute_categorical_scores(fcst: np.ndarray, obs: np.ndarray, threshold: float = 10.0) -> Dict[str, float]:
    """Compute POD, FAR, CSI, ETS at given threshold:
    - POD = H / (H + M) in [0, 1]
    - FAR = F / (H + F) in [0, 1]
    - CSI = H / (H + M + F) in [0, 1]
    - ETS = (H - H_random) / (H + M + F - H_random) in [-1/3, 1]
    """
    H, F, M, C = compute_contingency_table(fcst, obs, threshold)
    total = H + F + M + C
    
    # POD (Probability of Detection / Hit Rate)
    pod = float(H / (H + M)) if (H + M) > 0 else 0.0
    
    # FAR (False Alarm Ratio)
    far = float(F / (H + F)) if (H + F) > 0 else 0.0
    
    # CSI (Critical Success Index / Threat Score)
    csi = float(H / (H + M + F)) if (H + M + F) > 0 else 0.0
    
    # ETS (Equitable Threat Score / Gilbert Skill Score)
    h_random = ((H + M) * (H + F)) / total if total > 0 else 0.0
    ets_denom = H + M + F - h_random
    ets = float((H - h_random) / ets_denom) if ets_denom != 0 else 0.0
    
    return {
        "threshold": threshold,
        "hits": H,
        "false_alarms": F,
        "misses": M,
        "correct_negatives": C,
        "pod": round(np.clip(pod, 0.0, 1.0), 3),
        "far": round(np.clip(far, 0.0, 1.0), 3),
        "csi": round(np.clip(csi, 0.0, 1.0), 3),
        "ets": round(np.clip(ets, -0.333, 1.0), 3)
    }


def compute_fractions_skill_score(
    fcst_2d: np.ndarray,
    obs_2d: np.ndarray,
    threshold: float = 15.0,
    window_size: int = 3
) -> float:
    """Compute Fractions Skill Score (FSS) over a 2D spatial grid for neighborhood verification:
    FSS = 1 - [ sum( (P_fcst - P_obs)^2 ) / sum( P_fcst^2 + P_obs^2 ) ]
    where P is the spatial fraction of binary threshold exceedances in the neighborhood window.
    """
    f_bin = (fcst_2d >= threshold).astype(np.float64)
    o_bin = (obs_2d >= threshold).astype(np.float64)
    
    kernel = np.ones((window_size, window_size), dtype=np.float64) / (window_size * window_size)
    
    # Compute neighborhood fractions via 2D convolution
    f_frac = convolve2d(f_bin, kernel, mode="same", boundary="symm")
    o_frac = convolve2d(o_bin, kernel, mode="same", boundary="symm")
    
    mse = np.mean((f_frac - o_frac)**2)
    mse_ref = np.mean(f_frac**2) + np.mean(o_frac**2)
    
    if mse_ref < 1e-8:
        return 1.0  # Both forecast and obs agree on zero events
        
    fss = 1.0 - (mse / mse_ref)
    return round(float(np.clip(fss, 0.0, 1.0)), 3)


def compute_full_verification_suite(
    raw_fcst: np.ndarray,
    corr_fcst: np.ndarray,
    obs: np.ndarray,
    regimes: Optional[np.ndarray] = None,
    threshold: float = 10.0
) -> Dict[str, any]:
    """Compute continuous, categorical, and FSS verification metrics
    comparing Raw GFS vs. VarshaMitra Corrected Forecast, stratified overall and by regime.
    """
    # Overall continuous
    raw_cont = compute_continuous_metrics(raw_fcst, obs)
    corr_cont = compute_continuous_metrics(corr_fcst, obs)
    
    # Categorical
    raw_cat = compute_categorical_scores(raw_fcst, obs, threshold)
    corr_cat = compute_categorical_scores(corr_fcst, obs, threshold)
    
    # Heavy rainfall threshold (64.5 mm)
    raw_heavy = compute_categorical_scores(raw_fcst, obs, threshold=64.5)
    corr_heavy = compute_categorical_scores(corr_fcst, obs, threshold=64.5)
    
    # Stratification by regime
    regime_breakdown = {}
    from src.regime_labels import REGIME_NAMES
    
    if regimes is not None:
        for r_id, r_name in REGIME_NAMES.items():
            mask = (regimes == r_id)
            if np.sum(mask) > 5:
                r_raw = compute_continuous_metrics(raw_fcst[mask], obs[mask])
                r_corr = compute_continuous_metrics(corr_fcst[mask], obs[mask])
                r_cat = compute_categorical_scores(corr_fcst[mask], obs[mask], threshold=threshold)
                
                is_degenerate = bool(np.max(obs[mask]) == 0.0)
                if is_degenerate:
                    regime_breakdown[r_name] = {
                        "sample_count": int(np.sum(mask)),
                        "raw_rmse": r_raw["rmse"],
                        "corr_rmse": None,
                        "rmse_skill_gain_pct": None,
                        "pod": None,
                        "far": None,
                        "csi": None,
                        "ets": None,
                        "note": "N/A — offshore zero-rain test split"
                    }
                else:
                    skill_gain = (
                        ((r_raw["rmse"] - r_corr["rmse"]) / r_raw["rmse"]) * 100.0
                        if r_raw["rmse"] > 0
                        else 0.0
                    )
                    regime_breakdown[r_name] = {
                        "sample_count": int(np.sum(mask)),
                        "raw_rmse": r_raw["rmse"],
                        "corr_rmse": r_corr["rmse"],
                        "rmse_skill_gain_pct": round(skill_gain, 1),
                        "pod": r_cat["pod"],
                        "far": r_cat["far"],
                        "csi": r_cat["csi"],
                        "ets": r_cat["ets"]
                    }
                
    overall_skill_gain = ((raw_cont["rmse"] - corr_cont["rmse"]) / raw_cont["rmse"] * 100.0) if raw_cont["rmse"] > 0 else 0.0
    
    return {
        "overall": {
            "raw": {**raw_cont, **raw_cat},
            "corrected": {**corr_cont, **corr_cat},
            "rmse_skill_gain_pct": round(overall_skill_gain, 1),
            "heavy_rain_scores": {
                "raw": raw_heavy,
                "corrected": corr_heavy
            }
        },
        "by_regime": regime_breakdown
    }
