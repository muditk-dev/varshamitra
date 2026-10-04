"""VarshaMitra Uncertainty Quantification & Conformal Prediction Suite.
=====================================================================
Implements scientifically defensible uncertainty quantification for monsoon rainfall:
1. Predictive Quantiles (P10, P50, P90) via HistGradientBoostingRegressor(loss="quantile")
   with monotonic rearrangement (Chernozhukov et al., 2010) ensuring P10 <= P50 <= P90.
2. Split-Conformal Prediction calibrated strictly on the Validation split (August 2024)
   with finite-sample adjustment and absolute residual nonconformity scoring:
   s_i = |y_obs,i - y_hat,i|.
3. Regime Probability Entropy: Normalized Shannon entropy across all 6 regime probabilities
   H_norm = -sum(p_r * ln(p_r + eps)) / ln(6) in [0, 1].
4. Uncertainty Decomposition:
   - Predictive uncertainty spread: P90 - P10
   - Conformal interval width: Conformal Upper - Conformal Lower
   - Regime classification uncertainty: Regime entropy H_norm
   - Model disagreement proxy (epistemic proxy): |y_hat_moe - P50|
   - Aleatoric uncertainty proxy: Predictive quantile spread P90 - P10
   (All proxies explicitly documented as empirical proxies, not physically identified constants).

Strict Temporal & Scientific Integrity:
- Quantile models fit strictly on the Training split (June-July 2024).
- Conformal calibration performed strictly on the Validation split (August 2024).
- Locked Test split (September 2024) is strictly held-out for final evaluation.
- Zero PyTorch dependency: Pure scikit-learn & NumPy inference (< 50 MB RAM, Render-compatible).
- Fail-closed Leakage Guard: All feature vectors validated through src.feature_registry.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
from typing import Dict, Tuple, List, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.feature_registry import (
    validate_features_for_training,
    validate_features_for_inference,
    PRODUCTION_19_FEATURES,
    FeatureLeakageError
)
from src.regime_labels import REGIME_NAMES

logger = logging.getLogger("varshamitra.uncertainty")


# -----------------------------------------------------------------------------
# 1. MATHEMATICAL LOSS & ENTROPY FORMULATIONS
# -----------------------------------------------------------------------------

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Computes exact pinball loss (check function) for target quantile tau in (0, 1):
    L_tau(y, y_hat) = (1/N) * sum( max(tau * (y - y_hat), (tau - 1) * (y - y_hat)) )
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    err = y_t - y_p
    loss = np.maximum(tau * err, (tau - 1.0) * err)
    return float(np.mean(loss))


def compute_regime_entropy(regime_probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Computes normalized Shannon entropy for regime probability distributions.
    
    Formula:
        H(p) = -sum_{r=0}^{5} p_r * ln(p_r + eps)
        H_norm = H(p) / ln(6.0)

    Properties:
        - H_norm in [0.0, 1.0]
        - H_norm == 0.0 when one regime has probability 1.0 (complete certainty)
        - H_norm == 1.0 when all 6 regimes have equal probability 1/6 (complete ambiguity)

    Terminology:
        Referred to strictly as "Regime Classification Uncertainty" or "Regime Probability Entropy".
        Not conflated with physical epistemic uncertainty.
    """
    probs = np.asarray(regime_probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    # Normalize if slightly off due to float rounding
    sums = np.sum(probs, axis=1, keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    p_norm = probs / sums

    # Clip to avoid log(0)
    p_safe = np.clip(p_norm, eps, 1.0)
    entropy = -np.sum(p_safe * np.log(p_safe), axis=1)

    num_regimes = probs.shape[1] if probs.shape[1] > 1 else 6
    max_entropy = np.log(float(num_regimes))
    normalized_entropy = np.clip(entropy / max_entropy, 0.0, 1.0)

    return normalized_entropy.astype(np.float32)


# -----------------------------------------------------------------------------
# 2. QUANTILE PREDICTOR (P10 / P50 / P90) WITH MONOTONIC REARRANGEMENT
# -----------------------------------------------------------------------------

class QuantilePredictor:
    """Fits 3 independent HistGradientBoosting quantile regressors (tau = 0.10, 0.50, 0.90)
    and strictly enforces physical non-negativity and non-crossing via Monotonic Rearrangement:
    
    Monotonic Rearrangement (Chernozhukov et al., 2010):
        P10_adj = min(P10_raw, P50_raw)
        P90_adj = max(P90_raw, P50_raw)
        P10 = max(0.0, P10_adj)
        P50 = max(0.0, P50_raw)
        P90 = max(0.0, P90_adj)
        
    Guarantee:
        0 <= P10 <= P50 <= P90 for all predictions with 0 crossing violations.
    """

    def __init__(self, tau_list: Optional[List[float]] = None, max_iter: int = 100, max_depth: int = 6, random_state: int = 42):
        self.tau_list = tau_list or [0.10, 0.50, 0.90]
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.random_state = random_state
        self.models: Dict[float, HistGradientBoostingRegressor] = {}
        self.feature_names: List[str] = PRODUCTION_19_FEATURES

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """Fit quantile regressors on training split. Validates features against Leakage Guard."""
        features = feature_names or self.feature_names
        validate_features_for_training(features, strict=True)
        self.feature_names = features

        logger.info(f"Fitting QuantilePredictor for tau in {self.tau_list} on {len(X)} samples...")
        for tau in self.tau_list:
            logger.info(f"  Fitting HistGradientBoostingRegressor(loss='quantile', quantile={tau})...")
            reg = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=tau,
                max_iter=self.max_iter,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            reg.fit(X, y)
            self.models[tau] = reg

        logger.info("All quantile regressors trained successfully.")
        return self

    def predict_raw(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict raw unconstrained quantiles before monotonic rearrangement."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        validate_features_for_inference(self.feature_names, strict=True)

        preds = {}
        for tau, name in [(0.10, "p10_raw"), (0.50, "p50_raw"), (0.90, "p90_raw")]:
            if tau in self.models:
                preds[name] = self.models[tau].predict(X).astype(np.float32)
            else:
                preds[name] = np.zeros(len(X), dtype=np.float32)
        return preds

    def check_quantile_crossing(self, p10_raw: np.ndarray, p50_raw: np.ndarray, p90_raw: np.ndarray) -> Dict[str, Any]:
        """Analyzes crossing violations in raw quantile estimates before rearrangement."""
        n_total = len(p10_raw)
        viol_10_50 = p10_raw > p50_raw
        viol_50_90 = p50_raw > p90_raw

        count_10_50 = int(np.sum(viol_10_50))
        count_50_90 = int(np.sum(viol_50_90))
        total_viol = int(np.sum(viol_10_50 | viol_50_90))

        diff_10_50 = np.where(viol_10_50, p10_raw - p50_raw, 0.0)
        diff_50_90 = np.where(viol_50_90, p50_raw - p90_raw, 0.0)

        max_viol = float(max(np.max(diff_10_50) if count_10_50 > 0 else 0.0,
                             np.max(diff_50_90) if count_50_90 > 0 else 0.0))
        mean_viol = float(np.mean(np.maximum(diff_10_50, diff_50_90)))

        return {
            "total_samples": n_total,
            "violations_p10_gt_p50": count_10_50,
            "pct_p10_gt_p50": round((count_10_50 / n_total) * 100.0, 2) if n_total > 0 else 0.0,
            "violations_p50_gt_p90": count_50_90,
            "pct_p50_gt_p90": round((count_50_90 / n_total) * 100.0, 2) if n_total > 0 else 0.0,
            "total_violations": total_viol,
            "total_pct": round((total_viol / n_total) * 100.0, 2) if n_total > 0 else 0.0,
            "max_violation_mm": round(max_viol, 4),
            "mean_violation_mm": round(mean_viol, 4)
        }

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict conditionally rearranged quantiles strictly guaranteeing P10 <= P50 <= P90 and R >= 0."""
        raw = self.predict_raw(X)
        p10_r = raw["p10_raw"]
        p50_r = raw["p50_raw"]
        p90_r = raw["p90_raw"]

        # Chernozhukov Monotonic Rearrangement
        p10 = np.minimum(p10_r, p50_r)
        p90 = np.maximum(p90_r, p50_r)
        p50 = p50_r

        # Non-negativity physical precipitation constraint
        p10 = np.clip(p10, 0.0, None).astype(np.float32)
        p50 = np.clip(p50, 0.0, None).astype(np.float32)
        p90 = np.clip(p90, 0.0, None).astype(np.float32)

        # Secondary assurance step
        p10 = np.minimum(p10, p50)
        p90 = np.maximum(p90, p50)

        # Confirm 0 violations
        assert np.all(p10 <= p50 + 1e-6), "Rearrangement failed for P10 <= P50"
        assert np.all(p50 <= p90 + 1e-6), "Rearrangement failed for P50 <= P90"
        assert np.all(p10 >= 0.0), "Negative rainfall in P10"

        return {
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "spread": (p90 - p10).astype(np.float32)
        }


# -----------------------------------------------------------------------------
# 3. SPLIT-CONFORMAL CALIBRATOR
# -----------------------------------------------------------------------------

class SplitConformalCalibrator:
    """Split-Conformal Prediction Calibrator for Postprocessed Rainfall.
    
    Calibration Protocol:
    1. Base predictor (VarshaMitra Soft-Gated MoE) is fitted strictly on the Training Split.
    2. Nonconformity scores are computed strictly on the held-out Validation Split:
       s_i = |y_obs,i - y_hat,i|
    3. The empirical conformal quantile q_hat is computed at nominal coverage 1 - alpha:
       level = ceil((n_val + 1) * (1 - alpha)) / n_val
       q_hat = Quantile(scores_val, level)
    4. Conformal Prediction Interval for new unseen sample x:
       [ max(0.0, y_hat - q_hat), y_hat + q_hat ]
       
    Strict Temporal Independence:
    - Never calibrated on Training data (prevents in-sample underestimation).
    - Never calibrated or tuned on Locked Test data (prevents post-hoc data leakage).
    """

    def __init__(self, nominal_coverage: float = 0.90):
        self.nominal_coverage = nominal_coverage
        self.alpha = 1.0 - nominal_coverage
        self.q_hat: Optional[float] = None
        self.n_calibration_samples: int = 0
        self.calibration_mean_error: float = 0.0
        self.calibration_median_error: float = 0.0

    def calibrate(self, y_obs_val: np.ndarray, y_hat_val: np.ndarray):
        """Calibrates conformal nonconformity scores strictly on held-out validation data."""
        y_obs = np.asarray(y_obs_val, dtype=np.float64).ravel()
        y_hat = np.asarray(y_hat_val, dtype=np.float64).ravel()

        assert len(y_obs) == len(y_hat), "Mismatched validation observations and predictions"
        n = len(y_obs)
        self.n_calibration_samples = n

        # Absolute residual nonconformity score
        scores = np.abs(y_obs - y_hat)
        self.calibration_mean_error = float(np.mean(scores))
        self.calibration_median_error = float(np.median(scores))

        # Finite-sample adjusted quantile level
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / float(n))
        self.q_hat = float(np.quantile(scores, q_level))

        logger.info(
            f"Split-conformal calibration complete:\n"
            f"  Samples: {n}\n"
            f"  Nominal Coverage: {self.nominal_coverage * 100:.1f}%\n"
            f"  Finite-sample quantile level: {q_level:.6f}\n"
            f"  Conformal Quantile (q_hat): {self.q_hat:.4f} mm\n"
            f"  Validation Mean Absolute Residual: {self.calibration_mean_error:.2f} mm"
        )
        return self

    def predict_interval(self, y_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generates conformal intervals for predictions y_hat: [max(0, y_hat - q_hat), y_hat + q_hat]."""
        if self.q_hat is None:
            raise ValueError("SplitConformalCalibrator has not been calibrated. Call calibrate() first.")

        preds = np.asarray(y_hat, dtype=np.float32)
        lower = np.clip(preds - self.q_hat, 0.0, None).astype(np.float32)
        upper = (preds + self.q_hat).astype(np.float32)
        return lower, upper


# -----------------------------------------------------------------------------
# 4. UNIFIED UNCERTAINTY ENGINE & DECOMPOSITION
# -----------------------------------------------------------------------------

class UncertaintyEngine:
    """Master orchestrator for VarshaMitra uncertainty quantification:
    - QuantilePredictor (P10, P50, P90)
    - SplitConformalCalibrator (Nominal 90% coverage)
    - Regime Probability Entropy
    - Uncertainty Decomposition Proxies
    """

    def __init__(
        self,
        quantile_predictor: Optional[QuantilePredictor] = None,
        conformal_calibrator: Optional[SplitConformalCalibrator] = None
    ):
        self.quantile_predictor = quantile_predictor or QuantilePredictor()
        self.conformal_calibrator = conformal_calibrator or SplitConformalCalibrator(nominal_coverage=0.90)

    def fit_and_calibrate(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        moe_postprocessor: Any,
        regime_classifier: Any,
        feature_cols: Optional[List[str]] = None
    ):
        """Executes full training and calibration obeying strict temporal split:
        - Quantile models fitted on train_df
        - Conformal scores calibrated on val_df using MoE predictions
        """
        features = feature_cols or PRODUCTION_19_FEATURES
        validate_features_for_training(features, strict=True)

        X_train = train_df[features].values
        y_train = train_df["precip_obs"].values

        # 1. Fit Quantile Predictor on Train Split
        self.quantile_predictor.fit(X_train, y_train, feature_names=features)

        # 2. Calibrate Split-Conformal on Validation Split
        X_val = val_df[features].values
        y_val = val_df["precip_obs"].values
        raw_val = val_df["precip_raw"].values

        val_probs = regime_classifier.predict_proba(X_val)
        val_moe_pred = moe_postprocessor.predict(X_val, raw_val, val_probs)

        self.conformal_calibrator.calibrate(y_val, val_moe_pred)
        return self

    def predict_full_uncertainty(
        self,
        X: np.ndarray,
        point_prediction: float,
        regime_probs: np.ndarray
    ) -> Dict[str, Any]:
        """Inference for a single feature vector:
        Returns P10, P50, P90, Conformal Interval, Regime Entropy, and Decomposition Proxies.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 1. Predictive Quantiles (Monotonically Rearranged)
        q_res = self.quantile_predictor.predict(X)
        p10 = float(q_res["p10"][0])
        p50 = float(q_res["p50"][0])
        p90 = float(q_res["p90"][0])
        q_spread = float(q_res["spread"][0])

        # 2. Conformal Interval
        pt_arr = np.array([point_prediction], dtype=np.float32)
        c_low, c_up = self.conformal_calibrator.predict_interval(pt_arr)
        conf_low = float(c_low[0])
        conf_up = float(c_up[0])
        conf_width = round(conf_up - conf_low, 2)

        # 3. Regime Classification Entropy
        entropy_norm = float(compute_regime_entropy(regime_probs)[0])

        # 4. Uncertainty Decomposition Proxies (clearly designated as proxies)
        model_disagreement = round(abs(point_prediction - p50), 2)
        aleatoric_proxy = round(q_spread, 2)

        # Narrative summary
        summary = (
            f"Predictive 80% range [{p10:.1f}, {p90:.1f}] mm (median: {p50:.1f} mm). "
            f"Conformal 90% interval [{conf_low:.1f}, {conf_up:.1f}] mm (calibrated width: {conf_width:.1f} mm). "
            f"Regime classification entropy: {entropy_norm:.3f} (normalized 0-1)."
        )

        return {
            "p10": round(p10, 2),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "quantile_spread": round(q_spread, 2),
            "conformal_lower": round(conf_low, 2),
            "conformal_upper": round(conf_up, 2),
            "conformal_width": conf_width,
            "conformal_coverage_target": self.conformal_calibrator.nominal_coverage,
            "regime_entropy": round(entropy_norm, 4),
            "model_disagreement_proxy": model_disagreement,
            "aleatoric_proxy": aleatoric_proxy,
            "summary": summary
        }


# -----------------------------------------------------------------------------
# 5. METEOROLOGICAL EVALUATION BENCHMARK UTILITIES
# -----------------------------------------------------------------------------

def evaluate_quantile_performance(y_true: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> Dict[str, Any]:
    """Calculates comprehensive quantile performance metrics."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p10_a = np.asarray(p10, dtype=np.float64).ravel()
    p50_a = np.asarray(p50, dtype=np.float64).ravel()
    p90_a = np.asarray(p90, dtype=np.float64).ravel()

    pb10 = pinball_loss(y, p10_a, 0.10)
    pb50 = pinball_loss(y, p50_a, 0.50)
    pb90 = pinball_loss(y, p90_a, 0.90)

    cov_p10 = float(np.mean(y <= p10_a) * 100.0)
    cov_p90 = float(np.mean(y <= p90_a) * 100.0)
    cov_80 = float(np.mean((y >= p10_a) & (y <= p90_a)) * 100.0)

    spread = p90_a - p10_a
    mean_width = float(np.mean(spread))
    median_width = float(np.median(spread))

    mae_p50 = float(np.mean(np.abs(y - p50_a)))
    rmse_p50 = float(np.sqrt(np.mean((y - p50_a)**2)))

    return {
        "p10_pinball_loss": round(pb10, 4),
        "p10_empirical_lower_tail_pct": round(cov_p10, 2),
        "p50_pinball_loss": round(pb50, 4),
        "p50_mae": round(mae_p50, 2),
        "p50_rmse": round(rmse_p50, 2),
        "p90_pinball_loss": round(pb90, 4),
        "p90_empirical_upper_tail_pct": round(cov_p90, 2),
        "interval_p10_p90_coverage_pct": round(cov_80, 2),
        "interval_p10_p90_mean_width_mm": round(mean_width, 2),
        "interval_p10_p90_median_width_mm": round(median_width, 2)
    }


def evaluate_conformal_performance(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, nominal_coverage: float = 0.90) -> Dict[str, Any]:
    """Evaluates split-conformal coverage, sharpness, and error metrics."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    low = np.asarray(lower, dtype=np.float64).ravel()
    up = np.asarray(upper, dtype=np.float64).ravel()

    covered = (y >= low) & (y <= up)
    empirical_coverage = float(np.mean(covered) * 100.0)
    coverage_error = round(empirical_coverage - (nominal_coverage * 100.0), 2)

    width = up - low
    mean_width = float(np.mean(width))
    median_width = float(np.median(width))

    return {
        "nominal_coverage_pct": round(nominal_coverage * 100.0, 1),
        "empirical_coverage_pct": round(empirical_coverage, 2),
        "coverage_error_pp": coverage_error,
        "mean_interval_width_mm": round(mean_width, 2),
        "median_interval_width_mm": round(median_width, 2)
    }


def evaluate_uncertainty_by_intensity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
    conf_low: np.ndarray,
    conf_up: np.ndarray
) -> List[Dict[str, Any]]:
    """Evaluates uncertainty metrics across standard rainfall intensity bins:
    0-10 mm, 10-25 mm, 25-50 mm, 50-100 mm, >= 100 mm.
    """
    bins = [
        ("0–10 mm (Light)", 0.0, 10.0),
        ("10–25 mm (Moderate)", 10.0, 25.0),
        ("25–50 mm (Rather Heavy)", 25.0, 50.0),
        ("50–100 mm (Heavy)", 50.0, 100.0),
        (">=100 mm (Very Heavy/Extreme)", 100.0, 1e9)
    ]

    results = []
    for label, b_min, b_max in bins:
        mask = (y_true >= b_min) & (y_true < b_max)
        n = int(np.sum(mask))
        if n == 0:
            continue

        y_b = y_true[mask]
        pred_b = y_pred[mask]
        p10_b = p10[mask]
        p90_b = p90[mask]
        cl_b = conf_low[mask]
        cu_b = conf_up[mask]

        mae = float(np.mean(np.abs(y_b - pred_b)))
        med_pred = float(np.median(pred_b))
        q_width = float(np.mean(p90_b - p10_b))
        conf_width = float(np.mean(cu_b - cl_b))
        conf_cov = float(np.mean((y_b >= cl_b) & (y_b <= cu_b)) * 100.0)

        results.append({
            "bin": label,
            "samples": n,
            "median_prediction_mm": round(med_pred, 2),
            "mae_mm": round(mae, 2),
            "p90_p10_width_mm": round(q_width, 2),
            "conformal_width_mm": round(conf_width, 2),
            "conformal_coverage_pct": round(conf_cov, 2)
        })

    return results


def evaluate_uncertainty_by_regime(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf_low: np.ndarray,
    conf_up: np.ndarray,
    regimes: np.ndarray,
    entropy: np.ndarray
) -> List[Dict[str, Any]]:
    """Evaluates uncertainty across all six canonical regimes."""
    results = []
    for r_id in range(6):
        mask = (regimes == r_id)
        n = int(np.sum(mask))
        name = REGIME_NAMES.get(r_id, f"Regime {r_id}")
        if n == 0:
            results.append({
                "regime_id": r_id,
                "regime": name,
                "samples": 0,
                "rmse_mm": 0.0,
                "mae_mm": 0.0,
                "median_width_mm": 0.0,
                "conformal_coverage_pct": 0.0,
                "median_regime_entropy": 0.0,
                "sparse_warning": True
            })
            continue

        y_r = y_true[mask]
        pred_r = y_pred[mask]
        cl_r = conf_low[mask]
        cu_r = conf_up[mask]
        ent_r = entropy[mask]

        rmse = float(np.sqrt(np.mean((y_r - pred_r)**2)))
        mae = float(np.mean(np.abs(y_r - pred_r)))
        med_width = float(np.median(cu_r - cl_r))
        cov = float(np.mean((y_r >= cl_r) & (y_r <= cu_r)) * 100.0)
        med_ent = float(np.median(ent_r))

        results.append({
            "regime_id": r_id,
            "regime": name,
            "samples": n,
            "rmse_mm": round(rmse, 2),
            "mae_mm": round(mae, 2),
            "median_width_mm": round(med_width, 2),
            "conformal_coverage_pct": round(cov, 2),
            "median_regime_entropy": round(med_ent, 4),
            "sparse_warning": bool(n < 100)
        })

    return results


def evaluate_heavy_rainfall_interaction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf_low: np.ndarray,
    conf_up: np.ndarray,
    entropy: np.ndarray
) -> Dict[str, Any]:
    """Evaluates uncertainty behavior during high-risk precipitation events
    using the existing IMD thresholds (64.5 mm, 115.5 mm, 204.5 mm).
    """
    levels = [
        ("Non-Heavy (<64.5 mm)", y_true < 64.5),
        ("Heavy (>=64.5 mm)", y_true >= 64.5),
        ("Very Heavy (>=115.5 mm)", y_true >= 115.5),
        ("Extremely Heavy (>=204.5 mm)", y_true >= 204.5)
    ]

    report = {}
    for name, mask in levels:
        n = int(np.sum(mask))
        if n == 0:
            report[name] = {"samples": 0}
            continue

        y_sub = y_true[mask]
        pred_sub = y_pred[mask]
        cl_sub = conf_low[mask]
        cu_sub = conf_up[mask]
        ent_sub = entropy[mask]

        mae = float(np.mean(np.abs(y_sub - pred_sub)))
        cov = float(np.mean((y_sub >= cl_sub) & (y_sub <= cu_sub)) * 100.0)
        med_w = float(np.median(cu_sub - cl_sub))
        med_ent = float(np.median(ent_sub))

        report[name] = {
            "samples": n,
            "mae_mm": round(mae, 2),
            "conformal_coverage_pct": round(cov, 2),
            "median_interval_width_mm": round(med_w, 2),
            "median_regime_entropy": round(med_ent, 4)
        }

    return report
