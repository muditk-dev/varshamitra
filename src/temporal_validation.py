"""VarshaMitra Temporal Validation & B0-B4 Benchmark Suite.
========================================================
Implements strict chronological data splitting (Train / Validation / Locked Test)
and the official WMO/IMD B0-B4 benchmark progression ladder for SIH 26080.

Temporal Integrity Rules:
1. Chronological Ordering: Train (earliest) < Validation (middle) < Locked Test (unseen latest).
2. Zero Backward Information Leakage: Preprocessing, scalers, and models are fitted
   exclusively on the training split.
3. Locked Test Independence: Locked test data is never seen during model fitting,
   hyperparameter selection, or threshold tuning.
4. Feature Availability: All predictors are validated against the Feature Registry
   and Leakage Guard prior to splitting and training.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.feature_registry import validate_features_for_training, validate_features_for_inference, PRODUCTION_19_FEATURES, FeatureLeakageError
from src.regime_classifier import FEATURE_COLS, NUM_REGIMES, XGBoostRegimeClassifier
from src.regime_labels import REGIME_NAMES
from src.bias_correction import RegimeAwarePostProcessor
from src.heavy_rainfall_probability import FocalLossXGBoostProbabilityModel
from src.metrics import compute_continuous_metrics, compute_categorical_scores

logger = logging.getLogger("varshamitra.temporal_validation")


def generate_oof_regime_probabilities(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    time_col: str = "time",
    feature_cols: Optional[List[str]] = None
) -> np.ndarray:
    """Generate strictly out-of-fold (OOF) cross-fitted regime probabilities on the training split.

    Temporal Structure:
    - Partitions unique training dates into n_splits contiguous temporal blocks.
    - For fold k, fits XGBoostRegimeClassifier on all training dates outside block k.
    - Generates predicted probabilities on block k.

    Guarantees:
    - Zero in-sample probability generation (every sample is predicted by a model that never saw it).
    - Validation and Locked Test data are never touched.
    - Probabilities are non-negative and sum to 1.0.

    Returns:
        np.ndarray of shape (len(train_df), NUM_REGIMES) with OOF probabilities.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    validate_features_for_training(feature_cols, strict=True)

    unique_dates = pd.to_datetime(train_df[time_col].unique())
    date_blocks = np.array_split(unique_dates, n_splits)

    oof_probs = np.zeros((len(train_df), NUM_REGIMES), dtype=np.float32)

    for k, block_dates in enumerate(date_blocks):
        block_set = set(pd.to_datetime(block_dates).date)
        is_val_block = train_df[time_col].dt.date.isin(block_set)
        is_train_block = ~is_val_block

        X_tr = train_df.loc[is_train_block, feature_cols].values
        y_tr = train_df.loc[is_train_block, "regime"].values
        X_val = train_df.loc[is_val_block, feature_cols].values

        clf_k = XGBoostRegimeClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
        clf_k.fit(X_tr, y_tr)

        probs_k = clf_k.predict_proba(X_val)
        oof_probs[is_val_block] = probs_k

    # Sanity checks
    assert not np.isnan(oof_probs).any(), "NaN detected in OOF probabilities"
    assert np.all(oof_probs >= 0.0), "Negative probability detected in OOF probabilities"
    sums = np.sum(oof_probs, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4), "OOF probabilities must sum to 1.0"

    return oof_probs


def get_chronological_splits(
    df: pd.DataFrame,
    train_end: str = "2024-07-31",
    val_end: str = "2024-08-31",
    time_col: str = "time"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition dataset into strict chronological Train, Validation, and Locked Test splits.

    Default Temporal Windows (Monsoon 2024):
    - Train:       2024-06-01 to 2024-07-31 (61 days, Early/Mid Monsoon)
    - Validation:  2024-08-01 to 2024-08-31 (31 days, Peak Monsoon)
    - Locked Test: 2024-09-01 to 2024-09-30 (30 days, Late/Withdrawal Monsoon)

    Returns:
        (train_df, val_df, test_df)
    Raises:
        ValueError: If temporal ordering or non-overlap constraints are violated.
    """
    if time_col not in df.columns:
        raise ValueError(f"DataFrame must contain temporal column '{time_col}'.")

    # Validate predictor features through Leakage Guard before splitting
    validate_features_for_training(FEATURE_COLS, strict=True)

    df_sorted = df.sort_values(by=time_col).copy()
    time_series = pd.to_datetime(df_sorted[time_col])

    train_mask = time_series <= train_end
    val_mask = (time_series > train_end) & (time_series <= val_end)
    test_mask = time_series > val_end

    train_df = df_sorted[train_mask].copy()
    val_df = df_sorted[val_mask].copy()
    test_df = df_sorted[test_mask].copy()

    # Integrity Assertions
    train_max = pd.to_datetime(train_df[time_col]).max()
    val_min = pd.to_datetime(val_df[time_col]).min()
    val_max = pd.to_datetime(val_df[time_col]).max()
    test_min = pd.to_datetime(test_df[time_col]).min()

    if not (train_max < val_min):
        raise ValueError(f"Temporal overlap detected: Train max ({train_max}) >= Val min ({val_min})")
    if not (val_max < test_min):
        raise ValueError(f"Temporal overlap detected: Val max ({val_max}) >= Test min ({test_min})")

    logger.info(
        f"Chronological split successfully verified:\n"
        f"  Train:       {pd.to_datetime(train_df[time_col]).min().date()} to {train_max.date()} (N = {len(train_df)})\n"
        f"  Validation:  {val_min.date()} to {val_max.date()} (N = {len(val_df)})\n"
        f"  Locked Test: {test_min.date()} to {pd.to_datetime(test_df[time_col]).max().date()} (N = {len(test_df)})"
    )

    return train_df, val_df, test_df


def compute_regime_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """Compute exact cell counts and percentages for all 6 regimes across splits."""
    splits = {
        "train": train_df,
        "validation": val_df,
        "locked_test": test_df
    }

    distribution_report = {}
    for split_name, df_split in splits.items():
        total = len(df_split)
        counts = df_split["regime"].value_counts().to_dict()
        regime_stats = {}
        for r_id in range(NUM_REGIMES):
            c = counts.get(r_id, 0)
            pct = round((c / total) * 100.0, 2) if total > 0 else 0.0
            regime_stats[REGIME_NAMES[r_id]] = {
                "regime_id": r_id,
                "count": int(c),
                "percentage": pct
            }
        distribution_report[split_name] = {
            "total_samples": total,
            "regimes": regime_stats
        }

    return distribution_report


def run_b0_b4_benchmarks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold_standard: float = 10.0,
    threshold_heavy: float = 64.5
) -> Dict[str, Any]:
    """Execute official B0-B4 benchmark ladder evaluated on held-out locked test set:
    - B0: Raw NWP (Uncalibrated GFS 0.25 deg)
    - B1: Global Statistical Correction (Linear Bias Scaling on Train)
    - B2: Global ML Correction (HistGBDT on 19 features without regime awareness)
    - B3: Hard Regime Routing (Argmax classifier routing to regime experts)
    - B4: Full VarshaMitra (Soft-gated probability-weighted Mixture of Experts)
    """
    X_train = train_df[FEATURE_COLS].values
    raw_train = train_df["precip_raw"].values
    obs_train = train_df["precip_obs"].values
    reg_train = train_df["regime"].values

    X_test = test_df[FEATURE_COLS].values
    raw_test = test_df["precip_raw"].values
    obs_test = test_df["precip_obs"].values
    reg_test = test_df["regime"].values

    # B0: Raw NWP
    b0_pred = raw_test

    # B1: Global Statistical Correction (Linear Regression fit strictly on train)
    b1_model = LinearRegression()
    b1_model.fit(raw_train.reshape(-1, 1), obs_train)
    b1_pred = np.maximum(0.0, b1_model.predict(raw_test.reshape(-1, 1)))

    # B2: Global ML (HistGradientBoosting fit strictly on train without regime awareness)
    b2_model = HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)
    b2_model.fit(X_train, obs_train - raw_train)
    b2_pred = np.maximum(0.0, raw_test + b2_model.predict(X_test))

    # Fit Regime Classifier on Train
    clf = XGBoostRegimeClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
    clf.fit(X_train, reg_train)
    test_probs = clf.predict_proba(X_test)
    test_pred_regimes = np.argmax(test_probs, axis=1)

    # Fit Regime-Aware Postprocessor on Train
    moe = RegimeAwarePostProcessor()
    moe.fit(X_train, raw_train, obs_train, reg_train)

    # B3: Hard Regime Routing
    b3_pred = np.zeros(len(X_test), dtype=np.float32)
    for r in range(NUM_REGIMES):
        mask_r = (test_pred_regimes == r)
        if np.sum(mask_r) > 0:
            b3_pred[mask_r] = moe.correctors[r].predict(X_test[mask_r], raw_test[mask_r])
    b3_pred = np.maximum(0.0, b3_pred)

    # B4: Full VarshaMitra (Soft-Gated Mixture of Experts)
    b4_pred = moe.predict(X_test, raw_test, test_probs)

    benchmarks = {
        "B0_raw_nwp": {"name": "B0: Raw NWP (GFS Uncalibrated)", "pred": b0_pred},
        "B1_global_statistical": {"name": "B1: Global Linear Scaling", "pred": b1_pred},
        "B2_global_ml": {"name": "B2: Global ML (HistGBDT)", "pred": b2_pred},
        "B3_hard_regime": {"name": "B3: Hard Regime Routing", "pred": b3_pred},
        "B4_varshamitra": {"name": "B4: VarshaMitra Soft MoE", "pred": b4_pred}
    }

    results = {}
    for key, b_info in benchmarks.items():
        pred = b_info["pred"]
        cont = compute_continuous_metrics(pred, obs_test)
        cat_std = compute_categorical_scores(pred, obs_test, threshold=threshold_standard)
        cat_heavy = compute_categorical_scores(pred, obs_test, threshold=threshold_heavy)

        results[key] = {
            "name": b_info["name"],
            "continuous": cont,
            "standard_rain_10mm": cat_std,
            "heavy_rain_64_5mm": cat_heavy
        }

    return results


def evaluate_temporal_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, Any]:
    """Train XGBoost regime classifier on train_df and evaluate on val_df and test_df."""
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["regime"].values

    clf = XGBoostRegimeClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
    clf.fit(X_train, y_train)

    splits_eval = {}
    for s_name, df_s in [("validation", val_df), ("locked_test", test_df)]:
        X_s = df_s[FEATURE_COLS].values
        y_s = df_s["regime"].values

        probs = clf.predict_proba(X_s)
        preds = np.argmax(probs, axis=1)

        acc = float(accuracy_score(y_s, preds))
        macro_f1 = float(f1_score(y_s, preds, average="macro", zero_division=0))
        cm = confusion_matrix(y_s, preds, labels=list(range(NUM_REGIMES))).tolist()
        report = classification_report(
            y_s, preds,
            target_names=[REGIME_NAMES[i] for i in range(NUM_REGIMES)],
            zero_division=0,
            output_dict=True
        )

        prob_sums = np.sum(probs, axis=1)
        is_normalized = bool(np.allclose(prob_sums, 1.0, atol=1e-4))

        splits_eval[s_name] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "confusion_matrix": cm,
            "classification_report": report,
            "prob_shape": list(probs.shape),
            "prob_normalized": is_normalized
        }

    return splits_eval
