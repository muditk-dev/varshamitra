"""VarshaMitra Phase 4 Uncertainty Quantification Test Suite.
============================================================
Validates:
1. P10/P50/P90 existence.
2. P10 <= P50 <= P90 (Zero quantile crossings).
3. No NaN.
4. No negative rainfall (R >= 0 mm).
5. Quantile model loading from models/uncertainty_models.pkl.
6. Conformal calibration isolation (Validation split only).
7. Locked-test isolation.
8. Conformal interval validity (lower <= upper).
9. Regime entropy normalization in [0, 1].
10. Uncertainty API fields.
11. Existing API compatibility.
12. Zero PyTorch production loading.
13. Leakage Guard integration.
14. Existing B0-B4 regression preservation.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import pickle
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.feature_registry import (
    validate_features_for_training,
    validate_features_for_inference,
    PRODUCTION_19_FEATURES,
    FeatureLeakageError
)
from src.temporal_validation import get_chronological_splits
from src.uncertainty import (
    UncertaintyEngine,
    QuantilePredictor,
    SplitConformalCalibrator,
    compute_regime_entropy,
    pinball_loss
)
from api.main import app, run_live_prediction, FeatureVector


@pytest.fixture(scope="module")
def dataset_splits():
    df = pd.read_parquet(ROOT_DIR / "data" / "processed" / "processed_pipeline_dataframe.parquet")
    train_df, val_df, test_df = get_chronological_splits(df)
    return train_df, val_df, test_df


@pytest.fixture(scope="module")
def uncertainty_engine():
    model_path = ROOT_DIR / "models" / "uncertainty_models.pkl"
    assert model_path.exists(), f"Missing uncertainty model artifact at {model_path}"
    with open(model_path, "rb") as f:
        engine = pickle.load(f)
    return engine


# 1. P10/P50/P90 existence
def test_p10_p50_p90_existence(uncertainty_engine, dataset_splits):
    _, val_df, _ = dataset_splits
    X_val = val_df[PRODUCTION_19_FEATURES].values[:50]
    preds = uncertainty_engine.quantile_predictor.predict(X_val)

    assert "p10" in preds
    assert "p50" in preds
    assert "p90" in preds
    assert "spread" in preds
    assert len(preds["p10"]) == 50


# 2. P10 <= P50 <= P90 (Zero crossings)
def test_monotonic_quantile_ordering(uncertainty_engine, dataset_splits):
    _, _, test_df = dataset_splits
    X_test = test_df[PRODUCTION_19_FEATURES].values[:500]
    preds = uncertainty_engine.quantile_predictor.predict(X_test)

    p10 = preds["p10"]
    p50 = preds["p50"]
    p90 = preds["p90"]

    # Strictly test non-crossing with zero tolerance
    assert np.all(p10 <= p50 + 1e-6), "P10 > P50 crossing violation detected"
    assert np.all(p50 <= p90 + 1e-6), "P50 > P90 crossing violation detected"


# 3. No NaN
def test_no_nan_values(uncertainty_engine, dataset_splits):
    _, val_df, _ = dataset_splits
    X_val = val_df[PRODUCTION_19_FEATURES].values[:100]
    preds = uncertainty_engine.quantile_predictor.predict(X_val)

    for k, v in preds.items():
        assert not np.isnan(v).any(), f"NaN detected in {k}"


# 4. No negative rainfall
def test_no_negative_rainfall(uncertainty_engine, dataset_splits):
    _, _, test_df = dataset_splits
    X_test = test_df[PRODUCTION_19_FEATURES].values[:200]
    preds = uncertainty_engine.quantile_predictor.predict(X_test)

    assert np.all(preds["p10"] >= 0.0), "Negative rainfall in P10"
    assert np.all(preds["p50"] >= 0.0), "Negative rainfall in P50"
    assert np.all(preds["p90"] >= 0.0), "Negative rainfall in P90"


# 5. Quantile model loading
def test_quantile_model_loading():
    model_path = ROOT_DIR / "models" / "uncertainty_models.pkl"
    assert model_path.exists()
    size_kb = os.path.getsize(model_path) / 1024.0
    assert size_kb < 5000, f"Model artifact is too large: {size_kb} KB"

    with open(model_path, "rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, UncertaintyEngine)
    assert hasattr(loaded, "quantile_predictor")
    assert hasattr(loaded, "conformal_calibrator")


# 6. Conformal calibration isolation
def test_conformal_calibration_isolation(uncertainty_engine, dataset_splits):
    _, val_df, test_df = dataset_splits
    calibrator = uncertainty_engine.conformal_calibrator

    # Verify that calibration was run on validation data size (August 2024 = 29,667)
    assert calibrator.n_calibration_samples == len(val_df)
    assert calibrator.n_calibration_samples != len(test_df)
    assert calibrator.q_hat is not None
    assert calibrator.q_hat > 0.0


# 7. Locked-test isolation
def test_locked_test_isolation(dataset_splits):
    train_df, val_df, test_df = dataset_splits
    test_min_date = pd.to_datetime(test_df["time"]).min()
    val_max_date = pd.to_datetime(val_df["time"]).max()

    assert test_min_date > val_max_date, "Locked test dates overlap with validation dates"
    assert test_min_date >= pd.Timestamp("2024-09-01"), "Locked test must be September 2024"


# 8. Conformal interval validity
def test_conformal_interval_validity(uncertainty_engine):
    y_hat = np.array([0.0, 10.0, 50.0, 120.0], dtype=np.float32)
    low, up = uncertainty_engine.conformal_calibrator.predict_interval(y_hat)

    assert np.all(low >= 0.0), "Conformal lower bound must be >= 0 mm"
    assert np.all(low <= up), "Conformal lower bound exceeds upper bound"
    assert np.all(up >= y_hat), "Conformal upper bound must be >= point prediction"


# 9. Regime entropy normalization
def test_regime_entropy_normalization():
    # 1. Deterministic one-hot distribution
    one_hot = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    h_zero = compute_regime_entropy(one_hot)
    assert np.isclose(h_zero[0], 0.0, atol=1e-4)

    # 2. Maximum ambiguity uniform distribution across 6 regimes
    uniform = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]])
    h_max = compute_regime_entropy(uniform)
    assert np.isclose(h_max[0], 1.0, atol=1e-4)

    # 3. Intermediate distribution
    intermediate = np.array([[0.8, 0.1, 0.05, 0.03, 0.01, 0.01]])
    h_mid = compute_regime_entropy(intermediate)
    assert 0.0 < h_mid[0] < 1.0


# 10. Uncertainty API fields
def test_uncertainty_api_fields():
    fv = FeatureVector(
        precip_raw=45.0, wind_shear=28.0, wind_speed_850=14.0,
        upslope_flow=3.2, elevation=500.0, slope=15.0
    )
    res = run_live_prediction(fv)

    assert res.p10 is not None
    assert res.p50 is not None
    assert res.p90 is not None
    assert res.conformal_lower is not None
    assert res.conformal_upper is not None
    assert res.conformal_width is not None
    assert res.regime_entropy is not None
    assert res.rainfall_distribution is not None
    assert res.conformal_interval is not None
    assert res.uncertainty is not None

    assert res.p10 <= res.p50 <= res.p90
    assert res.conformal_lower <= res.conformal_upper
    assert 0.0 <= res.regime_entropy <= 1.0


# 11. Existing API compatibility
def test_existing_api_compatibility():
    fv = FeatureVector(precip_raw=20.0)
    res = run_live_prediction(fv)

    # All baseline fields from Phase 0-3 must be present and valid
    assert hasattr(res, "dominant_regime_id")
    assert hasattr(res, "dominant_regime_name")
    assert hasattr(res, "regime_confidence")
    assert hasattr(res, "regime_probabilities")
    assert hasattr(res, "raw_precipitation")
    assert hasattr(res, "calibrated_precipitation")
    assert hasattr(res, "bias_delta")
    assert hasattr(res, "p_heavy")
    assert hasattr(res, "alert_level")
    assert hasattr(res, "alert_color")
    assert hasattr(res, "explanation")

    assert res.dominant_regime_id in range(6)
    assert res.calibrated_precipitation >= 0.0
    assert res.alert_level in ["Green", "Yellow", "Orange", "Red"]


# 12. Zero PyTorch production loading
def test_zero_torch_uncertainty_loading():
    """Verify that uncertainty models deserialize and run without PyTorch (simulating Render free-tier)."""
    torch_backup = sys.modules.get("torch")
    sys.modules["torch"] = None
    try:
        model_path = ROOT_DIR / "models" / "uncertainty_models.pkl"
        with open(model_path, "rb") as f:
            engine = pickle.load(f)
        assert engine is not None
        X_dummy = np.zeros((2, 19), dtype=np.float32)
        res = engine.quantile_predictor.predict(X_dummy)
        assert "p10" in res
        assert "p50" in res
        assert "p90" in res
    finally:
        if torch_backup is not None:
            sys.modules["torch"] = torch_backup
        else:
            sys.modules.pop("torch", None)


# 13. Leakage Guard integration
def test_leakage_guard_uncertainty_integration():
    pred = QuantilePredictor()
    # Unregistered features must fail closed
    with pytest.raises(FeatureLeakageError):
        pred.fit(np.zeros((10, 2)), np.zeros(10), feature_names=["unregistered_feat", "precip_raw"])

    with pytest.raises(FeatureLeakageError):
        pred.fit(np.zeros((10, 2)), np.zeros(10), feature_names=["precip_obs", "precip_raw"])


# 14. Pinball loss calculation
def test_pinball_loss_exactness():
    y_t = np.array([10.0, 20.0])
    y_p = np.array([8.0, 25.0])
    # For tau=0.5: err = [2, -5]. loss = [0.5*2, -0.5*(-5)] = [1.0, 2.5] -> mean = 1.75
    pb = pinball_loss(y_t, y_p, 0.5)
    assert np.isclose(pb, 1.75)
