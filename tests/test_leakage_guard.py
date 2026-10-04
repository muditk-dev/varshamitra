"""Unit and integration test suite for VarshaMitra Phase 2:
Leakage Guard, Feature Registry, and Chronological Temporal Validation.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.feature_registry import (
    FEATURE_REGISTRY,
    PRODUCTION_19_FEATURES,
    FeatureLeakageError,
    validate_features_for_training,
    validate_features_for_inference,
    get_feature_metadata,
    LeakageGuard
)
from src.temporal_validation import (
    get_chronological_splits,
    compute_regime_distribution,
    run_b0_b4_benchmarks
)
from src.regime_classifier import NUM_REGIMES
from src.regime_labels import REGIME_NAMES


# 1. Feature Registry Completeness
def test_feature_registry_completeness():
    required_keys = [
        "name", "source", "description", "calculation_method",
        "forecast_time_availability", "target_dependency",
        "leakage_risk", "allowed_for_training", "allowed_for_inference"
    ]
    for feat_name, meta in FEATURE_REGISTRY.items():
        assert meta["name"] == feat_name
        for k in required_keys:
            assert k in meta, f"Feature '{feat_name}' missing required registry key '{k}'"


# 2. Unknown Feature Rejection (Fail Closed)
def test_unknown_feature_rejection():
    with pytest.raises(FeatureLeakageError, match="Unknown feature"):
        validate_features_for_training(["fake_satellite_microwave"], strict=True)

    with pytest.raises(FeatureLeakageError, match="Unknown feature"):
        validate_features_for_inference(["unregistered_barometer_flux"], strict=True)


# 3. Target Feature Rejection
def test_target_feature_rejection():
    target_features = ["precip_obs", "target_diff"]
    for t_feat in target_features:
        with pytest.raises(FeatureLeakageError, match="target"):
            validate_features_for_training([t_feat], strict=True)
        with pytest.raises(FeatureLeakageError, match="target"):
            validate_features_for_inference([t_feat], strict=True)


# 4. Future Observation Feature Rejection
def test_future_observation_rejection():
    future_features = ["lead_precip_obs", "future_obs_rain"]
    for f_feat in future_features:
        with pytest.raises(FeatureLeakageError, match="target|TEMPORAL|available"):
            validate_features_for_training([f_feat], strict=True)
        with pytest.raises(FeatureLeakageError, match="target|available"):
            validate_features_for_inference([f_feat], strict=True)


# 5. Inference Safe Validation
def test_inference_safe_validation():
    # Variables that are not allowed for live model inference
    forbidden_for_inference = ["precip_obs", "time", "lat", "lon", "regime", "precip_corr"]
    for feat in forbidden_for_inference:
        with pytest.raises(FeatureLeakageError):
            validate_features_for_inference([feat], strict=True)


# 6. All 19 Production Diagnostic Features
def test_all_19_production_features_pass():
    assert len(PRODUCTION_19_FEATURES) == 19
    # Training validation
    val_train = validate_features_for_training(PRODUCTION_19_FEATURES, strict=True)
    assert val_train == PRODUCTION_19_FEATURES
    # Inference validation
    val_inf = validate_features_for_inference(PRODUCTION_19_FEATURES, strict=True)
    assert val_inf == PRODUCTION_19_FEATURES
    # LeakageGuard static verification
    assert LeakageGuard.verify_production_19() is True


# 7. Chronological Ordering
def test_chronological_ordering():
    df_path = ROOT_DIR / "data" / "processed" / "processed_pipeline_dataframe.parquet"
    if not df_path.exists():
        pytest.skip("processed_pipeline_dataframe.parquet missing")

    df = pd.read_parquet(df_path)
    train_df, val_df, test_df = get_chronological_splits(
        df,
        train_end="2024-07-31",
        val_end="2024-08-31"
    )

    t_max_train = pd.to_datetime(train_df["time"]).max()
    t_min_val = pd.to_datetime(val_df["time"]).min()
    t_max_val = pd.to_datetime(val_df["time"]).max()
    t_min_test = pd.to_datetime(test_df["time"]).min()

    assert t_max_train < t_min_val, "Train must strictly precede Validation"
    assert t_max_val < t_min_test, "Validation must strictly precede Locked Test"


# 8. Train / Validation / Test Non-Overlap
def test_train_val_test_non_overlap():
    df_path = ROOT_DIR / "data" / "processed" / "processed_pipeline_dataframe.parquet"
    if not df_path.exists():
        pytest.skip("processed_pipeline_dataframe.parquet missing")

    df = pd.read_parquet(df_path)
    train_df, val_df, test_df = get_chronological_splits(df)

    train_dates = set(pd.to_datetime(train_df["time"]).dt.date)
    val_dates = set(pd.to_datetime(val_df["time"]).dt.date)
    test_dates = set(pd.to_datetime(test_df["time"]).dt.date)

    assert len(train_dates.intersection(val_dates)) == 0, "Train and Val dates must not overlap"
    assert len(val_dates.intersection(test_dates)) == 0, "Val and Test dates must not overlap"
    assert len(train_dates.intersection(test_dates)) == 0, "Train and Test dates must not overlap"
    assert len(train_df) + len(val_df) + len(test_df) == len(df), "Splits must sum to total rows"


# 9. Temporal Leakage Checks
def test_temporal_leakage_exception():
    # If date range is inverted, get_chronological_splits must fail closed
    dummy_df = pd.DataFrame({
        "time": pd.date_range("2024-06-01", periods=10, freq="D"),
        "precip_raw": np.random.randn(10),
        "precip_obs": np.random.randn(10)
    })
    for f in PRODUCTION_19_FEATURES:
        if f not in dummy_df:
            dummy_df[f] = 1.0

    # Inverted date bounds should raise ValueError
    with pytest.raises(ValueError, match="Temporal overlap"):
        get_chronological_splits(dummy_df, train_end="2024-06-08", val_end="2024-06-05")


# 10. Six Regime Distribution Across Splits
def test_six_regime_distribution():
    df_path = ROOT_DIR / "data" / "processed" / "processed_pipeline_dataframe.parquet"
    if not df_path.exists():
        pytest.skip("processed_pipeline_dataframe.parquet missing")

    df = pd.read_parquet(df_path)
    train_df, val_df, test_df = get_chronological_splits(df)
    report = compute_regime_distribution(train_df, val_df, test_df)

    for split in ["train", "validation", "locked_test"]:
        assert split in report
        assert report[split]["total_samples"] > 0
        regimes = report[split]["regimes"]
        assert len(regimes) == 6
        # Sum of regime counts equals total samples
        total_regimes = sum(r["count"] for r in regimes.values())
        assert total_regimes == report[split]["total_samples"]


# 11. B0-B4 Benchmark Execution
def test_b0_b4_benchmark_execution():
    df_path = ROOT_DIR / "data" / "processed" / "processed_pipeline_dataframe.parquet"
    if not df_path.exists():
        pytest.skip("processed_pipeline_dataframe.parquet missing")

    df = pd.read_parquet(df_path)
    train_df, val_df, test_df = get_chronological_splits(df)

    # Subsample test set for fast unit test execution
    sub_train = train_df.iloc[:2000].copy()
    sub_test = test_df.iloc[:500].copy()

    results = run_b0_b4_benchmarks(sub_train, sub_test, threshold_standard=10.0, threshold_heavy=64.5)

    expected_benchmarks = [
        "B0_raw_nwp", "B1_global_statistical", "B2_global_ml",
        "B3_hard_regime", "B4_varshamitra"
    ]
    for b in expected_benchmarks:
        assert b in results, f"Benchmark {b} missing from results"
        assert "continuous" in results[b]
        assert "standard_rain_10mm" in results[b]
        assert "heavy_rain_64_5mm" in results[b]
        assert results[b]["continuous"]["rmse"] >= 0.0


# 12. API Leakage Guard Integration
def test_api_leakage_guard_integration():
    from api.main import run_live_prediction, FeatureVector
    fv = FeatureVector(
        precip_raw=40.0,
        wind_shear=25.0,
        wind_speed_850=12.0,
        upslope_flow=2.5,
        vorticity=1.2,
        mslp_anomaly=-2.0,
        q_850=14.0,
        rh850=80.0,
        mfc=4.0,
        elevation=400.0,
        slope=12.0
    )
    res = run_live_prediction(fv)
    assert res.dominant_regime_id in range(NUM_REGIMES)
    assert res.calibrated_precipitation >= 0.0
    assert len(res.regime_probabilities) == 6
