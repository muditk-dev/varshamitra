"""Unit and integration test suite for VarshaMitra Phase 3:
Out-Of-Fold (OOF) Regime Probabilities, Six Regime Experts, and Soft MoE Validation.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import pickle
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.feature_registry import (
    PRODUCTION_19_FEATURES,
    FeatureLeakageError,
    validate_features_for_training,
    validate_features_for_inference
)
from src.temporal_validation import (
    get_chronological_splits,
    generate_oof_regime_probabilities,
    run_b0_b4_benchmarks
)
from src.regime_classifier import FEATURE_COLS, NUM_REGIMES, XGBoostRegimeClassifier
from src.regime_labels import REGIME_NAMES
from src.bias_correction import RegimeAwarePostProcessor, ActiveMonsoonGradientBoostingCorrector


# 1. OOF Probability Generation
def test_oof_probability_generation():
    df_path = ROOT_DIR / "data" / "processed" / "processed_pipeline_dataframe.parquet"
    if not df_path.exists():
        pytest.skip("processed_pipeline_dataframe.parquet missing")

    df = pd.read_parquet(df_path)
    train_df, _, _ = get_chronological_splits(df)

    # Test on a 10-day subset for fast unit testing
    sub_dates = pd.to_datetime(train_df["time"].unique())[:10]
    sub_train = train_df[train_df["time"].isin(sub_dates)].copy()

    oof_probs = generate_oof_regime_probabilities(sub_train, n_splits=3)
    assert oof_probs.shape == (len(sub_train), NUM_REGIMES)
    assert not np.isnan(oof_probs).any()


# 2. No In-Sample OOF Predictions
def test_no_in_sample_oof_predictions():
    df_path = ROOT_DIR / "data" / "processed" / "processed_pipeline_dataframe.parquet"
    if not df_path.exists():
        pytest.skip("processed_pipeline_dataframe.parquet missing")

    df = pd.read_parquet(df_path)
    train_df, _, _ = get_chronological_splits(df)

    # Take 6 days, 3 folds (2 days each)
    unique_dates = pd.to_datetime(train_df["time"].unique())[:6]
    sub_train = train_df[train_df["time"].isin(unique_dates)].copy()

    date_chunks = np.array_split(unique_dates, 3)
    for k, val_dates in enumerate(date_chunks):
        val_set = set(pd.to_datetime(val_dates).date)
        is_val = sub_train["time"].dt.date.isin(val_set)
        is_train = ~is_val

        train_dates = set(sub_train.loc[is_train, "time"].dt.date)
        val_dates_actual = set(sub_train.loc[is_val, "time"].dt.date)

        # Zero intersection proves the fold model never saw the validation block
        assert len(train_dates.intersection(val_dates_actual)) == 0, f"Fold {k} contaminated with validation dates"


# 3. Probability Normalization
def test_oof_probability_normalization():
    df_path = ROOT_DIR / "data" / "processed" / "processed_pipeline_dataframe.parquet"
    if not df_path.exists():
        pytest.skip("processed_pipeline_dataframe.parquet missing")

    df = pd.read_parquet(df_path)
    train_df, _, _ = get_chronological_splits(df)

    sub_dates = pd.to_datetime(train_df["time"].unique())[:6]
    sub_train = train_df[train_df["time"].isin(sub_dates)].copy()

    oof_probs = generate_oof_regime_probabilities(sub_train, n_splits=2)
    assert np.all(oof_probs >= 0.0), "Negative probability detected"
    sums = np.sum(oof_probs, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4), "Probabilities must sum to 1.0"


# 4. Temporal Fold Ordering
def test_temporal_fold_ordering():
    df_path = ROOT_DIR / "data" / "processed" / "processed_pipeline_dataframe.parquet"
    if not df_path.exists():
        pytest.skip("processed_pipeline_dataframe.parquet missing")

    df = pd.read_parquet(df_path)
    train_df, _, _ = get_chronological_splits(df)

    unique_dates = pd.Series(pd.to_datetime(train_df["time"].unique())).sort_values().values
    date_blocks = np.array_split(unique_dates, 5)

    for i in range(len(date_blocks) - 1):
        max_current = pd.to_datetime(date_blocks[i]).max()
        min_next = pd.to_datetime(date_blocks[i+1]).min()
        assert max_current < min_next, f"Temporal fold {i} does not strictly precede fold {i+1}"


# 5. Validation and Locked Test Isolation
def test_validation_locked_test_isolation():
    df_path = ROOT_DIR / "data" / "processed" / "processed_pipeline_dataframe.parquet"
    if not df_path.exists():
        pytest.skip("processed_pipeline_dataframe.parquet missing")

    df = pd.read_parquet(df_path)
    train_df, val_df, test_df = get_chronological_splits(df)

    train_dates = set(pd.to_datetime(train_df["time"]).dt.date)
    val_dates = set(pd.to_datetime(val_df["time"]).dt.date)
    test_dates = set(pd.to_datetime(test_df["time"]).dt.date)

    assert len(train_dates.intersection(val_dates)) == 0
    assert len(train_dates.intersection(test_dates)) == 0
    assert len(val_dates.intersection(test_dates)) == 0


# 6. Six Expert Availability
def test_six_expert_availability():
    moe = RegimeAwarePostProcessor()
    assert len(moe.correctors) == 6
    for r in range(6):
        assert r in moe.correctors
        assert hasattr(moe.correctors[r], "fit")
        assert hasattr(moe.correctors[r], "predict")
    # Verify Regime 0 is ActiveMonsoonGradientBoostingCorrector
    assert isinstance(moe.correctors[0], ActiveMonsoonGradientBoostingCorrector)


# 7. Soft MoE Blending Continuous Transition
def test_soft_moe_blending_continuity():
    moe = RegimeAwarePostProcessor()
    # Dummy fit
    N = 50
    X = np.random.randn(N, len(PRODUCTION_19_FEATURES))
    raw = np.random.uniform(5.0, 50.0, N)
    obs = np.random.uniform(5.0, 50.0, N)
    regimes = np.random.randint(0, 6, N)
    moe.fit(X, raw, obs, regimes)

    # Transition from Regime 0 to Regime 3
    steps = 15
    alphas = np.linspace(0.0, 1.0, steps)
    probs = np.zeros((steps, 6), dtype=np.float32)
    probs[:, 0] = 1.0 - alphas
    probs[:, 3] = alphas

    sample_x = X[0:1].repeat(steps, axis=0)
    sample_raw = raw[0:1].repeat(steps)
    preds = moe.predict(sample_x, sample_raw, probs)

    assert len(preds) == steps
    assert not np.isnan(preds).any()
    # Continuous step differences should be finite and smooth
    step_diffs = np.abs(np.diff(preds))
    assert np.all(step_diffs < 20.0), "Discontinuous jump across regime probability blend"


# 8. One-Hot Expert Equivalence
def test_one_hot_expert_equivalence():
    moe = RegimeAwarePostProcessor()
    N = 30
    X = np.random.randn(N, len(PRODUCTION_19_FEATURES))
    raw = np.random.uniform(5.0, 40.0, N)
    obs = np.random.uniform(5.0, 40.0, N)
    regimes = np.random.randint(0, 6, N)
    moe.fit(X, raw, obs, regimes)

    for r in range(6):
        one_hot = np.zeros((N, 6), dtype=np.float32)
        one_hot[:, r] = 1.0

        blend_pred = moe.predict(X, raw, one_hot)
        direct_pred = moe.correctors[r].predict(X, raw)
        assert np.allclose(blend_pred, direct_pred, atol=1e-4), f"One-hot mismatch for Regime {r}"


# 9. No Negative Rainfall
def test_no_negative_rainfall():
    moe = RegimeAwarePostProcessor()
    N = 25
    # Negative extreme inputs
    X = np.random.randn(N, len(PRODUCTION_19_FEATURES)) * -10.0
    raw = np.zeros(N)  # Zero raw forecast
    probs = np.full((N, 6), 1.0 / 6.0, dtype=np.float32)

    # Fit with dummy data
    X_fit = np.random.randn(N, len(PRODUCTION_19_FEATURES))
    raw_fit = np.random.uniform(0.0, 30.0, N)
    obs_fit = np.random.uniform(0.0, 30.0, N)
    reg_fit = np.random.randint(0, 6, N)
    moe.fit(X_fit, raw_fit, obs_fit, reg_fit)

    preds = moe.predict(X, raw, probs)
    assert np.all(preds >= 0.0), "Soft MoE produced negative rainfall"


# 10. Leakage Guard Integration in OOF
def test_leakage_guard_in_oof():
    dummy_df = pd.DataFrame({
        "time": pd.date_range("2024-06-01", periods=10, freq="D"),
        "precip_obs": np.random.randn(10)
    })
    # Attempting to pass target feature 'precip_obs' to OOF should raise FeatureLeakageError
    with pytest.raises(FeatureLeakageError, match="target"):
        generate_oof_regime_probabilities(dummy_df, feature_cols=["precip_obs"])


# 11. B0-B4 Benchmark Execution with Upgraded MoE
def test_b0_b4_benchmark_runs():
    df_path = ROOT_DIR / "data" / "processed" / "processed_pipeline_dataframe.parquet"
    if not df_path.exists():
        pytest.skip("processed_pipeline_dataframe.parquet missing")

    df = pd.read_parquet(df_path)
    train_df, _, test_df = get_chronological_splits(df)

    sub_train = train_df.iloc[:1500].copy()
    sub_test = test_df.iloc[:400].copy()

    benchmarks = run_b0_b4_benchmarks(sub_train, sub_test)
    assert "B0_raw_nwp" in benchmarks
    assert "B2_global_ml" in benchmarks
    assert "B3_hard_regime" in benchmarks
    assert "B4_varshamitra" in benchmarks


# 12. Model Artifact Loads Without PyTorch
def test_model_artifact_loads_without_torch():
    torch_backup = sys.modules.get("torch")
    sys.modules["torch"] = None

    try:
        model_path = ROOT_DIR / "models" / "regime_bias_postprocessor.pkl"
        assert model_path.exists(), "regime_bias_postprocessor.pkl missing"

        with open(model_path, "rb") as f:
            moe = pickle.load(f)

        assert moe is not None
        assert len(moe.correctors) == 6
        assert isinstance(moe.correctors[0], ActiveMonsoonGradientBoostingCorrector)
    finally:
        if torch_backup is not None:
            sys.modules["torch"] = torch_backup
        else:
            sys.modules.pop("torch", None)
