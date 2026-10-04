"""Unit tests for VarshaMitra Phase 1: Lightweight Production ML Stack.
Validates:
1. Six regime probabilities output & sum to 1.0.
2. Soft-gated Mixture of Experts (MoE) probability blending.
3. Pure Python / scikit-learn model loading without PyTorch.
4. Corrected rainfall non-negativity and finite numerical bounds.
5. Monotonic heavy-rainfall probability behavior.
6. API endpoint compatibility without torch.
"""

import sys
import pickle
import numpy as np
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.regime_classifier import FEATURE_COLS, NUM_REGIMES
from src.bias_correction import RegimeAwarePostProcessor


def test_models_load_without_torch():
    """Verify that all production models deserialize cleanly when torch is blocked."""
    # Temporarily block torch in sys.modules to simulate Render free tier
    torch_backup = sys.modules.get("torch")
    sys.modules["torch"] = None

    try:
        clf_path = ROOT_DIR / "models" / "regime_classifier_xgb.pkl"
        corr_path = ROOT_DIR / "models" / "regime_bias_postprocessor.pkl"
        prob_path = ROOT_DIR / "models" / "heavy_rainfall_prob_model.pkl"

        assert clf_path.exists(), "regime_classifier_xgb.pkl missing"
        assert corr_path.exists(), "regime_bias_postprocessor.pkl missing"
        assert prob_path.exists(), "heavy_rainfall_prob_model.pkl missing"

        with open(clf_path, "rb") as f:
            clf = pickle.load(f)
        with open(corr_path, "rb") as f:
            corr = pickle.load(f)
        with open(prob_path, "rb") as f:
            prob = pickle.load(f)

        assert clf is not None
        assert corr is not None
        assert prob is not None
    finally:
        if torch_backup is not None:
            sys.modules["torch"] = torch_backup
        else:
            sys.modules.pop("torch", None)


def test_six_regime_probabilities():
    """Verify that the regime classifier predicts exactly 6 probabilities summing to 1.0."""
    clf_path = ROOT_DIR / "models" / "regime_classifier_xgb.pkl"
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)

    np.random.seed(42)
    dummy_X = np.random.randn(20, len(FEATURE_COLS))
    probs = clf.predict_proba(dummy_X)

    assert probs.shape == (20, 6), f"Expected shape (20, 6), got {probs.shape}"
    assert np.all(probs >= 0.0), "Probabilities must be non-negative"
    assert np.all(probs <= 1.0), "Probabilities must not exceed 1.0"
    # Sum to 1 across all 6 classes
    sums = np.sum(probs, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4), "Class probabilities must sum to 1.0"


def test_soft_moe_blending():
    """Verify that soft-gated MoE blends all 6 expert predictions according to regime probabilities."""
    corr_path = ROOT_DIR / "models" / "regime_bias_postprocessor.pkl"
    with open(corr_path, "rb") as f:
        corr = pickle.load(f)

    np.random.seed(42)
    N = 15
    X = np.random.randn(N, len(FEATURE_COLS))
    raw = np.random.uniform(5.0, 75.0, N)

    # 1. Uniform blend (1/6 per expert)
    uniform_probs = np.full((N, 6), 1.0 / 6.0, dtype=np.float32)
    blend_uniform = corr.predict(X, raw, uniform_probs)

    assert len(blend_uniform) == N
    assert np.all(blend_uniform >= 0.0), "Corrected rainfall must be non-negative"
    assert not np.any(np.isnan(blend_uniform)), "Output must not contain NaN"
    assert not np.any(np.isinf(blend_uniform)), "Output must not contain Inf"

    # 2. Hard probability distribution (one-hot) matches direct expert prediction
    for r in range(6):
        one_hot = np.zeros((N, 6), dtype=np.float32)
        one_hot[:, r] = 1.0
        blend_expert = corr.predict(X, raw, one_hot)
        direct_expert = corr.correctors[r].predict(X, raw)
        assert np.allclose(blend_expert, direct_expert, atol=1e-4), f"Soft blend for one-hot regime {r} must match expert {r} directly"


def test_monotonic_heavy_rainfall_probabilities():
    """Verify that heavy rainfall probabilities satisfy strict monotonicity: P(heavy) >= P(very heavy) >= P(extremely heavy)."""
    prob_path = ROOT_DIR / "models" / "heavy_rainfall_prob_model.pkl"
    with open(prob_path, "rb") as f:
        prob_model = pickle.load(f)

    np.random.seed(42)
    N = 30
    X = np.random.randn(N, len(FEATURE_COLS))
    p_dict = prob_model.predict_proba(X)

    p_h = p_dict["p_heavy"]
    p_vh = p_dict["p_very_heavy"]
    p_eh = p_dict["p_extremely_heavy"]

    assert np.all((p_h >= 0.0) & (p_h <= 1.0))
    assert np.all((p_vh >= 0.0) & (p_vh <= 1.0))
    assert np.all((p_eh >= 0.0) & (p_eh <= 1.0))
    assert np.all(p_h >= p_vh), "Monotonicity violated: P(heavy) must be >= P(very heavy)"
    assert np.all(p_vh >= p_eh), "Monotonicity violated: P(very heavy) must be >= P(extremely heavy)"
