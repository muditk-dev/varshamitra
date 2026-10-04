"""VarshaMitra Phase 5 True Explainability & Evidence Chain Test Suite.
====================================================================
Tests scientifically defensible, model-derived feature attributions:
1. Real SHAP values (model-derived, non-synthetic, non-zero).
2. Adherence to PRODUCTION_19_FEATURES registry.
3. Zero target leakage (fail-closed guard enforcement).
4. Deterministic output across repeated calls.
5. Exact TreeSHAP additivity reconstruction error (< 1e-5):
   - XGBoost native C++ TreeSHAP (multiclass regime classifier)
   - HistGradientBoosting TreeExplainer (Active Monsoon bias corrector)
   - XGBoost native C++ TreeSHAP (Heavy rainfall exceedance)
6. Schema compliance (top_features, ranks, directions).
7. Soft MoE mixture decomposition consistency.
8. Heavy rainfall risk exceedance attribution.
9. District-level true model-derived TreeSHAP (replaces legacy synthetic).
10. Dedicated POST /api/explain endpoint validation.
11. Precomputed GET /api/explain/summary endpoint validation.
12. Zero PyTorch production loading during explainability inference.
13. Full backwards compatibility & regression preservation.
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
    validate_features_for_inference,
    PRODUCTION_19_FEATURES,
    FeatureLeakageError
)
from src.regime_labels import REGIME_NAMES
from src.explainability import TreeSHAPExplainer
from api.main import (
    app,
    run_live_prediction,
    get_district_by_name,
    explain_prediction,
    get_explainability_summary,
    FeatureVector,
    MODELS,
    DATA_STORE
)


@pytest.fixture(scope="module")
def loaded_explainer():
    """Initializes TreeSHAPExplainer using production models."""
    clf = MODELS.get("regime_classifier")
    moe = MODELS.get("bias_corrector")
    hprob = MODELS.get("heavy_prob_model")
    return TreeSHAPExplainer(
        regime_classifier=clf,
        bias_postprocessor=moe,
        heavy_prob_model=hprob,
        feature_names=PRODUCTION_19_FEATURES
    )


@pytest.fixture(scope="module")
def sample_feature_array():
    """Constructs a valid 1x19 feature array matching PRODUCTION_19_FEATURES."""
    raw = 45.0
    elev = 500.0
    slope_rad = np.radians(12.0)
    slope_lon = float(np.sin(slope_rad) * 0.05)
    slope_lat = float(np.cos(slope_rad) * 0.05)
    u850 = 14.0
    v850 = 4.5
    u200 = float(u850 - 25.0)
    v200 = 1.0
    mslp = 1002.5
    rh = 85.0
    shear = 25.0
    w850 = 14.0
    upslope = 3.5
    vort = 1.8
    mslp_a = -2.5
    q850 = 15.0
    mfc = 5.2
    precip_roll3 = float(raw * 0.85)
    coastal_prox = float(np.clip(1.0 - (elev / 600.0), 0.0, 1.0))

    arr = np.array([[
        raw, elev, slope_lon, slope_lat, u850, v850, u200, v200, mslp, rh,
        shear, w850, upslope, vort, mslp_a, q850, mfc, precip_roll3, coastal_prox
    ]], dtype=np.float32)
    return arr


# -----------------------------------------------------------------------------
# 1. Feature Registry & Leakage Guard
# -----------------------------------------------------------------------------
def test_explainability_feature_registry_adherence():
    """Verify that the explainer validates against PRODUCTION_19_FEATURES."""
    assert len(PRODUCTION_19_FEATURES) == 19
    validate_features_for_inference(PRODUCTION_19_FEATURES, strict=True)


def test_explainability_leakage_guard_fail_closed(loaded_explainer, sample_feature_array):
    """Verify that passing an unauthorized/future feature causes fail-closed rejection."""
    leak_features = PRODUCTION_19_FEATURES[:-1] + ["precip_obs_future"]
    bad_explainer = TreeSHAPExplainer(
        regime_classifier=loaded_explainer.clf,
        feature_names=leak_features
    )
    with pytest.raises(FeatureLeakageError):
        bad_explainer.explain_regime_classification(sample_feature_array)


# -----------------------------------------------------------------------------
# 2. Regime Classifier TreeSHAP (Native C++)
# -----------------------------------------------------------------------------
def test_regime_classifier_native_treeshap_additivity(loaded_explainer, sample_feature_array):
    """Verify exact additivity of XGBoost C++ TreeSHAP: sum(phi_j) + phi_0 == margin."""
    res = loaded_explainer.explain_regime_classification(sample_feature_array, top_k=5)

    assert "regime_id" in res
    assert "regime_name" in res
    assert "base_score_logodds" in res
    assert "margin_score_logodds" in res
    assert "reconstruction_error" in res
    assert res["reconstruction_error"] < 1e-5
    assert len(res["top_features"]) == 5

    # Check top feature schema
    top1 = res["top_features"][0]
    assert top1["rank"] == 1
    assert top1["feature"] in PRODUCTION_19_FEATURES
    assert isinstance(top1["shap_value_logodds"], float)
    assert top1["direction"] in ["promotes_regime", "suppresses_regime"]


# -----------------------------------------------------------------------------
# 3. Active Monsoon Corrector TreeSHAP (HistGBDT)
# -----------------------------------------------------------------------------
def test_active_monsoon_corrector_treeshap_additivity(loaded_explainer, sample_feature_array):
    """Verify exact additivity of HistGBDT TreeExplainer: sum(phi_j) + E[f] == delta_pred."""
    res = loaded_explainer.explain_active_monsoon_correction(sample_feature_array, top_k=5)

    assert res["regime_id"] == 0
    assert res["regime_name"] == "Active Monsoon"
    assert "base_expected_delta_mm" in res
    assert "predicted_delta_mm" in res
    assert "reconstruction_error_mm" in res
    assert res["reconstruction_error_mm"] < 1e-5
    assert len(res["top_features"]) == 5

    top1 = res["top_features"][0]
    assert top1["rank"] == 1
    assert top1["feature"] in PRODUCTION_19_FEATURES
    assert isinstance(top1["shap_value_mm"], float)
    assert top1["direction"] in ["increases_rainfall_delta", "decreases_rainfall_delta"]


# -----------------------------------------------------------------------------
# 4. Soft MoE Mixture Decomposition
# -----------------------------------------------------------------------------
def test_moe_mixture_decomposition(loaded_explainer, sample_feature_array):
    """Verify that MoE mixture contributions sum accurately to blended prediction."""
    probs = np.array([0.70, 0.05, 0.10, 0.08, 0.05, 0.02], dtype=np.float32)
    raw = 50.0

    res = loaded_explainer.explain_moe_mixture(sample_feature_array, raw_fcst=raw, regime_probs=probs)

    assert "dominant_regime" in res
    assert "dominant_weight" in res
    assert "expert_contributions" in res
    assert len(res["expert_contributions"]) == 6

    # Verify sum of weights == 1.0
    sum_w = sum(c["weight_p_r"] for c in res["expert_contributions"].values())
    assert abs(sum_w - 1.0) < 1e-4

    # Verify sum of mixture contributions == final_calibrated_mm
    sum_c = sum(c["mixture_contribution_mm"] for c in res["expert_contributions"].values())
    assert abs(sum_c - res["final_calibrated_mm"]) < 0.1


# -----------------------------------------------------------------------------
# 5. Heavy Rainfall Risk Exceedance TreeSHAP
# -----------------------------------------------------------------------------
def test_heavy_rainfall_treeshap_additivity(loaded_explainer, sample_feature_array):
    """Verify exact additivity of heavy rainfall exceedance model TreeSHAP."""
    res = loaded_explainer.explain_heavy_rainfall_risk(sample_feature_array, threshold_key="heavy", top_k=5)

    assert res["threshold_category"] == "heavy"
    assert res["threshold_mm"] == 64.5
    assert "reconstruction_error" in res
    assert res["reconstruction_error"] < 1e-5
    assert len(res["top_features"]) == 5

    top1 = res["top_features"][0]
    assert top1["rank"] == 1
    assert top1["feature"] in PRODUCTION_19_FEATURES
    assert top1["direction"] in ["escalates_heavy_risk", "suppresses_heavy_risk"]


# -----------------------------------------------------------------------------
# 6. Explanation Stability & Determinism
# -----------------------------------------------------------------------------
def test_explainability_determinism(loaded_explainer, sample_feature_array):
    """Verify that repeated calls on identical input produce identical SHAP values."""
    res1 = loaded_explainer.explain_active_monsoon_correction(sample_feature_array, top_k=5)
    res2 = loaded_explainer.explain_active_monsoon_correction(sample_feature_array, top_k=5)

    for f1, f2 in zip(res1["top_features"], res2["top_features"]):
        assert f1["feature"] == f2["feature"]
        assert f1["shap_value_mm"] == f2["shap_value_mm"]


# -----------------------------------------------------------------------------
# 7. District-Level True Model-Derived Attribution (Replaces Heuristic)
# -----------------------------------------------------------------------------
def test_district_true_treeshap():
    """Verify that district-level explanation is model-derived, non-synthetic, and schema-compatible."""
    d = get_district_by_name("Ahmednagar")

    assert "district" in d
    assert "shap_attribution" in d
    shap_attr = d["shap_attribution"]

    assert len(shap_attr) >= 5
    for item in shap_attr:
        assert "feature" in item
        assert "impact" in item
        assert item["feature"] in PRODUCTION_19_FEATURES
        assert isinstance(item["impact"], (float, int))

    # Crucial test: Ensure legacy synthetic names are completely absent!
    legacy_features = [
        "Regime Routing",
        "Moisture Flux Convergence",
        "Topographic Slope Gradient",
        "Boundary Layer Humidity",
        "Vertical Wind Shear",
        "Raw GFS Wet Diffusion"
    ]
    current_features = [item["feature"] for item in shap_attr]
    for lf in legacy_features:
        assert lf not in current_features, f"Legacy synthetic feature '{lf}' found in district response!"


# -----------------------------------------------------------------------------
# 8. POST /api/predict Explainability Integration
# -----------------------------------------------------------------------------
def test_predict_endpoint_explainability_integration():
    """Verify that POST /api/predict contains valid model-derived explainability payload."""
    fv = FeatureVector(
        precip_raw=45.0, elevation=500.0, slope=12.0, wind_speed_850=14.0,
        wind_shear=25.0, mslp_anomaly=-3.0, rh850=85.0, upslope_flow=3.5
    )
    res = run_live_prediction(fv)

    assert hasattr(res, "explainability")
    assert res.explainability is not None

    exp = res.explainability
    assert "evidence_narrative" in exp
    assert "regime_attribution" in exp
    assert "correction_attribution" in exp
    assert "mixture_decomposition" in exp
    assert isinstance(exp["evidence_narrative"], str)
    assert len(exp["evidence_narrative"]) > 50


# -----------------------------------------------------------------------------
# 9. Dedicated POST /api/explain Endpoint
# -----------------------------------------------------------------------------
def test_dedicated_explain_endpoint():
    """Verify that POST /api/explain returns a comprehensive evidence chain."""
    fv = FeatureVector(
        precip_raw=45.0, elevation=500.0, slope=12.0, wind_speed_850=14.0,
        wind_shear=25.0, mslp_anomaly=-3.0, rh850=85.0, upslope_flow=3.5
    )
    exp = explain_prediction(fv)

    assert "evidence_narrative" in exp
    assert "regime_attribution" in exp
    assert "correction_attribution" in exp
    assert "mixture_decomposition" in exp
    assert "heavy_risk_attribution" in exp
    assert "model_provenance" in exp

    assert exp["model_provenance"]["pytorch_runtime_dependency"] is False
    assert exp["model_provenance"]["leakage_guard"] == "VERIFIED_FAIL_CLOSED"


# -----------------------------------------------------------------------------
# 10. Precomputed GET /api/explain/summary Endpoint
# -----------------------------------------------------------------------------
def test_explainability_summary_endpoint():
    """Verify that GET /api/explain/summary serves the verified precomputed artifact."""
    summary = get_explainability_summary()

    assert "metadata" in summary
    assert "global_feature_importance" in summary
    assert "shap_additivity_verification" in summary
    assert "explanation_stability" in summary
    assert "district_shap_attributions" in summary

    additivity = summary["shap_additivity_verification"]
    assert additivity["active_monsoon_corrector"]["is_additive"] is True
    assert additivity["regime_classifier"]["is_additive"] is True

    stability = summary["explanation_stability"]
    assert stability["active_monsoon_corrector_pearson_stability"] >= 0.95
    assert stability["regime_classifier_pearson_stability"] >= 0.95


# -----------------------------------------------------------------------------
# 11. Zero PyTorch Production Dependency During Explanation
# -----------------------------------------------------------------------------
def test_zero_torch_explainability_runtime(loaded_explainer, sample_feature_array):
    """Verify that explainability runs in pure scikit-learn / XGBoost without PyTorch."""
    torch_backup = sys.modules.get("torch")
    sys.modules["torch"] = None
    try:
        res = loaded_explainer.explain_active_monsoon_correction(sample_feature_array)
        assert res is not None
        assert res["reconstruction_error_mm"] < 1e-5
    finally:
        if torch_backup is not None:
            sys.modules["torch"] = torch_backup
        else:
            sys.modules.pop("torch", None)


# -----------------------------------------------------------------------------
# 12. Regression Preservation: Uncertainty & Core Prediction Fields
# -----------------------------------------------------------------------------
def test_uncertainty_and_core_regression_preserved():
    """Verify that all Phase 0-4 fields remain intact and uncorrupted."""
    fv = FeatureVector(precip_raw=30.0)
    res = run_live_prediction(fv)

    # Core predictions
    assert res.dominant_regime_id in range(6)
    assert res.calibrated_precipitation >= 0.0
    assert res.alert_level in ["Green", "Yellow", "Orange", "Red"]

    # Phase 4 uncertainty fields
    assert res.p10 is not None
    assert res.p50 is not None
    assert res.p90 is not None
    assert res.conformal_lower is not None
    assert res.conformal_upper is not None
    assert res.conformal_width is not None
    assert res.regime_entropy is not None
    assert res.p10 <= res.p50 <= res.p90
    assert res.conformal_lower <= res.conformal_upper
    assert 0.0 <= res.regime_entropy <= 1.0
