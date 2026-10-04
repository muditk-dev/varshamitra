"""VarshaMitra Phase 6 Final Production API Contract & Registry Test Suite.
========================================================================
Validates all 11 REST endpoints, model registry integrity, startup verification,
PyTorch-free inference, feature schema conformance, and backward compatibility.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import json
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.feature_registry import (
    PRODUCTION_19_FEATURES,
    validate_features_for_inference,
    FeatureLeakageError
)
from src.model_registry import (
    ModelRegistry,
    validate_production_environment,
    FEATURE_SCHEMA_VERSION,
    ModelRegistryError
)
from api.main import (
    app,
    root,
    healthcheck,
    get_all_districts,
    get_district_by_name,
    run_live_prediction,
    explain_prediction,
    get_explainability_summary,
    get_verification_benchmarks,
    get_regime_profiles,
    get_provenance,
    run_what_if_simulation,
    FeatureVector,
    WhatIfRequest,
    MODELS,
    DATA_STORE
)


# -----------------------------------------------------------------------------
# 1. Root & Health Endpoints
# -----------------------------------------------------------------------------
def test_root_endpoint():
    """Verify GET / returns operational status and system identity."""
    res = root()
    assert res["system"] == "VarshaMitra Meteorological AI Engine"
    assert res["status"] == "OPERATIONAL"
    assert res["version"] == "1.0.0"
    assert "uptime_seconds" in res


def test_healthcheck_endpoint():
    """Verify GET /health reports healthy status, active models, and cached datasets."""
    res = healthcheck()
    assert res["status"] == "healthy"
    assert "regime_classifier" in res["models_loaded"]
    assert "bias_corrector" in res["models_loaded"]
    assert "heavy_prob_model" in res["models_loaded"]
    assert "uncertainty_engine" in res["models_loaded"]
    assert res["active_districts"] == 36


# -----------------------------------------------------------------------------
# 2. GIS & District Endpoints
# -----------------------------------------------------------------------------
def test_get_all_districts_geojson():
    """Verify GET /api/districts returns a valid 36-district FeatureCollection."""
    geojson = get_all_districts()
    assert geojson.get("type") == "FeatureCollection"
    features = geojson.get("features", [])
    assert len(features) == 36
    sample = features[0]["properties"]
    assert "district" in sample
    assert "corr_mean" in sample
    assert "alert_level" in sample


def test_get_district_by_name():
    """Verify GET /api/districts/{name} returns true model-derived TreeSHAP."""
    d = get_district_by_name("Ahmednagar")
    assert d["district"] == "Ahmednagar"
    assert "shap_attribution" in d
    shap_attr = d["shap_attribution"]
    assert len(shap_attr) >= 5

    # Verify no legacy synthetic feature names exist
    feature_names = [s["feature"] for s in shap_attr]
    assert "Regime Routing" not in feature_names
    assert "Raw GFS Wet Diffusion" not in feature_names
    for item in shap_attr:
        assert item["feature"] in PRODUCTION_19_FEATURES
        assert isinstance(item["impact"], (float, int))


# -----------------------------------------------------------------------------
# 3. Live Prediction Contract Endpoint (POST /api/predict)
# -----------------------------------------------------------------------------
def test_predict_endpoint_full_contract():
    """Verify POST /api/predict returns complete end-to-end prediction contract."""
    fv = FeatureVector(
        precip_raw=45.0, elevation=500.0, slope=12.0, wind_speed_850=14.0,
        wind_shear=25.0, mslp_anomaly=-3.0, rh850=85.0, upslope_flow=3.5
    )
    res = run_live_prediction(fv)

    # 1. Core predictions
    assert res.dominant_regime_id in range(6)
    assert res.dominant_regime_name in [
        "Active Monsoon", "Break Monsoon", "Monsoon Depression / Low",
        "Orographic", "Coastal", "Western Disturbance"
    ]
    assert 0.0 <= res.regime_confidence <= 1.0
    assert len(res.regime_probabilities) == 6
    assert abs(sum(res.regime_probabilities.values()) - 1.0) < 1e-3
    assert res.raw_precipitation == 45.0
    assert res.calibrated_precipitation >= 0.0
    assert res.alert_level in ["Green", "Yellow", "Orange", "Red"]

    # 2. Uncertainty fields (Phase 4)
    assert res.p10 is not None and res.p50 is not None and res.p90 is not None
    assert res.p10 <= res.p50 <= res.p90
    assert res.conformal_lower <= res.conformal_upper
    assert 0.0 <= res.regime_entropy <= 1.0

    # 3. Explainability payload (Phase 5)
    assert res.explainability is not None
    assert "evidence_narrative" in res.explainability
    assert "regime_attribution" in res.explainability
    assert "correction_attribution" in res.explainability

    # 4. Provenance & schema metadata (Phase 6)
    assert res.feature_schema_version == "production-19-v1"
    assert res.model_provenance is not None
    assert "regime_classifier" in res.model_provenance
    assert "bias_postprocessor" in res.model_provenance
    assert "heavy_rainfall" in res.model_provenance
    assert "uncertainty" in res.model_provenance


# -----------------------------------------------------------------------------
# 4. Contingency Lab Endpoint (POST /api/what-if)
# -----------------------------------------------------------------------------
def test_what_if_simulation_endpoint():
    """Verify POST /api/what-if simulates dynamic atmospheric response curves."""
    req = WhatIfRequest(mfc_delta_pct=15.0, shear_delta=-5.0, mslp_delta=-4.0)
    res = run_what_if_simulation(req)
    assert res.baseline_basin_mean == 26.8
    assert res.perturbed_basin_mean > 0.0
    assert len(res.time_series_hours) == 5
    assert len(res.baseline_hourly) == 5
    assert len(res.perturbed_hourly) == 5
    assert "Konkan Coastal" in res.regional_shifts
    assert "Western Ghats Crest" in res.regional_shifts


# -----------------------------------------------------------------------------
# 5. Verification Endpoint (GET /api/verification)
# -----------------------------------------------------------------------------
def test_verification_endpoint_benchmarks():
    """Verify GET /api/verification exposes B0-B4 progression ladder and disclosures."""
    res = get_verification_benchmarks()
    assert "benchmarks" in res
    assert len(res["benchmarks"]) == 5
    tiers = [b["tier"] for b in res["benchmarks"]]
    assert any("B0" in t for t in tiers)
    assert any("B2" in t for t in tiers)
    assert any("B4" in t for t in tiers)
    assert "statistical_significance_status" in res
    assert "Descriptive" in res["statistical_significance_status"]

    # Verify existing frontend fields are preserved
    assert "overall" in res
    assert "rmse_skill_gain_pct" in res["overall"]


# -----------------------------------------------------------------------------
# 6. Monsoon Intelligence Regimes (GET /api/regimes)
# -----------------------------------------------------------------------------
def test_regimes_endpoint():
    """Verify GET /api/regimes returns the 6 canonical synoptic regime profiles."""
    res = get_regime_profiles()
    assert len(res["regimes"]) == 6
    regime_ids = [r["id"] for r in res["regimes"]]
    assert set(regime_ids) == {0, 1, 2, 3, 4, 5}
    for r in res["regimes"]:
        assert "name" in r
        assert "color" in r
        assert "westerly_850" in r
        assert "description" in r


# -----------------------------------------------------------------------------
# 7. Dedicated Explainability Endpoints
# -----------------------------------------------------------------------------
def test_dedicated_explain_endpoint():
    """Verify POST /api/explain returns full evidence chain and model provenance."""
    fv = FeatureVector(precip_raw=55.0, wind_shear=30.0, wind_speed_850=16.0)
    exp = explain_prediction(fv)
    assert "evidence_narrative" in exp
    assert "regime_attribution" in exp
    assert "correction_attribution" in exp
    assert "mixture_decomposition" in exp
    assert "heavy_risk_attribution" in exp
    assert "model_provenance" in exp
    assert exp["model_provenance"]["pytorch_runtime_dependency"] is False


def test_explain_summary_endpoint():
    """Verify GET /api/explain/summary returns precomputed global TreeSHAP artifact."""
    summary = get_explainability_summary()
    assert "metadata" in summary
    assert "global_feature_importance" in summary
    assert "shap_additivity_verification" in summary
    assert "district_shap_attributions" in summary
    assert len(summary["district_shap_attributions"]) == 36


# -----------------------------------------------------------------------------
# 8. Provenance Endpoint (GET /api/provenance)
# -----------------------------------------------------------------------------
def test_provenance_endpoint():
    """Verify GET /api/provenance returns comprehensive registry, schema, and lineage specs."""
    prov = get_provenance()
    assert prov["system"] == "VarshaMitra Meteorological AI Engine"
    assert prov["status"] == "OPERATIONAL"

    # Feature schema check
    schema = prov["feature_schema"]
    assert schema["schema_version"] == "production-19-v1"
    assert schema["feature_count"] == 19
    assert schema["features"] == PRODUCTION_19_FEATURES

    # Model registry check
    models = prov["model_registry"]["models"]
    assert len(models) == 4
    for mname in ["regime_classifier", "bias_postprocessor", "heavy_rainfall", "uncertainty"]:
        assert mname in models
        assert "version" in models[mname]
        assert "sha256_short" in models[mname]
        assert models[mname]["pytorch_free"] is True

    # Data lineage check
    assert prov["data_lineage"] is not None
    assert "spatial_domain" in prov["data_lineage"]
    assert "data_sources" in prov["data_lineage"]


# -----------------------------------------------------------------------------
# 9. Model Registry & Startup Validation Suite
# -----------------------------------------------------------------------------
def test_model_registry_manifest_hashes():
    """Verify that ModelRegistry validates all artifact hashes and detects mismatches."""
    registry = ModelRegistry()
    results = registry.verify_all_artifacts()
    assert len(results) == 4
    assert all(results.values())


def test_model_registry_tamper_detection(tmp_path):
    """Verify that ModelRegistry raises ModelRegistryError if a model artifact is tampered."""
    manifest_file = tmp_path / "tampered_manifest.json"
    manifest_data = {
        "manifest_version": "1.0.0",
        "models": {
            "fake_model": {
                "artifact_filename": "regime_classifier_xgb.pkl",
                "sha256": "bad_hash_000000000000000000000000000000000000000000000000000000000000",
                "features_required": 19
            }
        }
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest_data, f)

    bad_registry = ModelRegistry(manifest_path=manifest_file, models_dir=ROOT_DIR / "models")
    with pytest.raises(ModelRegistryError, match="hash mismatch"):
        bad_registry.verify_all_artifacts()


def test_startup_validation_gatekeeper():
    """Verify that validate_production_environment executes and passes cleanly."""
    report = validate_production_environment()
    assert report["status"] == "VALIDATED"
    assert report["features_count"] == 19
    assert report["regimes_count"] == 6
    assert report["pytorch_free"] is True


# -----------------------------------------------------------------------------
# 10. PyTorch-Free Runtime Independence
# -----------------------------------------------------------------------------
def test_pytorch_free_complete_inference_pipeline():
    """Verify that entire inference, uncertainty, and TreeSHAP pipeline runs with PyTorch disabled."""
    torch_backup = sys.modules.get("torch")
    sys.modules["torch"] = None
    try:
        fv = FeatureVector(precip_raw=40.0, wind_shear=20.0, elevation=450.0)
        res = run_live_prediction(fv)
        assert res.calibrated_precipitation >= 0.0
        assert res.conformal_width > 0.0
        assert res.explainability is not None
    finally:
        if torch_backup is not None:
            sys.modules["torch"] = torch_backup
        else:
            sys.modules.pop("torch", None)
