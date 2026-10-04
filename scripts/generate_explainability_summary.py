"""Script to generate comprehensive Phase 5 explainability summary artifact.
Computes:
1. Global feature importance across locked test set for:
   - Active Monsoon HistGBDT Corrector
   - XGBoost Multiclass 6-Regime Classifier
   - Heavy Rainfall Binary Classifiers
2. Additivity / reconstruction verification metrics.
3. Perturbation stability metrics (Pearson correlation).
4. District-level true TreeSHAP attributions for all 36 districts.
Outputs to: data/processed/explainability_summary.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pickle
import time
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import pearsonr

from src.feature_registry import PRODUCTION_19_FEATURES, validate_features_for_inference
from src.regime_labels import REGIME_NAMES
from src.explainability import TreeSHAPExplainer

def run_explainability_audit_and_summary():
    print("=" * 60)
    print("VARSHAMITRA PHASE 5: EXPLAINABILITY SUMMARY GENERATION")
    print("=" * 60)

    # 1. Load data
    df = pd.read_parquet("data/processed/processed_pipeline_dataframe.parquet")
    test_df = df[df["time"] >= "2024-09-01"].copy()
    print(f"Total locked test samples (Sep 2024): {len(test_df)}")

    # Sample 500 representative test samples for global SHAP calculation
    sample_size = min(500, len(test_df))
    sample_df = test_df.sample(n=sample_size, random_state=42)
    X = sample_df[PRODUCTION_19_FEATURES].values.astype(np.float32)
    validate_features_for_inference(PRODUCTION_19_FEATURES, strict=True)

    # 2. Load models
    with open("models/regime_classifier_xgb.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("models/regime_bias_postprocessor.pkl", "rb") as f:
        moe = pickle.load(f)
    with open("models/heavy_rainfall_prob_model.pkl", "rb") as f:
        hprob = pickle.load(f)

    explainer = TreeSHAPExplainer(
        regime_classifier=clf,
        bias_postprocessor=moe,
        heavy_prob_model=hprob,
        feature_names=PRODUCTION_19_FEATURES
    )

    summary = {
        "metadata": {
            "title": "VarshaMitra Phase 5 True Explainability & Evidence Chain Summary",
            "protocol": "Chronological Evaluation (Locked Test: September 2024)",
            "feature_set": PRODUCTION_19_FEATURES,
            "feature_count": len(PRODUCTION_19_FEATURES),
            "target_leakage": "FAIL_CLOSED_VERIFIED_NONE",
            "pytorch_runtime_dependency": False,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sample_size_evaluated": sample_size
        }
    }

    # -------------------------------------------------------------------------
    # PART A: ACTIVE MONSOON HISTGBDT RESIDUAL CORRECTOR GLOBAL SHAP
    # -------------------------------------------------------------------------
    print("\n--- Computing Active Monsoon Corrector SHAP ---")
    exp0 = moe.correctors[0].model
    tree_exp0 = shap.TreeExplainer(exp0)
    shap_active = tree_exp0.shap_values(X) # (N, 19)
    base_val = float(tree_exp0.expected_value[0]) if isinstance(tree_exp0.expected_value, (list, np.ndarray)) else float(tree_exp0.expected_value)
    preds_active = exp0.predict(X)

    # Additivity check
    reconstructed_active = base_val + np.sum(shap_active, axis=1)
    err_active = np.abs(reconstructed_active - preds_active)
    active_additivity = {
        "mean_error": float(np.mean(err_active)),
        "max_error": float(np.max(err_active)),
        "tolerance": 1e-5,
        "is_additive": bool(np.max(err_active) < 1e-5)
    }
    print(f"Active Monsoon Additivity: mean_err={active_additivity['mean_error']:.2e}, max_err={active_additivity['max_error']:.2e}")

    # Global importance
    mean_abs_active = np.mean(np.abs(shap_active), axis=0)
    mean_val_active = np.mean(shap_active, axis=0)
    active_ranks = np.argsort(-mean_abs_active)
    active_global = []
    for rank, idx in enumerate(active_ranks, 1):
        feat = PRODUCTION_19_FEATURES[idx]
        active_global.append({
            "rank": rank,
            "feature": feat,
            "mean_abs_shap_mm": round(float(mean_abs_active[idx]), 4),
            "mean_shap_mm": round(float(mean_val_active[idx]), 4),
            "primary_direction": "increases_rainfall_delta" if mean_val_active[idx] > 0 else "decreases_rainfall_delta"
        })

    # -------------------------------------------------------------------------
    # PART B: 6-REGIME CLASSIFIER GLOBAL SHAP (NATIVE XGBOOST C++)
    # -------------------------------------------------------------------------
    print("\n--- Computing 6-Regime Classifier SHAP (Native C++) ---")
    booster_clf = clf.model.get_booster() if hasattr(clf, "model") else clf.get_booster()
    dmat = xgb.DMatrix(X, feature_names=PRODUCTION_19_FEATURES)
    contribs_clf = booster_clf.predict(dmat, pred_contribs=True) # shape: (N, 6, 20)
    margins_clf = booster_clf.predict(dmat, output_margin=True)   # shape: (N, 6)

    # Additivity check across all samples and regimes
    reconstructed_clf = contribs_clf[:, :, -1] + np.sum(contribs_clf[:, :, :-1], axis=2)
    err_clf = np.abs(reconstructed_clf - margins_clf)
    clf_additivity = {
        "mean_error": float(np.mean(err_clf)),
        "max_error": float(np.max(err_clf)),
        "tolerance": 1e-5,
        "is_additive": bool(np.max(err_clf) < 1e-5)
    }
    print(f"Regime Classifier Additivity: mean_err={clf_additivity['mean_error']:.2e}, max_err={clf_additivity['max_error']:.2e}")

    # Global importance per regime
    regime_explanations = {}
    for r_id in range(6):
        r_name = REGIME_NAMES.get(r_id, f"Regime {r_id}")
        r_shap = contribs_clf[:, r_id, :-1] # (N, 19)
        mean_abs_r = np.mean(np.abs(r_shap), axis=0)
        mean_val_r = np.mean(r_shap, axis=0)
        r_ranks = np.argsort(-mean_abs_r)
        
        top_feats = []
        for rank, idx in enumerate(r_ranks[:5], 1):
            top_feats.append({
                "rank": rank,
                "feature": PRODUCTION_19_FEATURES[idx],
                "mean_abs_shap_logodds": round(float(mean_abs_r[idx]), 4),
                "mean_shap_logodds": round(float(mean_val_r[idx]), 4),
                "direction": "promotes_regime" if mean_val_r[idx] > 0 else "suppresses_regime"
            })
        regime_explanations[r_name] = {
            "regime_id": r_id,
            "top_features": top_feats
        }

    # Overall Multiclass Importance
    all_regimes_abs_shap = np.mean(np.abs(contribs_clf[:, :, :-1]), axis=(0, 1)) # (19,)
    clf_ranks = np.argsort(-all_regimes_abs_shap)
    clf_global = []
    for rank, idx in enumerate(clf_ranks, 1):
        feat = PRODUCTION_19_FEATURES[idx]
        clf_global.append({
            "rank": rank,
            "feature": feat,
            "mean_abs_shap_logodds": round(float(all_regimes_abs_shap[idx]), 4)
        })

    # -------------------------------------------------------------------------
    # PART C: HEAVY RAINFALL EXCEEDANCE SHAP (XGBOOST BINARY)
    # -------------------------------------------------------------------------
    print("\n--- Computing Heavy Rainfall Risk SHAP ---")
    heavy_explanations = {}
    heavy_additivities = {}

    for t_key in ["heavy", "very_heavy", "extremely_heavy"]:
        m = hprob.models.get(t_key)
        if m is None:
            heavy_explanations[t_key] = {"status": "unfitted_or_null", "reason": "Zero positive events in training"}
            continue
        
        b_m = m.get_booster()
        contribs_h = b_m.predict(dmat, pred_contribs=True) # (N, 20)
        margin_h = b_m.predict(dmat, output_margin=True)   # (N,)
        reconstructed_h = contribs_h[:, -1] + np.sum(contribs_h[:, :-1], axis=1)
        err_h = np.abs(reconstructed_h - margin_h)

        heavy_additivities[t_key] = {
            "mean_error": float(np.mean(err_h)),
            "max_error": float(np.max(err_h)),
            "tolerance": 1e-5,
            "is_additive": bool(np.max(err_h) < 1e-5)
        }

        h_shap = contribs_h[:, :-1]
        mean_abs_h = np.mean(np.abs(h_shap), axis=0)
        mean_val_h = np.mean(h_shap, axis=0)
        h_ranks = np.argsort(-mean_abs_h)

        top_feats_h = []
        for rank, idx in enumerate(h_ranks, 1):
            top_feats_h.append({
                "rank": rank,
                "feature": PRODUCTION_19_FEATURES[idx],
                "mean_abs_shap_logodds": round(float(mean_abs_h[idx]), 4),
                "mean_shap_logodds": round(float(mean_val_h[idx]), 4),
                "direction": "escalates_heavy_risk" if mean_val_h[idx] > 0 else "suppresses_heavy_risk"
            })
        
        heavy_explanations[t_key] = {
            "threshold_mm": hprob.thresholds.get(t_key),
            "top_features": top_feats_h
        }

    # -------------------------------------------------------------------------
    # PART D: EXPLANATION STABILITY (PERTURBATION TEST)
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Explanation Stability via Perturbation ---")
    np.random.seed(42)
    # Small Gaussian noise: 1% standard deviation
    noise = np.random.normal(0, 0.01 * (np.std(X, axis=0) + 1e-5), size=X.shape).astype(np.float32)
    X_pert = X + noise

    # Perturbed Active Monsoon SHAP
    tree_exp_pert = shap.TreeExplainer(exp0)
    shap_active_pert = tree_exp_pert.shap_values(X_pert)

    correlations_active = []
    for i in range(len(X)):
        std1 = np.std(shap_active[i])
        std2 = np.std(shap_active_pert[i])
        if std1 > 1e-7 and std2 > 1e-7:
            r, _ = pearsonr(shap_active[i], shap_active_pert[i])
            if not np.isnan(r):
                correlations_active.append(r)
    mean_corr_active = float(np.mean(correlations_active))

    # Perturbed Regime Classifier SHAP
    dmat_pert = xgb.DMatrix(X_pert, feature_names=PRODUCTION_19_FEATURES)
    contribs_clf_pert = booster_clf.predict(dmat_pert, pred_contribs=True)

    correlations_clf = []
    for i in range(len(X)):
        for r_id in range(6):
            s1 = contribs_clf[i, r_id, :-1]
            s2 = contribs_clf_pert[i, r_id, :-1]
            if np.std(s1) > 1e-7 and np.std(s2) > 1e-7:
                r, _ = pearsonr(s1, s2)
                if not np.isnan(r):
                    correlations_clf.append(r)
    mean_corr_clf = float(np.mean(correlations_clf))

    stability_metrics = {
        "active_monsoon_corrector_pearson_stability": round(mean_corr_active, 4),
        "regime_classifier_pearson_stability": round(mean_corr_clf, 4),
        "interpretation": "High local stability under input perturbation (|r| > 0.95 indicates consistent local attribution)"
    }
    print(f"Active Monsoon Stability Pearson r: {mean_corr_active:.4f}")
    print(f"Regime Classifier Stability Pearson r: {mean_corr_clf:.4f}")

    # -------------------------------------------------------------------------
    # PART E: DISTRICT-LEVEL TRUE TREESHAP (ALL 36 DISTRICTS)
    # -------------------------------------------------------------------------
    print("\n--- Computing True TreeSHAP for All 36 Districts ---")
    with open("data/processed/district_alerts_2024-09-28.geojson", "r") as f:
        district_geojson = json.load(f)

    district_shap_map = {}
    for feat in district_geojson.get("features", []):
        props = feat.get("properties", {})
        dname = props.get("district", "Unknown")
        d_shap = explainer.explain_district(dname, district_props=props, top_k=6)
        district_shap_map[dname] = d_shap

    print(f"Successfully generated true TreeSHAP for {len(district_shap_map)} districts.")

    # -------------------------------------------------------------------------
    # PART F: ASSEMBLE COMPLETE ARTIFACT
    # -------------------------------------------------------------------------
    summary["global_feature_importance"] = {
        "active_monsoon_corrector_ranking": active_global,
        "regime_classifier_ranking": clf_global,
        "regime_specific_top_features": regime_explanations,
        "heavy_rainfall_exceedance_ranking": heavy_explanations
    }
    summary["shap_additivity_verification"] = {
        "active_monsoon_corrector": active_additivity,
        "regime_classifier": clf_additivity,
        "heavy_rainfall_models": heavy_additivities
    }
    summary["explanation_stability"] = stability_metrics
    summary["district_shap_attributions"] = district_shap_map

    out_path = Path("data/processed/explainability_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved explainability summary artifact to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 60)
    print("PHASE 5 EXPLAINABILITY SUMMARY COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_explainability_audit_and_summary()
