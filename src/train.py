"""VarshaMitra End-to-End Pipeline Orchestrator.
==============================================
Orchestrates data ingestion, preprocessing, physics feature engineering,
regime classification, regime-tailored bias correction, heavy rainfall probability,
district aggregation, and verification scoring.
"""

import os
import json
import logging
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, Any

import numpy as np
import pandas as pd
import xarray as xr

from src.data_ingestion import ingest_all_pilot_data
from src.preprocessing import merge_meteorological_streams
from src.feature_engineering import compute_physical_features, get_feature_matrix_dataframe
from src.regime_labels import label_monsoon_regimes, REGIME_NAMES
from src.regime_classifier import train_and_evaluate_regime_classifiers, FEATURE_COLS
from src.bias_correction import RegimeAwarePostProcessor
from src.heavy_rainfall_probability import FocalLossXGBoostProbabilityModel
from src.district_aggregation import aggregate_grid_to_districts
from src.explainability import MeteorologicalExplainer
from src.metrics import compute_full_verification_suite, compute_fractions_skill_score

import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("varshamitra.pipeline")


def run_full_pipeline(
    start_date: str = "2024-06-01",
    end_date: str = "2024-09-30",
    data_dir: str = "data",
    models_dir: str = "models"
) -> Dict[str, any]:
    """Execute complete VarshaMitra pipeline from raw data to verified products."""
    logger.info("=================================================================")
    logger.info("STARTING VARSHAMITRA REGIME-AWARE RAINFALL POST-PROCESSING PIPELINE")
    logger.info("Smart India Hackathon Problem Statement 26080 (NCMRWF / MoES)")
    logger.info("=================================================================")
    
    m_path = Path(models_dir)
    m_path.mkdir(parents=True, exist_ok=True)
    p_path = Path(data_dir) / "processed"
    p_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Ingestion
    logger.info("\n>>> PHASE 1: Data Ingestion")
    ingest_results = ingest_all_pilot_data(start_date=start_date, end_date=end_date, data_dir=data_dir)
    dem_ds = ingest_results["dem"]
    forecast_ds = ingest_results["forecast"]
    observed_ds = ingest_results["observed"]
    atmos_ds = ingest_results["atmospheric"]
    districts_gdf = ingest_results["districts"]
    
    # 2. Preprocessing
    logger.info("\n>>> PHASE 2: Preprocessing & Merging")
    unified_ds = merge_meteorological_streams(forecast_ds, observed_ds, atmos_ds, dem_ds)
    
    # 3. Physics-based Feature Engineering
    logger.info("\n>>> PHASE 2 (cont): Physics-based Feature Engineering")
    featured_ds = compute_physical_features(unified_ds)
    
    # 4. Weak Supervision Regime Labeling
    logger.info("\n>>> PHASE 3: Weak Supervision Regime Labeling")
    regime_da = label_monsoon_regimes(featured_ds)
    featured_ds["regime"] = regime_da
    
    # Save processed analysis-ready dataset
    processed_nc = p_path / f"varshamitra_features_{start_date}_{end_date}.nc"
    featured_ds.to_netcdf(processed_nc)
    logger.info(f"Saved feature-engineered dataset to {processed_nc}")
    
    # 5. Regime Classification Training
    logger.info("\n>>> PHASE 3 (cont): Regime Classifier Training")
    clf_results = train_and_evaluate_regime_classifiers(featured_ds, train_split_ratio=0.75, epochs=10)
    xgb_regime_model = clf_results["xgb_model"]
    split_idx = clf_results["split_idx"]
    
    # Save classifier
    with open(m_path / "regime_classifier_xgb.pkl", "wb") as f:
        pickle.dump(xgb_regime_model, f)
        
    # 6. Tabular DataFrame for Bias Correction and Probability Modeling
    df = get_feature_matrix_dataframe(featured_ds)
    df["regime"] = featured_ds["regime"].values.ravel()
    
    # Train / Test split along temporal dimension
    time_unique = featured_ds["time"].values
    split_date = time_unique[split_idx]
    train_mask = df["time"] < split_date
    test_mask = df["time"] >= split_date
    
    X_train = df.loc[train_mask, FEATURE_COLS].values
    raw_train = df.loc[train_mask, "precip_raw"].values
    obs_train = df.loc[train_mask, "precip_obs"].values
    reg_train = df.loc[train_mask, "regime"].values
    
    X_test = df.loc[test_mask, FEATURE_COLS].values
    raw_test = df.loc[test_mask, "precip_raw"].values
    obs_test = df.loc[test_mask, "precip_obs"].values
    reg_test = df.loc[test_mask, "regime"].values
    
    # 7. Regime-Specific Bias Correction Suite
    logger.info("\n>>> PHASE 4: Regime-Specific Bias Correction Training")
    post_processor = RegimeAwarePostProcessor()
    post_processor.fit(X_train, raw_train, obs_train, reg_train)
    
    # Predict on test set
    corr_test = post_processor.predict(X_test, raw_test, reg_test)
    df.loc[test_mask, "precip_corr"] = corr_test
    # Predict on train set for full coverage
    corr_train = post_processor.predict(X_train, raw_train, reg_train)
    df.loc[train_mask, "precip_corr"] = corr_train
    
    # Save post-processor
    with open(m_path / "regime_bias_postprocessor.pkl", "wb") as f:
        pickle.dump(post_processor, f)
        
    # 8. Heavy Rainfall Probability Model
    logger.info("\n>>> PHASE 5: Heavy Rainfall Exceedance Probability Modeling")
    prob_model = FocalLossXGBoostProbabilityModel()
    prob_model.fit(X_train, obs_train)
    
    test_probs = prob_model.predict_proba(X_test)
    all_probs = prob_model.predict_proba(df[FEATURE_COLS].values)
    df["p_heavy"] = all_probs["p_heavy"]
    df["p_very_heavy"] = all_probs["p_very_heavy"]
    df["p_extremely_heavy"] = all_probs["p_extremely_heavy"]
    
    reliability_data = prob_model.compute_reliability_curve(X_test, obs_test)
    
    with open(m_path / "heavy_rainfall_prob_model.pkl", "wb") as f:
        pickle.dump(prob_model, f)
        
    # 9. Explainability
    logger.info("\n>>> PHASE 6: Explainability & Natural Language Generator")
    explainer = MeteorologicalExplainer(xgb_regime_model.model, FEATURE_COLS)
    
    # 10. District Aggregation
    logger.info("\n>>> PHASE 6 (cont): District Zonal Aggregation")
    # Sample recent date
    sample_date = str(time_unique[-3])[:10]
    district_alerts_gdf = aggregate_grid_to_districts(df, districts_gdf, date_str=sample_date)
    
    # Attach explainability narrative to districts
    narratives = []
    for _, d_row in district_alerts_gdf.iterrows():
        narrative = explainer.generate_plain_language_narrative(d_row)
        narratives.append(narrative)
    district_alerts_gdf["explanation"] = narratives
    
    # Save sample district alerts
    district_alerts_gdf.to_file(p_path / f"district_alerts_{sample_date}.geojson", driver="GeoJSON")
    
    # 11. Verification Scoring
    logger.info("\n>>> PHASE 7: Meteorological Verification Suite")
    verification_matrix = compute_full_verification_suite(
        raw_fcst=raw_test,
        corr_fcst=corr_test,
        obs=obs_test,
        regimes=reg_test,
        threshold=10.0
    )
    
    # Compute Fractions Skill Score for a sample test 2D spatial slice
    sample_t = featured_ds["time"].values[-1]
    fcst_slice = featured_ds["precip_raw"].sel(time=sample_t).values
    obs_slice = featured_ds["precip_obs"].sel(time=sample_t).values
    fss_score = compute_fractions_skill_score(fcst_slice, obs_slice, threshold=15.0, window_size=3)
    verification_matrix["spatial_fss_sample"] = fss_score
    
    verif_file = p_path / "verification_scores_summary.json"
    with open(verif_file, "w") as f:
        json.dump(verification_matrix, f, indent=2)
    logger.info(f"Saved complete verification summary to {verif_file}")
    
    # Save processed tabular DataFrame for fast dashboard loading
    df_file = p_path / "processed_pipeline_dataframe.parquet"
    df.to_parquet(df_file, index=False)
    logger.info(f"Saved processed pipeline dataframe to {df_file}")
    
    logger.info("\n=================================================================")
    logger.info("VARSHAMITRA PIPELINE EXECUTION SUCCESSFUL!")
    logger.info(f"  Overall Raw RMSE:        {verification_matrix['overall']['raw']['rmse']:.2f} mm")
    logger.info(f"  Overall Corrected RMSE:  {verification_matrix['overall']['corrected']['rmse']:.2f} mm")
    logger.info(f"  Skill Gain (RMSE Red.):  {verification_matrix['overall']['rmse_skill_gain_pct']:.1f}%")
    logger.info(f"  Regime Classifier Acc:   {clf_results['xgb_accuracy']*100:.1f}% (XGB) | {clf_results['cnn_accuracy']*100:.1f}% (CNN)")
    logger.info("=================================================================")
    
    return {
        "dataset": featured_ds,
        "dataframe": df,
        "districts_gdf": district_alerts_gdf,
        "verification": verification_matrix,
        "classifier_results": clf_results,
        "reliability": reliability_data
    }


if __name__ == "__main__":
    results = run_full_pipeline()
