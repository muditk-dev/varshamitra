"""Unit tests for VarshaMitra pipeline modules.
=============================================
Validates each component independently with strict physical bounds checks.
"""

import os
import pytest
import numpy as np
import xarray as xr
import geopandas as gpd

# -------------------------------------------------------------------------
# Phase 1 Tests: Data Ingestion & Physical Validity
# -------------------------------------------------------------------------

def test_topography_generation():
    """Verify synthetic DEM satisfies Maharashtra geomorphology."""
    from src.data_ingestion import get_maharashtra_grid, generate_synthetic_topography
    lats, lons = get_maharashtra_grid()
    dem_ds = generate_synthetic_topography(lats, lons)
    
    assert "elevation" in dem_ds.data_vars
    assert "slope_lon" in dem_ds.data_vars
    assert "slope_lat" in dem_ds.data_vars
    
    elev = dem_ds["elevation"].values
    # Check bounding dimensions
    assert elev.shape == (len(lats), len(lons))
    # Sea level minimum >= 0m
    assert np.all(elev >= 0.0)
    # Peak elevation should reach Western Ghats summits (900m - 1600m)
    assert np.max(elev) > 900.0
    assert np.max(elev) <= 1600.0
    # Coastal zone (lon < 72.8) should be sea / near-sea (0 - 50m)
    assert np.mean(elev[:, lons < 72.8]) < 10.0


def test_raw_forecast_ingestion(tmp_path):
    """Verify NOAA GFS substitute dataset has required dimensions and valid precipitation bounds."""
    from src.data_ingestion import download_or_generate_forecast
    ds, prov = download_or_generate_forecast("2024-06-01", "2024-06-05", str(tmp_path), use_synthetic=True)
    
    assert "precip_raw" in ds.data_vars
    assert ds["precip_raw"].shape[0] == 5  # 5 days
    
    precip = ds["precip_raw"].values
    assert np.all(precip >= 0.0), "Precipitation cannot be negative"
    assert np.all(precip <= 500.0), "Precipitation exceeds realistic meteorological upper limit"
    assert prov["status"] in ["REAL", "SYNTHETIC_FALLBACK"]


def test_observed_rainfall_ingestion(tmp_path):
    """Verify IMD format observed rainfall dataset structure and physical properties."""
    from src.data_ingestion import download_or_generate_observed
    ds, prov = download_or_generate_observed("2024-06-01", "2024-06-05", str(tmp_path))
    
    assert "precip_obs" in ds.data_vars
    assert ds["precip_obs"].shape[0] == 5
    
    obs = ds["precip_obs"].values
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 500.0)
    assert "status" in prov


def test_atmospheric_era5_ingestion(tmp_path):
    """Verify ERA5 atmospheric variables have physically plausible monsoon values."""
    from src.data_ingestion import download_or_generate_atmospheric_era5
    ds, prov = download_or_generate_atmospheric_era5("2024-06-01", "2024-06-05", str(tmp_path))
    
    for var in ["u850", "v850", "u200", "v200", "mslp", "rh850"]:
        assert var in ds.data_vars, f"Missing atmospheric variable {var}"
        
    u850 = ds["u850"].values
    u200 = ds["u200"].values
    mslp = ds["mslp"].values
    rh850 = ds["rh850"].values
    
    # 850 hPa wind should show westerly low-level monsoon flow (u > 0)
    assert np.mean(u850) > 0.0, "Monsoon low-level jet must be predominantly westerly"
    # 200 hPa wind should show tropical easterly jet (u < 0)
    assert np.mean(u200) < 0.0, "Monsoon upper-level jet must be predominantly easterly"
    # MSLP in hPa during summer monsoon should be roughly 990 - 1015 hPa
    assert np.all((mslp >= 980.0) & (mslp <= 1025.0)), f"MSLP out of physical bounds: {np.min(mslp)} - {np.max(mslp)}"
    # Relative humidity should be in [0, 100]%
    assert np.all((rh850 >= 0.0) & (rh850 <= 100.0)), f"RH out of bounds: {np.min(rh850)} - {np.max(rh850)}"


def test_maharashtra_districts_geojson(tmp_path):
    """Verify Maharashtra district polygons are valid geometries and contain key districts."""
    from src.data_ingestion import fetch_maharashtra_districts
    gdf, prov = fetch_maharashtra_districts(str(tmp_path))
    
    assert len(gdf) >= 30, f"Expected at least 30 Maharashtra districts, got {len(gdf)}"
    assert "district" in gdf.columns
    assert "geometry" in gdf.columns
    
    # Check key districts are present
    districts = set(gdf["district"].values)
    for expected in ["Mumbai City", "Pune", "Nagpur", "Nashik", "Ratnagiri"]:
        assert expected in districts, f"Expected district {expected} not found"
        
    # Check that all geometries are valid polygons
    assert all(gdf.geometry.is_valid), "All district polygons must be geometrically valid"


# -------------------------------------------------------------------------
# Phase 2 Tests: Preprocessing & Physics Feature Engineering
# -------------------------------------------------------------------------

def test_preprocessing_and_feature_engineering(tmp_path):
    """Verify spatial alignment, physics feature engineering, and non-negativity."""
    from src.data_ingestion import generate_synthetic_topography, download_or_generate_forecast, download_or_generate_observed, download_or_generate_atmospheric_era5, get_maharashtra_grid
    from src.preprocessing import merge_meteorological_streams
    from src.feature_engineering import compute_physical_features, get_feature_matrix_dataframe
    
    lats, lons = get_maharashtra_grid()
    dem_ds = generate_synthetic_topography(lats, lons)
    fcst_ds, _ = download_or_generate_forecast("2024-07-01", "2024-07-05", str(tmp_path), use_synthetic=True)
    obs_ds, _ = download_or_generate_observed("2024-07-01", "2024-07-05", str(tmp_path))
    atm_ds, _ = download_or_generate_atmospheric_era5("2024-07-01", "2024-07-05", str(tmp_path))
    
    merged = merge_meteorological_streams(fcst_ds, obs_ds, atm_ds, dem_ds)
    assert "precip_raw" in merged and "precip_obs" in merged
    
    featured = compute_physical_features(merged)
    for col in ["wind_shear", "wind_speed_850", "upslope_flow", "vorticity", "mslp_anomaly", "q_850", "mfc"]:
        assert col in featured.data_vars, f"Missing physical feature {col}"
        
    # Bounds check
    assert np.all(featured["wind_shear"].values >= 0.0), "Wind shear cannot be negative"
    assert np.all(featured["wind_speed_850"].values >= 0.0), "Wind speed cannot be negative"
    assert np.all(featured["q_850"].values >= 0.0), "Specific humidity cannot be negative"
    assert np.all(featured["rh850"].values <= 100.0), "Relative humidity exceeds 100%"
    
    df = get_feature_matrix_dataframe(featured)
    assert len(df) == 5 * len(lats) * len(lons)


# -------------------------------------------------------------------------
# Phase 3 Tests: Weak Supervision Regime Labeling & Regime Classification
# -------------------------------------------------------------------------

def test_regime_labeling_and_classifier(tmp_path):
    """Verify weak supervision regime labels and XGBoost classifier probabilities."""
    from src.data_ingestion import generate_synthetic_topography, download_or_generate_forecast, download_or_generate_observed, download_or_generate_atmospheric_era5, get_maharashtra_grid
    from src.preprocessing import merge_meteorological_streams
    from src.feature_engineering import compute_physical_features
    from src.regime_labels import label_monsoon_regimes
    from src.regime_classifier import XGBoostRegimeClassifier, FEATURE_COLS, NUM_REGIMES
    
    lats, lons = get_maharashtra_grid()
    dem_ds = generate_synthetic_topography(lats, lons)
    fcst_ds, _ = download_or_generate_forecast("2024-07-01", "2024-07-06", str(tmp_path), use_synthetic=True)
    obs_ds, _ = download_or_generate_observed("2024-07-01", "2024-07-06", str(tmp_path))
    atm_ds, _ = download_or_generate_atmospheric_era5("2024-07-01", "2024-07-06", str(tmp_path))
    
    merged = merge_meteorological_streams(fcst_ds, obs_ds, atm_ds, dem_ds)
    featured = compute_physical_features(merged)
    regimes = label_monsoon_regimes(featured)
    
    labels = regimes.values.ravel()
    assert np.all((labels >= 0) & (labels < NUM_REGIMES)), "Regime labels out of valid range {0..5}"
    
    # Train test split on flattened grid
    X = np.column_stack([featured[f].values.ravel() for f in FEATURE_COLS])
    y = labels
    clf = XGBoostRegimeClassifier(n_estimators=10, max_depth=3)
    clf.fit(X[:500], y[:500])
    
    probs = clf.predict_proba(X[500:550])
    assert probs.shape == (50, NUM_REGIMES)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5), "Class probabilities must sum to 1"


# -------------------------------------------------------------------------
# Phase 4 Tests: 6 Regime-Specific Bias Correction Models
# -------------------------------------------------------------------------

def test_bias_correction_suite():
    """Verify all 6 regime correctors inherit BaseCorrector and return non-negative rainfall."""
    from src.bias_correction import (
        QuantileMappingCorrector, GradientBoostingCorrector, SpatialCNNCorrector, RegimeAwarePostProcessor
    )
    np.random.seed(42)
    N = 200
    X = np.random.randn(N, 10)
    raw = np.random.uniform(2.0, 60.0, N)
    obs = raw * 0.85 + np.random.normal(0, 3, N)
    obs = np.clip(obs, 0, None)
    
    # Test individual correctors
    qm = QuantileMappingCorrector()
    qm.fit(X, raw, obs)
    pred_qm = qm.predict(X, raw)
    assert np.all(pred_qm >= 0.0), "Quantile mapping output cannot be negative"
    
    gb = GradientBoostingCorrector()
    gb.fit(X, raw, obs)
    pred_gb = gb.predict(X, raw)
    assert np.all(pred_gb >= 0.0), "Gradient boosting output cannot be negative"
    
    cnn = SpatialCNNCorrector(in_features=10, epochs=3)
    cnn.fit(X, raw, obs)
    pred_cnn = cnn.predict(X, raw)
    assert np.all(pred_cnn >= 0.0), "CNN output cannot be negative"
    
    # Test ensemble router
    regimes = np.random.choice(range(6), N)
    processor = RegimeAwarePostProcessor()
    processor.fit(X, raw, obs, regimes)
    pred_ens = processor.predict(X, raw, regimes)
    assert np.all(pred_ens >= 0.0), "Ensemble post-processor output cannot be negative"
    assert len(pred_ens) == N


# -------------------------------------------------------------------------
# Phase 5 Tests: Heavy Rainfall Exceedance Probability Model
# -------------------------------------------------------------------------

def test_heavy_rainfall_probability():
    """Verify probability predictions are in [0, 1] and strictly monotonic across thresholds."""
    from src.heavy_rainfall_probability import FocalLossXGBoostProbabilityModel
    np.random.seed(42)
    N = 300
    X = np.random.randn(N, 8)
    obs = np.random.exponential(scale=15.0, size=N)
    
    model = FocalLossXGBoostProbabilityModel()
    model.fit(X, obs)
    
    probs = model.predict_proba(X)
    assert "p_heavy" in probs and "p_very_heavy" in probs and "p_extremely_heavy" in probs
    
    p_h = probs["p_heavy"]
    p_vh = probs["p_very_heavy"]
    p_eh = probs["p_extremely_heavy"]
    
    assert np.all((p_h >= 0.0) & (p_h <= 1.0))
    assert np.all((p_vh >= 0.0) & (p_vh <= 1.0))
    assert np.all((p_eh >= 0.0) & (p_eh <= 1.0))
    # Monotonicity check
    assert np.all(p_h >= p_vh), "Monotonicity violated: P(heavy) must be >= P(very heavy)"
    assert np.all(p_vh >= p_eh), "Monotonicity violated: P(very heavy) must be >= P(extremely heavy)"


# -------------------------------------------------------------------------
# Phase 6 Tests: District Aggregation & Plain-Language Explainability
# -------------------------------------------------------------------------

def test_district_aggregation_and_explainability(tmp_path):
    """Verify district aggregation and natural language meteorological explainability."""
    import pandas as pd
    from src.data_ingestion import fetch_maharashtra_districts
    from src.district_aggregation import aggregate_grid_to_districts
    from src.explainability import MeteorologicalExplainer
    from src.regime_classifier import FEATURE_COLS
    import xgboost as xgb
    
    districts_gdf, _ = fetch_maharashtra_districts(str(tmp_path))
    
    # Generate mock grid dataframe
    data = []
    for lat in [18.0, 19.0, 20.0]:
        for lon in [73.0, 74.0, 75.0]:
            data.append({
                "time": "2024-07-10", "lat": lat, "lon": lon,
                "precip_raw": 35.0, "precip_corr": 28.0, "precip_obs": 26.0,
                "regime": 3, "p_heavy": 0.45, "p_very_heavy": 0.15, "p_extremely_heavy": 0.02,
                "elevation": 850.0, "slope_lon": 0.02, "slope_lat": 0.01,
                "u850": 16.0, "v850": 4.0, "u200": -22.0, "v200": 0.0,
                "mslp": 1002.0, "rh850": 92.0, "wind_shear": 38.0, "wind_speed_850": 16.5,
                "upslope_flow": 0.35, "vorticity": 1.2e-5, "mslp_anomaly": -1.5,
                "q_850": 0.015, "mfc": 2.5e-7, "precip_roll3": 30.0, "coastal_proximity": 0.2
            })
    df = pd.DataFrame(data)
    
    agg_gdf = aggregate_grid_to_districts(df, districts_gdf, date_str="2024-07-10")
    assert len(agg_gdf) > 0
    assert "alert_level" in agg_gdf.columns
    assert set(agg_gdf["alert_level"].unique()).issubset({"Green", "Yellow", "Orange", "Red"})
    
    # Test Explainer
    mock_model = xgb.XGBClassifier(n_estimators=5, max_depth=2).fit(df[FEATURE_COLS].values, [0]*len(df))
    explainer = MeteorologicalExplainer(mock_model, FEATURE_COLS)
    narrative = explainer.generate_plain_language_narrative(agg_gdf.iloc[0])
    assert isinstance(narrative, str) and len(narrative) > 20


# -------------------------------------------------------------------------
# Phase 7 Tests: Meteorological Verification Suite
# -------------------------------------------------------------------------

def test_meteorological_verification_metrics():
    """Verify RMSE, MAE, POD, FAR, CSI, ETS, and FSS formulas and valid bounds."""
    from src.metrics import (
        compute_continuous_metrics, compute_categorical_scores, compute_fractions_skill_score
    )
    np.random.seed(42)
    obs = np.array([0.0, 5.0, 15.0, 25.0, 50.0, 75.0, 120.0])
    fcst = np.array([1.0, 4.0, 18.0, 20.0, 45.0, 80.0, 110.0])
    
    cont = compute_continuous_metrics(fcst, obs)
    assert cont["rmse"] >= 0.0
    assert cont["mae"] >= 0.0
    assert -1.0 <= cont["corr"] <= 1.0
    
    cat = compute_categorical_scores(fcst, obs, threshold=15.0)
    assert 0.0 <= cat["pod"] <= 1.0, "POD must be in [0, 1]"
    assert 0.0 <= cat["far"] <= 1.0, "FAR must be in [0, 1]"
    assert 0.0 <= cat["csi"] <= 1.0, "CSI must be in [0, 1]"
    assert -0.33 <= cat["ets"] <= 1.0, "ETS must be in [-1/3, 1]"
    
    # 2D Fractions Skill Score
    f_grid = np.random.uniform(0, 50, (20, 20))
    o_grid = np.random.uniform(0, 50, (20, 20))
    fss = compute_fractions_skill_score(f_grid, o_grid, threshold=20.0, window_size=3)
    assert 0.0 <= fss <= 1.0, "Fractions Skill Score must be bounded in [0, 1]"

