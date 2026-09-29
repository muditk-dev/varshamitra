"""VarshaMitra Physics-Based Feature Engineering Module.
=====================================================
Computes physically meaningful dynamical and thermodynamical diagnostic features
governing Indian Summer Monsoon rainfall regimes:

1. Vertical Wind Shear (200 hPa - 850 hPa)
   - Magnitude: ||V_200 - V_850|| (m/s)
   - Diagnostic for deep convection organization and tropical depression maintenance.
2. Moisture Flux Convergence (MFC)
   - MFC = -div(q * V_850) (g/kg / s)
   - Directly indicates areas of dynamic moisture pooling prior to heavy rain bursts.
3. Relative Vorticity and MSLP Pressure Anomaly
   - Vorticity: zeta = dv/dx - du/dy (10^-5 s^-1)
   - MSLP Anomaly: Delta_MSLP = MSLP - MSLP_mean
   - Core signature of Monsoon Low Pressure Systems (LPS / Depressions).
4. Orographic Upslope Moisture Flux
   - Upslope velocity: W_orog = V_850 . grad(z) = u*dz/dx + v*dz/dy (m/s)
   - Distinguishes windward orographic lifting along Western Ghats from rain shadow.
5. Column Moisture Proxy
   - Specific humidity estimated from RH850.
6. Antecedent Forecast Accumulation
   - 3-day rolling accumulation capturing wet spell persistence.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("varshamitra.features")


def compute_physical_features(ds: xr.Dataset) -> xr.Dataset:
    """Compute complete suite of meteorological features from unified dataset."""
    logger.info("Computing physics-based meteorological features...")
    
    lats = ds["lat"].values
    lons = ds["lon"].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Grid spacing in meters for derivative computation:
    # 0.25 deg ~ 27.5 km
    dx_2d = 0.25 * 111000.0 * np.cos(np.radians(lat_grid))
    dy = 0.25 * 111000.0
    
    u850 = ds["u850"].values
    v850 = ds["v850"].values
    u200 = ds["u200"].values
    v200 = ds["v200"].values
    mslp = ds["mslp"].values
    rh850 = ds["rh850"].values
    elev = ds["elevation"].values
    slope_x = ds["slope_lon"].values
    slope_y = ds["slope_lat"].values
    precip_raw = ds["precip_raw"].values
    
    n_times, n_lats, n_lons = u850.shape
    
    # 1. Vertical Wind Shear (m/s)
    u_shear = u200 - u850
    v_shear = v200 - v850
    wind_shear = np.sqrt(u_shear**2 + v_shear**2).astype(np.float32)
    
    # 2. 850 hPa Wind Speed (m/s)
    wind_speed_850 = np.sqrt(u850**2 + v850**2).astype(np.float32)
    
    # 3. Orographic Upslope Velocity (m/s)
    # W_orog = u * slope_x + v * slope_y
    upslope_flow = (u850 * slope_x + v850 * slope_y).astype(np.float32)
    
    # 4. Relative Vorticity (10^-5 s^-1)
    # zeta = dv/dx - du/dy
    vorticity = np.zeros_like(u850, dtype=np.float32)
    for t in range(n_times):
        # gradient of v along x (axis 1 of 2D)
        dv_dx = np.gradient(v850[t], axis=1) / dx_2d
        # gradient of u along y (axis 0 of 2D)
        du_dy = np.gradient(u850[t], axis=0) / dy
        vorticity[t] = (dv_dx - du_dy) * 1e5
        
    # 5. MSLP Anomaly (hPa) relative to regional mean
    mslp_domain_mean = np.mean(mslp, axis=(1, 2), keepdims=True)
    mslp_anomaly = (mslp - mslp_domain_mean).astype(np.float32)
    
    # 6. Specific Humidity Proxy (g/kg) at 850 hPa
    # Standard monsoon T850 ~ 18°C = 291.15 K -> Saturation vapor pressure ~ 20.6 hPa
    e_sat = 20.64
    q_850 = (rh850 / 100.0) * (0.622 * e_sat / (850.0 - 0.378 * e_sat)) * 1000.0
    q_850 = q_850.astype(np.float32)
    
    # 7. Moisture Flux Convergence (MFC)
    # MFC = -div(q * V) = - (d(q*u)/dx + d(q*v)/dy)
    mfc = np.zeros_like(u850, dtype=np.float32)
    for t in range(n_times):
        qu = q_850[t] * u850[t]
        qv = q_850[t] * v850[t]
        dqu_dx = np.gradient(qu, axis=1) / dx_2d
        dqv_dy = np.gradient(qv, axis=0) / dy
        mfc[t] = -(dqu_dx + dqv_dy) * 1e3  # scale to g/(kg*s) * 10^3
        
    # 8. Rolling 3-day forecast rain accumulation (persistence indicator)
    precip_roll3 = np.zeros_like(precip_raw, dtype=np.float32)
    for t in range(n_times):
        t_start = max(0, t - 2)
        precip_roll3[t] = np.mean(precip_raw[t_start:t+1], axis=0)
        
    # 9. Coastal Proximity Factor (decaying eastward from coast ~72.8°E)
    lon_dist_coast = np.clip((lon_grid - 72.8) / 1.5, 0.0, 5.0)
    coastal_proximity_2d = np.exp(-lon_dist_coast).astype(np.float32)
    coastal_proximity = np.repeat(coastal_proximity_2d[np.newaxis, :, :], n_times, axis=0)
    
    featured_ds = ds.copy()
    featured_ds["wind_shear"] = (["time", "lat", "lon"], wind_shear, {"units": "m/s", "description": "200-850 hPa bulk vertical wind shear"})
    featured_ds["wind_speed_850"] = (["time", "lat", "lon"], wind_speed_850, {"units": "m/s", "description": "850 hPa horizontal wind speed"})
    featured_ds["upslope_flow"] = (["time", "lat", "lon"], upslope_flow, {"units": "m/s", "description": "Orographic upslope wind component V.grad(z)"})
    featured_ds["vorticity"] = (["time", "lat", "lon"], vorticity, {"units": "10^-5 s^-1", "description": "850 hPa relative vorticity"})
    featured_ds["mslp_anomaly"] = (["time", "lat", "lon"], mslp_anomaly, {"units": "hPa", "description": "MSLP deviation from regional daily mean"})
    featured_ds["q_850"] = (["time", "lat", "lon"], q_850, {"units": "g/kg", "description": "Specific humidity proxy at 850 hPa"})
    featured_ds["mfc"] = (["time", "lat", "lon"], mfc, {"units": "10^-3 g/(kg s)", "description": "Moisture flux convergence"})
    featured_ds["precip_roll3"] = (["time", "lat", "lon"], precip_roll3, {"units": "mm/day", "description": "3-day rolling mean raw forecast"})
    featured_ds["coastal_proximity"] = (["time", "lat", "lon"], coastal_proximity, {"units": "dimensionless", "description": "Proximity to Arabian Sea coastline"})
    
    logger.info("Physics-based feature engineering complete.")
    return featured_ds


def get_feature_matrix_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """Flatten 3D spatio-temporal xarray dataset into tabular DataFrame
    suitable for scikit-learn / XGBoost training.
    """
    feature_vars = [
        "precip_raw", "elevation", "slope_lon", "slope_lat",
        "u850", "v850", "u200", "v200", "mslp", "rh850",
        "wind_shear", "wind_speed_850", "upslope_flow",
        "vorticity", "mslp_anomaly", "q_850", "mfc",
        "precip_roll3", "coastal_proximity"
    ]
    
    df_dict = {}
    
    # Coordinate indexing
    time_arr = ds["time"].values
    lat_arr = ds["lat"].values
    lon_arr = ds["lon"].values
    
    # Broadcast coordinates
    t_idx, lat_idx, lon_idx = np.meshgrid(
        np.arange(len(time_arr)),
        np.arange(len(lat_arr)),
        np.arange(len(lon_arr)),
        indexing="ij"
    )
    
    df_dict["time"] = time_arr[t_idx.ravel()]
    df_dict["lat"] = lat_arr[lat_idx.ravel()]
    df_dict["lon"] = lon_arr[lon_idx.ravel()]
    
    for f in feature_vars:
        if f in ds.data_vars:
            df_dict[f] = ds[f].values.ravel()
            
    if "precip_obs" in ds.data_vars:
        df_dict["precip_obs"] = ds["precip_obs"].values.ravel()
        
    df = pd.DataFrame(df_dict)
    return df


if __name__ == "__main__":
    from src.preprocessing import process_and_save_pipeline_dataset
    ds = process_and_save_pipeline_dataset()
    feat_ds = compute_physical_features(ds)
    df = get_feature_matrix_dataframe(feat_ds)
    print("Feature statistics summary:")
    print(df.describe().T[["mean", "std", "min", "max"]])
