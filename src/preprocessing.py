"""VarshaMitra Preprocessing Module.
==================================
Aligns spatial grids, harmonizes temporal accumulation intervals, and merges
all disparate meteorological data streams into a unified, analysis-ready xarray Dataset.

Standard Spatial Grid:
- Region: Maharashtra Pilot Bounding Box (15.5°N - 22.5°N, 72.5°E - 80.5°E)
- Resolution: 0.25° x 0.25° (~27.5 km grid cell)
- Bounding Box Dimensions: 29 latitude steps x 33 longitude steps = 957 grid cells
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr

import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("varshamitra.preprocessing")


def align_and_regrid_dataset(
    ds: xr.Dataset,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    method: str = "nearest"
) -> xr.Dataset:
    """Interpolate / regrid a spatial dataset to the uniform 0.25° Maharashtra grid."""
    # Ensure coordinate names are standard
    lat_col = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
    lon_col = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
    
    if lat_col != "lat" or lon_col != "lon":
        ds = ds.rename({lat_col: "lat", lon_col: "lon"})
        
    # Reindex or interpolate
    regridded = ds.interp(lat=target_lats, lon=target_lons, method=method)
    return regridded


def merge_meteorological_streams(
    forecast_ds: xr.Dataset,
    observed_ds: xr.Dataset,
    atmos_ds: xr.Dataset,
    dem_ds: xr.Dataset,
    target_lats: Optional[np.ndarray] = None,
    target_lons: Optional[np.ndarray] = None
) -> xr.Dataset:
    """Combine forecast, observations, synoptic atmospheric variables, and topography
    into a single cohesive xarray Dataset with consistent coordinate indices.
    """
    logger.info("Aligning and merging all meteorological data streams...")
    
    if target_lats is None or target_lons is None:
        target_lats = forecast_ds["lat"].values
        target_lons = forecast_ds["lon"].values
        
    # Standardize spatial coordinates across all streams
    dem_aligned = align_and_regrid_dataset(dem_ds, target_lats, target_lons)
    obs_aligned = align_and_regrid_dataset(observed_ds, target_lats, target_lons)
    atmos_aligned = align_and_regrid_dataset(atmos_ds, target_lats, target_lons)
    fcst_aligned = align_and_regrid_dataset(forecast_ds, target_lats, target_lons)
    
    # Broadcast static DEM to the temporal dimension
    time_coords = fcst_aligned["time"].values
    n_times = len(time_coords)
    
    elev_2d = dem_aligned["elevation"].values
    slope_x_2d = dem_aligned["slope_lon"].values
    slope_y_2d = dem_aligned["slope_lat"].values
    
    elev_3d = np.repeat(elev_2d[np.newaxis, :, :], n_times, axis=0)
    slope_x_3d = np.repeat(slope_x_2d[np.newaxis, :, :], n_times, axis=0)
    slope_y_3d = np.repeat(slope_y_2d[np.newaxis, :, :], n_times, axis=0)
    
    merged = xr.Dataset(
        data_vars={
            "precip_raw": fcst_aligned["precip_raw"],
            "precip_obs": obs_aligned["precip_obs"],
            "u850": atmos_aligned["u850"],
            "v850": atmos_aligned["v850"],
            "u200": atmos_aligned["u200"],
            "v200": atmos_aligned["v200"],
            "mslp": atmos_aligned["mslp"],
            "rh850": atmos_aligned["rh850"],
            "elevation": (["time", "lat", "lon"], elev_3d.astype(np.float32), {"units": "m"}),
            "slope_lon": (["time", "lat", "lon"], slope_x_3d.astype(np.float32), {"units": "m/m"}),
            "slope_lat": (["time", "lat", "lon"], slope_y_3d.astype(np.float32), {"units": "m/m"}),
        },
        coords={
            "time": time_coords,
            "lat": target_lats,
            "lon": target_lons
        },
        attrs={
            "title": "VarshaMitra Unified Analysis-Ready Dataset",
            "region": "Maharashtra Pilot Region (15.5°N - 22.5°N, 72.5°E - 80.5°E)",
            "grid_resolution": "0.25 degree"
        }
    )
    
    # Ensure no NaN values in active domain
    for var_name in merged.data_vars:
        nan_count = int(np.isnan(merged[var_name].values).sum())
        if nan_count > 0:
            logger.warning(f"Filling {nan_count} NaNs in {var_name} with forward/backward fill")
            merged[var_name] = merged[var_name].bfill(dim="lon").ffill(dim="lon").fillna(0.0)
            
    logger.info(f"Successfully merged dataset. Dimensions: {dict(merged.dims)}")
    return merged


def process_and_save_pipeline_dataset(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    start_date: str = "2024-06-01",
    end_date: str = "2024-09-30"
) -> xr.Dataset:
    """Load ingested raw NetCDFs, preprocess, and save analysis-ready processed NetCDF."""
    r_path = Path(raw_dir)
    p_path = Path(processed_dir)
    p_path.mkdir(parents=True, exist_ok=True)
    
    dem_path = r_path / "topography_dem_maharashtra.nc"
    fcst_path = r_path / f"raw_forecast_gfs_{start_date}_{end_date}.nc"
    obs_path = r_path / f"observed_rainfall_imd_{start_date}_{end_date}.nc"
    atmos_path = r_path / f"atmospheric_era5_{start_date}_{end_date}.nc"
    
    dem_ds = xr.open_dataset(dem_path)
    fcst_ds = xr.open_dataset(fcst_path)
    obs_ds = xr.open_dataset(obs_path)
    atmos_ds = xr.open_dataset(atmos_path)
    
    unified_ds = merge_meteorological_streams(fcst_ds, obs_ds, atmos_ds, dem_ds)
    
    out_file = p_path / f"unified_preprocessed_{start_date}_{end_date}.nc"
    unified_ds.to_netcdf(out_file)
    logger.info(f"Saved preprocessed unified dataset to {out_file}")
    return unified_ds


if __name__ == "__main__":
    ds = process_and_save_pipeline_dataset()
    print("Preprocessed dataset summary:")
    print(ds)
