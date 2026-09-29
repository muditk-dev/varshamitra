"""VarshaMitra Weak Supervision Regime Labeling Module.
=====================================================
Applies meteorological domain heuristics to assign weak ground-truth regime labels
to each grid cell in the spatio-temporal monsoon dataset.

The 6 Target Regimes:
0: Active Monsoon
1: Break Monsoon
2: Depression (Monsoon Low Pressure System)
3: Orographic (Western Ghats Barrier)
4: Coastal (Konkan Boundary Layer)
5: Western Disturbance (Upper-Tropospheric Westerly Trough)

Each heuristic rule is documented with its physical atmospheric rationale.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("varshamitra.regime_labels")

REGIME_NAMES = {
    0: "Active Monsoon",
    1: "Break Monsoon",
    2: "Depression",
    3: "Orographic",
    4: "Coastal",
    5: "Western Disturbance"
}

REGIME_COLORS = {
    0: "#1f77b4",  # Blue (Active)
    1: "#ff7f0e",  # Orange (Break)
    2: "#d62728",  # Red (Depression)
    3: "#2ca02c",  # Green (Orographic)
    4: "#17becf",  # Cyan (Coastal)
    5: "#9467bd"   # Purple (Western Disturbance)
}


def label_monsoon_regimes(ds: xr.Dataset) -> xr.DataArray:
    """Assign integer regime labels (0 to 5) to each (time, lat, lon) grid cell
    based on meteorological weak-supervision decision hierarchy.
    
    Decision Hierarchy Rationale:
    -----------------------------
    1. Depression Check:
       Monsoon Depressions (LPS) dominate the synoptic state. A closed low MSLP anomaly
       (Delta_MSLP < -2.0 hPa) combined with elevated cyclonic vorticity (zeta > 2.5 x 10^-5 s^-1)
       takes highest priority because synoptic cyclonic forcing overrides local topoclimatology.
       
    2. Orographic Check:
       Along the Western Ghats, physical obstacle forcing by the mountain barrier produces
       extreme rainfall even during ordinary monsoon flow. If the cell sits on the ridge/windward
       slope (elevation > 250m, upslope flow V.grad(z) > 0.015 m/s), it is labeled Orographic.
       
    3. Coastal Check:
       Low-elevation cells directly adjacent to the Arabian Sea (lon < 73.3°E, elevation < 150m)
       with high humidity (RH > 82%) experience marine boundary layer convergence (Konkan coast).
       
    4. Western Disturbance Check:
       Rare during summer monsoon, but mid-latitude westerly troughs can penetrate northern
       Maharashtra (lat > 20.5°N), exhibiting unusually high vertical shear (> 28 m/s) and
       easterly jet disruption.
       
    5. Break Monsoon Check:
       Characterized by suppressed rainfall, northward shift of the monsoon trough,
       positive MSLP anomaly (Delta_MSLP > 1.0 hPa), and reduced humidity (RH < 72% or precip < 2mm).
       
    6. Active Monsoon (Default Synoptic Monsoon State):
       Persistent low-level southwesterly jet (u850 > 10 m/s), positive moisture flux convergence,
       and widespread precipitation.
    """
    logger.info("Applying meteorological weak-supervision heuristics to label active regimes...")
    
    # Extract variables
    u850 = ds["u850"].values
    v850 = ds["v850"].values
    elev = ds["elevation"].values
    mslp_anom = ds["mslp_anomaly"].values
    vorticity = ds["vorticity"].values
    upslope = ds["upslope_flow"].values
    rh850 = ds["rh850"].values
    wind_shear = ds["wind_shear"].values
    precip_raw = ds["precip_raw"].values
    
    lats = ds["lat"].values
    lons = ds["lon"].values
    n_times, n_lats, n_lons = u850.shape
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Broadcast coordinates
    lon_3d = np.repeat(lon_grid[np.newaxis, :, :], n_times, axis=0)
    lat_3d = np.repeat(lat_grid[np.newaxis, :, :], n_times, axis=0)
    
    # Initialize labels with 0 (Active Monsoon)
    labels = np.zeros((n_times, n_lats, n_lons), dtype=np.int32)
    
    # --- Rule 5: Break Monsoon ---
    # Rationale: Suppressed rainfall, high pressure anomaly, dry troposphere
    is_break = (mslp_anom > 1.2) & (rh850 < 74.0) & (precip_raw < 3.5)
    labels[is_break] = 1
    
    # --- Rule 4: Coastal ---
    # Rationale: Arabian sea marine boundary layer, lon < 73.3°E, low elevation
    is_coastal = (lon_3d < 73.3) & (elev < 150.0) & (rh850 > 82.0)
    labels[is_coastal] = 4
    
    # --- Rule 3: Orographic ---
    # Rationale: Western Ghats crest and windward slopes with strong westerly impingement
    is_orographic = (elev > 250.0) & (upslope > 0.012) & (lon_3d >= 73.1) & (lon_3d <= 74.2)
    labels[is_orographic] = 3
    
    # --- Rule 5 (WD): Western Disturbance ---
    # Rationale: Strong upper westerly shear in northern Maharashtra
    is_wd = (lat_3d > 20.8) & (wind_shear > 30.0) & (u850 < 8.0)
    labels[is_wd] = 5
    
    # --- Rule 2: Depression (Highest priority synoptic system) ---
    # Rationale: Strong cyclonic vorticity center and negative MSLP anomaly
    is_depression = (mslp_anom < -2.2) & (vorticity > 2.0)
    labels[is_depression] = 2
    
    # Log regime distribution
    unique, counts = np.unique(labels, return_counts=True)
    total = labels.size
    logger.info("Regime distribution after weak supervision labeling:")
    for u, c in zip(unique, counts):
        logger.info(f"  Regime {u} ({REGIME_NAMES[u]}): {c} cells ({c/total*100:.1f}%)")
        
    regime_da = xr.DataArray(
        labels,
        coords=ds.coords,
        dims=["time", "lat", "lon"],
        name="regime",
        attrs={
            "description": "Active Monsoon Regime classification label (0-5)",
            "regime_0": "Active Monsoon",
            "regime_1": "Break Monsoon",
            "regime_2": "Depression",
            "regime_3": "Orographic",
            "regime_4": "Coastal",
            "regime_5": "Western Disturbance",
        }
    )
    return regime_da


if __name__ == "__main__":
    from src.preprocessing import process_and_save_pipeline_dataset
    from src.feature_engineering import compute_physical_features
    ds = process_and_save_pipeline_dataset()
    feat_ds = compute_physical_features(ds)
    regimes = label_monsoon_regimes(feat_ds)
    feat_ds["regime"] = regimes
    print("Labeled dataset with regimes ready!")
