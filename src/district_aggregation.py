"""VarshaMitra District Zonal Aggregation Module.
==============================================
Performs spatial overlay and zonal aggregation mapping grid-level forecasts
and exceedance probabilities to Maharashtra administrative district polygons.

Computes:
- Mean and 90th percentile raw vs. corrected forecasts
- Dominant active regime per district
- District-level exceedance probabilities
- Operational IMD Color Alert categorization (Green, Yellow, Orange, Red)
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("varshamitra.district_aggregation")


def map_alert_level(mean_rain: float, p_heavy: float, p_vheavy: float, p_eheavy: float) -> Tuple[str, str]:
    """Map district metrics to IMD 4-tier alert system:
    - Green (No warning): light/moderate rain, low heavy probability
    - Yellow (Watch / Be Updated): moderate rain or rising heavy probability
    - Orange (Alert / Be Prepared): heavy rain likely or very heavy risk
    - Red (Warning / Take Action): very heavy to extremely heavy rain imminent
    """
    if mean_rain >= 115.5 or p_vheavy >= 0.50 or p_eheavy >= 0.25:
        return "Red", "#d62728"
    elif mean_rain >= 64.5 or p_heavy >= 0.50 or p_vheavy >= 0.30:
        return "Orange", "#ff7f0e"
    elif mean_rain >= 15.6 or p_heavy >= 0.20:
        return "Yellow", "#bcbd22"
    else:
        return "Green", "#2ca02c"


def aggregate_grid_to_districts(
    grid_df: pd.DataFrame,
    districts_gdf: gpd.GeoDataFrame,
    date_str: Optional[str] = None
) -> gpd.GeoDataFrame:
    """Perform point-in-polygon spatial join from grid cells to Maharashtra districts."""
    if date_str is not None and "time" in grid_df.columns:
        grid_df = grid_df[grid_df["time"].astype(str).str.startswith(date_str)].copy()
        
    # Create GeoDataFrame of grid cell centers
    geometry = [Point(xy) for xy in zip(grid_df["lon"], grid_df["lat"])]
    grid_gdf = gpd.GeoDataFrame(grid_df, geometry=geometry, crs="EPSG:4326")
    
    # Spatial join
    joined = gpd.sjoin(grid_gdf, districts_gdf, how="inner", predicate="intersects")
    
    # Group by district
    records = []
    for district_name, group in joined.groupby("district"):
        raw_mean = float(group["precip_raw"].mean())
        corr_mean = float(group["precip_corr"].mean()) if "precip_corr" in group.columns else raw_mean
        corr_p90 = float(np.percentile(group["precip_corr"], 90)) if "precip_corr" in group.columns else raw_mean * 1.2
        corr_max = float(group["precip_corr"].max()) if "precip_corr" in group.columns else raw_mean * 1.5
        
        # Dominant regime
        dominant_regime = int(group["regime"].mode()[0]) if "regime" in group.columns else 0
        
        # Probabilities
        p_heavy = float(group["p_heavy"].mean()) if "p_heavy" in group.columns else 0.1
        p_vheavy = float(group["p_very_heavy"].mean()) if "p_very_heavy" in group.columns else 0.05
        p_eheavy = float(group["p_extremely_heavy"].mean()) if "p_extremely_heavy" in group.columns else 0.01
        
        alert_name, alert_color = map_alert_level(corr_mean, p_heavy, p_vheavy, p_eheavy)
        
        records.append({
            "district": district_name,
            "raw_mean": round(raw_mean, 1),
            "corr_mean": round(corr_mean, 1),
            "corr_p90": round(corr_p90, 1),
            "corr_max": round(corr_max, 1),
            "dominant_regime": dominant_regime,
            "p_heavy": round(p_heavy, 3),
            "p_very_heavy": round(p_vheavy, 3),
            "p_extremely_heavy": round(p_eheavy, 3),
            "alert_level": alert_name,
            "alert_color": alert_color,
            "cell_count": len(group)
        })
        
    res_df = pd.DataFrame(records)
    merged_gdf = districts_gdf.merge(res_df, on="district", how="left")
    # Fill any unassigned districts with low default
    merged_gdf["alert_level"] = merged_gdf["alert_level"].fillna("Green")
    merged_gdf["alert_color"] = merged_gdf["alert_color"].fillna("#2ca02c")
    merged_gdf["corr_mean"] = merged_gdf["corr_mean"].fillna(5.0)
    merged_gdf["raw_mean"] = merged_gdf["raw_mean"].fillna(5.0)
    merged_gdf["dominant_regime"] = merged_gdf["dominant_regime"].fillna(0).astype(int)
    
    return merged_gdf
