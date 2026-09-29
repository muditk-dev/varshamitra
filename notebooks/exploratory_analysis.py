"""Exploratory Data Analysis and Verification Plotter for VarshaMitra.
==================================================================
Produces multi-panel inspection plots of all 5 data streams for Maharashtra:
1. Topography DEM (Western Ghats & Deccan Plateau)
2. Raw NWP Forecast (NOAA GFS substitute for NCMRWF)
3. Observed Daily Rainfall (IMD 0.25° format)
4. Atmospheric Variables: MSLP & 850 hPa Low-Level Jet
5. Atmospheric Variables: 200 hPa Tropical Easterly Jet & RH
6. Maharashtra Administrative Districts Boundaries & Zonal Context
"""

import sys
import os
from pathlib import Path

# Ensure Windows conda environment native DLLs are loaded properly
conda_dll_dir = Path(sys.executable).parent / "Library" / "bin"
if conda_dll_dir.exists():
    try:
        os.add_dll_directory(str(conda_dll_dir))
    except Exception:
        pass
    os.environ["PATH"] = str(conda_dll_dir) + os.pathsep + os.environ.get("PATH", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_ingestion import ingest_all_pilot_data


def generate_exploratory_plots(output_image: str = "exploratory_data_preview.png"):
    """Run ingestion on a representative monsoon date range and render inspection panel."""
    print("Running data ingestion for exploratory analysis...")
    data = ingest_all_pilot_data(start_date="2024-07-15", end_date="2024-07-20", data_dir="data")
    
    dem_ds = data["dem"]
    forecast_ds = data["forecast"]
    observed_ds = data["observed"]
    atmos_ds = data["atmospheric"]
    districts_gdf = data["districts"]
    
    lats = dem_ds["lat"].values
    lons = dem_ds["lon"].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Pick mid-spell monsoon date (July 17, 2024)
    t_idx = 2
    sample_date = str(forecast_ds["time"].values[t_idx])[:10]
    print(f"Plotting multi-stream meteorological inspection for date: {sample_date}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=150)
    fig.suptitle(f"VarshaMitra Data Ingestion Inspection: Maharashtra Pilot Region ({sample_date})", fontsize=16, fontweight="bold", y=0.98)
    
    # Panel 1: Topography DEM
    ax1 = axes[0, 0]
    elev = dem_ds["elevation"].values
    im1 = ax1.contourf(lon_grid, lat_grid, elev, levels=25, cmap="terrain")
    plt.colorbar(im1, ax=ax1, label="Elevation (m MSL)", fraction=0.046, pad=0.04)
    ax1.set_title("1. Topography DEM (Western Ghats Ridge)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Longitude (°E)")
    ax1.set_ylabel("Latitude (°N)")
    ax1.axvline(73.65, color="red", linestyle="--", alpha=0.7, label="Ghats Crest (~73.65°E)")
    ax1.legend(loc="upper right", fontsize=8)
    
    # Panel 2: Raw Forecast (GFS substitute)
    ax2 = axes[0, 1]
    raw_rain = forecast_ds["precip_raw"].values[t_idx]
    im2 = ax2.contourf(lon_grid, lat_grid, raw_rain, levels=np.linspace(0, 150, 16), cmap="YlGnBu", extend="max")
    plt.colorbar(im2, ax=ax2, label="Precipitation (mm/day)", fraction=0.046, pad=0.04)
    ax2.set_title(f"2. Raw Forecast (NOAA GFS Sub)\nMax: {np.max(raw_rain):.1f} mm | Mean: {np.mean(raw_rain):.1f} mm", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Longitude (°E)")
    ax2.set_ylabel("Latitude (°N)")
    
    # Panel 3: Observed Rainfall (IMD ground truth format)
    ax3 = axes[0, 2]
    obs_rain = observed_ds["precip_obs"].values[t_idx]
    im3 = ax3.contourf(lon_grid, lat_grid, obs_rain, levels=np.linspace(0, 150, 16), cmap="YlGnBu", extend="max")
    plt.colorbar(im3, ax=ax3, label="Precipitation (mm/day)", fraction=0.046, pad=0.04)
    ax3.set_title(f"3. Observed Rainfall (IMD Format)\nMax: {np.max(obs_rain):.1f} mm | Mean: {np.mean(obs_rain):.1f} mm", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Longitude (°E)")
    ax3.set_ylabel("Latitude (°N)")
    
    # Panel 4: Atmospheric - MSLP & 850 hPa Wind
    ax4 = axes[1, 0]
    mslp = atmos_ds["mslp"].values[t_idx]
    u850 = atmos_ds["u850"].values[t_idx]
    v850 = atmos_ds["v850"].values[t_idx]
    wind_speed_850 = np.sqrt(u850**2 + v850**2)
    im4 = ax4.contourf(lon_grid, lat_grid, mslp, levels=15, cmap="coolwarm")
    plt.colorbar(im4, ax=ax4, label="MSLP (hPa)", fraction=0.046, pad=0.04)
    step = 2
    ax4.quiver(lon_grid[::step, ::step], lat_grid[::step, ::step],
               u850[::step, ::step], v850[::step, ::step],
               scale=250, color="black", width=0.004)
    ax4.set_title("4. Synoptic MSLP & 850 hPa Low-Level Jet", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Longitude (°E)")
    ax4.set_ylabel("Latitude (°N)")
    
    # Panel 5: Atmospheric - Relative Humidity & 200 hPa Wind
    ax5 = axes[1, 1]
    rh850 = atmos_ds["rh850"].values[t_idx]
    u200 = atmos_ds["u200"].values[t_idx]
    v200 = atmos_ds["v200"].values[t_idx]
    im5 = ax5.contourf(lon_grid, lat_grid, rh850, levels=np.linspace(50, 100, 11), cmap="Blues")
    plt.colorbar(im5, ax=ax5, label="850 hPa Relative Humidity (%)", fraction=0.046, pad=0.04)
    ax5.quiver(lon_grid[::step, ::step], lat_grid[::step, ::step],
               u200[::step, ::step], v200[::step, ::step],
               scale=450, color="purple", width=0.004)
    ax5.set_title("5. 850 hPa Humidity & 200 hPa Easterly Jet", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Longitude (°E)")
    ax5.set_ylabel("Latitude (°N)")
    
    # Panel 6: District Boundaries Overlaid on Observation
    ax6 = axes[1, 2]
    ax6.contourf(lon_grid, lat_grid, obs_rain, levels=np.linspace(0, 150, 16), cmap="YlGnBu", alpha=0.6)
    for geom in districts_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            coords = np.array(geom.exterior.coords)
            ax6.plot(coords[:, 0], coords[:, 1], color="darkred", linewidth=0.8, alpha=0.8)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                coords = np.array(poly.exterior.coords)
                ax6.plot(coords[:, 0], coords[:, 1], color="darkred", linewidth=0.8, alpha=0.8)
    
    # Annotate key district centers
    key_districts = ["Mumbai City", "Pune", "Nagpur", "Nashik", "Ratnagiri", "Solapur"]
    for _, row in districts_gdf.iterrows():
        if row["district"] in key_districts:
            centroid = row.geometry.centroid
            ax6.plot(centroid.x, centroid.y, "ro", markersize=3)
            ax6.text(centroid.x + 0.08, centroid.y + 0.08, row["district"], fontsize=7, fontweight="bold", color="darkred")
            
    ax6.set_title(f"6. Maharashtra Districts Overlay ({len(districts_gdf)} Districts)", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Longitude (°E)")
    ax6.set_ylabel("Latitude (°N)")
    ax6.set_xlim(72.5, 80.5)
    ax6.set_ylim(15.5, 22.5)
    
    out_file = Path(output_image)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(str(out_file), bbox_inches="tight")
    plt.close("all")
    print(f"Exploratory inspection figure saved to: {out_file.resolve()}")
    return str(out_file.resolve())


if __name__ == "__main__":
    try:
        out = generate_exploratory_plots("data/exploratory_data_preview.png")
        print("Exploratory script finished successfully:", out)
    except Exception as e:
        import traceback
        print("ERROR occurred in exploratory_analysis.py:")
        traceback.print_exc()
        sys.exit(1)
