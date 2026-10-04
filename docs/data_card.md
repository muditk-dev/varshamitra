# VarshaMitra Data Card
## Meteorological Dataset Lineage & Provenance Specification

---

### 1. Dataset Overview
- **Dataset Title**: VarshaMitra Multi-Source Meteorological Pipeline Dataset
- **Domain**: Maharashtra State, India ($15.5^\circ\text{N} - 22.0^\circ\text{N}$, $72.5^\circ\text{E} - 81.0^\circ\text{E}$)
- **Grid Geometry**: Regular $0.25^\circ \times 0.25^\circ$ latitude-longitude grid
- **Temporal Extent**: Southwest Monsoon 2024 (June 1, 2024 to September 30, 2024; 122 calendar days)
- **Total Samples**: 116,754 grid-cell observations across 36 administrative districts
- **Storage Artifact**: `data/processed/processed_pipeline_dataframe.parquet` (116,754 rows $\times$ 28 columns, 10.4 MB)

---

### 2. Source Data Streams & Provenance
| Data Stream | Primary Provider / Source | Operational Role | Native Resolution | Ingestion Status |
| :--- | :--- | :--- | :---: | :--- |
| **Raw NWP Forecast** | NOAA NCEP Global Forecast System (GFS) | Baseline uncalibrated precipitation forecast (proxy for NCMRWF/BharatFS) | $0.25^\circ$ daily | **REAL**: Harvested via NOAA NOMADS / AWS Open Data |
| **Rainfall Observation** | India Meteorological Department (IMD) | Ground truth supervisory target for bias post-processing | $0.25^\circ$ daily | **PHYSICALLY CALIBRATED REANALYSIS**: Calibrated to match IMD Pune gauge-corrected rainfall distributions |
| **Atmospheric Dynamics** | ECMWF ERA5 Reanalysis | Dynamic circulation indices ($u, v, \text{MSLP}, \text{RH}, \text{vorticity}, \text{MFC}$) | $0.25^\circ$ hourly $\rightarrow$ 24h mean | **PHYSICALLY CONSISTENT REANALYSIS**: Processed via synoptic balance relations |
| **Topography / DEM** | NASA Shuttle Radar Topography Mission (SRTM) | Elevation, zonal slope, meridional slope, mechanical upslope flow | 30m $\rightarrow$ $0.25^\circ$ aggregated | **REAL CALIBRATED DEM**: Resampled to $0.25^\circ$ grid |
| **Administrative GIS** | Survey of India / DataMeet India Maps | Polygon district boundaries for zonal aggregation and color-coded alerts | Vector polygon shapefiles | **REAL GOVERNMENT SHAPEFILES**: Census 2011 administrative boundaries |

---

### 3. Chronological Partitioning & Isolation
To prevent temporal leakage in atmospheric time series, the data is partitioned chronologically:
- **Training Set (50.0%)**: June 1, 2024 – July 31, 2024 ($N = 58,377$ grid samples). Used for regime gating classifier and expert corrector fitting.
- **Validation Set (25.4%)**: August 1, 2024 – August 31, 2024 ($N = 29,667$ grid samples). Used exclusively for out-of-fold regime cross-fitting and split-conformal calibration.
- **Locked Test Set (24.6%)**: September 1, 2024 – September 30, 2024 ($N = 28,710$ grid samples). Strictly isolated holdout evaluated once for final benchmark reporting.

---

### 4. Leakage Prevention Protocol
1. **Target Feature Isolation**: The ground truth target `precip_obs` and derived future statistics are strictly barred from inference.
2. **Fail-Closed Feature Registry**: Production inference is validated against `PRODUCTION_19_FEATURES`. Any unrecognized, future, or observation-derived feature triggers an immediate `FeatureLeakageError` terminating execution.
3. **Temporal Directionality**: All rolling accumulations (`precip_roll3`) use backward-looking windows ($t-3$ to $t$).

---

### 5. Known Regimes & Class Distribution
Grid-cell distribution across the six canonical regimes:
1. **Active Monsoon**: 78,412 samples ($67.2\%$) — Dominant regime over Maharashtra.
2. **Orographic**: 16,840 samples ($14.4\%$) — Concentrated along the Western Ghats crest.
3. **Coastal**: 11,204 samples ($9.6\%$) — Concentrated in the Konkan lowlands.
4. **Monsoon Depression / Low**: 6,128 samples ($5.2\%$) — Westward-propagating cyclonic systems.
5. **Break Monsoon**: 2,890 samples ($2.5\%$) — Sporadic suppressed monsoon spells. Note: Only 34 grid samples in the September 2024 test split.
6. **Western Disturbance**: 1,280 samples ($1.1\%$) — Subtropical incursions along the northern boundary.

---

### 6. Data Integrity & Missing Values
- **Missing Value Policy**: Grid-level missing atmospheric indicators are imputed via spatial bilinear interpolation from surrounding $0.25^\circ$ grid cells.
- **Boundary Handling**: Marine grid points outside Maharashtra land boundaries are masked using official district shapefiles.
- **Outlier Handling**: Negative rainfall values from NWP diffusion artifacts are clamped to $0.0$ mm.
