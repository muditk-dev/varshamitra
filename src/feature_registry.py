"""VarshaMitra Feature Registry & Strict Leakage Guard Module.
============================================================
Establishes the single authoritative source of truth for all meteorological
features in the VarshaMitra regime-aware AI pipeline (SIH 26080).

Enforces strict separation between:
1. FORECAST-SIDE INFORMATION (genuinely available at forecast issuance)
2. OBSERVATION / TARGET-SIDE INFORMATION (available only post-facto)

FAILS CLOSED:
- Unknown/unregistered features -> Rejected (FeatureLeakageError)
- Target-dependent features     -> Rejected (FeatureLeakageError)
- Future-observation features    -> Rejected (FeatureLeakageError)
- Observation-derived features   -> Rejected (FeatureLeakageError)
"""

import logging
from typing import Dict, List, Any, Optional, Set
import sys

logger = logging.getLogger("varshamitra.leakage_guard")


class FeatureLeakageError(ValueError):
    """Raised when a feature violates forecast-time availability, target independence,
    or registry authorization constraints.
    """
    pass


# -----------------------------------------------------------------------------
# 1. CANONICAL FEATURE REGISTRY
# -----------------------------------------------------------------------------
FEATURE_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- NWP Raw Precipitation Forecast ---
    "precip_raw": {
        "name": "precip_raw",
        "source": "NOAA GFS (0.25 deg)",
        "description": "Raw uncalibrated NWP 24h accumulated total precipitation forecast",
        "calculation_method": "Direct accumulation from GFS prate / apcp forecast fields",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "nwp_forecast",
        "units": "mm/24h"
    },

    # --- Topographic Static Features ---
    "elevation": {
        "name": "elevation",
        "source": "SRTM Topography DEM",
        "description": "Surface elevation above mean sea level",
        "calculation_method": "Static spatial DEM regridded to 0.25 deg Maharashtra domain",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "topographic_static",
        "units": "meters"
    },
    "slope_lon": {
        "name": "slope_lon",
        "source": "SRTM Topography DEM",
        "description": "Zonal topographic elevation gradient (dZ/dx)",
        "calculation_method": "Finite difference of DEM along longitude grid spacing (0.25 deg)",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "topographic_static",
        "units": "m/m"
    },
    "slope_lat": {
        "name": "slope_lat",
        "source": "SRTM Topography DEM",
        "description": "Meridional topographic elevation gradient (dZ/dy)",
        "calculation_method": "Finite difference of DEM along latitude grid spacing (0.25 deg)",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "topographic_static",
        "units": "m/m"
    },
    "coastal_proximity": {
        "name": "coastal_proximity",
        "source": "Geographical Static / Coastline Definition",
        "description": "Exponential proximity metric to Arabian Sea coastline (72.8 deg E)",
        "calculation_method": "exp(-max(0, (lon - 72.8) / 1.5)) decaying inland eastward",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "topographic_static",
        "units": "dimensionless (0 to 1)"
    },

    # --- Synoptic Atmospheric Variables (Forecast Side) ---
    "u850": {
        "name": "u850",
        "source": "ECMWF ERA5 / NOAA GFS",
        "description": "Zonal wind velocity at 850 hPa pressure level",
        "calculation_method": "Low-level monsoon westerly jet component from atmospheric model",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "synoptic_atmospheric",
        "units": "m/s"
    },
    "v850": {
        "name": "v850",
        "source": "ECMWF ERA5 / NOAA GFS",
        "description": "Meridional wind velocity at 850 hPa pressure level",
        "calculation_method": "Low-level monsoon southerly flow component from atmospheric model",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "synoptic_atmospheric",
        "units": "m/s"
    },
    "u200": {
        "name": "u200",
        "source": "ECMWF ERA5 / NOAA GFS",
        "description": "Zonal wind velocity at 200 hPa upper troposphere",
        "calculation_method": "Upper-level Tropical Easterly Jet / Westerly flow component",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "synoptic_atmospheric",
        "units": "m/s"
    },
    "v200": {
        "name": "v200",
        "source": "ECMWF ERA5 / NOAA GFS",
        "description": "Meridional wind velocity at 200 hPa upper troposphere",
        "calculation_method": "Upper-level meridional wind from atmospheric model",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "synoptic_atmospheric",
        "units": "m/s"
    },
    "mslp": {
        "name": "mslp",
        "source": "ECMWF ERA5 / NOAA GFS",
        "description": "Mean Sea Level Pressure",
        "calculation_method": "Barometric surface pressure reduced to MSL from atmospheric model",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "synoptic_atmospheric",
        "units": "hPa"
    },
    "rh850": {
        "name": "rh850",
        "source": "ECMWF ERA5 / NOAA GFS",
        "description": "Relative humidity at 850 hPa pressure level",
        "calculation_method": "Low-level saturation percentage from atmospheric model",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "synoptic_atmospheric",
        "units": "%"
    },

    # --- Physics-Derived Diagnostic Features (Forecast-Side Only) ---
    "wind_shear": {
        "name": "wind_shear",
        "source": "Derived (Atmospheric Dynamics)",
        "description": "Bulk vertical wind shear between 200 hPa and 850 hPa",
        "calculation_method": "sqrt((u200 - u850)^2 + (v200 - v850)^2)",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "physics_diagnostic",
        "units": "m/s"
    },
    "wind_speed_850": {
        "name": "wind_speed_850",
        "source": "Derived (Atmospheric Dynamics)",
        "description": "Horizontal wind speed magnitude at 850 hPa low-level jet",
        "calculation_method": "sqrt(u850^2 + v850^2)",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "physics_diagnostic",
        "units": "m/s"
    },
    "upslope_flow": {
        "name": "upslope_flow",
        "source": "Derived (Orographic Dynamics)",
        "description": "Topographic orographic upslope wind velocity component V . grad(Z)",
        "calculation_method": "u850 * slope_lon + v850 * slope_lat",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "physics_diagnostic",
        "units": "m/s"
    },
    "vorticity": {
        "name": "vorticity",
        "source": "Derived (Atmospheric Dynamics)",
        "description": "Relative cyclonic vorticity at 850 hPa",
        "calculation_method": "(dv850/dx - du850/dy) * 1e5 using 2D finite difference",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "physics_diagnostic",
        "units": "1e-5 s^-1"
    },
    "mslp_anomaly": {
        "name": "mslp_anomaly",
        "source": "Derived (Synoptic Climatology)",
        "description": "Mean Sea Level Pressure deviation from regional daily spatial mean",
        "calculation_method": "mslp - spatial_mean(mslp, domain)",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "physics_diagnostic",
        "units": "hPa"
    },
    "q_850": {
        "name": "q_850",
        "source": "Derived (Atmospheric Thermodynamics)",
        "description": "Specific humidity proxy at 850 hPa",
        "calculation_method": "Tetens formulation: (rh850/100) * (0.622 * e_sat / (850 - 0.378 * e_sat)) * 1000",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "physics_diagnostic",
        "units": "g/kg"
    },
    "mfc": {
        "name": "mfc",
        "source": "Derived (Atmospheric Thermodynamics & Kinematics)",
        "description": "Horizontal moisture flux convergence at 850 hPa",
        "calculation_method": "-div(q * V_850) = -(d(q*u)/dx + d(q*v)/dy) * 1e3",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "physics_diagnostic",
        "units": "10^-3 g/(kg s)"
    },
    "precip_roll3": {
        "name": "precip_roll3",
        "source": "Derived (Trailing NWP Forecast Accumulation)",
        "description": "3-day backward rolling mean of raw GFS forecast precipitation",
        "calculation_method": "mean(precip_raw[max(0, t-2) : t+1]) - strictly trailing forecast window",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": True,
        "allowed_for_inference": True,
        "category": "physics_diagnostic",
        "units": "mm/day"
    },

    # --- Target & Observation Variables (STRICTLY DISALLOWED AS MODEL INPUT FEATURES) ---
    "precip_obs": {
        "name": "precip_obs",
        "source": "IMD Daily Gridded Rainfall (0.25 deg)",
        "description": "Observed 24h ground-truth precipitation from IMD rain gauge network",
        "calculation_method": "Daily rain gauge interpolation (08:30 IST to 08:30 IST next day)",
        "forecast_time_availability": False,
        "target_dependency": True,
        "leakage_risk": "CRITICAL",
        "allowed_for_training": False,
        "allowed_for_inference": False,
        "category": "observation_target",
        "units": "mm/24h"
    },
    "lead_precip_obs": {
        "name": "lead_precip_obs",
        "source": "IMD Future Observation",
        "description": "Future observed rainfall for subsequent verification days",
        "calculation_method": "Shifted observation precip_obs[t + lead]",
        "forecast_time_availability": False,
        "target_dependency": True,
        "leakage_risk": "CRITICAL",
        "allowed_for_training": False,
        "allowed_for_inference": False,
        "category": "future_observation",
        "units": "mm/24h"
    },
    "future_obs_rain": {
        "name": "future_obs_rain",
        "source": "IMD Future Observation",
        "description": "Hypothetical future observed rainfall",
        "calculation_method": "Observation at t > forecast_issuance_time",
        "forecast_time_availability": False,
        "target_dependency": True,
        "leakage_risk": "CRITICAL",
        "allowed_for_training": False,
        "allowed_for_inference": False,
        "category": "future_observation",
        "units": "mm/24h"
    },
    "target_diff": {
        "name": "target_diff",
        "source": "Target Observation Residual",
        "description": "True forecast error / observed bias (precip_obs - precip_raw)",
        "calculation_method": "precip_obs - precip_raw",
        "forecast_time_availability": False,
        "target_dependency": True,
        "leakage_risk": "CRITICAL",
        "allowed_for_training": False,
        "allowed_for_inference": False,
        "category": "observation_target",
        "units": "mm"
    },

    # --- Coordinate & Metadata Variables ---
    "time": {
        "name": "time",
        "source": "Temporal Coordinate",
        "description": "UTC timestamp of the forecast verification date",
        "calculation_method": "Daily datetime index",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": False,
        "allowed_for_inference": False,
        "category": "coordinate",
        "units": "datetime"
    },
    "lat": {
        "name": "lat",
        "source": "Spatial Coordinate",
        "description": "Latitude coordinate in degrees North (15.5 to 22.5 N)",
        "calculation_method": "Grid cell center latitude",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": False,
        "allowed_for_inference": False,
        "category": "coordinate",
        "units": "deg_N"
    },
    "lon": {
        "name": "lon",
        "source": "Spatial Coordinate",
        "description": "Longitude coordinate in degrees East (72.5 to 80.5 E)",
        "calculation_method": "Grid cell center longitude",
        "forecast_time_availability": True,
        "target_dependency": False,
        "leakage_risk": "NONE",
        "allowed_for_training": False,
        "allowed_for_inference": False,
        "category": "coordinate",
        "units": "deg_E"
    },

    # --- Supervised Classification / Postprocessing Targets ---
    "regime": {
        "name": "regime",
        "source": "Weak Supervision Rule Labeling",
        "description": "Integer synoptic monsoon regime class (0 to 5)",
        "calculation_method": "Domain rule hierarchy evaluated on forecast atmospheric/terrain fields",
        "forecast_time_availability": False,
        "target_dependency": False,
        "leakage_risk": "HIGH_IF_FEATURE",
        "allowed_for_training": False,
        "allowed_for_inference": False,
        "category": "model_target",
        "units": "class_id (0-5)"
    },
    "precip_corr": {
        "name": "precip_corr",
        "source": "VarshaMitra Model Prediction",
        "description": "Calibrated rainfall output from regime-aware postprocessor",
        "calculation_method": "Soft mixture of experts blend across 6 regime regressors",
        "forecast_time_availability": False,
        "target_dependency": False,
        "leakage_risk": "CIRCULAR_DEPENDENCY",
        "allowed_for_training": False,
        "allowed_for_inference": False,
        "category": "model_prediction",
        "units": "mm/24h"
    }
}


# Canonical ordered list of the 19 production diagnostic features
PRODUCTION_19_FEATURES: List[str] = [
    "precip_raw", "elevation", "slope_lon", "slope_lat",
    "u850", "v850", "u200", "v200", "mslp", "rh850",
    "wind_shear", "wind_speed_850", "upslope_flow",
    "vorticity", "mslp_anomaly", "q_850", "mfc",
    "precip_roll3", "coastal_proximity"
]


# -----------------------------------------------------------------------------
# 2. LEAKAGE GUARD VALIDATION FUNCTIONS (FAIL CLOSED)
# -----------------------------------------------------------------------------
def get_feature_metadata(feature_name: str) -> Dict[str, Any]:
    """Retrieve metadata entry for a feature. Raises FeatureLeakageError if unknown."""
    if feature_name not in FEATURE_REGISTRY:
        raise FeatureLeakageError(
            f"Unregistered feature: '{feature_name}' not found in canonical FEATURE_REGISTRY. "
            f"All features must be registered with data lineage and leakage risk analysis."
        )
    return FEATURE_REGISTRY[feature_name]


def validate_features_for_training(feature_names: List[str], strict: bool = True) -> List[str]:
    """Validate that candidate predictor features are strictly eligible for model training.

    Rejection Criteria (Fail Closed):
    1. Unknown / unregistered feature in strict mode.
    2. Target-dependent features (e.g. precip_obs, target_diff).
    3. Future-observation derived features.
    4. Not marked allowed_for_training = True.

    Returns:
        List of validated feature names if all checks pass.
    Raises:
        FeatureLeakageError if any feature fails validation.
    """
    validated = []
    for feat in feature_names:
        if feat not in FEATURE_REGISTRY:
            if strict:
                raise FeatureLeakageError(
                    f"LEAKAGE GUARD REJECTION: Unknown feature '{feat}' is not registered. "
                    f"Failing closed to prevent accidental data leakage."
                )
            else:
                logger.warning(f"Unregistered feature '{feat}' passed with strict=False.")
                continue

        meta = FEATURE_REGISTRY[feat]

        if meta.get("target_dependency", False):
            raise FeatureLeakageError(
                f"CRITICAL LEAKAGE DETECTED: Feature '{feat}' is target-dependent "
                f"(Source: {meta['source']}). Target information must never be used as a predictor!"
            )

        if not meta.get("forecast_time_availability", False):
            raise FeatureLeakageError(
                f"TEMPORAL LEAKAGE DETECTED: Feature '{feat}' is not available at forecast issuance time "
                f"(Category: {meta['category']})."
            )

        if not meta.get("allowed_for_training", False):
            raise FeatureLeakageError(
                f"LEAKAGE GUARD REJECTION: Feature '{feat}' is explicitly disallowed for training "
                f"(Category: {meta['category']}, Risk: {meta['leakage_risk']})."
            )

        validated.append(feat)

    logger.debug(f"Leakage Guard: Successfully validated {len(validated)} features for training.")
    return validated


def validate_features_for_inference(feature_names: List[str], strict: bool = True) -> List[str]:
    """Validate that candidate features are safe for live production inference.

    Rejection Criteria (Fail Closed):
    1. Unknown / unregistered feature in strict mode.
    2. Target-dependent features.
    3. Future observations.
    4. Not marked forecast_time_availability = True.
    5. Not marked allowed_for_inference = True.

    Returns:
        List of validated feature names if all checks pass.
    Raises:
        FeatureLeakageError if any feature fails validation.
    """
    validated = []
    for feat in feature_names:
        if feat not in FEATURE_REGISTRY:
            if strict:
                raise FeatureLeakageError(
                    f"INFERENCE LEAKAGE GUARD REJECTION: Unknown feature '{feat}' not registered. "
                    f"Production inference strictly requires registered, lineage-audited features."
                )
            else:
                logger.warning(f"Unregistered feature '{feat}' passed with strict=False.")
                continue

        meta = FEATURE_REGISTRY[feat]

        if meta.get("target_dependency", False):
            raise FeatureLeakageError(
                f"PRODUCTION INFERENCE LEAKAGE REJECTION: Feature '{feat}' depends on target observations! "
                f"Observations are physically unavailable during live operational forecast issuance."
            )

        if not meta.get("forecast_time_availability", False):
            raise FeatureLeakageError(
                f"PRODUCTION INFERENCE LEAKAGE REJECTION: Feature '{feat}' is not available at forecast time "
                f"(Category: {meta['category']})."
            )

        if not meta.get("allowed_for_inference", False):
            raise FeatureLeakageError(
                f"PRODUCTION INFERENCE REJECTION: Feature '{feat}' is not authorized for inference "
                f"(Category: {meta['category']}, Risk: {meta['leakage_risk']})."
            )

        validated.append(feat)

    return validated


class LeakageGuard:
    """Convenience class providing unified leakage validation and audit methods."""

    @staticmethod
    def audit_features(feature_names: List[str]) -> List[Dict[str, Any]]:
        """Return structured audit report for a list of features."""
        report = []
        for name in feature_names:
            if name in FEATURE_REGISTRY:
                report.append(FEATURE_REGISTRY[name])
            else:
                report.append({
                    "name": name,
                    "source": "UNKNOWN",
                    "description": "Unregistered feature",
                    "calculation_method": "Unknown",
                    "forecast_time_availability": False,
                    "target_dependency": True,  # Assume worst case
                    "leakage_risk": "CRITICAL_UNREGISTERED",
                    "allowed_for_training": False,
                    "allowed_for_inference": False,
                    "category": "unregistered"
                })
        return report

    @staticmethod
    def verify_production_19() -> bool:
        """Verify that all 19 canonical diagnostic features satisfy both training and inference safety."""
        validate_features_for_training(PRODUCTION_19_FEATURES, strict=True)
        validate_features_for_inference(PRODUCTION_19_FEATURES, strict=True)
        return True
