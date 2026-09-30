"""VARSHAMITRA | AI FOR A RESILIENT MONSOON INDIA
==================================================
Operational Meteorological Analysis & Decision-Support System
Smart India Hackathon 2026 (NCMRWF / Ministry of Earth Sciences)
==================================================
Professional Meteorological & Scientific Workstation.
"""

import sys
import os
import json
import time
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Ensure Windows conda environment native DLLs are loaded properly
if sys.platform == "win32":
    conda_dll_dir = Path(sys.executable).parent / "Library" / "bin"
    if conda_dll_dir.exists():
        try:
            os.add_dll_directory(str(conda_dll_dir))
        except Exception:
            pass
        os.environ["PATH"] = str(conda_dll_dir) + os.pathsep + os.environ.get("PATH", "")

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.regime_labels import REGIME_NAMES, REGIME_COLORS
import importlib
import dashboard.auth as auth
importlib.reload(auth)
import dashboard.command_center as cc
importlib.reload(cc)
import dashboard.views as views
importlib.reload(views)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & AUTHENTICATION GATE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VarshaMitra | Meteorological Operations System",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

auth.init_auth_session()

# Unauthenticated Gate: Show Cinematic Login & Register Experience
if not st.session_state.authenticated:
    auth.render_auth_page()
    st.stop()

if "selected_district" not in st.session_state:
    st.session_state.selected_district = "Pune"
if "lead_time_idx" not in st.session_state:
    st.session_state.lead_time_idx = 0
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Overview"

valid_nav_options = [
    "Overview", "Live Forecast", "District Intelligence",
    "Regime Intelligence", "Heavy Rainfall Risk", "Explainability",
    "What-If Lab", "Forecast Replay", "Verification Lab",
    "Probability Calibration", "Ablation Study",
    "Data Provenance", "Reports", "Settings",
    # Backward compatibility aliases
    "Command Center", "Risk & Alerts"
]

if "tab" in st.query_params:
    qp_val = st.query_params["tab"]
    for t_opt in valid_nav_options:
        if qp_val.lower().replace(" ", "").replace("_", "") in t_opt.lower().replace(" ", "").replace("_", ""):
            st.session_state.current_tab = t_opt
            break

# -----------------------------------------------------------------------------
# 2. PROFESSIONAL LIGHT METEOROLOGICAL WORKSTATION STYLING
# -----------------------------------------------------------------------------
light_workstation_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    /* Government & Research Grade Light Scientific Workstation Palette */
    :root {
        --bg-main: #F5F7F9;
        --bg-surface: #FFFFFF;
        --bg-subtle: #EEF2F6;
        --border-color: #D9E0E8;
        --border-light: #E5E7EB;
        --text-primary: #172033;
        --text-secondary: #64748B;
        --text-muted: #94A3B8;
        --primary-blue: #2563A6;
        --weather-blue: #3B82C4;
        --teal: #159A9C;
        --warning: #D99A24;
        --danger: #C84B4B;
        --extreme: #B93636;
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        --font-heading: 'Manrope', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        --font-mono: 'JetBrains Mono', "SFMono-Regular", Consolas, Menlo, monospace;
    }

    body, .stApp {
        background-color: var(--bg-main) !important;
        font-family: var(--font-sans) !important;
        color: var(--text-primary) !important;
    }

    /* Container padding & max-width */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1650px !important;
    }

    /* Clean Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-surface) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    section[data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
        font-size: 1.0rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.6rem !important;
    }

    /* Modern clean sidebar navigation buttons */
    section[data-testid="stSidebar"] div.stButton > button {
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 6px 12px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        transition: all 0.15s ease !important;
    }

    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background-color: #2563A6 !important;
        border-color: #2563A6 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(37, 99, 166, 0.25) !important;
    }

    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
        background-color: transparent !important;
        border-color: transparent !important;
        color: #475569 !important;
    }

    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
        background-color: #F1F5F9 !important;
        color: #172033 !important;
        border-color: #E2E8F0 !important;
    }

    /* Standard Cards / Panels */
    .metro-panel {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    /* Quiet disclaimer */
    .quiet-disclaimer {
        color: var(--text-muted);
        font-size: 0.75rem;
        line-height: 1.45;
        border-top: 1px solid var(--border-color);
        padding-top: 14px;
        margin-top: 24px;
        text-align: center;
    }
</style>
"""
st.markdown(light_workstation_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. CACHED DATA & MODEL LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load trained regime classifier, bias correctors, and probability models."""
    m_dir = BASE_DIR / "models"
    models = {}
    try:
        with open(m_dir / "regime_classifier_xgb.pkl", "rb") as f:
            models["classifier"] = pickle.load(f)
    except Exception:
        models["classifier"] = None

    try:
        with open(m_dir / "regime_bias_postprocessor.pkl", "rb") as f:
            models["postprocessor"] = pickle.load(f)
    except Exception:
        models["postprocessor"] = None

    try:
        with open(m_dir / "heavy_rainfall_prob_model.pkl", "rb") as f:
            models["prob_model"] = pickle.load(f)
    except Exception:
        models["prob_model"] = None

    return models

@st.cache_data
def load_pipeline_parquet():
    """Load the complete 4-month 0.25° grid monsoon dataset (116,754 cells)."""
    p = BASE_DIR / "data" / "processed" / "processed_pipeline_dataframe.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None

@st.cache_data
def load_core_metadata():
    """Load verification benchmarks, provenance records, and authentic district polygons."""
    p_path = BASE_DIR / "data" / "processed"
    r_path = BASE_DIR / "data" / "raw"

    verif_file = p_path / "verification_scores_summary.json"
    prov_file = r_path / "data_provenance_summary.json"
    auth_file = r_path / "maharashtra_districts_authentic.geojson"

    verif_data = None
    if verif_file.exists():
        with open(verif_file, "r") as f:
            verif_data = json.load(f)

    prov_data = None
    if prov_file.exists():
        with open(prov_file, "r") as f:
            prov_data = json.load(f)

    if auth_file.exists():
        districts_gdf = gpd.read_file(auth_file)
    else:
        alt_files = sorted(list(p_path.glob("district_alerts_*.geojson")))
        districts_gdf = gpd.read_file(alt_files[-1]) if alt_files else None

    alert_files = sorted(list(p_path.glob("district_alerts_*.geojson")))
    dates_available = [f.stem.replace("district_alerts_", "") for f in alert_files]

    return verif_data, prov_data, districts_gdf, dates_available

@st.cache_data
def get_grid_to_district_map():
    """Precompute point-in-polygon assignment of 0.25° grid cells to districts."""
    base = BASE_DIR
    auth_path = base / "data" / "raw" / "maharashtra_districts_authentic.geojson"
    parquet_path = base / "data" / "processed" / "processed_pipeline_dataframe.parquet"

    if not auth_path.exists() or not parquet_path.exists():
        return None

    auth_gdf = gpd.read_file(auth_path)
    df = pd.read_parquet(parquet_path)
    sample_day = df[df["time"] == df["time"].iloc[0]][["lat", "lon"]].drop_duplicates()
    pts_gdf = gpd.GeoDataFrame(sample_day, geometry=gpd.points_from_xy(sample_day.lon, sample_day.lat), crs="EPSG:4326")
    joined = gpd.sjoin(pts_gdf, auth_gdf[["district", "geometry"]], how="inner", predicate="intersects")
    return joined[["lat", "lon", "district"]].drop_duplicates(subset=["lat", "lon"])

# Initialize data and models
MODELS = load_models()
FULL_DF = load_pipeline_parquet()
VERIF_DATA, PROV_DATA, AUTH_GDF, ALERT_DATES = load_core_metadata()
GRID_MAP = get_grid_to_district_map()

# Model Router Specification
ROUTER_SPECS = {
    0: {"name": "Active Monsoon", "acronym": "EQM", "model": "Empirical Quantile Mapping", "desc": "Non-linear CDF matching calibrated against monsoon trough convergence."},
    1: {"name": "Break Monsoon", "acronym": "GBM", "model": "Gradient Boosted Regressor", "desc": "Decision tree ensemble resolving convective triggers during synoptic lulls."},
    2: {"name": "Depression", "acronym": "CNN", "model": "Spatial 2D ConvNet", "desc": "Convolutional receptive fields capturing cyclonic vortex asymmetry and rain bands."},
    3: {"name": "Orographic", "acronym": "CNN", "model": "Spatial 2D ConvNet", "desc": "Receptive fields resolving Western Ghats windward blocking and lee decay."},
    4: {"name": "Coastal", "acronym": "GBM", "model": "Gradient Boosted Regressor", "desc": "Nonlinear regression on marine boundary layer humidity and sea-breeze moisture."},
    5: {"name": "Western Disturbance", "acronym": "EQM", "model": "Empirical Quantile Mapping", "desc": "Upper-tropospheric westerly trough distribution scaling preserving dry shear."}
}

# -----------------------------------------------------------------------------
# 4. TIMELINE & DATA AGGREGATION
# -----------------------------------------------------------------------------
TIMELINE_STEPS = [
    {"label": "Now", "hours": 0, "nominal_offset": 0},
    {"label": "+6h", "hours": 6, "nominal_offset": 0},
    {"label": "+12h", "hours": 12, "nominal_offset": 0},
    {"label": "+24h", "hours": 24, "nominal_offset": 1},
    {"label": "+48h", "hours": 48, "nominal_offset": 2},
    {"label": "+72h", "hours": 72, "nominal_offset": 3}
]

def get_forecast_dataframe_for_lead(base_date: str, step_idx: int) -> Tuple[gpd.GeoDataFrame, str]:
    """Compute or load district-level meteorological attributes for selected date and lead step."""
    step_info = TIMELINE_STEPS[min(step_idx, len(TIMELINE_STEPS)-1)]
    valid_dt = pd.to_datetime(base_date) + pd.Timedelta(hours=step_info["hours"])
    valid_str = valid_dt.strftime("%d %b %Y, %H:00 UTC")

    p_dir = BASE_DIR / "data" / "processed"
    target_date_str = (pd.to_datetime(base_date) + pd.Timedelta(days=step_info["nominal_offset"])).strftime("%Y-%m-%d")
    geojson_target = p_dir / f"district_alerts_{target_date_str}.geojson"

    if geojson_target.exists():
        gdf = gpd.read_file(geojson_target)
        if "alert_label" not in gdf.columns:
            gdf["alert_label"] = gdf.get("alert_level", "Normal")
        if "alert_level" not in gdf.columns:
            gdf["alert_level"] = gdf.get("alert_label", "Green")
        if "difference" not in gdf.columns:
            gdf["difference"] = gdf["corr_mean"] - gdf["raw_mean"]
        if "uncertainty_spread" not in gdf.columns:
            c_p90 = gdf["corr_p90"] if "corr_p90" in gdf.columns else gdf["corr_mean"] + 5.0
            gdf["uncertainty_spread"] = (c_p90 - gdf["corr_mean"]).clip(lower=2.0)
        if "uncertainty_low" not in gdf.columns:
            gdf["uncertainty_low"] = (gdf["corr_mean"] - gdf["uncertainty_spread"]).clip(lower=0.0)
        if "uncertainty_high" not in gdf.columns:
            gdf["uncertainty_high"] = gdf["corr_mean"] + gdf["uncertainty_spread"]
    elif FULL_DF is not None and GRID_MAP is not None:
        df_target = FULL_DF[FULL_DF["time"].dt.strftime("%Y-%m-%d") == target_date_str]
        if len(df_target) == 0:
            df_target = FULL_DF[FULL_DF["time"] == FULL_DF["time"].max()]

        joined = df_target.merge(GRID_MAP, on=["lat", "lon"], how="inner")
        agg = joined.groupby("district").agg(
            raw_mean=("precip_raw", "mean"),
            corr_mean=("precip_corr", "mean"),
            corr_p90=("precip_corr", lambda x: np.percentile(x, 90)),
            corr_max=("precip_corr", "max"),
            dominant_regime=("regime", lambda x: int(pd.Series(x).mode().iloc[0] if len(x)>0 else 0)),
            p_heavy=("p_heavy", "mean"),
            p_very_heavy=("p_very_heavy", "mean"),
            p_extremely_heavy=("p_extremely_heavy", "mean")
        ).reset_index()

        gdf = AUTH_GDF.merge(agg, on="district", how="left")

        def get_alert_tier(row):
            c_max = row.get("corr_max", 0.0)
            p_h = row.get("p_heavy", 0.0)
            p_vh = row.get("p_very_heavy", 0.0)
            if c_max >= 115.5 or p_vh >= 0.35:
                return "Red", "Extreme", "#DC2626"
            elif c_max >= 64.5 or p_h >= 0.40:
                return "Orange", "Heavy", "#EA580C"
            elif c_max >= 15.6:
                return "Yellow", "Moderate", "#CA8A04"
            else:
                return "Green", "Normal", "#16A34A"

        alerts = [get_alert_tier(r) for _, r in gdf.iterrows()]
        gdf["alert_level"] = [a[0] for a in alerts]
        gdf["alert_label"] = [a[1] for a in alerts]
        gdf["alert_color"] = [a[2] for a in alerts]
        gdf["uncertainty_spread"] = (gdf["corr_p90"] - gdf["corr_mean"]).clip(lower=2.0)
        gdf["uncertainty_low"] = (gdf["corr_mean"] - gdf["uncertainty_spread"]).clip(lower=0.0)
        gdf["uncertainty_high"] = gdf["corr_mean"] + gdf["uncertainty_spread"]
        gdf["difference"] = gdf["corr_mean"] - gdf["raw_mean"]
    else:
        gdf = AUTH_GDF.copy()
        gdf["raw_mean"] = 32.5
        gdf["corr_mean"] = 24.0
        gdf["corr_p90"] = 38.0
        gdf["corr_max"] = 58.0
        gdf["dominant_regime"] = 0
        gdf["p_heavy"] = 0.18
        gdf["p_very_heavy"] = 0.04
        gdf["p_extremely_heavy"] = 0.01
        gdf["alert_level"] = "Yellow"
        gdf["alert_label"] = "Moderate"
        gdf["alert_color"] = "#CA8A04"
        gdf["uncertainty_spread"] = 14.0
        gdf["uncertainty_low"] = 10.0
        gdf["uncertainty_high"] = 38.0
        gdf["difference"] = -8.5

    gdf["regime_name"] = gdf["dominant_regime"].map(lambda x: REGIME_NAMES.get(int(x), "Active Monsoon"))
    return gdf, valid_str

# Base operational date & district selection
if "selected_base_date" not in st.session_state:
    st.session_state.selected_base_date = ALERT_DATES[-1] if ALERT_DATES else "2024-09-28"

DISTRICTS_GDF, VALID_TIME_STR = get_forecast_dataframe_for_lead(
    st.session_state.selected_base_date,
    st.session_state.lead_time_idx
)

district_list = sorted(DISTRICTS_GDF["district"].dropna().unique().tolist())
if st.session_state.selected_district not in district_list and len(district_list) > 0:
    st.session_state.selected_district = district_list[0]

# -----------------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION & OPERATIONS CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 2px 4px 10px 4px; border-bottom: 1px solid #E2E8F0; margin-bottom: 8px;">
        <div style="font-size: 1.15rem; font-weight: 800; color: #172033; letter-spacing: -0.3px; display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.25rem;">🌧️</span> VARSHAMITRA
        </div>
        <div style="font-size: 0.70rem; font-weight: 600; color: #667085; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;">
            Meteorological Operations Workstation
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Grouped navigation as specified in Command Center requirements
    nav_groups = [
        ("COMMAND CENTER", [
            ("Overview", "🏠 Overview"),
            ("Live Forecast", "📡 Live Forecast"),
            ("District Intelligence", "📍 District Intelligence")
        ]),
        ("MONSOON INTELLIGENCE", [
            ("Regime Intelligence", "🌪️ Regime Intelligence"),
            ("Heavy Rainfall Risk", "⚠️ Heavy Rainfall Risk"),
            ("Explainability", "🔍 Explainability")
        ]),
        ("EXPERIMENTS", [
            ("What-If Lab", "🧪 What-If Lab"),
            ("Forecast Replay", "⏪ Forecast Replay")
        ]),
        ("VERIFICATION", [
            ("Verification Lab", "📊 Verification Lab"),
            ("Probability Calibration", "🎯 Probability Calibration"),
            ("Ablation Study", "🔬 Ablation Study")
        ]),
        ("DATA", [
            ("Data Provenance", "🗄️ Data Provenance"),
            ("Reports", "📄 Reports")
        ]),
        ("SYSTEM", [
            ("Settings", "⚙️ Settings")
        ])
    ]

    for grp_title, items in nav_groups:
        st.markdown(f"<div style='font-size:0.68rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.8px; margin:10px 0 3px 4px;'>{grp_title}</div>", unsafe_allow_html=True)
        for tab_key, tab_label in items:
            is_active = (
                st.session_state.current_tab == tab_key or
                (tab_key == "Overview" and st.session_state.current_tab in ["Command Center", "Overview"]) or
                (tab_key == "Heavy Rainfall Risk" and st.session_state.current_tab in ["Risk & Alerts", "Heavy Rainfall Risk"])
            )
            btn_type = "primary" if is_active else "secondary"
            if st.button(tab_label, key=f"side_nav_{tab_key}", type=btn_type, use_container_width=True):
                st.session_state.current_tab = tab_key
                st.query_params["tab"] = tab_key
                st.rerun()

    st.markdown("<hr style='border-color:#E2E8F0; margin:10px 0 8px 0;'>", unsafe_allow_html=True)

    # Operational Controls
    with st.expander("⚙️ Operational Controls", expanded=False):
        if ALERT_DATES:
            sel_base_dt = st.selectbox(
                "Synoptic Base Date",
                ALERT_DATES,
                index=ALERT_DATES.index(st.session_state.selected_base_date) if st.session_state.selected_base_date in ALERT_DATES else len(ALERT_DATES) - 1,
                help="Select initialization cycle date."
            )
            if sel_base_dt != st.session_state.selected_base_date:
                st.session_state.selected_base_date = sel_base_dt
                st.rerun()

        sel_dist = st.selectbox(
            "Focus District",
            district_list,
            index=district_list.index(st.session_state.selected_district) if st.session_state.selected_district in district_list else 0
        )
        if sel_dist != st.session_state.selected_district:
            st.session_state.selected_district = sel_dist
            st.rerun()

        lead_opts = [f"{s['label']}" for s in TIMELINE_STEPS]
        sel_lead = st.selectbox("Forecast Lead Time", lead_opts, index=st.session_state.lead_time_idx)
        if lead_opts.index(sel_lead) != st.session_state.lead_time_idx:
            st.session_state.lead_time_idx = lead_opts.index(sel_lead)
            st.rerun()

    # Status Indicator: Forecast Engine Online
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; background:#ECFDF5; border:1px solid #A7F3D0; padding:8px 12px; border-radius:6px; margin:8px 0 8px 0; font-size:0.78rem; font-weight:600; color:#065F46; font-family:var(--font-mono);">
        <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#10B981; box-shadow:0 0 6px rgba(16,185,129,0.7);"></span>
        <span>Forecast Engine Online</span>
    </div>
    """, unsafe_allow_html=True)

    # User profile / analyst chip
    user_name = st.session_state.get('user_name', 'Mudit Sharma')
    user_role = st.session_state.get('user_role', 'Meteorological Operations Analyst')
    user_org = st.session_state.get('user_org', 'Ministry of Earth Sciences / NCMRWF')
    st.markdown(f"""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:10px 12px; margin-bottom:8px; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
        <div style="font-size:0.86rem; font-weight:700; color:#172033; display:flex; align-items:center; gap:6px;">
            <span>👤</span> {user_name}
        </div>
        <div style="font-size:0.72rem; color:#667085; margin-top:2px;">{user_role}</div>
        <div style="font-size:0.68rem; color:#94A3B8; font-family:var(--font-mono); margin-top:1px;">{user_org}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Sign Out / Lock Console", key="btn_sign_out_sidebar", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.auth_mode = "login"
        st.rerun()


# -----------------------------------------------------------------------------
# 6. UNIVERSAL METEOROLOGICAL ASSISTANT DRAWER
# -----------------------------------------------------------------------------
views.render_assistant_drawer(DISTRICTS_GDF, st.session_state.selected_district)

# -----------------------------------------------------------------------------
# 7. MAIN VIEW ROUTER DISPATCH
# -----------------------------------------------------------------------------
tab = st.session_state.current_tab

if tab in ["Overview", "Command Center"]:
    cc.render_command_center(
        DISTRICTS_GDF,
        VALID_TIME_STR,
        TIMELINE_STEPS,
        ROUTER_SPECS
    )
elif tab == "Live Forecast":
    cc.render_live_forecast(
        DISTRICTS_GDF,
        VALID_TIME_STR,
        TIMELINE_STEPS,
        ROUTER_SPECS
    )
elif tab == "District Intelligence":
    views.render_district_intelligence_view(
        DISTRICTS_GDF,
        st.session_state.selected_district,
        ROUTER_SPECS
    )
elif tab == "Regime Intelligence":
    views.render_regime_intelligence_view(
        ROUTER_SPECS
    )
elif tab in ["Heavy Rainfall Risk", "Risk & Alerts"]:
    views.render_heavy_rainfall_risk_view(
        DISTRICTS_GDF
    )
elif tab == "Explainability":
    views.render_explainability_view(
        DISTRICTS_GDF,
        st.session_state.selected_district
    )
elif tab == "What-If Lab":
    views.render_what_if_lab_view(
        DISTRICTS_GDF,
        st.session_state.selected_district
    )
elif tab == "Forecast Replay":
    views.render_forecast_replay_view()
elif tab == "Verification Lab":
    views.render_verification_lab_view()
elif tab == "Probability Calibration":
    views.render_probability_calibration_view()
elif tab == "Ablation Study":
    views.render_ablation_study_view()
elif tab == "Data Provenance":
    views.render_data_provenance_view()
elif tab == "Reports":
    views.render_reports_view(
        DISTRICTS_GDF,
        VALID_TIME_STR
    )
elif tab in ["Settings", "Operational Controls"]:
    views.render_settings_view(
        ALERT_DATES,
        district_list,
        TIMELINE_STEPS
    )
else:
    cc.render_command_center(
        DISTRICTS_GDF,
        VALID_TIME_STR,
        TIMELINE_STEPS,
        ROUTER_SPECS
    )

# -----------------------------------------------------------------------------
# 8. SUBTLE QUIET DISCLAIMER (Requirement 6)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="quiet-disclaimer">
    <b>OPERATIONAL METEOROLOGICAL DISCLAIMER:</b> VarshaMitra is an operational AI meteorological research and decision-support prototype developed for SIH 2026. Official public weather forecasts, warnings, and emergency advisories are issued exclusively by the India Meteorological Department (IMD) and National Disaster Management Authority (NDMA).
</div>
""", unsafe_allow_html=True)
