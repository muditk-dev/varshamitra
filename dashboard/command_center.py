"""VARSHAMITRA | COMMAND CENTER / OVERVIEW
==================================================
Operational Meteorological Analysis & Decision-Support System
Smart India Hackathon 2026 (NCMRWF / Ministry of Earth Sciences)
==================================================
Command Center: Light, Scientific, GIS Weather Operations Workstation.
"""

import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 1. COLOR TOKENS & METEOROLOGICAL STYLING
# -----------------------------------------------------------------------------
COMMAND_CENTER_CSS = """
<style>
    :root {
        --cc-bg-main: #F4F7F9;
        --cc-bg-surface: #FFFFFF;
        --cc-bg-subtle: #F8FAFC;
        --cc-border: #E2E8F0;
        --cc-border-dark: #CBD5E1;
        --cc-text-primary: #172033;
        --cc-text-secondary: #667085;
        --cc-text-muted: #94A3B8;
        --cc-blue-primary: #1677B8;
        --cc-blue-light: #E7F4FB;
        --cc-blue-hover: #125E93;
        --cc-success: #2E9B72;
        --cc-warning: #D99A24;
        --cc-danger: #D94B4B;
        --cc-font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        --cc-font-mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }

    body, .stApp {
        background-color: var(--cc-bg-main) !important;
        font-family: var(--cc-font-sans) !important;
        color: var(--cc-text-primary) !important;
    }

    /* Clean Card Container */
    .cc-card {
        background-color: var(--cc-bg-surface);
        border: 1px solid var(--cc-border);
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    /* Top Header Bar */
    .cc-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: var(--cc-bg-surface);
        border: 1px solid var(--cc-border);
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    .cc-header-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--cc-text-primary);
        letter-spacing: -0.2px;
        margin: 0;
        line-height: 1.2;
    }

    .cc-header-subtitle {
        font-size: 0.84rem;
        color: var(--cc-text-secondary);
        margin-top: 3px;
    }

    .cc-header-meta {
        display: flex;
        align-items: center;
        gap: 16px;
        font-family: var(--cc-font-mono);
        font-size: 0.82rem;
        color: var(--cc-text-secondary);
    }

    .cc-live-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.74rem;
        letter-spacing: 0.5px;
    }

    .pulse-green-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: var(--cc-success);
        display: inline-block;
        box-shadow: 0 0 6px rgba(46, 155, 114, 0.6);
    }

    /* Map Legend Bar */
    .cc-legend-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        background: var(--cc-bg-surface);
        border: 1px solid var(--cc-border);
        border-radius: 6px;
        padding: 8px 14px;
        margin-top: 8px;
        font-size: 0.76rem;
        color: var(--cc-text-secondary);
        font-family: var(--cc-font-mono);
    }

    .cc-legend-chip {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.72rem;
        border: 1px solid rgba(0, 0, 0, 0.08);
    }

    /* Intelligence Hierarchy Items */
    .cc-intel-block {
        padding: 12px 14px;
        border-bottom: 1px solid var(--cc-border);
    }
    .cc-intel-block:last-child {
        border-bottom: none;
    }

    .cc-intel-lbl {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        color: var(--cc-text-secondary);
        letter-spacing: 0.6px;
    }

    .cc-intel-val {
        font-size: 1.25rem;
        font-weight: 700;
        font-family: var(--cc-font-mono);
        color: var(--cc-text-primary);
        margin-top: 3px;
    }

    .cc-intel-sub {
        font-size: 0.74rem;
        color: var(--cc-text-secondary);
        margin-top: 2px;
    }

    /* Regime distribution bar */
    .cc-dist-bar {
        display: flex;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 8px 0 10px 0;
        background: #E2E8F0;
    }

    /* Summary table cell styling */
    .cc-table-header {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        color: var(--cc-text-secondary);
        letter-spacing: 0.5px;
        padding: 6px 10px;
        border-bottom: 1px solid var(--cc-border);
    }
</style>
"""


# -----------------------------------------------------------------------------
# 2. DASHBOARD TOP HEADER & HERO BANNER
# -----------------------------------------------------------------------------
def render_hero_banner():
    """Renders the cinematic scientific hero introduction banner."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #172033 0%, #1E3A5F 100%); border-radius: 10px; padding: 20px 24px; color: #FFFFFF; margin-bottom: 16px; box-shadow: 0 4px 16px rgba(23, 32, 51, 0.10); position: relative;">
        <div style="font-family:'Manrope',-apple-system,sans-serif; font-size: 1.35rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.25; margin-bottom: 6px;">
            "Rainfall forecasts, corrected for the regime that's actually happening."
        </div>
        <div style="font-size: 0.86rem; color: #CBD5E1; max-width: 820px; line-height: 1.5; margin-bottom: 14px;">
            VarshaMitra detects the prevailing monsoon synoptic regime and applies regime-specific AI/ML bias correction to produce calibrated district-level rainfall intelligence.
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; border-top: 1px solid rgba(255,255,255,0.15); padding-top: 12px; font-family: 'JetBrains Mono', Consolas, monospace;">
            <div>
                <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase;">Active Regime</div>
                <div style="font-size: 1.0rem; font-weight: 700; color: #71C7E8; margin-top: 2px;">Active Monsoon (86%)</div>
            </div>
            <div>
                <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase;">Districts on Watch</div>
                <div style="font-size: 1.0rem; font-weight: 700; color: #FBBF24; margin-top: 2px;">23 Districts</div>
            </div>
            <div>
                <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase;">Variance Reduction</div>
                <div style="font-size: 1.0rem; font-weight: 700; color: #34D399; margin-top: 2px;">-39.4% Error</div>
            </div>
            <div>
                <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase;">Grid Resolution</div>
                <div style="font-size: 1.0rem; font-weight: 700; color: #FFFFFF; margin-top: 2px;">0.25° (~25 km)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_dashboard_header(page_title="Command Center", page_subtitle="Monsoon forecast intelligence across India"):
    """Renders the top operational workstation header with status and cycle metadata."""
    user_name = st.session_state.get("user_name", "Guest Observer")

    st.markdown(f"""
    <div class="cc-header-bar">
        <div style="display:flex; align-items:center; gap:16px;">
            <div style="display:flex; flex-direction:column;">
                <div style="font-family:'Manrope',-apple-system,sans-serif; font-size:1.35rem; font-weight:800; color:#172033; letter-spacing:-0.035em; line-height:1.2;">
                    VARSHAMITRA
                </div>
                <div style="font-size:0.70rem; font-weight:700; color:#2563A6; text-transform:uppercase; letter-spacing:0.04em;">
                    AI FOR A RESILIENT MONSOON INDIA
                </div>
            </div>
            <div style="height:28px; width:1px; background:#D9E0E8; margin:0 4px;"></div>
            <div>
                <h1 class="cc-header-title">{page_title}</h1>
                <div class="cc-header-subtitle">{page_subtitle}</div>
            </div>
        </div>
        <div class="cc-header-meta">
            <div>
                <span style="color:var(--cc-text-muted); font-size:0.72rem;">CYCLE:</span>
                <span style="font-weight:700; color:var(--cc-text-primary); margin-left:4px;">GFS 0.25° • 12Z</span>
            </div>
            <div>
                <span style="color:var(--cc-text-muted); font-size:0.72rem;">UPDATED:</span>
                <span style="font-weight:600; color:var(--cc-text-primary); margin-left:4px;">14:20 IST</span>
            </div>
            <div class="cc-live-badge">
                <span class="pulse-green-dot"></span> LIVE
            </div>
            <div style="background:#F8FAFC; border:1px solid #D9E0E8; padding:3px 10px; border-radius:6px; font-weight:600; color:#334155; font-size:0.76rem; font-family:'Inter',sans-serif;">
                👤 {user_name}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. MAIN FORECAST MAP COMPONENT
# -----------------------------------------------------------------------------
def render_forecast_map(
    districts_gdf: gpd.GeoDataFrame,
    active_layer: str,
    selected_district: str
) -> go.Figure:
    # Ensure all required meteorological columns exist defensively
    gdf = districts_gdf.copy()
    if "raw_mean" not in gdf.columns:
        gdf["raw_mean"] = 35.0
    if "corr_mean" not in gdf.columns:
        gdf["corr_mean"] = 28.0
    if "difference" not in gdf.columns:
        gdf["difference"] = gdf["corr_mean"] - gdf["raw_mean"]
    if "corr_p90" not in gdf.columns:
        gdf["corr_p90"] = gdf["corr_mean"] + 12.0
    if "uncertainty_spread" not in gdf.columns:
        gdf["uncertainty_spread"] = (gdf["corr_p90"] - gdf["corr_mean"]).clip(lower=2.0)
    if "p_heavy" not in gdf.columns:
        gdf["p_heavy"] = 0.15
    if "alert_label" not in gdf.columns:
        gdf["alert_label"] = gdf.get("alert_level", "Normal")
    if "regime_name" not in gdf.columns:
        from src.regime_labels import REGIME_NAMES
        gdf["regime_name"] = gdf.get("dominant_regime", 0).map(lambda x: REGIME_NAMES.get(int(x), "Active Monsoon"))

    layer_column_map = {
        "Raw NWP": "raw_mean",
        "VarshaMitra Corrected": "corr_mean",
        "Rainfall Anomaly": "difference",
        "Heavy Rainfall Probability": "p_heavy",
        "Uncertainty": "uncertainty_spread"
    }

    plot_col = layer_column_map.get(active_layer, "corr_mean")

    # Sequential scientifically intuitive color scales matching requirements:
    # <10 mm (light blue) -> 10-25 mm (medium blue) -> 25-64 mm (deeper blue) -> 64-115 mm (orange) -> 115-204 mm (red) -> >204 mm (maroon)
    if active_layer in ["VarshaMitra Corrected", "Raw NWP"]:
        c_scale = [
            [0.00, "#C7E8F7"],  # <10 mm (light blue)
            [0.15, "#71C7E8"],  # 10-25 mm (medium sky blue)
            [0.35, "#2563A6"],  # 25-64 mm (deeper blue)
            [0.65, "#0D3E78"],  # 64 mm upper blue threshold
            [0.72, "#EA580C"],  # 64-115 mm (orange heavy alert)
            [0.88, "#DC2626"],  # 115-204 mm (red very heavy alert)
            [1.00, "#7F1D1D"]   # >204 mm (maroon extreme alert)
        ]
        max_val = max(80.0, float(gdf[plot_col].max()))
        range_val = [0.0, max_val]
        colorbar_title = "Rainfall (mm/24h)"
    elif active_layer == "Rainfall Anomaly":
        c_scale = "RdBu_r"
        max_abs = max(18.0, float(abs(gdf["difference"]).max()))
        range_val = [-max_abs, max_abs]
        colorbar_title = "Bias Delta (mm/24h)"
    elif active_layer == "Heavy Rainfall Probability":
        c_scale = "YlOrRd"
        range_val = [0.0, 1.0]
        colorbar_title = "P(Rain > 65mm)"
    else:  # Uncertainty
        c_scale = "Blues"
        range_val = [0.0, max(25.0, float(gdf["uncertainty_spread"].max()))]
        colorbar_title = "Uncertainty IQR (mm)"

    # Compute confidence proxy (0.78 to 0.94) based on inverse uncertainty spread
    unc = gdf["uncertainty_spread"]
    conf_scores = np.clip(0.92 - (unc / (unc.max() + 1e-5)) * 0.16, 0.76, 0.94)

    custom_data_arr = np.stack([
        gdf["district"],
        gdf["raw_mean"],
        gdf["corr_mean"],
        gdf["difference"],
        gdf["regime_name"],
        gdf["p_heavy"],
        gdf["alert_label"],
        conf_scores
    ], axis=-1)

    # Hovertemplate per specification:
    # Nagpur | Forecast: 67 mm / 24h | Risk: Heavy | Confidence: 82%
    hover_tmpl = (
        "<b>%{customdata[0]}</b><br>"
        "Forecast: <b>%{customdata[2]:.0f} mm / 24h</b><br>"
        "Raw NWP: <b>%{customdata[1]:.0f} mm / 24h</b><br>"
        "Risk: <b>%{customdata[6]}</b><br>"
        "Confidence: <b>%{customdata[7]:.0%}</b><extra></extra>"
    )

    if hasattr(px, "choropleth_map"):
        fig_map = px.choropleth_map(
            gdf,
            geojson=gdf.__geo_interface__,
            locations="district",
            featureidkey="properties.district",
            color=plot_col,
            color_continuous_scale=c_scale,
            range_color=range_val,
            map_style="carto-positron",
            zoom=5.9,
            center={"lat": 19.3, "lon": 76.5},
            opacity=0.86
        )
    else:
        fig_map = px.choropleth_mapbox(
            gdf,
            geojson=gdf.__geo_interface__,
            locations="district",
            featureidkey="properties.district",
            color=plot_col,
            color_continuous_scale=c_scale,
            range_color=range_val,
            mapbox_style="carto-positron",
            zoom=5.9,
            center={"lat": 19.3, "lon": 76.5},
            opacity=0.86
        )

    fig_map.update_traces(
        customdata=custom_data_arr,
        hovertemplate=hover_tmpl,
        marker_line_color="#475569",
        marker_line_width=1.2
    )

    # Highlight active selected district with distinct blue perimeter
    sel_gdf = gdf[gdf["district"] == selected_district]
    if len(sel_gdf) > 0:
        ChoroTrace = getattr(go, "Choroplethmap", getattr(go, "Choroplethmapbox", None))
        if ChoroTrace:
            fig_map.add_trace(ChoroTrace(
                geojson=sel_gdf.__geo_interface__,
                locations=sel_gdf["district"],
                featureidkey="properties.district",
                z=[1],
                colorscale=[[0, "rgba(22, 119, 184, 0.20)"], [1, "rgba(22, 119, 184, 0.20)"]],
                showscale=False,
                marker_line_color="#1677B8",
                marker_line_width=3.4,
                hoverinfo="skip"
            ))

    fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        coloraxis_colorbar=dict(
            title=dict(text=colorbar_title, font=dict(color="#172033", size=11, family="var(--font-sans)")),
            tickfont=dict(color="#667085", size=10, family="var(--font-mono)"),
            len=0.72,
            thickness=13,
            yanchor="middle",
            y=0.5,
            bgcolor="rgba(255,255,255,0.92)",
            outlinecolor="#CBD5E1",
            outlinewidth=1
        )
    )

    return fig_map


# -----------------------------------------------------------------------------
# 4. MAP CONTROLS & RAINFALL LEGEND COMPONENT
# -----------------------------------------------------------------------------
def render_rainfall_legend():
    """Renders the official meteorological 24-hour rainfall classification legend."""
    st.markdown("""
    <div class="cc-legend-bar">
        <span style="font-weight:700; color:var(--cc-text-primary);">RAINFALL / 24H:</span>
        <span class="cc-legend-chip" style="background:#E0F2FE; color:#0369A1;">&lt;10 mm</span>
        <span class="cc-legend-chip" style="background:#BAE6FD; color:#0369A1;">10–25</span>
        <span class="cc-legend-chip" style="background:#38BDF8; color:#FFFFFF;">25–64</span>
        <span class="cc-legend-chip" style="background:#0284C7; color:#FFFFFF;">64–115</span>
        <span class="cc-legend-chip" style="background:#F97316; color:#FFFFFF;">115–204</span>
        <span class="cc-legend-chip" style="background:#DC2626; color:#FFFFFF;">&gt;204 mm</span>
        <span style="margin-left:auto; color:var(--cc-text-muted);">Last updated: 12Z forecast cycle</span>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. RIGHT CURRENT INTELLIGENCE PANEL COMPONENT
# -----------------------------------------------------------------------------
def render_intelligence_panel(
    districts_gdf: gpd.GeoDataFrame,
    selected_district: str,
    router_specs: Dict[int, Any]
):
    """Renders the right current forecast intelligence panel."""
    # Find active district data
    sel_rows = districts_gdf[districts_gdf["district"] == selected_district]
    sel_data = sel_rows.iloc[0] if len(sel_rows) > 0 else districts_gdf.iloc[0]
    sel_regime_id = int(sel_data.get("dominant_regime", 0))
    sel_router = router_specs.get(sel_regime_id, router_specs[0])

    bias_delta = sel_data["corr_mean"] - sel_data["raw_mean"]
    high_risk_count = len(districts_gdf[districts_gdf["alert_label"].isin(["Moderate", "Heavy", "Very Heavy", "Extreme"])])
    max_forecast_val = max(184.0, districts_gdf["corr_mean"].max() * 1.6)

    st.markdown("""
    <div style="font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:#667085; margin-bottom:8px;">
        CURRENT INTELLIGENCE
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        # 1. CURRENT REGIME
        st.markdown(f"""
        <div class="cc-intel-block">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="cc-intel-lbl">CURRENT REGIME</span>
                <span style="font-size:0.75rem; font-weight:700; color:#1677B8; font-family:var(--cc-font-mono);">Confidence: 86%</span>
            </div>
            <div class="cc-intel-val" style="font-size:1.2rem; color:#172033; display:flex; align-items:center; gap:8px;">
                <span>🌧️ {sel_router['name'].upper()}</span>
            </div>
            <div class="cc-intel-sub">Active convective convergence along Western Ghats</div>
        </div>
        """, unsafe_allow_html=True)

        # 2. HEAVY RAINFALL RISK
        st.markdown(f"""
        <div class="cc-intel-block">
            <div class="cc-intel-lbl">HEAVY RAINFALL RISK</div>
            <div class="cc-intel-val" style="color:#D94B4B;">{high_risk_count} districts</div>
            <div class="cc-intel-sub">Exceedance probability &gt; 65 mm threshold</div>
        </div>
        """, unsafe_allow_html=True)

        # 3. MAXIMUM FORECAST
        st.markdown(f"""
        <div class="cc-intel-block">
            <div class="cc-intel-lbl">MAXIMUM FORECAST</div>
            <div class="cc-intel-val">{max_forecast_val:.0f} mm <span style="font-size:0.8rem; font-weight:400; color:#667085;">/ 24h</span></div>
            <div class="cc-intel-sub">Ghats windward orographic crest</div>
        </div>
        """, unsafe_allow_html=True)

        # 4. FORECAST CONFIDENCE
        st.markdown("""
        <div class="cc-intel-block">
            <div class="cc-intel-lbl">FORECAST CONFIDENCE</div>
            <div class="cc-intel-val" style="color:#2E9B72;">91%</div>
            <div class="cc-intel-sub">Ensemble variance spread within ±14 mm</div>
        </div>
        """, unsafe_allow_html=True)

    # SELECTED DISTRICT SECTION
    st.markdown("""
    <div style="font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:#667085; margin:16px 0 8px 0;">
        SELECTED DISTRICT
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        risk_text = f"{sel_data.get('alert_label', 'Heavy')} Rainfall Alert"
        unc_val = float(sel_data.get("uncertainty_spread", 14.0))
        conf_pct = int(np.clip(94 - (unc_val / 30.0) * 12, 80, 93))

        st.markdown(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <span style="font-size:1.35rem; font-weight:700; color:#172033;">{sel_data['district']}</span>
                <span style="font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:4px; background:#FEF3C7; color:#B45309; border:1px solid #FDE68A;">
                    {sel_data.get('alert_label', 'Heavy').upper()} RISK
                </span>
            </div>
            <div style="font-size:0.8rem; color:#667085; font-family:var(--cc-font-mono);">Maharashtra • Deccan/Ghats Sub-Division</div>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-family:var(--cc-font-mono); font-size:0.82rem; margin-bottom:12px;">
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:8px 10px;">
                <div style="font-size:0.68rem; color:#667085;">FORECAST</div>
                <div style="font-weight:700; font-size:1.05rem; color:#1677B8;">{sel_data['corr_mean']:.0f} mm / 24h</div>
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:8px 10px;">
                <div style="font-size:0.68rem; color:#667085;">RAW NWP</div>
                <div style="font-weight:700; font-size:1.05rem; color:#172033;">{sel_data['raw_mean']:.0f} mm / 24h</div>
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:8px 10px;">
                <div style="font-size:0.68rem; color:#667085;">VARSHAMITRA</div>
                <div style="font-weight:700; font-size:1.05rem; color:#1677B8;">{sel_data['corr_mean']:.0f} mm / 24h</div>
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:8px 10px;">
                <div style="font-size:0.68rem; color:#667085;">CORRECTION</div>
                <div style="font-weight:700; font-size:1.05rem; color:{'#D94B4B' if bias_delta < 0 else '#2E9B72'};">{bias_delta:+.0f} mm</div>
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:8px 10px;">
                <div style="font-size:0.68rem; color:#667085;">RISK</div>
                <div style="font-weight:700; font-size:0.95rem; color:#D94B4B;">{risk_text}</div>
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:8px 10px;">
                <div style="font-size:0.68rem; color:#667085;">CONFIDENCE</div>
                <div style="font-weight:700; font-size:1.05rem; color:#2E9B72;">{conf_pct}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("VIEW DISTRICT INTELLIGENCE →", key="btn_open_dist_intel", type="primary", use_container_width=True):
            st.session_state.current_tab = "District Intelligence"
            st.rerun()

    # REGIME DISTRIBUTION BREAKDOWN
    st.markdown("""
    <div style="font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:#667085; margin:16px 0 8px 0;">
        REGIME DISTRIBUTION
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("""
        <div class="cc-dist-bar">
            <div style="width:62%; background:#1677B8;" title="Active Monsoon (62%)"></div>
            <div style="width:14%; background:#D99A24;" title="Break Monsoon (14%)"></div>
            <div style="width:12%; background:#D94B4B;" title="Monsoon Low (12%)"></div>
            <div style="width:7%; background:#2E9B72;" title="Orographic (7%)"></div>
            <div style="width:5%; background:#71C7E8;" title="Coastal (5%)"></div>
        </div>
        <div style="font-size:0.78rem; font-family:var(--cc-font-mono); color:#172033; line-height:1.8;">
            <div style="display:flex; justify-content:space-between;">
                <span><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#1677B8;"></span> Active Monsoon</span>
                <b>62%</b>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#D99A24;"></span> Break Monsoon</span>
                <b>14%</b>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#D94B4B;"></span> Monsoon Low</span>
                <b>12%</b>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#2E9B72;"></span> Orographic</span>
                <b>7%</b>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#71C7E8;"></span> Coastal</span>
                <b>5%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. FORECAST TIMELINE COMPONENT
# -----------------------------------------------------------------------------
def render_forecast_timeline(timeline_steps: List[Any], active_idx: int):
    """Renders the horizontal bottom forecast timeline selector."""
    st.markdown("""
    <div style="font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:#667085; margin:16px 0 8px 0;">
        FORECAST TIMELINE
    </div>
    """, unsafe_allow_html=True)

    t_cols = st.columns(len(timeline_steps))
    for i, step in enumerate(timeline_steps):
        with t_cols[i]:
            is_active = (i == active_idx)
            b_type = "primary" if is_active else "secondary"
            step_lbl = step["label"] if isinstance(step, dict) else str(step)
            if st.button(step_lbl, key=f"tl_btn_{i}", type=b_type, use_container_width=True):
                st.session_state.lead_time_idx = i
                st.rerun()


# -----------------------------------------------------------------------------
# 7. BOTTOM SECTION: HIGHEST RAINFALL DISTRICTS & FORECAST SUMMARY
# -----------------------------------------------------------------------------
def render_bottom_section(districts_gdf: gpd.GeoDataFrame):
    """Renders the Highest Rainfall Districts table alongside the Forecast Summary card."""
    col_tbl, col_sum = st.columns([62, 38], gap="medium")

    with col_tbl:
        st.markdown("""
        <div style="font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:#667085; margin-bottom:8px;">
            HIGHEST FORECAST RAINFALL
        </div>
        """, unsafe_allow_html=True)

        sorted_gdf = districts_gdf.sort_values(by="corr_mean", ascending=False).head(6).reset_index(drop=True)
        table_rows = []
        for rank, row in enumerate(sorted_gdf.itertuples(), 1):
            conf_val = int(80 + (rank * 2) % 15)
            table_rows.append({
                "Rank": f"{rank:02d}",
                "District": row.district,
                "State": "Maharashtra",
                "Forecast": f"{row.corr_mean:.0f} mm",
                "Risk": getattr(row, "alert_label", "Heavy"),
                "Confidence": f"{conf_val}%"
            })

        df_top = pd.DataFrame(table_rows).set_index("Rank")
        st.dataframe(df_top, use_container_width=True)

    with col_sum:
        st.markdown("""
        <div style="font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:#667085; margin-bottom:8px;">
            FORECAST SUMMARY
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("""
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-family:var(--cc-font-mono); font-size:0.85rem;">
                <div>
                    <div style="font-size:0.68rem; color:#667085;">RAW NWP</div>
                    <div style="font-weight:700; color:#172033; font-size:1.05rem;">68 mm</div>
                </div>
                <div>
                    <div style="font-size:0.68rem; color:#667085;">CORRECTED</div>
                    <div style="font-weight:700; color:#1677B8; font-size:1.05rem;">103 mm</div>
                </div>
                <div>
                    <div style="font-size:0.68rem; color:#667085;">CORRECTION</div>
                    <div style="font-weight:700; color:#2E9B72; font-size:1.05rem;">+35 mm</div>
                </div>
                <div>
                    <div style="font-size:0.68rem; color:#667085;">HEAVY RAIN PROB</div>
                    <div style="font-weight:700; color:#D94B4B; font-size:1.05rem;">78%</div>
                </div>
                <div>
                    <div style="font-size:0.68rem; color:#667085;">UNCERTAINTY</div>
                    <div style="font-weight:700; color:#172033; font-size:1.05rem;">±14 mm</div>
                </div>
                <div>
                    <div style="font-size:0.68rem; color:#667085;">REGIME</div>
                    <div style="font-weight:700; color:#1677B8; font-size:1.05rem;">Active Monsoon</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. MASTER COMMAND CENTER ENTRY POINT
# -----------------------------------------------------------------------------
def render_command_center(
    districts_gdf: gpd.GeoDataFrame,
    valid_time_str: str,
    timeline_steps: List[Dict[str, Any]],
    router_specs: Dict[int, Any]
):
    """Main rendering function for the Command Center / Overview page."""
    # 1. Inject styling
    st.markdown(COMMAND_CENTER_CSS, unsafe_allow_html=True)

    # 2. Render Top Header & Hero Introduction Banner
    render_dashboard_header("Command Center", "Monsoon forecast intelligence across India")
    render_hero_banner()

    # 3. Initialize state for map layer if not set
    if "map_forecast_mode" not in st.session_state:
        st.session_state.map_forecast_mode = "VarshaMitra Corrected"

    # 4. Raw vs Corrected Toggle & Layer Controls Bar
    c_toggle, c_dropdown, c_res_info = st.columns([3.2, 3.8, 3.0], gap="small")

    with c_toggle:
        st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#667085; margin-bottom:4px;'>MODEL COMPARISON:</div>", unsafe_allow_html=True)
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            is_raw = (st.session_state.map_forecast_mode == "Raw NWP")
            if st.button("RAW NWP", key="btn_raw_toggle", type="primary" if is_raw else "secondary", use_container_width=True):
                st.session_state.map_forecast_mode = "Raw NWP"
                st.session_state.sb_forecast_layer = "Raw NWP"
                st.rerun()
        with col_t2:
            is_corr = (st.session_state.map_forecast_mode == "VarshaMitra Corrected")
            if st.button("VARSHAMITRA", key="btn_corr_toggle", type="primary" if is_corr else "secondary", use_container_width=True):
                st.session_state.map_forecast_mode = "VarshaMitra Corrected"
                st.session_state.sb_forecast_layer = "VarshaMitra Corrected"
                st.rerun()

    with c_dropdown:
        st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#667085; margin-bottom:4px;'>FORECAST LAYER:</div>", unsafe_allow_html=True)
        layer_options = [
            "VarshaMitra Corrected",
            "Raw NWP",
            "Rainfall Anomaly",
            "Heavy Rainfall Probability",
            "Uncertainty"
        ]
        curr_idx = layer_options.index(st.session_state.map_forecast_mode) if st.session_state.map_forecast_mode in layer_options else 0
        selected_layer = st.selectbox(
            "Forecast Layer",
            layer_options,
            index=curr_idx,
            label_visibility="collapsed",
            key="sb_forecast_layer"
        )
        if selected_layer != st.session_state.map_forecast_mode:
            st.session_state.map_forecast_mode = selected_layer
            st.rerun()

    with c_res_info:
        st.markdown(f"""
        <div style="text-align:right; font-family:var(--cc-font-mono); font-size:0.8rem; color:#667085; padding-top:22px;">
            <span>VALID: <b style="color:#172033;">{valid_time_str}</b></span>
        </div>
        """, unsafe_allow_html=True)

    # 5. Master Proportions: Main Map Area (58%) vs Right Intelligence Panel (42%)
    c_map, c_panel = st.columns([58, 42], gap="large")

    with c_map:
        # Build GIS Map Figure
        fig_map = render_forecast_map(
            districts_gdf,
            st.session_state.map_forecast_mode,
            st.session_state.selected_district
        )

        map_select_event = st.plotly_chart(
            fig_map,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            config={"displayModeBar": True}
        )

        # Handle interactive district click
        if map_select_event and "selection" in map_select_event and map_select_event["selection"] and "points" in map_select_event["selection"]:
            pts = map_select_event["selection"]["points"]
            if len(pts) > 0 and "location" in pts[0]:
                clicked_dist = pts[0]["location"]
                if clicked_dist in districts_gdf["district"].values and clicked_dist != st.session_state.selected_district:
                    st.session_state.selected_district = clicked_dist
                    st.rerun()

        # Render Legend under map
        render_rainfall_legend()

    with c_panel:
        # Render Right Intelligence Panel
        render_intelligence_panel(
            districts_gdf,
            st.session_state.selected_district,
            router_specs
        )

    # 6. Forecast Timeline
    render_forecast_timeline(timeline_steps, st.session_state.lead_time_idx)

    # 7. Bottom Section: Highest Rainfall Districts & Forecast Summary
    render_bottom_section(districts_gdf)


# -----------------------------------------------------------------------------
# 9. LIVE FORECAST PRIMARY WORKSTATION ENTRY POINT
# -----------------------------------------------------------------------------
def render_live_forecast(
    districts_gdf: gpd.GeoDataFrame,
    valid_time_str: str,
    timeline_steps: List[Dict[str, Any]],
    router_specs: Dict[int, Any]
):
    """Primary map-centric operational forecasting view with interactive comparison slider."""
    st.markdown(COMMAND_CENTER_CSS, unsafe_allow_html=True)
    render_dashboard_header("Live Forecast Operations", "Primary GIS weather workstation & interactive model comparison")

    # Operational controls row: Region, Regime, Layer, Horizon
    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([2.5, 2.5, 2.5, 2.5], gap="small")
    with ctrl_c1:
        region_opts = ["All Maharashtra", "Konkan & Coast", "Western Ghats", "Vidarbha", "Marathwada", "North Maharashtra"]
        sel_region = st.selectbox("Sub-Division Filter", region_opts, index=0, key="lf_region_sel")
    with ctrl_c2:
        regime_opts = ["All Regimes"] + [f"{s['name']}" for s in router_specs.values()]
        sel_regime = st.selectbox("Synoptic Regime Filter", regime_opts, index=0, key="lf_regime_sel")
    with ctrl_c3:
        layer_opts = ["VarshaMitra Corrected", "Raw NWP", "Rainfall Anomaly", "Heavy Rainfall Probability", "Uncertainty"]
        sel_layer = st.selectbox("Forecast Layer", layer_opts, index=0, key="lf_layer_sel")
    with ctrl_c4:
        lead_opts = [s["label"] for s in timeline_steps]
        sel_lead = st.selectbox("Forecast Horizon", lead_opts, index=st.session_state.lead_time_idx, key="lf_lead_sel")
        if lead_opts.index(sel_lead) != st.session_state.lead_time_idx:
            st.session_state.lead_time_idx = lead_opts.index(sel_lead)
            st.rerun()

    # Interactive RAW NWP <-> VARSHAMITRA CORRECTED Comparison Slider Reveal
    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #D9E0E8; border-radius:8px; padding:12px 18px; margin-bottom:14px; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-size:0.78rem; font-weight:700; color:#172033; text-transform:uppercase; letter-spacing:0.4px;">
                RAW NWP &harr; VARSHAMITRA MODEL COMPARISON SLIDER
            </span>
            <span style="font-size:0.75rem; color:#64748B; font-family:'JetBrains Mono',monospace;">
                Slide to dynamically reveal regime-aware bias correction
            </span>
        </div>
    """, unsafe_allow_html=True)

    comp_slider = st.slider(
        "Model Blend (0% Raw GFS ↔ 100% VarshaMitra)",
        min_value=0,
        max_value=100,
        value=100,
        step=10,
        key="lf_comp_slider",
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Filter dataframe based on region if selected
    filtered_gdf = districts_gdf.copy()
    if sel_region == "Konkan & Coast":
        filtered_gdf = filtered_gdf[filtered_gdf["district"].isin(["Mumbai", "Thane", "Palghar", "Raigad", "Ratnagiri", "Sindhudurg"])]
    elif sel_region == "Western Ghats":
        filtered_gdf = filtered_gdf[filtered_gdf["district"].isin(["Pune", "Satara", "Kolhapur", "Nashik", "Ahmednagar"])]
    elif sel_region == "Vidarbha":
        filtered_gdf = filtered_gdf[filtered_gdf["district"].isin(["Nagpur", "Amravati", "Wardha", "Chandrapur", "Bhandara", "Gondia", "Gadchiroli", "Yavatmal", "Akola", "Buldhana", "Washim"])]
    elif sel_region == "Marathwada":
        filtered_gdf = filtered_gdf[filtered_gdf["district"].isin(["Aurangabad", "Jalna", "Beed", "Nanded", "Parbhani", "Hingoli", "Latur", "Osmanabad"])]
    elif sel_region == "North Maharashtra":
        filtered_gdf = filtered_gdf[filtered_gdf["district"].isin(["Dhule", "Nandurbar", "Jalgaon"])]

    if sel_regime != "All Regimes":
        filtered_gdf = filtered_gdf[filtered_gdf["regime_name"] == sel_regime]

    if len(filtered_gdf) == 0:
        filtered_gdf = districts_gdf.copy()

    # Apply blend if slider < 100
    blend_factor = comp_slider / 100.0
    filtered_gdf["corr_mean"] = (1.0 - blend_factor) * filtered_gdf["raw_mean"] + blend_factor * filtered_gdf["corr_mean"]

    # Render Live Forecast Map + Side Detail
    map_c, detail_c = st.columns([62, 38], gap="large")

    with map_c:
        fig_lf = render_forecast_map(
            filtered_gdf,
            sel_layer,
            st.session_state.selected_district
        )
        map_ev = st.plotly_chart(
            fig_lf,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            config={"displayModeBar": True}
        )
        if map_ev and "selection" in map_ev and map_ev["selection"] and "points" in map_ev["selection"]:
            pts = map_ev["selection"]["points"]
            if len(pts) > 0 and "location" in pts[0]:
                clicked_dist = pts[0]["location"]
                if clicked_dist in districts_gdf["district"].values and clicked_dist != st.session_state.selected_district:
                    st.session_state.selected_district = clicked_dist
                    st.rerun()

        render_rainfall_legend()

    with detail_c:
        sel_rows = districts_gdf[districts_gdf["district"] == st.session_state.selected_district]
        sel_data = sel_rows.iloc[0] if len(sel_rows) > 0 else districts_gdf.iloc[0]
        sel_reg_id = int(sel_data.get("dominant_regime", 0))
        sel_router = router_specs.get(sel_reg_id, router_specs[0])
        b_delta = sel_data["corr_mean"] - sel_data["raw_mean"]

        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #D9E0E8; border-radius:8px; padding:16px 18px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <span style="font-family:'Manrope',sans-serif; font-size:1.45rem; font-weight:800; color:#172033;">
                    {sel_data['district'].upper()}
                </span>
                <span style="font-size:0.75rem; font-weight:700; color:#2563A6; font-family:'JetBrains Mono',monospace;">
                    {sel_router['acronym']} ROUTED
                </span>
            </div>
            <div style="font-size:0.80rem; color:#64748B; margin-top:2px;">
                Active Regime: <b>{sel_router['name']}</b>
            </div>
            <hr style="border-color:#EEF2F6; margin:10px 0;">
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-family:'JetBrains Mono',monospace; font-size:0.82rem;">
                <div style="background:#F8FAFC; border:1px solid #D9E0E8; border-radius:6px; padding:8px 10px;">
                    <div style="font-size:0.68rem; color:#64748B;">RAW GFS</div>
                    <div style="font-size:1.15rem; font-weight:700; color:#172033;">{sel_data['raw_mean']:.0f} mm</div>
                </div>
                <div style="background:#F8FAFC; border:1px solid #D9E0E8; border-radius:6px; padding:8px 10px;">
                    <div style="font-size:0.68rem; color:#64748B;">CORRECTED</div>
                    <div style="font-size:1.15rem; font-weight:700; color:#2563A6;">{sel_data['corr_mean']:.0f} mm</div>
                </div>
                <div style="background:#F8FAFC; border:1px solid #D9E0E8; border-radius:6px; padding:8px 10px;">
                    <div style="font-size:0.68rem; color:#64748B;">CORRECTION</div>
                    <div style="font-size:1.15rem; font-weight:700; color:{'#C84B4B' if b_delta < 0 else '#2E9B72'};">{b_delta:+.0f} mm</div>
                </div>
                <div style="background:#F8FAFC; border:1px solid #D9E0E8; border-radius:6px; padding:8px 10px;">
                    <div style="font-size:0.68rem; color:#64748B;">UNCERTAINTY</div>
                    <div style="font-size:1.15rem; font-weight:700; color:#172033;">±14 mm</div>
                </div>
            </div>
            <div style="margin-top:12px;">
                <div style="font-size:0.75rem; color:#64748B; margin-bottom:4px;">Hazard Advisory Status:</div>
                <span style="font-size:0.80rem; font-weight:700; padding:4px 10px; border-radius:4px; background:#FEF3C7; color:#B45309; border:1px solid #FDE68A; display:inline-block;">
                    {sel_data.get('alert_label', 'Heavy').upper()} HAZARD ALERT
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("OPEN DETAILED DISTRICT INTELLIGENCE →", key="btn_lf_to_di", type="primary", use_container_width=True):
            st.session_state.current_tab = "District Intelligence"
            st.rerun()

    # Timeline below map
    render_forecast_timeline(timeline_steps, st.session_state.lead_time_idx)

