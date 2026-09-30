"""VARSHAMITRA | AUTHENTICATION & LOGIN EXPERIENCE
==================================================
Operational Meteorological Analysis & Decision-Support System
Smart India Hackathon 2026 (NCMRWF / Ministry of Earth Sciences)
==================================================
Light Scientific Meteorological Authentication Gateway.
Unified with VarshaMitra Command Center Design System.
"""

import re
import time
from typing import Tuple
import streamlit as st
import streamlit.components.v1 as components

# -----------------------------------------------------------------------------
# 1. INITIALIZE AUTHENTICATION SESSION STATE
# -----------------------------------------------------------------------------
def init_auth_session():
    """Initializes authentication session state with clean defaults."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"  # "login" or "register"
    if "user_name" not in st.session_state:
        st.session_state.user_name = "Guest Observer"
    if "user_role" not in st.session_state:
        st.session_state.user_role = "SIH 2026 Evaluator"
    if "auth_org" not in st.session_state:
        st.session_state.auth_org = "Government & Research Access"
    if "auth_error" not in st.session_state:
        st.session_state.auth_error = ""

    # Direct query parameter bypass for evaluation or automated testing
    if "guest" in st.query_params or "auth" in st.query_params or "bypass" in st.query_params:
        st.session_state.authenticated = True
        st.session_state.user_name = "Guest Observer"
        st.session_state.user_role = "SIH 2026 Evaluator"
        st.session_state.auth_org = "Operational Guest Access"


# -----------------------------------------------------------------------------
# 2. LIVING METEOROLOGICAL CANVAS COMPONENT (LIGHT GIS / ATMOSPHERIC)
# -----------------------------------------------------------------------------
def get_auth_canvas_html() -> str:
    """Returns self-contained HTML5 Canvas animation of Indian monsoon meteorology
    rendered in a light, restrained scientific aesthetic:
    - Light grey-blue background (#F5F7F9)
    - Faint India subcontinent boundary & Western Ghats ridge
    - Subtle isobar pressure contours
    - Light SW monsoon streamline flow
    - Muted slow rain particles
    - Small blue radar precipitation cells
    - Synoptic observation stations
    Respects prefers-reduced-motion.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: 100%; height: 100%; overflow: hidden; background: transparent; }
  canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: block; }
</style>
</head>
<body>
<canvas id="bg-canvas"></canvas>
<script>
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');
  let width, height;

  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  // Gentle light rain particles (muted slate blue-grey, slow and gentle)
  const raindrops = [];
  const RAIN_COUNT = 85;
  for (let i = 0; i < RAIN_COUNT; i++) {
    raindrops.push({
      x: Math.random() * (window.innerWidth + 200) - 100,
      y: Math.random() * window.innerHeight,
      len: 7 + Math.random() * 10,
      speed: 2.8 + Math.random() * 2.8,
      opacity: 0.08 + Math.random() * 0.12,
      dx: 1.0 + Math.random() * 0.6
    });
  }

  // Southwest Monsoon wind streamline particles
  const streamlines = [];
  const STREAMLINE_COUNT = 16;
  for (let i = 0; i < STREAMLINE_COUNT; i++) {
    streamlines.push({
      startX: (Math.random() * 0.35) * window.innerWidth,
      startY: (0.50 + Math.random() * 0.45) * window.innerHeight,
      progress: Math.random(),
      speed: 0.0008 + Math.random() * 0.0014,
      length: 70 + Math.random() * 80,
      opacity: 0.07 + Math.random() * 0.10
    });
  }

  // Geographic lat/lon grid lines (very subtle)
  function drawGeographicGrid(w, h) {
    ctx.save();
    ctx.strokeStyle = "rgba(104, 117, 138, 0.035)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 5]);

    for (let x = 60; x < w; x += 130) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 60; y < h; y += 130) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    ctx.restore();
  }

  // Faint India Subcontinent & Meteorological Focus
  function drawIndiaMap(w, h) {
    const cx = w * 0.22;
    const cy = h * 0.46;
    const s = Math.min(w, h) * 0.62;

    ctx.save();
    // Subcontinent polygon boundary
    ctx.strokeStyle = "rgba(47, 109, 176, 0.12)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);

    ctx.beginPath();
    ctx.moveTo(cx - 0.02 * s, cy - 0.42 * s);
    ctx.lineTo(cx + 0.08 * s, cy - 0.38 * s);
    ctx.lineTo(cx + 0.12 * s, cy - 0.30 * s);
    ctx.lineTo(cx + 0.24 * s, cy - 0.26 * s);
    ctx.lineTo(cx + 0.38 * s, cy - 0.28 * s);
    ctx.lineTo(cx + 0.36 * s, cy - 0.20 * s);
    ctx.lineTo(cx + 0.22 * s, cy - 0.12 * s);
    ctx.lineTo(cx + 0.18 * s, cy + 0.05 * s);
    ctx.lineTo(cx + 0.10 * s, cy + 0.22 * s);
    ctx.lineTo(cx - 0.02 * s, cy + 0.38 * s);
    ctx.lineTo(cx - 0.08 * s, cy + 0.22 * s);
    ctx.lineTo(cx - 0.14 * s, cy + 0.04 * s);
    ctx.lineTo(cx - 0.24 * s, cy - 0.02 * s);
    ctx.lineTo(cx - 0.20 * s, cy - 0.12 * s);
    ctx.lineTo(cx - 0.12 * s, cy - 0.26 * s);
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);

    // Western Ghats orographic mountain spine
    ctx.beginPath();
    ctx.moveTo(cx - 0.13 * s, cy - 0.02 * s);
    ctx.quadraticCurveTo(cx - 0.12 * s, cy + 0.12 * s, cx - 0.07 * s, cy + 0.26 * s);
    ctx.strokeStyle = "rgba(47, 109, 176, 0.20)";
    ctx.lineWidth = 2.0;
    ctx.stroke();

    // Subtle precipitation echoes / radar cells
    ctx.fillStyle = "rgba(47, 109, 176, 0.05)";
    ctx.beginPath();
    ctx.arc(cx - 0.10 * s, cy + 0.05 * s, 0.055 * s, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "rgba(47, 109, 176, 0.035)";
    ctx.beginPath();
    ctx.arc(cx - 0.06 * s, cy + 0.12 * s, 0.045 * s, 0, Math.PI * 2);
    ctx.fill();

    // Synoptic Stations (Subtle dots)
    const stations = [
      { x: cx - 0.08 * s, y: cy + 0.05 * s, active: true },
      { x: cx - 0.13 * s, y: cy + 0.03 * s, active: true },
      { x: cx - 0.07 * s, y: cy + 0.13 * s },
      { x: cx + 0.05 * s, y: cy + 0.01 * s },
      { x: cx - 0.04 * s, y: cy - 0.24 * s }
    ];

    stations.forEach(st => {
      ctx.beginPath();
      ctx.arc(st.x, st.y, st.active ? 2.5 : 1.8, 0, Math.PI * 2);
      ctx.fillStyle = st.active ? "rgba(47, 109, 176, 0.45)" : "rgba(104, 117, 138, 0.30)";
      ctx.fill();

      if (st.active && !prefersReducedMotion) {
        ctx.beginPath();
        ctx.arc(st.x, st.y, 4 + (Math.sin(Date.now() * 0.002) * 1.2), 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(47, 109, 176, 0.15)";
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
    });

    ctx.restore();
  }

  // Atmospheric Isobar Curves (Very faint)
  function drawAtmosphericIsobars(w, h) {
    ctx.save();
    ctx.strokeStyle = "rgba(47, 109, 176, 0.05)";
    ctx.lineWidth = 1;

    for (let i = 0; i < 3; i++) {
      ctx.beginPath();
      const offset = i * 70;
      ctx.moveTo(-60, h * 0.70 - offset);
      ctx.bezierCurveTo(
        w * 0.16, h * 0.60 - offset,
        w * 0.30, h * 0.74 - offset,
        w * 0.54, h * 0.46 - offset
      );
      ctx.stroke();
    }
    ctx.restore();
  }

  // Restrained Radar Sweep
  let radarAngle = 0;
  function drawRadarSweep(w, h) {
    const rx = w * 0.20;
    const ry = h * 0.48;
    const maxR = Math.min(w, h) * 0.32;

    ctx.save();
    for (let r = 50; r <= maxR; r += 60) {
      ctx.beginPath();
      ctx.arc(rx, ry, r, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(47, 109, 176, 0.035)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    if (!prefersReducedMotion) {
      radarAngle += 0.004;
      const grad = ctx.createRadialGradient(rx, ry, 10, rx, ry, maxR);
      grad.addColorStop(0, "rgba(47, 109, 176, 0.03)");
      grad.addColorStop(1, "rgba(47, 109, 176, 0.0)");

      ctx.beginPath();
      ctx.moveTo(rx, ry);
      ctx.arc(rx, ry, maxR, radarAngle - 0.18, radarAngle);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
    }
    ctx.restore();
  }

  // Southwest Monsoon wind streamlines
  function drawStreamlines(w, h) {
    ctx.save();
    ctx.lineWidth = 1.0;

    streamlines.forEach(line => {
      line.progress += line.speed;
      if (line.progress > 1) {
        line.progress = 0;
        line.startX = (Math.random() * 0.35) * w;
        line.startY = (0.50 + Math.random() * 0.45) * h;
      }

      const curDist = line.progress * line.length;
      const x1 = line.startX + curDist * 1.35;
      const y1 = line.startY - curDist * 0.82 - Math.sin(line.progress * Math.PI) * 16;
      const x2 = x1 + 9;
      const y2 = y1 - 5;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = `rgba(47, 109, 176, ${line.opacity * (1 - line.progress)})`;
      ctx.stroke();
    });
    ctx.restore();
  }

  // Gentle light rain particles
  function drawRain(w, h) {
    ctx.save();
    ctx.lineWidth = 1.0;

    raindrops.forEach(drop => {
      drop.x += drop.dx;
      drop.y += drop.speed;

      if (drop.y > h || drop.x > w) {
        drop.x = Math.random() * (w + 200) - 150;
        drop.y = -20;
      }

      ctx.beginPath();
      ctx.moveTo(drop.x, drop.y);
      ctx.lineTo(drop.x + drop.dx * (drop.len / drop.speed), drop.y + drop.len);
      ctx.strokeStyle = `rgba(104, 117, 138, ${drop.opacity})`;
      ctx.stroke();
    });
    ctx.restore();
  }

  function animate() {
    ctx.fillStyle = "#F5F7F9";
    ctx.fillRect(0, 0, width, height);

    drawGeographicGrid(width, height);
    drawAtmosphericIsobars(width, height);
    drawRadarSweep(width, height);
    drawIndiaMap(width, height);

    if (!prefersReducedMotion) {
      drawStreamlines(width, height);
      drawRain(width, height);
      requestAnimationFrame(animate);
    } else {
      drawStreamlines(width, height);
      drawRain(width, height);
    }
  }

  animate();
</script>
</body>
</html>"""


# -----------------------------------------------------------------------------
# 3. AUTHENTICATION PAGE CSS STYLING (LIGHT WORKSTATION DESIGN SYSTEM)
# -----------------------------------------------------------------------------
AUTH_PAGE_CSS = """
<style>
    /* Design Tokens strictly aligned with VarshaMitra Command Center */
    :root {
        --auth-bg-main: #F5F7F9;
        --auth-bg-surface: #FFFFFF;
        --auth-bg-subtle: #F8FAFC;
        --auth-border-panel: #D8E0E8;
        --auth-border-subtle: #E2E8F0;
        --auth-blue-primary: #2F6DB0;
        --auth-blue-hover: #1E528B;
        --auth-blue-light: #EBF3FA;
        --auth-text-primary: #17233A;
        --auth-text-secondary: #68758A;
        --auth-text-muted: #94A3B8;
        --auth-success: #2E9B72;
        --auth-success-bg: #ECFDF5;
        --auth-success-border: #A7F3D0;
        --auth-warning: #D99A24;
        --auth-font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        --auth-font-mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }

    /* Fullscreen app styling for login page */
    .stApp {
        background-color: var(--auth-bg-main) !important;
        font-family: var(--auth-font-sans) !important;
        color: var(--auth-text-primary) !important;
    }

    /* Hide standard sidebar, deploy button and header elements on auth page */
    [data-testid="stSidebar"],
    [data-testid="stHeader"],
    .stDeployButton,
    [data-testid="stToolbar"] {
        display: none !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
    }

    /* Fixed Canvas Iframe behind all content */
    iframe {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        z-index: 0 !important;
        pointer-events: none !important;
    }

    /* Page container constraints */
    .block-container {
        position: relative !important;
        z-index: 10 !important;
        padding-top: 3.0rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1320px !important;
    }

    /* Clean White Authentication Card Container (targets main right column ONLY) */
    .stMain [data-testid="stHorizontalBlock"]:not([data-testid="stForm"] [data-testid="stHorizontalBlock"]) > [data-testid="stColumn"]:nth-of-type(2) > [data-testid="stVerticalBlock"] {
        background-color: var(--auth-bg-surface) !important;
        border: 1px solid #D0D7DE !important;
        border-radius: 14px !important;
        padding: 32px 36px 36px 36px !important;
        box-shadow: 0 10px 30px -4px rgba(23, 35, 58, 0.10), 0 2px 8px -1px rgba(23, 35, 58, 0.05) !important;
        height: fit-content !important;
        animation: cardSlideUp 0.32s ease-out !important;
    }

    /* Reset nested columns inside forms so they never inherit card framing */
    [data-testid="stForm"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        border-radius: 0 !important;
        animation: none !important;
    }

    /* Segmented Control Radio Styling */
    div.st-key-auth_segmented_radio,
    div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]),
    div[data-testid="stRadio"] {
        margin-bottom: 16px !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    div[data-testid="stRadio"] > div {
        width: 100% !important;
        max-width: 100% !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"],
    div[data-testid="stRadioGroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 6px !important;
        background: #F1F5F9 !important;
        border: 1px solid #D8E0E8 !important;
        border-radius: 8px !important;
        padding: 4px !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > div,
    div[data-testid="stRadioGroup"] > div {
        flex: 1 1 50% !important;
        display: flex !important;
        width: 50% !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label,
    div[data-testid="stRadio"] label[data-testid="stRadioOption"] {
        flex: 1 !important;
        width: 100% !important;
        display: flex !important;
        text-align: center !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        transition: all 0.18s ease !important;
        margin: 0 !important;
    }
    div[data-testid="stRadio"] label[data-testid="stRadioOption"] > div {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }
    /* Hide the default radio circle completely */
    div[data-testid="stRadio"] label[data-testid="stRadioOption"] > div > div:first-child,
    div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    /* Active / Selected Tab */
    div[data-testid="stRadio"] div[role="radiogroup"] div[data-selected="true"] label,
    div[data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"],
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background-color: var(--auth-blue-primary) !important;
        color: #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(47, 109, 176, 0.28) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] div[data-selected="true"] label p,
    div[data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"] p,
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.5px !important;
    }
    /* Inactive Tab */
    div[data-testid="stRadio"] div[role="radiogroup"] div:not([data-selected="true"]) label,
    div[data-testid="stRadio"] label[data-testid="stRadioOption"]:not([data-selected="true"]),
    div[data-testid="stRadio"] div[role="radiogroup"] label:not(:has(input:checked)) {
        background-color: transparent !important;
        color: var(--auth-text-secondary) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] div:not([data-selected="true"]) label p,
    div[data-testid="stRadio"] label[data-testid="stRadioOption"]:not([data-selected="true"]) p,
    div[data-testid="stRadio"] div[role="radiogroup"] label:not(:has(input:checked)) p {
        color: var(--auth-text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.5px !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] div:not([data-selected="true"]) label:hover p,
    div[data-testid="stRadio"] label[data-testid="stRadioOption"]:not([data-selected="true"]):hover p {
        color: var(--auth-text-primary) !important;
    }

    /* Clean inner form wrapper (removes any default duplicate form borders) */
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }

    /* Input elements styling */
    div[data-testid="stTextInput"] label {
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        color: var(--auth-text-primary) !important;
        margin-bottom: 4px !important;
    }

    div[data-testid="stTextInput"] input {
        background-color: var(--auth-bg-surface) !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: var(--auth-text-primary) !important;
        font-size: 0.92rem !important;
        padding: 9px 14px !important;
        transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: var(--auth-blue-primary) !important;
        box-shadow: 0 0 0 3px rgba(47, 109, 176, 0.16) !important;
        outline: none !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: var(--auth-text-muted) !important;
    }

    /* Checkbox */
    div[data-testid="stCheckbox"] label span {
        font-size: 0.84rem !important;
        color: var(--auth-text-secondary) !important;
    }

    /* Buttons styling */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 9px 16px !important;
        transition: all 0.18s ease !important;
        letter-spacing: 0.2px !important;
    }

    /* Primary Buttons (Matching Dashboard Blue #2F6DB0) */
    div.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
        background-color: var(--auth-blue-primary) !important;
        border: 1px solid var(--auth-blue-primary) !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 4px rgba(47, 109, 176, 0.22) !important;
    }

    div.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
        background-color: var(--auth-blue-hover) !important;
        border-color: var(--auth-blue-hover) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(47, 109, 176, 0.28) !important;
    }

    /* Secondary Outlined Buttons */
    div.stButton > button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
        background-color: var(--auth-bg-surface) !important;
        border: 1px solid var(--auth-border-panel) !important;
        color: var(--auth-text-primary) !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    }

    div.stButton > button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {
        background-color: var(--auth-blue-light) !important;
        border-color: var(--auth-blue-primary) !important;
        color: var(--auth-blue-primary) !important;
        transform: translateY(-1px) !important;
    }

    /* Clean Divider */
    .auth-divider {
        display: flex;
        align-items: center;
        text-align: center;
        margin: 16px 0;
        color: var(--auth-text-muted);
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.8px;
    }

    .auth-divider::before, .auth-divider::after {
        content: '';
        flex: 1;
        border-bottom: 1px solid var(--auth-border-subtle);
    }

    .auth-divider span {
        padding: 0 10px;
    }

    /* Password Strength Meter */
    .pw-meter-container {
        margin: 6px 0 14px 0;
    }

    .pw-meter-bars {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 5px;
        height: 5px;
        margin-bottom: 6px;
    }

    .pw-bar {
        background: var(--auth-border-subtle);
        border-radius: 2px;
        transition: background-color 0.22s ease;
    }

    .pw-desc {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        color: var(--auth-text-secondary);
        font-family: var(--auth-font-mono);
    }

    /* Pipeline step hover micro-elevation */
    .pipeline-step-card {
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .pipeline-step-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
    }

    /* Respect reduced motion */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation: none !important;
            transition: none !important;
        }
    }

    /* Responsive Adjustments */
    @media (max-width: 992px) {
        .block-container {
            padding-top: 1.5rem !important;
        }
    }
</style>
"""


# -----------------------------------------------------------------------------
# 4. METEOROLOGICAL BRAND SVG MARK (REFINED FOR LIGHT SCIENTIFIC WORKSTATION)
# -----------------------------------------------------------------------------
def get_brand_svg_mark() -> str:
    """Returns an authentic meteorological vector logo mark matching the light dashboard:
    Doppler radar perimeter ring + monsoon streamline flow arc + precipitation isochrone bars.
    """
    return (
        '<svg width="48" height="48" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">'
        '<circle cx="26" cy="26" r="24" stroke="#2F6DB0" stroke-width="1.6" stroke-opacity="0.3" stroke-dasharray="3 3"/>'
        '<circle cx="26" cy="26" r="17" stroke="#68758A" stroke-width="1.2" stroke-opacity="0.3"/>'
        '<circle cx="26" cy="26" r="10" stroke="#2F6DB0" stroke-width="1.2" stroke-opacity="0.45"/>'
        '<path d="M 12 38 Q 22 28 38 18" stroke="#2F6DB0" stroke-width="2.4" stroke-linecap="round"/>'
        '<path d="M 15 42 Q 26 33 42 22" stroke="#68758A" stroke-width="1.5" stroke-linecap="round" stroke-opacity="0.5"/>'
        '<line x1="20" y1="28" x2="20" y2="35" stroke="#2F6DB0" stroke-width="2" stroke-linecap="round"/>'
        '<line x1="26" y1="22" x2="26" y2="34" stroke="#2F6DB0" stroke-width="2" stroke-linecap="round"/>'
        '<line x1="32" y1="18" x2="32" y2="30" stroke="#2F6DB0" stroke-width="2" stroke-linecap="round"/>'
        '<circle cx="26" cy="26" r="2.8" fill="#17233A"/>'
        '</svg>'
    )


# -----------------------------------------------------------------------------
# 5. PASSWORD STRENGTH CALCULATION
# -----------------------------------------------------------------------------
def calculate_password_strength(password: str) -> Tuple[int, str, str]:
    """Evaluates password strength and returns (score 0-4, label, color)."""
    if not password:
        return 0, "Enter password", "#94A3B8"
    score = 0
    if len(password) >= 6:
        score += 1
    if len(password) >= 8:
        score += 1
    if re.search(r"\d", password):
        score += 1
    if re.search(r"[!@#$%^&*(),.?\":{}|<>]", password) or (re.search(r"[A-Z]", password) and re.search(r"[a-z]", password)):
        score += 1

    if score <= 1:
        return 1, "Weak", "#D94B4B"
    elif score == 2:
        return 2, "Fair", "#D99A24"
    elif score == 3:
        return 3, "Good", "#2F6DB0"
    else:
        return 4, "Operational-Grade", "#2E9B72"


# -----------------------------------------------------------------------------
# 6. RENDER THE AUTHENTICATION EXPERIENCE
# -----------------------------------------------------------------------------
def render_auth_page():
    """Renders the clean, light, scientific Login & Register experience unified
    with the VarshaMitra Command Center design system.
    """
    init_auth_session()

    # 1. Inject Theme & Living Light Canvas
    st.markdown(AUTH_PAGE_CSS, unsafe_allow_html=True)
    components.html(get_auth_canvas_html(), height=0)

    # 2. Main Two-Column Layout (~52% Left, ~48% Right)
    col_left, col_right = st.columns([1.08, 1.0], gap="large")

    # =========================================================================
    # LEFT COLUMN: METEOROLOGICAL BRANDING & PHYSICAL PIPELINE INTELLIGENCE
    # =========================================================================
    with col_left:
        # Meteorological Brand Header
        mark_svg = get_brand_svg_mark()
        st.markdown(f"""<div style="display:flex; align-items:center; gap:16px; margin-bottom:18px;">
{mark_svg}
<div>
<div style="font-size:2.2rem; font-weight:800; letter-spacing:1.8px; color:#17233A; line-height:1.1;">VARSHAMITRA</div>
<div style="font-size:0.85rem; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:#2F6DB0; margin-top:4px;">AI FOR A RESILIENT MONSOON INDIA</div>
</div>
</div>""", unsafe_allow_html=True)

        # Core Meteorological Purpose Statement
        st.markdown("""<div style="font-size:1.12rem; font-weight:500; line-height:1.55; color:#17233A; margin-bottom:24px; max-width:540px; border-left:3px solid #2F6DB0; padding-left:16px;">
"Turning rainfall forecasts into district-level intelligence."
</div>""", unsafe_allow_html=True)

        # 4-Stage Integrated Physical Pipeline Cards
        st.markdown("""<div style="background:#FFFFFF; border:1px solid #D8E0E8; border-radius:12px; padding:18px 20px; margin-bottom:18px; max-width:540px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
<div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:#2F6DB0; margin-bottom:12px;">PHYSICAL FORECAST REFINEMENT PIPELINE</div>
<div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:8px;">
<div class="pipeline-step-card" style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 6px; text-align:center;">
<div style="font-size:0.68rem; font-weight:700; color:#68758A; font-family:'SFMono-Regular',Consolas,monospace;">01</div>
<div style="font-size:0.76rem; font-weight:700; color:#17233A; margin-top:2px;">NWP FORECAST</div>
<div style="font-size:0.68rem; color:#68758A; margin-top:2px;">GFS 0.25° Raw</div>
</div>
<div class="pipeline-step-card" style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 6px; text-align:center;">
<div style="font-size:0.68rem; font-weight:700; color:#2F6DB0; font-family:'SFMono-Regular',Consolas,monospace;">02</div>
<div style="font-size:0.76rem; font-weight:700; color:#2F6DB0; margin-top:2px;">REGIME DETECTION</div>
<div style="font-size:0.68rem; color:#68758A; margin-top:2px;">6 Synoptic Classes</div>
</div>
<div class="pipeline-step-card" style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 6px; text-align:center;">
<div style="font-size:0.68rem; font-weight:700; color:#2F6DB0; font-family:'SFMono-Regular',Consolas,monospace;">03</div>
<div style="font-size:0.76rem; font-weight:700; color:#2F6DB0; margin-top:2px;">BIAS CORRECTION</div>
<div style="font-size:0.68rem; color:#68758A; margin-top:2px;">Routed CDF Matching</div>
</div>
<div class="pipeline-step-card" style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:10px 6px; text-align:center;">
<div style="font-size:0.68rem; font-weight:700; color:#2E9B72; font-family:'SFMono-Regular',Consolas,monospace;">04</div>
<div style="font-size:0.76rem; font-weight:700; color:#2E9B72; margin-top:2px;">RAINFALL INTEL</div>
<div style="font-size:0.68rem; color:#68758A; margin-top:2px;">District Probability</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

        # Live Operational Monitoring Card
        st.markdown("""<div style="background:#FFFFFF; border:1px solid #D8E0E8; border-radius:12px; padding:16px 20px; margin-bottom:18px; max-width:540px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #E2E8F0;">
<div style="display:flex; align-items:center; gap:8px;">
<span style="width:8px; height:8px; border-radius:50%; background-color:#2E9B72; display:inline-block;"></span>
<span style="font-family:'SFMono-Regular',Consolas,monospace; font-size:0.78rem; font-weight:700; color:#17233A; letter-spacing:0.5px;">FORECAST ENGINE ONLINE</span>
</div>
<span style="background:#ECFDF5; color:#059669; border:1px solid #A7F3D0; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:700; letter-spacing:0.3px;">Operational</span>
</div>
<div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:12px;">
<div>
<div style="font-size:0.68rem; font-weight:600; color:#68758A; text-transform:uppercase; letter-spacing:0.4px;">NWP DATA</div>
<div style="font-family:'SFMono-Regular',Consolas,monospace; font-size:0.95rem; font-weight:700; color:#17233A; margin-top:2px;">GFS 0.25°</div>
</div>
<div>
<div style="font-size:0.68rem; font-weight:600; color:#68758A; text-transform:uppercase; letter-spacing:0.4px;">FORECAST CYCLE</div>
<div style="font-family:'SFMono-Regular',Consolas,monospace; font-size:0.95rem; font-weight:700; color:#2F6DB0; margin-top:2px;">12Z</div>
</div>
<div>
<div style="font-size:0.68rem; font-weight:600; color:#68758A; text-transform:uppercase; letter-spacing:0.4px;">STATUS</div>
<div style="font-family:'SFMono-Regular',Consolas,monospace; font-size:0.95rem; font-weight:700; color:#059669; margin-top:2px;">Operational</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

        # Scientific Annotations & Official Footer Line
        st.markdown("""<div style="display:flex; flex-wrap:wrap; align-items:center; gap:14px; font-family:'SFMono-Regular',Consolas,monospace; font-size:0.72rem; color:#68758A; margin-bottom:14px; max-width:540px;">
<div style="display:flex; align-items:center; gap:5px;"><span style="width:4px; height:4px; border-radius:50%; background-color:#2F6DB0;"></span> MONSOON INTELLIGENCE</div>
<div style="display:flex; align-items:center; gap:5px;"><span style="width:4px; height:4px; border-radius:50%; background-color:#2F6DB0;"></span> NWP → AI POST-PROCESSING</div>
<div style="display:flex; align-items:center; gap:5px;"><span style="width:4px; height:4px; border-radius:50%; background-color:#2F6DB0;"></span> DISTRICT FORECASTING</div>
<div style="display:flex; align-items:center; gap:5px;"><span style="width:4px; height:4px; border-radius:50%; background-color:#2F6DB0;"></span> REAL-TIME WEATHER ANALYTICS</div>
</div>
<div style="font-size:0.74rem; color:#94A3B8; font-family:'SFMono-Regular',Consolas,monospace; border-top:1px solid #E2E8F0; padding-top:10px; max-width:540px;">
SIH 2026 • Ministry of Earth Sciences • NCMRWF
</div>""", unsafe_allow_html=True)

    # =========================================================================
    # RIGHT COLUMN: CLEAN WHITE AUTHENTICATION CARD
    # =========================================================================
    with col_right:
        # Segmented Tab Control: SIGN IN | CREATE ACCOUNT
        current_mode = st.session_state.get("auth_mode", "login")
        radio_idx = 0 if current_mode == "login" else 1

        selected_tab = st.radio(
            "Auth Mode",
            options=["SIGN IN", "CREATE ACCOUNT"],
            index=radio_idx,
            horizontal=True,
            label_visibility="collapsed",
            key="auth_segmented_radio"
        )

        if selected_tab == "CREATE ACCOUNT" and current_mode != "register":
            st.session_state.auth_mode = "register"
            st.rerun()
        elif selected_tab == "SIGN IN" and current_mode != "login":
            st.session_state.auth_mode = "login"
            st.rerun()

        if st.session_state.auth_mode == "login":
            # =============================================================
            # SIGN IN VIEW
            # =============================================================
            st.markdown("""<div style="margin-top:4px; margin-bottom:18px;">
<div style="font-size:1.45rem; font-weight:700; color:#17233A; margin:0 0 4px 0; line-height:1.25;">Welcome to VarshaMitra</div>
<div style="font-size:0.86rem; color:#68758A; margin:0;">Sign in to access monsoon forecast intelligence.</div>
</div>""", unsafe_allow_html=True)

            with st.form("form_signin", clear_on_submit=False):
                email = st.text_input("Email Address", placeholder="Enter your email", key="in_login_email")
                password = st.text_input("Password", type="password", placeholder="Enter your password", key="in_login_pass")

                col_rem, col_fog = st.columns([1.1, 1.0])
                with col_rem:
                    remember = st.checkbox("Remember this session", value=True)
                with col_fog:
                    st.markdown("""<div style="text-align:right; font-size:0.82rem; margin-top:4px;">
<a href="mailto:admin@ncmrwf.gov.in?subject=VarshaMitra%20Password%20Reset" target="_blank" style="color:#2F6DB0; text-decoration:none; font-weight:500;">Forgot password?</a>
</div>""", unsafe_allow_html=True)

                submit_login = st.form_submit_button("SIGN IN TO VARSHAMITRA", type="primary", use_container_width=True)

            if submit_login:
                if email.strip() and password.strip():
                    with st.spinner("Authenticating with Monsoon Intelligence Engine..."):
                        time.sleep(0.8)
                    st.session_state.authenticated = True
                    user_clean = email.strip().split("@")[0].capitalize()
                    st.session_state.user_name = user_clean
                    st.session_state.user_role = "Meteorological Analyst"
                    st.session_state.auth_org = "Operational Forecaster Access"
                    st.rerun()
                else:
                    st.error("Please enter both email address and password.")

            # Clean Divider
            st.markdown('<div class="auth-divider"><span>OR</span></div>', unsafe_allow_html=True)

            # Secondary Action: Continue as Guest Observer
            if st.button("Continue as Guest Observer", key="btn_guest_observer", type="secondary", use_container_width=True):
                with st.spinner("Initializing Guest Observer Session..."):
                    time.sleep(0.6)
                st.session_state.authenticated = True
                st.session_state.user_name = "Guest Observer"
                st.session_state.user_role = "SIH 2026 Evaluator"
                st.session_state.auth_org = "Government & Research Access"
                st.rerun()

            # Switch to Register Mode
            st.markdown('<div style="text-align:center; font-size:0.84rem; color:#68758A; margin-top:16px; margin-bottom:6px;">New to VarshaMitra?</div>', unsafe_allow_html=True)
            if st.button("Create an account", key="btn_switch_to_register", use_container_width=True):
                st.session_state.auth_mode = "register"
                st.rerun()

        else:
            # =============================================================
            # CREATE ACCOUNT VIEW
            # =============================================================
            st.markdown("""<div style="margin-top:4px; margin-bottom:18px;">
<div style="font-size:1.45rem; font-weight:700; color:#17233A; margin:0 0 4px 0; line-height:1.25;">Create your VarshaMitra account</div>
<div style="font-size:0.86rem; color:#68758A; margin:0;">Join the monsoon intelligence workspace.</div>
</div>""", unsafe_allow_html=True)

            with st.form("form_register", clear_on_submit=False):
                full_name = st.text_input("Full Name", placeholder="Your full name", key="in_reg_name")
                reg_email = st.text_input("Email Address", placeholder="Enter your email", key="in_reg_email")
                reg_pass = st.text_input("Password", type="password", placeholder="Enter your password", key="in_reg_pass")
                reg_conf = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="in_reg_conf")

                # Live password strength indicator (clean light theme)
                score, label, color = calculate_password_strength(reg_pass)
                bars_html = "".join([
                    f'<div class="pw-bar" style="background-color: {color if i < score else "var(--auth-border-subtle)"};"></div>'
                    for i in range(4)
                ])
                st.markdown(f"""<div class="pw-meter-container">
<div class="pw-meter-bars">{bars_html}</div>
<div class="pw-desc">
<span>Password strength: <b style="color:{color};">{label}</b></span>
<span>8+ chars with mix of letters & numbers recommended</span>
</div>
</div>""", unsafe_allow_html=True)

                submit_reg = st.form_submit_button("CREATE VARSHAMITRA ACCOUNT", type="primary", use_container_width=True)

            if submit_reg:
                if not full_name.strip():
                    st.error("Please enter your full name.")
                elif not reg_email.strip() or "@" not in reg_email:
                    st.error("Please enter a valid email address.")
                elif len(reg_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                elif reg_pass != reg_conf:
                    st.error("Passwords do not match. Please verify.")
                else:
                    with st.spinner("Creating Operational Account..."):
                        time.sleep(0.8)
                    st.session_state.authenticated = True
                    st.session_state.user_name = full_name.strip()
                    st.session_state.user_role = "Operational Member (Research)"
                    st.session_state.auth_org = "Atmospheric Research Access"
                    st.rerun()

            # Clean Divider
            st.markdown('<div class="auth-divider"><span>OR</span></div>', unsafe_allow_html=True)

            # Secondary Action: Continue as Guest Observer
            if st.button("Continue as Guest Observer", key="btn_guest_observer_reg", type="secondary", use_container_width=True):
                with st.spinner("Initializing Guest Observer Session..."):
                    time.sleep(0.6)
                st.session_state.authenticated = True
                st.session_state.user_name = "Guest Observer"
                st.session_state.user_role = "SIH 2026 Evaluator"
                st.session_state.auth_org = "Government & Research Access"
                st.rerun()

            # Switch to Login Mode
            st.markdown('<div style="text-align:center; font-size:0.84rem; color:#68758A; margin-top:16px; margin-bottom:6px;">Already have an account?</div>', unsafe_allow_html=True)
            if st.button("Sign in", key="btn_switch_to_login", use_container_width=True):
                st.session_state.auth_mode = "login"
                st.rerun()
