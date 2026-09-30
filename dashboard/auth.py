"""VARSHAMITRA | AUTHENTICATION & LOGIN EXPERIENCE
==================================================
Operational Meteorological Analysis & Decision-Support System
Smart India Hackathon 2026 (NCMRWF / Ministry of Earth Sciences)
==================================================
Cinematic Monsoon Intelligence Authentication Gateway.
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
# 2. LIVING ATMOSPHERIC CANVAS COMPONENT (HTML5 / JS)
# -----------------------------------------------------------------------------
def get_auth_canvas_html() -> str:
    """Returns self-contained HTML5 Canvas animation of Indian monsoon meteorology:
    subtle rainfall particles, slow radar sweep, SW monsoon streamline flow, isobars, and faint India map.
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

  // Subtle rainfall particles
  const raindrops = [];
  const RAIN_COUNT = 150;
  for (let i = 0; i < RAIN_COUNT; i++) {
    raindrops.push({
      x: Math.random() * (window.innerWidth + 200) - 100,
      y: Math.random() * window.innerHeight,
      len: 10 + Math.random() * 16,
      speed: 7 + Math.random() * 7,
      opacity: 0.12 + Math.random() * 0.22,
      dx: 2.0 + Math.random() * 1.4
    });
  }

  // Southwest Monsoon wind streamline particles
  const streamlines = [];
  const STREAMLINE_COUNT = 24;
  for (let i = 0; i < STREAMLINE_COUNT; i++) {
    streamlines.push({
      startX: (Math.random() * 0.38) * window.innerWidth,
      startY: (0.58 + Math.random() * 0.42) * window.innerHeight,
      progress: Math.random(),
      speed: 0.0016 + Math.random() * 0.0024,
      length: 100 + Math.random() * 110,
      opacity: 0.10 + Math.random() * 0.22
    });
  }

  // Radar sweep & pulse
  let radarAngle = 0;
  let pulseRadius = 0;
  let flashAlpha = 0;
  let lastFlash = Date.now();

  function drawIndiaMap(w, h) {
    const cx = w * 0.28;
    const cy = h * 0.52;
    const s = Math.min(w, h) * 0.68;

    ctx.save();
    ctx.strokeStyle = "rgba(59, 167, 216, 0.15)";
    ctx.lineWidth = 1.3;
    ctx.setLineDash([4, 4]);

    // Subcontinent polygon outline
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
    ctx.strokeStyle = "rgba(113, 199, 232, 0.32)";
    ctx.lineWidth = 2.4;
    ctx.stroke();

    // Western Ghats annotation
    ctx.font = "8px 'SFMono-Regular', Consolas, monospace";
    ctx.fillStyle = "rgba(113, 199, 232, 0.40)";
    ctx.fillText("WESTERN GHATS RIDGE", cx - 0.24 * s, cy + 0.13 * s);

    // Maharashtra Focus Region
    ctx.fillStyle = "rgba(59, 167, 216, 0.04)";
    ctx.beginPath();
    ctx.arc(cx - 0.08 * s, cy + 0.05 * s, 0.085 * s, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(59, 167, 216, 0.24)";
    ctx.lineWidth = 1;
    ctx.stroke();

    // Synoptic Stations
    const stations = [
      { name: "PUNE (HQ / IMD)", x: cx - 0.08 * s, y: cy + 0.05 * s, active: true },
      { name: "MUMBAI RADAR", x: cx - 0.13 * s, y: cy + 0.03 * s, active: true },
      { name: "KOLHAPUR", x: cx - 0.07 * s, y: cy + 0.13 * s },
      { name: "NAGPUR DWR", x: cx + 0.05 * s, y: cy + 0.01 * s },
      { name: "NCMRWF NOIDA", x: cx - 0.04 * s, y: cy - 0.24 * s },
      { name: "CHENNAI", x: cx + 0.08 * s, y: cy + 0.24 * s }
    ];

    stations.forEach(st => {
      ctx.beginPath();
      ctx.arc(st.x, st.y, st.active ? 3 : 2, 0, Math.PI * 2);
      ctx.fillStyle = st.active ? "#71C7E8" : "rgba(155, 174, 194, 0.45)";
      ctx.fill();

      if (st.active && !prefersReducedMotion) {
        ctx.beginPath();
        ctx.arc(st.x, st.y, 6 + (Math.sin(Date.now() * 0.0028) * 1.8), 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(113, 199, 232, 0.32)";
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }

      ctx.font = "8px 'SFMono-Regular', Consolas, monospace";
      ctx.fillStyle = "rgba(155, 174, 194, 0.55)";
      ctx.fillText(st.name, st.x + 6, st.y + 3);
    });

    ctx.restore();
  }

  function drawAtmosphericIsobars(w, h) {
    ctx.save();
    ctx.strokeStyle = "rgba(59, 167, 216, 0.07)";
    ctx.lineWidth = 1;

    const isobarLabels = ["1004 hPa", "1008 hPa", "1012 hPa"];
    for (let i = 0; i < 3; i++) {
      ctx.beginPath();
      const offset = i * 65;
      ctx.moveTo(-60, h * 0.72 - offset);
      ctx.bezierCurveTo(
        w * 0.18, h * 0.62 - offset,
        w * 0.32, h * 0.76 - offset,
        w * 0.58, h * 0.48 - offset
      );
      ctx.stroke();

      ctx.font = "8px 'SFMono-Regular', Consolas, monospace";
      ctx.fillStyle = "rgba(113, 199, 232, 0.20)";
      ctx.fillText(isobarLabels[i], w * 0.19, h * 0.64 - offset);
    }
    ctx.restore();
  }

  function drawRadarSweep(w, h) {
    const rx = w * 0.25;
    const ry = h * 0.54;
    const maxR = Math.min(w, h) * 0.38;

    ctx.save();
    for (let r = 45; r <= maxR; r += 55) {
      ctx.beginPath();
      ctx.arc(rx, ry, r, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(59, 167, 216, 0.05)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    if (!prefersReducedMotion) {
      pulseRadius += 0.75;
      if (pulseRadius > maxR) pulseRadius = 12;
      ctx.beginPath();
      ctx.arc(rx, ry, pulseRadius, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(59, 167, 216, ${0.16 * (1 - pulseRadius / maxR)})`;
      ctx.lineWidth = 1.1;
      ctx.stroke();

      radarAngle += 0.009;
      const grad = ctx.createRadialGradient(rx, ry, 10, rx, ry, maxR);
      grad.addColorStop(0, "rgba(113, 199, 232, 0.10)");
      grad.addColorStop(1, "rgba(59, 167, 216, 0.0)");

      ctx.beginPath();
      ctx.moveTo(rx, ry);
      ctx.arc(rx, ry, maxR, radarAngle - 0.26, radarAngle);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
    }

    ctx.restore();
  }

  function drawStreamlines(w, h) {
    ctx.save();
    ctx.lineWidth = 1.2;

    streamlines.forEach(line => {
      line.progress += line.speed;
      if (line.progress > 1) {
        line.progress = 0;
        line.startX = (Math.random() * 0.38) * w;
        line.startY = (0.58 + Math.random() * 0.42) * h;
      }

      const curDist = line.progress * line.length;
      const x1 = line.startX + curDist * 1.35;
      const y1 = line.startY - curDist * 0.82 - Math.sin(line.progress * Math.PI) * 22;
      const x2 = x1 + 13;
      const y2 = y1 - 8;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = `rgba(113, 199, 232, ${line.opacity * (1 - line.progress)})`;
      ctx.stroke();
    });
    ctx.restore();
  }

  function drawRain(w, h) {
    ctx.save();
    ctx.lineWidth = 1.1;

    raindrops.forEach(drop => {
      drop.x += drop.dx;
      drop.y += drop.speed;

      if (drop.y > h || drop.x > w) {
        drop.x = Math.random() * (w + 200) - 150;
        drop.y = -25;
      }

      ctx.beginPath();
      ctx.moveTo(drop.x, drop.y);
      ctx.lineTo(drop.x + drop.dx * (drop.len / drop.speed), drop.y + drop.len);
      ctx.strokeStyle = `rgba(165, 218, 245, ${drop.opacity})`;
      ctx.stroke();
    });
    ctx.restore();
  }

  function animate() {
    const bgGrad = ctx.createLinearGradient(0, 0, width, height);
    bgGrad.addColorStop(0, "#06111F");
    bgGrad.addColorStop(0.5, "#0A1728");
    bgGrad.addColorStop(1, "#0D1B2E");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    if (!prefersReducedMotion) {
      if (Date.now() - lastFlash > 15000 && Math.random() < 0.015) {
        flashAlpha = 0.045;
        lastFlash = Date.now();
      }
      if (flashAlpha > 0.001) {
        ctx.fillStyle = `rgba(113, 199, 232, ${flashAlpha})`;
        ctx.fillRect(0, 0, width, height);
        flashAlpha *= 0.94;
      }
    }

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
# 3. AUTHENTICATION PAGE CSS STYLING
# -----------------------------------------------------------------------------
AUTH_PAGE_CSS = """
<style>
    /* Exact Palette per Specification */
    :root {
        --auth-bg-main: #06111F;
        --auth-bg-secondary: #0A1728;
        --auth-bg-panel: #0D1B2E;
        --auth-border-panel: #24384D;
        --auth-border-subtle: rgba(36, 56, 77, 0.65);
        --auth-blue-primary: #3BA7D8;
        --auth-blue-hover: #2B90C0;
        --auth-blue-light: #71C7E8;
        --auth-text-primary: #F4F7FA;
        --auth-text-secondary: #9BAEC2;
        --auth-text-muted: #65798E;
        --auth-success: #3BBF8A;
        --auth-warning: #E8B24A;
        --auth-danger: #DF7078;
        --auth-font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        --auth-font-mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }

    /* Fullscreen app styling when unauthenticated */
    .stApp {
        background-color: var(--auth-bg-main) !important;
        font-family: var(--auth-font-sans) !important;
        color: var(--auth-text-primary) !important;
    }

    /* Hide standard sidebar and default header elements on auth page */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
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

    /* Container constraints */
    .block-container {
        position: relative !important;
        z-index: 10 !important;
        padding-top: 2.8rem !important;
        padding-bottom: 2rem !important;
        max-width: 1440px !important;
    }

    /* Style the right-column card container natively */
    [data-testid="column"]:nth-of-type(3) div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--auth-bg-panel) !important;
        border: 1px solid var(--auth-border-panel) !important;
        border-radius: 12px !important;
        padding: 26px 30px 30px 30px !important;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45) !important;
        backdrop-filter: blur(14px) !important;
    }

    /* Input elements styling */
    div[data-testid="stTextInput"] label {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #E2E8F0 !important;
        margin-bottom: 4px !important;
    }

    div[data-testid="stTextInput"] input {
        background-color: var(--auth-bg-secondary) !important;
        border: 1px solid var(--auth-border-panel) !important;
        border-radius: 8px !important;
        color: var(--auth-text-primary) !important;
        font-size: 0.9rem !important;
        padding: 9px 14px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: var(--auth-blue-primary) !important;
        box-shadow: 0 0 0 3px rgba(59, 167, 216, 0.22) !important;
        outline: none !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: var(--auth-text-muted) !important;
    }

    /* Checkbox */
    div[data-testid="stCheckbox"] label span {
        font-size: 0.82rem !important;
        color: var(--auth-text-secondary) !important;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 9px 16px !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.3px !important;
    }

    /* Primary Buttons (with subtle highlight) */
    div.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
        background-color: var(--auth-blue-primary) !important;
        border: 1px solid var(--auth-blue-primary) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(59, 167, 216, 0.28) !important;
    }

    div.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
        background-color: var(--auth-blue-hover) !important;
        border-color: var(--auth-blue-hover) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(59, 167, 216, 0.40) !important;
    }

    /* Secondary Buttons */
    div.stButton > button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
        background-color: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid var(--auth-border-panel) !important;
        color: var(--auth-text-primary) !important;
    }

    div.stButton > button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(113, 199, 232, 0.4) !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
    }

    /* Divider */
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
        margin: 4px 0 14px 0;
    }

    .pw-meter-bars {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 5px;
        height: 4px;
        margin-bottom: 5px;
    }

    .pw-bar {
        background: rgba(255, 255, 255, 0.12);
        border-radius: 2px;
        transition: background-color 0.25s ease;
    }

    .pw-desc {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        color: var(--auth-text-muted);
        font-family: var(--auth-font-mono);
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
# 4. METEOROLOGICAL BRAND SVG MARK
# -----------------------------------------------------------------------------
def get_brand_svg_mark() -> str:
    """Returns an authentic, non-generic meteorological vector logo mark:
    Doppler radar concentric perimeter ring + monsoon streamline flow vector arc + precipitation isochrone bars.
    """
    return (
        '<svg width="50" height="50" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">'
        '<circle cx="26" cy="26" r="24" stroke="#3BA7D8" stroke-width="1.6" stroke-opacity="0.4" stroke-dasharray="3 3"/>'
        '<circle cx="26" cy="26" r="17" stroke="#71C7E8" stroke-width="1.2" stroke-opacity="0.3"/>'
        '<circle cx="26" cy="26" r="10" stroke="#3BA7D8" stroke-width="1" stroke-opacity="0.5"/>'
        '<path d="M 12 38 Q 22 28 38 18" stroke="#71C7E8" stroke-width="2.2" stroke-linecap="round"/>'
        '<path d="M 15 42 Q 26 33 42 22" stroke="#3BA7D8" stroke-width="1.5" stroke-linecap="round" stroke-opacity="0.7"/>'
        '<line x1="20" y1="28" x2="20" y2="35" stroke="#93E0FB" stroke-width="2" stroke-linecap="round"/>'
        '<line x1="26" y1="22" x2="26" y2="34" stroke="#93E0FB" stroke-width="2" stroke-linecap="round"/>'
        '<line x1="32" y1="18" x2="32" y2="30" stroke="#93E0FB" stroke-width="2" stroke-linecap="round"/>'
        '<circle cx="26" cy="26" r="2.5" fill="#FFFFFF"/>'
        '</svg>'
    )


# -----------------------------------------------------------------------------
# 5. PASSWORD STRENGTH CALCULATION
# -----------------------------------------------------------------------------
def calculate_password_strength(password: str) -> Tuple[int, str, str]:
    """Evaluates password strength and returns (score 0-4, label, color)."""
    if not password:
        return 0, "Enter password", "#65798E"
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
        return 1, "Weak", "#DF7078"
    elif score == 2:
        return 2, "Fair", "#E8B24A"
    elif score == 3:
        return 3, "Good", "#3BA7D8"
    else:
        return 4, "Operational-Grade", "#3BBF8A"


# -----------------------------------------------------------------------------
# 6. RENDER THE AUTHENTICATION EXPERIENCE
# -----------------------------------------------------------------------------
def render_auth_page():
    """Renders the complete 16:9 split-screen atmospheric login & register experience."""
    init_auth_session()

    # 1. Inject Theme & Background Canvas
    st.markdown(AUTH_PAGE_CSS, unsafe_allow_html=True)
    components.html(get_auth_canvas_html(), height=0)

    # 2. Main 16:9 Split Layout (Left ~55%, Right ~45%)
    col_left, col_spacer, col_right = st.columns([1.25, 0.08, 1.0], gap="medium")

    # =========================================================================
    # LEFT COLUMN: BRANDING & ATMOSPHERIC MONSOON INTELLIGENCE
    # =========================================================================
    with col_left:
        # Meteorological Mark + Header
        mark_svg = get_brand_svg_mark()
        st.markdown(f"""<div style="display:flex; align-items:center; gap:16px; margin-bottom:18px;">
{mark_svg}
<div>
<div style="font-size:2.2rem; font-weight:800; letter-spacing:2px; color:#F4F7FA; line-height:1.1;">VARSHAMITRA</div>
<div style="font-size:0.85rem; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; color:#71C7E8; margin-top:4px;">AI FOR A RESILIENT MONSOON INDIA</div>
</div>
</div>""", unsafe_allow_html=True)

        # Core Meteorological Purpose Statement
        st.markdown("""<div style="font-size:1.12rem; font-weight:400; line-height:1.55; color:#E2E8F0; margin-bottom:24px; max-width:520px; border-left:2px solid #3BA7D8; padding-left:16px;">
"Turning rainfall forecasts into district-level intelligence."
</div>""", unsafe_allow_html=True)

        # 4-Stage Integrated Pipeline Flow Representation
        st.markdown("""<div style="background:rgba(10, 23, 40, 0.75); border:1px solid #24384D; border-radius:10px; padding:16px 20px; margin-bottom:20px; max-width:540px;">
<div style="font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:#71C7E8; margin-bottom:12px;">PHYSICAL FORECAST REFINEMENT PIPELINE</div>
<div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:8px; position:relative;">
<div style="background:rgba(13, 27, 46, 0.9); border:1px solid rgba(59, 167, 216, 0.25); border-radius:6px; padding:8px 6px; text-align:center;">
<div style="font-size:0.64rem; font-weight:700; color:#9BAEC2; font-family:'SFMono-Regular',Consolas,monospace;">STAGE 1</div>
<div style="font-size:0.75rem; font-weight:700; color:#F4F7FA; margin-top:2px;">NWP FORECAST</div>
<div style="font-size:0.65rem; color:#65798E; margin-top:2px;">GFS 0.25° Raw</div>
</div>
<div style="background:rgba(13, 27, 46, 0.9); border:1px solid rgba(59, 167, 216, 0.25); border-radius:6px; padding:8px 6px; text-align:center;">
<div style="font-size:0.64rem; font-weight:700; color:#9BAEC2; font-family:'SFMono-Regular',Consolas,monospace;">STAGE 2</div>
<div style="font-size:0.75rem; font-weight:700; color:#71C7E8; margin-top:2px;">REGIME DETECTION</div>
<div style="font-size:0.65rem; color:#65798E; margin-top:2px;">6 Synoptic Classes</div>
</div>
<div style="background:rgba(13, 27, 46, 0.9); border:1px solid rgba(59, 167, 216, 0.25); border-radius:6px; padding:8px 6px; text-align:center;">
<div style="font-size:0.64rem; font-weight:700; color:#9BAEC2; font-family:'SFMono-Regular',Consolas,monospace;">STAGE 3</div>
<div style="font-size:0.75rem; font-weight:700; color:#3BA7D8; margin-top:2px;">BIAS CORRECTION</div>
<div style="font-size:0.65rem; color:#65798E; margin-top:2px;">Routed CDF Matching</div>
</div>
<div style="background:rgba(13, 27, 46, 0.9); border:1px solid rgba(59, 184, 138, 0.4); border-radius:6px; padding:8px 6px; text-align:center;">
<div style="font-size:0.64rem; font-weight:700; color:#3BBF8A; font-family:'SFMono-Regular',Consolas,monospace;">STAGE 4</div>
<div style="font-size:0.75rem; font-weight:700; color:#3BBF8A; margin-top:2px;">RAINFALL INTEL</div>
<div style="font-size:0.65rem; color:#65798E; margin-top:2px;">District Probability</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

        # Live Operational Status Block
        st.markdown("""<div style="background:rgba(10, 23, 40, 0.75); border:1px solid #24384D; border-radius:10px; padding:14px 20px; margin-bottom:20px; max-width:540px;">
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid rgba(36, 56, 77, 0.65);">
<div style="display:flex; align-items:center; gap:8px;">
<span style="width:7px; height:7px; border-radius:50%; background-color:#3BBF8A; display:inline-block; box-shadow:0 0 8px rgba(59, 191, 138, 0.6);"></span>
<span style="font-family:'SFMono-Regular',Consolas,monospace; font-size:0.78rem; font-weight:700; color:#F4F7FA; letter-spacing:0.6px;">FORECAST ENGINE ONLINE</span>
</div>
<span style="background:rgba(59, 191, 138, 0.15); color:#3BBF8A; border:1px solid rgba(59, 191, 138, 0.35); border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:700; letter-spacing:0.4px;">Operational</span>
</div>
<div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:12px;">
<div>
<div style="font-size:0.68rem; font-weight:600; color:#9BAEC2; text-transform:uppercase; letter-spacing:0.4px;">NWP DATA</div>
<div style="font-family:'SFMono-Regular',Consolas,monospace; font-size:0.95rem; font-weight:700; color:#F4F7FA; margin-top:2px;">GFS 0.25°</div>
</div>
<div>
<div style="font-size:0.68rem; font-weight:600; color:#9BAEC2; text-transform:uppercase; letter-spacing:0.4px;">FORECAST CYCLE</div>
<div style="font-family:'SFMono-Regular',Consolas,monospace; font-size:0.95rem; font-weight:700; color:#71C7E8; margin-top:2px;">12Z</div>
</div>
<div>
<div style="font-size:0.68rem; font-weight:600; color:#9BAEC2; text-transform:uppercase; letter-spacing:0.4px;">STATUS</div>
<div style="font-family:'SFMono-Regular',Consolas,monospace; font-size:0.95rem; font-weight:700; color:#3BBF8A; margin-top:2px;">Operational</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

        # Subtle Scientific Annotations & Footer Line
        st.markdown("""<div style="display:flex; flex-wrap:wrap; align-items:center; gap:14px; font-family:'SFMono-Regular',Consolas,monospace; font-size:0.74rem; color:#9BAEC2; margin-bottom:14px; max-width:540px;">
<div style="display:flex; align-items:center; gap:5px;"><span style="width:4px; height:4px; border-radius:50%; background-color:#71C7E8;"></span> MONSOON INTELLIGENCE</div>
<div style="display:flex; align-items:center; gap:5px;"><span style="width:4px; height:4px; border-radius:50%; background-color:#71C7E8;"></span> NWP → AI POST-PROCESSING</div>
<div style="display:flex; align-items:center; gap:5px;"><span style="width:4px; height:4px; border-radius:50%; background-color:#71C7E8;"></span> DISTRICT FORECASTING</div>
<div style="display:flex; align-items:center; gap:5px;"><span style="width:4px; height:4px; border-radius:50%; background-color:#71C7E8;"></span> REAL-TIME WEATHER ANALYTICS</div>
</div>
<div style="font-size:0.74rem; color:#65798E; font-family:'SFMono-Regular',Consolas,monospace; border-top:1px solid rgba(36, 56, 77, 0.45); padding-top:8px; max-width:540px;">
SIH 2026 • Ministry of Earth Sciences • NCMRWF
</div>""", unsafe_allow_html=True)

    # =========================================================================
    # RIGHT COLUMN: REFINED METEOROLOGICAL OPERATIONS LOGIN CONSOLE
    # =========================================================================
    with col_right:
        with st.container(border=True):
            # Mode Segmented Toggle: SIGN IN | CREATE ACCOUNT
            col_tab1, col_tab2 = st.columns([1, 1])
            with col_tab1:
                if st.button("SIGN IN", key="btn_toggle_signin", type="primary" if st.session_state.auth_mode == "login" else "secondary", use_container_width=True):
                    st.session_state.auth_mode = "login"
                    st.rerun()
            with col_tab2:
                if st.button("CREATE ACCOUNT", key="btn_toggle_register", type="primary" if st.session_state.auth_mode == "register" else "secondary", use_container_width=True):
                    st.session_state.auth_mode = "register"
                    st.rerun()

            if st.session_state.auth_mode == "login":
                # SIGN IN VIEW
                st.markdown("""<div style="margin-top:8px; margin-bottom:16px;">
<div style="font-size:1.45rem; font-weight:700; color:#F4F7FA; margin:0 0 3px 0; line-height:1.25;">Welcome to VarshaMitra</div>
<div style="font-size:0.86rem; color:#9BAEC2; margin:0;">Sign in to access monsoon forecast intelligence.</div>
</div>""", unsafe_allow_html=True)

                with st.form("form_signin", clear_on_submit=False):
                    email = st.text_input("Email Address", placeholder="Enter your email", key="in_login_email")
                    password = st.text_input("Password", type="password", placeholder="Enter your password", key="in_login_pass")

                    col_rem, col_fog = st.columns([1.1, 1.0])
                    with col_rem:
                        remember = st.checkbox("Remember this session", value=True)
                    with col_fog:
                        st.markdown("""<div style="text-align:right; font-size:0.82rem; margin-top:4px;">
<a href="mailto:admin@ncmrwf.gov.in?subject=VarshaMitra%20Password%20Reset" target="_blank" style="color:#71C7E8; text-decoration:none;">Forgot password?</a>
</div>""", unsafe_allow_html=True)

                    submit_login = st.form_submit_button("SIGN IN TO VARSHAMITRA", type="primary", use_container_width=True)

                if submit_login:
                    if email.strip() and password.strip():
                        # Cinematic initialization transition under 1.5s
                        with st.spinner("Initializing Forecast Intelligence..."):
                            time.sleep(1.0)
                        st.session_state.authenticated = True
                        user_clean = email.strip().split("@")[0].capitalize()
                        st.session_state.user_name = user_clean
                        st.session_state.user_role = "Meteorological Analyst"
                        st.session_state.auth_org = "Operational Forecaster Access"
                        st.rerun()
                    else:
                        st.error("Please enter both email address and password.")

                # Divider
                st.markdown('<div class="auth-divider"><span>OR</span></div>', unsafe_allow_html=True)

                # Secondary Action: Continue as Guest Observer
                if st.button("Continue as Guest Observer", key="btn_guest_observer", type="secondary", use_container_width=True):
                    with st.spinner("Initializing Forecast Intelligence..."):
                        time.sleep(0.8)
                    st.session_state.authenticated = True
                    st.session_state.user_name = "Guest Observer"
                    st.session_state.user_role = "SIH 2026 Evaluator"
                    st.session_state.auth_org = "Government & Research Access"
                    st.rerun()

                # Switch to Register Mode
                st.markdown('<div style="text-align:center; font-size:0.84rem; color:#9BAEC2; margin-top:14px; margin-bottom:4px;">New to VarshaMitra?</div>', unsafe_allow_html=True)
                if st.button("Create an account", key="btn_switch_to_register", use_container_width=True):
                    st.session_state.auth_mode = "register"
                    st.rerun()

            else:
                # CREATE ACCOUNT VIEW
                st.markdown("""<div style="margin-top:8px; margin-bottom:16px;">
<div style="font-size:1.45rem; font-weight:700; color:#F4F7FA; margin:0 0 3px 0; line-height:1.25;">Create your VarshaMitra account</div>
<div style="font-size:0.86rem; color:#9BAEC2; margin:0;">Join the monsoon intelligence workspace.</div>
</div>""", unsafe_allow_html=True)

                with st.form("form_register", clear_on_submit=False):
                    full_name = st.text_input("Full Name", placeholder="Your full name", key="in_reg_name")
                    reg_email = st.text_input("Email Address", placeholder="Enter your email", key="in_reg_email")
                    reg_pass = st.text_input("Password", type="password", placeholder="Enter your password", key="in_reg_pass")
                    reg_conf = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="in_reg_conf")

                    # Live password strength indicator
                    score, label, color = calculate_password_strength(reg_pass)
                    bars_html = "".join([
                        f'<div class="pw-bar" style="background-color: {color if i < score else "rgba(255,255,255,0.12)"};"></div>'
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
                        with st.spinner("Initializing Forecast Intelligence..."):
                            time.sleep(1.0)
                        st.session_state.authenticated = True
                        st.session_state.user_name = full_name.strip()
                        st.session_state.user_role = "Operational Member (Research)"
                        st.session_state.auth_org = "Atmospheric Research Access"
                        st.rerun()

                # Divider
                st.markdown('<div class="auth-divider"><span>OR</span></div>', unsafe_allow_html=True)

                # Secondary Action: Continue as Guest Observer
                if st.button("Continue as Guest Observer", key="btn_guest_observer_reg", type="secondary", use_container_width=True):
                    with st.spinner("Initializing Forecast Intelligence..."):
                        time.sleep(0.8)
                    st.session_state.authenticated = True
                    st.session_state.user_name = "Guest Observer"
                    st.session_state.user_role = "SIH 2026 Evaluator"
                    st.session_state.auth_org = "Government & Research Access"
                    st.rerun()

                # Switch to Login Mode
                st.markdown('<div style="text-align:center; font-size:0.84rem; color:#9BAEC2; margin-top:14px; margin-bottom:4px;">Already have an account?</div>', unsafe_allow_html=True)
                if st.button("Sign in", key="btn_switch_to_login", use_container_width=True):
                    st.session_state.auth_mode = "login"
                    st.rerun()
