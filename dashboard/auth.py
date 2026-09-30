"""VARSHAMITRA | AUTHENTICATION & LOGIN EXPERIENCE
==================================================
Operational Meteorological Analysis & Decision-Support System
Smart India Hackathon 2026 (NCMRWF / Ministry of Earth Sciences)
==================================================
Cinematic Monsoon Intelligence Authentication Experience.
"""

import re
from typing import Tuple
import streamlit as st
import streamlit.components.v1 as components

# -----------------------------------------------------------------------------
# 1. INITIALIZE AUTHENTICATION SESSION STATE
# -----------------------------------------------------------------------------
def init_auth_session():
    """Initializes auth state in session_state, checking for URL bypass params."""
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
    rainfall particles, radar sweep, SW monsoon streamline flow, isobars, and faint India map.
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

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  // Rain particle system
  const raindrops = [];
  const RAIN_COUNT = 175;
  for (let i = 0; i < RAIN_COUNT; i++) {
    raindrops.push({
      x: Math.random() * (window.innerWidth + 200) - 100,
      y: Math.random() * window.innerHeight,
      len: 12 + Math.random() * 18,
      speed: 8 + Math.random() * 9,
      opacity: 0.15 + Math.random() * 0.28,
      dx: 2.2 + Math.random() * 1.6
    });
  }

  // Southwest Monsoon wind streamline particles
  const streamlines = [];
  const STREAMLINE_COUNT = 28;
  for (let i = 0; i < STREAMLINE_COUNT; i++) {
    streamlines.push({
      startX: (Math.random() * 0.38) * window.innerWidth,
      startY: (0.58 + Math.random() * 0.42) * window.innerHeight,
      progress: Math.random(),
      speed: 0.0018 + Math.random() * 0.0028,
      length: 110 + Math.random() * 120,
      opacity: 0.12 + Math.random() * 0.24
    });
  }

  // Radar circular sweep & pulse
  let radarAngle = 0;
  let pulseRadius = 0;
  let flashAlpha = 0;
  let lastFlash = Date.now();

  function drawIndiaMap(w, h) {
    const cx = w * 0.28;
    const cy = h * 0.52;
    const s = Math.min(w, h) * 0.68;

    ctx.save();
    ctx.strokeStyle = "rgba(47, 167, 216, 0.16)";
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
    ctx.strokeStyle = "rgba(99, 197, 232, 0.34)";
    ctx.lineWidth = 2.4;
    ctx.stroke();

    // Western Ghats label
    ctx.font = "8px 'SFMono-Regular', Consolas, monospace";
    ctx.fillStyle = "rgba(99, 197, 232, 0.45)";
    ctx.fillText("WESTERN GHATS OROGRAPHIC RIDGE", cx - 0.26 * s, cy + 0.13 * s);

    // Maharashtra Focus Halo
    ctx.fillStyle = "rgba(47, 167, 216, 0.05)";
    ctx.beginPath();
    ctx.arc(cx - 0.08 * s, cy + 0.05 * s, 0.085 * s, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(47, 167, 216, 0.28)";
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
      ctx.fillStyle = st.active ? "#63C5E8" : "rgba(155, 174, 194, 0.45)";
      ctx.fill();

      if (st.active) {
        ctx.beginPath();
        ctx.arc(st.x, st.y, 6 + (Math.sin(Date.now() * 0.003) * 2), 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(99, 197, 232, 0.35)";
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }

      ctx.font = "8px 'SFMono-Regular', Consolas, monospace";
      ctx.fillStyle = "rgba(155, 174, 194, 0.6)";
      ctx.fillText(st.name, st.x + 6, st.y + 3);
    });

    ctx.restore();
  }

  function drawAtmosphericIsobars(w, h) {
    ctx.save();
    ctx.strokeStyle = "rgba(47, 167, 216, 0.08)";
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
      ctx.fillStyle = "rgba(99, 197, 232, 0.22)";
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
      ctx.strokeStyle = "rgba(47, 167, 216, 0.05)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    pulseRadius += 0.85;
    if (pulseRadius > maxR) pulseRadius = 12;
    ctx.beginPath();
    ctx.arc(rx, ry, pulseRadius, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(47, 167, 216, ${0.18 * (1 - pulseRadius / maxR)})`;
    ctx.lineWidth = 1.1;
    ctx.stroke();

    radarAngle += 0.011;
    const grad = ctx.createRadialGradient(rx, ry, 10, rx, ry, maxR);
    grad.addColorStop(0, "rgba(99, 197, 232, 0.12)");
    grad.addColorStop(1, "rgba(47, 167, 216, 0.0)");

    ctx.beginPath();
    ctx.moveTo(rx, ry);
    ctx.arc(rx, ry, maxR, radarAngle - 0.28, radarAngle);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

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
      ctx.strokeStyle = `rgba(99, 197, 232, ${line.opacity * (1 - line.progress)})`;
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
    bgGrad.addColorStop(0, "#050B14");
    bgGrad.addColorStop(0.5, "#081424");
    bgGrad.addColorStop(1, "#0A182A");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    // Subtle atmospheric flash
    if (Date.now() - lastFlash > 14000 && Math.random() < 0.015) {
      flashAlpha = 0.05;
      lastFlash = Date.now();
    }
    if (flashAlpha > 0.001) {
      ctx.fillStyle = `rgba(130, 190, 240, ${flashAlpha})`;
      ctx.fillRect(0, 0, width, height);
      flashAlpha *= 0.93;
    }

    drawAtmosphericIsobars(width, height);
    drawRadarSweep(width, height);
    drawIndiaMap(width, height);
    drawStreamlines(width, height);
    drawRain(width, height);

    requestAnimationFrame(animate);
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
    /* Reset & Deep Midnight Atmospheric Theme */
    :root {
        --auth-bg-main: #07111F;
        --auth-bg-card: rgba(11, 24, 42, 0.86);
        --auth-border: rgba(255, 255, 255, 0.14);
        --auth-border-subtle: rgba(255, 255, 255, 0.08);
        --auth-text-primary: #F5F7FA;
        --auth-text-secondary: #9BAEC2;
        --auth-text-muted: #64748B;
        --auth-accent-blue: #2FA7D8;
        --auth-accent-hover: #1E8EC0;
        --auth-accent-cyan: #63C5E8;
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
        padding-top: 3.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1400px !important;
    }

    /* Style the right-column card container natively */
    [data-testid="column"]:nth-of-type(3) div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--auth-bg-card) !important;
        border: 1px solid var(--auth-border) !important;
        border-radius: 20px !important;
        padding: 28px 32px 32px 32px !important;
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.52), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(24px) !important;
    }

    /* Input elements styling */
    div[data-testid="stTextInput"] label {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #E2E8F0 !important;
        margin-bottom: 4px !important;
    }

    div[data-testid="stTextInput"] input {
        background-color: rgba(14, 27, 46, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
        font-size: 0.9rem !important;
        padding: 10px 14px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: var(--auth-accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(47, 167, 216, 0.22) !important;
        outline: none !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #64748B !important;
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
        font-size: 0.9rem !important;
        padding: 9px 16px !important;
        transition: all 0.2s ease !important;
    }

    /* Primary Buttons */
    div.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
        background-color: var(--auth-accent-blue) !important;
        border: 1px solid var(--auth-accent-blue) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(47, 167, 216, 0.28) !important;
    }

    div.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
        background-color: var(--auth-accent-hover) !important;
        border-color: var(--auth-accent-hover) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(47, 167, 216, 0.4) !important;
    }

    /* Secondary Buttons */
    div.stButton > button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        color: var(--auth-text-primary) !important;
    }

    div.stButton > button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
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
    Doppler radar concentric ring + monsoon streamline flow vectors + precipitation isochrone bars.
    """
    return (
        '<svg width="50" height="50" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">'
        '<circle cx="26" cy="26" r="24" stroke="#2FA7D8" stroke-width="1.6" stroke-opacity="0.4" stroke-dasharray="3 3"/>'
        '<circle cx="26" cy="26" r="17" stroke="#63C5E8" stroke-width="1.2" stroke-opacity="0.3"/>'
        '<circle cx="26" cy="26" r="10" stroke="#2FA7D8" stroke-width="1" stroke-opacity="0.5"/>'
        '<path d="M 12 38 Q 22 28 38 18" stroke="#63C5E8" stroke-width="2.2" stroke-linecap="round"/>'
        '<path d="M 15 42 Q 26 33 42 22" stroke="#2FA7D8" stroke-width="1.5" stroke-linecap="round" stroke-opacity="0.7"/>'
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
        return 0, "Enter password", "#64748B"
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
        return 1, "Weak", "#E56B6F"
    elif score == 2:
        return 2, "Fair", "#F2B84B"
    elif score == 3:
        return 3, "Good", "#2FA7D8"
    else:
        return 4, "Operational-Grade", "#35B879"


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
        st.markdown(f"""<div style="display:flex; align-items:center; gap:16px; margin-bottom:20px;">
{mark_svg}
<div>
<div style="font-size:2.2rem; font-weight:800; letter-spacing:2px; color:#F5F7FA; line-height:1.1;">VARSHAMITRA</div>
<div style="font-size:0.85rem; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; color:#63C5E8; margin-top:4px;">AI FOR A RESILIENT MONSOON INDIA</div>
</div>
</div>""", unsafe_allow_html=True)

        # Scientific Statement
        st.markdown("""<div style="font-size:1.15rem; font-weight:400; line-height:1.55; color:#CBD5E1; margin-bottom:28px; max-width:520px; border-left:2px solid #2FA7D8; padding-left:16px;">
"Regime-aware rainfall intelligence for a more resilient India."
</div>""", unsafe_allow_html=True)

        # Synoptic Regime Live Preview Card
        st.markdown("""<div style="background:rgba(16, 32, 54, 0.72); border:1px solid rgba(255,255,255,0.12); border-radius:14px; padding:20px 24px; margin-bottom:28px; backdrop-filter:blur(16px); box-shadow:0 10px 30px rgba(0,0,0,0.35); max-width:540px;">
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; padding-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.08);">
<div style="display:flex; align-items:center; gap:8px;">
<span style="width:8px; height:8px; border-radius:50%; background-color:#35B879; display:inline-block; box-shadow:0 0 8px rgba(53,184,121,0.6);"></span>
<span style="font-family:'SFMono-Regular',Consolas,monospace; font-size:0.76rem; font-weight:600; color:#9BAEC2; letter-spacing:0.6px;">SYNOPTIC METEOROLOGY • GFS 0.25°</span>
</div>
<span style="background:rgba(53,184,121,0.15); color:#35B879; border:1px solid rgba(53,184,121,0.35); border-radius:4px; padding:3px 9px; font-size:0.74rem; font-weight:600; letter-spacing:0.4px;">ACTIVE MONSOON</span>
</div>
<div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:14px; margin-bottom:14px;">
<div>
<div style="font-size:0.7rem; font-weight:600; color:#9BAEC2; text-transform:uppercase; letter-spacing:0.4px;">ESTIMATED PRECIPITATION</div>
<div style="font-family:'SFMono-Regular',Consolas,monospace; font-size:1.28rem; font-weight:700; color:#F5F7FA; margin-top:4px;">19.9 <span style="font-size:0.76rem; font-weight:400; color:#9BAEC2;">mm/day</span></div>
</div>
<div>
<div style="font-size:0.7rem; font-weight:600; color:#9BAEC2; text-transform:uppercase; letter-spacing:0.4px;">ENSEMBLE CONFIDENCE</div>
<div style="font-family:'SFMono-Regular',Consolas,monospace; font-size:1.28rem; font-weight:700; color:#F5F7FA; margin-top:4px;">82<span style="font-size:0.76rem; font-weight:400; color:#9BAEC2;">%</span></div>
</div>
<div>
<div style="font-size:0.7rem; font-weight:600; color:#9BAEC2; text-transform:uppercase; letter-spacing:0.4px;">DOMINANT REGIME</div>
<div style="font-family:'SFMono-Regular',Consolas,monospace; font-size:1.15rem; font-weight:700; color:#63C5E8; margin-top:4px;">Active Monsoon</div>
</div>
</div>
<div style="display:flex; justify-content:space-between; align-items:center; font-size:0.74rem; color:#64748B; font-family:'SFMono-Regular',Consolas,monospace; margin-top:8px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.08);">
<span>MAHARASHTRA • 36 DISTRICTS • OROGRAPHIC RIDGE</span>
<span>RUN: 00Z OPERATIONAL</span>
</div>
</div>""", unsafe_allow_html=True)

        # Unobtrusive Status Metadata Strip
        st.markdown("""<div style="display:flex; flex-wrap:wrap; align-items:center; gap:16px; font-family:'SFMono-Regular',Consolas,monospace; font-size:0.76rem; color:#9BAEC2; max-width:540px;">
<div style="display:flex; align-items:center; gap:6px;"><span style="width:5px; height:5px; border-radius:50%; background-color:#63C5E8;"></span> MONSOON INTELLIGENCE</div>
<div style="display:flex; align-items:center; gap:6px;"><span style="width:5px; height:5px; border-radius:50%; background-color:#63C5E8;"></span> REGIME-AWARE FORECASTING</div>
<div style="display:flex; align-items:center; gap:6px;"><span style="width:5px; height:5px; border-radius:50%; background-color:#63C5E8;"></span> GFS 0.25° NWP</div>
<div style="display:flex; align-items:center; gap:6px;"><span style="width:5px; height:5px; border-radius:50%; background-color:#63C5E8;"></span> DISTRICT QUANTILE MAPPING</div>
</div>""", unsafe_allow_html=True)

    # =========================================================================
    # RIGHT COLUMN: PREMIUM GLASS-FROSTED AUTHENTICATION PANEL
    # =========================================================================
    with col_right:
        with st.container(border=True):
            # Mode Tab Switcher Header (Pill indicators)
            col_tab1, col_tab2 = st.columns([1, 1])
            with col_tab1:
                if st.button("Sign In", key="pill_tab_signin", type="primary" if st.session_state.auth_mode == "login" else "secondary", use_container_width=True):
                    st.session_state.auth_mode = "login"
                    st.rerun()
            with col_tab2:
                if st.button("Create Account", key="pill_tab_register", type="primary" if st.session_state.auth_mode == "register" else "secondary", use_container_width=True):
                    st.session_state.auth_mode = "register"
                    st.rerun()

            if st.session_state.auth_mode == "login":
                st.markdown("""<div style="margin-top:8px; margin-bottom:18px;">
<div style="font-size:1.45rem; font-weight:700; color:#F5F7FA; margin:0 0 4px 0; line-height:1.25;">Welcome back</div>
<div style="font-size:0.86rem; color:#9BAEC2; margin:0;">Continue to VarshaMitra meteorological workspace</div>
</div>""", unsafe_allow_html=True)

                # LOGIN FORM
                with st.form("form_login", clear_on_submit=False):
                    email = st.text_input("Email address", placeholder="meteorologist@ncmrwf.gov.in", key="in_login_email")
                    password = st.text_input("Password", type="password", placeholder="••••••••", key="in_login_pass")

                    col_rem, col_fog = st.columns([1, 1])
                    with col_rem:
                        remember = st.checkbox("Remember me", value=True)
                    with col_fog:
                        st.markdown("""<div style="text-align:right; font-size:0.82rem; margin-top:4px;">
<a href="mailto:admin@ncmrwf.gov.in?subject=VarshaMitra%20Password%20Reset" target="_blank" style="color:#63C5E8; text-decoration:none;">Forgot password?</a>
</div>""", unsafe_allow_html=True)

                    submit_login = st.form_submit_button("Sign in →", type="primary", use_container_width=True)

                if submit_login:
                    if email.strip() and password.strip():
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
                if st.button("Continue as Guest Observer →", key="btn_guest_observer", type="secondary", use_container_width=True):
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
                # REGISTER FORM
                st.markdown("""<div style="margin-top:8px; margin-bottom:18px;">
<div style="font-size:1.45rem; font-weight:700; color:#F5F7FA; margin:0 0 4px 0; line-height:1.25;">Create your VarshaMitra account</div>
<div style="font-size:0.86rem; color:#9BAEC2; margin:0;">Join the monsoon intelligence workspace.</div>
</div>""", unsafe_allow_html=True)

                with st.form("form_register", clear_on_submit=False):
                    full_name = st.text_input("Full name", placeholder="Dr. Rajesh Kulkarni", key="in_reg_name")
                    reg_email = st.text_input("Email address", placeholder="r.kulkarni@imd.gov.in", key="in_reg_email")
                    reg_pass = st.text_input("Password", type="password", placeholder="••••••••", key="in_reg_pass")
                    reg_conf = st.text_input("Confirm password", type="password", placeholder="••••••••", key="in_reg_conf")
                    reg_org = st.text_input("Organization / Institution (Optional)", placeholder="e.g. IMD Pune / IITM / State DMA", key="in_reg_org")

                    # Password Strength preview based on current input
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

                    submit_reg = st.form_submit_button("Create account →", type="primary", use_container_width=True)

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
                        st.session_state.authenticated = True
                        st.session_state.user_name = full_name.strip()
                        org_str = reg_org.strip() if reg_org.strip() else "Atmospheric Research"
                        st.session_state.user_role = f"Operational Member ({org_str})"
                        st.session_state.auth_org = org_str
                        st.rerun()

                # Divider
                st.markdown('<div class="auth-divider"><span>OR</span></div>', unsafe_allow_html=True)

                # Secondary Action: Continue as Guest Observer
                if st.button("Continue as Guest Observer →", key="btn_guest_observer_reg", type="secondary", use_container_width=True):
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
