"""
app.py  —  Oblique Aerial Survey Planner v3
============================================
Per-camera table, portrait/landscape, tilt-axis selector.
Three diagram views: footprint plan (all frames), cross-section, multi-strip.

Geometry verified against Oblique_setup9_working_2.xls — all values match.

Spreadsheet convention notes
─────────────────────────────
The Landscape sheet uses portrait-mounted L/R cameras (narrow axis across-track).
Our geometry.py orientation='portrait' reproduces those values exactly.
The Portrait sheet tab uses landscape-mounted oblique cameras (long axis across-track)
at a different flying height — it represents a different physical rig configuration.

Run:
    streamlit run app.py
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st

from geometry import (
    normalize_tilt_angle,
    half_fov_deg,
    diag_pp_to_long_edge_mm,
    flying_height_for_gsd,
    calculate_camera_solution,
    calculate_multicamera_solution,
    m_to_unit,
    unit_to_m,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CAM_COLOURS = ["#58a6ff", "#f85149", "#3fb950", "#d29922", "#bc8cff", "#39c5cf"]

BODY_PRESETS = {
    "Sony A7R IV":       {"w_mm": 35.7,    "h_mm": 23.8,    "w_px": 9504,  "h_px": 6336},
    "Sony A7R V":        {"w_mm": 36.1152, "h_mm": 24.0768, "w_px": 9504,  "h_px": 6336},
    "Sony A6500":        {"w_mm": 23.5,    "h_mm": 15.6,    "w_px": 6000,  "h_px": 4000},
    "Phase One iXM-100": {"w_mm": 53.4,    "h_mm": 40.0,    "w_px": 11664, "h_px": 8750},
    "Canon 5DS R":       {"w_mm": 36.0,    "h_mm": 24.0,    "w_px": 8688,  "h_px": 5792},
    "Nikon D850":        {"w_mm": 35.9,    "h_mm": 23.9,    "w_px": 8256,  "h_px": 5504},
    "Canon 760D":        {"w_mm": 22.3,    "h_mm": 14.9,    "w_px": 6000,  "h_px": 4000},
}

# Default 3-camera rig matching Oblique_setup9_working_2.xls Landscape sheet exactly
DEFAULT_CAMERAS = [
    {"enabled": True,  "label": "Nadir",        "body": "Sony A7R V", "focal_mm": 21.0,
     "tilt_deg": 0.0,  "tilt_conv": "horiz", "orientation": "landscape", "tilt_axis": "across"},
    {"enabled": True,  "label": "Right oblique", "body": "Sony A7R V", "focal_mm": 50.0,
     "tilt_deg": 50.0, "tilt_conv": "horiz", "orientation": "portrait",  "tilt_axis": "across"},
    {"enabled": True,  "label": "Left oblique",  "body": "Sony A7R V", "focal_mm": 50.0,
     "tilt_deg": 50.0, "tilt_conv": "horiz", "orientation": "portrait",  "tilt_axis": "across"},
    {"enabled": False, "label": "Fore oblique",  "body": "Sony A7R V", "focal_mm": 50.0,
     "tilt_deg": 50.0, "tilt_conv": "horiz", "orientation": "portrait",  "tilt_axis": "along"},
    {"enabled": False, "label": "Aft oblique",   "body": "Sony A7R V", "focal_mm": 50.0,
     "tilt_deg": 50.0, "tilt_conv": "horiz", "orientation": "portrait",  "tilt_axis": "along"},
]

PRESET_FILE = Path("presets.json")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Oblique Survey Planner", page_icon="✈️", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_saved_bodies():
    if PRESET_FILE.exists():
        try:
            return json.loads(PRESET_FILE.read_text())
        except Exception:
            pass
    return {}

def save_body_preset(name, data):
    existing = load_saved_bodies()
    existing[name] = data
    PRESET_FILE.write_text(json.dumps(existing, indent=2))

def save_scenario(data, path):
    Path(path).write_text(json.dumps(data, indent=2, default=str))

def load_scenario(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None

def all_body_names():
    return list(BODY_PRESETS.keys()) + list(load_saved_bodies().keys())

def get_body(name):
    saved = load_saved_bodies()
    if name in BODY_PRESETS:
        return BODY_PRESETS[name]
    if name in saved:
        d = saved[name]
        return {"w_mm": d["w_mm"], "h_mm": d["h_mm"], "w_px": d["w_px"], "h_px": d["h_px"]}
    return BODY_PRESETS["Sony A7R V"]

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

if "cameras" not in st.session_state:
    st.session_state.cameras = [dict(c) for c in DEFAULT_CAMERAS]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — geometry and display
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v_m, unit, d=1):
    return f"{m_to_unit(v_m, unit):.{d}f} {unit}"

def fmt_gsd(v_m, d=2):
    return f"{v_m * 100:.{d}f} cm/px"

def obliqueness_ratio(sol):
    """
    Ratio of far-edge GSD to inner-edge GSD.  1.0 = nadir (no variation).
    Higher = more oblique — GSD varies more across the image.
    """
    inner_gsd = min(sol.near_gsd_m, sol.far_gsd_m)
    outer_gsd = max(sol.near_gsd_m, sol.far_gsd_m)
    if inner_gsd <= 0:
        return float("inf")
    return outer_gsd / inner_gsd

def dark_fig(w=12, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    return fig, ax

def corner_inner_outer(sol):
    """
    Return (inner_gx_or_gy, outer_gx_or_gy) where inner = edge closer to nadir.
    Works for both positive-tilt (right) and negative-tilt (left) cameras.
    """
    a, b = sol.near_edge_m, sol.far_edge_m
    return (a, b) if abs(a) <= abs(b) else (b, a)

def inner_outer_corners(sol):
    """
    Split the 4 footprint corners into (inner_top, inner_bot) and (outer_top, outer_bot).
    Inner = nadir side (smaller |Gx|), outer = far side (larger |Gx|).
    """
    nt, nb = sol.corner_near_top, sol.corner_near_bot
    ft, fb = sol.corner_far_top,  sol.corner_far_bot
    if abs(nt[0]) <= abs(ft[0]):
        return (nt, nb), (ft, fb)
    else:
        return (ft, fb), (nt, nb)

def along_lengths_for_display(sol):
    """Inner and outer along-track (or across-track for along-tilt) footprint lengths."""
    (it, ib), (ot, ob) = inner_outer_corners(sol)
    return abs(it[1] - ib[1]), abs(ot[1] - ob[1])

def safe_corners(sol):
    """Return all 4 corner (Gx, Gy) tuples, filtered to finite values only."""
    raw = [sol.corner_near_top, sol.corner_far_top,
           sol.corner_near_bot, sol.corner_far_bot]
    return [c for c in raw if np.isfinite(c[0]) and np.isfinite(c[1])]

def axis_limits_from_solutions(solutions, pad=0.18):
    """
    Compute square axis limits that fit all camera footprints with a margin.
    Returns (lim,) where axes run from -lim to +lim.
    Falls back to 1.0 if no finite corners found.
    """
    all_x, all_y = [], []
    for _, sol, _ in solutions:
        for c in safe_corners(sol):
            all_x.append(c[0])
            all_y.append(c[1])
    if not all_x:
        return 1.0
    x_span = max(abs(min(all_x)), abs(max(all_x))) * (1 + pad)
    y_span = max(abs(min(all_y)), abs(max(all_y))) * (1 + pad)
    lim = max(x_span, y_span)
    return lim if np.isfinite(lim) and lim > 0 else 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("✈️ Oblique Survey Planner")
st.sidebar.markdown("---")
st.sidebar.subheader("Flight Parameters")

dist_unit = st.sidebar.selectbox("Distance unit", ["m", "ft"], index=0)

alt_input = st.sidebar.number_input(
    f"Altitude AGL ({dist_unit})",
    value=round(m_to_unit(470.0, dist_unit), 1),
    min_value=1.0, max_value=m_to_unit(10000.0, dist_unit),
    step=m_to_unit(10.0, dist_unit),
)
altitude_m = unit_to_m(alt_input, dist_unit)

speed_ms = st.sidebar.number_input(
    "Aircraft speed (m/s)", value=50.0, min_value=1.0, max_value=300.0, step=1.0
)
st.sidebar.caption(f"≈ {speed_ms * 1.94384:.1f} kts  |  {speed_ms * 3.6:.1f} km/h")
reciprocal = st.sidebar.checkbox("Reciprocal (bidirectional) strips", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Overlap Targets")
fwd_pct  = st.sidebar.slider("Forward overlap (%)", 10, 95, 60)
side_pct = st.sidebar.slider("Sidelap (%)",          10, 95, 30)
fwd_frac  = fwd_pct  / 100.0
side_frac = side_pct / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("💾 Save / Load")
sc_name = st.sidebar.text_input("Scenario name", "my_survey")
if st.sidebar.button("Save scenario"):
    save_scenario({
        "cameras": st.session_state.cameras,
        "altitude_m": altitude_m, "speed_ms": speed_ms,
        "fwd_overlap_pct": fwd_pct, "sidelap_pct": side_pct, "reciprocal": reciprocal,
    }, f"{sc_name}.json")
    st.sidebar.success(f"Saved {sc_name}.json")

load_name = st.sidebar.text_input("Load scenario filename", "")
if st.sidebar.button("Load scenario"):
    sc = load_scenario(load_name)
    if sc and "cameras" in sc:
        st.session_state.cameras = sc["cameras"]
        st.sidebar.success(f"Loaded {load_name}")
        st.rerun()
    else:
        st.sidebar.error(f"Could not load '{load_name}'")

# ─────────────────────────────────────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────────────────────────────────────

st.title("✈️ Oblique Aerial Survey Planner")
st.caption(
    f"Altitude: **{fmt(altitude_m, dist_unit)}**  |  "
    f"Speed: **{speed_ms:.0f} m/s ({speed_ms * 1.94384:.0f} kts)**  |  "
    f"Fwd overlap: **{fwd_pct}%**  |  Sidelap: **{side_pct}%**"
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Camera configuration table
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Camera Configuration")

with st.expander("Orientation & tilt-axis reference", expanded=False):
    st.markdown("""
| Setting | Option | Sensor across-track | Best for |
|---|---|---|---|
| **Orientation** | Portrait | **Narrow** (short) axis | L/R oblique — limits far-edge GSD stretch |
| | Landscape | **Long** axis | Nadir — maximises swath width |
| **Tilt axis** | Across (L/R) | Tilts left/right about along-track axis | Left & Right oblique |
| | Along (F/A) | Tilts fore/aft about across-track axis | Fore & Aft oblique |

**Spreadsheet match:** The Landscape sheet in the reference spreadsheet uses portrait-mounted
L/R cameras (narrow axis across-track). The default settings here reproduce those values exactly.
For the Left camera, enter the same tilt angle as Right — the mirroring is handled automatically.
    """)

cameras      = st.session_state.cameras
bodies_avail = all_body_names()

# Column headers
hdr = st.columns([0.35, 1.5, 1.7, 0.7, 0.65, 0.8, 0.85, 1.0, 0.3])
for col, lbl in zip(hdr, ["✓", "Label", "Body", "FL mm", "Tilt °", "From", "Orient.", "Tilt axis", ""]):
    col.markdown(f"<span style='color:#8b949e;font-size:0.78em;font-weight:600'>{lbl}</span>",
                 unsafe_allow_html=True)

to_delete = None
for i, cam in enumerate(cameras):
    c_en, c_lbl, c_body, c_fl, c_tilt, c_conv, c_orient, c_axis, c_del = \
        st.columns([0.35, 1.5, 1.7, 0.7, 0.65, 0.8, 0.85, 1.0, 0.3])
    colour = CAM_COLOURS[i % len(CAM_COLOURS)]

    cam["enabled"] = c_en.checkbox("##e", value=cam["enabled"], key=f"en_{i}",
                                    label_visibility="collapsed")
    c_lbl.markdown(f"<span style='color:{colour}'>●</span>", unsafe_allow_html=True)
    cam["label"] = c_lbl.text_input("##l", value=cam["label"], key=f"lbl_{i}",
                                     label_visibility="collapsed")
    body_idx = bodies_avail.index(cam["body"]) if cam["body"] in bodies_avail else 0
    cam["body"] = c_body.selectbox("##b", bodies_avail, index=body_idx, key=f"body_{i}",
                                    label_visibility="collapsed")
    cam["focal_mm"] = float(c_fl.number_input("##f", value=float(cam["focal_mm"]),
                                               min_value=1.0, max_value=2000.0, step=1.0,
                                               key=f"fl_{i}", label_visibility="collapsed"))
    cam["tilt_deg"] = float(c_tilt.number_input("##t", value=float(cam["tilt_deg"]),
                                                  min_value=0.0, max_value=85.0, step=0.5,
                                                  key=f"tilt_{i}", label_visibility="collapsed"))
    cam["tilt_conv"] = c_conv.selectbox("##c", ["horiz", "nadir"], key=f"conv_{i}",
                                         index=0 if cam["tilt_conv"] == "horiz" else 1,
                                         label_visibility="collapsed",
                                         format_func=lambda x: "Horizontal" if x == "horiz" else "Nadir")
    cam["orientation"] = c_orient.selectbox("##o", ["portrait", "landscape"], key=f"orient_{i}",
                                             index=0 if cam["orientation"] == "portrait" else 1,
                                             label_visibility="collapsed",
                                             format_func=lambda x: "Portrait" if x == "portrait" else "Landscape")
    cam["tilt_axis"] = c_axis.selectbox("##a", ["across", "along"], key=f"axis_{i}",
                                         index=0 if cam["tilt_axis"] == "across" else 1,
                                         label_visibility="collapsed",
                                         format_func=lambda x: "Across (L/R)" if x == "across" else "Along (F/A)")
    if c_del.button("✕", key=f"del_{i}", help="Remove camera"):
        to_delete = i

if to_delete is not None:
    cameras.pop(to_delete)
    st.rerun()

b1, b2, b3 = st.columns([1, 1, 6])
if b1.button("➕ Add camera"):
    cameras.append({"enabled": True, "label": f"Camera {len(cameras)+1}",
                    "body": "Sony A7R V", "focal_mm": 50.0, "tilt_deg": 50.0,
                    "tilt_conv": "horiz", "orientation": "portrait", "tilt_axis": "across"})
    st.rerun()
if b2.button("↺ Reset defaults"):
    st.session_state.cameras = [dict(c) for c in DEFAULT_CAMERAS]
    st.rerun()

with st.expander("Save a camera body as a custom preset"):
    pc1, pc2 = st.columns([2, 1])
    p_base = pc1.selectbox("Base body", list(BODY_PRESETS.keys()), key="p_base")
    p_name = pc2.text_input("Preset name", value=p_base, key="p_name")
    p_wmm  = pc1.number_input("Sensor long axis mm", value=BODY_PRESETS[p_base]["w_mm"], key="p_wmm")
    p_hmm  = pc1.number_input("Sensor short axis mm", value=BODY_PRESETS[p_base]["h_mm"], key="p_hmm")
    p_wpx  = pc1.number_input("Pixel count long axis", value=BODY_PRESETS[p_base]["w_px"], step=100, key="p_wpx")
    p_hpx  = pc1.number_input("Pixel count short axis", value=BODY_PRESETS[p_base]["h_px"], step=100, key="p_hpx")
    if pc2.button("Save preset"):
        save_body_preset(p_name, {"w_mm": p_wmm, "h_mm": p_hmm,
                                   "w_px": int(p_wpx), "h_px": int(p_hpx)})
        st.success(f"Saved '{p_name}'. Reload page to use it.")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Build camera solutions
# ─────────────────────────────────────────────────────────────────────────────

active   = [c for c in cameras if c["enabled"]]
solutions = []   # list of (cam_dict, CameraSolution, colour_str)
errors    = []

for i, cam in enumerate(active):
    try:
        bd     = get_body(cam["body"])
        tilt_n = normalize_tilt_angle(cam["tilt_deg"], cam["tilt_conv"])
        sol    = calculate_camera_solution(
            altitude_m          = altitude_m,
            tilt_from_nadir_deg = tilt_n,
            sensor_w_native_mm  = bd["w_mm"],
            sensor_h_native_mm  = bd["h_mm"],
            image_w_native_px   = bd["w_px"],
            image_h_native_px   = bd["h_px"],
            focal_length_mm     = cam["focal_mm"],
            orientation         = cam["orientation"],
            tilt_axis           = cam["tilt_axis"],
            label               = cam["label"],
        )
        solutions.append((cam, sol, CAM_COLOURS[i % len(CAM_COLOURS)]))
    except Exception as e:
        errors.append(f"**{cam['label']}**: {e}")

for e in errors:
    st.error(e)

if not solutions:
    st.warning("No cameras enabled. Enable at least one camera in the table above.")
    st.stop()

sol_list = [s for _, s, _ in solutions]

mc = None
try:
    mc = calculate_multicamera_solution(
        camera_solutions      = sol_list,
        arrangement           = "custom",
        altitude_m            = altitude_m,
        aircraft_speed_ms     = speed_ms,
        forward_overlap_fraction = fwd_frac,
        sidelap_fraction      = side_frac,
        reciprocal_flying     = reciprocal,
    )
except Exception as e:
    st.error(f"System calculation error: {e}")

if mc and mc.warnings:
    with st.expander("⚠️ Warnings", expanded=True):
        for w in mc.warnings:
            st.warning(w)

# ─────────────────────────────────────────────────────────────────────────────
# Summary cards
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("System Summary")
if mc:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Combined swath",    fmt(mc.combined_swath_m,             dist_unit))
    c2.metric("Line spacing",      fmt(mc.recommended_line_spacing_m,  dist_unit))
    c3.metric("Photo spacing",     fmt(mc.recommended_photo_spacing_m, dist_unit))
    c4.metric("Exposure interval", f"{mc.photo_interval_s:.2f} s")
    c5.metric("Sidelap achieved",  f"{mc.sidelap_achieved * 100:.1f}%")

    rep = next((s for _, s, _ in solutions if abs(s.tilt_from_nadir_deg) > 1), sol_list[0])
    c6, c7, c8, c9 = st.columns(4)
    c6.metric(f"GSD inner ({rep.label})", fmt_gsd(min(rep.near_gsd_m, rep.far_gsd_m)))
    c7.metric("GSD centre",               fmt_gsd(rep.centre_gsd_m))
    c8.metric("GSD outer",                fmt_gsd(max(rep.near_gsd_m, rep.far_gsd_m)))
    c9.metric("Fwd overlap near/ctr/far",
              f"{mc.forward_overlap_near*100:.0f}% / "
              f"{mc.forward_overlap_centre*100:.0f}% / "
              f"{mc.forward_overlap_far*100:.0f}%")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Per-camera results table  (includes obliqueness ratio)
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Per-Camera Results")

rows = []
for cam, sol, _ in solutions:
    inner_gx, outer_gx = corner_inner_outer(sol)
    inner_len, outer_len = along_lengths_for_display(sol)
    inner_gsd = min(sol.near_gsd_m, sol.far_gsd_m)
    outer_gsd = max(sol.near_gsd_m, sol.far_gsd_m)
    tilt_h = 90.0 - sol.tilt_from_nadir_deg

    rows.append({
        "Camera":                    sol.label,
        "Body / FL":                 f"{cam['body']}  {cam['focal_mm']:.0f} mm",
        "Orient.":                   sol.orientation,
        "Tilt axis":                 sol.tilt_axis,
        "Tilt nadir °":              round(sol.tilt_from_nadir_deg, 1),
        "Tilt horiz °":              round(tilt_h, 1),
        # ── Obliqueness ──
        "Obliqueness ratio":         round(obliqueness_ratio(sol), 2),
        # (far GSD / inner GSD — 1.0 = nadir; higher = more oblique)
        # ──────────────────
        "Pixel µm":                  round(sol.pixel_size_mm * 1000, 2),
        "FOV across °":              round(sol.full_fov_across_deg, 2),
        "FOV along °":               round(sol.full_fov_along_deg,  2),
        f"Inner edge ({dist_unit})": round(m_to_unit(abs(inner_gx), dist_unit), 1),
        f"Outer edge ({dist_unit})": round(m_to_unit(abs(outer_gx), dist_unit), 1),
        f"Inner length ({dist_unit})": round(m_to_unit(inner_len, dist_unit), 1),
        f"Outer length ({dist_unit})": round(m_to_unit(outer_len, dist_unit), 1),
        "Inner GSD cm":              round(inner_gsd * 100, 3),
        "Centre GSD cm":             round(sol.centre_gsd_m * 100, 3),
        "Outer GSD cm":              round(outer_gsd * 100, 3),
        "Inner slant m":             round(min(sol.near_slant_m, sol.far_slant_m), 1),
        "Outer slant m":             round(max(sol.near_slant_m, sol.far_slant_m), 1),
        "Diag image mm":             round(sol.diag_image_mm, 4),
    })

st.dataframe(rows, use_container_width=True)
st.caption(
    "**Obliqueness ratio** = outer GSD ÷ inner GSD. "
    "1.0 = nadir (uniform); higher values mean greater GSD variation across the frame."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 1 — Footprint plan view — ALL cameras, ALL frames shown
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Footprint Plan View — All Cameras")
st.caption(
    "Plan view centred on the aircraft nadir point (✕). "
    "**x** = across-track (right positive), **y** = along-track (forward = up). "
    "Each polygon is the exact 4-corner footprint of one camera. "
    "Arrows show distance from nadir to inner and outer edges. "
    "Side labels show inner/outer along-track (or across-track) footprint lengths."
)

lim = axis_limits_from_solutions(solutions)

fig_fp, ax_fp = dark_fig(w=10, h=10)
ax_fp.set_aspect("equal")
ax_fp.set_xlim(-lim, lim)
ax_fp.set_ylim(-lim, lim)

# Grid / nadir
ax_fp.axhline(0, color="#21262d", lw=0.8, zorder=1)
ax_fp.axvline(0, color="#21262d", lw=0.8, zorder=1)
ax_fp.scatter([0], [0], s=200, color="#f0c040", zorder=8, marker="x", linewidths=2.5)
ax_fp.annotate("Nadir", (0, 0), xytext=(8, -14), textcoords="offset points",
               color="#f0c040", fontsize=8.5, fontweight="bold")

for cam, sol, colour in solutions:
    corners_xy = [
        (sol.corner_near_top[0], sol.corner_near_top[1]),
        (sol.corner_far_top[0],  sol.corner_far_top[1]),
        (sol.corner_far_bot[0],  sol.corner_far_bot[1]),
        (sol.corner_near_bot[0], sol.corner_near_bot[1]),
    ]
    # Only draw if all corners are finite
    if not all(np.isfinite(v) for xy in corners_xy for v in xy):
        continue

    # Filled polygon
    poly = plt.Polygon(corners_xy, closed=True,
                       facecolor=colour, alpha=0.18,
                       edgecolor=colour, linewidth=2.2, zorder=3)
    ax_fp.add_patch(poly)
    ax_fp.scatter(*zip(*corners_xy), color=colour, s=30, zorder=5, edgecolors="none")

    # Camera label at centroid
    cx = sum(c[0] for c in corners_xy) / 4
    cy = sum(c[1] for c in corners_xy) / 4
    ax_fp.text(cx, cy, sol.label, color=colour, fontsize=8, fontweight="bold",
               ha="center", va="center", zorder=7,
               bbox=dict(facecolor="#0d1117", alpha=0.70, pad=2.5,
                         edgecolor=colour, linewidth=0.8, boxstyle="round,pad=0.3"))

    inner_gx, outer_gx = corner_inner_outer(sol)
    inner_len, outer_len = along_lengths_for_display(sol)
    (it, ib), (ot, ob) = inner_outer_corners(sol)

    if sol.tilt_axis == "across":
        # Dimension arrows along x-axis (y=0)
        ap = dict(arrowstyle="-|>", lw=1.3, mutation_scale=11)
        ax_fp.annotate("", xy=(inner_gx, 0), xytext=(0, 0),
                        arrowprops={**ap, "color": colour}, zorder=4)
        ax_fp.annotate("", xy=(outer_gx, 0), xytext=(inner_gx, 0),
                        arrowprops={**ap, "color": colour}, zorder=4)

        # Distance labels below arrows
        dy = lim * 0.035
        ax_fp.text(inner_gx / 2, -dy,
                   f"{m_to_unit(abs(inner_gx), dist_unit):.0f} {dist_unit}",
                   color=colour, fontsize=7, ha="center", va="top")
        ax_fp.text((inner_gx + outer_gx) / 2, -dy * 2.4,
                   f"{m_to_unit(abs(outer_gx), dist_unit):.0f} {dist_unit}",
                   color=colour, fontsize=7, ha="center", va="top")

        # Inner edge along-track length label (on the nadir-side edge)
        ixm = (it[0] + ib[0]) / 2
        iym = (it[1] + ib[1]) / 2
        dx_off = -9 if inner_gx >= 0 else 9
        ax_fp.annotate(f"{m_to_unit(inner_len, dist_unit):.0f} {dist_unit}",
                        xy=(ixm, iym), xytext=(dx_off, 0), textcoords="offset points",
                        color=colour, fontsize=6.5, va="center",
                        ha="right" if dx_off < 0 else "left")

        # Outer edge along-track length label (on the far edge)
        oxm = (ot[0] + ob[0]) / 2
        oym = (ot[1] + ob[1]) / 2
        dx_off2 = 9 if outer_gx >= 0 else -9
        ax_fp.annotate(f"{m_to_unit(outer_len, dist_unit):.0f} {dist_unit}",
                        xy=(oxm, oym), xytext=(dx_off2, 0), textcoords="offset points",
                        color=colour, fontsize=6.5, va="center",
                        ha="left" if dx_off2 > 0 else "right")

        # GSD labels at inner/outer corners
        inner_gsd = min(sol.near_gsd_m, sol.far_gsd_m)
        outer_gsd = max(sol.near_gsd_m, sol.far_gsd_m)
        ax_fp.text(ixm, iym * 0.65,
                   f"{inner_gsd*100:.2f} cm/px",
                   color=colour, fontsize=5.5, alpha=0.85, ha="center", va="center")
        ax_fp.text(oxm, oym * 0.65,
                   f"{outer_gsd*100:.2f} cm/px",
                   color=colour, fontsize=5.5, alpha=0.85, ha="center", va="center")

    else:  # tilt_axis == "along"
        # Dimension arrows along y-axis (x=0)
        ap = dict(arrowstyle="-|>", lw=1.3, mutation_scale=11)
        inner_gy = sol.near_edge_m if abs(sol.near_edge_m) < abs(sol.far_edge_m) else sol.far_edge_m
        outer_gy = sol.far_edge_m  if abs(sol.far_edge_m)  > abs(sol.near_edge_m) else sol.near_edge_m
        ax_fp.annotate("", xy=(0, inner_gy), xytext=(0, 0),
                        arrowprops={**ap, "color": colour}, zorder=4)
        ax_fp.annotate("", xy=(0, outer_gy), xytext=(0, inner_gy),
                        arrowprops={**ap, "color": colour}, zorder=4)
        dx = lim * 0.025
        ax_fp.text(dx, inner_gy / 2,
                   f"{m_to_unit(abs(inner_gy), dist_unit):.0f} {dist_unit}",
                   color=colour, fontsize=7, ha="left", va="center")
        ax_fp.text(dx, (inner_gy + outer_gy) / 2,
                   f"{m_to_unit(abs(outer_gy), dist_unit):.0f} {dist_unit}",
                   color=colour, fontsize=7, ha="left", va="center")

# Tick labels in display units
xt = ax_fp.get_xticks()
ax_fp.set_xticklabels([f"{m_to_unit(t, dist_unit):.0f}" for t in xt], color="#8b949e")
yt = ax_fp.get_yticks()
ax_fp.set_yticklabels([f"{m_to_unit(t, dist_unit):.0f}" for t in yt], color="#8b949e")
ax_fp.set_xlabel(f"← Across-track ({dist_unit}) →", color="#8b949e")
ax_fp.set_ylabel(f"↑ Along-track / forward ({dist_unit})", color="#8b949e")
ax_fp.set_title("Single-Frame Footprint Plan View — All Cameras", color="#c9d1d9",
                fontsize=11, pad=10)

legend_fp = [mpatches.Patch(color=col, label=cam["label"], alpha=0.85)
             for cam, _, col in solutions]
ax_fp.legend(handles=legend_fp, loc="upper right", fontsize=8, framealpha=0.4,
             labelcolor="#c9d1d9", facecolor="#161b22")

# Forward arrow
ax_fp.annotate("", xy=(0, lim * 0.90), xytext=(0, lim * 0.74),
               arrowprops=dict(arrowstyle="->", color="#c9d1d9", lw=1.8))
ax_fp.text(lim * 0.03, lim * 0.82, "Fwd", color="#c9d1d9", fontsize=7.5, va="center")

fig_fp.tight_layout()
st.pyplot(fig_fp)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 2 — Cross-section (across-track cameras)
# ─────────────────────────────────────────────────────────────────────────────

across_sols = [(c, s, col) for c, s, col in solutions if s.tilt_axis == "across"]

st.subheader("Cross-Section View — Across-Track")
st.caption(
    "Side view looking forward. "
    "Dotted line = inner edge ray, solid = centre ray, dashed = outer edge ray. "
    "GSD values shown at ground intercepts."
)

if across_sols:
    fig_xs, ax_xs = dark_fig(12, 5)
    H = altitude_m

    ax_xs.axvline(0, color="#30363d", lw=1.0, ls="--", zorder=1)
    ax_xs.axhline(0, color="#444",    lw=1.5, zorder=1)
    ax_xs.scatter([0], [H], s=210, color="#f0c040", marker="^", zorder=6)
    ax_xs.annotate(f"{fmt(H, dist_unit)} AGL", (0, H), xytext=(7, -4),
                   textcoords="offset points", color="#f0c040", fontsize=7.5)

    for cam, sol, colour in across_sols:
        inner_gx, outer_gx = corner_inner_outer(sol)
        centre_gx = sol.centre_m
        inner_gsd = min(sol.near_gsd_m, sol.far_gsd_m)
        outer_gsd = max(sol.near_gsd_m, sol.far_gsd_m)
        tilt_h = 90.0 - sol.tilt_from_nadir_deg

        for gx, ls, lw in [(inner_gx, ":", 1.4), (centre_gx, "-", 2.0), (outer_gx, "--", 1.4)]:
            ax_xs.plot([0, gx], [H, 0], color=colour, ls=ls, lw=lw, zorder=3)

        ax_xs.scatter([inner_gx, centre_gx, outer_gx], [0, 0, 0],
                       color=colour, s=50, zorder=5, edgecolors="none")

        ax_xs.text(inner_gx, -H * 0.04,
                   f"{sol.label}\n"
                   f"tilt {tilt_h:.1f}° from horiz\n"
                   f"{m_to_unit(abs(inner_gx), dist_unit):.0f} {dist_unit}",
                   color=colour, fontsize=6, ha="center", va="top")
        ax_xs.text(outer_gx, -H * 0.09,
                   f"{m_to_unit(abs(outer_gx), dist_unit):.0f} {dist_unit}",
                   color=colour, fontsize=6, ha="center", va="top")

        # GSD labels at mid-height on each ray
        ax_xs.text(inner_gx * 0.55, H * 0.18,
                   f"{inner_gsd*100:.1f} cm/px",
                   color=colour, fontsize=6, ha="center", alpha=0.9)
        ax_xs.text(outer_gx * 0.55, H * 0.18,
                   f"{outer_gsd*100:.1f} cm/px",
                   color=colour, fontsize=6, ha="center", alpha=0.9)

        # Obliqueness annotation
        ob_ratio = obliqueness_ratio(sol)
        ax_xs.text(centre_gx * 0.5, H * 0.55,
                   f"Obliqueness: {ob_ratio:.2f}×",
                   color=colour, fontsize=6, ha="center", alpha=0.75,
                   style="italic")

    max_x = max(abs(s.far_edge_m) for _, s, _ in across_sols) * 1.25
    max_x = max(max_x, max(abs(s.near_edge_m) for _, s, _ in across_sols) * 1.25)
    ax_xs.set_xlim(-max_x, max_x)
    ax_xs.set_ylim(-H * 0.18, H * 1.15)

    xt = ax_xs.get_xticks()
    ax_xs.set_xticklabels([f"{m_to_unit(t, dist_unit):.0f}" for t in xt], color="#8b949e")
    yt = ax_xs.get_yticks()
    ax_xs.set_yticklabels([f"{m_to_unit(t, dist_unit):.0f}" for t in yt], color="#8b949e")
    ax_xs.set_xlabel(f"← Across-track ({dist_unit}) →", color="#8b949e")
    ax_xs.set_ylabel(f"Altitude ({dist_unit})", color="#8b949e")
    ax_xs.set_title("Cross-Section — Across-Track Cameras", color="#c9d1d9", fontsize=11)

    legend_xs = [mpatches.Patch(color=col, label=cam["label"], alpha=0.8)
                 for cam, _, col in across_sols]
    legend_xs += [
        mpatches.Patch(color="white", alpha=0, label=""),
        plt.Line2D([0],[0], color="white", ls=":", lw=1.4, label="Inner edge ray"),
        plt.Line2D([0],[0], color="white", ls="-", lw=2.0, label="Centre ray"),
        plt.Line2D([0],[0], color="white", ls="--", lw=1.4, label="Outer edge ray"),
    ]
    ax_xs.legend(handles=legend_xs, fontsize=7, framealpha=0.35,
                 labelcolor="#c9d1d9", facecolor="#161b22", loc="upper right", ncol=2)

    fig_xs.tight_layout()
    st.pyplot(fig_xs)
else:
    st.info("No across-track cameras enabled.")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 3 — Multi-strip plan view
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Multi-Strip Plan View")
st.caption(
    "Three adjacent flight strips offset by the recommended line spacing. "
    "Strip 1 is brightest. Only across-track cameras shown "
    "(fore/aft cameras repeat identically on each strip)."
)

if mc and across_sols:
    fig_ms, ax_ms = dark_fig(14, 7)
    ax_ms.set_aspect("equal")

    line_spacing = mc.recommended_line_spacing_m
    n_strips     = 3
    alphas       = [0.55, 0.38, 0.22]
    lws          = [1.6,  0.9,  0.5]
    strip_cols   = ["#c9d1d9", "#6e7681", "#444"]

    max_along = max(
        max(abs(s.corner_far_top[1]), abs(s.corner_far_bot[1]))
        for _, s, _ in across_sols
    )

    for si in range(n_strips):
        x_off = si * line_spacing
        for cam, sol, colour in across_sols:
            corners = [
                (sol.corner_near_top[0] + x_off, sol.corner_near_top[1]),
                (sol.corner_far_top[0]  + x_off, sol.corner_far_top[1]),
                (sol.corner_far_bot[0]  + x_off, sol.corner_far_bot[1]),
                (sol.corner_near_bot[0] + x_off, sol.corner_near_bot[1]),
            ]
            poly = plt.Polygon(corners, closed=True,
                               facecolor=colour, alpha=alphas[si],
                               edgecolor=colour, linewidth=lws[si], zorder=3)
            ax_ms.add_patch(poly)

        ax_ms.axvline(x_off, color=strip_cols[si], lw=0.9, ls="-.", alpha=0.7, zorder=2)
        ax_ms.text(x_off, max_along * 1.07, f"Strip {si+1}",
                   color=strip_cols[si], fontsize=8, ha="center", va="bottom")

    # Line spacing dimension arrow
    ax_ms.annotate("", xy=(line_spacing, 0), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="<->", color="white", lw=2.0,
                                   mutation_scale=13), zorder=5)
    ax_ms.text(line_spacing / 2, -max_along * 0.09,
               f"Line spacing\n{m_to_unit(line_spacing, dist_unit):.0f} {dist_unit}",
               color="white", fontsize=8, ha="center", va="top",
               bbox=dict(facecolor="#21262d", alpha=0.75, pad=3, edgecolor="#555", lw=0.5))

    # Sidelap zone annotation
    right_sols = [(c, s, col) for c, s, col in across_sols if s.tilt_from_nadir_deg > 0]
    if right_sols:
        _, rs, rcol = right_sols[0]
        outer_gx = max(abs(rs.near_edge_m), abs(rs.far_edge_m))
        if outer_gx > line_spacing:
            ax_ms.axvspan(line_spacing, outer_gx, color=rcol, alpha=0.07, zorder=2)
            ax_ms.text((line_spacing + outer_gx) / 2, max_along * 0.5,
                       f"Sidelap\n{mc.sidelap_achieved*100:.0f}%",
                       color=rcol, fontsize=7, ha="center", va="center",
                       alpha=0.9, rotation=90)

    # Axis limits
    all_x_ms = []
    for _, sol, _ in across_sols:
        for si in range(n_strips):
            all_x_ms += [sol.corner_near_top[0] + si * line_spacing,
                          sol.corner_far_top[0]  + si * line_spacing]
    x_min = min(all_x_ms) * 1.12
    x_max = max(all_x_ms) * 1.12
    y_rng = max_along * 1.18

    ax_ms.set_xlim(x_min, x_max)
    ax_ms.set_ylim(-y_rng, y_rng)

    xt = ax_ms.get_xticks()
    ax_ms.set_xticklabels([f"{m_to_unit(t, dist_unit):.0f}" for t in xt], color="#8b949e")
    yt = ax_ms.get_yticks()
    ax_ms.set_yticklabels([f"{m_to_unit(t, dist_unit):.0f}" for t in yt], color="#8b949e")
    ax_ms.set_xlabel(f"← Across-track ({dist_unit}) →", color="#8b949e")
    ax_ms.set_ylabel(f"↑ Along-track ({dist_unit})", color="#8b949e")
    ax_ms.set_title("Multi-Strip Plan View (3 strips)", color="#c9d1d9", fontsize=11)

    legend_ms = [mpatches.Patch(color=col, label=cam["label"], alpha=0.8)
                 for cam, _, col in across_sols]
    ax_ms.legend(handles=legend_ms, fontsize=8, framealpha=0.35,
                 labelcolor="#c9d1d9", facecolor="#161b22", loc="upper right")

    # Forward arrow
    ax_ms.annotate("", xy=(x_min * 0.55, y_rng * 0.90),
                   xytext=(x_min * 0.55, y_rng * 0.65),
                   arrowprops=dict(arrowstyle="->", color="#c9d1d9", lw=1.8))
    ax_ms.text(x_min * 0.55, y_rng * 0.77, "Fwd",
               color="#c9d1d9", fontsize=7.5, ha="center", va="center")

    fig_ms.tight_layout()
    st.pyplot(fig_ms)
elif not mc:
    st.info("System solution unavailable.")
else:
    st.info("No across-track cameras enabled.")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Formula trace
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Formula Trace")
with st.expander("Show intermediate calculations per camera", expanded=False):
    for cam, sol, colour in solutions:
        bd = get_body(cam["body"])
        inner_gx, outer_gx = corner_inner_outer(sol)
        inner_len, outer_len = along_lengths_for_display(sol)
        inner_gsd = min(sol.near_gsd_m, sol.far_gsd_m)
        outer_gsd = max(sol.near_gsd_m, sol.far_gsd_m)
        tilt_h = 90.0 - sol.tilt_from_nadir_deg
        ob = obliqueness_ratio(sol)

        st.markdown(
            f"<span style='color:{colour}'>●</span> **{sol.label}**",
            unsafe_allow_html=True
        )
        st.markdown(f"""
| Quantity | Formula | Value |
|---|---|---|
| Sensor native | — | {bd['w_mm']} × {bd['h_mm']} mm, {bd['w_px']} × {bd['h_px']} px |
| Orientation | **{sol.orientation}** | across: **{sol.sensor_across_mm:.4f} mm**, along: **{sol.sensor_along_mm:.4f} mm** |
| Focal length | — | **{cam['focal_mm']} mm** |
| Pixel size | `sensor_across / image_across_px` | **{sol.pixel_size_mm*1000:.3f} µm** |
| Tilt | — | **{sol.tilt_from_nadir_deg:.2f}° from nadir** / **{tilt_h:.2f}° from horizontal** |
| Tilt axis | — | **{sol.tilt_axis}** |
| Half FOV across | `atan(sensor_across / (2×fl))` | {sol.half_fov_across_deg:.4f}° → full {sol.full_fov_across_deg:.4f}° |
| Half FOV along | `atan(sensor_along / (2×fl))` | {sol.half_fov_along_deg:.4f}° → full {sol.full_fov_along_deg:.4f}° |
| Diag PP→edge | `sqrt((sensor_across/2)² + fl²)` | **{sol.diag_image_mm:.4f} mm** |
| Inner edge | `H × tan(θ − φ_w)` | **{m_to_unit(abs(inner_gx), dist_unit):.2f} {dist_unit}** from nadir |
| Outer edge | `H × tan(θ + φ_w)` | **{m_to_unit(abs(outer_gx), dist_unit):.2f} {dist_unit}** from nadir |
| Inner slant | `sqrt(H² + Gx²)` | **{min(sol.near_slant_m, sol.far_slant_m):.2f} m** |
| Outer slant | `sqrt(H² + Gx²)` | **{max(sol.near_slant_m, sol.far_slant_m):.2f} m** |
| Inner length | exact 4-corner | **{m_to_unit(inner_len, dist_unit):.2f} {dist_unit}** |
| Outer length | exact 4-corner | **{m_to_unit(outer_len, dist_unit):.2f} {dist_unit}** |
| Inner GSD | `px_mm × slant_mm / diag_mm` | **{fmt_gsd(inner_gsd)}** |
| Centre GSD | — | **{fmt_gsd(sol.centre_gsd_m)}** |
| Outer GSD | `px_mm × slant_mm / diag_mm` | **{fmt_gsd(outer_gsd)}** |
| **Obliqueness ratio** | `outer GSD / inner GSD` | **{ob:.3f}×** |
        """)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Assumptions
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("ℹ️ Assumptions, conventions and spreadsheet verification"):
    st.markdown("""
### Spreadsheet verification (Oblique_setup9_working_2.xls)

**Landscape sheet** — Sony A7R V, nadir fl=21 mm, oblique fl=50 mm, 50° from horizontal, GSD 8.5 cm:

| Value | Our app | Spreadsheet |
|---|---|---|
| Flying height | 469.737 m | 469.737 m |
| Oblique inner edge | 233.820 m | 233.820 m |
| Oblique outer edge | 635.679 m | 635.679 m |
| Inner length | 368.473 m | 368.473 m |
| Outer length | 555.051 m | 555.051 m |
| Inner GSD | 3.877 cm/px | 3.877 cm/px |
| Outer GSD | 5.840 cm/px | 5.840 cm/px |
| Slope to inner | 524.714 m | 524.714 m |
| Slope to outer | 790.405 m | 790.405 m |

All values match to 3+ decimal places.

**Portrait sheet** — uses landscape-mounted oblique cameras (long axis across-track) at a different
flying height (H=631.58 m, nadir fl=48 mm, GSD 5 cm). To reproduce that sheet, set the L/R oblique
cameras to **Landscape** orientation. Both sheets match our model when the correct orientation is used.

### Obliqueness ratio
Defined as `outer GSD / inner GSD`. Values:
- **1.0** = nadir camera (uniform GSD across the image)
- **1.5** = mild oblique (50% more GSD at the far edge than inner edge)
- **>3** = highly oblique (consider whether far-edge GSD meets mission requirements)

### Sensor orientation convention

| Setting | Sensor dimension across-track | Along-track | Typical use |
|---|---|---|---|
| **Portrait** | Short (narrow) axis | Long axis | L/R oblique |
| **Landscape** | Long axis | Short axis | Nadir camera |

### GSD formula (slant-plane, matching spreadsheet)
```
GSD = pixel_size_mm × slant_2d_mm / diag_image_mm
diag_image_mm = sqrt((sensor_across/2)² + focal_length²)
```

### Photo spacing
Uses the **inner (minimum) footprint length** so the target forward overlap is
met at the most constrained position. The far edge will have higher overlap than requested.

### Assumptions (v1)
- Flat terrain
- Pinhole camera, square pixels
- Pure single-axis tilt per camera
- No lens distortion, wind drift, or lever-arm effects
    """)

st.markdown("---")
st.caption(
    "Oblique Survey Planner v3  ·  Flat terrain  ·  Pinhole model  ·  "
    "Verified against Oblique_setup9_working_2.xls"
)
