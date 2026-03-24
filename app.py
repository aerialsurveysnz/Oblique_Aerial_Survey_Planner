"""
app.py  —  Oblique Aerial Survey Planner v3
============================================
Per-camera configuration table with portrait/landscape and tilt-axis selectors.
Three diagram views: footprint plan, cross-section, multi-strip.

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

PRESET_FILE = Path("presets.json")

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

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Oblique Survey Planner", page_icon="✈️", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_saved_bodies() -> dict:
    """Load any user-saved camera body presets from disk."""
    if PRESET_FILE.exists():
        try:
            return json.loads(PRESET_FILE.read_text())
        except Exception:
            pass
    return {}

def save_body_preset(name: str, data: dict):
    existing = load_saved_bodies()
    existing[name] = data
    PRESET_FILE.write_text(json.dumps(existing, indent=2))

def save_scenario(data: dict, path: str):
    Path(path).write_text(json.dumps(data, indent=2, default=str))

def load_scenario(path: str) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None

def all_body_names() -> list[str]:
    return list(BODY_PRESETS.keys()) + list(load_saved_bodies().keys())

def get_body(name: str) -> dict:
    """Return body dict for a given name, checking both built-ins and saved presets."""
    saved = load_saved_bodies()
    if name in BODY_PRESETS:
        return BODY_PRESETS[name]
    if name in saved:
        d = saved[name]
        return {"w_mm": d["w_mm"], "h_mm": d["h_mm"], "w_px": d["w_px"], "h_px": d["h_px"]}
    return BODY_PRESETS["Sony A7R V"]   # fallback

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

if "cameras" not in st.session_state:
    st.session_state.cameras = [dict(c) for c in DEFAULT_CAMERAS]

# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v_m: float, unit: str, d: int = 1) -> str:
    return f"{m_to_unit(v_m, unit):.{d}f} {unit}"

def fmt_gsd(v_m: float, d: int = 2) -> str:
    return f"{v_m * 100:.{d}f} cm/px"

def dark_fig(w: float = 12, h: float = 6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    return fig, ax

def corner_inner_outer(sol):
    """
    Return (inner_gx, outer_gx) where inner = edge closer to nadir (smaller |Gx|)
    and outer = edge farther from nadir (larger |Gx|).
    Works correctly for both positive-tilt (right) and negative-tilt (left) cameras.
    """
    a, b = sol.near_edge_m, sol.far_edge_m
    if abs(a) <= abs(b):
        return a, b   # near is inner, far is outer  (right camera, positive tilt)
    else:
        return b, a   # far is inner, near is outer   (left camera, negative tilt)

def inner_outer_corners(sol):
    """
    Return four corners split into two pairs: inner (nadir side) and outer.
    Each pair is (top_corner, bot_corner) = ((Gx, Gy), (Gx, Gy)).
    """
    nt, nb = sol.corner_near_top, sol.corner_near_bot
    ft, fb = sol.corner_far_top,  sol.corner_far_bot
    # Inner = pair with smaller |Gx|
    if abs(nt[0]) <= abs(ft[0]):
        inner_top, inner_bot = nt, nb
        outer_top, outer_bot = ft, fb
    else:
        inner_top, inner_bot = ft, fb
        outer_top, outer_bot = nt, nb
    return (inner_top, inner_bot), (outer_top, outer_bot)

def along_lengths_for_display(sol):
    """
    Return (inner_length, outer_length) — the along-track span of the inner and outer
    footprint edges, in metres. Correctly handles both tilt directions.
    """
    (it, ib), (ot, ob) = inner_outer_corners(sol)
    inner_len = abs(it[1] - ib[1])
    outer_len = abs(ot[1] - ob[1])
    return inner_len, outer_len

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — flight parameters
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
if st.sidebar.button("Save scenario (JSON)"):
    save_scenario({
        "cameras":        st.session_state.cameras,
        "altitude_m":     altitude_m,
        "speed_ms":       speed_ms,
        "fwd_overlap_pct": fwd_pct,
        "sidelap_pct":    side_pct,
        "reciprocal":     reciprocal,
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

with st.expander("Orientation & tilt-axis guide", expanded=False):
    st.markdown("""
| Setting | Option | Meaning | Best for |
|---|---|---|---|
| **Orientation** | Portrait | **Narrow** sensor axis across-track, long axis along-track | L/R oblique — limits far-edge GSD stretch |
| | Landscape | **Long** sensor axis across-track, narrow along-track | Nadir — maximises swath |
| **Tilt axis** | Across (L/R) | Camera rotates left/right about the along-track axis | Left & Right oblique cameras |
| | Along (F/A) | Camera rotates fore/aft about the across-track axis | Fore & Aft oblique cameras |

For a **left** camera, enter the same tilt angle as the right camera — the geometry mirrors automatically because
the camera body is mounted on the opposite side.  Both cameras share the same tilt magnitude.
    """)

cameras      = st.session_state.cameras
bodies_avail = all_body_names()

# Column headers
hdr = st.columns([0.4, 1.6, 1.8, 0.75, 0.7, 0.85, 0.9, 1.05, 0.35])
for col, lbl in zip(hdr, ["✓", "Label", "Body", "FL mm", "Tilt °", "From", "Orient.", "Tilt axis", ""]):
    col.markdown(f"<span style='color:#8b949e;font-size:0.8em'>{lbl}</span>", unsafe_allow_html=True)

to_delete = None
for i, cam in enumerate(cameras):
    col_en, col_lbl, col_body, col_fl, col_tilt, col_conv, col_orient, col_axis, col_del = \
        st.columns([0.4, 1.6, 1.8, 0.75, 0.7, 0.85, 0.9, 1.05, 0.35])

    colour = CAM_COLOURS[i % len(CAM_COLOURS)]
    dot    = f"<span style='color:{colour}'>●</span>"

    cam["enabled"] = col_en.checkbox(
        "##e", value=cam["enabled"], key=f"en_{i}", label_visibility="collapsed"
    )
    col_lbl.markdown(dot, unsafe_allow_html=True)
    cam["label"] = col_lbl.text_input(
        "##l", value=cam["label"], key=f"lbl_{i}", label_visibility="collapsed"
    )

    body_idx = bodies_avail.index(cam["body"]) if cam["body"] in bodies_avail else 0
    cam["body"] = col_body.selectbox(
        "##b", bodies_avail, index=body_idx, key=f"body_{i}", label_visibility="collapsed"
    )
    cam["focal_mm"] = float(col_fl.number_input(
        "##f", value=float(cam["focal_mm"]), min_value=1.0, max_value=2000.0, step=1.0,
        key=f"fl_{i}", label_visibility="collapsed",
    ))
    cam["tilt_deg"] = float(col_tilt.number_input(
        "##t", value=float(cam["tilt_deg"]), min_value=0.0, max_value=85.0, step=0.5,
        key=f"tilt_{i}", label_visibility="collapsed",
    ))
    cam["tilt_conv"] = col_conv.selectbox(
        "##c", ["horiz", "nadir"], key=f"conv_{i}", label_visibility="collapsed",
        index=0 if cam["tilt_conv"] == "horiz" else 1,
        format_func=lambda x: "Horizontal" if x == "horiz" else "Nadir",
    )
    cam["orientation"] = col_orient.selectbox(
        "##o", ["portrait", "landscape"], key=f"orient_{i}", label_visibility="collapsed",
        index=0 if cam["orientation"] == "portrait" else 1,
        format_func=lambda x: "Portrait" if x == "portrait" else "Landscape",
    )
    cam["tilt_axis"] = col_axis.selectbox(
        "##a", ["across", "along"], key=f"axis_{i}", label_visibility="collapsed",
        index=0 if cam["tilt_axis"] == "across" else 1,
        format_func=lambda x: "Across (L/R)" if x == "across" else "Along (F/A)",
    )
    if col_del.button("✕", key=f"del_{i}", help="Remove this camera"):
        to_delete = i

if to_delete is not None:
    cameras.pop(to_delete)
    st.rerun()

b1, b2, b3 = st.columns([1, 1, 6])
if b1.button("➕ Add camera"):
    cameras.append({
        "enabled": True, "label": f"Camera {len(cameras)+1}",
        "body": "Sony A7R V", "focal_mm": 50.0, "tilt_deg": 50.0,
        "tilt_conv": "horiz", "orientation": "portrait", "tilt_axis": "across",
    })
    st.rerun()
if b2.button("↺ Reset defaults"):
    st.session_state.cameras = [dict(c) for c in DEFAULT_CAMERAS]
    st.rerun()

# Save custom body preset
with st.expander("Save a camera body as a preset"):
    pc1, pc2 = st.columns([2, 1])
    p_base = pc1.selectbox("Base body", list(BODY_PRESETS.keys()), key="p_base")
    p_name = pc2.text_input("Preset name", value=p_base, key="p_name")
    p_wmm  = pc1.number_input("Sensor width mm", value=BODY_PRESETS[p_base]["w_mm"], key="p_wmm")
    p_hmm  = pc1.number_input("Sensor height mm", value=BODY_PRESETS[p_base]["h_mm"], key="p_hmm")
    p_wpx  = pc1.number_input("Width px", value=BODY_PRESETS[p_base]["w_px"], step=100, key="p_wpx")
    p_hpx  = pc1.number_input("Height px", value=BODY_PRESETS[p_base]["h_px"], step=100, key="p_hpx")
    if pc2.button("Save preset"):
        save_body_preset(p_name, {"w_mm": p_wmm, "h_mm": p_hmm, "w_px": int(p_wpx), "h_px": int(p_hpx)})
        st.success(f"Saved preset '{p_name}'. Reload page to use it in the body dropdown.")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Build camera solutions
# ─────────────────────────────────────────────────────────────────────────────

active = [c for c in cameras if c["enabled"]]
solutions: list[tuple[dict, object, str]] = []  # (cam_dict, CameraSolution, colour)
errors: list[str] = []

for i, cam in enumerate(active):
    try:
        bd = get_body(cam["body"])
        tilt_n = normalize_tilt_angle(cam["tilt_deg"], cam["tilt_conv"])
        sol = calculate_camera_solution(
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
        camera_solutions     = sol_list,
        arrangement          = "custom",
        altitude_m           = altitude_m,
        aircraft_speed_ms    = speed_ms,
        forward_overlap_fraction = fwd_frac,
        sidelap_fraction     = side_frac,
        reciprocal_flying    = reciprocal,
    )
except Exception as e:
    st.error(f"System calculation error: {e}")

if mc and mc.warnings:
    with st.expander("⚠️ Warnings", expanded=True):
        for w in mc.warnings:
            st.warning(w)

# ─────────────────────────────────────────────────────────────────────────────
# System summary cards
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("System Summary")
if mc:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Combined swath",    fmt(mc.combined_swath_m,             dist_unit))
    c2.metric("Line spacing",      fmt(mc.recommended_line_spacing_m,  dist_unit))
    c3.metric("Photo spacing",     fmt(mc.recommended_photo_spacing_m, dist_unit))
    c4.metric("Exposure interval", f"{mc.photo_interval_s:.2f} s")
    c5.metric("Sidelap achieved",  f"{mc.sidelap_achieved * 100:.1f}%")

    # Representative GSD from first oblique camera (skip nadir)
    rep = next((s for _, s, _ in solutions if abs(s.tilt_from_nadir_deg) > 1), sol_list[0])
    c6, c7, c8, c9 = st.columns(4)
    c6.metric(f"GSD near ({rep.label})",   fmt_gsd(rep.near_gsd_m))
    c7.metric("GSD centre",                fmt_gsd(rep.centre_gsd_m))
    c8.metric("GSD far",                   fmt_gsd(rep.far_gsd_m))
    c9.metric("Fwd overlap near / ctr / far",
              f"{mc.forward_overlap_near*100:.0f}% / "
              f"{mc.forward_overlap_centre*100:.0f}% / "
              f"{mc.forward_overlap_far*100:.0f}%")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Per-camera results table
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Per-Camera Results")

rows = []
for cam, sol, _ in solutions:
    inner_gx, outer_gx = corner_inner_outer(sol)
    inner_len, outer_len = along_lengths_for_display(sol)
    rows.append({
        "Camera":                   sol.label,
        "Body / FL":                f"{cam['body']}  {cam['focal_mm']:.0f} mm",
        "Orient.":                  sol.orientation,
        "Tilt axis":                sol.tilt_axis,
        "Tilt nadir °":             round(sol.tilt_from_nadir_deg, 1),
        "Tilt horiz °":             round(90 - sol.tilt_from_nadir_deg, 1),
        "Pixel µm":                 round(sol.pixel_size_mm * 1000, 2),
        "FOV across °":             round(sol.full_fov_across_deg, 2),
        "FOV along °":              round(sol.full_fov_along_deg,  2),
        f"Inner edge ({dist_unit})": round(m_to_unit(abs(inner_gx), dist_unit), 1),
        f"Outer edge ({dist_unit})": round(m_to_unit(abs(outer_gx), dist_unit), 1),
        f"Inner length ({dist_unit})": round(m_to_unit(inner_len, dist_unit), 1),
        f"Outer length ({dist_unit})": round(m_to_unit(outer_len, dist_unit), 1),
        "Inner GSD cm":             round(min(sol.near_gsd_m, sol.far_gsd_m) * 100, 3),
        "Outer GSD cm":             round(max(sol.near_gsd_m, sol.far_gsd_m) * 100, 3),
        "Centre GSD cm":            round(sol.centre_gsd_m * 100, 3),
        "Inner slant m":            round(min(sol.near_slant_m, sol.far_slant_m), 1),
        "Outer slant m":            round(max(sol.near_slant_m, sol.far_slant_m), 1),
        "Diag image mm":            round(sol.diag_image_mm, 4),
    })

st.dataframe(rows, use_container_width=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 1 — Footprint plan view (spreadsheet style)
# Centred on aircraft nadir.  x = across-track,  y = along-track (forward = up).
# Each camera drawn as filled quadrilateral.
# Dimension lines and labels at inner/outer edges, matching spreadsheet layout.
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Footprint Plan View — Single Frame")
st.caption(
    "Plan view centred on the aircraft nadir point (✕). "
    "**x** = across-track (right positive), **y** = along-track (forward up). "
    "Dimension arrows show distance from nadir to the inner and outer footprint edges. "
    "Inner / outer lengths annotated on the footprint sides."
)

fig_fp, ax_fp = dark_fig(w=9, h=9)
ax_fp.set_aspect("equal")

# Grid lines
ax_fp.axhline(0, color="#21262d", lw=0.8, zorder=1)
ax_fp.axvline(0, color="#21262d", lw=0.8, zorder=1)
ax_fp.scatter([0], [0], s=160, color="#f0c040", zorder=7, marker="x", linewidths=2.5)
ax_fp.annotate("Nadir", (0, 0), xytext=(6, -12), textcoords="offset points",
               color="#f0c040", fontsize=8, fontweight="bold")

for cam, sol, colour in solutions:
    # --- Footprint polygon ---
    corners_xy = [
        (sol.corner_near_top[0], sol.corner_near_top[1]),
        (sol.corner_far_top[0],  sol.corner_far_top[1]),
        (sol.corner_far_bot[0],  sol.corner_far_bot[1]),
        (sol.corner_near_bot[0], sol.corner_near_bot[1]),
    ]
    poly = plt.Polygon(corners_xy, closed=True,
                       facecolor=colour, alpha=0.20,
                       edgecolor=colour, linewidth=2.0, zorder=3)
    ax_fp.add_patch(poly)
    ax_fp.scatter(*zip(*corners_xy), color=colour, s=28, zorder=5, edgecolors="none")

    # --- Camera label at centroid ---
    cx = sum(c[0] for c in corners_xy) / 4
    cy = sum(c[1] for c in corners_xy) / 4
    ax_fp.text(cx, cy, sol.label, color=colour, fontsize=8, fontweight="bold",
               ha="center", va="center", zorder=6,
               bbox=dict(facecolor="#0d1117", alpha=0.65, pad=2,
                         edgecolor=colour, linewidth=0.6, boxstyle="round,pad=0.3"))

    # --- Dimension arrows and labels ---
    inner_gx, outer_gx = corner_inner_outer(sol)
    inner_len, outer_len = along_lengths_for_display(sol)
    (it, ib), (ot, ob) = inner_outer_corners(sol)

    if sol.tilt_axis == "across":
        # Arrows along x-axis from nadir to inner and outer edges
        y_arr = cy * 0.0   # draw arrow along y=0 line
        arrowprops = dict(arrowstyle="-|>", lw=1.2, mutation_scale=10)
        ax_fp.annotate("", xy=(inner_gx, y_arr), xytext=(0, y_arr),
                        arrowprops={**arrowprops, "color": colour}, zorder=4)
        ax_fp.annotate("", xy=(outer_gx, y_arr), xytext=(inner_gx, y_arr),
                        arrowprops={**arrowprops, "color": colour}, zorder=4)

        # Distance labels on the arrows
        lim_ref = abs(outer_gx)
        ha_inner = "left" if inner_gx > 0 else "right"
        ha_outer = "left" if outer_gx > 0 else "right"
        label_dy = lim_ref * 0.04
        ax_fp.text(inner_gx / 2, -label_dy,
                   f"{m_to_unit(abs(inner_gx), dist_unit):.0f} {dist_unit}",
                   color=colour, fontsize=7, ha="center", va="top", zorder=6)
        ax_fp.text((inner_gx + outer_gx) / 2, -label_dy * 2.2,
                   f"{m_to_unit(abs(outer_gx), dist_unit):.0f} {dist_unit}",
                   color=colour, fontsize=7, ha="center", va="top", zorder=6)

        # Inner / outer length annotations on the footprint sides
        # Inner edge: midpoint of inner_top to inner_bot
        ixm = (it[0] + ib[0]) / 2
        iym = (it[1] + ib[1]) / 2
        offset_x = -8 if inner_gx >= 0 else 8
        ax_fp.annotate(f"{m_to_unit(inner_len, dist_unit):.0f} {dist_unit}",
                        xy=(ixm, iym), xytext=(offset_x, 0), textcoords="offset points",
                        color=colour, fontsize=6.5, ha="right" if offset_x < 0 else "left",
                        va="center", zorder=6)
        # Outer edge
        oxm = (ot[0] + ob[0]) / 2
        oym = (ot[1] + ob[1]) / 2
        offset_x2 = 8 if outer_gx >= 0 else -8
        ax_fp.annotate(f"{m_to_unit(outer_len, dist_unit):.0f} {dist_unit}",
                        xy=(oxm, oym), xytext=(offset_x2, 0), textcoords="offset points",
                        color=colour, fontsize=6.5, ha="left" if offset_x2 > 0 else "right",
                        va="center", zorder=6)

        # GSD annotation inside footprint near inner and outer edge
        ax_fp.text(it[0], it[1] * 0.7,
                   f"GSD {m_to_unit(abs(inner_gx), dist_unit):.0f}{dist_unit}:\n"
                   f"{min(sol.near_gsd_m, sol.far_gsd_m)*100:.2f} cm/px",
                   color=colour, fontsize=5.5, alpha=0.8, ha="center", va="center", zorder=5)
        ax_fp.text(ot[0], ot[1] * 0.7,
                   f"GSD {m_to_unit(abs(outer_gx), dist_unit):.0f}{dist_unit}:\n"
                   f"{max(sol.near_gsd_m, sol.far_gsd_m)*100:.2f} cm/px",
                   color=colour, fontsize=5.5, alpha=0.8, ha="center", va="center", zorder=5)

    else:
        # Along-tilt camera: arrows along y-axis
        inner_gy, outer_gy = corner_inner_outer(sol)  # reuse — for along-tilt near/far is in y
        # For along cameras: near_edge_m / far_edge_m are G_y values
        inner_gy = sol.near_edge_m if abs(sol.near_edge_m) < abs(sol.far_edge_m) else sol.far_edge_m
        outer_gy = sol.far_edge_m  if abs(sol.far_edge_m)  > abs(sol.near_edge_m) else sol.near_edge_m

        arrowprops = dict(arrowstyle="-|>", lw=1.2, mutation_scale=10)
        ax_fp.annotate("", xy=(0, inner_gy), xytext=(0, 0),
                        arrowprops={**arrowprops, "color": colour}, zorder=4)
        ax_fp.annotate("", xy=(0, outer_gy), xytext=(0, inner_gy),
                        arrowprops={**arrowprops, "color": colour}, zorder=4)

        lim_ref = abs(outer_gy)
        label_dx = lim_ref * 0.04
        ax_fp.text(label_dx, inner_gy / 2,
                   f"{m_to_unit(abs(inner_gy), dist_unit):.0f} {dist_unit}",
                   color=colour, fontsize=7, ha="left", va="center", zorder=6)
        ax_fp.text(label_dx, (inner_gy + outer_gy) / 2,
                   f"{m_to_unit(abs(outer_gy), dist_unit):.0f} {dist_unit}",
                   color=colour, fontsize=7, ha="left", va="center", zorder=6)

# Axis limits — fit all corners with margin
all_x = [c[v][0] for _, sol, _ in solutions
          for c in [(sol.corner_near_top, sol.corner_far_top,
                     sol.corner_near_bot, sol.corner_far_bot)]
          for v in range(4)]
all_y = [c[v][1] for _, sol, _ in solutions
          for c in [(sol.corner_near_top, sol.corner_far_top,
                     sol.corner_near_bot, sol.corner_far_bot)]
          for v in range(4)]

# Simpler approach — flatten corner lists
all_x = [p for _, sol, _ in solutions
          for p in [sol.corner_near_top[0], sol.corner_far_top[0],
                    sol.corner_near_bot[0], sol.corner_far_bot[0]]]
all_y = [p for _, sol, _ in solutions
          for p in [sol.corner_near_top[1], sol.corner_far_top[1],
                    sol.corner_near_bot[1], sol.corner_far_bot[1]]]

if all_x and all_y:
    pad = 0.18
    x_span = max(abs(min(all_x)), abs(max(all_x))) * (1 + pad)
    y_span = max(abs(min(all_y)), abs(max(all_y))) * (1 + pad)
    lim = max(x_span, y_span)
    ax_fp.set_xlim(-lim, lim)
    ax_fp.set_ylim(-lim, lim)

# Tick labels in display units
xt = ax_fp.get_xticks()
ax_fp.set_xticklabels([f"{m_to_unit(t, dist_unit):.0f}" for t in xt], color="#8b949e")
yt = ax_fp.get_yticks()
ax_fp.set_yticklabels([f"{m_to_unit(t, dist_unit):.0f}" for t in yt], color="#8b949e")
ax_fp.set_xlabel(f"← Across-track ({dist_unit}) →", color="#8b949e")
ax_fp.set_ylabel(f"↑ Along-track / forward ({dist_unit})", color="#8b949e")
ax_fp.set_title("Single-Frame Footprint — Plan View", color="#c9d1d9", fontsize=11, pad=10)

legend_fp = [mpatches.Patch(color=col, label=cam["label"], alpha=0.8)
             for cam, _, col in solutions]
ax_fp.legend(handles=legend_fp, loc="upper right", fontsize=8, framealpha=0.4,
             labelcolor="#c9d1d9", facecolor="#161b22")

# North/flight direction arrow
ax_fp.annotate("", xy=(0, lim * 0.88), xytext=(0, lim * 0.72),
               arrowprops=dict(arrowstyle="->", color="#c9d1d9", lw=1.5))
ax_fp.text(lim * 0.04, lim * 0.80, "Fwd", color="#c9d1d9", fontsize=7, va="center")

fig_fp.tight_layout()
st.pyplot(fig_fp)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 2 — Cross-section (across-track cameras only)
# ─────────────────────────────────────────────────────────────────────────────

across_sols = [(c, s, col) for c, s, col in solutions if s.tilt_axis == "across"]

st.subheader("Cross-Section View — Across-Track")
st.caption(
    "Side view looking forward along the flight track. "
    "Solid lines = image centre ray. Dotted = inner edge. Dashed = outer edge."
)

if across_sols:
    fig_xs, ax_xs = dark_fig(12, 5)
    H = altitude_m

    ax_xs.axvline(0, color="#30363d", lw=1.0, ls="--", zorder=1)
    ax_xs.axhline(0, color="#444",    lw=1.5, zorder=1, label="Ground")
    ax_xs.scatter([0], [H], s=200, color="#f0c040", marker="^", zorder=6, label="Aircraft")
    ax_xs.annotate(f"{fmt(H, dist_unit)}", (0, H), xytext=(6, -4),
                   textcoords="offset points", color="#f0c040", fontsize=7.5)

    for cam, sol, colour in across_sols:
        inner_gx, outer_gx = corner_inner_outer(sol)
        centre_gx = sol.centre_m

        for gx, ls, lw in [(inner_gx, ":", 1.3), (centre_gx, "-", 1.8), (outer_gx, "--", 1.3)]:
            ax_xs.plot([0, gx], [H, 0], color=colour, ls=ls, lw=lw, zorder=3)

        ax_xs.scatter([inner_gx, centre_gx, outer_gx], [0, 0, 0],
                       color=colour, s=45, zorder=5, edgecolors="none")

        # Ground labels
        for gx, label_str, dy_frac in [
            (inner_gx, f"{sol.label}\n{m_to_unit(abs(inner_gx), dist_unit):.0f} {dist_unit}", 0.04),
            (outer_gx, f"{m_to_unit(abs(outer_gx), dist_unit):.0f} {dist_unit}", 0.09),
        ]:
            ax_xs.text(gx, -H * dy_frac, label_str, color=colour,
                       fontsize=6.5, ha="center", va="top", zorder=6)

        # GSD labels at ground intercepts
        inner_gsd = min(sol.near_gsd_m, sol.far_gsd_m)
        outer_gsd = max(sol.near_gsd_m, sol.far_gsd_m)
        ax_xs.text(inner_gx, H * 0.15,
                   f"{inner_gsd*100:.1f} cm/px", color=colour,
                   fontsize=6, ha="center", alpha=0.8)
        ax_xs.text(outer_gx, H * 0.15,
                   f"{outer_gsd*100:.1f} cm/px", color=colour,
                   fontsize=6, ha="center", alpha=0.8)

    max_x = max(abs(s.far_edge_m) for _, s, _ in across_sols) * 1.22
    max_x = max(max_x, max(abs(s.near_edge_m) for _, s, _ in across_sols) * 1.22)
    ax_xs.set_xlim(-max_x, max_x)
    ax_xs.set_ylim(-H * 0.16, H * 1.14)

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
        plt.Line2D([0], [0], color="white", ls=":", lw=1.3, label="Inner edge"),
        plt.Line2D([0], [0], color="white", ls="-", lw=1.8, label="Centre ray"),
        plt.Line2D([0], [0], color="white", ls="--", lw=1.3, label="Outer edge"),
    ]
    ax_xs.legend(handles=legend_xs, fontsize=7, framealpha=0.35,
                 labelcolor="#c9d1d9", facecolor="#161b22", loc="upper right", ncol=2)

    fig_xs.tight_layout()
    st.pyplot(fig_xs)
else:
    st.info("No across-track cameras enabled.")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 3 — Multi-strip plan view (3 adjacent strips)
# Across-track cameras only. Strip 1 is the reference; strips 2 & 3 are offset
# by the recommended line spacing along the across-track direction.
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Multi-Strip Plan View")
st.caption(
    "Three adjacent flight strips offset by the recommended line spacing. "
    "Strip 1 (brightest) is the reference. "
    "Across-track cameras only — fore/aft cameras repeat identically on every strip."
)

if mc and across_sols:
    fig_ms, ax_ms = dark_fig(14, 7)
    ax_ms.set_aspect("equal")

    line_spacing = mc.recommended_line_spacing_m
    n_strips     = 3
    alphas       = [0.55, 0.38, 0.22]
    lws          = [1.6,  0.9,  0.5]
    strip_colors = ["#c9d1d9", "#6e7681", "#444"]

    # Along-track extent for tick/limit calculation
    max_along = max(
        max(abs(s.corner_far_top[1]), abs(s.corner_far_bot[1]))
        for _, s, _ in across_sols
    )

    for si in range(n_strips):
        x_off = si * line_spacing   # across-track shift for this strip's nadir track
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

        # Nadir track
        ax_ms.axvline(x_off, color=strip_colors[si], lw=0.8, ls="-.", alpha=0.7, zorder=2)
        ax_ms.text(x_off, max_along * 1.06, f"Strip {si+1}",
                   color=strip_colors[si], fontsize=8, ha="center", va="bottom")

    # Line spacing dimension arrow
    ax_ms.annotate("", xy=(line_spacing, 0), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="<->", color="white", lw=1.8,
                                   mutation_scale=12), zorder=5)
    ax_ms.text(line_spacing / 2, -max_along * 0.09,
               f"Line spacing\n{m_to_unit(line_spacing, dist_unit):.0f} {dist_unit}",
               color="white", fontsize=8, ha="center", va="top",
               bbox=dict(facecolor="#21262d", alpha=0.75, pad=3,
                         edgecolor="#555", linewidth=0.5))

    # Overlap annotation — show where adjacent strips overlap
    # Overlap zone = from strip2 inner to strip1 outer (for right-side cameras)
    right_sols = [(c, s, col) for c, s, col in across_sols if s.tilt_from_nadir_deg > 0]
    if right_sols:
        _, rs, rcol = right_sols[0]
        outer_gx = max(abs(rs.near_edge_m), abs(rs.far_edge_m))
        # Overlap starts at line_spacing and ends at outer_gx
        if outer_gx > line_spacing:
            ax_ms.axvspan(line_spacing, outer_gx, color=rcol, alpha=0.08,
                          zorder=2, label="Overlap zone")
            ax_ms.text((line_spacing + outer_gx) / 2, max_along * 0.5,
                       f"Sidelap\n{mc.sidelap_achieved*100:.0f}%",
                       color=rcol, fontsize=7, ha="center", va="center",
                       alpha=0.9, rotation=90)

    # Axis limits
    all_x_ms = []
    for _, sol, _ in across_sols:
        for si in range(n_strips):
            all_x_ms += [sol.corner_near_top[0] + si*line_spacing,
                          sol.corner_far_top[0]  + si*line_spacing]
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

    # Forward flight arrow
    ax_ms.annotate("", xy=(x_min * 0.5, y_rng * 0.88), xytext=(x_min * 0.5, y_rng * 0.62),
                   arrowprops=dict(arrowstyle="->", color="#c9d1d9", lw=1.5))
    ax_ms.text(x_min * 0.5, y_rng * 0.75, "Fwd", color="#c9d1d9",
               fontsize=7, ha="center", va="center")

    fig_ms.tight_layout()
    st.pyplot(fig_ms)
elif not mc:
    st.info("System solution unavailable.")
else:
    st.info("No across-track cameras enabled.")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Formula trace — per camera
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Formula Trace")
with st.expander("Show intermediate calculations per camera", expanded=False):
    for cam, sol, colour in solutions:
        bd = get_body(cam["body"])
        inner_gx, outer_gx = corner_inner_outer(sol)
        inner_len, outer_len = along_lengths_for_display(sol)
        inner_gsd = min(sol.near_gsd_m, sol.far_gsd_m)
        outer_gsd = max(sol.near_gsd_m, sol.far_gsd_m)

        st.markdown(
            f"<span style='color:{colour}'>●</span> **{sol.label}**",
            unsafe_allow_html=True,
        )
        st.markdown(f"""
| Quantity | Formula | Value |
|---|---|---|
| Sensor native | — | {bd['w_mm']} × {bd['h_mm']} mm,  {bd['w_px']} × {bd['h_px']} px |
| Orientation | **{sol.orientation}** → across: {sol.sensor_across_mm:.4f} mm, along: {sol.sensor_along_mm:.4f} mm | |
| Focal length | — | **{cam['focal_mm']} mm** |
| Pixel size | `sensor_across / image_across_px` | **{sol.pixel_size_mm*1000:.3f} µm** |
| Tilt | — | {sol.tilt_from_nadir_deg:.2f}° from nadir / {90-sol.tilt_from_nadir_deg:.2f}° from horizontal |
| Tilt axis | — | **{sol.tilt_axis}** |
| Half FOV across | `atan(sensor_across / (2×fl))` | {sol.half_fov_across_deg:.4f}° → full {sol.full_fov_across_deg:.4f}° |
| Half FOV along  | `atan(sensor_along / (2×fl))` | {sol.half_fov_along_deg:.4f}° → full {sol.full_fov_along_deg:.4f}° |
| Diag PP→edge | `sqrt((sensor_across/2)² + fl²)` | **{sol.diag_image_mm:.4f} mm** |
| Inner edge | `H × tan(θ − φ_w)` | **{m_to_unit(abs(inner_gx), dist_unit):.2f} {dist_unit}** from nadir |
| Outer edge | `H × tan(θ + φ_w)` | **{m_to_unit(abs(outer_gx), dist_unit):.2f} {dist_unit}** from nadir |
| Inner slant | `sqrt(H² + Gx²)` | **{min(sol.near_slant_m, sol.far_slant_m):.2f} m** |
| Outer slant | `sqrt(H² + Gx²)` | **{max(sol.near_slant_m, sol.far_slant_m):.2f} m** |
| Inner length (perp) | exact 4-corner | **{m_to_unit(inner_len, dist_unit):.2f} {dist_unit}** |
| Outer length (perp) | exact 4-corner | **{m_to_unit(outer_len, dist_unit):.2f} {dist_unit}** |
| Inner GSD | `px_mm × slant_mm / diag_mm` | **{fmt_gsd(inner_gsd)}** |
| Centre GSD | — | **{fmt_gsd(sol.centre_gsd_m)}** |
| Outer GSD | `px_mm × slant_mm / diag_mm` | **{fmt_gsd(outer_gsd)}** |
        """)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Assumptions
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("ℹ️ Assumptions, conventions and limitations"):
    st.markdown("""
### Sensor orientation

| Setting | Across-track dim | Along-track dim | Typical use |
|---|---|---|---|
| **Portrait** | Short (narrow) axis | Long axis | L/R oblique — limits GSD stretch at far edge |
| **Landscape** | Long axis | Short axis | Nadir — maximises swath width |

The pixel count used for pixel-size calculation matches the chosen axis:
portrait uses the short-axis pixel count across-track; landscape uses the long-axis count.

### Tilt axis convention

| Axis | Camera rotates about | Footprint elongated in | Sign convention |
|---|---|---|---|
| **Across (L/R)** | Along-track (Y) | Across-track (X) | Positive = tilts right |
| **Along (F/A)** | Across-track (X) | Along-track (Y) | Positive = tilts forward |

For a **left-side** camera, enter the same tilt angle as the right-side camera.
The geometry mirrors correctly because negative tilt produces the symmetric footprint.

### GSD formula
```
GSD = pixel_size_mm × slant_2d_mm / diag_image_mm
diag_image_mm = sqrt((sensor_across / 2)² + focal_length²)
```
This is the slant-plane GSD, matching the reference spreadsheet (Oblique_setup9_working_2.xls).

### Photo spacing
Uses the **inner (minimum) footprint length** to ensure the target forward overlap is
met at the most constrained position (closest to nadir). The far edge will have somewhat
higher overlap than requested.

### Combined swath
Spans from the leftmost to rightmost footprint corner across all enabled cameras.
Line spacing = combined swath × (1 − sidelap).

### Terrain
v1 assumes flat terrain. For undulating terrain use altitude AGL above the highest
point as a conservative GSD estimate.
    """)

st.markdown("---")
st.caption(
    "Oblique Survey Planner v3  ·  Flat terrain  ·  Pinhole model  ·  "
    "Verified against Oblique_setup9_working_2.xls"
)
