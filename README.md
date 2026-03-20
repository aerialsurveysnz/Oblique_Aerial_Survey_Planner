# Oblique Aerial Survey Planner

A Streamlit calculator for planning 4-camera oblique aerial photogrammetry flights.

## What it does

Given your camera specs, flight altitude, speed, and overlap targets, the app computes:

- Ground intercept distances (near edge, centre, far edge) for each camera
- GSD at near edge, image centre, and far edge
- Across-track and along-track footprint dimensions
- Combined swath width across all cameras
- Recommended flight line spacing (to hit your sidelap target)
- Recommended photo spacing / exposure interval (to hit your forward overlap target)
- Achieved overlap at near/centre/far edge positions
- Warnings for physically impossible or risky configurations

It also renders:
- A **cross-section plot** showing camera rays to the ground
- A **plan-view plot** showing adjacent strips and their footprints

---

## Quick start

### 1. Install dependencies

```bash
pip install streamlit matplotlib numpy
```

### 2. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### 3. Run unit tests

```bash
pip install pytest
python -m pytest tests/test_geometry.py -v
```

---

## File structure

```
oblique_survey/
├── app.py                  # Streamlit UI
├── geometry.py             # Pure geometry functions (no Streamlit)
├── tests/
│   └── test_geometry.py    # Unit tests for geometry module
├── presets.json            # Saved camera presets (auto-created)
├── *.json                  # Saved scenarios
└── README.md               # This file
```

---

## Camera arrangements

| Mode | Description |
|------|-------------|
| **Single nadir** | One camera pointing straight down (tilt = 0°) |
| **2 oblique** | One camera left, one right at the specified tilt angle |
| **4 oblique** | Two cameras left, two right (simplified: all pointing purely across-track) |

> ⚠️ The 4-oblique model assumes all cameras are tilted **purely across-track**. Real rigs (e.g. Leica RCD30 Oblique, IGI Penta-DigiCAM) may have both across- and along-track angular components. In that case, extend `calculate_camera_solution()` to accept both tilt axes.

---

## Angle convention

The app supports two conventions, selectable in the sidebar:

| Convention | 0° means | 90° means |
|-----------|----------|-----------|
| **Tilt from nadir** | Camera pointing straight down | Camera pointing at horizon |
| **Tilt from horizontal** | Camera pointing at horizon | Camera pointing straight down |

Internally, all calculations use **tilt from nadir**.

---

## Geometry assumptions (v1)

1. **Flat terrain** — ground is a horizontal plane at the stated AGL altitude.
2. **Pinhole camera** — no lens distortion, no vignetting.
3. **Square pixels** — pixel size = sensor dimension / image dimension.
4. **GSD formula**: `GSD = (pixel_size / focal_length) × altitude / cos²(angle_from_nadir)`
   - The `cos²` factor: one for slant range, one for oblique foreshortening.
5. **Along-track footprint** uses the *image centre* slant range (approximation — see geometry.py).
6. **Line spacing** is derived from the combined swath across all cameras and the target sidelap.
7. **No wind drift, crab angle, or IMU effects**.

---

## Saving scenarios and presets

- **Save scenario**: stores all current inputs as a JSON file (named in the sidebar).
- **Save camera preset**: stores sensor/focal specs so you can reload them next session.
- Preset file: `presets.json` (auto-created on first save).

---

## Key formulas (summary)

```
pixel_size_mm         = sensor_width_mm / image_width_px

half_fov_across       = atan(sensor_width / (2 × focal_length))

near_edge_ground      = altitude × tan(tilt - half_fov_across)
centre_ground         = altitude × tan(tilt)
far_edge_ground       = altitude × tan(tilt + half_fov_across)

slant_range           = altitude / cos(angle_from_nadir)

GSD                   = (pixel_size_mm / focal_length_mm) × altitude / cos²(angle)

footprint_across      = far_edge - near_edge
footprint_along       = 2 × centre_slant × tan(half_fov_along)

combined_swath        = rightmost_far_edge - leftmost_near_edge
line_spacing          = combined_swath × (1 - sidelap)
photo_spacing         = footprint_along × (1 - forward_overlap)
exposure_interval     = photo_spacing / aircraft_speed
```

---

## Extending the app

The geometry module (`geometry.py`) is deliberately separated from the UI (`app.py`). All functions are pure (no side effects, no Streamlit calls), making them easy to:

- Test independently (see `tests/test_geometry.py`)
- Import into other scripts or Jupyter notebooks
- Extend with terrain variation (pass a DEM-derived height array to `ground_intersections_flat_terrain`)

---

## Verification checklist before operational use

- [ ] Confirm your camera's actual pixel size matches `sensor_dim / image_dim`
- [ ] Verify focal length is the calibrated value (not nominal)
- [ ] Cross-check GSD at nadir against your photogrammetry software
- [ ] Validate along-track footprint against a test flight log
- [ ] Confirm exposure interval is achievable by your camera (check manufacturer spec)
- [ ] For undulating terrain, apply a terrain safety margin to the altitude
- [ ] Test reciprocal strip coverage in your flight planning software (e.g. UgCS, Mission Planner)
