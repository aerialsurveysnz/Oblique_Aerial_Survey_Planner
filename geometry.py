"""
geometry.py
===========
Pure geometry functions for 4-camera oblique aerial survey planning.

Coordinate convention
---------------------
  x  →  across-track (positive to the right when looking forward)
  y  →  along-track  (positive forward)
  z  →  vertical     (positive up)

Angle convention
----------------
  tilt_from_nadir   : 0° = straight down (nadir), 90° = horizontal
  tilt_from_horiz   : 0° = horizontal, 90° = straight down (nadir)

The public API always works in *tilt from nadir* internally.
Use normalize_tilt_angle() to convert.

Camera model
------------
Pinhole camera.  GSD is derived from the slant range and the pixel
angular size, which itself comes from:
    pixel_size_mm = sensor_size_mm / image_dimension_px
    pixel angular size = pixel_size_mm / focal_length_mm   (radians, small angle)

Ground intersections (flat terrain)
-------------------------------------
For a camera tilted at angle θ (from nadir) at altitude H above flat ground:

  nadir_offset = H * tan(θ)          ← ground distance directly below camera

The image spans ±½ the sensor FOV.  For the across-track axis:

  half_fov = atan(sensor_width / (2 * focal_length))

  near_edge_ground  = H * tan(θ - half_fov_across)   # closer to track
  centre_ground     = H * tan(θ)                      # image centre ray
  far_edge_ground   = H * tan(θ + half_fov_across)   # farther from track

  CAUTION: if (θ - half_fov) ≤ 0 the near edge crosses nadir and the formula
  below nadir requires special handling (still works, gives negative value which
  is on the opposite side of the track).

GSD at a ground point
----------------------
For a point at slant range R and across-track angle α from camera axis:

  slant_range = H / cos(α)          where α = angle from nadir to that ray

  GSD ≈ pixel_size_mm / focal_length_mm * slant_range
      = (pixel_size_mm / focal_length_mm) * (H / cos(α))

This is the GSD in the slant plane.  On flat ground the across-track GSD is
further stretched by 1/cos(α), giving:

  GSD_ground = pixel_size_mm / focal_length_mm * H / cos²(α)

  NOTE: This is a first-order approximation (pinhole, flat terrain).
  Real-world atmospheric refraction and lens distortion are ignored.
"""

import math
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def m_to_unit(value_m: float, unit: str) -> float:
    """Convert metres to the requested display unit."""
    factors = {"m": 1.0, "ft": 3.28084, "cm": 100.0, "mm": 1000.0}
    return value_m * factors[unit]


def unit_to_m(value: float, unit: str) -> float:
    """Convert from display unit to metres."""
    factors = {"m": 1.0, "ft": 0.3048, "cm": 0.01, "mm": 0.001}
    return value * factors[unit]


def mm_to_unit(value_mm: float, unit: str) -> float:
    """Convert millimetres to the requested unit."""
    return m_to_unit(value_mm * 0.001, unit)


# ---------------------------------------------------------------------------
# Camera / sensor helpers
# ---------------------------------------------------------------------------

def pixel_size_mm(sensor_dim_mm: float, image_dim_px: int) -> float:
    """
    Return the physical size of one pixel in millimetres.

    pixel_size = sensor_dimension_mm / image_dimension_px

    Args:
        sensor_dim_mm: sensor width or height in mm
        image_dim_px:  corresponding image dimension in pixels

    Returns:
        pixel size in mm
    """
    if image_dim_px <= 0:
        raise ValueError("image_dim_px must be > 0")
    if sensor_dim_mm <= 0:
        raise ValueError("sensor_dim_mm must be > 0")
    return sensor_dim_mm / image_dim_px


def focal_length_px(focal_length_mm: float, px_size_mm: float) -> float:
    """
    Return focal length expressed in pixels.

    f_px = focal_length_mm / pixel_size_mm

    Args:
        focal_length_mm: focal length in mm
        px_size_mm:      pixel size in mm (from pixel_size_mm())

    Returns:
        focal length in pixels
    """
    if px_size_mm <= 0:
        raise ValueError("px_size_mm must be > 0")
    return focal_length_mm / px_size_mm


def normalize_tilt_angle(angle_deg: float, convention: str) -> float:
    """
    Return the tilt angle measured *from nadir* (degrees).

    Args:
        angle_deg:  tilt angle in degrees
        convention: 'nadir'  → angle already from nadir (0 = straight down)
                    'horiz'  → angle from horizontal (0 = horizontal, 90 = nadir)

    Returns:
        tilt from nadir in degrees  (0 … 90, or negative if past nadir)
    """
    if convention == "nadir":
        return float(angle_deg)
    elif convention == "horiz":
        return 90.0 - float(angle_deg)
    else:
        raise ValueError(f"Unknown angle convention: {convention!r}. Use 'nadir' or 'horiz'.")


def half_fov_deg(sensor_dim_mm: float, focal_length_mm: float) -> float:
    """
    Half field-of-view for a given sensor dimension and focal length.

    half_fov = atan(sensor_dim / (2 * focal_length))   (degrees)

    Args:
        sensor_dim_mm:   sensor width or height in mm
        focal_length_mm: focal length in mm

    Returns:
        half FOV in degrees
    """
    return math.degrees(math.atan(sensor_dim_mm / (2.0 * focal_length_mm)))


# ---------------------------------------------------------------------------
# Flat-terrain ground intersections
# ---------------------------------------------------------------------------

@dataclass
class GroundIntersections:
    """
    Across-track ground intercept distances from the aircraft nadir track.

    Positive values are on the far side (away from track centre for oblique),
    negative values are on the near side (towards or past the track).

    All distances in metres.
    """
    near_edge_m: float    # nearest edge of image footprint (across-track)
    centre_m: float       # image centre ray intercept
    far_edge_m: float     # farthest edge of image footprint
    near_slant_m: float   # slant range to near edge
    centre_slant_m: float # slant range to image centre
    far_slant_m: float    # slant range to far edge
    near_angle_deg: float # ray angle from nadir at near edge
    centre_angle_deg: float
    far_angle_deg: float


def ground_intersections_flat_terrain(
    altitude_m: float,
    tilt_from_nadir_deg: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    focal_length_mm: float,
) -> GroundIntersections:
    """
    Compute across-track ground intercepts on flat terrain (pinhole model).

    The camera is tilted *across track* by tilt_from_nadir_deg.
    We compute where the near edge, image centre, and far edge of the
    image footprint hit the ground.

    Geometry (side view, looking along track):

            Aircraft (H above ground)
                |
                |   (nadir)
               /|\\
              / | \\
             /  |  \\
    near_edge  nadir  far_edge

    For tilt θ from nadir, half-FOV φ (across track):

        near_edge_angle = θ - φ   (from nadir)
        centre_angle    = θ
        far_edge_angle  = θ + φ

        ground_distance = H * tan(angle_from_nadir)
        slant_range     = H / cos(angle_from_nadir)

    ASSUMPTIONS:
    - Flat terrain (z = 0)
    - Pure rotation about the along-track axis (roll)
    - Pinhole camera
    - No atmospheric refraction

    Args:
        altitude_m:            aircraft altitude AGL in metres
        tilt_from_nadir_deg:   camera tilt from nadir in degrees (0 = nadir)
        sensor_width_mm:       across-track sensor dimension in mm
        sensor_height_mm:      along-track sensor dimension in mm
        focal_length_mm:       focal length in mm

    Returns:
        GroundIntersections dataclass
    """
    if altitude_m <= 0:
        raise ValueError("altitude_m must be > 0")

    phi_deg = half_fov_deg(sensor_width_mm, focal_length_mm)  # across-track half-FOV
    theta = tilt_from_nadir_deg  # camera tilt from nadir

    near_angle = theta - phi_deg
    centre_angle = theta
    far_angle = theta + phi_deg

    def _ground_dist(angle_deg: float) -> float:
        """Signed ground distance from nadir (positive = away from aircraft track centre)."""
        a_rad = math.radians(angle_deg)
        if abs(a_rad) >= math.pi / 2:
            return math.copysign(float("inf"), angle_deg)
        return altitude_m * math.tan(a_rad)

    def _slant(angle_deg: float) -> float:
        a_rad = math.radians(angle_deg)
        if abs(a_rad) >= math.pi / 2:
            return float("inf")
        return altitude_m / math.cos(a_rad)

    return GroundIntersections(
        near_edge_m=_ground_dist(near_angle),
        centre_m=_ground_dist(centre_angle),
        far_edge_m=_ground_dist(far_angle),
        near_slant_m=_slant(near_angle),
        centre_slant_m=_slant(centre_angle),
        far_slant_m=_slant(far_angle),
        near_angle_deg=near_angle,
        centre_angle_deg=centre_angle,
        far_angle_deg=far_angle,
    )


# ---------------------------------------------------------------------------
# GSD calculation
# ---------------------------------------------------------------------------

def gsd_at_ground_point(
    altitude_m: float,
    angle_from_nadir_deg: float,
    sensor_width_mm: float,
    image_width_px: int,
    focal_length_mm: float,
) -> float:
    """
    Ground sample distance at a point defined by its nadir angle.

    For a pinhole camera on flat ground, the across-track GSD at angle α
    from nadir is:

        px_size = sensor_width_mm / image_width_px
        GSD     = px_size / focal_length_mm * H / cos²(α)   [metres]

    The cos²(α) comes from:
      - first cos(α) → slant range stretching (more distant)
      - second cos(α) → oblique viewing angle (foreshortening on ground)

    ASSUMPTION: flat terrain, pinhole model.

    Args:
        altitude_m:           aircraft altitude AGL in metres
        angle_from_nadir_deg: viewing angle from nadir in degrees
        sensor_width_mm:      sensor width in mm (across-track dimension)
        image_width_px:       image width in pixels
        focal_length_mm:      focal length in mm

    Returns:
        GSD in metres per pixel
    """
    px_size = pixel_size_mm(sensor_width_mm, image_width_px)  # mm
    # Convert to metres for consistent units
    px_size_m = px_size * 0.001
    fl_m = focal_length_mm * 0.001

    a_rad = math.radians(angle_from_nadir_deg)
    cos_a = math.cos(a_rad)
    if abs(cos_a) < 1e-9:
        return float("inf")

    # GSD = (pixel_size / focal_length) * slant_range
    # slant_range = H / cos(a)
    # Projected onto horizontal ground: multiply by 1/cos(a) again
    return (px_size_m / fl_m) * (altitude_m / (cos_a ** 2))


# ---------------------------------------------------------------------------
# Footprint and swath
# ---------------------------------------------------------------------------

@dataclass
class FootprintDimensions:
    """Footprint of one camera image on flat ground, in metres."""
    across_track_m: float   # width of footprint across the flight track
    along_track_m: float    # length of footprint along the flight track
    near_edge_m: float      # across-track distance from nadir to near edge
    far_edge_m: float       # across-track distance from nadir to far edge


def footprint_dimensions(
    altitude_m: float,
    tilt_from_nadir_deg: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    focal_length_mm: float,
) -> FootprintDimensions:
    """
    Compute the ground footprint of one camera (flat terrain, pinhole model).

    Across-track footprint = far_edge_ground - near_edge_ground

    Along-track footprint uses the *along-track* sensor dimension (height)
    and the slant range to the image centre:

        along_fov_half = atan(sensor_height / (2 * focal_length))
        along_track_footprint = 2 * centre_slant * tan(along_fov_half)

    This approximates the along-track extent as if the ground were
    perpendicular to the line of sight at the image centre.
    For oblique cameras over flat terrain this is approximate; a full
    4-corner projection is more accurate but adds complexity.

    ASSUMPTION: along-track footprint uses centre slant range (approximation).

    Args:
        altitude_m:           AGL altitude in metres
        tilt_from_nadir_deg:  camera tilt from nadir in degrees
        sensor_width_mm:      across-track sensor dimension in mm
        sensor_height_mm:     along-track sensor dimension in mm
        focal_length_mm:      focal length in mm

    Returns:
        FootprintDimensions dataclass
    """
    gi = ground_intersections_flat_terrain(
        altitude_m, tilt_from_nadir_deg, sensor_width_mm, sensor_height_mm, focal_length_mm
    )

    across_track = gi.far_edge_m - gi.near_edge_m  # always positive

    # Along-track footprint: use centre slant range
    along_half_fov = math.atan(sensor_height_mm / (2.0 * focal_length_mm))
    along_track = 2.0 * gi.centre_slant_m * math.tan(along_half_fov)

    return FootprintDimensions(
        across_track_m=across_track,
        along_track_m=along_track,
        near_edge_m=gi.near_edge_m,
        far_edge_m=gi.far_edge_m,
    )


def effective_swath_from_sidelap(
    footprint_across_m: float,
    sidelap_fraction: float,
) -> float:
    """
    Usable (non-overlapping) swath width for one camera strip.

    effective_swath = footprint_across * (1 - sidelap_fraction)

    Args:
        footprint_across_m: across-track footprint width in metres
        sidelap_fraction:   sidelap as a fraction (0.0–1.0), e.g. 0.3 for 30%

    Returns:
        Effective swath width in metres
    """
    if not 0 <= sidelap_fraction < 1:
        raise ValueError("sidelap_fraction must be in [0, 1)")
    return footprint_across_m * (1.0 - sidelap_fraction)


def line_spacing_from_sidelap(
    footprint_across_m: float,
    sidelap_fraction: float,
) -> float:
    """
    Recommended flight line spacing to achieve the target sidelap.

    line_spacing = footprint_across * (1 - sidelap_fraction)

    This is the distance between adjacent flight line nadir tracks such that
    the overlapping image strips have the requested sidelap.

    Args:
        footprint_across_m: across-track footprint width in metres
        sidelap_fraction:   desired sidelap as a fraction

    Returns:
        Line spacing in metres
    """
    return effective_swath_from_sidelap(footprint_across_m, sidelap_fraction)


def photo_spacing_from_forward_overlap(
    footprint_along_m: float,
    forward_overlap_fraction: float,
) -> float:
    """
    Recommended photo interval (metres along-track) for target forward overlap.

    photo_spacing = footprint_along * (1 - forward_overlap_fraction)

    Args:
        footprint_along_m:       along-track footprint in metres
        forward_overlap_fraction: desired forward overlap as a fraction

    Returns:
        Photo spacing in metres
    """
    if not 0 <= forward_overlap_fraction < 1:
        raise ValueError("forward_overlap_fraction must be in [0, 1)")
    return footprint_along_m * (1.0 - forward_overlap_fraction)


# ---------------------------------------------------------------------------
# Per-camera solution
# ---------------------------------------------------------------------------

@dataclass
class CameraSolution:
    """Complete geometry solution for one camera."""
    # Ground intercepts
    near_edge_m: float
    centre_m: float
    far_edge_m: float
    near_slant_m: float
    centre_slant_m: float
    far_slant_m: float
    near_angle_deg: float
    centre_angle_deg: float
    far_angle_deg: float
    # GSD
    near_gsd_m: float
    centre_gsd_m: float
    far_gsd_m: float
    # Footprint
    footprint_across_m: float
    footprint_along_m: float
    # Pixel size
    pixel_size_mm: float
    # FOV
    half_fov_across_deg: float
    half_fov_along_deg: float


def calculate_camera_solution(
    altitude_m: float,
    tilt_from_nadir_deg: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    image_width_px: int,
    image_height_px: int,
    focal_length_mm: float,
) -> CameraSolution:
    """
    Full geometry solution for a single oblique camera.

    Args:
        altitude_m:          AGL altitude in metres
        tilt_from_nadir_deg: camera tilt from nadir (degrees)
        sensor_width_mm:     across-track sensor dimension (mm)
        sensor_height_mm:    along-track sensor dimension (mm)
        image_width_px:      image width in pixels
        image_height_px:     image height in pixels
        focal_length_mm:     focal length in mm

    Returns:
        CameraSolution dataclass with all intermediate and final values
    """
    gi = ground_intersections_flat_terrain(
        altitude_m, tilt_from_nadir_deg,
        sensor_width_mm, sensor_height_mm, focal_length_mm,
    )

    fp = footprint_dimensions(
        altitude_m, tilt_from_nadir_deg,
        sensor_width_mm, sensor_height_mm, focal_length_mm,
    )

    px_sz = pixel_size_mm(sensor_width_mm, image_width_px)
    hfov_a = half_fov_deg(sensor_width_mm, focal_length_mm)
    hfov_l = half_fov_deg(sensor_height_mm, focal_length_mm)

    near_gsd = gsd_at_ground_point(altitude_m, gi.near_angle_deg, sensor_width_mm, image_width_px, focal_length_mm)
    ctr_gsd = gsd_at_ground_point(altitude_m, gi.centre_angle_deg, sensor_width_mm, image_width_px, focal_length_mm)
    far_gsd = gsd_at_ground_point(altitude_m, gi.far_angle_deg, sensor_width_mm, image_width_px, focal_length_mm)

    return CameraSolution(
        near_edge_m=gi.near_edge_m,
        centre_m=gi.centre_m,
        far_edge_m=gi.far_edge_m,
        near_slant_m=gi.near_slant_m,
        centre_slant_m=gi.centre_slant_m,
        far_slant_m=gi.far_slant_m,
        near_angle_deg=gi.near_angle_deg,
        centre_angle_deg=gi.centre_angle_deg,
        far_angle_deg=gi.far_angle_deg,
        near_gsd_m=near_gsd,
        centre_gsd_m=ctr_gsd,
        far_gsd_m=far_gsd,
        footprint_across_m=fp.across_track_m,
        footprint_along_m=fp.along_track_m,
        pixel_size_mm=px_sz,
        half_fov_across_deg=hfov_a,
        half_fov_along_deg=hfov_l,
    )


# ---------------------------------------------------------------------------
# Multi-camera system solution
# ---------------------------------------------------------------------------

@dataclass
class MultiCameraSolution:
    """System-level outputs for a multi-camera oblique array."""
    combined_swath_m: float              # total across-track ground coverage
    recommended_line_spacing_m: float   # between nadir track lines
    recommended_photo_spacing_m: float  # along-track exposure interval
    photo_interval_s: float             # time between exposures at given speed
    # Overlaps achieved at the footprint of one camera
    forward_overlap_near: float         # fraction
    forward_overlap_centre: float       # fraction
    forward_overlap_far: float          # fraction
    sidelap_achieved: float             # fraction (system level)
    warnings: list                      # list of warning strings
    reciprocal_recommended: bool


def calculate_multicamera_solution(
    camera_solutions: list,             # list of CameraSolution (one per camera)
    arrangement: str,                   # '4_oblique', '2_oblique', 'single_nadir'
    altitude_m: float,
    aircraft_speed_ms: float,
    forward_overlap_fraction: float,
    sidelap_fraction: float,
    reciprocal_flying: bool,
) -> MultiCameraSolution:
    """
    Compute system-level outputs from individual camera solutions.

    For a 4-camera oblique (2 left + 2 right or symmetric):
    - Combined swath = leftmost near edge to rightmost far edge
    - Line spacing is based on the combined swath and target sidelap

    For 2-camera oblique (1 left + 1 right):
    - Similar to 4-camera but with only one camera per side

    For single nadir:
    - Standard nadir geometry

    Overlap at near/centre/far edge is computed from the photo spacing
    vs. the along-track footprint.

    ASSUMPTION:
    - All cameras have the same along-track footprint (same sensor, same FL)
    - Along-track footprint is taken from the first camera solution
    - Sidelap applies to the combined swath

    Args:
        camera_solutions:        list of CameraSolution objects
        arrangement:             camera arrangement string
        altitude_m:              AGL altitude
        aircraft_speed_ms:       aircraft speed in m/s
        forward_overlap_fraction: target forward overlap (fraction)
        sidelap_fraction:        target sidelap (fraction)
        reciprocal_flying:       True if flying reciprocal strips

    Returns:
        MultiCameraSolution
    """
    warnings = []

    if not camera_solutions:
        raise ValueError("No camera solutions provided")

    # --- Combined swath ---
    # Find leftmost near edge and rightmost far edge across all cameras
    all_near = [cs.near_edge_m for cs in camera_solutions]
    all_far = [cs.far_edge_m for cs in camera_solutions]
    leftmost = min(all_near)   # could be negative (past nadir)
    rightmost = max(all_far)
    combined_swath = rightmost - leftmost

    # --- Line spacing ---
    line_spacing = line_spacing_from_sidelap(combined_swath, sidelap_fraction)

    # --- Photo spacing (along-track) ---
    # Use first camera's along-track footprint
    along_footprint = camera_solutions[0].footprint_along_m
    photo_spacing = photo_spacing_from_forward_overlap(along_footprint, forward_overlap_fraction)
    photo_interval_s = photo_spacing / aircraft_speed_ms if aircraft_speed_ms > 0 else float("inf")

    # --- Achieved overlaps ---
    # forward overlap achieved = 1 - photo_spacing / along_footprint_at_position
    # For near/centre/far, the along-track footprint varies slightly with slant range.
    # We approximate: scale along-track footprint by ratio of slant ranges.
    ref_slant = camera_solutions[0].centre_slant_m
    ref_along = camera_solutions[0].footprint_along_m

    def _overlap(slant_m):
        scaled_along = ref_along * (slant_m / ref_slant) if ref_slant > 0 else ref_along
        if scaled_along <= 0:
            return 0.0
        ov = 1.0 - photo_spacing / scaled_along
        return max(0.0, min(1.0, ov))

    fwd_near = _overlap(camera_solutions[0].near_slant_m)
    fwd_ctr = _overlap(camera_solutions[0].centre_slant_m)
    fwd_far = _overlap(camera_solutions[0].far_slant_m)

    # Sidelap achieved: based on combined swath vs line spacing
    sidelap_achieved = 1.0 - line_spacing / combined_swath if combined_swath > 0 else 0.0

    # --- Reciprocal recommendation ---
    # Recommend reciprocal if tilt > 0 (oblique) to ensure even coverage
    reciprocal_recommended = any(abs(cs.centre_angle_deg) > 5 for cs in camera_solutions)

    # --- Warnings ---
    for i, cs in enumerate(camera_solutions):
        if cs.far_angle_deg >= 85:
            warnings.append(f"Camera {i+1}: far edge angle {cs.far_angle_deg:.1f}° approaches horizon — GSD will be very large.")
        if cs.near_gsd_m > 0 and cs.far_gsd_m / cs.near_gsd_m > 5:
            warnings.append(f"Camera {i+1}: GSD ratio near/far > 5× ({cs.near_gsd_m*100:.1f} cm → {cs.far_gsd_m*100:.1f} cm). Consider a smaller tilt angle.")
        if cs.footprint_along_m <= 0:
            warnings.append(f"Camera {i+1}: along-track footprint is zero or negative — check inputs.")

    if photo_interval_s < 1.0:
        warnings.append(f"Exposure interval {photo_interval_s:.2f} s is very short. Verify your camera's minimum interval.")
    if line_spacing <= 0:
        warnings.append("Computed line spacing is zero or negative. Sidelap may be >= 1.")
    if combined_swath <= 0:
        warnings.append("Combined swath is zero — check altitude and tilt angle.")

    return MultiCameraSolution(
        combined_swath_m=combined_swath,
        recommended_line_spacing_m=line_spacing,
        recommended_photo_spacing_m=photo_spacing,
        photo_interval_s=photo_interval_s,
        forward_overlap_near=fwd_near,
        forward_overlap_centre=fwd_ctr,
        forward_overlap_far=fwd_far,
        sidelap_achieved=sidelap_achieved,
        warnings=warnings,
        reciprocal_recommended=reciprocal_recommended,
    )
