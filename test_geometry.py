"""
tests/test_geometry.py
======================
Unit tests for the oblique aerial survey geometry module.

Run with:
    python -m pytest tests/test_geometry.py -v
or from the project root:
    python -m pytest -v
"""

import math
import sys
import os
import pytest

# Allow import from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometry import (
    pixel_size_mm,
    focal_length_px,
    normalize_tilt_angle,
    half_fov_deg,
    ground_intersections_flat_terrain,
    gsd_at_ground_point,
    footprint_dimensions,
    effective_swath_from_sidelap,
    line_spacing_from_sidelap,
    photo_spacing_from_forward_overlap,
    calculate_camera_solution,
    calculate_multicamera_solution,
    m_to_unit,
    unit_to_m,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SONY_A7RIV = dict(
    sensor_width_mm=35.7,
    sensor_height_mm=23.8,
    image_width_px=9504,
    image_height_px=6336,
    focal_length_mm=35.0,
)


def approx(x, rel=1e-4):
    """Shorthand for pytest.approx with sensible tolerance."""
    return pytest.approx(x, rel=rel)


# ---------------------------------------------------------------------------
# pixel_size_mm
# ---------------------------------------------------------------------------

class TestPixelSizeMm:
    def test_known_value(self):
        # Sony A7R IV: 35.7 mm / 9504 px = 3.756 µm
        result = pixel_size_mm(35.7, 9504)
        assert result == approx(35.7 / 9504)

    def test_square_sensor(self):
        result = pixel_size_mm(24.0, 6000)
        assert result == approx(0.004)

    def test_raises_zero_pixels(self):
        with pytest.raises(ValueError):
            pixel_size_mm(24.0, 0)

    def test_raises_zero_sensor(self):
        with pytest.raises(ValueError):
            pixel_size_mm(0.0, 6000)

    def test_raises_negative_pixels(self):
        with pytest.raises(ValueError):
            pixel_size_mm(24.0, -100)


# ---------------------------------------------------------------------------
# focal_length_px
# ---------------------------------------------------------------------------

class TestFocalLengthPx:
    def test_known_value(self):
        # 35mm lens, 3.756µm pixel → f_px = 35 / 0.003756 ≈ 9318
        px_sz = pixel_size_mm(35.7, 9504)
        f_px = focal_length_px(35.0, px_sz)
        assert f_px == approx(35.0 / px_sz)

    def test_raises_zero_pixel_size(self):
        with pytest.raises(ValueError):
            focal_length_px(35.0, 0.0)


# ---------------------------------------------------------------------------
# normalize_tilt_angle
# ---------------------------------------------------------------------------

class TestNormalizeTiltAngle:
    def test_nadir_convention_passthrough(self):
        assert normalize_tilt_angle(45.0, "nadir") == approx(45.0)

    def test_horiz_convention_nadir_is_90(self):
        # 0° from horizontal = 90° from nadir
        assert normalize_tilt_angle(0.0, "horiz") == approx(90.0)

    def test_horiz_convention_45(self):
        # 45° from horizontal = 45° from nadir
        assert normalize_tilt_angle(45.0, "horiz") == approx(45.0)

    def test_horiz_convention_horizontal_is_0(self):
        # 90° from horizontal = 0° from nadir (pointing straight down)
        assert normalize_tilt_angle(90.0, "horiz") == approx(0.0)

    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError):
            normalize_tilt_angle(45.0, "unknown")


# ---------------------------------------------------------------------------
# half_fov_deg
# ---------------------------------------------------------------------------

class TestHalfFovDeg:
    def test_known_value(self):
        # sensor_width=35.7mm, fl=35mm → half_fov = atan(35.7/70) ≈ 27.0°
        result = half_fov_deg(35.7, 35.0)
        expected = math.degrees(math.atan(35.7 / 70.0))
        assert result == approx(expected)

    def test_nadir_camera_fov(self):
        # 24mm full-frame, 24mm lens → half_fov = atan(24/48) = atan(0.5) = 26.57°
        result = half_fov_deg(24.0, 24.0)
        expected = math.degrees(math.atan(0.5))
        assert result == approx(expected)


# ---------------------------------------------------------------------------
# ground_intersections_flat_terrain
# ---------------------------------------------------------------------------

class TestGroundIntersections:
    def test_nadir_camera_symmetric(self):
        """For 0° tilt, near and far edges should be symmetric about nadir."""
        gi = ground_intersections_flat_terrain(
            altitude_m=1000.0,
            tilt_from_nadir_deg=0.0,
            sensor_width_mm=35.7,
            sensor_height_mm=23.8,
            focal_length_mm=35.0,
        )
        # Centre should be at 0 (directly below)
        assert gi.centre_m == approx(0.0, abs=1e-9)
        # Near and far should be symmetric
        assert gi.near_edge_m == approx(-gi.far_edge_m)
        # All distances positive for far edge, negative for near
        assert gi.far_edge_m > 0
        assert gi.near_edge_m < 0

    def test_oblique_far_edge_farther(self):
        """For oblique camera, far edge must be farther than centre."""
        gi = ground_intersections_flat_terrain(
            altitude_m=1000.0,
            tilt_from_nadir_deg=45.0,
            **{k: SONY_A7RIV[k] for k in ("sensor_width_mm", "sensor_height_mm", "focal_length_mm")},
        )
        assert gi.far_edge_m > gi.centre_m
        assert gi.centre_m > gi.near_edge_m

    def test_centre_at_45_deg(self):
        """At 45° tilt, centre ray hits ground at H * tan(45°) = H."""
        h = 1000.0
        gi = ground_intersections_flat_terrain(
            altitude_m=h,
            tilt_from_nadir_deg=45.0,
            sensor_width_mm=35.7,
            sensor_height_mm=23.8,
            focal_length_mm=35.0,
        )
        assert gi.centre_m == approx(h * math.tan(math.radians(45.0)))

    def test_slant_range_nadir(self):
        """For 0° tilt, centre slant range equals altitude."""
        gi = ground_intersections_flat_terrain(
            altitude_m=500.0,
            tilt_from_nadir_deg=0.0,
            sensor_width_mm=35.7,
            sensor_height_mm=23.8,
            focal_length_mm=35.0,
        )
        assert gi.centre_slant_m == approx(500.0)

    def test_slant_range_oblique(self):
        """Slant range at 45° = H / cos(45°) = H * sqrt(2)."""
        h = 1000.0
        gi = ground_intersections_flat_terrain(
            altitude_m=h,
            tilt_from_nadir_deg=45.0,
            sensor_width_mm=35.7,
            sensor_height_mm=23.8,
            focal_length_mm=35.0,
        )
        expected_slant = h / math.cos(math.radians(45.0))
        assert gi.centre_slant_m == approx(expected_slant)

    def test_raises_zero_altitude(self):
        with pytest.raises(ValueError):
            ground_intersections_flat_terrain(0.0, 30.0, 35.7, 23.8, 35.0)

    def test_raises_negative_altitude(self):
        with pytest.raises(ValueError):
            ground_intersections_flat_terrain(-100.0, 30.0, 35.7, 23.8, 35.0)


# ---------------------------------------------------------------------------
# gsd_at_ground_point
# ---------------------------------------------------------------------------

class TestGsdAtGroundPoint:
    def test_nadir_gsd(self):
        """At 0° (nadir), GSD = pixel_size * altitude / focal_length."""
        h = 1000.0
        px_sz = pixel_size_mm(35.7, 9504)  # mm
        fl = 35.0
        expected = (px_sz * 0.001) / (fl * 0.001) * h  # metres
        result = gsd_at_ground_point(h, 0.0, 35.7, 9504, fl)
        assert result == approx(expected)

    def test_oblique_larger_gsd(self):
        """Oblique GSD must be larger than nadir GSD."""
        nadir_gsd = gsd_at_ground_point(1000.0, 0.0, 35.7, 9504, 35.0)
        oblique_gsd = gsd_at_ground_point(1000.0, 45.0, 35.7, 9504, 35.0)
        assert oblique_gsd > nadir_gsd

    def test_gsd_scales_with_altitude(self):
        """GSD should double when altitude doubles (at nadir)."""
        gsd1 = gsd_at_ground_point(500.0, 0.0, 35.7, 9504, 35.0)
        gsd2 = gsd_at_ground_point(1000.0, 0.0, 35.7, 9504, 35.0)
        assert gsd2 == approx(2 * gsd1)


# ---------------------------------------------------------------------------
# footprint_dimensions
# ---------------------------------------------------------------------------

class TestFootprintDimensions:
    def test_nadir_footprint_symmetric(self):
        """Nadir footprint: across = 2 * near_edge magnitude."""
        fp = footprint_dimensions(1000.0, 0.0, 35.7, 23.8, 35.0)
        # near and far should be symmetric, across = |near| + far
        assert fp.across_track_m > 0
        assert fp.along_track_m > 0
        # For nadir, near = -far
        assert fp.near_edge_m == approx(-fp.far_edge_m)

    def test_footprint_increases_with_altitude(self):
        fp1 = footprint_dimensions(500.0, 30.0, 35.7, 23.8, 35.0)
        fp2 = footprint_dimensions(1000.0, 30.0, 35.7, 23.8, 35.0)
        assert fp2.across_track_m > fp1.across_track_m
        assert fp2.along_track_m > fp1.along_track_m

    def test_footprint_increases_with_tilt(self):
        fp1 = footprint_dimensions(1000.0, 20.0, 35.7, 23.8, 35.0)
        fp2 = footprint_dimensions(1000.0, 40.0, 35.7, 23.8, 35.0)
        assert fp2.across_track_m > fp1.across_track_m


# ---------------------------------------------------------------------------
# effective_swath_from_sidelap
# ---------------------------------------------------------------------------

class TestEffectiveSwath:
    def test_zero_sidelap(self):
        assert effective_swath_from_sidelap(100.0, 0.0) == approx(100.0)

    def test_thirty_percent_sidelap(self):
        assert effective_swath_from_sidelap(100.0, 0.3) == approx(70.0)

    def test_raises_sidelap_ge_1(self):
        with pytest.raises(ValueError):
            effective_swath_from_sidelap(100.0, 1.0)

    def test_raises_negative_sidelap(self):
        with pytest.raises(ValueError):
            effective_swath_from_sidelap(100.0, -0.1)


# ---------------------------------------------------------------------------
# line_spacing_from_sidelap
# ---------------------------------------------------------------------------

class TestLineSpacing:
    def test_thirty_percent_sidelap(self):
        spacing = line_spacing_from_sidelap(200.0, 0.30)
        assert spacing == approx(140.0)

    def test_zero_sidelap(self):
        spacing = line_spacing_from_sidelap(200.0, 0.0)
        assert spacing == approx(200.0)


# ---------------------------------------------------------------------------
# photo_spacing_from_forward_overlap
# ---------------------------------------------------------------------------

class TestPhotoSpacing:
    def test_sixty_percent_overlap(self):
        spacing = photo_spacing_from_forward_overlap(100.0, 0.60)
        assert spacing == approx(40.0)

    def test_eighty_percent_overlap(self):
        spacing = photo_spacing_from_forward_overlap(100.0, 0.80)
        assert spacing == approx(20.0)

    def test_raises_overlap_ge_1(self):
        with pytest.raises(ValueError):
            photo_spacing_from_forward_overlap(100.0, 1.0)


# ---------------------------------------------------------------------------
# calculate_camera_solution (integration)
# ---------------------------------------------------------------------------

class TestCameraSolution:
    def test_returns_valid_solution(self):
        sol = calculate_camera_solution(
            altitude_m=1000.0,
            tilt_from_nadir_deg=35.0,
            sensor_width_mm=35.7,
            sensor_height_mm=23.8,
            image_width_px=9504,
            image_height_px=6336,
            focal_length_mm=35.0,
        )
        assert sol.far_edge_m > sol.centre_m > sol.near_edge_m
        assert sol.far_gsd_m > sol.centre_gsd_m > sol.near_gsd_m
        assert sol.footprint_across_m > 0
        assert sol.footprint_along_m > 0

    def test_nadir_camera_solution(self):
        sol = calculate_camera_solution(
            altitude_m=500.0,
            tilt_from_nadir_deg=0.0,
            sensor_width_mm=35.7,
            sensor_height_mm=23.8,
            image_width_px=9504,
            image_height_px=6336,
            focal_length_mm=35.0,
        )
        assert sol.centre_m == approx(0.0, abs=1e-9)
        # At nadir, centre angle = 0 → centre GSD = pixel_size * H / focal_length
        px_sz_m = pixel_size_mm(35.7, 9504) * 0.001
        expected_centre_gsd = px_sz_m / (35.0 * 0.001) * 500.0
        assert sol.centre_gsd_m == approx(expected_centre_gsd)


# ---------------------------------------------------------------------------
# calculate_multicamera_solution (integration)
# ---------------------------------------------------------------------------

class TestMulticameraSolution:
    def _make_solutions(self, tilt=35.0, altitude=1000.0, n=4):
        sol = calculate_camera_solution(
            altitude_m=altitude,
            tilt_from_nadir_deg=tilt,
            **SONY_A7RIV,
        )
        return [sol] * n

    def test_combined_swath_positive(self):
        sols = self._make_solutions()
        mc = calculate_multicamera_solution(
            camera_solutions=sols,
            arrangement="4_oblique",
            altitude_m=1000.0,
            aircraft_speed_ms=50.0,
            forward_overlap_fraction=0.60,
            sidelap_fraction=0.30,
            reciprocal_flying=False,
        )
        assert mc.combined_swath_m > 0

    def test_line_spacing_less_than_swath(self):
        sols = self._make_solutions()
        mc = calculate_multicamera_solution(
            camera_solutions=sols,
            arrangement="4_oblique",
            altitude_m=1000.0,
            aircraft_speed_ms=50.0,
            forward_overlap_fraction=0.60,
            sidelap_fraction=0.30,
            reciprocal_flying=False,
        )
        assert mc.recommended_line_spacing_m < mc.combined_swath_m

    def test_photo_interval_positive(self):
        sols = self._make_solutions()
        mc = calculate_multicamera_solution(
            camera_solutions=sols,
            arrangement="4_oblique",
            altitude_m=1000.0,
            aircraft_speed_ms=50.0,
            forward_overlap_fraction=0.60,
            sidelap_fraction=0.30,
            reciprocal_flying=False,
        )
        assert mc.photo_interval_s > 0

    def test_reciprocal_recommended_for_oblique(self):
        sols = self._make_solutions(tilt=35.0)
        mc = calculate_multicamera_solution(
            camera_solutions=sols,
            arrangement="4_oblique",
            altitude_m=1000.0,
            aircraft_speed_ms=50.0,
            forward_overlap_fraction=0.60,
            sidelap_fraction=0.30,
            reciprocal_flying=False,
        )
        assert mc.reciprocal_recommended is True


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

class TestUnitConversion:
    def test_m_to_ft(self):
        assert m_to_unit(1.0, "ft") == approx(3.28084)

    def test_m_to_cm(self):
        assert m_to_unit(1.0, "cm") == approx(100.0)

    def test_ft_to_m(self):
        assert unit_to_m(3.28084, "ft") == approx(1.0, rel=1e-4)

    def test_roundtrip(self):
        original = 123.456
        assert unit_to_m(m_to_unit(original, "ft"), "ft") == approx(original)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
