"""Tests for the leading-edge mapping core (no ROS runtime required).

Run from the package root:  python3 -m pytest test/ -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sonar_mapping.leading_edge_core import (          # noqa: E402
    make_detector, leading_edge, polar_to_sonar_xy,
    sonar_to_body, body_to_map, scan_to_map_points,
)


# ---------------------------------------------------------------- geometry
def test_polar_to_sonar_xy_boresight():
    """A cell on the central beam maps straight ahead."""
    az = np.linspace(-1.13, 1.13, 512).astype(np.float32)   # ±65°
    rng = np.linspace(0.05, 10.0, 633).astype(np.float32)
    cols = np.array([256])            # central beam, az ≈ 0
    rows = np.array([300])            # cropped row; bin = 500
    x, y = polar_to_sonar_xy(cols, rows, az, rng, crop_row=200)
    assert abs(y[0]) < 0.05           # essentially no lateral offset
    assert abs(x[0] - rng[500]) < 0.05


def test_polar_to_sonar_xy_side_beam_sign():
    """Positive azimuth gives positive y (port/starboard consistency)."""
    az = np.linspace(-1.13, 1.13, 512).astype(np.float32)
    rng = np.linspace(0.05, 10.0, 633).astype(np.float32)
    x, y = polar_to_sonar_xy(np.array([500]), np.array([300]), az, rng, 200)
    assert y[0] > 0
    x2, y2 = polar_to_sonar_xy(np.array([11]), np.array([300]), az, rng, 200)
    assert y2[0] < 0


def test_sonar_to_body_pitch_projection():
    """The 20° mounting compresses forward range by cos(20°) and adds depth."""
    p = np.deg2rad(20.0)
    x_b, y_b, z_b = sonar_to_body(np.array([10.0]), np.array([0.0]), p)
    assert x_b[0] == pytest.approx(10.0 * np.cos(p))
    assert z_b[0] == pytest.approx(10.0 * np.sin(p))
    assert y_b[0] == 0.0


def test_sonar_to_body_lateral_unchanged():
    """Sway lies along the tilt axis — it must NOT be scaled by the pitch."""
    p = np.deg2rad(20.0)
    _, y_b, _ = sonar_to_body(np.array([0.0]), np.array([3.0]), p)
    assert y_b[0] == pytest.approx(3.0)


def test_body_to_map_yaw_and_translation():
    """90° yaw turns body-x into map-east; translation adds."""
    n, e, d = body_to_map(np.array([1.0]), np.array([0.0]), np.array([0.5]),
                          north=10.0, east=20.0, yaw_rad=np.pi / 2)
    assert n[0] == pytest.approx(10.0, abs=1e-6)
    assert e[0] == pytest.approx(21.0, abs=1e-6)
    assert d[0] == pytest.approx(0.5)


def test_round_trip_identity_pose():
    """With zero pose and zero pitch, map coords equal sonar coords."""
    x_b, y_b, z_b = sonar_to_body(np.array([4.0]), np.array([-2.0]), 0.0)
    n, e, d = body_to_map(x_b, y_b, z_b, 0.0, 0.0, 0.0)
    assert n[0] == pytest.approx(4.0)
    assert e[0] == pytest.approx(-2.0)
    assert d[0] == pytest.approx(0.0)


# ---------------------------------------------------------------- leading edge
def _bright_blob(img, r, c, size=9, value=250):
    img[r - size // 2: r + size // 2 + 1, c - size // 2: c + size // 2 + 1] = value


def test_leading_edge_one_bin_per_beam():
    """Two blobs on the same beam -> exactly one selected cell for that beam."""
    rng = np.random.default_rng(0)
    img = (rng.random((433, 512)) * 20).astype(np.float32) / 255.0
    _bright_blob(img, 100, 200)
    _bright_blob(img, 300, 200)      # same beam, second feature
    _bright_blob(img, 150, 350)
    det = make_detector()
    cols, rows, resp = leading_edge(img, det, n_beams=512)
    assert len(cols) == len(set(cols.tolist()))      # one bin per beam, always
    assert len(cols) > 0


def test_leading_edge_empty_image():
    """A featureless image yields an empty leading edge, not an error."""
    img = np.zeros((433, 512), dtype=np.float32)
    det = make_detector()
    cols, rows, resp = leading_edge(img, det, n_beams=512)
    assert len(cols) == 0


def test_scan_to_map_points_shape_and_intensity():
    rng_ = np.random.default_rng(1)
    img = (rng_.random((433, 512)) * 30).astype(np.float32) / 255.0
    _bright_blob(img, 220, 256, size=13)
    raw = (img * 255).astype(np.uint8)
    az = np.linspace(-1.13, 1.13, 512).astype(np.float32)
    ranges = np.linspace(0.05, 10.0, 633).astype(np.float32)
    det = make_detector()
    pts = scan_to_map_points(img, raw, det, az, ranges, 200,
                             np.deg2rad(20.0), 5.0, -3.0, 0.3)
    assert pts.ndim == 2 and pts.shape[1] == 4
    assert len(pts) > 0
    assert np.all(pts[:, 3] >= 0)                    # intensities sampled


# ---------------------------------------------------------------- real data (optional)
REAL = '/root/bag/sonar_raw.npy'


@pytest.mark.skipif(not os.path.exists(REAL), reason='recorded sonar frames not available')
def test_real_frame_leading_edge():
    """On a real textured Skovshoved frame the leading edge covers many beams."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                    'sonar_odometry'))
    from sonar_odometry.image_processing import SonarImageProcessor
    raw = np.load(REAL).reshape(-1, 633, 512)
    proc = SonarImageProcessor(); proc.config['crop_row'] = 200
    pre = proc.process_image(raw[120])
    det = make_detector()
    cols, rows, resp = leading_edge(pre, det, n_beams=512)
    assert len(cols) > 100                            # dense coverage
    assert len(cols) == len(set(cols.tolist()))       # single bin per beam


# ---------------------------------------------------------------- strict gates
def test_strict_rejects_weak_and_dark():
    """Strict selection drops low-response keypoints and dark-echo cells."""
    from sonar_mapping.leading_edge_core import strict_leading_edge, make_detector
    det = make_detector()
    rng_ = np.random.default_rng(0)
    img = (rng_.random((433, 512)) * 40).astype(np.float32) / 255.0   # dark noise
    raw = (img * 255).astype(np.uint8)
    cols, rows = strict_leading_edge(img, raw, det, 512,
                                     min_response=0.0013, min_intensity=40)
    assert len(cols) == 0        # nothing should survive on dark noise


def test_median_consistency_drops_spikes():
    from sonar_mapping.leading_edge_core import median_consistency
    rows = np.full(40, 100)
    rows[17] = 300               # isolated range spike
    keep = median_consistency(np.arange(40), rows, window=15, tol_bins=30)
    assert not keep[17]
    assert keep.sum() == 39


def test_strict_scan_to_map_shape():
    from sonar_mapping.leading_edge_core import scan_to_map_points_strict, make_detector
    det = make_detector()
    rng_ = np.random.default_rng(1)
    img = rng_.random((433, 512)).astype(np.float32)
    raw = (img * 255).astype(np.uint8)
    az = np.linspace(-1.13, 1.13, 512).astype(np.float32)
    ranges = np.linspace(0.1, 10.0, 633).astype(np.float32)
    pts = scan_to_map_points_strict(img, raw, det, az, ranges, 200,
                                    np.deg2rad(20.0), 1.0, 2.0, 0.3)
    assert pts.shape[1] == 4 or len(pts) == 0
