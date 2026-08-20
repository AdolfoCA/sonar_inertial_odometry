#!/usr/bin/env python3
"""Core logic for AKAZE leading-edge sonar mapping.

Pure functions, no ROS dependencies — used by the offline bag replay and by the
sonar_mapping ROS 2 node. For each sonar scan the strongest AKAZE keypoint per
beam defines a 'leading edge' (a single range bin per beam); those polar cells
are projected to Cartesian points in the sonar frame, rotated through the
static sonar->body mounting pitch, and placed in the map frame with the vehicle
pose.
"""
import numpy as np
import cv2


def make_detector(threshold=0.001, n_octaves=8, n_octave_layers=8):
    """AKAZE detector with the same tuning as the odometry pipeline."""
    det = cv2.AKAZE_create(
        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
        descriptor_size=0,
        descriptor_channels=3,
        threshold=threshold,
    )
    det.setNOctaves(n_octaves)
    det.setNOctaveLayers(n_octave_layers)
    try:
        det.setDiffusivity(cv2.KAZE_DIFF_PM_G2)
    except AttributeError:
        pass
    return det


def leading_edge(processed_img, detector, n_beams):
    """Strongest AKAZE keypoint per beam -> leading edge.

    Args:
        processed_img: pre-processed (cropped) sonar image, rows = range bins.
        detector: cv2.AKAZE instance.
        n_beams: number of sonar beams (columns of the full image).

    Returns:
        (cols, rows, responses): int arrays of the selected keypoint per beam
        (only beams that have at least one keypoint), rows in CROPPED image
        coordinates.
    """
    kps = detector.detect(processed_img, None)
    if not kps:
        return (np.empty(0, int), np.empty(0, int), np.empty(0))
    best_row = np.full(n_beams, -1, dtype=int)
    best_resp = np.full(n_beams, -np.inf)
    for k in kps:
        c = int(round(k.pt[0]))
        if c < 0 or c >= n_beams:
            continue
        if k.response > best_resp[c]:
            best_resp[c] = k.response
            best_row[c] = int(round(k.pt[1]))
    sel = best_row >= 0
    cols = np.nonzero(sel)[0]
    return cols, best_row[sel], best_resp[sel]


def polar_to_sonar_xy(cols, rows_cropped, beam_azimuths_rad, ranges_m, crop_row):
    """Map (beam, cropped-row) cells to Cartesian x,y in the sonar frame (eq. 7)."""
    bins = rows_cropped + crop_row
    bins = np.clip(bins, 0, len(ranges_m) - 1)
    r = ranges_m[bins]
    az = beam_azimuths_rad[cols]
    return r * np.cos(az), r * np.sin(az)


def sonar_to_body(x_s, y_s, pitch_rad):
    """Static sonar->body transform: mounting pitch (down positive), NED body axes.

    The sonar image plane is tilted pitch_rad below the horizontal, so an
    in-plane forward distance x_s projects to x_s*cos(pitch) horizontally and
    x_s*sin(pitch) downward.
    """
    x_b = x_s * np.cos(pitch_rad)
    y_b = y_s
    z_b = x_s * np.sin(pitch_rad)
    return x_b, y_b, z_b


def body_to_map(x_b, y_b, z_b, north, east, yaw_rad):
    """Rigid planar pose: rotate by yaw (heading from North), translate."""
    n = north + x_b * np.cos(yaw_rad) - y_b * np.sin(yaw_rad)
    e = east + x_b * np.sin(yaw_rad) + y_b * np.cos(yaw_rad)
    d = z_b
    return n, e, d


def scan_to_map_points(processed_img, raw_cropped, detector, beam_azimuths_rad,
                       ranges_m, crop_row, pitch_rad, north, east, yaw_rad):
    """Full per-scan pipeline: leading edge -> map-frame points.

    Returns an (N, 4) array [north, east, down, intensity] (empty if no
    keypoints), intensity sampled from the raw (unfiltered) image.
    """
    n_beams = processed_img.shape[1]
    cols, rows, _ = leading_edge(processed_img, detector, n_beams)
    if len(cols) == 0:
        return np.empty((0, 4))
    x_s, y_s = polar_to_sonar_xy(cols, rows, beam_azimuths_rad, ranges_m, crop_row)
    x_b, y_b, z_b = sonar_to_body(x_s, y_s, pitch_rad)
    n, e, d = body_to_map(x_b, y_b, z_b, north, east, yaw_rad)
    rr = np.clip(rows, 0, raw_cropped.shape[0] - 1)
    cc = np.clip(cols, 0, raw_cropped.shape[1] - 1)
    inten = raw_cropped[rr, cc].astype(np.float32)
    return np.column_stack([n, e, d, inten])


def median_consistency(cols, rows, window=15, tol_bins=18):
    """Reject beams whose range bin deviates from the local (az-ordered) median.

    cols must be sorted (they are, by construction). Returns a boolean mask.
    """
    if len(rows) < 5:
        return np.ones(len(rows), bool)
    rows = np.asarray(rows, float)
    half = window // 2
    keep = np.ones(len(rows), bool)
    for i in range(len(rows)):
        lo, hi = max(0, i - half), min(len(rows), i + half + 1)
        med = np.median(rows[lo:hi])
        if abs(rows[i] - med) > tol_bins:
            keep[i] = False
    return keep


def strict_leading_edge(processed_img, raw_cropped, detector, n_beams,
                        min_response=0.002, min_intensity=50,
                        window=15, tol_bins=18, min_beams=8):
    """Leading edge with outlier rejection.

    Three gates on top of best-per-beam selection:
      1. keypoint response >= min_response      (weak detections dropped)
      2. raw echo intensity >= min_intensity    (points must sit on real return)
      3. sliding-median consistency across beams (isolated spikes dropped)
    """
    kps = detector.detect(processed_img, None)
    best_row = np.full(n_beams, -1, dtype=int)
    best_resp = np.full(n_beams, -np.inf)
    for k in kps:
        if k.response < min_response:
            continue
        c = int(round(k.pt[0]))
        if c < 0 or c >= n_beams:
            continue
        if k.response > best_resp[c]:
            best_resp[c] = k.response
            best_row[c] = int(round(k.pt[1]))
    sel = best_row >= 0
    cols = np.nonzero(sel)[0]
    rows = best_row[sel]
    if len(cols) == 0:
        return np.empty(0, int), np.empty(0, int)
    rr = np.clip(rows, 0, raw_cropped.shape[0] - 1)
    inten_ok = raw_cropped[rr, cols] >= min_intensity
    cols, rows = cols[inten_ok], rows[inten_ok]
    if len(cols) == 0:
        return cols, rows
    keep = median_consistency(cols, rows, window, tol_bins)
    cols, rows = cols[keep], rows[keep]
    if len(cols) < min_beams:
        return np.empty(0, int), np.empty(0, int)
    return cols, rows


def scan_to_map_points_strict(processed_img, raw_cropped, detector, beam_azimuths_rad,
                              ranges_m, crop_row, pitch_rad, north, east, yaw_rad,
                              min_response=0.0013, min_intensity=40,
                              window=15, tol_bins=30, min_beams=8):
    """Strict per-scan pipeline: gated leading edge -> map-frame points.

    Same as scan_to_map_points but with the response / intensity / consistency
    gates of strict_leading_edge — this is what the mapping node uses.
    """
    n_beams = processed_img.shape[1]
    cols, rows = strict_leading_edge(processed_img, raw_cropped, detector, n_beams,
                                     min_response=min_response,
                                     min_intensity=min_intensity,
                                     window=window, tol_bins=tol_bins,
                                     min_beams=min_beams)
    if len(cols) == 0:
        return np.empty((0, 4))
    x_s, y_s = polar_to_sonar_xy(cols, rows, beam_azimuths_rad, ranges_m, crop_row)
    x_b, y_b, z_b = sonar_to_body(x_s, y_s, pitch_rad)
    n, e, d = body_to_map(x_b, y_b, z_b, north, east, yaw_rad)
    rr = np.clip(rows, 0, raw_cropped.shape[0] - 1)
    inten = raw_cropped[rr, cols].astype(np.float32)
    return np.column_stack([n, e, d, inten])
