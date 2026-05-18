#!/usr/bin/env python3
"""
Generate trajectory comparison plots and a Google Maps satellite HTML overlay.

All outputs are saved to  <ws>/debug/plots/

  nis_timeseries.pdf        NIS over time with mean and expected lines
  inlier_histogram.pdf      Histogram of RANSAC inlier counts
  east_timeseries.pdf       East position vs time  (GPS, LiDAR, Sonar, error)
  north_timeseries.pdf      North position vs time (GPS, LiDAR, Sonar, error)
  heading_timeseries.pdf    Heading vs time        (GPS, LiDAR, Sonar)
  trajectory_map.html       Satellite map overlay  (requires GPS lat/lon in CSV
                              or --origin-lat/--origin-lon on command line)

GPS lever-arm
-------------
The GPS antenna is offset from the body frame.  The correction
  body = GPS + lever_m * [cos(heading), sin(heading)]
is applied before comparing GPS to LiDAR/Sonar.  Pass --lever-m to override
the default of 0.40 m.

Alignment
---------
Unless --no-align is given, LiDAR and Sonar are rotated to minimise the
RMS distance from the GPS path over their common time interval (closed-form
2-D Procrustes).  All paths share the same starting point (GPS first-fix).
The same rotation offset is applied to the yaw_deg heading series so the
heading plot is consistent with the trajectory plot.

Usage
-----
  python src/plot_trajectories.py
  python src/plot_trajectories.py --data-dir debug/data --zoom 18
  python src/plot_trajectories.py --origin-lat 43.xxxx --origin-lon 10.xxxx

B&W printing
------------
  GPS   #000000  black   luminance Y ≈   0   thickest line
  LiDAR #2166AC  blue    luminance Y ≈  89   medium line
  Sonar #F4A100  amber   luminance Y ≈ 168   thin line
  Error #666666  gray    luminance Y ≈ 102   dashed
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------
_WS_ROOT          = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _WS_ROOT / "debug" / "zigzag" / "data"
_DEFAULT_OUT_DIR  = _WS_ROOT / "debug" / "zigzag" / "plots"

EARTH_R = 6_371_000.0
NIS_DOF = 3          # measurement dimension [dx, dy, dtheta]

# ---------------------------------------------------------------------------
# GPS antenna offset from IMU / body-frame origin
# ---------------------------------------------------------------------------
# Physical distance (metres) between the GPS antenna phase-centre and the
# IMU / body-frame origin.  Positive = antenna is *forward* of the origin
# along the direction of travel.  This is applied to the GPS path before any
# comparison so that all three sources share a common body-frame origin.
#
# Change this constant to match your physical sensor layout; it is also
# exposed as --lever-m on the command line for quick overrides.
GPS_ANTENNA_OFFSET_M: float = 0.99   # North offset applied to LiDAR/Sonar (metres)
GPS_ANTENNA_OFFSET_EAST_M: float = 0.90  # East offset applied to LiDAR/Sonar (metres)

# Known physical offset between sensor starts (user-provided).
# LiDAR recording begins this many seconds after Sonar recording.
LIDAR_SONAR_OFFSET_S: float = 3.0

# Sensors recorded more than this many seconds apart are assumed to come from
# different ROS bag sessions and cannot be synchronised by absolute timestamp.
_SESSION_GAP_S: float = 1000.0

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
C_GPS   = "#000000"
C_LIDAR = "#2166AC"
C_SONAR = "#F4A100"
C_ERR   = "#666666"

LW_GPS   = 2.2
LW_LIDAR = 1.8
LW_SONAR = 1.5
LW_ERR   = 1.2

FS_LABEL  = 22
FS_LEGEND = 14
FS_TICK   = 18

_STYLE = "seaborn-v0_8-whitegrid"


def _setup_style() -> None:
    try:
        plt.style.use(_STYLE)
    except OSError:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update({
        "axes.labelsize":  FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "legend.framealpha": 0.9,
        "figure.dpi": 150,
    })


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


def _label_mean_on_yaxis(ax: plt.Axes, mean_val: float, color: str,
                         fmt: str = "{:.1f}", fontsize: int = 24) -> None:
    """Write the mean value just outside the left spine at the height of the
    mean line, coloured to match. Drop any tick that would collide with it."""
    y0, y1 = ax.get_ylim()
    ticks  = [t for t in ax.get_yticks() if y0 <= t <= y1]
    if len(ticks) >= 2:
        spacing = min(b - a for a, b in zip(ticks, ticks[1:]))
        ax.set_yticks([t for t in ticks if abs(t - mean_val) > 0.45 * spacing])
    ax.annotate(fmt.format(mean_val),
                xy=(0, mean_val), xycoords=("axes fraction", "data"),
                xytext=(-8, 0), textcoords="offset points",
                ha="right", va="center", fontsize=fontsize,
                color=color, fontweight="bold", clip_on=False)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def enu_to_latlon(east_m: np.ndarray, north_m: np.ndarray,
                  lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    lats = lat0 + np.rad2deg(north_m / EARTH_R)
    lons = lon0 + np.rad2deg(east_m / (EARTH_R * np.cos(np.deg2rad(lat0))))
    return lats, lons


def _smooth_heading(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """ENU heading (rad, CCW from East) via finite differences + Savitzky-Golay."""
    n   = len(east)
    win = min(21, (n // 10) | 1)
    if n >= win + 2:
        e_sm = savgol_filter(east,  win, 3)
        n_sm = savgol_filter(north, win, 3)
    else:
        e_sm, n_sm = east, north
    return np.arctan2(np.gradient(n_sm), np.gradient(e_sm))


def _wrap_deg(deg: np.ndarray) -> np.ndarray:
    """Wrap an angle (degrees) to (-180, 180]."""
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


def apply_lever_arm(east: np.ndarray, north: np.ndarray,
                    lever_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Shift GPS path by lever_m in the North direction."""
    return east, north + lever_m


def rotate_enu(east: np.ndarray, north: np.ndarray,
               angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return c * east - s * north, s * east + c * north


def optimal_rotation_2d(gps_e: np.ndarray, gps_n: np.ndarray, gps_t: np.ndarray,
                        other_e: np.ndarray, other_n: np.ndarray,
                        other_t: np.ndarray) -> float:
    """
    Closed-form least-squares rotation that minimises Σ‖R(θ)·p_i − q_i‖²
    where p = other (interpolated onto GPS timestamps) and q = GPS.

    Solution: θ = atan2(Σ(p_e·q_n − p_n·q_e), Σ(p_e·q_e + p_n·q_n))
    """
    p_e = np.interp(gps_t, other_t, other_e)
    p_n = np.interp(gps_t, other_t, other_n)

    t_lo = max(gps_t[0],  other_t[0])
    t_hi = min(gps_t[-1], other_t[-1])
    mask = (gps_t >= t_lo) & (gps_t <= t_hi)
    if mask.sum() < 10:
        return 0.0

    pe, pn = p_e[mask], p_n[mask]
    qe, qn = gps_e[mask], gps_n[mask]
    return float(np.arctan2(
        np.sum(pe * qn - pn * qe),
        np.sum(pe * qe + pn * qn),
    ))


def _interp(src_t: np.ndarray, src_v: np.ndarray,
            dst_t: np.ndarray) -> np.ndarray:
    f = interp1d(src_t, src_v, bounds_error=False, fill_value=np.nan)
    return f(dst_t)


# ---------------------------------------------------------------------------
# Central path preprocessing  (called once; results shared by all outputs)
# ---------------------------------------------------------------------------

def _build_aligned_paths(gps_df: pd.DataFrame, lidar_df: pd.DataFrame,
                         sonar_df: pd.DataFrame,
                         lever_m: float, lever_east_m: float, no_align: bool):
    """
    Align three sensor trajectories in four explicit steps.

    Step 1 — Time alignment: trim all three to the common recording window.
              Physical time reference T=0 is the Sonar start.
              GPS starts at T = elapsed_delay (GPS_t0 − Sonar_t0).
              LiDAR starts at T = LIDAR_SONAR_OFFSET_S (user-provided).
              Common window = [T_start_max, T_end_min].
    Step 2 — Starting-point: zero-centre each path at its own first sample
              inside the common window (all ICs → (0, 0) in local ENU).
    Step 3 — Lever-arm: already applied to raw positions before Step 2
              so the zero-centring anchors the corrected body-frame origin.
    Step 4 — Heading alignment: rotate LiDAR and Sonar so their initial
              heading matches GPS (GPS is the reference, kept fixed).

    Returns (gps, lidar, sonar, rot_lidar, rot_sonar, i_gps_df, i_lidar_df, i_sonar_df).
    i_*_df are row indices into the original dataframes at the trimmed start
    (used downstream for yaw columns and lat/lon look-ups).
    """
    # ── Session detection ──────────────────────────────────────────────────────
    gps_abs_t0   = float(gps_df["time_sec"].iloc[0])
    lidar_abs_t0 = float(lidar_df["time_sec"].iloc[0])
    sonar_abs_t0 = float(sonar_df["time_sec"].iloc[0])
    lidar_same = abs(lidar_abs_t0 - gps_abs_t0) < _SESSION_GAP_S
    sonar_same = abs(sonar_abs_t0 - gps_abs_t0) < _SESSION_GAP_S

    print(f"\n  Session detection (gap = {_SESSION_GAP_S:.0f} s threshold):")
    print(f"    GPS   t0 = {gps_abs_t0:.3f}  (reference)")
    print(f"    LiDAR gap = {lidar_abs_t0-gps_abs_t0:+.1f} s → "
          f"{'same session' if lidar_same else 'DIFFERENT session'}")
    print(f"    Sonar gap = {sonar_abs_t0-gps_abs_t0:+.1f} s → "
          f"{'same session' if sonar_same else 'DIFFERENT session'}")

    # Each sensor's elapsed-time axis (seconds since own first sample)
    gps_t   = gps_df["time_sec"].values.astype(float)   - gps_abs_t0
    lidar_t = lidar_df["time_sec"].values.astype(float) - lidar_abs_t0
    sonar_t = sonar_df["time_sec"].values.astype(float) - sonar_abs_t0

    # Raw ENU positions
    gps_e_raw   = gps_df["east_m"].values.astype(float)
    gps_n_raw   = gps_df["north_m"].values.astype(float)
    lidar_e_raw = lidar_df["east_m"].values.astype(float)
    lidar_n_raw = lidar_df["north_m"].values.astype(float)
    sonar_e_raw = sonar_df["east_m"].values.astype(float)
    sonar_n_raw = sonar_df["north_m"].values.astype(float)

    # lever_m north-shift is applied AFTER alignment (see end of function)

    # ── Step 1: Time alignment ─────────────────────────────────────────────────
    # Physical time T = 0 at Sonar start.
    #   Sonar elapsed t  ↔  physical T = sonar_t
    #   GPS elapsed t    ↔  physical T = gps_t + gps_phys_offset
    #   LiDAR elapsed t  ↔  physical T = lidar_t + lidar_phys_offset
    gps_phys_offset   = gps_abs_t0 - sonar_abs_t0 if sonar_same else 0.0
    lidar_phys_offset = LIDAR_SONAR_OFFSET_S if not lidar_same else (lidar_abs_t0 - sonar_abs_t0 if sonar_same else 0.0)

    T_start = max(gps_phys_offset, 0.0, lidar_phys_offset)   # latest sensor start
    T_end   = min(gps_t[-1]   + gps_phys_offset,             # earliest sensor end
                  sonar_t[-1],
                  lidar_t[-1] + lidar_phys_offset)

    # Convert common window edges to each sensor's elapsed time
    gps_t_start   = T_start - gps_phys_offset
    gps_t_end     = T_end   - gps_phys_offset
    sonar_t_start = T_start
    sonar_t_end   = T_end
    lidar_t_start = T_start - lidar_phys_offset
    lidar_t_end   = T_end   - lidar_phys_offset

    # Small epsilon: sensors have different rates so T_start/T_end rarely land
    # exactly on a sample.  Accept any sample within half a typical period (0.5 s).
    _EPS = 0.5

    def _idx_start(arr, val):
        i = int(np.searchsorted(arr, val - _EPS))
        return max(0, min(i, len(arr) - 1))

    def _idx_end(arr, val):
        i = int(np.searchsorted(arr, val + _EPS, side="right")) - 1
        return max(0, min(i, len(arr) - 1))

    i_gps_0   = _idx_start(gps_t,   gps_t_start)
    i_gps_1   = _idx_end  (gps_t,   gps_t_end)
    i_sonar_0 = _idx_start(sonar_t, sonar_t_start)
    i_sonar_1 = _idx_end  (sonar_t, sonar_t_end)
    i_lidar_0 = _idx_start(lidar_t, lidar_t_start)
    i_lidar_1 = _idx_end  (lidar_t, lidar_t_end)

    i_gps_1   = max(i_gps_0,   i_gps_1)
    i_sonar_1 = max(i_sonar_0, i_sonar_1)
    i_lidar_1 = max(i_lidar_0, i_lidar_1)

    print(f"\n  Step 1 — time alignment  (physical T=[{T_start:.3f}, {T_end:.3f}] s):")
    print(f"    GPS:   rows [{i_gps_0}, {i_gps_1}]  "
          f"({i_gps_1-i_gps_0+1} pts, elapsed [{gps_t[i_gps_0]:.3f}, {gps_t[i_gps_1]:.3f}] s)")
    print(f"    Sonar: rows [{i_sonar_0}, {i_sonar_1}]  "
          f"({i_sonar_1-i_sonar_0+1} pts, elapsed [{sonar_t[i_sonar_0]:.3f}, {sonar_t[i_sonar_1]:.3f}] s)")
    print(f"    LiDAR: rows [{i_lidar_0}, {i_lidar_1}]  "
          f"({i_lidar_1-i_lidar_0+1} pts, elapsed [{lidar_t[i_lidar_0]:.3f}, {lidar_t[i_lidar_1]:.3f}] s)")

    # Trim all arrays to the common window
    sl_g = slice(i_gps_0,   i_gps_1   + 1)
    sl_l = slice(i_lidar_0, i_lidar_1 + 1)
    sl_s = slice(i_sonar_0, i_sonar_1 + 1)

    gps_e_raw   = gps_e_raw[sl_g];   gps_n_raw   = gps_n_raw[sl_g]
    lidar_e_raw = lidar_e_raw[sl_l]; lidar_n_raw = lidar_n_raw[sl_l]
    sonar_e_raw = sonar_e_raw[sl_s]; sonar_n_raw = sonar_n_raw[sl_s]
    gps_t   = gps_t[sl_g]   - gps_t[i_gps_0]
    lidar_t = lidar_t[sl_l] - lidar_t[i_lidar_0]
    sonar_t = sonar_t[sl_s] - sonar_t[i_sonar_0]

    # ── Step 2: Zero-centre each path at its own first sample ─────────────────
    gps_e   = gps_e_raw   - gps_e_raw[0]
    gps_n   = gps_n_raw   - gps_n_raw[0]
    lidar_e = lidar_e_raw - lidar_e_raw[0]
    lidar_n = lidar_n_raw - lidar_n_raw[0]
    sonar_e = sonar_e_raw - sonar_e_raw[0]
    sonar_n = sonar_n_raw - sonar_n_raw[0]

    print(f"\n  Step 2 — starting points anchored to (0, 0) for each sensor.")

    rot_lidar = rot_sonar = 0.0

    if not no_align:
        from scipy.spatial import KDTree

        # ── Step 3: Heading alignment ─────────────────────────────────────────
        # Coarse rotation: match each sensor's initial heading to GPS.
        # Rotating around (0,0) = rotating around the IC → start stays fixed.
        def _init_hdg(e, n):
            for k in range(1, min(50, len(e))):
                de, dn = e[k] - e[0], n[k] - n[0]
                if de * de + dn * dn > 0.01:
                    return float(np.arctan2(dn, de))
            return 0.0

        gps_h0   = _init_hdg(gps_e,   gps_n)
        lidar_h0 = _init_hdg(lidar_e, lidar_n)
        sonar_h0 = _init_hdg(sonar_e, sonar_n)
        rot_lidar = gps_h0 - lidar_h0
        rot_sonar = gps_h0 - sonar_h0
        lidar_e, lidar_n = rotate_enu(lidar_e, lidar_n, rot_lidar)
        sonar_e, sonar_n = rotate_enu(sonar_e, sonar_n, rot_sonar)

        print(f"\n  Step 3 — heading alignment (GPS = reference):")
        print(f"    GPS   {np.rad2deg(gps_h0):+.1f}°")
        print(f"    LiDAR {np.rad2deg(lidar_h0):+.1f}° → rotated {np.rad2deg(rot_lidar):+.1f}°")
        print(f"    Sonar {np.rad2deg(sonar_h0):+.1f}° → rotated {np.rad2deg(rot_sonar):+.1f}°")

        # ── Step 4: Rotation-only ICP ─────────────────────────────────────────
        # Fine rotation starting from the heading-aligned pose.
        # Still rotates around (0,0) → IC stays at (0,0).
        def _icp_rot(ref_e, ref_n, path_e, path_n, n_iter=100, tol=1e-7):
            e, n  = path_e.copy(), path_n.copy()
            total = 0.0
            tree  = KDTree(np.column_stack([ref_e, ref_n]))
            for _ in range(n_iter):
                _, idx = tree.query(np.column_stack([e, n]))
                qe, qn = ref_e[idx], ref_n[idx]
                theta  = float(np.arctan2(
                    np.sum(e * qn - n * qe),
                    np.sum(e * qe + n * qn),
                ))
                e, n   = rotate_enu(e, n, theta)
                total += theta
                if abs(theta) < tol:
                    break
            return e, n, total

        lidar_e, lidar_n, drot_L = _icp_rot(gps_e, gps_n, lidar_e, lidar_n)
        sonar_e, sonar_n, drot_S = _icp_rot(gps_e, gps_n, sonar_e, sonar_n)
        rot_lidar += drot_L
        rot_sonar += drot_S

        print(f"\n  Step 4 — ICP refinement (rotation-only, around IC):")
        print(f"    LiDAR +{np.rad2deg(drot_L):+.2f}° → total {np.rad2deg(rot_lidar):+.2f}°")
        print(f"    Sonar +{np.rad2deg(drot_S):+.2f}° → total {np.rad2deg(rot_sonar):+.2f}°")

    # Apply antenna offsets after all alignment so they are not cancelled by
    # zero-centering and not absorbed by the ICP rotation.
    if lever_m != 0 or lever_east_m != 0:
        lidar_n = lidar_n - lever_m
        sonar_n = sonar_n - lever_m
        lidar_e = lidar_e - lever_east_m
        sonar_e = sonar_e - lever_east_m
        print(f"  Antenna offset: LiDAR/Sonar N={-lever_m:+.3f} m  E={-lever_east_m:+.3f} m")

    i_gps_df   = i_gps_0
    i_lidar_df = i_lidar_0
    i_sonar_df = i_sonar_0

    return (
        (gps_t,   gps_e,   gps_n),
        (lidar_t, lidar_e, lidar_n),
        (sonar_t, sonar_e, sonar_n),
        rot_lidar, rot_sonar,
        i_gps_df, i_lidar_df, i_sonar_df,
    )


# ---------------------------------------------------------------------------
# Plot 1: NIS timeseries
# ---------------------------------------------------------------------------

NIS_THRESHOLD = 30.0


def plot_nis(nis_df: pd.DataFrame, out_dir: Path) -> None:
    t   = nis_df["time_sec"].values - nis_df["time_sec"].iloc[0]
    nis = nis_df["nis"].values

    mask     = nis <= NIS_THRESHOLD
    t_filt   = t[mask]
    nis_filt = nis[mask]

    mean_nis = float(np.nanmean(nis_filt))
    expected = float(NIS_DOF)

    FS_BASE = 28
    FS_BIG  = FS_BASE + 14
    FS_TICK = FS_BASE + 8
    FS_MEAN = FS_BASE + 4
    FS_LEG  = FS_BASE + 6

    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(t_filt, nis_filt, color="#444444", linewidth=0.9, marker="o",
            markersize=1.4, alpha=0.75)
    ax.axhline(mean_nis, color="#2ca02c", linewidth=2.0)
    ax.axhline(expected, color="#d62728", linewidth=2.0, linestyle="--")

    ax.set_ylim(bottom=0, top=NIS_THRESHOLD)
    ax.set_xlabel("Time (s)", fontsize=FS_BIG)
    ax.set_ylabel("NIS", fontsize=FS_BIG)
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    _label_mean_on_yaxis(ax, mean_nis, "#2ca02c", fmt="{:.1f}", fontsize=FS_MEAN)

    handles = [
        Line2D([0], [0], color="#444444", lw=1.5, marker="o", markersize=4),
        Line2D([0], [0], color="#2ca02c", lw=2.0),
        Line2D([0], [0], color="#d62728", lw=2.0, linestyle="--"),
    ]
    labels = ["NIS", "Mean", f"Expected ({NIS_DOF})"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, fontsize=FS_LEG, framealpha=0.9, handlelength=2.0,
               columnspacing=1.4, borderpad=0.5)
    fig.tight_layout()
    _savefig(fig, out_dir / "nis_timeseries.png")


# ---------------------------------------------------------------------------
# Plot 2: Inlier count over time
# ---------------------------------------------------------------------------

def plot_inliers(nis_df: pd.DataFrame, out_dir: Path) -> None:
    t      = nis_df["time_sec"].values - nis_df["time_sec"].iloc[0]
    counts = nis_df["inlier_count"].values
    mean_c = float(np.mean(counts))

    FS_BASE = 28
    FS_BIG  = FS_BASE + 14
    FS_TICK = FS_BASE + 8
    FS_MEAN = FS_BASE + 4
    FS_LEG  = FS_BASE + 6

    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(t, counts, color="#3a7d44", linewidth=0.9, marker="o",
            markersize=1.4, alpha=0.85)
    ax.axhline(mean_c, color="#444444", linewidth=2.0, linestyle="--")
    ax.set_xlabel("Time (s)", fontsize=FS_BIG)
    ax.set_ylabel("Inlier Count", fontsize=FS_BIG)
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    _label_mean_on_yaxis(ax, mean_c, "#444444", fmt="{:.0f}", fontsize=FS_MEAN)

    handles = [
        Line2D([0], [0], color="#3a7d44", lw=1.5, marker="o", markersize=4),
        Line2D([0], [0], color="#444444", lw=2.0, linestyle="--"),
    ]
    labels = ["Inliers", "Mean"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, fontsize=FS_LEG, framealpha=0.9, handlelength=2.0,
               columnspacing=1.4, borderpad=0.5)
    fig.tight_layout()
    _savefig(fig, out_dir / "inlier_timeseries.png")


# ---------------------------------------------------------------------------
# Plot 3 & 4: East and North position vs time
# ---------------------------------------------------------------------------

def _error_on_gps_grid(gps_t, gps_v, other_t, other_v):
    return gps_t, gps_v - _interp(other_t, other_v, gps_t)


def plot_position(gps: tuple, lidar: tuple, sonar: tuple, out_dir: Path) -> None:
    gps_t,   gps_e,   gps_n   = gps
    lidar_t, lidar_e, lidar_n = lidar
    sonar_t, sonar_e, sonar_n = sonar

    err_t_e, err_e = _error_on_gps_grid(gps_t, gps_e, sonar_t, sonar_e)
    err_t_n, err_n = _error_on_gps_grid(gps_t, gps_n, sonar_t, sonar_n)

    for ylabel, gps_v, lidar_v, sonar_v, err_t, err_v, fname in [
        (r"$p^E$ (m)", gps_e, lidar_e, sonar_e, err_t_e, err_e, "east_timeseries.png"),
        (r"$p^N$ (m)", gps_n, lidar_n, sonar_n, err_t_n, err_n, "north_timeseries.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(gps_t,   gps_v,   color=C_GPS,   lw=LW_GPS,   label="GPS")
        ax.plot(lidar_t, lidar_v, color=C_LIDAR, lw=LW_LIDAR, label="LiDAR")
        ax.plot(sonar_t, sonar_v, color=C_SONAR, lw=LW_SONAR, label="Sonar")
        ax.plot(err_t, err_v, color=C_ERR, lw=LW_ERR, linestyle="--",
                label="GPS − Sonar error")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", framealpha=0.9)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        fig.tight_layout()
        _savefig(fig, out_dir / fname)


# ---------------------------------------------------------------------------
# Plot 5: Heading vs time
# ---------------------------------------------------------------------------

def plot_heading(gps_e: np.ndarray, gps_n: np.ndarray, gps_t: np.ndarray,
                 lidar_df: pd.DataFrame, lidar_t: np.ndarray,
                 sonar_df: pd.DataFrame, sonar_t: np.ndarray,
                 rot_lidar: float, rot_sonar: float,
                 out_dir: Path) -> None:
    """
    GPS heading from lever-arm-corrected ENU via smoothed finite differences.
    LiDAR/Sonar yaw_deg shifted by the same Procrustes rotation applied to
    the ENU path so all three series share a common heading reference.
    """
    gps_hdg = _wrap_deg(np.rad2deg(_smooth_heading(gps_e, gps_n)))

    if "yaw_deg" in lidar_df.columns:
        lidar_hdg = _wrap_deg(lidar_df["yaw_deg"].values.astype(float)
                              + np.rad2deg(rot_lidar))
    else:
        lidar_hdg = None

    if "yaw_deg" in sonar_df.columns:
        sonar_hdg = _wrap_deg(sonar_df["yaw_deg"].values.astype(float)
                              + np.rad2deg(rot_sonar))
    else:
        sonar_hdg = None

    def _break_wraps(t, hdg):
        t, hdg = np.asarray(t, float), np.asarray(hdg, float)
        jump = np.abs(np.diff(hdg)) > 180.0
        if not jump.any():
            return t, hdg
        idx = np.where(jump)[0] + 1
        return np.insert(t, idx, np.nan), np.insert(hdg, idx, np.nan)

    fig, ax = plt.subplots(figsize=(9, 4))
    gt, gh = _break_wraps(gps_t, gps_hdg)
    ax.plot(gt, gh, color=C_GPS, lw=LW_GPS, label="GPS")
    if lidar_hdg is not None:
        lt, lh = _break_wraps(lidar_t, lidar_hdg)
        ax.plot(lt, lh, color=C_LIDAR, lw=LW_LIDAR, label="LiDAR")
    if sonar_hdg is not None:
        st, sh = _break_wraps(sonar_t, sonar_hdg)
        ax.plot(st, sh, color=C_SONAR, lw=LW_SONAR, label="Sonar")

    ax.set_ylim(-180, 180)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading (°, CCW from East)")
    ax.legend(loc="best", framealpha=0.9)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    _savefig(fig, out_dir / "heading_timeseries.png")


# ---------------------------------------------------------------------------
# Plot 6: Combined error plot — Sonar−LiDAR and Sonar−GPS, in E / N / heading
# ---------------------------------------------------------------------------

def plot_error_summary(gps: tuple, lidar: tuple, sonar: tuple,
                       gps_hdg_rad: np.ndarray,
                       lidar_hdg_rad: np.ndarray | None,
                       sonar_hdg_rad: np.ndarray | None,
                       out_dir: Path) -> None:
    """One figure, 6 curves:
       Sonar−LiDAR : East, North, Heading
       Sonar−GPS   : East, North, Heading
    Everything is evaluated on the Sonar time grid (GPS / LiDAR interpolated).
    Position errors (m) on the left y-axis, heading errors (deg) on the right.
    Heading arrays are passed in already unwrapped and rotation-corrected.
    """
    gps_t,   gps_e,   gps_n   = gps
    lidar_t, lidar_e, lidar_n = lidar
    sonar_t, sonar_e, sonar_n = sonar

    e_err_lidar = sonar_e - _interp(lidar_t, lidar_e, sonar_t)
    n_err_lidar = sonar_n - _interp(lidar_t, lidar_n, sonar_t)
    e_err_gps   = sonar_e - _interp(gps_t,   gps_e,   sonar_t)
    n_err_gps   = sonar_n - _interp(gps_t,   gps_n,   sonar_t)

    if sonar_hdg_rad is not None:
        sonar_hdg = np.rad2deg(sonar_hdg_rad)
        h_err_gps = _wrap_deg(sonar_hdg - np.rad2deg(_interp(gps_t, gps_hdg_rad, sonar_t)))
        h_err_lidar = (_wrap_deg(sonar_hdg - np.rad2deg(_interp(lidar_t, lidar_hdg_rad, sonar_t)))
                       if lidar_hdg_rad is not None else None)
    else:
        h_err_gps = h_err_lidar = None

    fig, ax = plt.subplots(figsize=(24, 12))
    ax_h = ax.twinx()

    # East = blue, North = red, Heading = green. Solid = Sonar − LiDAR, dashed = Sonar − GPS.
    C_EAST, C_NORTH, C_HDG = "#1f77b4", "#d62728", "#2ca02c"
    LW_LD, LW_GP = 3.6, 5.2
    ax.plot(sonar_t, e_err_lidar, color=C_EAST, lw=LW_LD, zorder=2,
            label="East: Sonar − LiDAR")
    ax.plot(sonar_t, n_err_lidar, color=C_NORTH, lw=LW_LD, zorder=2,
            label="North: Sonar − LiDAR")
    ax.plot(sonar_t, e_err_gps, color=C_EAST, lw=LW_GP, linestyle="--", zorder=4,
            label="East: Sonar − GPS")
    ax.plot(sonar_t, n_err_gps, color=C_NORTH, lw=LW_GP, linestyle="--", zorder=4,
            label="North: Sonar − GPS")
    if h_err_lidar is not None:
        ax_h.plot(sonar_t, h_err_lidar, color=C_HDG, lw=LW_LD, zorder=2,
                  label="Heading: Sonar − LiDAR")
    if h_err_gps is not None:
        ax_h.plot(sonar_t, h_err_gps, color=C_HDG, lw=LW_GP, linestyle="--", zorder=4,
                  label="Heading: Sonar − GPS")

    FS_LBL  = 48   # axis labels
    FS_TICK = 40   # tick labels
    FS_LEG  = 30   # legend (smaller so all 3 columns fit inside the axes)
    ax.axhline(0.0, color="#999999", lw=0.8, zorder=0)
    ax.set_xlabel("Time (s)", fontsize=FS_LBL)
    ax.set_ylabel("Position error (m)", fontsize=FS_LBL)
    ax_h.set_ylabel("Heading error (°)", fontsize=FS_LBL)
    ax.tick_params(labelsize=FS_TICK)
    ax_h.tick_params(labelsize=FS_TICK)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    # Add headroom on the position axis so the inside-top legend never overlaps
    # the curves.
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y0 + (y1 - y0) * 1.5)

    # Heading axis: scale to match the position axis headroom so the twin
    # axes share zero and the gridline at y=0 lines up for both.
    ax_h.set_ylim(-45, 45 * 1.5)
    ax_h.set_yticks([-45, -30, -15, 0, 15, 30, 45])

    # Clean tick locators on the position axis to avoid clutter after expansion.
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="upper"))

    # Gridlines only on the left axis to avoid double-grid from the twin.
    ax.grid(True, alpha=0.4)
    ax_h.grid(False)

    # Legend inside the axes at the top.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_h.get_legend_handles_labels()
    leg = ax.legend(h1 + h2, l1 + l2, loc="upper center",
                    ncol=3, fontsize=FS_LEG, framealpha=0.95, handlelength=2.0,
                    columnspacing=1.2, borderpad=0.4, handletextpad=0.5)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    fig.subplots_adjust(left=0.08, right=0.93, top=0.98, bottom=0.10)
    _savefig(fig, out_dir / "error_summary.png")


# ---------------------------------------------------------------------------
# HTML map generation
# ---------------------------------------------------------------------------

def _js_latlng_array(lats: np.ndarray, lons: np.ndarray, max_pts: int = 4000) -> str:
    """Return a JS array of [lat, lng] pairs for Leaflet polylines."""
    n    = len(lats)
    step = max(1, n // max_pts)
    pts  = [f"[{lats[i]:.7f},{lons[i]:.7f}]" for i in range(0, n, step)]
    return "[" + ",".join(pts) + "]"


GOOGLE_MAPS_API_KEY: str = "AIzaSyA2T1UdCtUsR5izwnzkSXtB4c5ZjVhMDvw"


def write_html(out_path: Path, center_lat: float, center_lon: float,
               zoom: int, paths: list[dict], markers: list[dict]) -> None:
    """Write a Google Maps JS API map."""
    styles = [("GPS", C_GPS, 9), ("LiDAR", C_LIDAR, 6), ("Sonar", C_SONAR, 4)]
    legend_items = "".join(
        '<div><span style="display:inline-block;width:30px;height:4px;'
        'background:{col};margin-right:6px;vertical-align:middle;'
        'border:1px solid #555"></span>{name}</div>'.format(col=col, name=name)
        for name, col, _ in styles
    )

    def _gm_latlng_array(lats, lons, max_pts=4000):
        n    = len(lats)
        step = max(1, n // max_pts)
        pts  = ["{{lat:{:.7f},lng:{:.7f}}}".format(lats[i], lons[i])
                for i in range(0, n, step)]
        return "[" + ",".join(pts) + "]"

    polylines_js = "\n".join(
        "  new google.maps.Polyline({{path:{pts},strokeColor:'{col}',"
        "strokeWeight:{w},strokeOpacity:1,map:map}});".format(
            pts=_gm_latlng_array(p["lats"], p["lons"]),
            col=p["color"], w=p["weight"],
        )
        for p in paths
    )
    def _gm_icon(color, shape):
        sym = ("google.maps.SymbolPath.BACKWARD_CLOSED_ARROW"
               if shape == "end" else "google.maps.SymbolPath.CIRCLE")
        return ("{{path:{sym},scale:8,fillColor:'{col}',"
                "fillOpacity:1,strokeColor:'white',strokeWeight:2}}").format(
                    sym=sym, col=color)

    markers_js = "\n".join(
        "  new google.maps.Marker({{position:{{lat:{lat:.7f},lng:{lon:.7f}}},"
        "map:map,title:'{title}',icon:{icon}}});".format(
            lat=m["lat"], lon=m["lon"], title=m["title"],
            icon=_gm_icon(m.get("color", "#ff0000"), m.get("shape", "circle")),
        )
        for m in markers
    )
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Trajectory Map</title>
<style>html,body,#map{width:100%;height:100%;margin:0;padding:0}</style>
</head><body>
<div id="map"></div>
<div style="position:absolute;top:10px;right:10px;z-index:1000;
            background:rgba(255,255,255,0.88);padding:8px 14px;
            font-family:Arial,sans-serif;font-size:13px;border-radius:4px;
            box-shadow:0 2px 6px rgba(0,0,0,.3)">""" + legend_items + """</div>
<script>
function initMap() {
  var map = new google.maps.Map(document.getElementById('map'), {
    center: {lat:""" + f"{center_lat:.7f}" + """, lng:""" + f"{center_lon:.7f}" + """},
    zoom: """ + str(zoom) + """,
    mapTypeId: 'roadmap'
  });
""" + polylines_js + "\n" + markers_js + """
}
</script>
<script async defer
  src="https://maps.googleapis.com/maps/api/js?key=""" + GOOGLE_MAPS_API_KEY + """&callback=initMap">
</script>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"  saved → {out_path.name}")


# ---------------------------------------------------------------------------
# Static map image (OSM tiles + matplotlib overlay)
# ---------------------------------------------------------------------------

def _tile_coords(lat_deg: float, lon_deg: float, zoom: int) -> tuple[float, float]:
    """Fractional tile x,y for a given lat/lon and zoom level."""
    n   = 2 ** zoom
    tx  = (lon_deg + 180.0) / 360.0 * n
    lat_r = math.radians(lat_deg)
    ty  = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return tx, ty


def _fetch_tile(z: int, x: int, y: int) -> "PIL.Image.Image | None":
    from PIL import Image as _PILImage
    # Esri World Imagery satellite tiles (note: y before x in URL path)
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "plot_trajectories/1.0 ros2_ws (educational use)"},
            timeout=10,
        )
        resp.raise_for_status()
        return _PILImage.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        print(f"    [tile {z}/{x}/{y} failed: {exc}]")
        return None


def _build_basemap(all_lats: np.ndarray, all_lons: np.ndarray,
                   zoom: int, pad: int = 1):
    """
    Fetch and stitch OSM tiles covering the bounding box of all lat/lon points.

    Returns (PIL.Image, x0_tile, y0_tile, tile_px) where (x0,y0) are the
    top-left tile indices and tile_px=256 is the size of each tile in pixels.
    Auto-reduces zoom if more than 64 tiles would be needed.
    """
    from PIL import Image as _PILImage
    TILE_PX  = 256
    MAX_TILES = 64

    for z in range(zoom, max(zoom - 4, 10) - 1, -1):
        ntotal = 2 ** z
        tx_min, ty_min = _tile_coords(max(all_lats), min(all_lons), z)
        tx_max, ty_max = _tile_coords(min(all_lats), max(all_lons), z)

        x0 = max(0, int(tx_min) - pad)
        y0 = max(0, int(ty_min) - pad)
        x1 = min(ntotal - 1, int(tx_max) + pad)
        y1 = min(ntotal - 1, int(ty_max) + pad)

        nx, ny = x1 - x0 + 1, y1 - y0 + 1
        if nx * ny <= MAX_TILES:
            break
        print(f"    zoom {z}: {nx*ny} tiles > {MAX_TILES}, trying zoom {z-1} …")

    print(f"    fetching {nx}×{ny}={nx*ny} tiles at zoom {z} …")
    canvas = _PILImage.new("RGB", (nx * TILE_PX, ny * TILE_PX), (210, 210, 210))
    for xi in range(nx):
        for yi in range(ny):
            tile = _fetch_tile(z, x0 + xi, y0 + yi)
            if tile is not None:
                canvas.paste(tile, (xi * TILE_PX, yi * TILE_PX))

    return canvas, x0, y0, z, TILE_PX


def _latlon_to_px(lats: np.ndarray, lons: np.ndarray,
                  zoom: int, x0_tile: int, y0_tile: int,
                  tile_px: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Convert arrays of lat/lon to pixel coordinates on the stitched basemap."""
    n   = 2 ** zoom
    px  = (lons + 180.0) / 360.0 * n * tile_px - x0_tile * tile_px
    lat_r = np.radians(lats)
    py  = ((1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / math.pi)
           / 2.0 * n * tile_px - y0_tile * tile_px)
    return px, py


def _downsample(arr: np.ndarray, max_pts: int = 3000) -> np.ndarray:
    step = max(1, len(arr) // max_pts)
    return arr[::step]


def plot_trajectory_map_image(
        gps_lats: np.ndarray, gps_lons: np.ndarray,
        lidar_lats: np.ndarray, lidar_lons: np.ndarray,
        sonar_lats: np.ndarray, sonar_lons: np.ndarray,
        zoom: int, out_dir: Path) -> None:
    """
    Fetch Esri satellite tiles, stitch a basemap, then draw the three
    trajectories on top with start (●) / end (▼) markers and a clear
    bottom-right legend.  Saved as trajectory_map.png.
    """
    print("  Building static trajectory map image …")

    all_lats = np.concatenate([gps_lats, lidar_lats, sonar_lats])
    all_lons = np.concatenate([gps_lons, lidar_lons, sonar_lons])

    try:
        basemap, x0, y0, used_zoom, tile_px = _build_basemap(all_lats, all_lons, zoom)
    except Exception as exc:
        print(f"    basemap fetch failed: {exc} — skipping trajectory_map.png")
        return

    img_w, img_h = basemap.size

    def to_px(lats: np.ndarray, lons: np.ndarray):
        return _latlon_to_px(
            _downsample(lats), _downsample(lons), used_zoom, x0, y0, tile_px,
        )

    gx, gy = to_px(gps_lats,   gps_lons)
    lx, ly = to_px(lidar_lats, lidar_lons)
    sx, sy = to_px(sonar_lats, sonar_lons)

    # — start / end pixel coords (full-resolution, not downsampled) —
    def endpoints(lats, lons):
        px0, py0 = _latlon_to_px(lats[[0]],  lons[[0]],  used_zoom, x0, y0, tile_px)
        px1, py1 = _latlon_to_px(lats[[-1]], lons[[-1]], used_zoom, x0, y0, tile_px)
        return (px0[0], py0[0]), (px1[0], py1[0])

    gps_s,   gps_e_pt   = endpoints(gps_lats,   gps_lons)
    lidar_s, lidar_e_pt = endpoints(lidar_lats, lidar_lons)
    sonar_s, sonar_e_pt = endpoints(sonar_lats, sonar_lons)

    # — tight crop: bounding box of all trajectory pixels + 8 % margin —
    all_px = np.concatenate([gx, lx, sx])
    all_py = np.concatenate([gy, ly, sy])
    pad_x  = max(40, (all_px.max() - all_px.min()) * 0.08)
    pad_y  = max(40, (all_py.max() - all_py.min()) * 0.08)
    xlim   = (all_px.min() - pad_x, all_px.max() + pad_x)
    ylim   = (all_py.max() + pad_y, all_py.min() - pad_y)   # y-axis flipped (down = positive)

    # — figure: portrait aspect matching the cropped region —
    crop_w = xlim[1] - xlim[0]
    crop_h = ylim[0] - ylim[1]   # note: ylim[0] > ylim[1] because y-axis is flipped
    base_w = 10.0
    base_h = max(base_w * crop_h / max(crop_w, 1.0), 6.0)
    fig, ax = plt.subplots(figsize=(base_w, base_h), dpi=150)
    ax.imshow(np.asarray(basemap), extent=[0, img_w, img_h, 0],
              origin="upper", aspect="equal", zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # trajectories — thick lines for visibility over satellite imagery
    LW_MAP_GPS   = 4.0
    LW_MAP_LIDAR = 3.0
    LW_MAP_SONAR = 2.5
    ax.plot(gx, gy, color=C_GPS,   lw=LW_MAP_GPS,   zorder=3, alpha=0.95, label="GPS")
    ax.plot(lx, ly, color=C_LIDAR, lw=LW_MAP_LIDAR, zorder=2, alpha=0.92, label="LiDAR")
    ax.plot(sx, sy, color=C_SONAR, lw=LW_MAP_SONAR, zorder=1, alpha=0.90, label="Sonar")

    # start markers (●) and end markers (▼) — white edge for contrast
    mk_start = dict(zorder=6, markersize=13, markeredgewidth=2.0,
                    markeredgecolor="white", linestyle="None")
    mk_end   = dict(zorder=6, markersize=13, markeredgewidth=2.0,
                    markeredgecolor="white", linestyle="None")
    for (spx, spy), (epx, epy), col in [
        (gps_s,   gps_e_pt,   C_GPS),
        (lidar_s, lidar_e_pt, C_LIDAR),
        (sonar_s, sonar_e_pt, C_SONAR),
    ]:
        ax.plot(spx, spy, marker="o", color=col, **mk_start)
        ax.plot(epx, epy, marker="v", color=col, **mk_end)

    # legend — bottom right, white semi-transparent box, large font
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=C_GPS,   lw=LW_MAP_GPS,   label="GPS"),
        Line2D([0], [0], color=C_LIDAR, lw=LW_MAP_LIDAR, label="LiDAR"),
        Line2D([0], [0], color=C_SONAR, lw=LW_MAP_SONAR, label="Sonar"),
        Line2D([0], [0], marker="o", color="#333333", linestyle="None",
               markersize=10, markeredgecolor="white", markeredgewidth=1.5,
               label="Start"),
        Line2D([0], [0], marker="v", color="#333333", linestyle="None",
               markersize=10, markeredgecolor="white", markeredgewidth=1.5,
               label="End"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="lower right",
        framealpha=0.92,
        facecolor="white",
        edgecolor="#aaaaaa",
        fontsize=max(FS_LEGEND + 2, 12),
        labelspacing=0.6,
        handlelength=2.5,
        borderpad=0.8,
    )
    leg.set_zorder(10)

    ax.set_axis_off()
    fig.tight_layout(pad=0.3)
    _savefig(fig, out_dir / "trajectory_map.png")


# ---------------------------------------------------------------------------
# Interactive initial-condition picker
# ---------------------------------------------------------------------------

def _pick_ic_interactive(
    gps_e, gps_n, gps_t,
    lidar_e, lidar_n, lidar_t,
    sonar_e, sonar_n, sonar_t,
):
    """
    Show aligned trajectories.  User clicks once on the desired starting point
    (anywhere near the GPS path).  Returns trimmed, re-zeroed arrays and the
    index offsets (di_gps, di_lidar, di_sonar) into the current arrays.
    """
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass  # keep whatever backend is available

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.plot(gps_e,   gps_n,   color=C_GPS,   lw=1.5, label="GPS",   zorder=3)
    ax.plot(lidar_e, lidar_n, color=C_LIDAR, lw=1.5, label="LiDAR", zorder=2)
    ax.plot(sonar_e, sonar_n, color=C_SONAR, lw=1.5, label="Sonar", zorder=1)
    ax.plot(0, 0, "k*", markersize=12, zorder=6, label="Current IC")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Pan/zoom freely.\nSingle-click on the GPS path to select the starting point.")
    ax.legend(loc="best")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    print("\n  ── Interactive IC selection ──────────────────────────────")
    print("  Pan / zoom, then click once on the GPS path to pick the IC.")
    print("  Close the window without clicking to keep the current IC.")

    pts = plt.ginput(1, timeout=0)
    plt.close(fig)

    if not pts:
        print("  No point selected — keeping current IC.")
        return (gps_e, gps_n, gps_t,
                lidar_e, lidar_n, lidar_t,
                sonar_e, sonar_n, sonar_t,
                0, 0, 0)

    click_e, click_n = pts[0]

    # Nearest GPS point to click
    gps_dists = (gps_e - click_e) ** 2 + (gps_n - click_n) ** 2
    i_gps = int(np.argmin(gps_dists))
    ic = np.array([gps_e[i_gps], gps_n[i_gps]])

    # Nearest LiDAR / Sonar to the selected GPS position
    lidar_dists = (lidar_e - ic[0]) ** 2 + (lidar_n - ic[1]) ** 2
    sonar_dists = (sonar_e - ic[0]) ** 2 + (sonar_n - ic[1]) ** 2
    i_lidar = int(np.argmin(lidar_dists))
    i_sonar = int(np.argmin(sonar_dists))

    print(f"  Selected: GPS[{i_gps}] = ({ic[0]:.2f}, {ic[1]:.2f}) m  "
          f"→ LiDAR[{i_lidar}], Sonar[{i_sonar}]")

    def _trim(e, n, t, i):
        return e[i:] - e[i], n[i:] - n[i], t[i:] - t[i]

    gps_e2,   gps_n2,   gps_t2   = _trim(gps_e,   gps_n,   gps_t,   i_gps)
    lidar_e2, lidar_n2, lidar_t2 = _trim(lidar_e, lidar_n, lidar_t, i_lidar)
    sonar_e2, sonar_n2, sonar_t2 = _trim(sonar_e, sonar_n, sonar_t, i_sonar)

    return (gps_e2, gps_n2, gps_t2,
            lidar_e2, lidar_n2, lidar_t2,
            sonar_e2, sonar_n2, sonar_t2,
            i_gps, i_lidar, i_sonar)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",   default=str(_DEFAULT_DATA_DIR))
    ap.add_argument("--out-dir",    default=str(_DEFAULT_OUT_DIR))
    ap.add_argument("--origin-lat", type=float, default=None)
    ap.add_argument("--origin-lon", type=float, default=None)
    ap.add_argument("--zoom",       type=int,   default=17)
    ap.add_argument("--lever-m",      type=float, default=GPS_ANTENNA_OFFSET_M,
                    help=f"North offset for LiDAR/Sonar (m). Default: {GPS_ANTENNA_OFFSET_M}")
    ap.add_argument("--lever-east-m", type=float, default=GPS_ANTENNA_OFFSET_EAST_M,
                    help=f"East offset for LiDAR/Sonar (m). Default: {GPS_ANTENNA_OFFSET_EAST_M}")
    ap.add_argument("--no-align",   action="store_true",
                    help="Skip path alignment")
    ap.add_argument("--pick-ic",    action="store_true",
                    help="Show interactive plot to manually select the initial condition")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()

    # ── Load data ─────────────────────────────────────────────────────────────
    gps_df   = pd.read_csv(data_dir / "gps_path.csv")
    lidar_df = pd.read_csv(data_dir / "lidar_slam.csv")
    sonar_df = pd.read_csv(data_dir / "sonar_odometry.csv")
    nis_df   = pd.read_csv(data_dir / "nis.csv")

    print(f"GPS: {len(gps_df)} rows  |  LiDAR: {len(lidar_df)} rows  "
          f"|  Sonar: {len(sonar_df)} rows  |  NIS: {len(nis_df)} rows")

    # ── Build aligned paths once — shared by all plots and the map ────────────
    print("Aligning paths …")
    gps, lidar, sonar, rot_lidar, rot_sonar, i_gps_df, i_lidar_df, i_sonar_df = \
        _build_aligned_paths(gps_df, lidar_df, sonar_df, args.lever_m, args.lever_east_m,
                             args.no_align)
    gps_t,   gps_e,   gps_n   = gps
    lidar_t, lidar_e, lidar_n = lidar
    sonar_t, sonar_e, sonar_n = sonar

    # ── Optional: interactive IC selection ────────────────────────────────────
    if args.pick_ic:
        (gps_e,   gps_n,   gps_t,
         lidar_e, lidar_n, lidar_t,
         sonar_e, sonar_n, sonar_t,
         di_gps, di_lidar, di_sonar) = _pick_ic_interactive(
            gps_e, gps_n, gps_t,
            lidar_e, lidar_n, lidar_t,
            sonar_e, sonar_n, sonar_t,
        )
        i_gps_df   += di_gps
        i_lidar_df += di_lidar
        i_sonar_df += di_sonar
        gps   = (gps_t,   gps_e,   gps_n)
        lidar = (lidar_t, lidar_e, lidar_n)
        sonar = (sonar_t, sonar_e, sonar_n)

    # Trimmed dataframes: start and end must match the trimmed arrays exactly
    lidar_df_h = lidar_df.iloc[i_lidar_df : i_lidar_df + len(lidar_t)].reset_index(drop=True)
    sonar_df_h = sonar_df.iloc[i_sonar_df : i_sonar_df + len(sonar_t)].reset_index(drop=True)

    # ── Error statistics ──────────────────────────────────────────────────────
    def _err_stats(ref_t, ref_v, other_t, other_v, wrap_rad=False):
        diff = ref_v - _interp(other_t, other_v, ref_t)
        if wrap_rad:
            diff = (diff + np.pi) % (2.0 * np.pi) - np.pi   # heading error → (-π, π]
        err = np.abs(diff)
        err = err[~np.isnan(err)]
        return float(np.mean(err)), float(np.std(err))

    # Heading series (same logic as plot_heading)
    gps_hdg_rad = np.unwrap(_smooth_heading(gps_e, gps_n))

    lidar_hdg_rad = sonar_hdg_rad = None
    if "yaw_deg" in lidar_df_h.columns:
        raw = lidar_df_h["yaw_deg"].values.astype(float) + np.rad2deg(rot_lidar)
        lidar_hdg_rad = np.unwrap(np.deg2rad(raw))
    if "yaw_deg" in sonar_df_h.columns:
        raw = sonar_df_h["yaw_deg"].values.astype(float) + np.rad2deg(rot_sonar)
        sonar_hdg_rad = np.unwrap(np.deg2rad(raw))

    rows: list[tuple[str, float, float]] = []
    rows.append(("Sonar − LiDAR  East  (m)", *_err_stats(sonar_t, sonar_e, lidar_t, lidar_e)))
    rows.append(("Sonar − LiDAR  North (m)", *_err_stats(sonar_t, sonar_n, lidar_t, lidar_n)))
    if sonar_hdg_rad is not None and lidar_hdg_rad is not None:
        mu, sd = _err_stats(sonar_t, sonar_hdg_rad, lidar_t, lidar_hdg_rad, wrap_rad=True)
        rows.append(("Sonar − LiDAR  Heading (°)", np.rad2deg(mu), np.rad2deg(sd)))
    rows.append(("Sonar − GNSS   East  (m)", *_err_stats(sonar_t, sonar_e, gps_t, gps_e)))
    rows.append(("Sonar − GNSS   North (m)", *_err_stats(sonar_t, sonar_n, gps_t, gps_n)))
    if sonar_hdg_rad is not None:
        mu, sd = _err_stats(sonar_t, sonar_hdg_rad, gps_t, gps_hdg_rad, wrap_rad=True)
        rows.append(("Sonar − GNSS   Heading (°)", np.rad2deg(mu), np.rad2deg(sd)))

    header = "Test A (zigzag) — absolute error statistics (Sonar vs LiDAR / GNSS)"
    lines = [header, "=" * len(header), "",
             f"{'':28s}  {'mean |err|':>12s}  {'std |err|':>12s}"]
    for name, mu, sd in rows:
        lines.append(f"{name:28s}  {mu:>12.4f}  {sd:>12.4f}")
    stats_text = "\n".join(lines) + "\n"

    print("\n  ── Absolute error statistics ───────────────────────────────")
    for ln in lines[3:]:
        print("  " + ln)

    results_path = _WS_ROOT / "results" / "error_stats_testA.txt"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(stats_text, encoding="utf-8")
    print(f"\n  stats written → {results_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("Generating plots …")
    plot_nis(nis_df, out_dir)
    plot_inliers(nis_df, out_dir)
    plot_error_summary(gps, lidar, sonar,
                       gps_hdg_rad, lidar_hdg_rad, sonar_hdg_rad, out_dir)

    # ── Trajectory map ────────────────────────────────────────────────────────
    lat0, lon0 = None, None
    if "latitude" in gps_df.columns and "longitude" in gps_df.columns:
        # Use the GPS lat/lon at the trimmed starting row (the common overlap point).
        lat0 = float(gps_df["latitude"].iloc[i_gps_df])
        lon0 = float(gps_df["longitude"].iloc[i_gps_df])
    elif args.origin_lat is not None and args.origin_lon is not None:
        lat0, lon0 = args.origin_lat, args.origin_lon

    if lat0 is None:
        print("\nSkipping trajectory_map.html — GPS lat/lon unavailable.")
        print("  Re-run after bag replay with updated gps_path.py, or pass")
        print("  --origin-lat LAT --origin-lon LON on the command line.")
    else:
        print(f"GPS origin: lat={lat0:.7f}  lon={lon0:.7f}")

        gps_lats,   gps_lons   = enu_to_latlon(gps_e,   gps_n,   lat0, lon0)
        lidar_lats, lidar_lons = enu_to_latlon(lidar_e, lidar_n, lat0, lon0)
        sonar_lats, sonar_lons = enu_to_latlon(sonar_e, sonar_n, lat0, lon0)

        paths = [
            {"lats": gps_lats,   "lons": gps_lons,   "color": C_GPS,   "weight": 9},
            {"lats": lidar_lats, "lons": lidar_lons, "color": C_LIDAR, "weight": 6},
            {"lats": sonar_lats, "lons": sonar_lons, "color": C_SONAR, "weight": 4},
        ]
        # All three paths share the GPS first-fix as their common (0,0) origin
        # after zero-centering + Procrustes alignment, so start markers coincide.
        markers = [
            {"lat": lat0, "lon": lon0, "title": "GPS start",
             "color": C_GPS,   "shape": "circle"},
            {"lat": lat0, "lon": lon0, "title": "LiDAR start",
             "color": C_LIDAR, "shape": "circle"},
            {"lat": lat0, "lon": lon0, "title": "Sonar start",
             "color": C_SONAR, "shape": "circle"},
            {"lat": float(gps_lats[-1]),   "lon": float(gps_lons[-1]),
             "title": "GPS end",   "color": C_GPS,   "shape": "end"},
            {"lat": float(lidar_lats[-1]), "lon": float(lidar_lons[-1]),
             "title": "LiDAR end", "color": C_LIDAR, "shape": "end"},
            {"lat": float(sonar_lats[-1]), "lon": float(sonar_lons[-1]),
             "title": "Sonar end", "color": C_SONAR, "shape": "end"},
        ]
        write_html(out_dir / "trajectory_map.html",
                   float(np.mean(gps_lats)), float(np.mean(gps_lons)),
                   args.zoom, paths, markers)

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
