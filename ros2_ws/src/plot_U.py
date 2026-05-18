#!/usr/bin/env python3
"""
Same outputs as plot_trajectories.py.

Alignment pipeline (same as plot_zigzag.py):
  1. Common time window, trim, re-zero time at window start
  2. Zero-centre each path at its own first sample within the window
  3. Heading alignment  (LiDAR & Sonar rotated about (0,0) to match GPS initial heading)
  4. Rotation-only ICP  (LiDAR & Sonar refined about (0,0) against the GPS path)

Additional drift outputs:
  drift_inliers.pdf      Inlier count over time with low-inlier spans highlighted
  drift_trajectory.pdf   ENU trajectories; Sonar drift segments coloured red

Usage
-----
  python src/analyze_drift.py
  python src/analyze_drift.py --data-dir debug/data --out-dir debug/plots_drift
  python src/analyze_drift.py --threshold 15   # fixed inlier threshold
  python src/analyze_drift.py --no-align       # raw paths, no heading rotation / ICP
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
_DEFAULT_DATA_DIR = _WS_ROOT / "debug" / "U" / "data"
_DEFAULT_OUT_DIR  = _WS_ROOT / "debug" / "U" / "plots_drift"

EARTH_R = 6_371_000.0
NIS_DOF = 3

# ---------------------------------------------------------------------------
# Timing constants (same as plot_trajectories.py)
# ---------------------------------------------------------------------------
GPS_ANTENNA_OFFSET_M: float      = 0.50
GPS_ANTENNA_OFFSET_EAST_M: float = 0.90
LIDAR_SONAR_OFFSET_S: float      = 3.0   # LiDAR starts this many real seconds after Sonar
_SESSION_GAP_S: float            = 1000.0

# ROS bag playback rate for GPS/Sonar (LiDAR ran live at full speed).
BAG_RATE: float = 0.2

# ---------------------------------------------------------------------------
# Visual constants  (identical to plot_trajectories.py)
# ---------------------------------------------------------------------------
C_GPS   = "#000000"
C_LIDAR = "#2166AC"
C_SONAR = "#F4A100"
C_ERR   = "#666666"

C_SONAR_DRIFT = "#d62728"   # red for drift segments on the trajectory plot
C_DRIFT_SPAN  = "#F4A100"   # orange shading on the inlier drift plot
C_THRESH      = "#d62728"
C_MEAN        = "#444444"

LW_GPS   = 2.2
LW_LIDAR = 1.8
LW_SONAR = 1.5
LW_ERR   = 1.2

FS_LABEL  = 13
FS_LEGEND = 12
FS_TICK   = 11

NIS_THRESHOLD = 30.0

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
# Coordinate helpers  (identical to plot_trajectories.py)
# ---------------------------------------------------------------------------

def enu_to_latlon(east_m: np.ndarray, north_m: np.ndarray,
                  lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    lats = lat0 + np.rad2deg(north_m / EARTH_R)
    lons = lon0 + np.rad2deg(east_m / (EARTH_R * np.cos(np.deg2rad(lat0))))
    return lats, lons


def _smooth_heading(east: np.ndarray, north: np.ndarray) -> np.ndarray:
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


def _break_wraps(t: np.ndarray, hdg_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaNs where a wrapped angle jumps >180° so a line plot doesn't
    draw a vertical connector across the ±180° seam."""
    t, hdg = np.asarray(t, float), np.asarray(hdg_deg, float)
    jump = np.abs(np.diff(hdg)) > 180.0
    if not jump.any():
        return t, hdg
    idx = np.where(jump)[0] + 1
    return np.insert(t, idx, np.nan), np.insert(hdg, idx, np.nan)


def rotate_enu(east: np.ndarray, north: np.ndarray,
               angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return c * east - s * north, s * east + c * north


def _interp(src_t: np.ndarray, src_v: np.ndarray,
            dst_t: np.ndarray) -> np.ndarray:
    f = interp1d(src_t, src_v, bounds_error=False, fill_value=np.nan)
    return f(dst_t)


def _init_hdg(e: np.ndarray, n: np.ndarray) -> float:
    for k in range(1, min(50, len(e))):
        de, dn = e[k] - e[0], n[k] - n[0]
        if de * de + dn * dn > 0.01:
            return float(np.arctan2(dn, de))
    return 0.0


def _icp_rot(ref_e: np.ndarray, ref_n: np.ndarray,
             path_e: np.ndarray, path_n: np.ndarray,
             n_iter: int = 100, tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotation-only ICP of `path` onto `ref`, rotating about (0, 0).

    Each iteration: nearest-neighbour correspondences to `ref`, then the
    closed-form least-squares rotation that aligns the current path to those
    matches. Rotating about the origin keeps the (already zero-centred) start
    pinned at (0, 0). Returns (rotated_e, rotated_n, total_angle_rad).
    """
    from scipy.spatial import KDTree
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


# ---------------------------------------------------------------------------
# Alignment  (no time trimming; zero-centre + heading + rotation-only ICP)
# ---------------------------------------------------------------------------

def _build_simple_paths(gps_df: pd.DataFrame, lidar_df: pd.DataFrame,
                        sonar_df: pd.DataFrame,
                        lever_m: float, lever_east_m: float,
                        no_align: bool,
                        bag_rate: float = BAG_RATE):
    """
    Alignment with common time window (same pipeline as plot_zigzag.py):
      Step 1 — Compute physical times, find common window, trim, re-zero at window start.
      Step 2 — Zero-centre each path at its own first sample within the window.
      Step 3 — (optional) Heading alignment: rotate LiDAR & Sonar about (0,0) so
                their initial heading matches GPS.
      Step 4 — (optional) Rotation-only ICP: refine LiDAR & Sonar rotation about
                (0,0) to minimise nearest-neighbour distance to the GPS path.

    GPS and Sonar elapsed times are divided by bag_rate (LiDAR ran live, unchanged).
    Physical reference: T=0 at Sonar start; LiDAR starts at +LIDAR_SONAR_OFFSET_S.
    After trimming, all returned times are relative to the common window start (T=0).

    Returns (gps, lidar, sonar, rot_lidar, rot_sonar, lidar_yaw, sonar_yaw, t_common_start)
      gps/lidar/sonar  — (t, east, north) trimmed to common window, t re-zeroed
      lidar_yaw        — trimmed yaw_deg array for LiDAR, or None
      sonar_yaw        — trimmed yaw_deg array for Sonar, or None
      t_common_start   — physical offset (from Sonar start) of the common window start
    """
    gps_abs_t0   = float(gps_df["time_sec"].iloc[0])
    lidar_abs_t0 = float(lidar_df["time_sec"].iloc[0])
    sonar_abs_t0 = float(sonar_df["time_sec"].iloc[0])

    sonar_same = abs(sonar_abs_t0 - gps_abs_t0) < _SESSION_GAP_S
    gps_phys_offset = (gps_abs_t0 - sonar_abs_t0) / bag_rate if sonar_same else 0.0

    # Elapsed times in real time (bag_rate applied to GPS/Sonar)
    gps_t_raw   = (gps_df["time_sec"].values.astype(float)   - gps_abs_t0) / bag_rate
    lidar_t_raw =  lidar_df["time_sec"].values.astype(float) - lidar_abs_t0
    sonar_t_raw = (sonar_df["time_sec"].values.astype(float) - sonar_abs_t0) / bag_rate

    # Physical time axes (T=0 at Sonar start)
    gps_phys_t   = gps_t_raw   + gps_phys_offset
    lidar_phys_t = lidar_t_raw + LIDAR_SONAR_OFFSET_S
    sonar_phys_t = sonar_t_raw

    # Common window
    T_start = max(float(gps_phys_t[0]), float(sonar_phys_t[0]), float(lidar_phys_t[0]))
    T_end   = min(float(gps_phys_t[-1]), float(sonar_phys_t[-1]), float(lidar_phys_t[-1]))
    print(f"  Common time window: [{T_start:.2f}, {T_end:.2f}] s  "
          f"(duration {T_end - T_start:.1f} s)")

    if T_end <= T_start:
        raise ValueError(f"No common time window: T_start={T_start:.2f} T_end={T_end:.2f}")

    i_g0 = int(np.searchsorted(gps_phys_t,   T_start))
    i_g1 = int(np.searchsorted(gps_phys_t,   T_end, side="right"))
    i_l0 = int(np.searchsorted(lidar_phys_t, T_start))
    i_l1 = int(np.searchsorted(lidar_phys_t, T_end, side="right"))
    i_s0 = int(np.searchsorted(sonar_phys_t, T_start))
    i_s1 = int(np.searchsorted(sonar_phys_t, T_end, side="right"))

    # Re-zero time at common window start (T_start → 0)
    gps_t   = gps_phys_t  [i_g0:i_g1] - T_start
    lidar_t = lidar_phys_t[i_l0:i_l1] - T_start
    sonar_t = sonar_phys_t[i_s0:i_s1] - T_start

    gps_e_raw   = gps_df["east_m"].values.astype(float)  [i_g0:i_g1]
    gps_n_raw   = gps_df["north_m"].values.astype(float) [i_g0:i_g1]
    lidar_e_raw = lidar_df["east_m"].values.astype(float)[i_l0:i_l1]
    lidar_n_raw = lidar_df["north_m"].values.astype(float)[i_l0:i_l1]
    sonar_e_raw = sonar_df["east_m"].values.astype(float)[i_s0:i_s1]
    sonar_n_raw = sonar_df["north_m"].values.astype(float)[i_s0:i_s1]

    lidar_yaw = (lidar_df["yaw_deg"].values.astype(float)[i_l0:i_l1]
                 if "yaw_deg" in lidar_df.columns else None)
    sonar_yaw = (sonar_df["yaw_deg"].values.astype(float)[i_s0:i_s1]
                 if "yaw_deg" in sonar_df.columns else None)

    print(f"  Samples in window — GPS: {len(gps_t)}  LiDAR: {len(lidar_t)}  "
          f"Sonar: {len(sonar_t)}")

    # Zero-centre at first sample within common window
    gps_e   = gps_e_raw   - gps_e_raw[0]
    gps_n   = gps_n_raw   - gps_n_raw[0]
    lidar_e = lidar_e_raw - lidar_e_raw[0]
    lidar_n = lidar_n_raw - lidar_n_raw[0]
    sonar_e = sonar_e_raw - sonar_e_raw[0]
    sonar_n = sonar_n_raw - sonar_n_raw[0]

    rot_lidar = rot_sonar = 0.0

    if not no_align:
        # ── Step 3: heading alignment — match each sensor's initial heading to GPS.
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

        # ── Step 4: rotation-only ICP — refine LiDAR & Sonar against the GPS path.
        lidar_e, lidar_n, drot_L = _icp_rot(gps_e, gps_n, lidar_e, lidar_n)
        sonar_e, sonar_n, drot_S = _icp_rot(gps_e, gps_n, sonar_e, sonar_n)
        rot_lidar += drot_L
        rot_sonar += drot_S

        print(f"\n  Step 4 — ICP refinement (rotation-only, about IC):")
        print(f"    LiDAR {np.rad2deg(drot_L):+.2f}° → total {np.rad2deg(rot_lidar):+.2f}°")
        print(f"    Sonar {np.rad2deg(drot_S):+.2f}° → total {np.rad2deg(rot_sonar):+.2f}°")

    if lever_m != 0 or lever_east_m != 0:
        lidar_n = lidar_n - lever_m
        sonar_n = sonar_n - lever_m
        lidar_e = lidar_e - lever_east_m
        sonar_e = sonar_e - lever_east_m
        print(f"  Antenna offset: LiDAR/Sonar N={-lever_m:+.3f} m  E={-lever_east_m:+.3f} m")

    return (
        (gps_t,   gps_e,   gps_n),
        (lidar_t, lidar_e, lidar_n),
        (sonar_t, sonar_e, sonar_n),
        rot_lidar, rot_sonar,
        lidar_yaw, sonar_yaw,
        T_start,
    )


# ---------------------------------------------------------------------------
# Plot 1: NIS timeseries  (identical to plot_trajectories.py)
# ---------------------------------------------------------------------------

def plot_nis(nis_df: pd.DataFrame, out_dir: Path,
             bag_rate: float = BAG_RATE, t_offset: float = 0.0) -> None:
    # Same time-axis treatment as plot_inliers (real seconds, common-window start).
    t   = (nis_df["time_sec"].values - nis_df["time_sec"].iloc[0]) / bag_rate - t_offset
    nis = nis_df["nis"].values

    mask     = nis <= NIS_THRESHOLD
    t_filt   = t[mask]
    nis_filt = nis[mask]

    mean_nis = float(np.nanmean(nis_filt))
    expected = float(NIS_DOF)

    FS_BASE = 28
    FS_BIG  = FS_BASE + 14   # axis labels
    FS_TICK = FS_BASE + 8    # tick labels
    FS_MEAN = FS_BASE + 4    # mean annotation on the y-axis
    FS_LEG  = FS_BASE + 6    # legend

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
# Plot 2: Inlier count over time  (identical to plot_trajectories.py)
# ---------------------------------------------------------------------------

def plot_inliers(nis_df: pd.DataFrame, out_dir: Path,
                 bag_rate: float = BAG_RATE, t_offset: float = 0.0) -> None:
    # NIS is recorded on the Sonar clock during bag playback; divide by bag_rate
    # to get real seconds, then shift to the common-window start so the x-axis
    # matches the other plots.
    t      = (nis_df["time_sec"].values - nis_df["time_sec"].iloc[0]) / bag_rate - t_offset
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

def plot_position(gps: tuple, lidar: tuple, sonar: tuple, out_dir: Path) -> None:
    gps_t,   gps_e,   gps_n   = gps
    lidar_t, lidar_e, lidar_n = lidar
    sonar_t, sonar_e, sonar_n = sonar

    err_t = gps_t
    err_e = gps_e - _interp(sonar_t, sonar_e, gps_t)
    err_n = gps_n - _interp(sonar_t, sonar_n, gps_t)

    for ylabel, gps_v, lidar_v, sonar_v, err_v, fname in [
        (r"$p^E$ (m)", gps_e, lidar_e, sonar_e, err_e, "east_timeseries.pdf"),
        (r"$p^N$ (m)", gps_n, lidar_n, sonar_n, err_n, "north_timeseries.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(gps_t,   gps_v,   color=C_GPS,   lw=LW_GPS,   label="GPS")
        ax.plot(lidar_t, lidar_v, color=C_LIDAR, lw=LW_LIDAR, label="LiDAR")
        ax.plot(sonar_t, sonar_v, color=C_SONAR, lw=LW_SONAR, label="Sonar")
        ax.plot(err_t,   err_v,   color=C_ERR,   lw=LW_ERR,   linestyle="--",
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
                 lidar_t: np.ndarray, lidar_yaw: np.ndarray | None,
                 sonar_t: np.ndarray, sonar_yaw: np.ndarray | None,
                 rot_lidar: float, rot_sonar: float,
                 out_dir: Path) -> None:
    # Headings wrapped to (-180°, 180°].
    gps_hdg = _wrap_deg(np.rad2deg(_smooth_heading(gps_e, gps_n)))
    lidar_hdg = (_wrap_deg(lidar_yaw + np.rad2deg(rot_lidar))
                 if lidar_yaw is not None else None)
    sonar_hdg = (_wrap_deg(sonar_yaw + np.rad2deg(rot_sonar))
                 if sonar_yaw is not None else None)

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
    _savefig(fig, out_dir / "heading_timeseries.pdf")


# ---------------------------------------------------------------------------
# Plot 6: Combined error plot — Sonar−LiDAR and Sonar−GPS, in E / N / heading
# ---------------------------------------------------------------------------

def plot_error_summary(gps: tuple, lidar: tuple, sonar: tuple,
                       gps_e_arr: np.ndarray, gps_n_arr: np.ndarray,
                       lidar_yaw: np.ndarray | None, sonar_yaw: np.ndarray | None,
                       rot_lidar: float, rot_sonar: float,
                       out_dir: Path) -> None:
    """One figure, 6 curves:
       Sonar−LiDAR : East, North, Heading
       Sonar−GPS   : East, North, Heading
    Everything is evaluated on the Sonar time grid (GPS / LiDAR interpolated).
    Position errors (m) on the left y-axis, heading errors (deg) on the right.
    """
    gps_t,   _, _ = gps
    lidar_t, lidar_e, lidar_n = lidar
    sonar_t, sonar_e, sonar_n = sonar

    # Positions on the Sonar grid.
    e_err_lidar = sonar_e - _interp(lidar_t, lidar_e, sonar_t)
    n_err_lidar = sonar_n - _interp(lidar_t, lidar_n, sonar_t)
    e_err_gps   = sonar_e - _interp(gps_t,   gps[1],  sonar_t)
    n_err_gps   = sonar_n - _interp(gps_t,   gps[2],  sonar_t)

    # Headings unwrapped (continuous) so interpolation/subtraction are well-defined;
    # the heading *error* is computed first, then wrapped to (-180°, 180°].
    sonar_hdg = (np.rad2deg(np.unwrap(np.deg2rad(sonar_yaw + np.rad2deg(rot_sonar))))
                 if sonar_yaw is not None else None)
    lidar_hdg = (np.rad2deg(np.unwrap(np.deg2rad(lidar_yaw + np.rad2deg(rot_lidar))))
                 if lidar_yaw is not None else None)
    gps_hdg   = np.rad2deg(np.unwrap(_smooth_heading(gps_e_arr, gps_n_arr)))

    h_err_lidar = (_wrap_deg(sonar_hdg - _interp(lidar_t, lidar_hdg, sonar_t))
                   if (sonar_hdg is not None and lidar_hdg is not None) else None)
    h_err_gps   = (_wrap_deg(sonar_hdg - _interp(gps_t, gps_hdg, sonar_t))
                   if sonar_hdg is not None else None)

    fig, ax = plt.subplots(figsize=(24, 12))
    ax_h = ax.twinx()

    # East = blue, North = red, Heading = green. Solid = Sonar − LiDAR, dashed = Sonar − GPS.
    # Sonar − GPS drawn thicker and on top so it stays visible where it overlaps Sonar − LiDAR.
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
        ht, hh = _break_wraps(sonar_t, h_err_lidar)
        ax_h.plot(ht, hh, color=C_HDG, lw=LW_LD, zorder=2,
                  label="Heading: Sonar − LiDAR")
    if h_err_gps is not None:
        ht, hh = _break_wraps(sonar_t, h_err_gps)
        ax_h.plot(ht, hh, color=C_HDG, lw=LW_GP, linestyle="--", zorder=4,
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
    ax_h.set_ylim(-180, 180 * 1.5)
    ax_h.set_yticks([-180, -90, 0, 90, 180])

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
# Absolute error statistics  (Sonar ↔ LiDAR, Sonar ↔ GNSS)
# ---------------------------------------------------------------------------

def write_error_stats(gps: tuple, lidar: tuple, sonar: tuple,
                      gps_e_arr: np.ndarray, gps_n_arr: np.ndarray,
                      lidar_yaw: np.ndarray | None, sonar_yaw: np.ndarray | None,
                      rot_lidar: float, rot_sonar: float,
                      test_label: str, results_path: Path) -> None:
    """Print and persist mean / std of the absolute error between the Sonar path
    and the LiDAR / GNSS paths, in East, North and Heading."""
    gps_t,   _, _ = gps
    lidar_t, lidar_e, lidar_n = lidar
    sonar_t, sonar_e, sonar_n = sonar

    def _stats(ref_t, ref_v, other_t, other_v, wrap_deg=False):
        diff = ref_v - _interp(other_t, other_v, ref_t)
        if wrap_deg:
            diff = _wrap_deg(diff)          # heading error → (-180°, 180°]
        err = np.abs(diff)
        err = err[~np.isnan(err)]
        return float(np.mean(err)), float(np.std(err))

    # Headings (deg), unwrapped, with the alignment rotations applied. The
    # *error* is wrapped to (-180°, 180°] after subtraction (see _stats).
    sonar_hdg = (np.rad2deg(np.unwrap(np.deg2rad(sonar_yaw + np.rad2deg(rot_sonar))))
                 if sonar_yaw is not None else None)
    lidar_hdg = (np.rad2deg(np.unwrap(np.deg2rad(lidar_yaw + np.rad2deg(rot_lidar))))
                 if lidar_yaw is not None else None)
    gps_hdg   = np.rad2deg(np.unwrap(_smooth_heading(gps_e_arr, gps_n_arr)))

    rows: list[tuple[str, float, float]] = []
    # Sonar − LiDAR
    rows.append(("Sonar − LiDAR  East  (m)", *_stats(sonar_t, sonar_e, lidar_t, lidar_e)))
    rows.append(("Sonar − LiDAR  North (m)", *_stats(sonar_t, sonar_n, lidar_t, lidar_n)))
    if sonar_hdg is not None and lidar_hdg is not None:
        rows.append(("Sonar − LiDAR  Heading (°)",
                     *_stats(sonar_t, sonar_hdg, lidar_t, lidar_hdg, wrap_deg=True)))
    # Sonar − GNSS
    rows.append(("Sonar − GNSS   East  (m)", *_stats(sonar_t, sonar_e, gps_t, gps_e_arr)))
    rows.append(("Sonar − GNSS   North (m)", *_stats(sonar_t, sonar_n, gps_t, gps_n_arr)))
    if sonar_hdg is not None:
        rows.append(("Sonar − GNSS   Heading (°)",
                     *_stats(sonar_t, sonar_hdg, gps_t, gps_hdg, wrap_deg=True)))

    header = (f"{test_label} — absolute error statistics "
              f"(Sonar vs LiDAR / GNSS)")
    lines = [header, "=" * len(header), "",
             f"{'':28s}  {'mean |err|':>12s}  {'std |err|':>12s}"]
    for name, mu, sd in rows:
        lines.append(f"{name:28s}  {mu:>12.4f}  {sd:>12.4f}")
    text = "\n".join(lines) + "\n"

    print("\n  ── Absolute error statistics ───────────────────────────────")
    for ln in lines[3:]:
        print("  " + ln)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(text, encoding="utf-8")
    print(f"\n  stats written → {results_path}")


# ---------------------------------------------------------------------------
# HTML map  (identical to plot_trajectories.py)
# ---------------------------------------------------------------------------

GOOGLE_MAPS_API_KEY: str = "AIzaSyA2T1UdCtUsR5izwnzkSXtB4c5ZjVhMDvw"


def write_html(out_path: Path, center_lat: float, center_lon: float,
               zoom: int, paths: list[dict], markers: list[dict]) -> None:
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
# Static satellite map image  (identical to plot_trajectories.py)
# ---------------------------------------------------------------------------

def _tile_coords(lat_deg: float, lon_deg: float, zoom: int) -> tuple[float, float]:
    n   = 2 ** zoom
    tx  = (lon_deg + 180.0) / 360.0 * n
    lat_r = math.radians(lat_deg)
    ty  = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return tx, ty


def _fetch_tile(z: int, x: int, y: int):
    from PIL import Image as _PILImage
    url = (f"https://server.arcgisonline.com/ArcGIS/rest/services/"
           f"World_Imagery/MapServer/tile/{z}/{y}/{x}")
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "analyze_drift/1.0 ros2_ws (educational use)"},
            timeout=10,
        )
        resp.raise_for_status()
        return _PILImage.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        print(f"    [tile {z}/{x}/{y} failed: {exc}]")
        return None


def _build_basemap(all_lats: np.ndarray, all_lons: np.ndarray,
                   zoom: int, pad: int = 1):
    from PIL import Image as _PILImage
    TILE_PX   = 256
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
    print("  Building static trajectory map image …")
    all_lats = np.concatenate([gps_lats, lidar_lats, sonar_lats])
    all_lons = np.concatenate([gps_lons, lidar_lons, sonar_lons])

    try:
        basemap, x0, y0, used_zoom, tile_px = _build_basemap(all_lats, all_lons, zoom)
    except Exception as exc:
        print(f"    basemap fetch failed: {exc} — skipping trajectory_map.png")
        return

    img_w, img_h = basemap.size

    def to_px(lats, lons):
        return _latlon_to_px(
            _downsample(lats), _downsample(lons), used_zoom, x0, y0, tile_px,
        )

    gx, gy = to_px(gps_lats,   gps_lons)
    lx, ly = to_px(lidar_lats, lidar_lons)
    sx, sy = to_px(sonar_lats, sonar_lons)

    def endpoints(lats, lons):
        px0, py0 = _latlon_to_px(lats[[0]],  lons[[0]],  used_zoom, x0, y0, tile_px)
        px1, py1 = _latlon_to_px(lats[[-1]], lons[[-1]], used_zoom, x0, y0, tile_px)
        return (px0[0], py0[0]), (px1[0], py1[0])

    gps_s,   gps_e_pt   = endpoints(gps_lats,   gps_lons)
    lidar_s, lidar_e_pt = endpoints(lidar_lats, lidar_lons)
    sonar_s, sonar_e_pt = endpoints(sonar_lats, sonar_lons)

    all_px = np.concatenate([gx, lx, sx])
    all_py = np.concatenate([gy, ly, sy])
    pad_x  = max(40, (all_px.max() - all_px.min()) * 0.08)
    pad_y  = max(40, (all_py.max() - all_py.min()) * 0.08)
    xlim   = (all_px.min() - pad_x, all_px.max() + pad_x)
    ylim   = (all_py.max() + pad_y, all_py.min() - pad_y)

    crop_w = xlim[1] - xlim[0]
    crop_h = ylim[0] - ylim[1]
    base_w = 10.0
    base_h = max(base_w * crop_h / max(crop_w, 1.0), 6.0)
    fig, ax = plt.subplots(figsize=(base_w, base_h), dpi=150)
    ax.imshow(np.asarray(basemap), extent=[0, img_w, img_h, 0],
              origin="upper", aspect="equal", zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    LW_MAP_GPS   = 4.0
    LW_MAP_LIDAR = 3.0
    LW_MAP_SONAR = 2.5
    ax.plot(gx, gy, color=C_GPS,   lw=LW_MAP_GPS,   zorder=3, alpha=0.95, label="GPS")
    ax.plot(lx, ly, color=C_LIDAR, lw=LW_MAP_LIDAR, zorder=2, alpha=0.92, label="LiDAR")
    ax.plot(sx, sy, color=C_SONAR, lw=LW_MAP_SONAR, zorder=1, alpha=0.90, label="Sonar")

    mk = dict(zorder=6, markersize=13, markeredgewidth=2.0,
              markeredgecolor="white", linestyle="None")
    for (spx, spy), (epx, epy), col in [
        (gps_s,   gps_e_pt,   C_GPS),
        (lidar_s, lidar_e_pt, C_LIDAR),
        (sonar_s, sonar_e_pt, C_SONAR),
    ]:
        ax.plot(spx, spy, marker="o", color=col, **mk)
        ax.plot(epx, epy, marker="v", color=col, **mk)

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
    leg = ax.legend(handles=legend_handles, loc="lower right", framealpha=0.92,
                    facecolor="white", edgecolor="#aaaaaa",
                    fontsize=max(FS_LEGEND + 2, 12),
                    labelspacing=0.6, handlelength=2.5, borderpad=0.8)
    leg.set_zorder(10)
    ax.set_axis_off()
    fig.tight_layout(pad=0.3)
    _savefig(fig, out_dir / "trajectory_map.png")


# ---------------------------------------------------------------------------
# Drift-specific plots
# ---------------------------------------------------------------------------

def detect_drift_regions(values: np.ndarray, threshold: float,
                         min_duration: int = 3) -> list[tuple[int, int]]:
    below = values < threshold
    regions = []
    i = 0
    while i < len(below):
        if below[i]:
            j = i
            while j < len(below) and below[j]:
                j += 1
            if j - i >= min_duration:
                regions.append((i, j - 1))
            i = j
        else:
            i += 1
    return regions


def _plot_segmented(ax: plt.Axes,
                    e: np.ndarray, n: np.ndarray, mask: np.ndarray,
                    normal_color: str, drift_color: str, lw: float) -> None:
    """Plot path in two colours determined by a boolean mask."""
    n_pts = len(e)
    i = 0
    drift_used = normal_used = False
    while i < n_pts:
        j, cur = i, mask[i]
        while j < n_pts and mask[j] == cur:
            j += 1
        end = min(j + 1, n_pts)   # include one bridge point
        seg_e, seg_n = e[i:end], n[i:end]
        if cur:
            label = "Sonar (drift)" if not drift_used else None
            drift_used = True
            ax.plot(seg_e, seg_n, color=drift_color, lw=lw * 1.5,
                    label=label, zorder=3, alpha=0.95)
        else:
            label = "Sonar" if not normal_used else None
            normal_used = True
            ax.plot(seg_e, seg_n, color=normal_color, lw=lw,
                    label=label, zorder=2, alpha=0.85)
        i = j


def plot_drift_inliers(nis_df: pd.DataFrame,
                       regions: list[tuple[int, int]],
                       threshold: float, mean_c: float,
                       smooth: np.ndarray,
                       out_dir: Path,
                       bag_rate: float = BAG_RATE,
                       t_offset: float = 0.0) -> None:
    t      = (nis_df["time_sec"].values - nis_df["time_sec"].iloc[0]) / bag_rate - t_offset
    counts = nis_df["inlier_count"].values

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, counts, color=C_MEAN, linewidth=0.6, alpha=0.35,
            marker="o", markersize=1.2, label="Inlier count (raw)")
    ax.plot(t, smooth, color="#2166AC", linewidth=1.8, label="Smoothed")
    ax.axhline(mean_c, color=C_MEAN, linewidth=1.2, linestyle="--",
               label=f"Mean ({mean_c:.1f})")
    ax.axhline(threshold, color=C_THRESH, linewidth=1.2, linestyle=":",
               label=f"Drift threshold ({threshold:.1f})")
    for k, (s, e) in enumerate(regions):
        ax.axvspan(t[s], t[e], color=C_DRIFT_SPAN, alpha=0.35,
                   label="Heading drift" if k == 0 else None)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RANSAC Inlier Count")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=10)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    _savefig(fig, out_dir / "drift_inliers.pdf")


def plot_drift_trajectory(gps: tuple, lidar: tuple, sonar: tuple,
                          nis_df: pd.DataFrame,
                          regions: list[tuple[int, int]],
                          out_dir: Path,
                          bag_rate: float = BAG_RATE,
                          t_offset: float = 0.0) -> None:
    gps_t,   gps_e,   gps_n   = gps
    lidar_t, lidar_e, lidar_n = lidar
    sonar_t, sonar_e, sonar_n = sonar

    # NIS times scaled to real time and shifted to match common window (T=0)
    nis_t = (nis_df["time_sec"].values - nis_df["time_sec"].iloc[0]) / bag_rate - t_offset
    drift_mask = np.zeros(len(sonar_t), dtype=bool)
    for s, e in regions:
        drift_mask |= (sonar_t >= nis_t[s]) & (sonar_t <= nis_t[e])

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.plot(gps_e,   gps_n,   color=C_GPS,   lw=LW_GPS,   label="GPS",   zorder=4)
    ax.plot(lidar_e, lidar_n, color=C_LIDAR, lw=LW_LIDAR, label="LiDAR", zorder=3)
    _plot_segmented(ax, sonar_e, sonar_n, drift_mask, C_SONAR, C_SONAR_DRIFT, LW_SONAR)
    ax.plot(0, 0, "ko", markersize=9, zorder=7, label="Start")
    for col, xe, xn, title in [
        (C_GPS,   gps_e[-1],   gps_n[-1],   "GPS end"),
        (C_LIDAR, lidar_e[-1], lidar_n[-1], "LiDAR end"),
        (C_SONAR_DRIFT if drift_mask[-1] else C_SONAR,
         sonar_e[-1], sonar_n[-1], "Sonar end"),
    ]:
        ax.plot(xe, xn, marker="v", color=col, markersize=9,
                zorder=7, linestyle="None", label=title)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=10)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    _savefig(fig, out_dir / "drift_trajectory.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Same outputs as plot_trajectories.py but without time-window "
                    "trimming for the standard plots (alignment = zero-centre + "
                    "heading + rotation-only ICP, as in plot_zigzag.py)."
    )
    ap.add_argument("--data-dir",     default=str(_DEFAULT_DATA_DIR))
    ap.add_argument("--out-dir",      default=str(_DEFAULT_OUT_DIR))
    ap.add_argument("--origin-lat",   type=float, default=None)
    ap.add_argument("--origin-lon",   type=float, default=None)
    ap.add_argument("--zoom",         type=int,   default=17)
    ap.add_argument("--lever-m",      type=float, default=GPS_ANTENNA_OFFSET_M,
                    help=f"North offset for LiDAR/Sonar (m). Default: {GPS_ANTENNA_OFFSET_M}")
    ap.add_argument("--lever-east-m", type=float, default=GPS_ANTENNA_OFFSET_EAST_M,
                    help=f"East offset for LiDAR/Sonar (m). Default: {GPS_ANTENNA_OFFSET_EAST_M}")
    ap.add_argument("--no-align",     action="store_true",
                    help="Skip heading alignment and ICP (raw zero-centred paths)")
    ap.add_argument("--bag-rate",     type=float, default=BAG_RATE,
                    help=f"ROS bag playback rate for GPS/Sonar (default {BAG_RATE}). "
                         "LiDAR is assumed to run at full speed (1.0).")
    ap.add_argument("--threshold",    type=float, default=None,
                    help="Inlier count drift threshold. Default: mean − 1×std")
    ap.add_argument("--min-dur",      type=int,   default=5,
                    help="Minimum consecutive samples below threshold to flag as drift")
    ap.add_argument("--window",       type=int,   default=20,
                    help="Savitzky-Golay smoothing window for drift detection")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()

    # ── Load data ──────────────────────────────────────────────────────────────
    gps_df   = pd.read_csv(data_dir / "gps_path.csv")
    lidar_df = pd.read_csv(data_dir / "lidar_slam.csv")
    sonar_df = pd.read_csv(data_dir / "sonar_odometry.csv")
    nis_df   = pd.read_csv(data_dir / "nis.csv")

    print(f"GPS: {len(gps_df)} rows  |  LiDAR: {len(lidar_df)} rows  "
          f"|  Sonar: {len(sonar_df)} rows  |  NIS: {len(nis_df)} rows")

    # ── Build aligned paths ───────────────────────────────────────────────────
    print("Building paths (common time window, zero-centre, heading + ICP) …")
    gps, lidar, sonar, rot_lidar, rot_sonar, lidar_yaw, sonar_yaw, T_start = \
        _build_simple_paths(
            gps_df, lidar_df, sonar_df, args.lever_m, args.lever_east_m, args.no_align,
            bag_rate=args.bag_rate,
        )
    gps_t,   gps_e,   gps_n   = gps
    lidar_t, lidar_e, lidar_n = lidar
    sonar_t, sonar_e, sonar_n = sonar

    # All times are already re-zeroed to the common window start — use directly.

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("Generating plots …")
    plot_nis(nis_df, out_dir, bag_rate=args.bag_rate, t_offset=T_start)
    plot_inliers(nis_df, out_dir, bag_rate=args.bag_rate, t_offset=T_start)
    plot_error_summary(gps, lidar, sonar,
                       gps_e, gps_n,
                       lidar_yaw, sonar_yaw,
                       rot_lidar, rot_sonar, out_dir)

    # ── Error statistics → results/error_stats_testB.txt ──────────────────────
    write_error_stats(gps, lidar, sonar, gps_e, gps_n, lidar_yaw, sonar_yaw,
                      rot_lidar, rot_sonar,
                      test_label="Test B (U)",
                      results_path=_WS_ROOT / "results" / "error_stats_testB.txt")

    # ── Trajectory map ────────────────────────────────────────────────────────
    lat0, lon0 = None, None
    if "latitude" in gps_df.columns and "longitude" in gps_df.columns:
        lat0 = float(gps_df["latitude"].iloc[0])
        lon0 = float(gps_df["longitude"].iloc[0])
    elif args.origin_lat is not None and args.origin_lon is not None:
        lat0, lon0 = args.origin_lat, args.origin_lon

    if lat0 is None:
        print("\nSkipping trajectory_map.html — GPS lat/lon unavailable.")
        print("  Pass --origin-lat LAT --origin-lon LON on the command line.")
    else:
        print(f"\nGPS origin: lat={lat0:.7f}  lon={lon0:.7f}")

        gps_lats,   gps_lons   = enu_to_latlon(gps_e,   gps_n,   lat0, lon0)
        lidar_lats, lidar_lons = enu_to_latlon(lidar_e, lidar_n, lat0, lon0)
        sonar_lats, sonar_lons = enu_to_latlon(sonar_e, sonar_n, lat0, lon0)

        paths = [
            {"lats": gps_lats,   "lons": gps_lons,   "color": C_GPS,   "weight": 9},
            {"lats": lidar_lats, "lons": lidar_lons, "color": C_LIDAR, "weight": 6},
            {"lats": sonar_lats, "lons": sonar_lons, "color": C_SONAR, "weight": 4},
        ]
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
