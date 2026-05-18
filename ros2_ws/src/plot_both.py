#!/usr/bin/env python3
"""
Combined NIS and inlier-count figures for the two runs.

Produces two PNGs, each with two stacked subplots (Test B = U on top,
Test A = zigzag below) and a single shared legend, intended to drop into
Overleaf as one figure:

  nis_both.png            stacked: Test A NIS (top), Test B NIS (bottom)
  inliers_both.png        stacked: Test A inliers (top), Test B inliers (bottom)
  trajectories_both.png   side by side: Test A | Test B trajectories on
                          Esri satellite imagery (GPS / LiDAR / Sonar each panel)

NIS/inlier x-axes use raw elapsed time for Test A and bag-rate-corrected real
time for Test B.

Usage
-----
  python src/plot_both.py
  python src/plot_both.py --u-data debug/U/data --zigzag-data debug/zigzag/data \
                          --out-dir debug/both
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

_WS_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_U_DATA       = _WS_ROOT / "debug" / "U" / "data"
_DEFAULT_ZIGZAG_DATA  = _WS_ROOT / "debug" / "zigzag" / "data"
_DEFAULT_OUT_DIR      = _WS_ROOT / "debug" / "both"

NIS_DOF       = 3
NIS_THRESHOLD = 30.0
EARTH_R       = 6_371_000.0

FS = 28

# Trajectory colours (match plot_U.py / plot_zigzag.py).
C_GPS, C_LIDAR, C_SONAR = "#000000", "#2166AC", "#F4A100"
MAP_ZOOM = 18

# ROS bag playback rate for the U run (Test B): its CSV timestamps are in
# bag-playback seconds, so divide elapsed time by this to get real seconds.
# The zigzag run (Test A) was recorded at full speed → no scaling.
U_BAG_RATE = 0.2

# (label, dataframe-key) — top to bottom in the figure.
TESTS = [("Test A", "z"), ("Test B", "u")]
# Time-axis scale factor per test (elapsed seconds → real seconds).
TIME_SCALE = {"Test B": 1.0 / U_BAG_RATE, "Test A": 1.0}

C_NIS  = "#444444"
C_INL  = "#3a7d44"
C_MEAN = "#2ca02c"   # mean line — same in both subplots; value shown on the y-axis
C_EXP  = "#d62728"


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


def _elapsed(df: pd.DataFrame) -> np.ndarray:
    return df["time_sec"].values.astype(float) - float(df["time_sec"].iloc[0])


def _shared_legend(fig: plt.Figure, handles: list, labels: list, ncol: int) -> None:
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=ncol, fontsize=FS - 4, framealpha=0.9, handlelength=2.0,
               columnspacing=1.4, borderpad=0.5)


def _label_mean_on_yaxis(ax: plt.Axes, mean_val: float, color: str,
                         fmt: str = "{:.1f}", fontsize: int | None = None) -> None:
    """Write the mean value on the y-axis (just outside the left spine) at the
    height of the mean line, coloured to match it. Drop any default y-tick that
    would collide with the mean label."""
    y0, y1 = ax.get_ylim()
    ticks  = [t for t in ax.get_yticks() if y0 <= t <= y1]
    if len(ticks) >= 2:
        spacing = min(b - a for a, b in zip(ticks, ticks[1:]))
        ax.set_yticks([t for t in ticks if abs(t - mean_val) > 0.45 * spacing])
    ax.annotate(fmt.format(mean_val),
                xy=(0, mean_val), xycoords=("axes fraction", "data"),
                xytext=(-8, 0), textcoords="offset points",
                ha="right", va="center",
                fontsize=fontsize if fontsize is not None else FS - 6,
                color=color, fontweight="bold", clip_on=False)


# ---------------------------------------------------------------------------
# NIS figure
# ---------------------------------------------------------------------------

def plot_nis_both(dfs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    from matplotlib.lines import Line2D

    FS_BIG = FS + 14   # axis labels, titles
    FS_TICK = FS + 8   # tick labels
    FS_MEAN = FS + 4   # mean annotation on the y-axis
    FS_LEG  = FS + 6   # shared legend

    fig, axes = plt.subplots(2, 1, figsize=(16, 17))

    for ax, (title, key) in zip(axes, TESTS):
        nis_df = dfs[key]
        t   = _elapsed(nis_df) * TIME_SCALE[title]
        nis = nis_df["nis"].values.astype(float)
        mask = nis <= NIS_THRESHOLD
        t_f, nis_f = t[mask], nis[mask]
        mean_nis = float(np.nanmean(nis_f))

        ax.plot(t_f, nis_f, color=C_NIS, lw=0.9, marker="o", markersize=1.4,
                alpha=0.75)
        ax.axhline(mean_nis, color=C_MEAN, lw=2.0)
        ax.axhline(float(NIS_DOF), color=C_EXP, lw=2.0, linestyle="--")
        ax.set_ylim(bottom=0, top=NIS_THRESHOLD)
        ax.set_ylabel("NIS", fontsize=FS_BIG)
        ax.set_title(title, fontsize=FS_BIG, style="italic")
        ax.tick_params(labelsize=FS_TICK)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        _label_mean_on_yaxis(ax, mean_nis, C_MEAN, fmt="{:.1f}", fontsize=FS_MEAN)
    axes[-1].set_xlabel("Time (s)", fontsize=FS_BIG)

    handles = [
        Line2D([0], [0], color=C_NIS, lw=1.5, marker="o", markersize=4),
        Line2D([0], [0], color=C_MEAN, lw=2.0),
        Line2D([0], [0], color=C_EXP, lw=2.0, linestyle="--"),
    ]
    labels = ["NIS", "Mean", f"Expected ({NIS_DOF})"]
    fig.tight_layout()
    # Centre the legend over the plotting area (axes bbox), not the full figure
    # — the y-axis label + mean annotation make the axes offset to the right.
    top_bbox = axes[0].get_position()
    legend_x = (top_bbox.x0 + top_bbox.x1) / 2
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(legend_x, top_bbox.y1 + 0.04),
               bbox_transform=fig.transFigure,
               ncol=3, fontsize=FS_LEG, framealpha=0.9, handlelength=2.0,
               columnspacing=1.4, borderpad=0.5)
    _savefig(fig, out_dir / "nis_both.png")


# ---------------------------------------------------------------------------
# Inlier-count figure
# ---------------------------------------------------------------------------

def plot_inliers_both(dfs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    from matplotlib.lines import Line2D

    FS_BIG = FS + 14   # axis labels, titles
    FS_TICK = FS + 8   # tick labels
    FS_MEAN = FS + 4   # mean annotation on the y-axis
    FS_LEG  = FS + 6   # shared legend

    fig, axes = plt.subplots(2, 1, figsize=(16, 17))

    for ax, (title, key) in zip(axes, TESTS):
        nis_df = dfs[key]
        t      = _elapsed(nis_df) * TIME_SCALE[title]
        counts = nis_df["inlier_count"].values.astype(float)
        mean_c = float(np.mean(counts))

        ax.plot(t, counts, color=C_INL, lw=0.9, marker="o", markersize=1.4,
                alpha=0.85)
        ax.axhline(mean_c, color="#444444", lw=2.0, linestyle="--")
        ax.set_ylabel("Inlier Count", fontsize=FS_BIG)
        ax.set_title(title, fontsize=FS_BIG, style="italic")
        ax.tick_params(labelsize=FS_TICK)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        _label_mean_on_yaxis(ax, mean_c, "#444444", fmt="{:.0f}", fontsize=FS_MEAN)
    axes[-1].set_xlabel("Time (s)", fontsize=FS_BIG)

    handles = [
        Line2D([0], [0], color=C_INL, lw=1.5, marker="o", markersize=4),
        Line2D([0], [0], color="#444444", lw=2.0, linestyle="--"),
    ]
    labels = ["Inliers", "Mean"]
    fig.tight_layout()
    # Centre the legend over the plotting area (axes bbox), not the full figure
    # — the y-axis label + mean annotation make the axes offset to the right.
    top_bbox = axes[0].get_position()
    legend_x = (top_bbox.x0 + top_bbox.x1) / 2
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(legend_x, top_bbox.y1 + 0.04),
               bbox_transform=fig.transFigure,
               ncol=2, fontsize=FS_LEG, framealpha=0.9, handlelength=2.0,
               columnspacing=1.4, borderpad=0.5)
    _savefig(fig, out_dir / "inliers_both.png")


# ---------------------------------------------------------------------------
# Satellite-backed trajectory panels
# ---------------------------------------------------------------------------

def _enu_to_latlon(e: np.ndarray, n: np.ndarray, lat0: float, lon0: float):
    lats = lat0 + np.rad2deg(n / EARTH_R)
    lons = lon0 + np.rad2deg(e / (EARTH_R * np.cos(np.deg2rad(lat0))))
    return lats, lons


def _rotate(e: np.ndarray, n: np.ndarray, a: float):
    c, s = np.cos(a), np.sin(a)
    return c * e - s * n, s * e + c * n


def _init_hdg(e: np.ndarray, n: np.ndarray) -> float:
    for k in range(1, min(80, len(e))):
        de, dn = e[k] - e[0], n[k] - n[0]
        if de * de + dn * dn > 0.04:
            return float(np.arctan2(dn, de))
    return 0.0


def _aligned_latlon(data_dir: Path):
    """Load a run's GPS/LiDAR/Sonar, zero-centre each, heading-align LiDAR &
    Sonar to GPS, convert to lat/lon using the GPS first fix as origin."""
    gps = pd.read_csv(data_dir / "gps_path.csv")
    lid = pd.read_csv(data_dir / "lidar_slam.csv")
    son = pd.read_csv(data_dir / "sonar_odometry.csv")
    lat0, lon0 = float(gps["latitude"].iloc[0]), float(gps["longitude"].iloc[0])

    def _zc(df):
        e = df["east_m"].values.astype(float);  e = e - e[0]
        n = df["north_m"].values.astype(float); n = n - n[0]
        return e, n

    ge, gn = _zc(gps); le, ln = _zc(lid); se, sn = _zc(son)
    g_h = _init_hdg(ge, gn)
    le, ln = _rotate(le, ln, g_h - _init_hdg(le, ln))
    se, sn = _rotate(se, sn, g_h - _init_hdg(se, sn))

    return {
        "GPS":   _enu_to_latlon(ge, gn, lat0, lon0),
        "LiDAR": _enu_to_latlon(le, ln, lat0, lon0),
        "Sonar": _enu_to_latlon(se, sn, lat0, lon0),
    }


def _tile_coords(lat_deg: float, lon_deg: float, z: int):
    n   = 2 ** z
    tx  = (lon_deg + 180.0) / 360.0 * n
    lat_r = math.radians(lat_deg)
    ty  = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return tx, ty


def _fetch_tile(z: int, x: int, y: int):
    from PIL import Image
    url = (f"https://server.arcgisonline.com/ArcGIS/rest/services/"
           f"World_Imagery/MapServer/tile/{z}/{y}/{x}")
    try:
        r = requests.get(url, headers={"User-Agent": "plot_both/1.0 (educational)"},
                         timeout=10)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as exc:
        print(f"    [tile {z}/{x}/{y} failed: {exc}]")
        return None


def _build_basemap(all_lats, all_lons, zoom, pad=1, max_tiles=64):
    from PIL import Image
    TILE = 256
    for z in range(zoom, max(zoom - 5, 10) - 1, -1):
        ntot = 2 ** z
        tx0, ty0 = _tile_coords(max(all_lats), min(all_lons), z)
        tx1, ty1 = _tile_coords(min(all_lats), max(all_lons), z)
        x0 = max(0, int(tx0) - pad); y0 = max(0, int(ty0) - pad)
        x1 = min(ntot - 1, int(tx1) + pad); y1 = min(ntot - 1, int(ty1) + pad)
        nx, ny = x1 - x0 + 1, y1 - y0 + 1
        if nx * ny <= max_tiles:
            break
    print(f"    fetching {nx}x{ny}={nx*ny} tiles at zoom {z} …")
    canvas = Image.new("RGB", (nx * TILE, ny * TILE), (210, 210, 210))
    for xi in range(nx):
        for yi in range(ny):
            t = _fetch_tile(z, x0 + xi, y0 + yi)
            if t is not None:
                canvas.paste(t, (xi * TILE, yi * TILE))
    return canvas, x0, y0, z, TILE


def _latlon_to_px(lats, lons, z, x0, y0, tile=256):
    n  = 2 ** z
    px = (np.asarray(lons) + 180.0) / 360.0 * n * tile - x0 * tile
    lat_r = np.radians(np.asarray(lats))
    py = ((1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / math.pi)
          / 2.0 * n * tile - y0 * tile)
    return px, py


def _downsample(a, max_pts=3000):
    a = np.asarray(a)
    return a[:: max(1, len(a) // max_pts)]


def _draw_panel(ax: plt.Axes, paths: dict, zoom: int) -> None:
    all_lats = np.concatenate([np.asarray(v[0]) for v in paths.values()])
    all_lons = np.concatenate([np.asarray(v[1]) for v in paths.values()])
    basemap, x0, y0, zz, tile = _build_basemap(all_lats, all_lons, zoom)
    img_w, img_h = basemap.size
    ax.imshow(np.asarray(basemap), extent=[0, img_w, img_h, 0],
              origin="upper", aspect="equal", zorder=0)

    styling = [("GPS", C_GPS, 4.5, 4), ("LiDAR", C_LIDAR, 3.2, 3), ("Sonar", C_SONAR, 2.6, 2)]
    px_all, py_all = [], []
    for name, col, lw, zo in styling:
        lats, lons = paths[name]
        px, py = _latlon_to_px(_downsample(lats), _downsample(lons), zz, x0, y0, tile)
        ax.plot(px, py, color=col, lw=lw, alpha=0.95, zorder=zo)
        px_all.append(px); py_all.append(py)
        # start ● / end ▼
        spx, spy = _latlon_to_px([lats[0]],  [lons[0]],  zz, x0, y0, tile)
        epx, epy = _latlon_to_px([lats[-1]], [lons[-1]], zz, x0, y0, tile)
        ax.plot(spx, spy, marker="o", color=col, markersize=11, mec="white", mew=1.8,
                ls="None", zorder=6)
        ax.plot(epx, epy, marker="v", color=col, markersize=11, mec="white", mew=1.8,
                ls="None", zorder=6)

    px_all = np.concatenate(px_all); py_all = np.concatenate(py_all)
    pad_x = max(30, (px_all.max() - px_all.min()) * 0.10)
    pad_y = max(30, (py_all.max() - py_all.min()) * 0.10)
    ax.set_xlim(px_all.min() - pad_x, px_all.max() + pad_x)
    ax.set_ylim(py_all.max() + pad_y, py_all.min() - pad_y)   # y flipped
    ax.set_axis_off()


def plot_trajectories_both(u_data: Path, z_data: Path, out_dir: Path,
                           zoom: int = MAP_ZOOM) -> None:
    from matplotlib.lines import Line2D
    print("Building satellite trajectory panels …")
    panels = [("Test A", _aligned_latlon(z_data)), ("Test B", _aligned_latlon(u_data))]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for ax, (title, paths) in zip(axes, panels):
        _draw_panel(ax, paths, zoom)
        ax.set_title(title, fontsize=FS, style="italic")

    handles = [
        Line2D([0], [0], color=C_GPS,   lw=4.0, label="GPS"),
        Line2D([0], [0], color=C_LIDAR, lw=3.2, label="LiDAR"),
        Line2D([0], [0], color=C_SONAR, lw=2.6, label="Sonar"),
        Line2D([0], [0], marker="o", color="#333", ls="None", ms=10, mec="white",
               mew=1.5, label="Start"),
        Line2D([0], [0], marker="v", color="#333", ls="None", ms=10, mec="white",
               mew=1.5, label="End"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=5, fontsize=FS - 4, framealpha=0.95, handlelength=2.0,
               columnspacing=1.4, borderpad=0.5)
    fig.tight_layout()
    _savefig(fig, out_dir / "trajectories_both.png")


# ---------------------------------------------------------------------------
# Side-by-side of pre-rendered satellite trajectory PNGs
# ---------------------------------------------------------------------------

# Line colours used in the pre-rendered *_path.png images.
C_PATH_GPS, C_PATH_LIDAR, C_PATH_SONAR = "#00AA00", "#2166AC", "#F4A100"

_DEFAULT_U_PATH_PNG      = _WS_ROOT / "debug" / "U" / "plots_drift" / "U_path.png"
_DEFAULT_ZIGZAG_PATH_PNG = _WS_ROOT / "debug" / "zigzag" / "plots" / "zigzag_path.png"


def plot_paths_both(test_a_png: Path, test_b_png: Path, out_dir: Path,
                    panel_h: float = 7.0) -> None:
    """Place two pre-rendered satellite trajectory PNGs side by side
    (Test A | Test B) with italic titles and a shared legend.

    Column widths are made proportional to each image's aspect ratio so the
    panels sit at the same height with no wasted space around them.
    """
    import matplotlib.image as mpimg
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    print("Composing side-by-side trajectory PNGs …")
    img_a = mpimg.imread(str(test_a_png))
    img_b = mpimg.imread(str(test_b_png))
    asp_a = img_a.shape[1] / img_a.shape[0]   # width / height
    asp_b = img_b.shape[1] / img_b.shape[0]

    fig = plt.figure(figsize=((asp_a + asp_b) * panel_h, panel_h))
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[asp_a, asp_b],
                            wspace=0.03, left=0.0, right=1.0, bottom=0.0, top=0.92)
    for col, img, title in [(0, img_a, "Test A"), (1, img_b, "Test B")]:
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.set_title(title, fontsize=FS, style="italic")
        ax.axis("off")

    handles = [
        Line2D([0], [0], color=C_PATH_GPS,   lw=4.0, label="GPS"),
        Line2D([0], [0], color=C_PATH_LIDAR, lw=3.2, label="LiDAR"),
        Line2D([0], [0], color=C_PATH_SONAR, lw=2.6, label="Sonar"),
        Line2D([0], [0], marker="o", color="#333", ls="None", ms=10, mec="white",
               mew=1.5, label="Start"),
        Line2D([0], [0], marker="v", color="#333", ls="None", ms=10, mec="white",
               mew=1.5, label="End"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=5, fontsize=FS - 4, framealpha=0.95, handlelength=2.0,
               columnspacing=1.4, borderpad=0.5)
    _savefig(fig, out_dir / "paths_both.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--u-data",      default=str(_DEFAULT_U_DATA))
    ap.add_argument("--zigzag-data", default=str(_DEFAULT_ZIGZAG_DATA))
    ap.add_argument("--u-path-png",      default=str(_DEFAULT_U_PATH_PNG))
    ap.add_argument("--zigzag-path-png", default=str(_DEFAULT_ZIGZAG_PATH_PNG))
    ap.add_argument("--out-dir",     default=str(_DEFAULT_OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = {
        "u": pd.read_csv(Path(args.u_data) / "nis.csv"),
        "z": pd.read_csv(Path(args.zigzag_data) / "nis.csv"),
    }
    print(f"Test B (U) NIS: {len(dfs['u'])} rows  |  Test A (zigzag) NIS: {len(dfs['z'])} rows")

    print("Generating plots …")
    plot_nis_both(dfs, out_dir)
    plot_inliers_both(dfs, out_dir)
    plot_trajectories_both(Path(args.u_data), Path(args.zigzag_data), out_dir)
    plot_paths_both(Path(args.zigzag_path_png), Path(args.u_path_png), out_dir)

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
