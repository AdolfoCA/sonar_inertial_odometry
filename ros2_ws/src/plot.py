#!/usr/bin/env python3
"""
Combines the zigzag and U trajectory map HTML files into a single
overlay map where both sets of trajectories appear together.
"""

import re
from pathlib import Path
from bs4 import BeautifulSoup


# Trajectory colours on the combined map (GPS recoloured from black → green).
C_GPS, C_LIDAR, C_SONAR = "#00AA00", "#2166AC", "#F4A100"


def _legend_html() -> str:
    """Self-contained legend overlay: GPS / LiDAR / Sonar lines + Start ● / End ▼."""
    def line_row(color, name):
        return (f'<div style="display:flex;align-items:center;margin:6px 0">'
                f'<span style="display:inline-block;width:52px;height:8px;'
                f'background:{color};margin-right:14px;border:1px solid #555"></span>'
                f'{name}</div>')

    def marker_row(symbol, name):
        return (f'<div style="display:flex;align-items:center;margin:6px 0">'
                f'<span style="display:inline-block;width:52px;text-align:center;'
                f'margin-right:14px;font-size:30px;line-height:1;color:#333">{symbol}</span>'
                f'{name}</div>')

    rows = (line_row(C_GPS, "GPS")
            + line_row(C_LIDAR, "LiDAR")
            + line_row(C_SONAR, "Sonar")
            + marker_row("&#9679;", "Start")    # ●
            + marker_row("&#9660;", "End"))     # ▼
    return ('<div style="position:absolute;top:18px;right:18px;z-index:1000;'
            'background:#ffffff;padding:22px 36px;border-radius:6px;'
            'box-shadow:0 2px 8px rgba(0,0,0,.35);'
            'font-family:Arial,sans-serif;font-size:28px;color:#222">'
            + rows + '</div>')


def extract_parts(html_path):
    """Return (center, overlay_code, api_key) from a trajectory map HTML."""
    content = Path(html_path).read_text()
    soup = BeautifulSoup(content, "html.parser")

    init_script = None
    api_key = None

    for tag in soup.find_all("script"):
        src = tag.get("src", "")
        if src and "maps.googleapis.com" in src:
            m = re.search(r"key=([^&]+)", src)
            if m:
                api_key = m.group(1)
        elif tag.string and "initMap" in tag.string:
            init_script = tag.string

    center_m = re.search(r"center:\s*\{lat:([\d.]+),\s*lng:([\d.]+)\}", init_script)
    center = (float(center_m.group(1)), float(center_m.group(2))) if center_m else (0.0, 0.0)

    # All Polyline and Marker calls are single (very long) lines — extract them
    # directly and wrap each in __push(...) so the combined map can frame them.
    overlay_lines = []
    for line in init_script.split("\n"):
        s = line.strip()
        if s.startswith("new google.maps.Polyline(") or s.startswith("new google.maps.Marker("):
            s = s.rstrip(";")
            overlay_lines.append("  __push(" + s + ");")

    # GPS path/markers were drawn black in the source map → recolour to green.
    overlay_code = re.sub(r"#000000", C_GPS, "\n".join(overlay_lines))
    return center, overlay_code, api_key


def make_combined(zigzag_html, u_html, output_path):
    center1, overlay1, api_key = extract_parts(zigzag_html)
    center2, overlay2, _       = extract_parts(u_html)

    mid_lat = (center1[0] + center2[0]) / 2
    mid_lng = (center1[1] + center2[1]) / 2

    html = f"""\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Trajectory Map – Zigzag + U</title>
  <style>html, body, #map {{ width: 100%; height: 100%; margin: 0; padding: 0; }}</style>
</head>
<body>
  <div id="map"></div>
  {_legend_html()}

  <script>
function initMap() {{
  var map = new google.maps.Map(document.getElementById('map'), {{
    center: {{lat:{mid_lat:.7f}, lng:{mid_lng:.7f}}},
    zoom: 16,
    mapTypeId: 'roadmap'
  }});

  // Collect every polyline so we can frame all of them.
  var bounds = new google.maps.LatLngBounds();
  var __overlays = [];
  var __push = function(o) {{ __overlays.push(o); return o; }};

{overlay1}
{overlay2}

  __overlays.forEach(function(o) {{
    if (o.getPath) {{ o.getPath().forEach(function(p) {{ bounds.extend(p); }}); }}
    else if (o.getPosition) {{ bounds.extend(o.getPosition()); }}
  }});
  if (!bounds.isEmpty()) map.fitBounds(bounds);
}}
  </script>
  <script async defer
    src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap">
  </script>
</body>
</html>"""

    Path(output_path).write_text(html)
    print(f"Written: {output_path}")


if __name__ == "__main__":
    ws = Path(__file__).parent.parent
    zigzag = ws / "debug/zigzag/plots/trajectory_map.html"
    u      = ws / "debug/U/plots_drift/trajectory_map.html"
    out    = ws / "debug/combined_trajectory_map.html"

    make_combined(zigzag, u, out)
