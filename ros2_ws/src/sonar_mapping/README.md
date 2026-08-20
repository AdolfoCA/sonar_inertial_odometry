# sonar_mapping

AKAZE **leading-edge** seabed mapping — the first step from sonar-inertial
odometry towards sonar SLAM.

For every forward-looking-sonar scan the node

1. pre-processes the polar image with the same chain as `sonar_odometry`
   (crop → log-compress → bilateral → CLAHE),
2. detects AKAZE keypoints and keeps, **per beam, the keypoint with the
   strongest response** — a *leading edge*: a pseudo-scan with a single range
   bin for each beam,
3. maps the selected polar cells to Cartesian points in the sonar frame
   (paper eq. 7),
4. applies the static sonar→body mounting transform (20° pitch, paper eq. 18)
   and the vehicle pose from `/odometry` (the `sonar_odometry` estimate),
5. publishes the scan's points as a `sensor_msgs/PointCloud2` in the map
   frame, and accumulates them into a full-run seabed map.

On shutdown (Ctrl-C after the bag ends) the accumulated cloud is saved as an
ASCII PLY (`save_path`, default `~/sonar_mapping_map.ply`) — open it in
CloudCompare/Open3D, or render it top-down with the trajectory.

## Run

```bash
ros2 launch sonar_mapping sonar_mapping.launch.py
# in parallel: the odometry and the bag
ros2 launch sonar_odometry sonar_odometry.launch.py
ros2 bag play zigzag.mcap
```

## Topics

| direction | topic                          | type                                      |
|-----------|--------------------------------|-------------------------------------------|
| in        | `/oculus/sonar_image`          | `marine_acoustic_msgs/ProjectedSonarImage` |
| in        | `/odometry`                    | `nav_msgs/Odometry`                        |
| out       | `/sonar_mapping/leading_edge`  | `sensor_msgs/PointCloud2` (per scan)       |
| out       | `/sonar_mapping/map`           | `sensor_msgs/PointCloud2` (accumulated)    |

Cloud fields: `x y z intensity` — map frame, `z` positive down, intensity is
the raw echo strength of the selected bin.

## Parameters

`crop_row` (200) · `sonar_pitch_deg` (20.0) · `akaze_threshold` (0.001) ·
`akaze_octaves` (8) · `akaze_octave_layers` (8) · `map_publish_every` (20) ·
`save_path` (`~/sonar_mapping_map.ply`) · `sonar_topic` · `odom_topic` ·
`cloud_topic` · `map_topic` · `map_frame`.

## Tests

The mapping core (`sonar_mapping/leading_edge_core.py`) is pure Python/NumPy/
OpenCV with no ROS dependency, and is covered by unit tests:

```bash
cd ros2_ws/src/sonar_mapping && python3 -m pytest test/ -v
```

Geometry (eq. 7 mapping, 20° pitch projection, pose composition), the
one-bin-per-beam guarantee, and empty-scan handling are tested synthetically;
one test additionally validates dense leading-edge coverage on a real
Skovshoved frame when the recorded data is present.

Validated offline against the full `zigzag.mcap` replay (3,494 scans →
1,013,807 map points): the accumulated cloud reproduces the seabed texture
bands of the marina with the zig-zag trajectory over them.
