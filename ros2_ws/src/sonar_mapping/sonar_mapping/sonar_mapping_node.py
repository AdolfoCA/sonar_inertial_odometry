#!/usr/bin/env python3
"""
ROS2 node: AKAZE leading-edge seabed mapping.

For every incoming sonar scan the node:
  1. pre-processes the polar image (same chain as sonar_odometry),
  2. detects AKAZE keypoints and keeps, for each beam, the keypoint with the
     strongest response — a "leading edge" of one range bin per beam,
  3. maps those polar cells to Cartesian points in the sonar frame (eq. 7),
  4. applies the static sonar->body transform (mounting pitch, default 20°
     down) and the latest vehicle pose from the odometry topic,
  5. publishes the scan's points as a sensor_msgs/PointCloud2 in the map
     frame and accumulates them.

On shutdown the accumulated cloud is written to an ASCII PLY file, so the
whole run's seabed map can be inspected offline (e.g. CloudCompare, Open3D)
or rendered top-down with the vehicle trajectory.

Topics
  in:  sonar_topic  (marine_acoustic_msgs/ProjectedSonarImage)  default /oculus/sonar_image
  in:  odom_topic   (nav_msgs/Odometry)                          default /odometry
  out: cloud_topic  (sensor_msgs/PointCloud2, per scan)          default /sonar_mapping/leading_edge
  out: map_topic    (sensor_msgs/PointCloud2, accumulated,       default /sonar_mapping/map
                     published every map_publish_every scans)

Parameters
  crop_row            int    range bins above this row are water column (default 200)
  sonar_pitch_deg     float  mounting pitch, positive down             (default 20.0)
  akaze_threshold     float  AKAZE response threshold                  (default 0.001)
  akaze_octaves       int                                              (default 8)
  akaze_octave_layers int                                              (default 8)
  map_publish_every   int    accumulate N scans between map publishes  (default 20)
  save_path           str    PLY written on shutdown (default ~/sonar_mapping_map.ply)

Fields of the published clouds: x, y, z (map frame, z down along the vehicle
D axis) and intensity (raw echo strength of the selected bin).
"""
import os
import struct

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from marine_acoustic_msgs.msg import ProjectedSonarImage

from sonar_odometry.image_processing import SonarImageProcessor
from sonar_mapping.leading_edge_core import (
    make_detector, scan_to_map_points,
)


def quat_to_yaw(q):
    """Yaw about +z from a geometry_msgs Quaternion."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny, cosy))


def cloud_msg(points_xyzi, stamp, frame_id):
    """Build a PointCloud2 (x, y, z, intensity float32) from an (N,4) array."""
    msg = PointCloud2()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height = 1
    msg.width = len(points_xyzi)
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * len(points_xyzi)
    msg.is_dense = True
    msg.data = np.ascontiguousarray(points_xyzi, dtype=np.float32).tobytes()
    return msg


class SonarMappingNode(Node):

    def __init__(self):
        super().__init__('sonar_mapping_node')

        self.declare_parameter('sonar_topic', '/oculus/sonar_image')
        self.declare_parameter('odom_topic', '/odometry')
        self.declare_parameter('cloud_topic', '/sonar_mapping/leading_edge')
        self.declare_parameter('map_topic', '/sonar_mapping/map')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('crop_row', 200)
        self.declare_parameter('sonar_pitch_deg', 20.0)
        self.declare_parameter('akaze_threshold', 0.001)
        self.declare_parameter('akaze_octaves', 8)
        self.declare_parameter('akaze_octave_layers', 8)
        self.declare_parameter('map_publish_every', 20)
        self.declare_parameter('save_path', os.path.expanduser('~/sonar_mapping_map.ply'))

        g = lambda n: self.get_parameter(n).value
        self.crop_row = int(g('crop_row'))
        self.pitch = float(np.deg2rad(g('sonar_pitch_deg')))
        self.map_frame = g('map_frame')
        self.map_publish_every = int(g('map_publish_every'))
        self.save_path = os.path.expanduser(g('save_path'))

        self.processor = SonarImageProcessor()
        self.processor.config['crop_row'] = self.crop_row
        self.detector = make_detector(
            threshold=float(g('akaze_threshold')),
            n_octaves=int(g('akaze_octaves')),
            n_octave_layers=int(g('akaze_octave_layers')),
        )

        self._pose = None            # (north, east, yaw)
        self._chunks = []            # accumulated (N,4) arrays
        self._n_scans = 0

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(ProjectedSonarImage, g('sonar_topic'),
                                 self.sonar_cb, qos)
        self.create_subscription(Odometry, g('odom_topic'), self.odom_cb, 20)
        self.pub_scan = self.create_publisher(PointCloud2, g('cloud_topic'), 5)
        self.pub_map = self.create_publisher(PointCloud2, g('map_topic'), 1)

        self.get_logger().info('Sonar mapping node started (AKAZE leading edge).')

    # ------------------------------------------------------------ callbacks
    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self._pose = (p.x, p.y, quat_to_yaw(msg.pose.pose.orientation))

    def sonar_cb(self, msg: ProjectedSonarImage):
        if self._pose is None:
            return                                   # no pose yet
        n_beams = len(msg.beam_directions)
        img = np.frombuffer(bytes(msg.image.data), dtype=np.uint8)
        img = img.reshape(-1, n_beams)
        # marine_acoustic_msgs convention: Z forward, beams in the Y-Z plane
        az = np.array([np.arctan2(d.y, d.z) for d in msg.beam_directions],
                      dtype=np.float32)
        rng = np.asarray(msg.ranges, dtype=np.float32)

        processed = self.processor.process_image(img)
        raw_crop = img[self.crop_row:, :]
        north, east, yaw = self._pose
        pts = scan_to_map_points(processed, raw_crop, self.detector, az, rng,
                                 self.crop_row, self.pitch, north, east, yaw)
        self._n_scans += 1
        if len(pts) == 0:
            return
        self._chunks.append(pts.astype(np.float32))
        self.pub_scan.publish(cloud_msg(pts, msg.header.stamp, self.map_frame))
        if self.map_publish_every > 0 and self._n_scans % self.map_publish_every == 0:
            full = np.vstack(self._chunks)
            self.pub_map.publish(cloud_msg(full, msg.header.stamp, self.map_frame))
            self.get_logger().info(
                f'map: {len(full):,} points from {self._n_scans} scans')

    # ------------------------------------------------------------ shutdown
    def save_map(self):
        if not self._chunks:
            self.get_logger().warning('no points accumulated — nothing to save')
            return
        pts = np.vstack(self._chunks)
        with open(self.save_path, 'w') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write('comment AKAZE leading-edge seabed map (sonar_mapping)\n')
            f.write(f'element vertex {len(pts)}\n')
            f.write('property float x\nproperty float y\nproperty float z\n')
            f.write('property float intensity\nend_header\n')
            np.savetxt(f, pts, fmt='%.3f %.3f %.3f %.0f')
        self.get_logger().info(f'saved {len(pts):,} points to {self.save_path}')


def main(args=None):
    rclpy.init(args=args)
    node = SonarMappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_map()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
