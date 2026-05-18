"""
ROS2 node: Lidar SLAM path logger.

Subscribes to the nav_msgs/Odometry published by the lidar SLAM (Fast-LIO)
and writes each pose to a CSV file for offline plotting.

Subscriptions
-------------
  /odometry   nav_msgs/Odometry  – lidar SLAM odometry (configurable)

Parameters
----------
  lidar_odom_topic : string  (default: "/odometry")
  csv_data_dir     : string  (default: <ws_root>/debug/data)
"""

from __future__ import annotations

import csv
import os
import numpy as np

from ament_index_python.packages import get_package_share_directory as _get_share
_WS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    _get_share("sonar_odometry")
))))
_DEFAULT_DATA_DIR = os.path.join(_WS_ROOT, "debug", "data")

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry


class LidarPathLoggerNode(Node):

    def __init__(self):
        super().__init__("lidar_path_logger_node")

        self.declare_parameter("lidar_odom_topic", "/odometry")
        self.declare_parameter("csv_data_dir",     "")

        lidar_topic = self.get_parameter("lidar_odom_topic").value
        data_dir = self.get_parameter("csv_data_dir").value or _DEFAULT_DATA_DIR

        try:
            os.makedirs(data_dir, exist_ok=True)
            csv_path = os.path.join(data_dir, "lidar_slam.csv")
            self._csv_f = open(csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_f)
            self._csv_writer.writerow(["time_sec", "east_m", "north_m", "yaw_deg"])
            self.get_logger().info(f"Lidar SLAM CSV → {csv_path}")
        except Exception as e:
            self._csv_f = None
            self._csv_writer = None
            self.get_logger().error(f"Lidar CSV init failed: {e}")

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._sub = self.create_subscription(
            Odometry, lidar_topic, self._odom_callback, best_effort_qos
        )
        self.get_logger().info(
            f"Lidar path logger started — subscribing to '{lidar_topic}'."
        )

    def _odom_callback(self, msg: Odometry) -> None:
        if self._csv_writer is None:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = float(np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        ))
        self._csv_writer.writerow([
            f"{t:.6f}", f"{x:.6f}", f"{y:.6f}", f"{float(np.rad2deg(yaw)):.4f}"
        ])
        self._csv_f.flush()

    def destroy_node(self):
        if self._csv_f is not None:
            try:
                self._csv_f.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarPathLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
