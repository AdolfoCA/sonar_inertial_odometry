"""
ROS2 node: Sonar Inertial Odometry with EKF.

Subscriptions
-------------
  /imu/data           sensor_msgs/Imu                  – IMU at ~100 Hz
  /oculus/sonar_image marine_acoustic_msgs/ProjectedSonarImage

Publications
------------
  /odometry           nav_msgs/Odometry – fused position, heading, velocity

Frame conventions:
  - Local odom frame: X = vehicle-forward at startup, Y = vehicle-left at startup (matches Fast-LIO)
  - FLU (Forward-Left-Up) body frame (base_link)
  - nav_msgs/Odometry:
      frame_id       = "odom"       (local odom, aligned with base_link at t=0)
      child_frame_id = "base_link"  (FLU body)
      pose.position.x = forward from origin (m)
      pose.position.y = left from origin (m)
      orientation     = quaternion from yaw (CCW positive, 0 = vehicle-forward)

IMU pipeline:
  Raw IMU in sensor frame → rotated to FLU body via TF(imu_frame→base_link).
  u = [ax_flu, ay_flu, gz_flu]: forward accel, leftward accel, CCW yaw rate.

Sonar pipeline:
  Feature-matcher output → depression correction (TF-derived cos_phi) →
  lever-arm correction → FLU body frame → EKF update.
  z = [d_forward, d_left, d_theta] — all in FLU, at body/IMU origin.

Parameters (all settable via ROS params or a YAML config file)
----------
  imu_topic            : string  (default: "/imu/data")
  sonar_topic          : string  (default: "/oculus/sonar_image")
  odom_topic           : string  (default: "/odometry")
  initial_heading_deg  : float   (default: 0.0)
  accel_bias_x         : float   (default: 0.0)
  accel_bias_y         : float   (default: 0.0)
  gyro_bias_z          : float   (default: 0.002)
  nis_threshold        : float   (default: 30.0)
  min_inliers          : int     (default: 6)
  min_inlier_ratio     : float   (default: 0.3)
  publish_tf           : bool    (default: false)
"""

from __future__ import annotations

import csv
import os
import threading

# Default debug output dir: ros2_ws/debug
# ament_index gives us install/<pkg>/share/<pkg>; four levels up is the workspace root.
from ament_index_python.packages import get_package_share_directory as _get_share
_WS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    _get_share("sonar_odometry")
))))
_DEFAULT_DEBUG_DIR = os.path.join(_WS_ROOT, "debug")
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TransformStamped, PoseStamped
from builtin_interfaces.msg import Time
from std_msgs.msg import Float32, Int32
from marine_acoustic_msgs.msg import ProjectedSonarImage

try:
    from tf2_ros import TransformBroadcaster, Buffer, TransformListener
    TF2_AVAILABLE = True
except ImportError:
    TF2_AVAILABLE = False

from .ekf import ESEKF
from .image_processing import SonarImageProcessor
from .feature_matching import SonarFeatureMatcher


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a quaternion to a 3×3 rotation matrix."""
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-10:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s*qw*qx, s*qw*qy, s*qw*qz
    xx, xy, xz = s*qx*qx, s*qx*qy, s*qx*qz
    yy, yz, zz = s*qy*qy, s*qy*qz, s*qz*qz
    return np.array([
        [1.0-(yy+zz),    xy-wz,       xz+wy    ],
        [   xy+wz,    1.0-(xx+zz),    yz-wx    ],
        [   xz-wy,       yz+wx,    1.0-(xx+yy)],
    ])


def heading_to_quaternion(theta: float) -> tuple[float, float, float, float]:
    """Convert yaw angle (rad, CCW positive) to quaternion (x, y, z, w). theta=0 = vehicle-forward."""
    half = theta / 2.0
    return 0.0, 0.0, np.sin(half), np.cos(half)


def ros_time_to_sec(stamp: Time) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def projected_sonar_to_cv2(msg: ProjectedSonarImage) -> np.ndarray | None:
    """Convert marine_acoustic_msgs/ProjectedSonarImage to a uint8 grayscale OpenCV image."""
    n_ranges = len(msg.ranges)
    n_beams  = msg.image.beam_count
    if n_ranges == 0 or n_beams == 0:
        return None

    dtype_map = {
        0: np.uint8,  1: np.int8,
        2: np.uint16, 3: np.int16,
        4: np.uint32, 5: np.int32,
        6: np.uint64, 7: np.int64,
        8: np.float32, 9: np.float64,
    }
    dtype = dtype_map.get(msg.image.dtype, np.uint8)
    data = np.frombuffer(bytes(msg.image.data), dtype=dtype).reshape(n_ranges, n_beams)

    # Normalise to uint8
    if dtype != np.uint8:
        d_min, d_max = data.min(), data.max()
        if d_max > d_min:
            data = ((data - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            data = np.zeros((n_ranges, n_beams), dtype=np.uint8)

    return data


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class SonarOdometryNode(Node):

    def __init__(self):
        super().__init__("sonar_odometry_node")
        self._declare_parameters()
        self._load_parameters()
        self._init_ekf()
        self._init_processing()
        self._create_interfaces()
        self._init_csv_logging()
        self.get_logger().info("Sonar Odometry Node started.")

    # ---------------------------------------------------------------- params
    def _declare_parameters(self):
        self.declare_parameter("imu_topic",            "/imu/data")
        self.declare_parameter("sonar_topic",          "/oculus/sonar_image")
        self.declare_parameter("odom_topic",           "/odometry")
        self.declare_parameter("odom_frame_id",        "odom")
        self.declare_parameter("base_frame_id",        "base_link")
        self.declare_parameter("initial_heading_deg",  0.0)
        self.declare_parameter("accel_bias_x",         0.0)
        self.declare_parameter("accel_bias_y",         0.0)
        self.declare_parameter("gyro_bias_z",          0.002)
        self.declare_parameter("nis_threshold",        30.0)
        self.declare_parameter("lowe_ratio",               0.75)
        self.declare_parameter("use_pixel_ransac",         True)
        self.declare_parameter("pixel_ransac_threshold",   3.0)
        self.declare_parameter("distance_threshold",       0.5)
        self.declare_parameter("sonar_stride",            2)
        self.declare_parameter("min_inliers",             6)
        self.declare_parameter("min_inlier_ratio",        0.3)
        self.declare_parameter("publish_tf",           False)
        self.declare_parameter("imu_frame_id",         "imu_link")
        self.declare_parameter("sonar_frame_id",       "sonar_link")
        self.declare_parameter("debug_image_dir",      _DEFAULT_DEBUG_DIR)
        self.declare_parameter("debug",                False)
        self.declare_parameter("debug_match_interval", 10)
        self.declare_parameter("csv_data_dir",         "")
        # ── Q: IMU white-noise variances ──────────────────────────────────────
        self.declare_parameter("acc_covariance",    1.754e-4)  # (m/s²)²
        self.declare_parameter("gyro_covariance",   7.615e-7)  # (rad/s)²
        # ── Q: bias / scale random-walk variances ─────────────────────────
        self.declare_parameter("b_acc_covariance",  0.25)      # (m/s²)²
        self.declare_parameter("b_gyro_covariance", 0.25)      # (rad/s)²
        self.declare_parameter("scale_covariance",  1.0e-6)    # dimensionless²
        # ── R: sonar measurement noise variances ──────────────────────────
        self.declare_parameter("sonar_position_covariance", 5.506e-4)  # m²
        self.declare_parameter("sonar_yaw_covariance",      5.506e-4)  # rad²
        self.declare_parameter("crop_row",                 300)
        self.declare_parameter("bilateral_d",              9)
        self.declare_parameter("bilateral_sigma_color",    50.0)
        self.declare_parameter("bilateral_sigma_space",    50.0)
        self.declare_parameter("clahe_clip_limit",         2.0)
        self.declare_parameter("clahe_tile_grid_size",     8)

    def _load_parameters(self):
        gp = self.get_parameter
        self.imu_topic       = gp("imu_topic").value
        self.sonar_topic     = gp("sonar_topic").value
        self.odom_topic      = gp("odom_topic").value
        self.odom_frame_id   = gp("odom_frame_id").value
        self.base_frame_id   = gp("base_frame_id").value
        # theta=0 = vehicle-forward = odom +X (matches Fast-LIO local odom convention)
        self.initial_heading = np.deg2rad(gp("initial_heading_deg").value)
        self.accel_bias      = (gp("accel_bias_x").value, gp("accel_bias_y").value, 0.0)
        self.gyro_bias_z     = gp("gyro_bias_z").value
        self.nis_threshold   = gp("nis_threshold").value
        self.lowe_ratio               = gp("lowe_ratio").value
        self.use_pixel_ransac         = gp("use_pixel_ransac").value
        self.pixel_ransac_threshold   = gp("pixel_ransac_threshold").value
        self.distance_threshold       = gp("distance_threshold").value
        self.sonar_stride            = max(1, int(gp("sonar_stride").value))
        self.min_inliers             = gp("min_inliers").value
        self.min_inlier_ratio        = gp("min_inlier_ratio").value
        self.publish_tf_flag = gp("publish_tf").value
        self.imu_frame_id      = gp("imu_frame_id").value
        self.sonar_frame_id    = gp("sonar_frame_id").value
        self.debug_image_dir      = gp("debug_image_dir").value
        self.debug_enabled        = gp("debug").value
        self.debug_match_interval = gp("debug_match_interval").value
        self.csv_data_dir         = gp("csv_data_dir").value
        self.acc_cov       = gp("acc_covariance").value
        self.gyro_cov      = gp("gyro_covariance").value
        self.b_acc_cov     = gp("b_acc_covariance").value
        self.b_gyro_cov    = gp("b_gyro_covariance").value
        self.scale_cov     = gp("scale_covariance").value
        self.sonar_pos_cov = gp("sonar_position_covariance").value
        self.sonar_yaw_cov = gp("sonar_yaw_covariance").value
        tile = gp("clahe_tile_grid_size").value
        self._img_proc_config = {
            "crop_row":             gp("crop_row").value,
            "bilateral_d":          gp("bilateral_d").value,
            "bilateral_sigma_color":gp("bilateral_sigma_color").value,
            "bilateral_sigma_space":gp("bilateral_sigma_space").value,
            "clahe_clip_limit":     gp("clahe_clip_limit").value,
            "clahe_tile_grid":      (tile, tile),
        }

    # ----------------------------------------------------------------- init
    def _init_ekf(self):
        self.ekf = ESEKF(
            initial_heading_rad=self.initial_heading,
            accel_bias=self.accel_bias,
            gyro_bias_z=self.gyro_bias_z,
            acc_covariance=self.acc_cov,
            gyro_covariance=self.gyro_cov,
            b_acc_covariance=self.b_acc_cov,
            b_gyro_covariance=self.b_gyro_cov,
            scale_covariance=self.scale_cov,
            sonar_position_covariance=self.sonar_pos_cov,
            sonar_yaw_covariance=self.sonar_yaw_cov,
        )
        self._lock = threading.Lock()
        self._prev_imu_time: float | None = None
        self._is_initialized: bool = False       # becomes True after first IMU
        self._sonar_count: int = 0
        self._sonar_raw_count: int = 0   # increments every sonar frame (for stride)
        # Cached rotation matrices and translations from TF (looked up once on first use)
        self._R_imu_to_body:   np.ndarray | None = None
        self._R_sonar_to_body: np.ndarray | None = None
        self._t_sonar_to_body: np.ndarray | None = None  # sonar origin in FLU body (base_link) frame

        # Sonar-only dead-reckoning (accumulated body-frame displacements)
        self._dbg_sonar_px:    float = 0.0
        self._dbg_sonar_py:    float = 0.0
        self._dbg_sonar_theta: float = self.initial_heading

        # IMU-filter dead-reckoning (pure IMU integration, never corrected by sonar)
        self._imu_dr_px:    float = 0.0
        self._imu_dr_py:    float = 0.0
        self._imu_dr_vx:    float = 0.0
        self._imu_dr_vy:    float = 0.0
        self._imu_dr_theta: float = self.initial_heading

        # NIS windowed average
        self._nis_window: list[float] = []
        self._nis_window_size: int = 50

        # CSV log handles (set by _init_csv_logging)
        self._csv_sonar_writer = None
        self._csv_sonar_f      = None
        self._csv_nis_writer   = None
        self._csv_nis_f        = None

        # Debug match image state
        self._sonar_scan_count: int = 0
        self._prev_sonar_img_u8: np.ndarray | None = None  # uint8 version of previous processed image

    def _init_csv_logging(self):
        data_dir = self.csv_data_dir if self.csv_data_dir else os.path.join(self.debug_image_dir, "data")
        try:
            os.makedirs(data_dir, exist_ok=True)
            sonar_path = os.path.join(data_dir, "sonar_odometry.csv")
            self._csv_sonar_f = open(sonar_path, "w", newline="")
            self._csv_sonar_writer = csv.writer(self._csv_sonar_f)
            self._csv_sonar_writer.writerow(["time_sec", "east_m", "north_m", "yaw_deg"])

            nis_path = os.path.join(data_dir, "nis.csv")
            self._csv_nis_f = open(nis_path, "w", newline="")
            self._csv_nis_writer = csv.writer(self._csv_nis_f)
            self._csv_nis_writer.writerow(["time_sec", "nis", "accepted", "inlier_count", "inlier_ratio"])

            self.get_logger().info(f"CSV data logging → {data_dir}")
        except Exception as e:
            self.get_logger().error(f"CSV logging init failed: {e}")

    def _init_processing(self):
        self._processor = SonarImageProcessor()
        self._processor.config.update(self._img_proc_config)
        self._matcher   = SonarFeatureMatcher(
            lowe_ratio=self.lowe_ratio,
            use_pixel_ransac=self.use_pixel_ransac,
            pixel_ransac_threshold=self.pixel_ransac_threshold,
            distance_threshold=self.distance_threshold,
            crop_row=self._img_proc_config["crop_row"],
        )
        self._prev_sonar_img: np.ndarray | None = None
        self._prev_sonar_stamp: float | None = None

    def _create_interfaces(self):
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._imu_sub = self.create_subscription(
            Imu, self.imu_topic, self._imu_callback, best_effort_qos
        )
        self._sonar_sub = self.create_subscription(
            ProjectedSonarImage, self.sonar_topic, self._sonar_callback, reliable_qos
        )
        self._odom_pub = self.create_publisher(Odometry, self.odom_topic, reliable_qos)

        self._path_pub = self.create_publisher(Path, self.odom_topic + "/path", reliable_qos)
        self._path_msg = Path()
        self._path_msg.header.frame_id = self.odom_frame_id

        self._dbg_sonar_dr_path_pub = self.create_publisher(Path, "/debug/sonar_dr/path", reliable_qos)
        self._dbg_sonar_dr_path = Path()
        self._dbg_sonar_dr_path.header.frame_id = self.odom_frame_id

        self._imu_filter_path_pub = self.create_publisher(Path, "/debug/imu_filter/path", reliable_qos)
        self._imu_filter_path = Path()
        self._imu_filter_path.header.frame_id = self.odom_frame_id

        self._nis_pub          = self.create_publisher(Float32, "/debug/nis", reliable_qos)
        self._nis_avg_pub      = self.create_publisher(Float32, "/debug/nis/average", reliable_qos)
        self._inliers_pub      = self.create_publisher(Int32,   "/debug/inliers/count", reliable_qos)
        self._inlier_ratio_pub = self.create_publisher(Float32, "/debug/inliers/ratio", reliable_qos)

        self.create_timer(0.2, self._publish_sonar_dr_path)  # 5 Hz
        self.create_timer(0.2, self._publish_imu_filter_path)  # 5 Hz

        if TF2_AVAILABLE:
            self._tf_buffer   = Buffer(cache_time=Duration(seconds=3600))
            self._tf_listener = TransformListener(self._tf_buffer, self)
        else:
            self._tf_buffer   = None
            self._tf_listener = None
            self.get_logger().error("tf2_ros not available — frame transforms will not work.")

        if self.publish_tf_flag and TF2_AVAILABLE:
            self._tf_broadcaster = TransformBroadcaster(self)
        else:
            self._tf_broadcaster = None

    # -------------------------------------------------- TF rotation lookup
    def _lookup_transform(self, source_frame: str, target_frame: str
                          ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Look up the static TF from source_frame to target_frame.
        Returns (R, t) where R is the 3×3 rotation matrix and t is the
        3-vector translation, both expressing source origin in target frame.
        Returns None if the transform is not yet available.
        """
        if self._tf_buffer is None:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time()
            )
            q = tf.transform.rotation
            R = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
            tr = tf.transform.translation
            t = np.array([tr.x, tr.y, tr.z])
            self.get_logger().info(
                f"TF: cached {source_frame} → {target_frame}  t={t}  R=\n{R}"
            )
            return R, t
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup {source_frame} → {target_frame} failed: {e}",
                throttle_duration_sec=5.0,
            )
            return None

    # ---------------------------------------------------------- IMU callback
    def _imu_callback(self, msg: Imu) -> None:
        stamp = ros_time_to_sec(msg.header.stamp)

        with self._lock:
            # Lazy TF lookup — cached after first success
            if self._R_imu_to_body is None:
                result = self._lookup_transform(self.imu_frame_id, self.base_frame_id)
                if result is None:
                    return  # TF not ready yet
                self._R_imu_to_body, _ = result

            if self._prev_imu_time is None:
                self._prev_imu_time = stamp
                self._is_initialized = True
                self.get_logger().info("IMU: first message received — EKF initialized.")
                return

            dt = stamp - self._prev_imu_time
            if dt <= 0.0:
                self.get_logger().warn(
                    f"IMU: non-positive dt={dt:.4f}s, skipping.", throttle_duration_sec=5.0
                )
                self._prev_imu_time = stamp
                return

            # Remove gravity using the Madgwick orientation estimate.
            # R rotates sensor frame → world frame; gravity appears as -Z in world.
            q = msg.orientation
            if q.x == 0.0 and q.y == 0.0 and q.z == 0.0 and q.w == 0.0:
                self._prev_imu_time = stamp
                return
            R_sensor_to_world = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
            a_imu_raw = np.array([msg.linear_acceleration.x,
                                   msg.linear_acceleration.y,
                                   msg.linear_acceleration.z])
            a_world = R_sensor_to_world @ a_imu_raw
            a_world[2] -= 9.81  # remove gravity (ENU world: +Z up, FLU sensor at rest has +9.81 in Z)
            a_imu = R_sensor_to_world.T @ a_world  # back to sensor frame, gravity-free

            w_imu = np.array([msg.angular_velocity.x,
                               msg.angular_velocity.y,
                               msg.angular_velocity.z])
            a_body = self._R_imu_to_body @ a_imu
            w_body = self._R_imu_to_body @ w_imu

            # FLU convention: [forward_accel, left_accel, CCW_yaw_rate]
            # gz_flu (CCW+) = ω_z_enu — same sign, no negation needed.
            u = np.array([a_body[0], a_body[1], w_body[2]])
            self.ekf.prediction(u, dt)

            # IMU-filter dead-reckoning: same corrected inputs as EKF, never sonar-updated.
            ax_c = (u[0] - self.ekf.b_ax) / self.ekf.s_ax
            ay_c = (u[1] - self.ekf.b_ay) / self.ekf.s_ay
            gz_c = u[2] - self.ekf.b_gz
            c_dr = np.cos(self._imu_dr_theta)
            s_dr = np.sin(self._imu_dr_theta)
            ax_enu = c_dr * ax_c - s_dr * ay_c - gz_c * self._imu_dr_vy
            ay_enu = s_dr * ax_c + c_dr * ay_c + gz_c * self._imu_dr_vx
            self._imu_dr_px    += self._imu_dr_vx * dt + 0.5 * ax_enu * dt**2
            self._imu_dr_py    += self._imu_dr_vy * dt + 0.5 * ay_enu * dt**2
            self._imu_dr_vx    += ax_enu * dt
            self._imu_dr_vy    += ay_enu * dt
            self._imu_dr_theta += gz_c * dt
            self._imu_dr_theta  = self.ekf._wrap_angle(self._imu_dr_theta)

            now = self.get_clock().now().to_msg()
            ps = PoseStamped()
            ps.header.stamp    = now
            ps.header.frame_id = self.odom_frame_id
            ps.pose.position.x = self._imu_dr_px
            ps.pose.position.y = self._imu_dr_py
            qx, qy, qz, qw = heading_to_quaternion(self._imu_dr_theta)
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            self._imu_filter_path.poses.append(ps)

            self._prev_imu_time = stamp

        self._publish_odometry(msg.header.stamp, update_path=False)

    # -------------------------------------------------------- sonar callback
    def _sonar_callback(self, msg: ProjectedSonarImage) -> None:
        t0 = time.time()

        # Populate sonar geometry from this message (no-op after first call)
        self._matcher.update_sonar_params(msg)

        img = projected_sonar_to_cv2(msg)
        if img is None:
            self.get_logger().warn(
                "Sonar: failed to decode ProjectedSonarImage.", throttle_duration_sec=5.0
            )
            return

        stamp = ros_time_to_sec(msg.header.stamp)

        img_proc = self._processor.process_image(img)
        t_preproc = time.time()

        with self._lock:
            self._sonar_raw_count += 1

            # Stride: keep the previous reference image but skip matching for
            # intermediate frames.  This multiplies the inter-frame displacement
            # by `sonar_stride`, dramatically improving forward-estimate SNR.
            # (stride=1 = every frame, stride=2 = every other frame, etc.)
            if self.sonar_stride > 1 and (self._sonar_raw_count % self.sonar_stride != 0):
                # Store first frame as reference; otherwise just return.
                if self._prev_sonar_img is None:
                    self._prev_sonar_img   = img_proc
                    self._prev_sonar_stamp = stamp
                return

        with self._lock:
            if not self._is_initialized:
                self.get_logger().warn(
                    "Sonar: waiting for first IMU message before processing.", throttle_duration_sec=5.0
                )
                self._prev_sonar_img   = img_proc
                self._prev_sonar_stamp = stamp
                return

            if self._prev_sonar_img is None:
                self._prev_sonar_img   = img_proc
                self._prev_sonar_stamp = stamp
                self.ekf.prev_px    = self.ekf.px
                self.ekf.prev_py    = self.ekf.py
                self.ekf.prev_theta = self.ekf.theta
                self.get_logger().info("Sonar: first frame stored — ready for matching.")
                return

            result, kp1, kp2, matches = self._matcher.process_sonar_image_pair(
                self._prev_sonar_img, img_proc
            )
            t_match = time.time()

            self.get_logger().info(
                f"Sonar timing — preproc: {(t_preproc-t0)*1000:.1f}ms  "
                f"matching: {(t_match-t_preproc)*1000:.1f}ms  "
                f"total: {(t_match-t0)*1000:.1f}ms",
                throttle_duration_sec=2.0
            )
            self.get_logger().info(
                f"Sonar features — kp1={len(kp1)}  kp2={len(kp2)}  "
                f"matches={len(matches)}  inliers={result['num_inliers']}",
                throttle_duration_sec=2.0
            )

            img_proc_u8 = (img_proc * 255).astype(np.uint8) if img_proc.dtype != np.uint8 else img_proc

            # Save match debug image and raw sonar image every debug_match_interval scans
            if self.debug_enabled and (self._sonar_scan_count % self.debug_match_interval == 0):
                if self._prev_sonar_img_u8 is not None and result['inliers'] is not None:
                    try:
                        os.makedirs(self.debug_image_dir, exist_ok=True)
                        inlier_matches = [m for m, ok in zip(matches, result['inliers']) if ok]
                        vis = cv2.drawMatches(
                            self._prev_sonar_img_u8, kp1,
                            img_proc_u8, kp2,
                            inlier_matches, None,
                            matchColor=(0, 255, 0),
                            singlePointColor=(255, 0, 0),
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                        )
                        a = self._sonar_scan_count
                        b = self._sonar_scan_count + 1
                        fname = os.path.join(self.debug_image_dir, f"debug_sonar_{a}_{b}.png")
                        cv2.imwrite(fname, vis)
                        self.get_logger().info(f"Debug: saved match image {fname}")
                    except Exception as e:
                        self.get_logger().warn(f"Debug: could not save match image: {e}")
                try:
                    raw_dir = os.path.join(self.debug_image_dir, "raw")
                    os.makedirs(raw_dir, exist_ok=True)
                    raw_fname = os.path.join(raw_dir, f"raw_sonar_{self._sonar_scan_count}.png")
                    cv2.imwrite(raw_fname, img)
                    self.get_logger().info(f"Debug: saved raw sonar image {raw_fname}")
                except Exception as e:
                    self.get_logger().warn(f"Debug: could not save raw sonar image: {e}")

            self._prev_sonar_img_u8 = img_proc_u8
            self._sonar_scan_count += 1

            self._prev_sonar_img   = img_proc
            self._prev_sonar_stamp = stamp

            self._inliers_pub.publish(Int32(data=int(result['num_inliers'])))
            self._inlier_ratio_pub.publish(Float32(data=float(result['inlier_ratio'])))

            z_sonar = self._sonar_result_to_measurement(result)
            if z_sonar is None:
                self.get_logger().warn(
                    f"Sonar: rejected (inliers={result['num_inliers']}, "
                    f"ratio={result['inlier_ratio']:.2f}) — no EKF update.",
                    throttle_duration_sec=2.0
                )
                # Advance the EKF baseline so the next sonar measurement (always
                # frame-to-frame) is compared against the correct prediction window.
                self.ekf.prev_px    = self.ekf.px
                self.ekf.prev_py    = self.ekf.py
                self.ekf.prev_theta = self.ekf.theta
                return

            self.get_logger().info(
                f"Sonar update #{self._sonar_count}: "
                f"d_fwd={z_sonar[0]:.3f}m  d_left={z_sonar[1]:.3f}m  "
                f"dth={np.rad2deg(z_sonar[2]):.2f}°"
            )

            # Sonar dead-reckoning: body FLU → ENU.
            # z_sonar[0]=forward, z_sonar[1]=left, z_sonar[2]=CCW yaw.
            # ENU: East = cos(θ)*fwd - sin(θ)*left,  North = sin(θ)*fwd + cos(θ)*left
            c = np.cos(self._dbg_sonar_theta)
            s_th = np.sin(self._dbg_sonar_theta)
            self._dbg_sonar_px    += c * z_sonar[0] - s_th * z_sonar[1]
            self._dbg_sonar_py    += s_th * z_sonar[0] + c * z_sonar[1]
            self._dbg_sonar_theta += z_sonar[2]

            ps_s = PoseStamped()
            ps_s.header.stamp    = self.get_clock().now().to_msg()
            ps_s.header.frame_id = self.odom_frame_id
            ps_s.pose.position.x = self._dbg_sonar_px
            ps_s.pose.position.y = self._dbg_sonar_py
            qx, qy, qz, qw = heading_to_quaternion(self._dbg_sonar_theta)
            ps_s.pose.orientation.x = qx
            ps_s.pose.orientation.y = qy
            ps_s.pose.orientation.z = qz
            ps_s.pose.orientation.w = qw
            self._dbg_sonar_dr_path.poses.append(ps_s)

            nis, innovation, S, accepted = self.ekf.update(
                z_sonar, nis_threshold=self.nis_threshold
            )
            self._sonar_count += 1

            if not np.isnan(nis):
                tag = "ACCEPT" if accepted else "REJECT"
                self.get_logger().info(
                    f"EKF #{self._sonar_count} {tag}  NIS={nis:.2f}  "
                    f"pos=({self.ekf.px:.3f}, {self.ekf.py:.3f})m  "
                    f"hdg={np.rad2deg(self.ekf.theta):.1f}°"
                )
                self._nis_pub.publish(Float32(data=float(nis)))
                self._nis_window.append(float(nis))
                if len(self._nis_window) >= self._nis_window_size:
                    avg = float(np.mean(self._nis_window))
                    self._nis_avg_pub.publish(Float32(data=avg))
                    self.get_logger().info(f"NIS avg over {self._nis_window_size} updates: {avg:.2f}")
                    self._nis_window.clear()

                if self._csv_nis_writer is not None:
                    t = ros_time_to_sec(msg.header.stamp)
                    self._csv_nis_writer.writerow([
                        f"{t:.6f}", f"{float(nis):.6f}", int(accepted),
                        result["num_inliers"], f"{float(result['inlier_ratio']):.4f}"
                    ])
                    self._csv_nis_f.flush()

        self._publish_odometry(msg.header.stamp, update_path=True)

    # -------------------------------------------------- measurement builder
    def _sonar_result_to_measurement(self, result: dict) -> np.ndarray | None:
        T = result["transformation"]
        if T is None and result.get("transformation_pixel") is None:
            return None
        if result["num_inliers"] < self.min_inliers:
            return None
        if result["inlier_ratio"] < self.min_inlier_ratio:
            return None

        # Lazy TF lookup — cached after first success.
        # NOTE: use a separate variable to avoid shadowing the input `result` dict.
        if self._R_sonar_to_body is None:
            tf_result = self._lookup_transform(self.sonar_frame_id, self.base_frame_id)
            if tf_result is None:
                return None
            self._R_sonar_to_body, self._t_sonar_to_body = tf_result

            # --- One-time diagnostic log after TF is cached -------------------
            boresight_body = self._R_sonar_to_body[:, 2]   # +Z_sonar in body FLU
            cos_phi_diag   = float(boresight_body[0])       # forward projection
            depression_deg = float(np.rad2deg(np.arccos(np.clip(abs(cos_phi_diag), 0.0, 1.0))))
            r_fwd_diag  = float(self._t_sonar_to_body[0])
            r_left_diag = float(self._t_sonar_to_body[1])
            lever_at_10deg = np.deg2rad(10.0) * r_fwd_diag
            self.get_logger().info(
                f"\n=== Sonar TF cached ===\n"
                f"  boresight in body FLU : {boresight_body}\n"
                f"  cos_phi (fwd proj)    : {cos_phi_diag:.4f}\n"
                f"  depression angle      : {depression_deg:.1f} deg\n"
                f"  t_sonar_to_body (FLU) : {self._t_sonar_to_body}\n"
                f"  lever-arm at 10° CCW yaw: d_left_correction = {lever_at_10deg:.4f} m\n"
                f"Sign conventions (all should be positive for the named motion):\n"
                f"  forward motion  → d_fwd  > 0  (dp_image[0] / cos_phi)\n"
                f"  leftward motion → d_left > 0  (-dp_image[1])\n"
                f"  CCW yaw         → dtheta > 0  (dtheta_image / cos_phi)"
            )

        # --- Depression correction -------------------------------------------
        # cos_phi = projection of sonar boresight (+Z_sonar) onto body +X (forward).
        # A depression angle of phi compresses range-direction image motion by cos_phi.
        cos_phi = float(self._R_sonar_to_body[0, 2])
        if cos_phi <= 0.1:
            self.get_logger().error(
                f"Sonar boresight forward component cos_phi={cos_phi:.4f} <= 0.1. "
                "Check sonar TF — no EKF update.",
                throttle_duration_sec=5.0,
            )
            return None

        # ── Forward displacement: pixel-row estimate (preferred) ──────────────
        # In the sonar image, ALL feature points shift by the same number of rows
        # for pure forward vehicle motion, regardless of beam azimuth.  Fitting the
        # RANSAC in pixel (col, row) space therefore gives a uniform, low-noise
        # row-direction translation — the per-azimuth cos(az) variation that inflates
        # the Cartesian RANSAC residuals simply does not exist in pixel space.
        #
        # The row shift is converted to metres via the average range-per-bin scale:
        #   d_range = row_shift × Δrange_per_bin
        # and then corrected for the sonar depression angle:
        #   d_fwd_sonar = d_range / cos_phi
        #
        # Positive row_shift → sensor moved toward LARGER range → FORWARD.  ✓
        T_pixel = result.get("transformation_pixel")
        if T_pixel is not None and self._matcher._ranges_m is not None:
            T_sensor_px = cv2.invertAffineTransform(T_pixel)
            row_shift = float(T_sensor_px[1, 2])   # pixels; positive = forward
            # Average metres-per-range-bin (linear approximation; valid for small shifts)
            ranges = self._matcher._ranges_m
            range_per_bin = float((ranges[-1] - ranges[0]) / max(len(ranges) - 1, 1))
            d_fwd_sonar = row_shift * range_per_bin / cos_phi

            # Lateral from column direction (pixel space).
            # col shift → beam-angle displacement → lateral metres at mean seafloor range.
            col_shift   = float(T_sensor_px[0, 2])  # pixels; positive toward higher azimuth
            azimuths    = self._matcher._beam_azimuths_rad
            beam_spacing = abs(float(azimuths[-1] - azimuths[0])) / max(len(azimuths) - 1, 1)
            # Mean range of the seafloor region (rows above crop_row are water column)
            crop = self._matcher._crop_row
            mean_range = float(np.mean(ranges[crop:]) if crop < len(ranges) else np.mean(ranges))
            # Port-positive lateral: higher azimuth col → starboard → negative FLU Y
            d_left_sonar_px = -col_shift * mean_range * beam_spacing
        else:
            d_fwd_sonar  = None
            d_left_sonar_px = None

        # ── Lateral + heading: Cartesian transform from pixel inliers ─────────
        # The Cartesian stage uses only the pixel-RANSAC inlier subset, so it still
        # benefits from the improved inlier quality even though the forward estimate
        # now comes from the cleaner pixel-row calculation above.
        if T is not None:
            T_sensor = cv2.invertAffineTransform(T)
            dp       = T_sensor[0:2, 2].copy()
            dR       = T_sensor[0:2, 0:2]
            dtheta_image = float(np.arctan2(dR[1, 0], dR[0, 0]))
            d_left_sonar = float(dp[1])             # Cartesian lateral
            dtheta_body  = -dtheta_image / cos_phi  # CCW positive
        else:
            # Cartesian stage failed (too few inliers after Cartesian conversion).
            # Fall back: lateral from pixel col-direction; heading from pixel rotation.
            if T_pixel is None:
                return None
            T_sensor_px2 = cv2.invertAffineTransform(T_pixel)
            dtheta_image = float(np.arctan2(T_sensor_px2[1, 0], T_sensor_px2[0, 0]))
            dtheta_body  = -dtheta_image / cos_phi
            d_left_sonar = d_left_sonar_px if d_left_sonar_px is not None else 0.0

        # Use pixel-row forward if available, else Cartesian dp[0]
        if d_fwd_sonar is None:
            if T is None:
                return None
            T_sensor_fb = cv2.invertAffineTransform(T)
            d_fwd_sonar = float(T_sensor_fb[0, 2]) / cos_phi

        # Reconstruct FLU displacements at the sonar origin

        # --- Lever-arm correction -------------------------------------------
        # The sonar measures motion of its own origin; the EKF expects the body/IMU origin.
        # For small CCW rotation dtheta about base_link, a point at (r_fwd, r_left) moves:
        #   delta_p = [-r_left * dtheta,  +r_fwd * dtheta]  (in FLU body)
        # Body-origin displacement = sonar displacement - delta_p.
        # r_z (vertical) does not enter this planar correction.
        r_fwd  = float(self._t_sonar_to_body[0])   # FLU X: negative = sonar is aft
        r_left = float(self._t_sonar_to_body[1])   # FLU Y: zero for centred mount
        delta_p_lever = np.array([-dtheta_body * r_left, dtheta_body * r_fwd])
        dp_body_origin = np.array([d_fwd_sonar, d_left_sonar]) - delta_p_lever

        # Final EKF measurement: [d_forward, d_left, dtheta] in FLU body frame
        return np.array([dp_body_origin[0], dp_body_origin[1], dtheta_body])

    # ------------------------------------------------- publish sonar DR path
    def _publish_sonar_dr_path(self) -> None:
        self._dbg_sonar_dr_path.header.stamp = self.get_clock().now().to_msg()
        self._dbg_sonar_dr_path_pub.publish(self._dbg_sonar_dr_path)

    def _publish_imu_filter_path(self) -> None:
        self._imu_filter_path.header.stamp = self.get_clock().now().to_msg()
        self._imu_filter_path_pub.publish(self._imu_filter_path)

    # ---------------------------------------------------- CSV file cleanup
    def destroy_node(self):
        for f in (self._csv_sonar_f, self._csv_nis_f):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        super().destroy_node()

    # ----------------------------------------------------- publish odometry
    def _publish_odometry(self, stamp: Time, update_path: bool = False) -> None:
        with self._lock:
            px, py    = self.ekf.px, self.ekf.py
            vx, vy    = self.ekf.vx, self.ekf.vy
            theta     = self.ekf.theta
            P         = self.ekf.covariance

        msg = Odometry()
        msg.header.stamp    = stamp
        msg.header.frame_id = self.odom_frame_id
        msg.child_frame_id  = self.base_frame_id

        msg.pose.pose.position.x = px
        msg.pose.pose.position.y = py
        msg.pose.pose.position.z = 0.0

        qx, qy, qz, qw = heading_to_quaternion(theta)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # 6×6 pose covariance (x, y, z, roll, pitch, yaw)
        cov6 = np.zeros((6, 6))
        cov6[0, 0] = P[0, 0]  # px-px
        cov6[0, 1] = P[0, 1]  # px-py
        cov6[1, 0] = P[1, 0]
        cov6[1, 1] = P[1, 1]  # py-py
        cov6[5, 5] = P[4, 4]  # theta-theta
        cov6[0, 5] = P[0, 4]
        cov6[5, 0] = P[4, 0]
        cov6[1, 5] = P[1, 4]
        cov6[5, 1] = P[4, 1]
        msg.pose.covariance = cov6.flatten().tolist()

        # Twist in FLU body frame
        # ENU: vx=East, vy=North; Forward=[cosθ, sinθ], Left=[-sinθ, cosθ]
        c, s = np.cos(theta), np.sin(theta)
        vx_body =  c * vx + s * vy    # forward  (+X_FLU)
        vy_body = -s * vx + c * vy    # leftward (+Y_FLU)
        msg.twist.twist.linear.x = vx_body
        msg.twist.twist.linear.y = vy_body
        msg.twist.twist.linear.z = 0.0

        cov6t = np.zeros((6, 6))
        cov6t[0, 0] = P[2, 2]
        cov6t[0, 1] = P[2, 3]
        cov6t[1, 0] = P[3, 2]
        cov6t[1, 1] = P[3, 3]
        msg.twist.covariance = cov6t.flatten().tolist()

        self._odom_pub.publish(msg)

        if update_path:
            now = self.get_clock().now().to_msg()
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp    = now
            pose_stamped.header.frame_id = self.odom_frame_id
            pose_stamped.pose = msg.pose.pose
            self._path_msg.header.stamp = now
            self._path_msg.poses.append(pose_stamped)
            self._path_pub.publish(self._path_msg)

            if self._csv_sonar_writer is not None:
                t = ros_time_to_sec(stamp)
                self._csv_sonar_writer.writerow([
                    f"{t:.6f}", f"{px:.6f}", f"{py:.6f}", f"{float(np.rad2deg(theta)):.4f}"
                ])
                self._csv_sonar_f.flush()

        if self._tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp    = self.get_clock().now().to_msg()
            tf.header.frame_id = self.odom_frame_id
            tf.child_frame_id  = self.base_frame_id
            tf.transform.translation.x = px
            tf.transform.translation.y = py
            tf.transform.translation.z = 0.0
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self._tf_broadcaster.sendTransform(tf)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = SonarOdometryNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
