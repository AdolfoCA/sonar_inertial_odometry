"""
ROS2 node: IMU Dead-Reckoning comparison (raw vs Madgwick-filtered).

raw path      — double-integrates /ouster/imu with gravity included
filtered path — uses Madgwick roll/pitch to remove gravity, then integrates
                in the same ENU frame as the sonar/EKF path

Both paths start from the origin with initial_heading_deg and are expressed
in the odom (ENU) frame, so they can be directly compared with /odometry/path.

Publications
------------
  /imu_dr/raw/path      nav_msgs/Path
  /imu_dr/filtered/path nav_msgs/Path

Parameters
----------
  raw_imu_topic       string  (default: /ouster/imu)
  filt_imu_topic      string  (default: /imu/data)
  odom_frame_id       string  (default: odom)
  initial_heading_deg float   (default: 0.0)  compass degrees, 0=North CW
  accel_bias_x        float   (default: 0.0)
  accel_bias_y        float   (default: 0.0)
  gyro_bias_z         float   (default: 0.0)
"""

from __future__ import annotations

import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

from sensor_msgs.msg import Imu
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time

try:
    from tf2_ros import Buffer, TransformListener
    TF2_AVAILABLE = True
except ImportError:
    TF2_AVAILABLE = False

_G = 9.81


def _ros_time_to_sec(stamp: Time) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _heading_to_quaternion(theta: float) -> tuple[float, float, float, float]:
    half = theta / 2.0
    return 0.0, 0.0, float(np.sin(half)), float(np.cos(half))


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion → 3×3 rotation matrix."""
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-10:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s*(qy*qy + qz*qz),   s*(qx*qy - qw*qz),   s*(qx*qz + qw*qy)],
        [    s*(qx*qy + qw*qz), 1 - s*(qx*qx + qz*qz),  s*(qy*qz - qw*qx)],
        [    s*(qx*qz - qw*qy),     s*(qy*qz + qw*qx), 1 - s*(qx*qx + qy*qy)],
    ])


class _RawDeadReckoner:
    """Naive double-integration — gravity included, shows raw IMU drift."""

    def __init__(self, initial_heading: float, b_ax: float, b_ay: float, b_gz: float,
                 R_imu_to_body: np.ndarray | None = None):
        self.theta = initial_heading
        self.px    = 0.0
        self.py    = 0.0
        self._vx   = 0.0
        self._vy   = 0.0
        self._prev_t: float | None = None
        self._b_ax = b_ax
        self._b_ay = b_ay
        self._b_gz = b_gz
        self._R = R_imu_to_body if R_imu_to_body is not None else np.eye(3)

    def update(self, msg: Imu) -> tuple[float, float, float] | None:
        t = _ros_time_to_sec(msg.header.stamp)
        if self._prev_t is None:
            self._prev_t = t
            return None
        dt = t - self._prev_t
        if dt <= 0.0:
            self._prev_t = t
            return None
        self._prev_t = t

        # Rotate sensor → body (FLU: X=fwd, Y=left, Z=up), then subtract body-frame biases.
        a_imu = np.array([msg.linear_acceleration.x,
                          msg.linear_acceleration.y,
                          msg.linear_acceleration.z])
        w_imu = np.array([msg.angular_velocity.x,
                          msg.angular_velocity.y,
                          msg.angular_velocity.z])
        a_body = self._R @ a_imu
        w_body = self._R @ w_imu

        ax = a_body[0] - self._b_ax
        ay = a_body[1] - self._b_ay
        gz = w_body[2] - self._b_gz

        c, s = np.cos(self.theta), np.sin(self.theta)
        ax_enu = c * ax - s * ay
        ay_enu = s * ax + c * ay

        self.px    += self._vx * dt + 0.5 * ax_enu * dt * dt
        self.py    += self._vy * dt + 0.5 * ay_enu * dt * dt
        self._vx   += ax_enu * dt
        self._vy   += ay_enu * dt
        self.theta += gz * dt  # CCW positive, matches EKF convention

        return self.px, self.py, self.theta


class _FilteredDeadReckoner:
    """
    Dead-reckoning with gravity removal using the Madgwick orientation.

    Replicates the exact pipeline used by sonar_odometry_node + EKF:
      1. Rotate sensor → body via R_imu_to_body
      2. Remove gravity: rotate to world, subtract g from Z, rotate back to sensor,
         then to body (equivalent to subtracting R_imu_to_body @ R_sensor^T @ [0,0,g])
      3. Subtract body-frame biases
      4. Integrate in ENU using independently-tracked yaw (gyro, CCW positive)

    Madgwick yaw is NOT used for horizontal rotation — only roll/pitch contribute
    via the gravity removal round-trip.
    """

    def __init__(self, initial_heading: float, b_ax: float, b_ay: float, b_gz: float,
                 R_imu_to_body: np.ndarray | None = None):
        self.theta  = initial_heading
        self.px     = 0.0
        self.py     = 0.0
        self._vx    = 0.0
        self._vy    = 0.0
        self._prev_t: float | None = None
        self._b_ax  = b_ax
        self._b_ay  = b_ay
        self._b_gz  = b_gz
        self._R = R_imu_to_body if R_imu_to_body is not None else np.eye(3)

    def update(self, msg: Imu) -> tuple[float, float, float] | None:
        t = _ros_time_to_sec(msg.header.stamp)
        if self._prev_t is None:
            self._prev_t = t
            return None
        dt = t - self._prev_t
        if dt <= 0.0:
            self._prev_t = t
            return None
        self._prev_t = t

        q = msg.orientation
        if q.x == 0.0 and q.y == 0.0 and q.z == 0.0 and q.w == 0.0:
            return None

        # Gravity removal in sensor frame, then rotate to body.
        # g_sensor = R_sensor_to_world^T @ [0,0,g] — specific force due to gravity in sensor frame.
        R_sensor_to_world = _quat_to_R(q.x, q.y, q.z, q.w)
        g_sensor = R_sensor_to_world.T @ np.array([0.0, 0.0, _G])

        a_sensor = np.array([msg.linear_acceleration.x,
                              msg.linear_acceleration.y,
                              msg.linear_acceleration.z])
        w_sensor = np.array([msg.angular_velocity.x,
                              msg.angular_velocity.y,
                              msg.angular_velocity.z])

        # Gravity-free acceleration and angular velocity in body frame
        a_body = self._R @ (a_sensor - g_sensor)
        w_body = self._R @ w_sensor

        # Subtract body-frame biases (same convention as main EKF)
        ax_body = a_body[0] - self._b_ax
        ay_body = a_body[1] - self._b_ay
        gz = w_body[2] - self._b_gz

        # FLU body → ENU: fwd=[cosθ, sinθ], left=[-sinθ, cosθ]
        c, s = np.cos(self.theta), np.sin(self.theta)
        ax_enu = c * ax_body - s * ay_body
        ay_enu = s * ax_body + c * ay_body

        self.px    += self._vx * dt + 0.5 * ax_enu * dt * dt
        self.py    += self._vy * dt + 0.5 * ay_enu * dt * dt
        self._vx   += ax_enu * dt
        self._vy   += ay_enu * dt
        self.theta += gz * dt  # CCW positive, matches EKF convention

        return self.px, self.py, self.theta


class ImuDeadReckoningNode(Node):

    def __init__(self):
        super().__init__("imu_dr_node")

        self.declare_parameter("raw_imu_topic",       "/ouster/imu")
        self.declare_parameter("filt_imu_topic",      "/imu/data")
        self.declare_parameter("odom_frame_id",       "odom")
        self.declare_parameter("imu_frame_id",        "imu_link")
        self.declare_parameter("base_frame_id",       "base_link")
        self.declare_parameter("initial_heading_deg",  0.0)
        self.declare_parameter("accel_bias_x",         0.0)
        self.declare_parameter("accel_bias_y",         0.0)
        self.declare_parameter("gyro_bias_z",          0.0)

        gp       = self.get_parameter
        frame_id = gp("odom_frame_id").value
        init_hdg = np.deg2rad(gp("initial_heading_deg").value)
        b_ax     = gp("accel_bias_x").value
        b_ay     = gp("accel_bias_y").value
        b_gz     = gp("gyro_bias_z").value

        self._imu_frame_id  = gp("imu_frame_id").value
        self._base_frame_id = gp("base_frame_id").value
        self._frame_id      = frame_id
        self._init_hdg      = init_hdg
        self._b_ax, self._b_ay, self._b_gz = b_ax, b_ay, b_gz
        self._lock          = threading.Lock()
        self._raw_dr:  _RawDeadReckoner      | None = None
        self._filt_dr: _FilteredDeadReckoner | None = None

        self._raw_path  = Path()
        self._filt_path = Path()
        self._raw_path.header.frame_id  = frame_id
        self._filt_path.header.frame_id = frame_id

        be_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        rel_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._raw_path_pub  = self.create_publisher(Path, "/imu_dr/raw/path",      rel_qos)
        self._filt_path_pub = self.create_publisher(Path, "/imu_dr/filtered/path", rel_qos)

        self.create_subscription(Imu, gp("raw_imu_topic").value,  self._raw_cb,  be_qos)
        self.create_subscription(Imu, gp("filt_imu_topic").value, self._filt_cb, be_qos)

        self.create_timer(0.2, self._publish_paths)

        if TF2_AVAILABLE:
            self._tf_buffer   = Buffer(cache_time=Duration(seconds=3600))
            self._tf_listener = TransformListener(self._tf_buffer, self)
        else:
            self._tf_buffer = None
            self.get_logger().error("tf2_ros not available — R_imu_to_body will be identity.")
            self._init_dr(np.eye(3))

        self.get_logger().info(
            f"IMU DR node started  raw→{gp('raw_imu_topic').value}  "
            f"filtered→{gp('filt_imu_topic').value}  "
            f"init_hdg={np.rad2deg(init_hdg):.1f}° ENU"
        )

    def _init_dr(self, R_imu_to_body: np.ndarray) -> None:
        self._raw_dr  = _RawDeadReckoner(self._init_hdg, self._b_ax, self._b_ay, self._b_gz,
                                         R_imu_to_body)
        self._filt_dr = _FilteredDeadReckoner(self._init_hdg, self._b_ax, self._b_ay, self._b_gz,
                                              R_imu_to_body)
        self._raw_path  = Path()
        self._filt_path = Path()
        self._raw_path.header.frame_id  = self._frame_id
        self._filt_path.header.frame_id = self._frame_id

    def _ensure_tf(self) -> bool:
        """Look up R_imu_to_body once and initialise DR objects. Returns True when ready."""
        if self._raw_dr is not None:
            return True
        if self._tf_buffer is None:
            return False
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame_id, self._imu_frame_id, rclpy.time.Time()
            )
            q = tf.transform.rotation
            R = _quat_to_R(q.x, q.y, q.z, q.w)
            self.get_logger().info(
                f"IMU DR: TF {self._imu_frame_id} → {self._base_frame_id} cached.\n"
                f"R_imu_to_body =\n{R}"
            )
            self._init_dr(R)
            return True
        except Exception as e:
            self.get_logger().warn(
                f"IMU DR: TF not yet available ({e})", throttle_duration_sec=5.0
            )
            return False

    def _raw_cb(self, msg: Imu) -> None:
        with self._lock:
            if not self._ensure_tf():
                return
            result = self._raw_dr.update(msg)
            if result is not None:
                self._raw_path.poses.append(self._make_pose(*result))

    def _filt_cb(self, msg: Imu) -> None:
        with self._lock:
            if not self._ensure_tf():
                return
            result = self._filt_dr.update(msg)
            if result is not None:
                self._filt_path.poses.append(self._make_pose(*result))

    def _make_pose(self, px: float, py: float, theta: float) -> PoseStamped:
        ps = PoseStamped()
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.header.frame_id = self._frame_id
        ps.pose.position.x = px
        ps.pose.position.y = py
        qx, qy, qz, qw = _heading_to_quaternion(theta)
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        return ps

    def _publish_paths(self) -> None:
        now = self.get_clock().now().to_msg()
        with self._lock:
            if self._raw_dr is None:
                return
            self._raw_path.header.stamp  = now
            self._filt_path.header.stamp = now
            self._raw_path_pub.publish(self._raw_path)
            self._filt_path_pub.publish(self._filt_path)


def main(args=None):
    rclpy.init(args=args)
    node = ImuDeadReckoningNode()
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
