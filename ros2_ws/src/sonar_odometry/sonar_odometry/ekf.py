"""
Extended Kalman Filter for Sonar Inertial Odometry.

State vector (10D):
  [px, py, vx, vy, theta, b_ax, b_ay, b_gz, s_ax, s_ay]
   px, py    : position in local odom frame (meters); +X = vehicle-forward at startup, +Y = vehicle-left
   vx, vy    : velocity in local odom frame (m/s)
   theta     : yaw (rad), zero = vehicle-forward at startup, CCW positive
   b_ax, b_ay: accelerometer biases in FLU body frame (m/s^2)
   b_gz      : gyro yaw-rate bias in FLU body frame (rad/s)
   s_ax, s_ay: accelerometer scale factors (dimensionless)

IMU input convention (FLU body frame):
  u = [ax_flu, ay_flu, gz_flu]
  ax_flu: forward acceleration  (m/s^2, gravity-compensated, +X_body)
  ay_flu: leftward acceleration (m/s^2, gravity-compensated, +Y_body)
  gz_flu: yaw-rate CCW positive (rad/s, +Z_body = up = ENU convention)

Sonar measurement (FLU body frame):
  z = [d_forward, d_left, d_theta]
  d_forward: displacement forward  (meters, +X_body)
  d_left   : displacement leftward (meters, +Y_body)
  d_theta  : heading change        (rad, CCW positive)
"""

from __future__ import annotations

import numpy as np


class ESEKF:
    # ------------------------------------------------------------------ init
    def __init__(
        self,
        initial_heading_rad: float = 0.0,
        accel_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
        gyro_bias_z: float = 0.002,
        # Q — IMU white-noise variances  [(unit)²]
        # These are the diagonal entries of the continuous-time noise covariance
        # that drive the IMU prediction step.  Increase to trust the IMU less.
        acc_covariance:    float = 1.754e-4,   # (m/s²)²   forward & lateral accel noise
        gyro_covariance:   float = 7.615e-7,   # (rad/s)²  yaw-rate noise
        # Q — bias / scale random-walk variances  [(unit)² per step]
        # Larger → filter allows biases to drift faster.
        b_acc_covariance:  float = 0.25,        # (m/s²)²
        b_gyro_covariance: float = 0.25,        # (rad/s)²
        scale_covariance:  float = 1.0e-6,      # dimensionless²
        # R — sonar measurement noise variances  [(unit)²]
        # Increase to trust sonar less, decrease to follow sonar more tightly.
        sonar_position_covariance: float = 5.506e-4,  # m²  (forward and lateral)
        sonar_yaw_covariance:      float = 5.506e-4,  # rad²
    ):
        # ---- Nominal state ----
        self.px: float = 0.0
        self.py: float = 0.0
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.theta: float = initial_heading_rad
        self.b_ax: float = float(accel_bias[0])
        self.b_ay: float = float(accel_bias[1])
        self.b_gz: float = gyro_bias_z
        self.s_ax: float = 1.0
        self.s_ay: float = 1.0

        # ---- Covariance ----
        self.P: np.ndarray = np.eye(10) * 10.0

        # ---- Q: process noise diagonal variances ----
        self.sigma_eta1_sq = acc_covariance     # forward accel white noise
        self.sigma_eta2_sq = acc_covariance     # lateral accel white noise
        self.sigma_eta3_sq = gyro_covariance    # yaw-rate white noise
        self.sigma_b_ax_sq = b_acc_covariance
        self.sigma_b_ay_sq = b_acc_covariance
        self.sigma_b_gz_sq = b_gyro_covariance
        self.ss_ax = scale_covariance
        self.ss_ay = scale_covariance

        # ---- R: measurement noise diagonal variances ----
        self.sigma_w4_sq = sonar_position_covariance   # forward position
        self.sigma_w5_sq = sonar_position_covariance   # lateral position
        self.sigma_w6_sq = sonar_yaw_covariance

        # ---- Bookkeeping (state at last sonar update) ----
        self.prev_px:    float | None = None
        self.prev_py:    float | None = None
        self.prev_theta: float | None = None

    # ----------------------------------------------------------------- helpers
    def _as_vector(self) -> np.ndarray:
        return np.array([
            self.px, self.py, self.vx, self.vy, self.theta,
            self.b_ax, self.b_ay, self.b_gz, self.s_ax, self.s_ay
        ])

    def _set_from_vector(self, x: np.ndarray) -> None:
        (self.px, self.py, self.vx, self.vy, self.theta,
         self.b_ax, self.b_ay, self.b_gz, self.s_ax, self.s_ay) = x

    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return float(np.arctan2(np.sin(a), np.cos(a)))

    @staticmethod
    def _wrap_to_2pi(a: float) -> float:
        """Wrap angle to [0, 2pi]."""
        a = np.arctan2(np.sin(a), np.cos(a))
        if a < 0:
            a += 2 * np.pi
        return float(a)

    def _build_R(self) -> np.ndarray:
        return np.diag([self.sigma_w4_sq, self.sigma_w5_sq, self.sigma_w6_sq])

    # --------------------------------------------------------------- prediction
    def prediction(self, u: np.ndarray, dt: float) -> None:
        """
        EKF prediction step driven by IMU measurements.

        Parameters
        ----------
        u  : array [ax_flu, ay_flu, gz_flu]  (FLU body frame, gravity-compensated)
             ax_flu: forward accel (m/s^2), ay_flu: leftward accel (m/s^2),
             gz_flu: CCW yaw rate (rad/s) — same sign as ENU ω_z
        dt : time step in seconds
        """
        if dt <= 0.0:
            return

        ax_m, ay_m, gz_m = float(u[0]), float(u[1]), float(u[2])

        # Cache previous state
        theta = self.theta
        vx, vy = self.vx, self.vy
        b_ax, b_ay, b_gz = self.b_ax, self.b_ay, self.b_gz
        s_ax, s_ay = self.s_ax, self.s_ay

        # Corrected body-frame accelerations and yaw rate
        ax_c = (ax_m - b_ax) / s_ax   # forward
        ay_c = (ay_m - b_ay) / s_ay   # left
        gz_c = gz_m - b_gz            # CCW+, equal to ω_z_enu

        # Rotation: FLU body → ENU  (theta = ENU yaw, CCW from East)
        # Forward  (+X_FLU) → ENU [cos θ,  sin θ]
        # Left     (+Y_FLU) → ENU [-sin θ, cos θ]
        c, s = np.cos(theta), np.sin(theta)
        ax_enu = c * ax_c - s * ay_c    # East component
        ay_enu = s * ax_c + c * ay_c    # North component

        # Centripetal: ω_enu × v_enu in 2-D (ω_z_enu = gz_c, CCW+)
        # ω × v = [-ω_z*vy_N, ω_z*vx_E]
        ax_total = ax_enu - gz_c * vy
        ay_total = ay_enu + gz_c * vx

        # ------ Linearised continuous-time Jacobian (10×10) ------
        Fc = np.zeros((10, 10))

        Fc[0, 2] = 1.0
        Fc[1, 3] = 1.0

        # Centripetal velocity partials: d(ax_total)/d(vy) = -gz_c, d(ay_total)/d(vx) = +gz_c
        Fc[2, 3] = -gz_c
        Fc[3, 2] =  gz_c

        # Theta partials of ENU accel:
        # d(ax_enu)/dθ = -s*ax_c - c*ay_c,  d(ay_enu)/dθ = c*ax_c - s*ay_c
        fvx_th = -s * ax_c - c * ay_c
        fvy_th =  c * ax_c - s * ay_c
        Fc[0, 4] = 0.5 * dt * fvx_th
        Fc[1, 4] = 0.5 * dt * fvy_th
        Fc[2, 4] = fvx_th
        Fc[3, 4] = fvy_th

        # Bias partials (b_ax=forward, b_ay=left):
        # ax_enu = c*ax_c - s*ay_c  → d/d(b_ax)=-c/s_ax, d/d(b_ay)=+s/s_ay
        # ay_enu = s*ax_c + c*ay_c  → d/d(b_ax)=-s/s_ax, d/d(b_ay)=-c/s_ay
        Fc[2, 5] = -c / s_ax
        Fc[2, 6] = +s / s_ay
        Fc[3, 5] = -s / s_ax
        Fc[3, 6] = -c / s_ay

        # b_gz partials (centripetal + theta):
        # ax_total = ax_enu - gz_c*vy → d/d(b_gz) = +vy  (gz_c = gz_m - b_gz)
        # ay_total = ay_enu + gz_c*vx → d/d(b_gz) = -vx
        # theta += gz_c*dt           → d(theta_dot)/d(b_gz) = -1
        Fc[2, 7] =  vy
        Fc[3, 7] = -vx
        Fc[4, 7] = -1.0

        # Scale factor partials:
        # ax_enu = c*ax_c - s*ay_c → d/d(s_ax)= c*dax_dsax, d/d(s_ay)= -s*day_dsay
        # ay_enu = s*ax_c + c*ay_c → d/d(s_ax)= s*dax_dsax, d/d(s_ay)= +c*day_dsay
        dax_dsax = -(ax_m - b_ax) / (s_ax ** 2)
        day_dsay = -(ay_m - b_ay) / (s_ay ** 2)
        Fc[2, 8] =  c * dax_dsax
        Fc[2, 9] = -s * day_dsay
        Fc[3, 8] =  s * dax_dsax
        Fc[3, 9] =  c * day_dsay

        F = np.eye(10) + dt * Fc

        # ------ Process noise mapping (10×8) ------
        Gd = np.zeros((10, 8))

        # eta_ax (forward accel noise): ax_enu += c/s_ax, ay_enu += s/s_ax
        Gd[0, 0] = 0.5 * dt**2 * c / s_ax
        Gd[1, 0] = 0.5 * dt**2 * s / s_ax
        Gd[2, 0] = dt * c / s_ax
        Gd[3, 0] = dt * s / s_ax

        # eta_ay (left accel noise): ax_enu += -s/s_ay, ay_enu += c/s_ay
        Gd[0, 1] = -0.5 * dt**2 * s / s_ay
        Gd[1, 1] =  0.5 * dt**2 * c / s_ay
        Gd[2, 1] = -dt * s / s_ay
        Gd[3, 1] =  dt * c / s_ay

        # eta_gz (CCW yaw noise): ax_total += -vy*eta, ay_total += +vx*eta, theta += +eta
        Gd[2, 2] = -dt * vy
        Gd[3, 2] =  dt * vx
        Gd[4, 2] =  dt

        Gd[5, 3] = dt
        Gd[6, 4] = dt
        Gd[7, 5] = dt
        Gd[8, 6] = dt
        Gd[9, 7] = dt

        Qn = np.diag([
            self.sigma_eta1_sq,
            self.sigma_eta2_sq,
            self.sigma_eta3_sq,
            self.sigma_b_ax_sq,
            self.sigma_b_ay_sq,
            self.sigma_b_gz_sq,
            self.ss_ax,
            self.ss_ay,
        ])
        Qd = Gd @ Qn @ Gd.T

        # ------ Propagate nominal state ------
        self.px    += vx * dt + 0.5 * ax_total * dt**2   # East
        self.py    += vy * dt + 0.5 * ay_total * dt**2   # North
        self.vx    += ax_total * dt
        self.vy    += ay_total * dt
        self.theta += gz_c * dt                           # ENU: CCW positive = FLU gz
        self.theta  = self._wrap_angle(self.theta)

        # ------ Propagate covariance ------
        self.P = F @ self.P @ F.T + Qd

    # ------------------------------------------------------------------ update
    def update(
        self,
        z_sonar: np.ndarray,
        nis_threshold: float = 30.0,
    ) -> tuple[float, np.ndarray | None, np.ndarray | None, bool]:
        """
        EKF update with sonar displacement measurement.

        Parameters
        ----------
        z_sonar       : array [d_forward, d_left, d_theta]  (FLU body frame, meters / rad)
        nis_threshold : reject outliers above this NIS value

        Returns
        -------
        (nis, innovation, innovation_covariance, accepted)
        nis is nan on first call or numerical failure.
        """
        if self.prev_px is None:
            self.prev_px    = self.px
            self.prev_py    = self.py
            self.prev_theta = self.theta
            return float('nan'), None, None, False

        # ENU displacement since last sonar update
        dp_e = self.px - self.prev_px
        dp_n = self.py - self.prev_py

        # Predicted FLU body-frame displacement from ENU displacement
        # Forward (+X_FLU) in ENU: [cos θ, sin θ]
        # Left   (+Y_FLU) in ENU: [-sin θ, cos θ]
        c = np.cos(self.prev_theta)
        s = np.sin(self.prev_theta)
        z_pred = np.array([
             c * dp_e + s * dp_n,                             # forward
            -s * dp_e + c * dp_n,                             # left
            self._wrap_angle(self.theta - self.prev_theta),   # dtheta CCW+
        ])

        y = z_sonar - z_pred
        y[2] = self._wrap_angle(y[2])

        # Measurement Jacobian (3×10) — FLU: d(left)/d(px_E, py_N) = [-sinθ, cosθ]
        H = np.zeros((3, 10))
        H[0, 0] =  c;  H[0, 1] =  s   # d(fwd)/d(px_E, py_N)
        H[1, 0] = -s;  H[1, 1] =  c   # d(left)/d(px_E, py_N)
        H[2, 4] =  1.0

        R = self._build_R()
        S = H @ self.P @ H.T + R

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return float('nan'), None, None, False

        nis = float(y.T @ S_inv @ y)
        if nis > nis_threshold:
            # Advance the baseline even on rejection so the next sonar measurement
            # (which is always frame-to-frame) is compared against the correct EKF
            # prediction window.  Without this, repeated rejections cause z_pred to
            # accumulate over many frames while z_sonar remains one-frame, making
            # the innovation grow without bound and locking the filter into a spiral.
            self.prev_px    = self.px
            self.prev_py    = self.py
            self.prev_theta = self.theta
            return nis, y, S, False

        K  = self.P @ H.T @ S_inv
        dx = K @ y
        self._set_from_vector(self._as_vector() + dx)
        self.theta = self._wrap_angle(self.theta)

        I_KH = np.eye(10) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        self.prev_px    = self.px
        self.prev_py    = self.py
        self.prev_theta = self.theta

        return nis, y, S, True

    # ---------------------------------------------------------- public getters
    @property
    def position(self) -> np.ndarray:
        return np.array([self.px, self.py])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])

    @property
    def heading(self) -> float:
        return self.theta

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()
