from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Tuple

from .imu import ImuSample

Quaternion = Tuple[float, float, float, float]
Matrix3 = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]


@dataclass(frozen=True)
class Pose:
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    dt_ms: float
    gx_dps: float
    gy_dps: float
    gz_dps: float
    still: bool
    bias_ready: bool
    rotation_matrix: Matrix3 | None = None


def wrap_angle_deg(deg: float) -> float:
    while deg > 180.0:
        deg -= 360.0
    while deg < -180.0:
        deg += 360.0
    return deg


class OrientationEstimator:
    def __init__(self, accel_blend: float, use_accel: bool) -> None:
        self.accel_blend = max(0.0, min(1.0, float(accel_blend)))
        self.use_accel = bool(use_accel)

        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        self.orientation_q: Quaternion = (1.0, 0.0, 0.0, 0.0)
        self.last_ts: float | None = None

        self.bias_gx_dps = 0.0
        self.bias_gy_dps = 0.0
        self.bias_gz_dps = 0.0
        self.bias_ready = False
        self.bias_calib_count = 0
        self.bias_calib_target = 100
        self.bias_track_alpha = 0.05
        self.gyro_deadband_dps = 0.45

        self.gravity_g: float | None = None
        self.still_locked = False
        self.still_count = 0
        self.move_count = 0
        self.still_enter_count = 6
        self.still_exit_count = 3
        self.was_still_locked = False
        self.yaw_hold_deg = 0.0

    def update(self, sample: ImuSample, dt_s: float | None = None) -> Pose:
        now = time.monotonic()
        if dt_s is not None and dt_s > 0.0:
            dt = max(0.0, min(0.2, float(dt_s)))
            self.last_ts = now
        else:
            if self.last_ts is None:
                dt = 0.0
            else:
                dt = max(0.0, min(0.2, now - self.last_ts))
            self.last_ts = now

        sample_gx_dps = sample.gx_dps
        sample_gy_dps = sample.gy_dps
        sample_gz_dps = sample.gz_dps

        still_sample = self._is_still(sample, sample_gx_dps, sample_gy_dps, sample_gz_dps)
        is_still = self._update_still_state(still_sample)
        entered_still = is_still and not self.was_still_locked
        self._update_bias(sample_gx_dps, sample_gy_dps, sample_gz_dps, is_still)

        if entered_still:
            self.yaw_hold_deg = self._current_euler_degrees()[2]

        gx_dps = self._apply_deadband(sample_gx_dps - self.bias_gx_dps, self.gyro_deadband_dps)
        gy_dps = self._apply_deadband(sample_gy_dps - self.bias_gy_dps, self.gyro_deadband_dps)
        gz_dps = self._apply_deadband(sample_gz_dps - self.bias_gz_dps, self.gyro_deadband_dps)

        if is_still:
            gx_dps = 0.0
            gy_dps = 0.0
            gz_dps = 0.0

        if dt > 0.0:
            if not is_still:
                self._integrate_gyro(gx_dps, gy_dps, gz_dps, dt)

            if self.use_accel:
                self._apply_accel_correction(sample, is_still)

            if is_still:
                self._set_yaw(self.yaw_hold_deg)

        matrix = self._quat_to_matrix(self.orientation_q)
        self.roll_deg, self.pitch_deg, self.yaw_deg = self._matrix_to_euler_degrees(matrix)

        self.was_still_locked = is_still
        return Pose(
            roll_deg=self.roll_deg,
            pitch_deg=self.pitch_deg,
            yaw_deg=self.yaw_deg,
            dt_ms=dt * 1000.0,
            gx_dps=gx_dps,
            gy_dps=gy_dps,
            gz_dps=gz_dps,
            still=is_still,
            bias_ready=self.bias_ready,
            rotation_matrix=matrix,
        )

    def _apply_accel_correction(self, sample: ImuSample, is_still: bool) -> None:
        ax = sample.ax_g
        ay = sample.ay_g
        az = sample.az_g
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm <= 1e-6:
            return

        roll_acc = math.degrees(math.atan2(ay, az))
        pitch_acc = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
        blend = max(self.accel_blend, 0.12) if is_still else min(self.accel_blend, 0.008)

        yaw_deg = self._current_euler_degrees()[2]
        target_q = self._quat_from_euler_degrees(roll_acc, pitch_acc, yaw_deg)
        self.orientation_q = self._quat_slerp(self.orientation_q, target_q, blend)

    def _is_still(self, sample: ImuSample, gx_dps: float, gy_dps: float, gz_dps: float) -> bool:
        ax = sample.ax_g
        ay = sample.ay_g
        az = sample.az_g
        acc_norm = math.sqrt(ax * ax + ay * ay + az * az)

        if self.gravity_g is None:
            self.gravity_g = max(1e-6, acc_norm)

        self.gravity_g = 0.995 * self.gravity_g + 0.005 * acc_norm

        gyro_mag = math.sqrt(gx_dps * gx_dps + gy_dps * gy_dps + gz_dps * gz_dps)
        acc_err = abs(acc_norm - self.gravity_g)
        acc_tol = max(0.08, self.gravity_g * 0.08)

        if self.still_locked:
            return gyro_mag < 2.5 and acc_err < acc_tol * 1.4
        return gyro_mag < 1.2 and acc_err < acc_tol

    def _update_still_state(self, still_sample: bool) -> bool:
        if still_sample:
            self.still_count += 1
            self.move_count = 0
            if self.still_count >= self.still_enter_count:
                self.still_locked = True
        else:
            self.move_count += 1
            self.still_count = 0
            if self.move_count >= self.still_exit_count:
                self.still_locked = False

        return self.still_locked

    def _update_bias(self, gx_dps: float, gy_dps: float, gz_dps: float, is_still: bool) -> None:
        if not is_still:
            return

        if self.bias_calib_count < self.bias_calib_target:
            self.bias_calib_count += 1
            gain = 1.0 / float(self.bias_calib_count)
            self.bias_gx_dps += (gx_dps - self.bias_gx_dps) * gain
            self.bias_gy_dps += (gy_dps - self.bias_gy_dps) * gain
            self.bias_gz_dps += (gz_dps - self.bias_gz_dps) * gain
            if self.bias_calib_count >= self.bias_calib_target:
                self.bias_ready = True
            return

        gain = self.bias_track_alpha
        self.bias_gx_dps += (gx_dps - self.bias_gx_dps) * gain
        self.bias_gy_dps += (gy_dps - self.bias_gy_dps) * gain
        self.bias_gz_dps += (gz_dps - self.bias_gz_dps) * gain

    @staticmethod
    def _apply_deadband(value: float, band: float) -> float:
        if -band < value < band:
            return 0.0
        return value

    def _integrate_gyro(self, gx_dps: float, gy_dps: float, gz_dps: float, dt_s: float) -> None:
        wx = math.radians(gx_dps)
        wy = math.radians(gy_dps)
        wz = math.radians(gz_dps)
        speed = math.sqrt(wx * wx + wy * wy + wz * wz)
        if speed <= 1e-12:
            return

        angle = speed * dt_s
        half_angle = angle * 0.5
        s = math.sin(half_angle) / speed
        delta_q = (
            math.cos(half_angle),
            wx * s,
            wy * s,
            wz * s,
        )
        self.orientation_q = self._quat_normalize(self._quat_multiply(self.orientation_q, delta_q))

    def _set_yaw(self, yaw_deg: float) -> None:
        roll_deg, pitch_deg, _ = self._current_euler_degrees()
        self.orientation_q = self._quat_from_euler_degrees(roll_deg, pitch_deg, yaw_deg)

    def _current_euler_degrees(self) -> tuple[float, float, float]:
        return self._matrix_to_euler_degrees(self._quat_to_matrix(self.orientation_q))

    @staticmethod
    def _quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )

    @staticmethod
    def _quat_normalize(q: Quaternion) -> Quaternion:
        w, x, y, z = q
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm <= 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        inv_norm = 1.0 / norm
        return (w * inv_norm, x * inv_norm, y * inv_norm, z * inv_norm)

    @classmethod
    def _quat_slerp(cls, a: Quaternion, b: Quaternion, blend: float) -> Quaternion:
        t = max(0.0, min(1.0, blend))
        qa = cls._quat_normalize(a)
        qb = cls._quat_normalize(b)
        dot = sum(qa[i] * qb[i] for i in range(4))
        if dot < 0.0:
            qb = (-qb[0], -qb[1], -qb[2], -qb[3])
            dot = -dot

        if dot > 0.9995:
            mixed = tuple((1.0 - t) * qa[i] + t * qb[i] for i in range(4))
            return cls._quat_normalize(mixed)  # type: ignore[arg-type]

        theta_0 = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return cls._quat_normalize(tuple(s0 * qa[i] + s1 * qb[i] for i in range(4)))  # type: ignore[arg-type]

    @staticmethod
    def _quat_from_euler_degrees(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Quaternion:
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        return OrientationEstimator._quat_normalize(
            (
                cy * cp * cr + sy * sp * sr,
                cy * cp * sr - sy * sp * cr,
                sy * cp * sr + cy * sp * cr,
                sy * cp * cr - cy * sp * sr,
            )
        )

    @staticmethod
    def _quat_to_matrix(q: Quaternion) -> Matrix3:
        w, x, y, z = OrientationEstimator._quat_normalize(q)
        return (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        )

    @staticmethod
    def _matrix_to_euler_degrees(matrix: Matrix3) -> tuple[float, float, float]:
        pitch = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
        cp = math.cos(pitch)
        if abs(cp) > 1e-6:
            roll = math.atan2(matrix[2][1], matrix[2][2])
            yaw = math.atan2(matrix[1][0], matrix[0][0])
        else:
            roll = 0.0
            yaw = math.atan2(-matrix[0][1], matrix[1][1])

        return (
            wrap_angle_deg(math.degrees(roll)),
            wrap_angle_deg(math.degrees(pitch)),
            wrap_angle_deg(math.degrees(yaw)),
        )
