from __future__ import annotations

import math
from typing import Any

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QSizePolicy, QWidget

from .imu import ImuSample
from .orientation import Matrix3, Pose


class Pose3DWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.pose: Pose | None = None
        self.sample: ImuSample | None = None
        self.read_ms: float | None = None
        self.interval_ms: float | None = None
        self.setMinimumSize(420, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def clear_pose(self) -> None:
        self.pose = None
        self.sample = None
        self.read_ms = None
        self.interval_ms = None
        self.update()

    def set_pose(self, pose: Pose, sample: ImuSample, read_ms: float, interval_ms: float) -> None:
        self.pose = pose
        self.sample = sample
        self.read_ms = read_ms
        self.interval_ms = interval_ms
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(self.rect(), QColor("#111111"))
            self._paint(painter, self.width(), self.height())
        finally:
            painter.end()

    @staticmethod
    def _rot_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list[list[float]]:
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]

    @staticmethod
    def _project(x: float, y: float, z: float, cx: float, cy: float, scale: float) -> tuple[float, float]:
        screen_x = x - y * 0.58
        screen_y = z - y * 0.34
        return cx + screen_x * scale, cy - screen_y * scale

    @staticmethod
    def _color(color: str, alpha: int | None = None) -> QColor:
        qcolor = QColor(color)
        if alpha is not None:
            qcolor.setAlpha(max(0, min(255, int(alpha))))
        return qcolor

    @staticmethod
    def _font(size: int, bold: bool = False) -> QFont:
        font = QFont("Consolas", size)
        font.setBold(bold)
        return font

    def _draw_text(self, painter: QPainter, x: float, y: float, text: str, color: str, size: int, bold: bool = False) -> None:
        painter.setPen(QColor(color))
        painter.setFont(self._font(size, bold=bold))
        metrics = painter.fontMetrics()
        painter.drawText(QPointF(x, y + metrics.ascent()), text)

    def _draw_axis_line(
        self,
        painter: QPainter,
        vec: tuple[float, float, float],
        color: str,
        width: int,
        cx: float,
        cy: float,
        scale: float,
        label: str,
    ) -> None:
        x2, y2 = self._project(vec[0], vec[1], vec[2], cx, cy, scale)
        pen = QPen(QColor(color))
        pen.setWidth(width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(cx, cy), QPointF(x2, y2))

        painter.setPen(QColor(color))
        painter.setFont(self._font(11, bold=True))
        painter.drawText(QPointF(x2 + 12.0, y2 + 4.0), label)

    @staticmethod
    def _transform_body_point(matrix: Matrix3 | list[list[float]], x: float, y: float, z: float) -> tuple[float, float, float]:
        return (
            matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
            matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
            matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
        )

    def _draw_reference_plane(
        self,
        painter: QPainter,
        matrix: Matrix3 | list[list[float]],
        cx: float,
        cy: float,
        scale: float,
    ) -> None:
        x_half = 1.2
        y_half = 1.0
        z = -0.04

        def project_body(x: float, y: float, z_value: float) -> QPointF:
            wx, wy, wz = self._transform_body_point(matrix, x, y, z_value)
            return QPointF(*self._project(wx, wy, wz, cx, cy, scale))

        corners = [
            project_body(-x_half, -y_half, z),
            project_body(x_half, -y_half, z),
            project_body(x_half, y_half, z),
            project_body(-x_half, y_half, z),
        ]

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color("#25303A", 115))
        painter.drawPolygon(QPolygonF(corners))

        grid_pen = QPen(self._color("#9EB2C2", 55))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        grid_steps = 4
        for step in range(-grid_steps, grid_steps + 1):
            x = x_half * step / grid_steps
            painter.drawLine(project_body(x, -y_half, z), project_body(x, y_half, z))

            y = y_half * step / grid_steps
            painter.drawLine(project_body(-x_half, y, z), project_body(x_half, y, z))

        outline_pen = QPen(self._color("#C6D7E0", 110))
        outline_pen.setWidth(2)
        outline_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(outline_pen)
        painter.drawPolygon(QPolygonF(corners))

    def _paint(self, painter: QPainter, width: int, height: int) -> None:
        canvas_w = max(400, int(width))
        canvas_h = max(320, int(height))
        cx = canvas_w * 0.5
        cy = canvas_h * 0.60
        scale = min(canvas_w * 0.28, canvas_h * 0.32)

        self._draw_text(painter, 12, 10, "Arducam IMU 3D Pose", "#E0E0E0", 12, bold=True)

        if self.pose is None or self.sample is None:
            self._draw_text(painter, 12, 36, "Waiting for IMU sample...", "#B8C7D9", 10)
            return

        pose = self.pose
        sample = self.sample
        lines = [
            "roll=%7.2f deg   pitch=%7.2f deg   yaw=%7.2f deg"
            % (pose.roll_deg, pose.pitch_deg, pose.yaw_deg),
            "gx=%7.2f dps   gy=%7.2f dps   gz=%7.2f dps"
            % (pose.gx_dps, pose.gy_dps, pose.gz_dps),
            "ax=%8.5f g   ay=%8.5f g   az=%8.5f g" % (sample.ax_g, sample.ay_g, sample.az_g),
            "imu gx=%7.2f dps   gy=%7.2f dps   gz=%7.2f dps"
            % (sample.gx_dps, sample.gy_dps, sample.gz_dps),
            "temp=%7.2f C  dt=%7.2f ms" % (sample.temperature_c, pose.dt_ms),
            "still=%d  bias_ready=%d" % (1 if pose.still else 0, 1 if pose.bias_ready else 0),
        ]
        if self.read_ms is not None and self.interval_ms is not None:
            lines.append("read=%7.3f ms   interval=%7.3f ms" % (self.read_ms, self.interval_ms))

        y = 36
        for line in lines:
            self._draw_text(painter, 12, y, line, "#B8C7D9", 10)
            y += 18

        self._draw_axis_line(painter, (1.3, 0.0, 0.0), "#772222", 2, cx, cy, scale, "Wx")
        self._draw_axis_line(painter, (0.0, 1.3, 0.0), "#3A3A3A", 2, cx, cy, scale, "Wy")
        self._draw_axis_line(painter, (0.0, 0.0, 1.3), "#227722", 2, cx, cy, scale, "Wz")

        matrix: Matrix3 | list[list[float]] | None = pose.rotation_matrix
        if matrix is None:
            matrix = self._rot_matrix(pose.roll_deg, pose.pitch_deg, pose.yaw_deg)
        bx = (matrix[0][0], matrix[1][0], matrix[2][0])
        by = (matrix[0][1], matrix[1][1], matrix[2][1])
        bz = (matrix[0][2], matrix[1][2], matrix[2][2])
        self._draw_reference_plane(painter, matrix, cx, cy, scale)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawEllipse(QPointF(cx, cy), 4.0, 4.0)

        self._draw_axis_line(painter, bx, "#FF5C5C", 4, cx, cy, scale, "Bx")
        self._draw_axis_line(painter, by, "#AAAAAA", 4, cx, cy, scale, "By")
        self._draw_axis_line(painter, bz, "#60FF60", 4, cx, cy, scale, "Bz")
