from __future__ import annotations

import math
import sys
from typing import Any

from .imu import ImuSample
from .orientation import Pose


class Axis3DViewer:
    def __init__(self, title: str) -> None:
        try:
            from PyQt6.QtCore import QPointF, Qt
            from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
            from PyQt6.QtWidgets import QApplication, QWidget
        except Exception as exc:
            raise RuntimeError(
                "PyQt6 is not installed in the active environment. "
                "Run this demo with the project virtual environment."
            ) from exc

        self.closed = False
        self.pose: Pose | None = None
        self.sample: ImuSample | None = None
        self.read_ms: float | None = None
        self.interval_ms: float | None = None

        self._QColor = QColor
        self._QFont = QFont
        self._QPainter = QPainter
        self._QPen = QPen
        self._QPointF = QPointF
        self._QPolygonF = QPolygonF
        self._Qt = Qt

        app = QApplication.instance()
        if app is None:
            app = QApplication([sys.argv[0] if sys.argv else "imu_axis3d_viewer"])
        app.setQuitOnLastWindowClosed(False)
        self.app = app

        owner = self

        class Axis3DWidget(QWidget):
            def __init__(self) -> None:
                super().__init__()
                self.setWindowTitle(title)
                self.resize(780, 560)
                self.setMinimumSize(400, 320)

            def closeEvent(self, event: Any) -> None:
                owner.closed = True
                event.accept()

            def paintEvent(self, event: Any) -> None:
                painter = QPainter(self)
                try:
                    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                    painter.fillRect(self.rect(), QColor("#111111"))
                    owner._paint(painter, self.width(), self.height())
                finally:
                    painter.end()

        self.widget = Axis3DWidget()
        self.widget.show()
        self.app.processEvents()

    def close(self) -> None:
        self.closed = True
        try:
            self.widget.close()
        except Exception:
            pass

    def tick(self) -> bool:
        if self.closed:
            return False
        try:
            self.app.processEvents()
            if not self.widget.isVisible():
                self.closed = True
                return False
        except Exception:
            self.closed = True
            return False
        return True

    def update_frame(
        self,
        pose: Pose,
        sample: ImuSample,
        read_ms: float,
        interval_ms: float,
    ) -> bool:
        self.pose = pose
        self.sample = sample
        self.read_ms = read_ms
        self.interval_ms = interval_ms
        self.widget.update()
        return self.tick()

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

    def _project(self, x: float, y: float, z: float, cx: float, cy: float, scale: float) -> tuple[float, float]:
        screen_x = x - y * 0.58
        screen_y = z - y * 0.34
        return cx + screen_x * scale, cy - screen_y * scale

    def _color(self, color: str, alpha: int | None = None) -> Any:
        qcolor = self._QColor(color)
        if alpha is not None:
            qcolor.setAlpha(max(0, min(255, int(alpha))))
        return qcolor

    def _font(self, size: int, bold: bool = False) -> Any:
        font = self._QFont("Consolas", size)
        font.setBold(bold)
        return font

    def _draw_text(self, painter: Any, x: float, y: float, text: str, color: str, size: int, bold: bool = False) -> None:
        painter.setPen(self._QColor(color))
        painter.setFont(self._font(size, bold=bold))
        metrics = painter.fontMetrics()
        painter.drawText(self._QPointF(x, y + metrics.ascent()), text)

    def _draw_axis_line(
        self,
        painter: Any,
        vec: tuple[float, float, float],
        color: str,
        width: int,
        cx: float,
        cy: float,
        scale: float,
        label: str,
    ) -> None:
        x2, y2 = self._project(vec[0], vec[1], vec[2], cx, cy, scale)
        pen = self._QPen(self._QColor(color))
        pen.setWidth(width)
        pen.setCapStyle(self._Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(self._QPointF(cx, cy), self._QPointF(x2, y2))

        painter.setPen(self._QColor(color))
        painter.setFont(self._font(11, bold=True))
        painter.drawText(self._QPointF(x2 + 12.0, y2 + 4.0), label)

    @staticmethod
    def _transform_body_point(
        matrix: list[list[float]],
        x: float,
        y: float,
        z: float,
    ) -> tuple[float, float, float]:
        return (
            matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
            matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
            matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
        )

    def _draw_reference_plane(
        self,
        painter: Any,
        matrix: list[list[float]],
        cx: float,
        cy: float,
        scale: float,
    ) -> None:
        x_half = 1.2
        y_half = 1.0
        z = -0.04

        def project_body(x: float, y: float, z: float) -> Any:
            wx, wy, wz = self._transform_body_point(matrix, x, y, z)
            return self._QPointF(*self._project(wx, wy, wz, cx, cy, scale))

        corners = [
            project_body(-x_half, -y_half, z),
            project_body(x_half, -y_half, z),
            project_body(x_half, y_half, z),
            project_body(-x_half, y_half, z),
        ]

        painter.setPen(self._Qt.PenStyle.NoPen)
        painter.setBrush(self._color("#25303A", 115))
        painter.drawPolygon(self._QPolygonF(corners))

        grid_pen = self._QPen(self._color("#9EB2C2", 55))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        painter.setBrush(self._Qt.BrushStyle.NoBrush)

        grid_steps = 4
        for step in range(-grid_steps, grid_steps + 1):
            x = x_half * step / grid_steps
            painter.drawLine(project_body(x, -y_half, z), project_body(x, y_half, z))

            y = y_half * step / grid_steps
            painter.drawLine(project_body(-x_half, y, z), project_body(x_half, y, z))

        outline_pen = self._QPen(self._color("#C6D7E0", 110))
        outline_pen.setWidth(2)
        outline_pen.setJoinStyle(self._Qt.PenJoinStyle.RoundJoin)
        painter.setPen(outline_pen)
        painter.drawPolygon(self._QPolygonF(corners))

    def _paint(self, painter: Any, width: int, height: int) -> None:
        canvas_w = max(400, int(width))
        canvas_h = max(320, int(height))
        cx = canvas_w * 0.5
        cy = canvas_h * 0.58
        scale = min(canvas_w * 0.28, canvas_h * 0.34)

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

        matrix = pose.rotation_matrix
        if matrix is None:
            matrix = self._rot_matrix(pose.roll_deg, pose.pitch_deg, pose.yaw_deg)
        bx = (matrix[0][0], matrix[1][0], matrix[2][0])
        by = (matrix[0][1], matrix[1][1], matrix[2][1])
        bz = (matrix[0][2], matrix[1][2], matrix[2][2])
        self._draw_reference_plane(painter, matrix, cx, cy, scale)

        painter.setPen(self._Qt.PenStyle.NoPen)
        painter.setBrush(self._QColor("#FFFFFF"))
        painter.drawEllipse(self._QPointF(cx, cy), 4.0, 4.0)

        self._draw_axis_line(painter, bx, "#FF5C5C", 4, cx, cy, scale, "Bx")
        self._draw_axis_line(painter, by, "#AAAAAA", 4, cx, cy, scale, "By")
        self._draw_axis_line(painter, bz, "#60FF60", 4, cx, cy, scale, "Bz")
