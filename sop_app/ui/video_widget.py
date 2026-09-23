"""影像顯示元件：等比例縮放顯示畫面、疊上偵測區域，並支援拖曳框選新區域。"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..sop_schema import ROI


class VideoWidget(QWidget):
    roi_drawn = Signal(list)       # 正規化座標的四個角點

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._placeholder = "尚未連線影像來源"
        self._rois: list[ROI] = []
        self._highlight: set[str] = set()
        self._selected: str | None = None
        self._banner = ""
        self._banner_color = QColor("#455a64")
        self._draw_mode = False
        self._drag_start: QPointF | None = None
        self._drag_end: QPointF | None = None
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ---- 外部設定 --------------------------------------------------------------
    def set_frame(self, bgr: np.ndarray):
        bgr = np.ascontiguousarray(bgr)
        height, width = bgr.shape[:2]
        self._image = QImage(bgr.data, width, height, bgr.strides[0], QImage.Format.Format_BGR888).copy()
        self.update()

    def clear_frame(self, placeholder: str = "尚未連線影像來源"):
        self._image, self._placeholder = None, placeholder
        self.update()

    def set_rois(self, rois: Iterable[ROI], highlight: Iterable[str] = (), selected: str | None = None):
        self._rois, self._highlight, self._selected = list(rois), set(highlight), selected
        self.update()

    def set_banner(self, text: str, color: str = "#455a64"):
        if text != self._banner or QColor(color) != self._banner_color:
            self._banner, self._banner_color = text, QColor(color)
            self.update()

    def set_draw_mode(self, enabled: bool):
        self._draw_mode = enabled
        self._drag_start = self._drag_end = None
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.update()

    # ---- 座標轉換 --------------------------------------------------------------
    def _image_rect(self) -> QRectF:
        if self._image is None:
            return QRectF()
        scale = min(self.width() / self._image.width(), self.height() / self._image.height())
        width, height = self._image.width() * scale, self._image.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    @staticmethod
    def _to_widget(rect: QRectF, x: float, y: float) -> QPointF:
        return QPointF(rect.x() + x * rect.width(), rect.y() + y * rect.height())

    @staticmethod
    def _to_normalized(rect: QRectF, point: QPointF) -> tuple[float, float]:
        x = (point.x() - rect.x()) / rect.width()
        y = (point.y() - rect.y()) / rect.height()
        return round(min(max(x, 0.0), 1.0), 4), round(min(max(y, 0.0), 1.0), 4)

    # ---- 繪製 ------------------------------------------------------------------
    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#1b1f23"))
        if self._image is None:
            painter.setPen(QColor("#8a949e"))
            painter.setFont(QFont(self.font().family(), 13))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return

        rect = self._image_rect()
        painter.drawImage(rect, self._image)
        label_font = QFont(self.font().family(), 10, QFont.Weight.Bold)

        for roi in self._rois:
            polygon = QPolygonF([self._to_widget(rect, x, y) for x, y in roi.points])
            if roi.name == self._selected:
                pen, fill = QPen(QColor("#00e5ff"), 3), QColor(0, 229, 255, 40)
            elif roi.name in self._highlight:
                pen, fill = QPen(QColor("#ffd400"), 3), QColor(255, 212, 0, 35)
            else:
                pen, fill = QPen(QColor(255, 255, 255, 200), 1.5, Qt.PenStyle.DashLine), QColor(0, 0, 0, 0)
            painter.setPen(pen)
            painter.setBrush(QBrush(fill))
            painter.drawPolygon(polygon)
            self._draw_tag(painter, polygon.boundingRect().topLeft() + QPointF(4, 4), roi.name,
                           pen.color(), label_font)

        if self._drag_start is not None and self._drag_end is not None:
            painter.setPen(QPen(QColor("#00e5ff"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(0, 229, 255, 40))
            painter.drawRect(QRectF(self._drag_start, self._drag_end).normalized())

        if self._banner:
            banner_font = QFont(self.font().family(), 16, QFont.Weight.Bold)
            painter.setFont(banner_font)
            height = painter.fontMetrics().height() + 16
            # 上方有黑邊時把橫幅放在黑邊裡，不遮住畫面內容
            top = rect.y() - height if rect.y() >= height else rect.y()
            band = QRectF(rect.x(), top, rect.width(), height)
            color = QColor(self._banner_color)
            color.setAlpha(220)
            painter.fillRect(band, color)
            painter.setPen(QColor("white"))
            painter.drawText(band.adjusted(12, 0, -12, 0),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._banner)

        if self._draw_mode:
            painter.setFont(label_font)
            self._draw_tag(painter, QPointF(rect.x() + 8, rect.bottom() - 30), "拖曳滑鼠框選偵測區域",
                           QColor("#00e5ff"), label_font)

    @staticmethod
    def _draw_tag(painter: QPainter, origin: QPointF, text: str, color: QColor, font: QFont):
        painter.setFont(font)
        metrics = painter.fontMetrics()
        box = QRectF(origin, QPointF(origin.x() + metrics.horizontalAdvance(text) + 10,
                                     origin.y() + metrics.height() + 4))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(color)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    # ---- 框選 ------------------------------------------------------------------
    def mousePressEvent(self, event):
        if self._draw_mode and event.button() == Qt.MouseButton.LeftButton \
                and self._image_rect().contains(event.position()):
            self._drag_start = self._drag_end = event.position()
            self.update()

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            self._drag_end = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._drag_start is None or event.button() != Qt.MouseButton.LeftButton:
            return
        rect = self._image_rect()
        (x1, y1), (x2, y2) = self._to_normalized(rect, self._drag_start), self._to_normalized(rect, event.position())
        self._drag_start = self._drag_end = None
        self.update()
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 > 0.01 and y2 - y1 > 0.01:
            self.roi_drawn.emit([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
