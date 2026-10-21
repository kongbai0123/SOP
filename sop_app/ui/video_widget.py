"""影像顯示元件：等比例縮放顯示畫面、疊上偵測區域，並支援拖曳框選新區域。"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..sop_schema import ROI
from ..overlay import class_color


class VideoWidget(QWidget):
    roi_drawn = Signal(list)       # 正規化座標的四個角點
    roi_selected = Signal(str)
    roi_edited = Signal(str, list)
    roi_rename_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._detections = []
        self._classes = ()
        self._placeholder = "尚未連線影像來源"
        self._rois: list[ROI] = []
        self._regions_visible = True
        self._highlight: set[str] = set()
        self._stale: set[str] = set()
        self._selected: str | None = None
        self._banner = ""
        self._banner_color = QColor("#455a64")
        self._banner_single_line = False
        self._alert = ""
        self._alert_regions: set[str] = set()
        self._draw_mode = False
        self._drag_start: QPointF | None = None
        self._drag_end: QPointF | None = None
        self._editable = set()
        self._edit_drag = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ---- 外部設定 --------------------------------------------------------------
    def set_frame(self, bgr: np.ndarray, detections=(), classes=(), min_score=0.3):
        self._detections = [d for d in detections if d.score >= min_score]
        self._classes = classes
        bgr = np.ascontiguousarray(bgr)
        height, width = bgr.shape[:2]
        self._image = QImage(bgr.data, width, height, bgr.strides[0], QImage.Format.Format_BGR888).copy()
        self.update()

    def clear_frame(self, placeholder: str = "尚未連線影像來源"):
        self._detections = []
        self._image, self._placeholder = None, placeholder
        self.update()

    def set_rois(self, rois: Iterable[ROI], highlight: Iterable[str] = (), selected: str | None = None,
                 stale: Iterable[str] = ()):
        self._rois, self._highlight, self._selected = list(rois), set(highlight), selected
        self._stale = set(stale)
        self.update()

    def set_banner(self, text: str, color: str = "#455a64", *, single_line=False):
        if text != self._banner or QColor(color) != self._banner_color or bool(single_line) != self._banner_single_line:
            self._banner, self._banner_color = text, QColor(color)
            self._banner_single_line = bool(single_line)
            self.setToolTip(text if self._banner_single_line else "")
            self.update()

    def set_alert(self, text: str, regions: Iterable[str] = ()):
        """Overlay an alert without changing the source frame or locator banner."""
        text = text.strip()
        regions = set(regions) if text else set()
        if text != self._alert or regions != self._alert_regions:
            self._alert, self._alert_regions = text, regions
            self.update()

    @property
    def regions_visible(self):
        return self._regions_visible

    def set_regions_visible(self, visible: bool):
        visible = bool(visible)
        if self._regions_visible != visible:
            self._regions_visible = visible
            self._edit_drag = None
            self.update()

    def set_draw_mode(self, enabled: bool):
        self._draw_mode = enabled
        self._drag_start = self._drag_end = None
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.update()

    def set_editable(self, names):
        self._editable = set(names)
        self._edit_drag = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._edit_drag = None
            self._drag_start = self._drag_end = None
            self.update()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._regions_visible and self._selected in self._editable and not self._draw_mode:
            self.roi_rename_requested.emit(self._selected)

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
            painter.drawText(self.rect().adjusted(12, 12, -12, -12),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap | Qt.TextFlag.TextWrapAnywhere,
                             self._placeholder)
            self._draw_alert(painter, QRectF(self.rect()))
            return

        rect = self._image_rect()
        painter.drawImage(rect, self._image)
        label_font = QFont(self.font().family(), 10, QFont.Weight.Bold)

        for roi in self._rois:
            alarmed = bool(self._alert) and roi.name in self._alert_regions
            if not self._regions_visible and not alarmed:
                continue
            points = self._edit_drag['preview'] if self._edit_drag and self._edit_drag['name'] == roi.name else roi.points
            polygon = QPolygonF([self._to_widget(rect, x, y) for x, y in points])
            if alarmed:
                pen, fill = QPen(QColor("#ff5252"), 3), QColor(255, 82, 82, 45)
            elif roi.name in self._stale:
                pen, fill = QPen(QColor("#ff9800"), 2, Qt.PenStyle.DashLine), QColor(255, 152, 0, 25)
            elif roi.name == self._selected:
                pen, fill = QPen(QColor("#00e5ff"), 3), QColor(0, 229, 255, 40)
            elif roi.name in self._highlight:
                pen, fill = QPen(QColor("#ffd400"), 3), QColor(255, 212, 0, 35)
            else:
                color = QColor.fromHsv(sum(ord(c) for c in roi.name) * 37 % 360, 150, 255)
                pen, fill = QPen(color, 2), QColor(0, 0, 0, 0)
            painter.setPen(pen)
            painter.setBrush(QBrush(fill))
            painter.drawPolygon(polygon)
            if self._regions_visible and roi.name == self._selected and roi.name in self._editable:
                painter.setBrush(QColor('white'))
                for point in polygon:
                    painter.drawRect(QRectF(point.x() - 5, point.y() - 5, 10, 10))
            tag = roi.name
            self._draw_tag(painter, polygon.boundingRect().topLeft() + QPointF(4, 4), tag,
                           pen.color(), label_font)

        if self._drag_start is not None and self._drag_end is not None:
            painter.setPen(QPen(QColor("#00e5ff"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(0, 229, 255, 40))
            painter.drawRect(QRectF(self._drag_start, self._drag_end).normalized())

        if self._banner and not self._alert:
            banner_font = QFont(self.font().family(), 16, QFont.Weight.Bold)
            text = self._banner
            flags = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
            if self._banner_single_line:
                painter.setFont(banner_font)
                text = painter.fontMetrics().elidedText(" ".join(text.split()), Qt.TextElideMode.ElideRight,
                                                       max(1, int(rect.width() - 24)))
                height = painter.fontMetrics().height() + 16
                flags |= Qt.TextFlag.TextSingleLine
            else:
                bounds = QRectF(0, 0, rect.width(), self.height()).adjusted(12, 8, -12, -8)
                banner_font, measured = self._fit_text(painter, text, banner_font, bounds)
                painter.setFont(banner_font)
                height = measured.height() + 16
                flags |= Qt.TextFlag.TextWordWrap | Qt.TextFlag.TextWrapAnywhere
            # 上方有黑邊時把橫幅放在黑邊裡，不遮住畫面內容
            top = rect.y() - height if rect.y() >= height else min(rect.y(), self.height() - height)
            band = QRectF(rect.x(), top, rect.width(), height)
            color = QColor(self._banner_color)
            color.setAlpha(220)
            painter.fillRect(band, color)
            painter.setPen(QColor("white"))
            painter.drawText(band.adjusted(12, 0, -12, 0), flags, text)

        if self._draw_mode:
            painter.setFont(label_font)
            self._draw_tag(painter, QPointF(rect.x() + 8, rect.bottom() - 30), "拖曳滑鼠框選偵測區域",
                           QColor("#00e5ff"), label_font)

        # 辨識框與名稱最後繪製，避免被區域填色、標籤或其他遮罩蓋住。
        painter.save()
        painter.setClipRect(rect)
        for det in self._detections:
            b, g, r = class_color(det.label, self._classes)
            color = QColor(r, g, b)
            x1, y1, x2, y2 = det.box
            box = QRectF(self._to_widget(rect, x1 / self._image.width(), y1 / self._image.height()),
                         self._to_widget(rect, x2 / self._image.width(), y2 / self._image.height()))
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(box)
            self._draw_tag(painter, QPointF(box.left(), max(rect.top(), box.top() - 22)),
                           f"{det.label} {det.score:.2f}", color, label_font)
        painter.restore()
        # 告警最後繪製，辨識框與標籤不得蓋住需要立即看見的警示。
        self._draw_alert(painter, rect)

    def _draw_alert(self, painter: QPainter, rect: QRectF):
        if not self._alert:
            return
        font = QFont(self.font().family(), 16, QFont.Weight.Bold)
        bounds = QRectF(0, 0, rect.width(), self.height()).adjusted(12, 8, -12, -8)
        font, measured = self._fit_text(painter, self._alert, font, bounds)
        height = min(self.height(), measured.height() + 16)
        top = rect.y() - height if rect.y() >= height else min(rect.y(), self.height() - height)
        band = QRectF(rect.x(), top, rect.width(), height)
        painter.fillRect(band, QColor(183, 28, 28, 242))
        painter.setFont(font)
        painter.setPen(QColor("white"))
        painter.drawText(band.adjusted(12, 8, -12, -8),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter |
                         Qt.TextFlag.TextWordWrap | Qt.TextFlag.TextWrapAnywhere, self._alert)

    @staticmethod
    def _fit_text(painter, text, font, bounds):
        font = QFont(font)
        flags = Qt.TextFlag.TextWordWrap | Qt.TextFlag.TextWrapAnywhere
        while True:
            painter.setFont(font)
            measured = painter.fontMetrics().boundingRect(
                QRect(0, 0, max(1, int(bounds.width())), 100000), flags, text)
            if measured.height() <= bounds.height() or font.pointSizeF() <= 7:
                return font, measured
            font.setPointSizeF(max(7, font.pointSizeF() - .5))

    @staticmethod
    def _draw_tag(painter: QPainter, origin: QPointF, text: str, color: QColor, font: QFont):
        bounds = QRectF(painter.viewport()).adjusted(6, 6, -6, -6)
        font, measured = VideoWidget._fit_text(painter, text, font, bounds.adjusted(6, 4, -6, -4))
        painter.setFont(font)
        width, height = min(bounds.width(), measured.width() + 12), measured.height() + 8
        x = max(bounds.left(), min(origin.x(), bounds.right() - width))
        y = max(bounds.top(), min(origin.y(), bounds.bottom() - height))
        box = QRectF(x, y, width, height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(color)
        painter.drawText(box.adjusted(6, 4, -6, -4), Qt.AlignmentFlag.AlignLeft |
                         Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap | Qt.TextFlag.TextWrapAnywhere, text)
        return box

    # ---- 框選 ------------------------------------------------------------------
    def mousePressEvent(self, event):
        self.setFocus()
        if self._regions_visible and not self._draw_mode and self._editable and event.button() == Qt.MouseButton.LeftButton:
            rect = self._image_rect()
            if rect.isEmpty() or not rect.contains(event.position()):
                return
            candidates = sorted(self._rois, key=lambda r: r.name == self._selected, reverse=True)
            for roi in candidates:
                if roi.name not in self._editable:
                    continue
                polygon = QPolygonF([self._to_widget(rect, x, y) for x, y in roi.points])
                corner = next((i for i, p in enumerate(polygon)
                               if (p - event.position()).manhattanLength() <= 14), None)
                if corner is not None or polygon.containsPoint(event.position(), Qt.FillRule.OddEvenFill):
                    self.roi_selected.emit(roi.name)
                    self._edit_drag = dict(name=roi.name, original=list(roi.points), preview=list(roi.points),
                                           corner=corner, start=self._to_normalized(rect, event.position()))
                    return
        if self._draw_mode and event.button() == Qt.MouseButton.LeftButton \
                and self._image_rect().contains(event.position()):
            self._drag_start = self._drag_end = event.position()
            self.update()

    def mouseMoveEvent(self, event):
        if self._edit_drag:
            drag = self._edit_drag
            x, y = self._to_normalized(self._image_rect(), event.position())
            points = drag['original']
            if drag['corner'] is None:
                dx = min(max(x - drag['start'][0], -min(p[0] for p in points)), 1 - max(p[0] for p in points))
                dy = min(max(y - drag['start'][1], -min(p[1] for p in points)), 1 - max(p[1] for p in points))
                drag['preview'] = [(a + dx, b + dy) for a, b in points]
            else:
                ox, oy = points[(drag['corner'] + 2) % 4]
                if abs(x - ox) > .01 and abs(y - oy) > .01:
                    x1, x2 = sorted((x, ox))
                    y1, y2 = sorted((y, oy))
                    drag['preview'] = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
            self.update()
            return
        if self._drag_start is not None:
            self._drag_end = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._edit_drag and event.button() == Qt.MouseButton.LeftButton:
            self.mouseMoveEvent(event)
            drag, self._edit_drag = self._edit_drag, None
            if drag['preview'] != drag['original']:
                self.roi_edited.emit(drag['name'], drag['preview'])
            self.update()
            return
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
