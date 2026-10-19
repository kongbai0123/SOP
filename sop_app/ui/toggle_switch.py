"""Keyboard-accessible switch with an animated sliding thumb."""
from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QCheckBox


class ToggleSwitch(QCheckBox):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._position = 0.0
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName(text)
        self.setMinimumHeight(28)
        self.animation = QPropertyAnimation(self, b"position", self)
        self.animation.setDuration(150)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.toggled.connect(self._animate)

    def _animate(self, checked):
        self.animation.stop()
        self.animation.setStartValue(self._position)
        self.animation.setEndValue(1.0 if checked else 0.0)
        self.animation.start()

    def _set_position(self, value):
        self._position = value
        self.update()

    position = Property(float, lambda self: self._position, _set_position)

    def sizeHint(self):
        size = super().sizeHint()
        size.setWidth(self.fontMetrics().horizontalAdvance(self.text()) + 56)
        size.setHeight(max(28, size.height()))
        return size

    def hitButton(self, pos):
        return self.rect().contains(pos)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(1.0 if self.isEnabled() else .4)
        y = (self.height() - 22) / 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#2563eb" if self.isChecked() else "#64748b"))
        painter.drawRoundedRect(QRectF(1, y, 40, 22), 11, 11)
        painter.setBrush(QColor("white"))
        painter.drawEllipse(QRectF(4 + 18 * self._position, y + 3, 16, 16))
        painter.setPen(self.palette().windowText().color())
        painter.drawText(self.rect().adjusted(49, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, self.text())
        if self.hasFocus():
            painter.setPen(self.palette().highlight().color())
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 4, 4)
