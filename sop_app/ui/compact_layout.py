"""Compact native layout helpers: wrapping tool rows, auto-height notes,
size-accurate spin boxes and an elided one-line status text."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QDoubleSpinBox, QLabel, QLayout, QPlainTextEdit, QSizePolicy,
                               QSpinBox, QStyle, QStyleOptionSpinBox, QWidget, QWidgetItem)


def relayout_now(root: QWidget):
    """Recompute nested layouts in one pass instead of one event-loop pass per level.

    Children are visited before their parents so every container's cached
    size hint is dropped before the root layout asks for it again.
    """
    def invalidate(layout):
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item.layout() is not None:  # nested sub-layouts keep their own cache
                invalidate(item.layout())
            item.invalidate()
        layout.invalidate()

    for widget in reversed(root.findChildren(QWidget)):
        if widget.layout() is not None:
            invalidate(widget.layout())
        widget.updateGeometry()
    if root.layout() is not None:
        invalidate(root.layout())
        root.layout().activate()


def text_label(text="", object_name="", wrap=True) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(wrap)
    label.setMinimumWidth(0)
    if object_name:
        label.setObjectName(object_name)
    label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    return label


def set_state(widget: QWidget, state: str):
    """Restyle a widget whose colors depend on a dynamic ``state`` property."""
    if widget.property("state") != state:
        widget.setProperty("state", state)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class FlowLayout(QLayout):
    """Left-to-right controls that wrap to the next line instead of clipping."""

    def __init__(self, parent=None, spacing=4):
        super().__init__(parent)
        self._items: list[QWidgetItem] = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _arrange(self, rect, apply):
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, line = area.x(), area.y(), 0
        for item in self._items:
            if item.widget() is not None and item.widget().isHidden():
                continue
            hint = item.sizeHint().expandedTo(item.minimumSize())
            if x > area.x() and x + hint.width() > area.right() + 1:
                x, y, line = area.x(), y + line + self._spacing, 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._spacing
            line = max(line, hint.height())
        return y + line - rect.y() + margins.bottom()


class _CompactSpin:
    """Size from the longest value the field can show instead of Qt's generous default."""
    def _value_width(self):
        metrics = self.fontMetrics()
        texts = [self.prefix() + self.textFromValue(value) + self.suffix()
                 for value in (self.minimum(), self.maximum())]
        if self.specialValueText():
            texts.append(self.specialValueText())
        # Measure the style's own frame, padding and arrow area at a probe width.
        self.ensurePolished()
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        option.rect.setWidth(200)
        option.rect.setHeight(max(self.height(), 30))
        field = self.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, option,
                                            QStyle.SubControl.SC_SpinBoxEditField, self)
        chrome = 200 - field.width() if 0 < field.width() < 200 else 40
        return max(metrics.horizontalAdvance(text) for text in texts) + chrome + 6

    def sizeHint(self):
        return QSize(self._value_width(), super().sizeHint().height())

    def minimumSizeHint(self):
        return QSize(self._value_width(), super().minimumSizeHint().height())


class CompactSpinBox(_CompactSpin, QSpinBox):
    pass


class CompactDoubleSpinBox(_CompactSpin, QDoubleSpinBox):
    pass


class CompactLog(QPlainTextEdit):
    """Read-only event list that fills spare height but does not demand it."""

    def sizeHint(self):
        return QSize(super().sizeHint().width(), self.minimumHeight() or 60)


class AutoHeightTextEdit(QPlainTextEdit):
    """Multi-line note that grows with its text, so nothing is hidden."""

    def __init__(self, text="", max_lines=6, parent=None):
        super().__init__(text, parent)
        self.max_lines = max_lines
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.document().setDocumentMargin(1)
        self.document().documentLayout().documentSizeChanged.connect(lambda _size: self._fit())
        self._fit()

    def _fit(self):
        lines = max(1, int(self.document().documentLayout().documentSize().height()))
        shown = min(lines, self.max_lines)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if lines > self.max_lines
                                        else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        margin = round(self.document().documentMargin())
        # Frame and style-sheet padding: measured once laid out, estimated before.
        chrome = (self.height() - self.viewport().height() if self.isVisible() and self.viewport().height() > 0
                  else self.frameWidth() * 2 + 6)
        height = shown * self.fontMetrics().lineSpacing() + 2 * margin + chrome + 2
        if self.height() != height or self.minimumHeight() != height:
            self.setFixedHeight(height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._fit()


class StatusText(QWidget):
    """One-line status text for the bottom bar; long text is elided in the middle.

    The middle is removed so both the model name and the compute device remain
    readable. The complete text is kept as the tooltip and via ``text()``.
    """

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._text = ""
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setText(text)

    def text(self):
        return self._text

    def setText(self, text):
        text = str(text).replace("\n", " ")
        if text != self._text:
            self._text = text
            self.setToolTip(text)
            self.updateGeometry()
            self.update()

    def sizeHint(self):
        metrics = self.fontMetrics()
        return QSize(metrics.horizontalAdvance(self._text) + 8, metrics.height() + 4)

    def minimumSizeHint(self):
        metrics = self.fontMetrics()
        return QSize(min(self.sizeHint().width(), metrics.horizontalAdvance("模型：…") + 8), metrics.height() + 4)

    def paintEvent(self, _event):
        painter = QPainter(self)
        rect = self.contentsRect().adjusted(4, 0, -4, 0)
        text = self.fontMetrics().elidedText(self._text, Qt.TextElideMode.ElideMiddle, rect.width())
        self.style().drawItemText(painter, rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                                  self.palette(), True, text, self.foregroundRole())
