"""Make the editor's wheel scroll its panes instead of changing settings."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QLineEdit, QScrollArea,
                               QWidget)


class EditorWheelGuard(QObject):
    """Protect current and future spin/combo fields within one editor only.

    An application filter catches a spin box's internal line edit as well as
    dynamically rebuilt fields without repeatedly scanning/registering the tree.
    It is installed only while the editor is visible; retained hidden pages do
    not add work to unrelated events. Qt also removes it on destruction.
    """

    def __init__(self, root: QWidget):
        super().__init__(root)
        self.root = root
        self._forwarding = False
        self._active = False
        root.installEventFilter(self)
        # Qt can assign wheel focus before delivering the wheel event. Remove
        # only that capability before input, leaving click/tab/focus intact.
        self._protect_existing_fields()
        if root.isVisible():
            self._activate()

    def _protect_existing_fields(self):
        for widget in self.root.findChildren(QWidget):
            self._protect_focus_policy(widget)

    def _activate(self):
        if self._active:
            return
        # Hidden pages can rebuild fields while no global filter is installed.
        # Protect them before the first user input on each subsequent show.
        self._protect_existing_fields()
        QApplication.instance().installEventFilter(self)
        self._active = True

    def _deactivate(self):
        if self._active:
            QApplication.instance().removeEventFilter(self)
            self._active = False

    def _field_and_scroll(self, watched):
        if not isinstance(watched, QWidget):
            return None, None
        # A dialog may have the editor as its QObject parent while belonging
        # to a different window. Its controls keep their own wheel behavior.
        if watched.window() is not self.root.window():
            return None, None

        field = scroll = None
        ancestor = watched
        while ancestor is not None and ancestor is not self.root:
            if field is None and isinstance(ancestor, (QAbstractSpinBox, QComboBox)):
                field = ancestor
            if field is not None and scroll is None and isinstance(ancestor, QScrollArea):
                scroll = ancestor
            ancestor = ancestor.parentWidget()
        if ancestor is not self.root or field is None:
            return None, None
        return field, scroll

    @staticmethod
    def _remove_wheel_focus(widget):
        # WheelFocus contains StrongFocus; masking the entire enum would also
        # remove click/tab focus. Compute the wheel-only bit from Qt itself.
        wheel_only = int(Qt.FocusPolicy.WheelFocus) & ~int(Qt.FocusPolicy.StrongFocus)
        original = int(widget.focusPolicy())
        protected = original & ~wheel_only
        if protected != original:
            widget.setFocusPolicy(Qt.FocusPolicy(protected))

    def _protect_focus_policy(self, watched):
        if not isinstance(watched, (QAbstractSpinBox, QComboBox, QLineEdit)):
            return
        field, _scroll = self._field_and_scroll(watched)
        if field is None:
            return
        self._remove_wheel_focus(field)
        if isinstance(field, QAbstractSpinBox) or field.isEditable():
            line = field.lineEdit()
            if line is not None:
                self._remove_wheel_focus(line)

    def eventFilter(self, watched, event):
        kind = event.type()
        if watched is self.root:
            if kind == QEvent.Type.Show:
                self._activate()
            elif kind == QEvent.Type.Hide:
                self._deactivate()
        if kind != QEvent.Type.Wheel:
            # New/reparented fields are protected before native wheel focus
            # processing. Unrelated frame/layout/paint events do no tree work.
            if kind in (QEvent.Type.Polish, QEvent.Type.Show, QEvent.Type.ParentChange):
                self._protect_focus_policy(watched)
            return False
        if self._forwarding:
            return False
        field, scroll = self._field_and_scroll(watched)
        if field is None:
            return False
        # An open dropdown owns its normal native scrolling, including its
        # view/viewport children. Plain text and log widgets never match above.
        if isinstance(field, QComboBox) and field.view().isVisible():
            return False

        if scroll is not None:
            viewport = scroll.viewport()
            local = event.globalPosition() - QPointF(viewport.mapToGlobal(QPoint()))
            forwarded = QWheelEvent(local, event.globalPosition(), event.pixelDelta(),
                                    event.angleDelta(), event.buttons(), event.modifiers(),
                                    event.phase(), event.inverted(), event.source(),
                                    event.pointingDevice())
            forwarded.setTimestamp(event.timestamp())
            self._forwarding = True
            try:
                QApplication.sendEvent(viewport, forwarded)
            finally:
                self._forwarding = False
        # Consume the original even at a pane boundary or without a containing
        # pane: focus must never make a wheel edit the field's value.
        event.accept()
        return True


def install_editor_wheel_guard(root: QWidget) -> EditorWheelGuard:
    if not hasattr(root, "_wheel_guard"):
        root._wheel_guard = EditorWheelGuard(root)
    return root._wheel_guard
