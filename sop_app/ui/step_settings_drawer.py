"""One step's existing form, temporarily hosted over the editor workspace."""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QFrame, QHBoxLayout,
                               QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from .workbench_widgets import workbench_palette


class StepSettingsDrawer(QFrame):
    """A child overlay: opening the form never takes width from the video.

    The row remains the sole owner of the controls and Step data. Borrowing its
    form avoids two editors overwriting each other's values on a later rename.
    Closing returns both widgets before a row can be rebuilt or destroyed.
    """

    closed = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("stepSettingsOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.current_editor = None
        self._borrowed = []
        self._filter_active = False
        self.panel = QFrame(self)
        self.panel.setObjectName("stepSettingsPanel")
        self.panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)
        header = QHBoxLayout()
        self.title = QLabel()
        self.title.setTextFormat(Qt.TextFormat.PlainText)
        self.title.setWordWrap(True)
        self.title.setObjectName("stepSettingsTitle")
        header.addWidget(self.title, 1)
        self.close_button = QPushButton("完成編輯")
        self.close_button.setObjectName("primaryButton")
        self.close_button.clicked.connect(self.close)
        header.addWidget(self.close_button)
        layout.addLayout(header)
        self.save_hint = QLabel("修改會即時保留；請使用上方「儲存」寫入 SOP 檔案。")
        self.save_hint.setWordWrap(True)
        self.save_hint.setObjectName("mutedLabel")
        layout.addWidget(self.save_hint)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.document = QWidget()
        self.document.setObjectName("stepSettingsDocument")
        self.form_layout = QVBoxLayout(self.document)
        self.form_layout.setContentsMargins(0, 0, 6, 8)
        self.form_layout.setSpacing(0)
        self.scroll.setWidget(self.document)
        layout.addWidget(self.scroll, 1)
        self.actions_layout = QVBoxLayout()
        self.actions_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.actions_layout)
        parent.installEventFilter(self)
        self.hide()

    def set_theme(self, dark):
        colors = workbench_palette(dark)
        self.setStyleSheet("""
            QFrame#stepSettingsOverlay { background:rgba(12, 20, 30, 95); }
            QFrame#stepSettingsPanel { background:%(panel)s; border-left:1px solid %(border)s; }
            QWidget#stepSettingsDocument { background:%(panel)s; color:%(fg)s; }
            QLabel#stepSettingsTitle { font-size:16px; font-weight:600; color:%(fg)s; }
            QFrame[stepSettingsSection="true"] { background:%(panel)s; border:1px solid %(border)s; }
            QLabel#groupLabel { font-size:14px; font-weight:600; color:%(fg)s; }
            QLabel#groupLabel[role="violation"] { color:%(danger)s; }
        """ % colors)

    def reposition(self):
        parent = self.parentWidget()
        self.setGeometry(parent.rect())
        width = min(self.width(), max(520, min(760, round(self.width() * .54))))
        self.panel.setGeometry(self.width() - width, 0, width, self.height())
        self.raise_()

    def open_editor(self, editor):
        if self.current_editor is editor:
            self.show()
            self.reposition()
            return
        self._release_editor()
        self.current_editor = editor
        for widget, target_layout in ((editor.body, self.form_layout),
                                      (editor.actions_row, self.actions_layout)):
            parent = widget.parentWidget()
            source_layout = parent.layout()
            index = source_layout.indexOf(widget)
            self._borrowed.append((widget, parent, source_layout, index))
            source_layout.removeWidget(widget)
            target_layout.addWidget(widget)
        editor.changed.connect(self.refresh_title)
        editor.set_details_open(True)
        self.refresh_title()
        self.scroll.verticalScrollBar().setValue(0)
        self.show()
        self.reposition()
        self.close_button.setFocus()

    def refresh_title(self):
        if self.current_editor is not None:
            editor = self.current_editor
            self.title.setText(f"工序 {editor.number_label.text()} · {editor.step.name}")

    def _release_editor(self):
        editor = self.current_editor
        if editor is None:
            return
        # Escape and structural changes also commit pending numeric text, just
        # like clicking the close button would by transferring focus.
        for spin in editor.body.findChildren(QAbstractSpinBox):
            spin.interpretText()
        editor.changed.disconnect(self.refresh_title)
        self.current_editor = None
        for widget, parent, source_layout, index in self._borrowed:
            self.form_layout.removeWidget(widget)
            self.actions_layout.removeWidget(widget)
            widget.setParent(parent)
            source_layout.insertWidget(index, widget)
        self._borrowed.clear()
        editor.set_details_open(False)
        editor.edit_button.setFocus()
        self.closed.emit()

    def showEvent(self, event):
        if not self._filter_active:
            QApplication.instance().installEventFilter(self)
            self._filter_active = True
        super().showEvent(event)

    def hideEvent(self, event):
        if self._filter_active:
            QApplication.instance().removeEventFilter(self)
            self._filter_active = False
        self._release_editor()
        super().hideEvent(event)

    def closeEvent(self, event):
        self._release_editor()
        super().closeEvent(event)

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() == QEvent.Type.Resize:
            if self.isVisible():
                self.reposition()
        if (event.type() != QEvent.Type.KeyPress or not self.isVisible()
                or not isinstance(watched, QWidget) or not self.isAncestorOf(watched)
                or QApplication.activePopupWidget() is not None):
            return False
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return True
        if event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            forward = (event.key() != Qt.Key.Key_Backtab
                       and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            candidate = watched
            seen = set()
            while id(candidate) not in seen:
                seen.add(id(candidate))
                candidate = candidate.nextInFocusChain() if forward else candidate.previousInFocusChain()
                if (self.isAncestorOf(candidate) and candidate.isVisible() and candidate.isEnabled()
                        and candidate.focusPolicy() & Qt.FocusPolicy.TabFocus):
                    candidate.setFocus(Qt.FocusReason.TabFocusReason if forward else Qt.FocusReason.BacktabFocusReason)
                    return True
        return False
