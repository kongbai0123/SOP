"""Shared native widgets and the compact industrial style sheet."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QApplication, QComboBox, QLabel, QStyle,
                              QStyleOptionComboBox, QScrollArea, QStackedWidget, QWidget)


def _invalidate_tree(layout):
    """Drop cached sizes of a widget's layout, including nested sub-layouts."""
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item.layout() is not None:
            _invalidate_tree(item.layout())
        item.invalidate()
    layout.invalidate()


def relayout_ancestors(widgets, top):
    """Settle the parents of resized widgets now, bottom-up, instead of one
    event-loop pass per nesting level (which shows a squeezed frame meanwhile)."""
    chains, seen = [], set()
    for widget in widgets:
        chain, current = [], widget.parentWidget()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append(current)
            if current is top:
                break
            current = current.parentWidget()
        chains.append(chain)
    for chain in chains:
        for ancestor in chain:
            if ancestor.layout() is not None:
                _invalidate_tree(ancestor.layout())
                ancestor.layout().activate()
            elif isinstance(ancestor, QScrollArea):
                QApplication.sendEvent(ancestor, QEvent(QEvent.Type.LayoutRequest))
            ancestor.updateGeometry()


class ReadabilityGuard(QObject):
    """Keep intrinsic control sizes after polishing, text changes and DPI changes.

    Scrollable containers handle overflow; shrinking the contents of a button is
    never the fallback. A coalesced layout-only timer does not run per video frame.
    """
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.sync)
        self.root.installEventFilter(self)
        self.schedule()

    def schedule(self, *_args):
        if not self.timer.isActive():
            self.timer.start(0)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.Show, QEvent.Type.LayoutRequest, QEvent.Type.Resize,
                            QEvent.Type.FontChange, QEvent.Type.StyleChange, QEvent.Type.ChildAdded):
            self.schedule()
        return False

    def sync(self):
        changed = []
        for widget in self.root.findChildren(QWidget):
            if widget.window() != self.root.window():
                continue
            if isinstance(widget, QLabel) and widget.wordWrap() and widget.isVisible():
                widget.installEventFilter(self)
                if widget.parentWidget() is not None:
                    widget.parentWidget().installEventFilter(self)
                if widget.property("wrapBaseMinimum") is None:
                    widget.setProperty("wrapBaseMinimum", widget.minimumHeight())
                height = max(widget.property("wrapBaseMinimum"), widget.heightForWidth(widget.width()))
                if widget.minimumHeight() != height:
                    widget.setMinimumHeight(height)
                    changed.append(widget)
                continue
            if not isinstance(widget, (QAbstractButton, QComboBox, QAbstractSpinBox)):
                continue
            if not widget.property("readabilityWatched"):
                widget.setProperty("readabilityWatched", True)
                widget.installEventFilter(self)
                if isinstance(widget, QComboBox):
                    widget.currentIndexChanged.connect(self.schedule)
                elif isinstance(widget, QAbstractSpinBox) and hasattr(widget, "valueChanged"):
                    widget.valueChanged.connect(self.schedule)
            widget.ensurePolished()
            hint = widget.sizeHint()
            width = hint.width()
            if isinstance(widget, QComboBox) and not widget.property("allowValueElision"):
                # Reserve the selected text plus the style's real padding and
                # arrow area, measured at a probe width rather than assumed.
                option = QStyleOptionComboBox()
                widget.initStyleOption(option)
                option.rect.setWidth(400)
                field = widget.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, option,
                                                      QStyle.SubControl.SC_ComboBoxEditField, widget)
                chrome = 400 - field.width() if 0 < field.width() < 400 else 52
                width = max(width, widget.fontMetrics().horizontalAdvance(widget.currentText()) + chrome + 2)
            before = (widget.minimumWidth(), widget.minimumHeight(), widget.maximumHeight())
            if widget.minimumWidth() != width:
                widget.setMinimumWidth(width)
            if widget.maximumHeight() < hint.height():
                widget.setMaximumHeight(hint.height())
            if widget.minimumHeight() < hint.height():
                widget.setMinimumHeight(hint.height())
            if before != (widget.minimumWidth(), widget.minimumHeight(), widget.maximumHeight()):
                changed.append(widget)
        if changed:
            relayout_ancestors(changed, self.root)


def install_readability_guard(root):
    if not hasattr(root, "_readability_guard"):
        root._readability_guard = ReadabilityGuard(root)
    return root._readability_guard


class PageStack(QStackedWidget):
    """Main pages without a tab strip; navigation lives in the window header.

    Pages that do not own a scroll are wrapped so a small or high-DPI screen can
    still reach every control without forcing the window beyond the display.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._labels: list[str] = []

    def addTab(self, page, label):
        self._labels.append(label)
        if page.property("ownsPageScroll"):
            return self.addWidget(page)
        scroll = QScrollArea()
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        return self.addWidget(scroll)

    def tabText(self, index):
        return self._labels[index]

    def page(self, index):
        widget = super().widget(index)
        return widget.widget() if isinstance(widget, QScrollArea) else widget

    def currentWidget(self):
        widget = super().currentWidget()
        return widget.widget() if isinstance(widget, QScrollArea) else widget

    def setCurrentWidget(self, page):
        for index in range(self.count()):
            container = super().widget(index)
            if container is page or (isinstance(container, QScrollArea) and container.widget() is page):
                return super().setCurrentIndex(index)
        return super().setCurrentWidget(page)


def workbench_palette(dark: bool) -> dict:
    """Neutral surfaces; saturated colors are reserved for state and the main action."""
    return (dict(variant="dark", bg="#1b2129", panel="#232b35", head="#2b3440", fg="#e3e9f0", muted="#a3b2c2",
                 border="#3b4857", blue="#8db8f0", soft="#2a3b50", button_text="#0f1f33",
                 ok="#8fd19e", warning="#f0c27a", danger="#ff9c94") if dark else
            dict(variant="light", bg="#eceff3", panel="#ffffff", head="#e2e7ed", fg="#1d2733", muted="#56657a",
                 border="#c3ccd6", blue="#1b5fad", soft="#e3edf8", button_text="#ffffff",
                 ok="#1d7a35", warning="#8a5300", danger="#b3261e"))


def _control_styles(colors):
    icons = Path(__file__).parent / "assets"
    colors = {**colors, "down": (icons / f"chevron-down-{colors['variant']}.svg").as_posix(),
              "up": (icons / f"chevron-up-{colors['variant']}.svg").as_posix()}
    return """
        QPushButton { font-size:13px; min-height:18px; padding:3px 9px; border:1px solid %(border)s;
                      border-radius:2px; background:%(panel)s; color:%(fg)s; }
        QPushButton:hover { border-color:%(blue)s; }
        QPushButton:pressed, QPushButton:checked { background:%(soft)s; color:%(blue)s; border-color:%(blue)s; }
        QPushButton#primaryButton { background:%(blue)s; color:%(button_text)s; border-color:%(blue)s; font-weight:600; }
        QPushButton#primaryButton:disabled { background:%(head)s; color:%(muted)s; border-color:%(border)s; }
        QPushButton:disabled { color:%(muted)s; background:%(bg)s; }
        QPushButton#dangerButton { color:%(danger)s; }
        QPushButton#dangerButton:hover { border-color:%(danger)s; }
        QPushButton#dangerButton:disabled { color:%(muted)s; }
        QLineEdit, QComboBox, QAbstractSpinBox { font-size:13px; min-height:18px;
              background:%(panel)s; color:%(fg)s; border:1px solid %(border)s; border-radius:2px; padding:3px 6px; }
        QComboBox { padding-right:24px; }
        QComboBox::drop-down { subcontrol-origin:padding; subcontrol-position:top right; width:20px; border:0; }
        QComboBox::down-arrow { image:url("%(down)s"); width:10px; height:10px; }
        QComboBox QAbstractItemView { background:%(panel)s; color:%(fg)s; selection-background-color:%(soft)s;
                                    selection-color:%(blue)s; border:1px solid %(border)s; }
        QAbstractSpinBox { padding-right:20px; }
        QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin:border; subcontrol-position:top right; width:18px; border:0; }
        QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin:border; subcontrol-position:bottom right; width:18px; border:0; }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image:url("%(up)s"); width:9px; height:9px; }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image:url("%(down)s"); width:9px; height:9px; }
        QDateEdit::drop-down { width:20px; border:0; }
        QDateEdit::down-arrow { image:url("%(down)s"); width:10px; height:10px; }
        QPlainTextEdit { font-size:13px; background:%(panel)s; color:%(fg)s;
                        border:1px solid %(border)s; border-radius:2px; padding:3px 4px; }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {
              border-color:%(blue)s; }
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QPlainTextEdit:disabled {
              color:%(muted)s; background:%(bg)s; }
        QCheckBox { font-size:13px; spacing:5px; }
        QListWidget { background:%(panel)s; color:%(fg)s; border:1px solid %(border)s; border-radius:0; }
        QListWidget::item { padding:5px 6px; }
        QListWidget::item:selected { background:%(soft)s; color:%(blue)s; }
        QScrollArea { border:0; background:transparent; }
        QScrollBar:vertical { background:%(bg)s; width:10px; margin:0; }
        QScrollBar::handle:vertical { background:%(border)s; min-height:30px; }
        QScrollBar::handle:vertical:hover { background:%(muted)s; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:none; }
        QScrollBar:horizontal { background:%(bg)s; height:10px; margin:0; }
        QScrollBar::handle:horizontal { background:%(border)s; min-width:30px; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width:0; }
        QLabel#sectionTitle { font-size:13px; font-weight:600; color:%(fg)s; }
        QLabel#mutedLabel, QLabel#fieldLabel { color:%(muted)s; font-size:12px; }
        QLabel#warningLabel { color:%(warning)s; font-size:12px; }
        QLabel#columnTitle { color:%(muted)s; font-size:12px; font-weight:600; }
        QWidget#columnHeader { background:%(head)s; border-bottom:1px solid %(border)s; }
        QLabel[state="ok"] { color:%(ok)s; }
        QLabel[state="ng"] { color:%(danger)s; }
        QLabel[state="warning"] { color:%(warning)s; }
        QLabel[state="running"] { color:%(blue)s; }
        QLabel[state="neutral"] { color:%(muted)s; }
    """ % colors


def application_styles(dark: bool) -> str:
    colors = workbench_palette(dark)
    return _control_styles(colors) + """
        QMainWindow#sopMainWindow { background:%(bg)s; color:%(fg)s; }
        QWidget#appHeader { background:%(panel)s; border-bottom:1px solid %(border)s; }
        QPushButton#navButton { font-size:13px; padding:5px 18px; border:1px solid %(border)s; border-radius:0;
                                background:%(bg)s; color:%(fg)s; }
        QPushButton#navButton[segment="middle"], QPushButton#navButton[segment="last"] { border-left:0; }
        QPushButton#navButton:checked { background:%(blue)s; color:%(button_text)s; border-color:%(blue)s; font-weight:600; }
        QToolBar#sourceToolbar { background:%(bg)s; border:0; border-bottom:1px solid %(border)s; padding:2px 6px; spacing:6px; }
        QToolBar#sourceToolbar QLabel { color:%(muted)s; font-size:12px; }
        QStackedWidget#mainPages { background:%(bg)s; }
        QStatusBar { background:%(panel)s; color:%(muted)s; font-size:12px; border-top:1px solid %(border)s; }
        QStatusBar::item { border:0; }
        QStatusBar QLabel, StatusText { color:%(muted)s; font-size:12px; }
        QTableWidget { background:%(panel)s; color:%(fg)s; border:1px solid %(border)s; gridline-color:%(border)s;
                       font-size:13px; selection-background-color:%(soft)s; selection-color:%(fg)s; }
        QHeaderView::section { background:%(head)s; color:%(muted)s; font-size:12px; font-weight:600; padding:3px 6px;
                               border:0; border-right:1px solid %(border)s; border-bottom:1px solid %(border)s; }
        QSplitter::handle { background:%(bg)s; }
    """ % colors


def workbench_styles(dark: bool) -> str:
    colors = workbench_palette(dark)
    return """
        QWidget#workbenchEditor, QWidget#editorDocument, QWidget#toolsDocument { background:%(bg)s; color:%(fg)s; font-size:13px; }
        QWidget#monitorPane { background:%(bg)s; }
        QFrame#panel, QFrame#stepBlock { background:%(panel)s; border:1px solid %(border)s; border-radius:0; }
        QWidget#stepHeader { background:%(head)s; }
        QLabel#stepNumber { font-size:14px; font-weight:700; color:%(fg)s; }
        QLabel#groupLabel { font-weight:600; color:%(fg)s; }
        QLabel#groupLabel[role="violation"] { color:%(danger)s; }
        QLabel#guideLabel { color:%(fg)s; background:%(soft)s; padding:3px 6px; font-size:12px; }
        QLabel#setupSummary { color:%(fg)s; font-size:12px; }
        QLabel#setupSummary[state="warning"] { color:%(warning)s; }
        QLabel#livePreview { font-size:12px; color:%(muted)s; }
        QLabel#validationStatus { font-size:12px; color:%(muted)s; }
        QLabel#validationStatus[state="warning"] { color:%(warning)s; }
        QLabel#validationStatus[state="running"] { color:%(blue)s; }
        QLabel#validationStatus[state="ng"] { color:%(danger)s; }
        QLabel#trialAlert { color:white; background:#b71c1c; border:1px solid #f26c6c;
                           padding:6px; font-size:12px; font-weight:600; }
        QFrame#conditionBlock { background:%(bg)s; border:1px solid %(border)s; }
        QFrame#conditionBlock[selected="true"] { background:%(soft)s; border-color:%(blue)s; }
        QLabel#conditionTitle { color:%(muted)s; font-size:12px; font-weight:600; }
        QFrame#regionRow { border:0; border-bottom:1px solid %(border)s; }
        QWidget#trackingBar { background:%(panel)s; border:1px solid %(border)s; }
        QLabel#rulesSummary { color:%(fg)s; font-size:12px; }
    """ % colors + _control_styles(colors)
