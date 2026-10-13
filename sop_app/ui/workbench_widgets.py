"""Small native widgets shared by the image-workbench editor."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QComboBox, QLabel, QPushButton,
                              QScrollArea, QSizePolicy, QTabWidget, QToolButton, QVBoxLayout, QWidget)


class ActionHint(QLabel):
    """Show actionable feedback without permanently repeating onboarding text."""
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setObjectName("guideLabel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.setText(text)

    def setText(self, text):
        super().setText(text)
        routine = text.startswith(("即時畫面：設定", "按「開始引導設定」"))
        self.setVisible(bool(text) and not routine)


class CollapsibleSection(QWidget):
    expanded_changed = Signal(bool)

    def __init__(self, title: str, content: QWidget, expanded=False, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.content = content
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setObjectName("sectionToggle")
        self.toggle.toggled.connect(self.set_expanded)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self.toggle)
        layout.addWidget(content)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool):
        self.toggle.blockSignals(True)
        self.toggle.setChecked(expanded)
        self.toggle.blockSignals(False)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.content.setVisible(expanded)
        self.expanded_changed.emit(expanded)


def apply_action_icons(root: QWidget, dark: bool):
    """Render action symbols independently of the text font's glyph bounds."""
    variant = "dark" if dark else "light"
    for button in root.findChildren(QPushButton):
        name = button.property("actionIcon")
        if name:
            button.setIcon(QIcon(str(Path(__file__).parent / "assets" / f"{name}-{variant}.svg")))
            button.setIconSize(QSize(12, 12))


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
            if isinstance(widget, QComboBox):
                # Some cards deliberately use AdjustToMinimumContentsLength.
                # Still reserve enough width for the selected value + arrow.
                width = max(width, widget.fontMetrics().horizontalAdvance(widget.currentText()) + 42)
            if widget.minimumWidth() != width:
                widget.setMinimumWidth(width)
            if widget.maximumHeight() < hint.height():
                widget.setMaximumHeight(hint.height())
            if widget.minimumHeight() < hint.height():
                widget.setMinimumHeight(hint.height())


def install_readability_guard(root):
    if not hasattr(root, "_readability_guard"):
        root._readability_guard = ReadabilityGuard(root)
    return root._readability_guard


class ScrollablePageTabs(QTabWidget):
    """Keep the page API while allowing a high-DPI/small screen to scroll.

    A hidden settings page must not force the whole window beyond the display.
    Ordinary desktop sizes have no extra scroll bars.
    """
    def addTab(self, page, label):
        scroll = QScrollArea()
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        return super().addTab(scroll, label)

    def currentWidget(self):
        widget = super().currentWidget()
        return widget.widget() if isinstance(widget, QScrollArea) else widget

    def setCurrentWidget(self, page):
        for index in range(self.count()):
            container = super().widget(index)
            if isinstance(container, QScrollArea) and container.widget() is page:
                return super().setCurrentWidget(container)
        return super().setCurrentWidget(page)


def workbench_palette(dark: bool) -> dict:
    return (dict(bg="#171e28", panel="#232d3b", fg="#e2eaf5", muted="#a8b9ce", border="#3b4a60",
                 blue="#91bfff", soft="#293f5d", button_text="#122238", warning="#ffd099", danger="#ff9f9f") if dark else
            dict(bg="#f4f7fb", panel="#ffffff", fg="#263549", muted="#627287", border="#dce4ee",
                 blue="#155fbc", soft="#edf4ff", button_text="#ffffff", warning="#885519", danger="#b42332"))


def _control_styles(colors):
    variant = "dark" if colors["bg"] == "#171e28" else "light"
    icons = Path(__file__).parent / "assets"
    colors = {**colors, "down": (icons / f"chevron-down-{variant}.svg").as_posix(),
              "up": (icons / f"chevron-up-{variant}.svg").as_posix()}
    return """
        QPushButton { font-size:13px; min-height:18px; padding:6px 10px; border:1px solid %(border)s;
                      border-radius:6px; background:%(panel)s; color:%(fg)s; }
        QPushButton:hover { border-color:%(blue)s; background:%(soft)s; }
        QPushButton:checked { background:%(soft)s; color:%(blue)s; border-color:%(blue)s; }
        QPushButton#primaryButton { background:%(blue)s; color:%(button_text)s; border-color:%(blue)s; }
        QPushButton#primaryButton:disabled { background:%(soft)s; color:%(muted)s; border-color:%(border)s; }
        QPushButton:disabled { color:%(muted)s; }
        QPushButton#dangerButton { border-color:transparent; color:%(muted)s; padding:4px 7px; }
        QPushButton#dangerButton:hover { color:%(danger)s; border-color:%(danger)s; }
        QLineEdit, QComboBox, QAbstractSpinBox { font-size:14px; min-height:20px;
              background:%(panel)s; color:%(fg)s; border:1px solid %(border)s; border-radius:5px; padding:5px 8px; }
        QComboBox { padding-right:28px; }
        QComboBox::drop-down { subcontrol-origin:padding; subcontrol-position:top right; width:24px; border:0; }
        QComboBox::down-arrow { image:url("%(down)s"); width:12px; height:12px; }
        QComboBox QAbstractItemView { background:%(panel)s; color:%(fg)s; selection-background-color:%(soft)s;
                                    selection-color:%(blue)s; border:1px solid %(border)s; }
        QAbstractSpinBox { padding-right:26px; }
        QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin:border; subcontrol-position:top right; width:20px; border:0; }
        QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin:border; subcontrol-position:bottom right; width:20px; border:0; }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image:url("%(up)s"); width:10px; height:10px; }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image:url("%(down)s"); width:10px; height:10px; }
        QDateEdit::drop-down { width:24px; border:0; }
        QDateEdit::down-arrow { image:url("%(down)s"); width:12px; height:12px; }
        QPlainTextEdit { font-size:13px; background:%(panel)s; color:%(fg)s;
                        border:1px solid %(border)s; border-radius:5px; padding:6px; }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {
              border-color:%(blue)s; }
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { color:%(muted)s; }
        QListWidget { background:%(panel)s; color:%(fg)s; border:1px solid %(border)s; border-radius:6px; }
        QListWidget::item { padding:10px 8px; margin:2px; border-radius:4px; }
        QListWidget::item:selected { background:%(soft)s; color:%(blue)s; }
        QScrollArea { border:0; background:transparent; }
        QScrollBar:vertical { background:%(bg)s; width:10px; margin:0; }
        QScrollBar::handle:vertical { background:%(border)s; min-height:30px; border-radius:5px; }
        QScrollBar::handle:vertical:hover { background:%(muted)s; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:none; }
    """ % colors


def application_styles(dark: bool) -> str:
    colors = workbench_palette(dark)
    return _control_styles(colors) + """
        QMainWindow#sopMainWindow { background:%(bg)s; color:%(fg)s; }
        QMenuBar { background:%(panel)s; color:%(fg)s; font-size:13px; padding:2px; }
        QMenuBar::item { padding:4px 8px; } QMenuBar::item:selected { background:%(soft)s; }
        QToolBar#sourceToolbar { background:%(panel)s; border:0; border-bottom:1px solid %(border)s; padding:4px; spacing:8px; }
        QTabWidget#mainTabs::pane { border:0; background:%(bg)s; }
        QTabBar#mainNavigation::tab { color:%(muted)s; background:%(panel)s; font-size:13px;
                                    padding:9px 18px; border-bottom:2px solid transparent; }
        QTabBar#mainNavigation::tab:selected { color:%(blue)s; background:%(soft)s; border-bottom-color:%(blue)s; }
        QStatusBar { background:%(panel)s; color:%(muted)s; font-size:12px; border-top:1px solid %(border)s; }
        QStatusBar::item { border:0; }
        QTableWidget { background:%(panel)s; color:%(fg)s; border:1px solid %(border)s; gridline-color:%(border)s; }
        QHeaderView::section { background:%(bg)s; color:%(muted)s; font-size:13px; padding:6px; border:0; }
    """ % colors


def workbench_styles(dark: bool) -> str:
    colors = workbench_palette(dark)
    return """
        QWidget#workbenchEditor { background:%(bg)s; color:%(fg)s; font-size:13px; }
        QWidget#inspectorContent { background:%(bg)s; }
        QTabWidget#workbenchInspector::pane { border:0; background:%(bg)s; }
        QGroupBox { background:%(panel)s; border:1px solid %(border)s; border-radius:8px;
                    margin-top:16px; padding:15px 10px 9px; color:%(fg)s; font-size:13px; }
        QGroupBox::title { subcontrol-origin:margin; left:12px; color:%(fg)s; padding:0 3px; }
        QGroupBox#validationBox { margin-top:14px; padding:12px 10px 8px; }
        QFrame#conditionCard { background:%(panel)s; border:1px solid %(border)s; border-radius:7px; }
        QFrame#conditionCard[selected="true"] { border:1px solid %(blue)s; background:%(soft)s; }
        QLabel#workspaceTitle { font-size:20px; font-weight:600; color:%(fg)s; }
        QLabel#contextTitle { font-size:14px; font-weight:500; color:%(fg)s; }
        QLabel#mutedLabel { color:%(muted)s; font-size:12px; }
        QLabel#fieldLabel { color:%(muted)s; font-size:12px; }
        QLabel#livePreview { font-size:12px; color:%(muted)s; }
        QLabel#documentState { color:%(blue)s; background:%(soft)s; padding:4px 8px; border-radius:5px; font-size:12px; }
        QLabel#validationStatus { font-size:12px; color:%(muted)s; }
        QLabel#validationStatus[state="warning"] { color:%(warning)s; }
        QLabel#validationStatus[state="running"] { color:%(blue)s; }
        QLabel#setupSummary { color:%(fg)s; background:%(panel)s; border:1px solid %(border)s;
                              padding:7px 10px; border-radius:6px; font-size:12px; }
        QLabel#guideLabel { color:%(blue)s; background:%(soft)s; padding:6px 8px; border-radius:5px; font-size:12px; }
        QPushButton#workbenchNav { padding:7px 15px; border-color:transparent; }
        QPushButton#workbenchNav:checked { border-color:%(blue)s; }
        QToolButton#sectionToggle { border:0; color:%(muted)s; padding:4px; min-height:18px; font-size:12px; }
        QSplitter::handle { background:transparent; }
        QSplitter::handle:hover { background:%(soft)s; }
    """ % colors + _control_styles(colors)
