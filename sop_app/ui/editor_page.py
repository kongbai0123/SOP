"""SOP editor: the live image in the centre with workpiece tracking under it,
steps on the left, regions and trial on the right."""
from __future__ import annotations

import copy
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
                               QHBoxLayout, QLabel, QLayout, QLineEdit, QMessageBox,
                               QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)
from ..conditions import evaluate_conditions
from ..engine import evaluate_step, violation_window
from ..paths import PROJECT_ROOT, relativize, resolve
from ..pipeline import FramePacket
from ..sop_schema import CONDITION_TYPES, Condition, ROI, SOPDefinition, Step
from ..detection import FrameResult
from ..tracking import WorkpieceLocator, capture_workpiece, decode_reference, display_rois, workpiece_key
from .compact_layout import CompactLog, relayout_now, set_state, text_label
from .condition_rows import ConditionRows
from .editor_validation import EditorValidation
from .step_editor import StepEditor, field_pair, flow_row
from .video_widget import VideoWidget
from .toggle_switch import ToggleSwitch
from .workbench_widgets import install_readability_guard, workbench_styles

SIDE_MARGIN = 6          # inner margin of the two side columns
CENTER_MIN_WIDTH = 440   # below this the right column moves under the image


def section_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    label.setTextFormat(Qt.TextFormat.PlainText)
    return label


def panel(*widgets, object_name="panel", spacing=4) -> QFrame:
    frame = QFrame()
    frame.setObjectName(object_name)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(6, 5, 6, 6)
    layout.setSpacing(spacing)
    for widget in widgets:
        layout.addWidget(widget)
    return frame


def titled(title: str, *widgets) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(3)
    layout.addWidget(section_title(title))
    for item in widgets:
        layout.addWidget(item)
    return widget


def side_scroll(content: QWidget) -> QScrollArea:
    """A side column scrolls on its own; columns are side by side, never nested."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(content)
    return scroll


def breakable(text: str, every=12) -> str:
    """Long file names without spaces may wrap instead of widening the column."""
    return "​".join(text[index:index + every] for index in range(0, len(text), every))


class ROIEditorRow(QFrame):
    def __init__(self, roi, parent=None):
        super().__init__(parent)
        self.setObjectName("regionRow")
        self.roi = roi
        self.name_edit = QLineEdit(roi.name)
        self.name_edit.setAccessibleName("區域名稱")
        self.anchor_combo = QComboBox()
        self.anchor_combo.addItem("固定畫面", "fixed")
        self.anchor_combo.addItem("跟隨工件", "workpiece")
        self.anchor_combo.setCurrentIndex(max(0, self.anchor_combo.findData(roi.anchor)))
        self.status_text = ""
        self.edit_button = QPushButton("移動／縮放")
        self.duplicate_button = QPushButton("複製")
        self.delete_button = QPushButton("刪除")
        self.delete_button.setObjectName("dangerButton")
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(4)
        top.addWidget(self.name_edit, 1)
        top.addWidget(self.anchor_combo)
        self.actions_row = QWidget()
        actions = QHBoxLayout(self.actions_row)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(4)
        for button in (self.edit_button, self.duplicate_button, self.delete_button):
            actions.addWidget(button)
        actions.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 5)
        layout.setSpacing(3)
        layout.addLayout(top)
        layout.addWidget(self.actions_row)
        self.set_status("本幀 —")

    def set_editing(self, editing):
        self.anchor_combo.setVisible(editing)
        self.actions_row.setVisible(editing)

    def set_status(self, text):
        """This frame's state of the region is a tooltip on the row."""
        self.status_text = text
        anchor = "跟隨工件" if self.roi.anchor == "workpiece" else "固定畫面"
        self.setToolTip(f"{self.roi.name}（{anchor}）\n{text}")


class EditorPage(QWidget):
    LIVE_PREVIEW_INTERVAL_SEC = .2
    modified = Signal()
    model_selected = Signal(str)
    workpiece_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workbenchEditor")
        self.setProperty("ownsPageScroll", True)
        self._data_root = PROJECT_ROOT
        self._resolve_path = resolve
        self._relativize_path = relativize
        # The arrangement adapts to the size it receives, so the three-column
        # minimum must not stop the window from shrinking.
        self.setMinimumSize(480, 320)
        self.sop, self.classes = SOPDefinition(), []
        self._loading, self._production_running = False, False
        self._packet = self._preview_result = self._packet_arrived_at = None
        self._last_live_preview_at = None
        self._locator = self._reference_frame = None
        self._workpiece_key = ""
        # Only image arrays and their exact pose are retained while selecting a
        # frame. FramePacket can own detection tensors and must not be cached.
        self._latest_frame = self._latest_preview = self._latest_matrix = None
        self._latest_arrived_at = None
        self._latest_key = ""
        self._edit_frame = self._edit_matrix = None
        self._edit_frame_kind = self._edit_key = ""
        self._capture_target = False
        self._regions_visible = True
        self._roi_undo, self._roi_redo = [], []
        self._selected_roi_name = self._highlighted_condition_roi = None
        self._model_state, self._model_detail = "unselected", ""
        self._theme_dark = None
        self._mode = None
        self._applying_layout = False
        self._image_focus = self._auto_focus = False
        self._editing_steps = False
        self._individual_step = None
        self._editing_rois = False
        self._expanded: set[int] = set()  # id() of steps whose settings are open
        self._validation = EditorValidation()
        self.step_editors, self.roi_rows = [], []
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._apply_layout_mode)
        self._build()
        if QApplication.instance():
            QApplication.instance().paletteChanged.connect(self._apply_workbench_theme)
        self._apply_workbench_theme()
        install_readability_guard(self)

    # ---- layout --------------------------------------------------------------
    def _build(self):
        self.settings_pane = self._build_settings()
        self.monitor_pane = self._build_center()
        self.tools_pane = self._build_tools()
        self.root_layout = QHBoxLayout(self)
        self.root_layout.setContentsMargins(0, 6, 0, 6)
        self.root_layout.setSpacing(6)
        self.root_layout.addWidget(self.settings_pane)
        self.root_layout.addWidget(self.monitor_pane, 1)
        self.root_layout.addWidget(self.tools_pane)
        # Opening a step or adding a long name may need a wider side column.
        self.page_content.installEventFilter(self)
        self.tools_content.installEventFilter(self)

    def _build_settings(self):
        """Left column: project, every step from top to bottom, then the next-cycle rules."""
        self.setup_summary = text_label("", "setupSummary")
        self.page_content = QWidget()
        self.page_content.setObjectName("editorDocument")
        self.document_layout = QVBoxLayout(self.page_content)
        self.document_layout.setContentsMargins(SIDE_MARGIN, 4, SIDE_MARGIN, 12)
        self.document_layout.setSpacing(10)
        self.document_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.settings_content = QWidget()
        settings = QVBoxLayout(self.settings_content)
        settings.setContentsMargins(0, 0, 0, 0)
        settings.setSpacing(12)
        settings.addWidget(self._build_project_area())
        settings.addWidget(self._build_steps_area())
        settings.addWidget(self._build_cycle_area())
        self.document_layout.addWidget(self.settings_content)
        self.document_layout.addStretch(1)
        self.page_scroll = side_scroll(self.page_content)
        self.page_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        pane = QWidget()
        pane.setObjectName("settingsPane")
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        summary = QWidget()
        summary_layout = QVBoxLayout(summary)
        summary_layout.setContentsMargins(SIDE_MARGIN + 2, 0, SIDE_MARGIN, 0)
        summary_layout.addWidget(self.setup_summary)
        layout.addWidget(summary)
        layout.addWidget(self.page_scroll, 1)
        return pane

    def _build_center(self):
        """Centre: the image is the main body, with only view controls under it."""
        pane = QWidget()
        pane.setObjectName("monitorPane")
        self.video = VideoWidget()
        self.video.setMinimumSize(240, 135)
        self.video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video.roi_drawn.connect(self._on_roi_drawn)
        self.video.roi_selected.connect(self._select_roi)
        self.video.roi_edited.connect(self._edit_roi_points)
        self.video.roi_rename_requested.connect(self._rename_selected_roi)
        # Framing and tracking checks may use the steps column's width as well.
        self.focus_button = QPushButton("放大影像")
        self.focus_button.setCheckable(True)
        self.focus_button.toggled.connect(lambda checked: self._focus_image(auto=False) if checked
                                          else self._set_image_focus(False))
        self.freeze_check = QCheckBox("凍結畫面")
        self.freeze_check.toggled.connect(self._on_freeze)
        self.live_button = QPushButton("返回即時影像")
        self.live_button.clicked.connect(self._return_live)
        self.target_button = QPushButton("框選參考工件")
        self.target_button.clicked.connect(self._begin_target)
        self.clear_target_button = QPushButton("清除工件跟隨")
        self.clear_target_button.clicked.connect(lambda: self._clear_workpiece())
        self.reference_button = QPushButton("編輯跟隨區域")
        self.reference_button.setCheckable(True)
        self.reference_button.setEnabled(False)
        self.reference_button.toggled.connect(self._show_reference)
        self.tracking_label = text_label("尚未設定工件跟隨；區域使用固定畫面座標。", "mutedLabel")
        self.tracking_tools = QWidget()
        self.tracking_tools.setObjectName("trackingBar")
        self.tracking_tools.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        tracking = QVBoxLayout(self.tracking_tools)
        tracking.setContentsMargins(6, 3, 6, 3)
        tracking.setSpacing(3)
        tracking.addWidget(flow_row(section_title("工件跟隨"), self.target_button,
                                    self.reference_button, self.clear_target_button, spacing=6))
        tracking.addWidget(self.tracking_label)
        self.retake_frame_button = QPushButton("重新取景")
        self.retake_frame_button.clicked.connect(self._retake_following_frame)
        self.use_frame_button = QPushButton("使用這一幀")
        self.use_frame_button.clicked.connect(self._use_latest_following_frame)
        self.original_frame_button = QPushButton("使用原始參考影像")
        self.original_frame_button.clicked.connect(self._use_original_reference)
        self.following_frame_tools = QWidget()
        frame_tools = QVBoxLayout(self.following_frame_tools)
        frame_tools.setContentsMargins(0, 0, 0, 0)
        frame_tools.setSpacing(3)
        frame_tools.addWidget(flow_row(self.retake_frame_button, self.use_frame_button,
                                      self.original_frame_button, spacing=4))
        self.following_frame_label = text_label("", "mutedLabel")
        frame_tools.addWidget(self.following_frame_label)
        self.following_frame_tools.hide()
        tracking.addWidget(self.following_frame_tools)
        self.view_bar = QWidget()
        bar = QHBoxLayout(self.view_bar)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(6)
        for widget in (self.focus_button, self.freeze_check):
            bar.addWidget(widget)
        self.live_label = text_label("單幀預覽：等待影像", "livePreview")
        bar.addWidget(self.live_label, 1)
        self.guide_label = text_label("", "guideLabel")
        self.center_layout = QVBoxLayout(pane)
        self.center_layout.setContentsMargins(0, 0, 0, 0)
        self.center_layout.setSpacing(4)
        self.center_layout.addWidget(self.video, 1)
        self.center_layout.addWidget(self.tracking_tools)
        self.center_layout.addWidget(self.view_bar)
        self.center_layout.addWidget(self.guide_label)
        return pane

    def _build_tools(self):
        """Right column: regions and the trial."""
        self.draw_button = QPushButton("新增區域")
        self.draw_button.setCheckable(True)
        self.draw_button.toggled.connect(self._toggle_draw)
        self.undo_roi_button, self.redo_roi_button = QPushButton("復原區域"), QPushButton("重作區域")
        self.undo_roi_button.clicked.connect(self._undo_roi)
        self.redo_roi_button.clicked.connect(self._redo_roi)
        self.edit_rois_button = QPushButton("編輯區域")
        self.edit_rois_button.setCheckable(True)
        self.edit_rois_button.toggled.connect(self._set_rois_editing)
        self.region_tools = flow_row(self.draw_button, self.edit_rois_button, self.live_button, spacing=4)
        self.roi_edit_toolbar = flow_row(self.undo_roi_button, self.redo_roi_button, spacing=4)
        self.roi_edit_note = text_label("可修改定位方式、移動／縮放、複製或刪除區域；復原／重作會保留區域與條件的引用。", "mutedLabel")
        self.roi_edit_toolbar.hide()
        self.roi_edit_note.hide()
        self.rois_layout = QVBoxLayout()
        self.rois_layout.setContentsMargins(0, 0, 0, 0)
        self.rois_layout.setSpacing(0)
        rows = QWidget()
        rows.setLayout(self.rois_layout)
        self.empty_rois_label = text_label("尚未建立作業區域；條件可使用整個畫面。", "mutedLabel")
        self.roi_status = text_label("尚未建立作業區域。", "mutedLabel")
        self.roi_status.hide()  # per-region status is shown in each row
        self.regions_panel = panel(self.region_tools, self.roi_edit_toolbar,
                                   self.roi_edit_note, rows, self.empty_rois_label)
        regions = titled("作業區域", self.regions_panel)

        self.validation_box = self._build_validation_area()
        self.tools_content = QWidget()
        self.tools_content.setObjectName("toolsDocument")
        layout = QVBoxLayout(self.tools_content)
        layout.setContentsMargins(SIDE_MARGIN, 0, SIDE_MARGIN, 6)
        layout.setSpacing(10)
        layout.addWidget(regions)
        layout.addWidget(self.validation_box, 1)
        self.tools_scroll = side_scroll(self.tools_content)
        self._update_roi_history_buttons()
        return self.tools_scroll

    def _build_project_area(self):
        self.name_edit = QLineEdit()
        self.name_edit.textEdited.connect(self._on_sop_name)
        self.version_label = QLabel("1")
        self.model_label = text_label("（未選擇）")
        self.model_state_label = text_label("", "livePreview")
        self.model_button = QPushButton("選擇模型…")
        self.model_button.clicked.connect(self._choose_model)
        form = QWidget()
        rows = QFormLayout(form)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setHorizontalSpacing(8)
        rows.setVerticalSpacing(4)
        name = QHBoxLayout()
        name.setSpacing(8)
        name.addWidget(self.name_edit, 1)
        name.addWidget(field_pair("版本", self.version_label))
        state = QHBoxLayout()
        state.setSpacing(6)
        state.addWidget(self.model_state_label, 1)
        state.addWidget(self.model_button)
        rows.addRow(self._form_label("SOP 名稱"), name)
        rows.addRow(self._form_label("辨識模型"), self.model_label)
        rows.addRow(self._form_label("狀態"), state)
        self.sop_box = panel(form)
        return titled("專案", self.sop_box)

    @staticmethod
    def _form_label(text):
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    def _build_steps_area(self):
        self.steps_box = QWidget()
        layout = QVBoxLayout(self.steps_box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.add_step_button = QPushButton("新增工序")
        self.add_step_button.clicked.connect(lambda: self._add_step())
        self.edit_steps_button = QPushButton("編輯")
        self.edit_steps_button.setCheckable(True)
        self.edit_steps_button.toggled.connect(self._set_steps_editing)
        title = QHBoxLayout()
        title.setContentsMargins(0, 0, 0, 0)
        title.setSpacing(4)
        title.addWidget(section_title("工序（由上而下依序執行）"), 1)
        title.addWidget(self.add_step_button)
        title.addWidget(self.edit_steps_button)
        layout.addLayout(title)
        self.steps_layout = QVBoxLayout()
        self.steps_layout.setSpacing(6)
        layout.addLayout(self.steps_layout)
        self.empty_steps_label = text_label("尚未新增工序。請按「新增工序」開始設定。", "mutedLabel")
        layout.addWidget(self.empty_steps_label)
        return self.steps_box

    def _build_cycle_area(self):
        self.auto_restart_spin = StepEditor.seconds(3, 600)
        self.reset_hold_spin = StepEditor.seconds(2, 60)
        self.reset_ratio_spin = StepEditor.percent(80, 1)
        self.sequence_check = ToggleSwitch("偵測跳步")
        self.cycle_enabled_switch = ToggleSwitch("自動開始下一輪")
        self.cycle_enabled_switch.setChecked(True)
        self.reset_table = ConditionRows("reset", "新增重置條件", "未設定重置條件：本輪完成後依延遲秒數開始下一輪")
        self.reset_table.changed.connect(self._on_cycle_form)
        self.reset_table.region_selected.connect(self._highlight_condition_region)
        self.cycle_rule_summary = text_label("", "mutedLabel")
        form = QWidget()
        rows = QFormLayout(form)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setHorizontalSpacing(8)
        rows.setVerticalSpacing(4)
        rows.addRow(self._form_label("無重置條件時延遲重啟"), self.auto_restart_spin)
        rows.addRow(self._form_label("重置觀察"), self.reset_hold_spin)
        rows.addRow(self._form_label("重置成立比例"), self.reset_ratio_spin)
        sequence_note = text_label("後續工序含「出現」條件且提前成立時發出順序錯誤警報。", "mutedLabel")
        self.cycle_settings = panel(form, self.cycle_rule_summary,
                               text_label("重置條件全部成立後才開始下一輪（選填）", "mutedLabel"),
                               self.reset_table)
        self.cycle_box = panel(self.cycle_enabled_switch, self.cycle_settings,
                               self.sequence_check, sequence_note)
        self.cycle_enabled_switch.toggled.connect(self._on_cycle_form)
        for spin in (self.auto_restart_spin, self.reset_hold_spin, self.reset_ratio_spin):
            spin.valueChanged.connect(self._on_cycle_form)
        self.sequence_check.toggled.connect(self._on_cycle_form)
        self._update_cycle_summary()
        return titled("下一輪與順序", self.cycle_box)

    def _build_validation_area(self):
        self.validation_mode = QComboBox()
        self.validation_mode.addItem("單步驗證", "single")
        self.validation_mode.addItem("完整流程", "flow")
        self.validation_step = QComboBox()
        self.validation_mode.currentIndexChanged.connect(self._validation_scope_changed)
        self.validation_step.currentIndexChanged.connect(self._validation_scope_changed)
        self.validation_start = QPushButton("開始試跑")
        self.validation_start.setObjectName("primaryButton")
        self.validation_start.clicked.connect(self._start_validation)
        self.validation_stop = QPushButton("停止試跑")
        self.validation_stop.clicked.connect(self._stop_validation)
        self.validation_confirm = QPushButton("人工確認")
        self.validation_confirm.clicked.connect(self._confirm_validation)
        scope = QWidget()
        scope_form = QFormLayout(scope)
        scope_form.setContentsMargins(0, 0, 0, 0)
        scope_form.setHorizontalSpacing(8)
        scope_form.setVerticalSpacing(4)
        scope_form.addRow(self._form_label("範圍"), self.validation_mode)
        scope_form.addRow(self._form_label("工序"), self.validation_step)
        controls = flow_row(self.validation_start, self.validation_stop, self.validation_confirm, spacing=4)
        self.validation_status = text_label("尚未開始試跑", "validationStatus")
        self.validation_target = text_label("尚未擷取測試設定", "mutedLabel")
        metrics = QWidget()
        metrics_layout = QHBoxLayout(metrics)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        self.validation_count, self.validation_elapsed, self.validation_ratio = (
            text_label(), text_label(), text_label())
        for label in (self.validation_count, self.validation_elapsed, self.validation_ratio):
            metrics_layout.addWidget(label, 1)
        self.validation_details = CompactLog()
        self.validation_details.setReadOnly(True)
        self.validation_details.setMinimumHeight(90)
        self.validation_details.setPlaceholderText("試跑事件")
        note = text_label("使用正式判定引擎；不寫入作業紀錄", "mutedLabel")
        box = panel(note, scope, controls, self.validation_status, self.validation_target, metrics,
                    self.validation_details)
        box.layout().setStretch(6, 1)
        self._validation_watchdog = QTimer(self)
        self._validation_watchdog.setInterval(500)
        self._validation_watchdog.timeout.connect(self._check_validation_feed)
        self._validation_watchdog.start()
        self._update_validation_view()
        wrapper = titled("試跑", box)
        wrapper.layout().setStretch(1, 1)
        self.validation_panel = box
        return wrapper

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_layout_mode()

    def showEvent(self, event):
        super().showEvent(event)
        self._schedule_layout()

    def _schedule_layout(self):
        if not self._layout_timer.isActive():
            self._layout_timer.start(0)

    def eventFilter(self, watched, event):
        if watched in (self.page_content, self.tools_content) and event.type() == QEvent.Type.LayoutRequest:
            self._schedule_layout()
        return super().eventFilter(watched, event)

    @staticmethod
    def _column_width(scroll, preferred, lowest, highest):
        """Side column width: the preferred share, but never less than its content needs."""
        needed = scroll.widget().minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 2
        return max(needed, lowest, min(highest, preferred))

    def _apply_layout_mode(self):
        """Image in the centre; steps left, tools right. A narrow window moves the
        tools under the image, and framing hides the steps to enlarge the image."""
        if self._applying_layout:
            return
        self._applying_layout = True
        try:
            self._arrange()
        finally:
            self._applying_layout = False

    def _arrange(self):
        width, height = self.width(), self.height()
        spacing = self.root_layout.spacing()
        left = self._column_width(self.page_scroll, int(width * 0.27), 330, 460)
        right = self._column_width(self.tools_scroll, int(width * 0.21), 290, 360)
        steps_shown = not self._image_focus
        center = width - right - spacing - ((left + spacing) if steps_shown else 0)
        mode = "focus" if self._image_focus else "narrow" if center < CENTER_MIN_WIDTH else "wide"
        if mode == "narrow":
            center = width - left - spacing
        if mode != self._mode:
            self._mode = mode
            tools_under_image = mode == "narrow"
            if tools_under_image != (self.center_layout.indexOf(self.tools_pane) >= 0):
                if tools_under_image:
                    self.root_layout.removeWidget(self.tools_pane)
                    self.center_layout.addWidget(self.tools_pane, 1)
                else:
                    self.center_layout.removeWidget(self.tools_pane)
                    self.root_layout.addWidget(self.tools_pane)
            self.center_layout.setStretchFactor(self.video, 0 if tools_under_image else 1)
            self.settings_pane.setVisible(steps_shown)
        self.settings_pane.setFixedWidth(left)
        if mode == "narrow":
            self.tools_pane.setMinimumWidth(0)
            self.tools_pane.setMaximumWidth(16777215)
            # Keep the image readable; the tools scroll in the remaining height.
            self.video.setFixedHeight(max(135, min(round(center * 9 / 16), int(height * 0.5))))
        else:
            self.tools_pane.setFixedWidth(right)
            self.video.setMinimumHeight(135)
            self.video.setMaximumHeight(16777215)
        self.root_layout.activate()

    def _focus_image(self, auto=True):
        """Enlarge the image; automatic enlargement ends with the framing action."""
        if self._image_focus:
            self._auto_focus = self._auto_focus and auto
            return
        self._auto_focus = auto
        self._set_image_focus(True)

    def _release_auto_focus(self):
        if self._image_focus and self._auto_focus:
            self._set_image_focus(False)

    def _set_image_focus(self, enabled):
        self._image_focus = bool(enabled)
        if not enabled:
            self._auto_focus = False
        self.focus_button.blockSignals(True)
        self.focus_button.setChecked(self._image_focus)
        self.focus_button.blockSignals(False)
        self.focus_button.setText("還原版面" if self._image_focus else "放大影像")
        self._apply_layout_mode()

    def _apply_workbench_theme(self, _palette=None):
        app = QApplication.instance()
        dark = (app.palette() if app else self.palette()).window().color().lightness() < 128
        if self._theme_dark != dark:
            self._theme_dark = dark
            self.setStyleSheet(workbench_styles(dark))

    def on_save_result(self, result):
        self.guide_label.setText({"saved": "SOP 已儲存。", "cancelled": "已取消儲存；目前編輯內容仍保留。"}.get(
            result, "儲存失敗；目前編輯內容仍保留，可重試儲存。"))

    # ---- SOP ---------------------------------------------------------------
    def set_sop(self, sop):
        self._last_live_preview_at = None
        self._validation.stop()
        self._validation = EditorValidation()
        self._loading = True
        self._clear_following_frames()
        self._return_live()
        if self._image_focus:
            self._set_image_focus(False)
        self.sop = sop
        self._clear_roi_history()
        self._selected_roi_name = self._highlighted_condition_roi = None
        # Every loaded SOP opens as a clean, reorderable name list.
        self._expanded = set()
        self._individual_step = None
        self.edit_steps_button.setChecked(False)
        self.edit_rois_button.setChecked(False)
        self._model_state = "selected" if sop.model_path else "unselected"
        self._model_detail = ""
        if sop.model_path and not self._resolve_path(sop.model_path).exists():
            self._model_state, self._model_detail = "failed", "模型路徑不存在"
        self.name_edit.setText(sop.name)
        self.version_label.setText(str(sop.version))
        self._set_model_label(sop.model_path)
        self._locator = self._reference_frame = None
        self._workpiece_key = workpiece_key(sop.workpiece)
        self._packet = self._preview_result = self._packet_arrived_at = None
        self.video.clear_frame("SOP 已更新，等待即時影像")
        if sop.workpiece:
            try:
                self._reference_frame = decode_reference(sop.workpiece)
                self._locator = WorkpieceLocator(sop.workpiece)
                self.tracking_label.setText("已設定工件跟隨；即時影像可檢查定位狀態。")
            except ValueError as exc:
                self.tracking_label.setText(str(exc))
        else:
            self.tracking_label.setText("尚未設定工件跟隨；區域使用固定畫面座標。")
        self.reference_button.setEnabled(self._locator is not None)
        self.cycle_enabled_switch.setChecked(sop.cycle.enabled)
        self.auto_restart_spin.setValue(sop.cycle.auto_restart_sec)
        self.reset_hold_spin.setValue(sop.cycle.reset_hold_sec)
        self.reset_ratio_spin.setValue(round(sop.cycle.reset_ratio * 100))
        self.sequence_check.setChecked(sop.cycle.detect_sequence)
        self._rebuild_steps()
        self._refresh_roi_rows()
        self._refresh_options()
        self._loading = False
        self._refresh_setup_summary()
        self._update_cycle_summary()
        self._update_live()
        self._update_validation_view()
        self.workpiece_changed.emit(sop.workpiece)

    def set_classes(self, classes):
        self.classes = list(classes)
        self._refresh_options()
        self._refresh_step_summaries()

    def set_model_state(self, state, detail=""):
        if state not in ("unselected", "selected", "loading", "available", "failed"):
            raise ValueError(f"Unknown model state: {state}")
        self._model_state, self._model_detail = state, detail
        self._set_model_label(self.sop.model_path)
        self._refresh_setup_summary()
        if state != "available":
            self._invalidate_validation("模型狀態已變更，請重新試跑")
        self._update_validation_view()

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            widget = layout.takeAt(0).widget()
            if widget:
                widget.hide()
                widget.deleteLater()

    def _step_index(self, step):
        return next((i for i, candidate in enumerate(self.sop.steps) if candidate is step), -1)

    def _keep_scroll(self, reveal=None):
        """Rebuilt rows must not move the page: restore the reading position once
        the new rows are laid out, then scroll only as far as a new row needs."""
        bar = self.page_scroll.verticalScrollBar()
        value = bar.value()

        def restore():
            relayout_now(self.page_content)
            QApplication.sendEvent(self.page_scroll, QEvent(QEvent.Type.LayoutRequest))
            bar.setValue(min(value, bar.maximum()))
            if reveal is not None and not reveal.isHidden():
                self.page_scroll.ensureWidgetVisible(reveal, 0, 40)
        QTimer.singleShot(0, restore)

    def _rebuild_steps(self, target=None, reveal=None):
        scroll = self.page_scroll.verticalScrollBar().value()
        self._clear_layout(self.steps_layout)
        self.step_editors = []
        names = [roi.name for roi in self.sop.rois]
        for step in self.sop.steps:
            editor = StepEditor(step, self.classes, names)
            editor.changed.connect(lambda s=step: self._on_step_form(s))
            editor.region_selected.connect(self._highlight_condition_region)
            editor.duplicate_requested.connect(lambda s=step: self._duplicate_step(self._step_index(s)))
            editor.delete_requested.connect(lambda s=step: self._delete_step(self._step_index(s)))
            editor.move_requested.connect(lambda delta, s=step: self._move_step(delta, self._step_index(s)))
            editor.trial_requested.connect(lambda s=step: self._start_step_validation(self._step_index(s)))
            editor.expanded_changed.connect(lambda expanded, s=step: self._on_step_expanded(s, expanded))
            editor.set_expanded(id(step) in self._expanded)
            individual = not self._editing_steps and step is self._individual_step
            editor.set_editing(self._editing_steps or individual, individual=individual)
            editor.reorder_requested.connect(lambda token, after, s=step: self._drop_step(token, s, after))
            self.step_editors.append(editor)
            self.steps_layout.addWidget(editor)
            # Show at once: a queued show would let the page shrink and reset the scroll.
            editor.setVisible(True)
        self.page_scroll.verticalScrollBar().setValue(scroll)
        self._keep_scroll(next((editor for editor in self.step_editors if editor.step is reveal), None))
        self._expanded &= {id(step) for step in self.sop.steps}
        self.empty_steps_label.setVisible(not self.sop.steps)
        self._update_steps_header()
        self._refresh_trial_steps(target)
        self._refresh_step_summaries()
        self._update_live()

    def _on_step_expanded(self, step, expanded):
        (self._expanded.add if expanded else self._expanded.discard)(id(step))
        if not expanded and step is self._individual_step and not self._editing_steps:
            self._individual_step = None
            for editor in self.step_editors:
                if editor.step is step:
                    editor.set_editing(False)
        self._update_steps_header()

    def _set_all_expanded(self, expanded):
        self._expanded = {id(step) for step in self.sop.steps} if expanded else set()
        for editor in self.step_editors:
            editor.set_expanded(expanded)
        self._update_steps_header()

    def _open_only(self, step):
        """A new step opens for editing; the others collapse to their summaries."""
        if not self._editing_steps:
            self._individual_step = step
        self._expanded = {id(step)}

    def _set_steps_editing(self, editing):
        self._editing_steps = bool(editing)
        self._individual_step = None
        if editing:
            self._set_all_expanded(True)
        else:
            self._set_all_expanded(False)
        for editor in self.step_editors:
            editor.set_editing(editing)
        self._update_steps_header()

    def _update_steps_header(self):
        self.edit_steps_button.setText("完成編輯" if self._editing_steps else "編輯")
        self.edit_steps_button.setEnabled(bool(self.sop.steps))

    def _drop_step(self, token, target, after):
        source = next((step for step in self.sop.steps if str(id(step)) == token), None)
        if source is None or source is target or self._step_index(target) < 0:
            return
        index = self._step_index(source)
        destination = self._step_index(target) + int(after)
        if index < destination:
            destination -= 1
        if index == destination:
            return
        selected = self._current_step()
        self.sop.steps.pop(index)
        self.sop.steps.insert(destination, source)
        self._clear_roi_history()
        self._rebuild_steps(selected)
        self._mark_modified()

    def _refresh_trial_steps(self, target=None):
        old = self.validation_step.currentIndex()
        self.validation_step.blockSignals(True)
        self.validation_step.clear()
        for i, step in enumerate(self.sop.steps):
            self.validation_step.addItem(f"工序 {i + 1} · {step.name}", i)
        row = self._step_index(target) if target is not None else old
        self.validation_step.setCurrentIndex(min(max(0, row), len(self.sop.steps) - 1))
        self.validation_step.blockSignals(False)

    def _current_step(self):
        row = self.validation_step.currentIndex()
        return self.sop.steps[row] if 0 <= row < len(self.sop.steps) else None

    def _on_step_form(self, step):
        if self._loading or self._step_index(step) < 0:
            return
        self._clear_roi_history()
        self._refresh_trial_steps()
        self._refresh_step_summaries()
        self._refresh_video_rois()
        self._update_live()
        self._mark_modified()

    def _add_step(self, index=None):
        insert_at = len(self.sop.steps) if index is None else max(0, min(len(self.sop.steps), index + 1))
        step = Step(name=f"工序 {len(self.sop.steps) + 1}")
        self.sop.steps.insert(insert_at, step)
        self._clear_roi_history()
        self._open_only(step)
        self._rebuild_steps(step, reveal=step)
        self._mark_modified()
        self.step_editors[insert_at].step_name_edit.setFocus()
        self.step_editors[insert_at].step_name_edit.selectAll()

    def _duplicate_step(self, index=None):
        index = self.validation_step.currentIndex() if index is None else index
        if not 0 <= index < len(self.sop.steps):
            return
        clone = copy.deepcopy(self.sop.steps[index])
        clone.name += "（複製）"
        self.sop.steps.insert(index + 1, clone)
        self._clear_roi_history()
        self._open_only(clone)
        self._rebuild_steps(clone, reveal=clone)
        self._mark_modified()

    def _delete_step(self, index=None):
        index = self.validation_step.currentIndex() if index is None else index
        if not 0 <= index < len(self.sop.steps):
            return
        if QMessageBox.question(self, "刪除工序", f"確定刪除「{self.sop.steps[index].name}」？") != QMessageBox.StandardButton.Yes:
            return
        del self.sop.steps[index]
        self._clear_roi_history()
        self._rebuild_steps()
        self._mark_modified()

    def _move_step(self, delta, index=None):
        index = self.validation_step.currentIndex() if index is None else index
        if index < 0 or not 0 <= index + delta < len(self.sop.steps):
            return
        target = self._current_step()
        self.sop.steps[index], self.sop.steps[index + delta] = self.sop.steps[index + delta], self.sop.steps[index]
        self._clear_roi_history()
        self._rebuild_steps(target)
        self._mark_modified()

    def _refresh_step_summaries(self):
        for i, editor in enumerate(self.step_editors):
            editor.set_position(i, len(self.sop.steps))
            editor.set_issues(self._step_issues(editor.step))
        self._refresh_setup_summary()

    def _refresh_setup_summary(self):
        invalid = sum(bool(self._step_issues(step)) for step in self.sop.steps)
        manual = sum(not step.conditions and not self._step_issues(step) for step in self.sop.steps)
        model = {"unselected": "未選擇", "selected": "已選路徑，尚未載入", "loading": "載入中", "available": "可用", "failed": "失敗"}[self._model_state]
        self.setup_summary.setText(f"模型：{model}　工序 {len(self.sop.steps) - invalid}/{len(self.sop.steps)} 已設定"
                                   f"（自動 {len(self.sop.steps) - invalid - manual} · 人工確認 {manual} · 待修正 {invalid}）　作業區域 {len(self.sop.rois)} 個")
        self.setup_summary.setToolTip(self._model_detail)
        set_state(self.setup_summary, "warning" if invalid or self._model_state == "failed" else "neutral")
        self.model_state_label.setText(f"{model}" + (f"：{self._model_detail}" if self._model_detail else ""))
        set_state(self.model_state_label, {"available": "ok", "failed": "ng", "loading": "running"}.get(self._model_state, "neutral"))
        self.clear_target_button.setEnabled(self.sop.workpiece is not None)

    def _refresh_options(self):
        names = [roi.name for roi in self.sop.rois]
        for editor in self.step_editors:
            editor.set_options(self.classes, names)
        self.reset_table.set_options(self.classes, names)
        if self.reset_table.conditions() != self.sop.cycle.reset_conditions:
            self.reset_table.set_conditions(self.sop.cycle.reset_conditions)

    # ---- live preview ------------------------------------------------------
    def on_packet(self, packet):
        result = self._tracked(packet.result)
        # Pipeline source reads hand over a new array and overlays draw on a
        # copy. Keep those owners here; copy full pixels only when freezing.
        self._latest_frame, self._latest_preview = packet.frame, packet.preview_display
        self._latest_arrived_at = time.monotonic()
        self._latest_key = result.workpiece_key
        self._latest_matrix = self._valid_edit_matrix(result.workpiece_transform)
        if self.reference_button.isChecked():
            if self._edit_frame_kind == "selecting":
                self._show_following_selection()
            return
        if not self.freeze_check.isChecked() or self._packet is None:
            self._packet_arrived_at = time.monotonic()
            self._packet = packet
            self._preview_result = result
            if self._locator:
                self.tracking_label.setText(self._preview_result.tracking_status)
            self.video.set_frame(packet.preview_display)
            self._refresh_video_rois()
            if self._validation.running:
                try:
                    for result in packet.all_results():
                        self._validation.update(self._tracked(result))
                        if not self._validation.running:
                            break
                except Exception as exc:
                    self._invalidate_validation(f"試跑發生錯誤：{exc}")
        # Only the explanatory one-frame text is sampled. Trial accumulation
        # above still consumes every background result in packet.all_results().
        now = time.monotonic()
        if self._last_live_preview_at is None or now - self._last_live_preview_at >= self.LIVE_PREVIEW_INTERVAL_SEC:
            self._update_live()
            self._last_live_preview_at = now
        self._update_validation_view()

    @staticmethod
    def _step_preview(step, judged):
        """One-frame result in the production engine's terms; time windows are not accumulated."""
        errors = list(dict.fromkeys(check.error for check in judged.completion_results + judged.forbidden_results
                                    if check.error))
        if not step.conditions:
            parts, state = ["需手動確認：未設定自動完成條件"], "neutral"
        elif judged.completion_blocked:
            parts, state = ["完成條件本幀無法判斷：正式執行暫停累積"], "warning"
        elif judged.completion_met:
            parts, state = [f"本幀成立，尚未代表工序完成（時間窗 {step.hold_sec:g} 秒、成立比例至少 {step.ratio:.0%}；"
                            "預覽未累積時間）"], "ok"
        else:
            parts, state = ["本幀條件未成立"], "neutral"
        if step.forbidden:
            hold, ratio = violation_window(step)
            if judged.forbidden_blocked:
                parts.append("違規條件本幀無法判斷")
                state = "warning" if state != "ng" else state
            elif judged.forbidden_met:
                parts.append(f"本幀違規條件成立（持續 {hold:g} 秒、比例至少 {ratio:.0%} 即警報）")
                state = "ng"
            else:
                parts.append("違規條件本幀未成立")
        if errors:
            parts.append("；".join(errors))
        return "單幀預覽：" + " · ".join(parts), state

    def _update_live(self):
        available = self._packet is not None and self._preview_result is not None and not self.reference_button.isChecked()
        if not available:
            text = "參考影像編輯中；返回即時影像後顯示判定。" if self.reference_button.isChecked() else "單幀預覽：等待影像"
            for editor in self.step_editors:
                editor.condition_table.set_live_results(None)
                editor.forbidden_table.set_live_results(None)
                editor.set_preview(text, "neutral")
            self.reset_table.set_live_results(None)
            self.live_label.setText(text)
            for row in self.roi_rows:
                row.set_status("本幀 —")
            return
        rois, result = self.sop.roi_map(), self._preview_result
        _, reset_results = evaluate_conditions(self.sop.cycle.reset_conditions, result, rois)
        self.reset_table.set_live_results(reset_results)
        positives, violations, roi_checks = 0, 0, {}
        for editor in self.step_editors:
            step = editor.step
            judged = evaluate_step(step, result, rois)
            editor.condition_table.set_live_results(judged.completion_results)
            editor.forbidden_table.set_live_results(judged.forbidden_results)
            for condition, check in zip(step.conditions, judged.completion_results):
                if condition.roi:
                    roi_checks.setdefault(condition.roi, []).append(check)
            positives += bool(step.conditions and judged.completion_met)
            violations += bool(step.forbidden and judged.forbidden_met)
            editor.set_preview(*self._step_preview(step, judged))
        self.live_label.setText(f"單幀預覽：{positives} 道工序完成條件成立 · {violations} 道違規條件成立")
        statuses = []
        for row in self.roi_rows:
            checks = roi_checks.get(row.roi.name, [])
            status = ("尚未用於完成條件" if not checks else "定位／辨識無法確認" if any(c.error for c in checks)
                      else f"{sum(c.met for c in checks)}/{len(checks)} 條件本幀成立")
            row.set_status(status)
            statuses.append(f"{row.roi.name}：{status}")
        self.roi_status.setText("\n".join(statuses) or "尚未建立作業區域。")

    def _start_step_validation(self, index):
        if not 0 <= index < len(self.sop.steps):
            return
        self.validation_mode.setCurrentIndex(self.validation_mode.findData("single"))
        self.validation_step.setCurrentIndex(index)
        self._start_validation()
        self.tools_scroll.ensureWidgetVisible(self.validation_box)

    def set_production_running(self, running):
        self._production_running = bool(running)
        if running and self._image_focus:
            self._set_image_focus(False)
        self.settings_content.setEnabled(not running)
        for widget in (self.tracking_tools, self.regions_panel, self.view_bar):
            widget.setEnabled(not running)
        self.validation_mode.setEnabled(not running)
        self.validation_step.setEnabled(not running and self.validation_mode.currentData() == "single")
        if running:
            self.draw_button.setChecked(False)
            self.video.set_draw_mode(False)
            self._invalidate_validation("正式監控已啟動；請停止監控後重新試跑")
        self.guide_label.setText("正式監控執行中，工序設定與試跑已鎖定。" if running else "正式監控已停止，可編輯設定並試跑。")
        self._refresh_video_rois()
        self._update_validation_view()

    # ---- regions -----------------------------------------------------------
    def _set_rois_editing(self, editing):
        self._editing_rois = bool(editing)
        self.edit_rois_button.setText("完成編輯" if editing else "編輯區域")
        self.roi_edit_toolbar.setVisible(editing)
        self.roi_edit_note.setVisible(editing)
        for row in self.roi_rows:
            row.set_editing(editing)
        if not editing:
            self.draw_button.setChecked(False)
        self._refresh_video_rois()

    def _refresh_roi_rows(self):
        self._clear_layout(self.rois_layout)
        self.roi_rows = []
        for roi in self.sop.rois:
            row = ROIEditorRow(roi)
            row.name_edit.editingFinished.connect(lambda r=roi, w=row: self._rename_roi(r, w.name_edit.text()))
            row.anchor_combo.currentIndexChanged.connect(lambda _index, r=roi, w=row: self._change_roi_anchor(r, w.anchor_combo.currentData()))
            row.edit_button.clicked.connect(lambda _checked=False, r=roi: self._begin_roi_edit(r))
            row.duplicate_button.clicked.connect(lambda _checked=False, r=roi: self._duplicate_roi(r))
            row.delete_button.clicked.connect(lambda _checked=False, r=roi: self._delete_roi(r))
            row.set_editing(self._editing_rois)
            self.roi_rows.append(row)
            self.rois_layout.addWidget(row)
            row.setVisible(True)
        self.empty_rois_label.setVisible(not self.sop.rois)
        self._update_roi_history_buttons()
        self._refresh_video_rois()
        self._refresh_step_summaries()
        self._update_live()

    def _selected_roi(self):
        return next((roi for roi in self.sop.rois if roi.name == self._selected_roi_name), None)

    def _roi_argument(self, roi):
        if isinstance(roi, str):
            return next((candidate for candidate in self.sop.rois if candidate.name == roi), None)
        return roi or self._selected_roi()

    def _select_roi(self, name):
        self._selected_roi_name = self._highlighted_condition_roi = name
        self._refresh_video_rois()

    def _highlight_condition_region(self, name):
        self._selected_roi_name = None
        self._highlighted_condition_roi = name or None
        self._refresh_video_rois()

    def _refresh_video_rois(self):
        selected = self._selected_roi_name or self._highlighted_condition_roi
        if self.reference_button.isChecked():
            matrix = (self._edit_matrix if self._edit_frame_kind in ("latest", "reference")
                      and self._edit_key == self._workpiece_key else
                      self._latest_matrix if self._edit_frame_kind == "selecting"
                      and self._latest_following_frame_valid() else None)
            rois = display_rois([roi for roi in self.sop.rois if roi.anchor == "workpiece"], matrix)
        else:
            # Returning from editing restores the latest preview immediately,
            # but no detection result is revived for evaluation or trials.
            matrix = (self._preview_result.workpiece_transform if self._preview_result else
                      self._latest_matrix if not self.freeze_check.isChecked()
                      and self._latest_following_frame_valid() else None)
            rois = display_rois(self.sop.rois, matrix)
        used = {c.roi for c in self._all_conditions() if c.roi}
        self.video.set_rois(rois, used, selected)
        self.video.set_regions_visible(self._regions_visible or self._editing_rois
                                       or self.reference_button.isChecked() or self.draw_button.isChecked()
                                       or self._capture_target)
        editable = [r.name for r in self.sop.rois if
                    (r.anchor == "workpiece" and self._following_edit_ready()) or
                    (r.anchor == "fixed" and self.freeze_check.isChecked() and not self.reference_button.isChecked())]
        self.video.set_editable(editable if self._editing_rois and not self._capture_target and not self._production_running else [])

    def set_regions_visible(self, visible):
        self._regions_visible = bool(visible)
        self._refresh_video_rois()

    def _roi_snapshot(self):
        return copy.deepcopy(self.sop.rois), [c.roi for c in self._all_conditions()]

    def _remember_roi(self):
        self._roi_undo.append(self._roi_snapshot())
        self._roi_undo = self._roi_undo[-50:]
        self._roi_redo.clear()
        self._update_roi_history_buttons()

    def _clear_roi_history(self):
        self._roi_undo.clear()
        self._roi_redo.clear()
        self._update_roi_history_buttons()

    def _update_roi_history_buttons(self):
        self.undo_roi_button.setEnabled(bool(self._roi_undo))
        self.redo_roi_button.setEnabled(bool(self._roi_redo))

    def _restore_roi(self, source, destination):
        if not source:
            self.guide_label.setText("沒有可復原／重做的區域修改。")
            return
        destination.append(self._roi_snapshot())
        self.sop.rois, refs = source.pop()
        for condition, name in zip(self._all_conditions(), refs):
            condition.roi = name
        self._refresh_roi_rows()
        self._sync_condition_references()
        self._update_live()
        self._mark_modified()

    def _undo_roi(self):
        self._restore_roi(self._roi_undo, self._roi_redo)

    def _redo_roi(self):
        self._restore_roi(self._roi_redo, self._roi_undo)

    def _sync_condition_references(self):
        self._refresh_options()

    def _rename_roi(self, roi, text):
        if not any(candidate is roi for candidate in self.sop.rois):
            return
        old, new = roi.name, text.strip()
        if new == old:
            return
        if not new or any(r is not roi and r.name == new for r in self.sop.rois):
            QMessageBox.warning(self, "無法重新命名", "名稱不可空白或重複")
            self._refresh_roi_rows()
            return
        self._remember_roi()
        roi.name = new
        for row in self.roi_rows:
            if row.roi is roi:
                row.name_edit.setText(new)
        for condition in self._all_conditions():
            if condition.roi == old:
                condition.roi = new
        if self._selected_roi_name == old:
            self._selected_roi_name = new
        self._sync_condition_references()
        self._refresh_video_rois()
        self._refresh_step_summaries()
        self._update_live()
        self._mark_modified()

    def _rename_selected_roi(self, name=None):
        if name:
            self._select_roi(name)
        roi = self._selected_roi()
        row = next((row for row in self.roi_rows if row.roi is roi), None)
        if row:
            self.tools_scroll.ensureWidgetVisible(row)
            row.name_edit.setFocus()
            row.name_edit.selectAll()

    def _begin_roi_edit(self, roi=None):
        if self._production_running:
            return False
        roi = self._roi_argument(roi)
        if roi is None:
            self.guide_label.setText("請按欲調整區域旁的移動／縮放。")
            return False
        self.edit_rois_button.setChecked(True)
        self._select_roi(roi.name)
        if roi.anchor == "workpiece":
            if self._reference_frame is None:
                self.guide_label.setText("請先設定有效的參考工件。")
                return False
            self.reference_button.setChecked(True)
            if not self._following_edit_ready():
                self.guide_label.setText("請等待定位有效並按「使用這一幀」，或明確選用「使用原始參考影像」，再調整跟隨區域。")
                return False
        else:
            if self.reference_button.isChecked():
                self._return_live()
            if self._packet is None:
                self.guide_label.setText("請等待即時影像，再調整固定區域。")
                return False
            self.freeze_check.setChecked(True)
        self.draw_button.setChecked(False)
        self._refresh_video_rois()
        self._focus_image()
        self.guide_label.setText(f"編輯「{roi.name}」：畫面已暫停並放大。拖曳框內移動、四角縮放；完成後按「返回即時影像」。")
        return True

    def _edit_roi_points(self, name, points):
        if self._production_running:
            return
        roi = next((r for r in self.sop.rois if r.name == name), None)
        if roi is None:
            return
        if roi.anchor == "workpiece":
            points = self._following_points_to_reference(points)
            if points is None:
                return
        elif self.reference_button.isChecked():
            return
        self._remember_roi()
        roi.points = [tuple(p) for p in points]
        self._refresh_video_rois()
        self._mark_modified()

    def _duplicate_roi(self, roi=None):
        roi = self._roi_argument(roi)
        if roi is None:
            return
        self._remember_roi()
        names = {r.name for r in self.sop.rois}
        name = next(f"{roi.name}（副本{i}）" for i in range(1, 10000) if f"{roi.name}（副本{i}）" not in names)
        clone = ROI(name, list(roi.points), roi.anchor)
        self.sop.rois.append(clone)
        self._selected_roi_name = name
        self._refresh_roi_rows()
        self._refresh_options()
        self._mark_modified()
        self._begin_roi_edit(clone)
        self.guide_label.setText("已複製區域，框線與原區域重疊；可移動新區域並直接修改名稱。")

    def _delete_roi(self, roi=None):
        roi = self._roi_argument(roi)
        if roi is None or not any(candidate is roi for candidate in self.sop.rois):
            return
        name = roi.name
        if any(c.roi == name for c in self._all_conditions()):
            affected = [f"工序 {i + 1} · {step.name}" for i, step in enumerate(self.sop.steps) if any(c.roi == name for c in step.conditions + step.forbidden)]
            if any(c.roi == name for c in self.sop.cycle.reset_conditions):
                affected.append("循環重置條件")
            QMessageBox.information(self, "區域仍在使用", "請先修改以下條件的區域，再刪除：\n" + "\n".join(affected))
            return
        if QMessageBox.question(self, "刪除作業區域", f"確定刪除區域「{name}」？") != QMessageBox.StandardButton.Yes:
            return
        self._remember_roi()
        self.sop.rois.remove(roi)
        self._refresh_roi_rows()
        self._refresh_options()
        self._mark_modified()

    def _change_roi_anchor(self, roi, anchor):
        if anchor == roi.anchor:
            return
        matrix = self._preview_result.workpiece_transform if self._preview_result else None
        if self._locator is None or matrix is None or self.freeze_check.isChecked() or self.reference_button.isChecked():
            self.guide_label.setText("變更定位方式需先設定有效參考工件，返回即時影像並等待定位穩定；目前設定已保留。")
            self._refresh_roi_rows()
            return
        transform = matrix
        if anchor == "workpiece":
            try:
                transform = np.linalg.inv(np.vstack([matrix, [0, 0, 1]]))[:2]
            except np.linalg.LinAlgError:
                self.guide_label.setText("目前定位無效，請等待重新定位。")
                self._refresh_roi_rows()
                return
        points = np.asarray(roi.points) @ transform[:, :2].T + transform[:, 2]
        if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1):
            self.guide_label.setText("轉換後的區域超出有效影像範圍；原設定已保留，請調整工件或區域位置後再試。")
            self._refresh_roi_rows()
            return
        self._remember_roi()
        roi.points = [tuple(p) for p in points]
        roi.anchor = anchor
        self._refresh_roi_rows()
        self._update_live()
        self._mark_modified()

    def _toggle_draw(self, enabled):
        if enabled and (self._production_running or
                        (self.reference_button.isChecked() and not self._following_edit_ready())):
            self.draw_button.setChecked(False)
            self.guide_label.setText("請先按「使用這一幀」固定有效定位畫面，或選用「使用原始參考影像」。")
            return
        if enabled and not self._capture_target and not self.reference_button.isChecked():
            if self._packet is None:
                self.draw_button.setChecked(False)
                self.guide_label.setText("請先連線影像來源，才能框選區域。")
                return
            self.freeze_check.setChecked(True)
        self.video.set_draw_mode(enabled)
        self._refresh_video_rois()
        if enabled:
            self._focus_image()
            self.guide_label.setText("畫面已暫停並放大。拖曳框選區域，放開後在「作業區域」直接修改名稱；完成後按「返回即時影像」。")

    def _on_freeze(self, frozen):
        self._last_live_preview_at = None
        if frozen:
            self._invalidate_validation("畫面已凍結，請返回即時後重新試跑")
        else:
            self._capture_target = False
            self.draw_button.setChecked(False)
            self.target_button.setText("框選參考工件")
            self._packet = self._preview_result = self._packet_arrived_at = None
            if not self.reference_button.isChecked() and self._latest_preview is not None:
                self.video.set_frame(self._latest_preview)
            self._release_auto_focus()
        self._refresh_video_rois()

    def _on_roi_drawn(self, points):
        if self._production_running:
            return
        self.draw_button.setChecked(False)
        self.video.set_draw_mode(False)
        if self._capture_target:
            try:
                target = capture_workpiece(self._packet.frame, points)
                locator, reference = WorkpieceLocator(target), decode_reference(target)
            except (ValueError, AttributeError) as exc:
                QMessageBox.warning(self, "無法設定參考工件", str(exc))
                self.draw_button.setChecked(True)
                self.guide_label.setText("框選未成功，可重新拖曳；按「返回即時影像」可取消。")
                return
            self._capture_target = False
            self.sop.workpiece = target
            self._workpiece_key = workpiece_key(target)
            self.workpiece_changed.emit(target)
            self._clear_roi_history()
            self._locator, self._reference_frame = locator, reference
            self.reference_button.setEnabled(True)
            self.reference_button.setChecked(True)
            self._use_original_reference()
            self.guide_label.setText("參考工件已設定。可在參考影像框選跟隨區域，或按「返回即時影像」。")
            self._refresh_setup_summary()
            self._mark_modified()
            return
        if self.reference_button.isChecked():
            points = self._following_points_to_reference(points)
            if points is None:
                return
        existing = {roi.name for roi in self.sop.rois}
        name = next(f"區域{i}" for i in range(1, 10000) if f"區域{i}" not in existing)
        self._remember_roi()
        anchor = "workpiece" if self.reference_button.isChecked() else "fixed"
        self.sop.rois.append(ROI(name, [tuple(p) for p in points], anchor))
        self._selected_roi_name = name
        self._refresh_roi_rows()
        self._refresh_options()
        row = self.roi_rows[-1]  # named in the tools column, beside the enlarged image
        self.tools_scroll.ensureWidgetVisible(row)
        row.name_edit.setFocus()
        row.name_edit.selectAll()
        self.guide_label.setText(f"已新增「{name}」，可在「作業區域」該列直接修改名稱；各工序的區域欄位可直接選用。")
        self._mark_modified()

    def _begin_target(self):
        if self._capture_target:
            self._return_live()
            return
        if any(r.anchor == "workpiece" for r in self.sop.rois):
            QMessageBox.information(self, "已有跟隨區域", "重設參考工件會改變座標基準。請先清除工件跟隨設定，再重新框選。")
            return
        if self._packet is None:
            QMessageBox.information(self, "等待影像", "請先連線影像來源並顯示即時影像。")
            return
        packet = self._packet
        self.reference_button.setChecked(False)
        self._packet = packet
        self.freeze_check.setChecked(True)
        self.video.set_frame(packet.frame)
        self._capture_target = True
        self.target_button.setText("取消框選參考工件")
        self.draw_button.setChecked(True)
        self.video.set_draw_mode(True)
        self._focus_image()
        self.guide_label.setText("畫面已暫停並放大。拖曳框住完整工件，盡量排除手部與背景。")

    def _clear_workpiece(self, restart=False):
        if self.sop.workpiece is None:
            return
        dependent = {r.name for r in self.sop.rois if r.anchor == "workpiece"}
        affected = sum(c.roi in dependent for c in self._all_conditions())
        if QMessageBox.question(self, "清除工件跟隨設定",
                f"確定清除參考工件、{len(dependent)} 個跟隨區域及 {affected} 個引用條件？\n固定區域與工序會保留。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        packet = self._packet
        self._return_live()
        self.sop.workpiece = None
        self.sop.rois = [r for r in self.sop.rois if r.name not in dependent]
        for step in self.sop.steps:
            step.conditions = [c for c in step.conditions if c.roi not in dependent]
            step.forbidden = [c for c in step.forbidden if c.roi not in dependent]
        self.sop.cycle.reset_conditions = [c for c in self.sop.cycle.reset_conditions if c.roi not in dependent]
        self._locator = self._reference_frame = None
        self._workpiece_key = ""
        self.workpiece_changed.emit(None)
        self.reference_button.setEnabled(False)
        self._clear_roi_history()
        self._rebuild_steps()
        self._refresh_roi_rows()
        self._sync_condition_references()
        self._update_cycle_summary()
        self._update_live()
        self.tracking_label.setText("尚未設定工件跟隨；固定區域與工序已保留。")
        self.guide_label.setText("已清除工件跟隨設定。需人工確認的工序會直接顯示在原位置。")
        self._mark_modified()
        if restart and packet is not None:
            self._packet = packet
            self._begin_target()

    @staticmethod
    def _valid_edit_matrix(matrix):
        """A normalized reference-to-frame affine, never a stale/singular pose."""
        try:
            value = np.asarray(matrix, dtype=float)
            if value.shape == (2, 3) and np.isfinite(value).all() and abs(np.linalg.det(value[:, :2])) > 1e-10:
                return value.copy()
        except (TypeError, ValueError, np.linalg.LinAlgError):
            pass
        return None

    def _clear_following_frames(self):
        self._latest_frame = self._latest_preview = self._latest_matrix = None
        self._latest_arrived_at = None
        self._latest_key = ""
        self._edit_frame = self._edit_matrix = None
        self._edit_frame_kind = self._edit_key = ""

    def _latest_following_frame_valid(self):
        return (self.sop.workpiece is not None and self._latest_frame is not None
                and self._latest_matrix is not None and self._latest_key == self._workpiece_key
                and self._latest_arrived_at is not None
                and time.monotonic() - self._latest_arrived_at <= 3)

    def _following_edit_ready(self):
        return (self.reference_button.isChecked() and not self._production_running
                and self._edit_frame_kind in ("latest", "reference")
                and self._edit_frame is not None and self._edit_matrix is not None
                and self._edit_key == self._workpiece_key and self.sop.workpiece is not None)

    def _show_following_selection(self):
        if self._latest_frame is not None:
            self.video.set_frame(self._latest_frame)
        else:
            self.video.clear_frame("等待即時影像；可選用原始參考影像")
        valid = self._latest_following_frame_valid()
        self.use_frame_button.setEnabled(valid and not self._production_running)
        detail = "定位有效；按「使用這一幀」固定後才可編輯。" if valid else "尚無新鮮有效定位；目前不能變更跟隨區域。"
        self.following_frame_label.setText("重新取景中（即時預覽）：" + detail)
        self._refresh_video_rois()

    def _retake_following_frame(self):
        if self._production_running or not self.reference_button.isChecked():
            return
        self.draw_button.setChecked(False)
        self.video.set_draw_mode(False)
        self._edit_frame = self._edit_matrix = None
        self._edit_key = ""
        self._edit_frame_kind = "selecting"
        self._show_following_selection()
        self.guide_label.setText("重新取景：調整工件或等待所需畫面，再按「使用這一幀」。既有區域與工序條件會保留。")

    def _freeze_following_frame(self, frame, matrix, kind):
        # Own the selected pixels and the pose belonging to these exact pixels;
        # subsequent packets must never change a frozen selection.
        self._edit_frame, self._edit_matrix = frame.copy(), matrix.copy()
        self._edit_frame_kind, self._edit_key = kind, self._workpiece_key
        self.video.set_frame(self._edit_frame)
        self.use_frame_button.setEnabled(False)
        text = "正在使用最新有效定位的凍結畫面" if kind == "latest" else "正在使用原始參考影像"
        self.following_frame_label.setText(text + "；修改會轉回原始參考座標，保留工序條件。")
        self.guide_label.setText(text + "。可新增或編輯跟隨區域；「重新取景」可選另一幀，完成後按「返回即時影像」。")
        self._refresh_video_rois()

    def _use_latest_following_frame(self):
        if self._production_running or not self.reference_button.isChecked():
            return False
        if not self._latest_following_frame_valid():
            if self._edit_frame_kind == "selecting":
                self._show_following_selection()
            self.guide_label.setText("此幀沒有新鮮有效定位，無法固定；請重新取景，或選用原始參考影像。")
            return False
        self._freeze_following_frame(self._latest_frame, self._latest_matrix, "latest")
        return True

    def _use_original_reference(self):
        if self._production_running or not self.reference_button.isChecked() or self._reference_frame is None:
            return False
        self.draw_button.setChecked(False)
        self.video.set_draw_mode(False)
        self._freeze_following_frame(self._reference_frame, np.eye(2, 3), "reference")
        return True

    def _following_points_to_reference(self, points):
        if not self._following_edit_ready():
            self.guide_label.setText("尚未固定有效定位畫面；請先使用這一幀或原始參考影像，原區域已保留。")
            return None
        try:
            displayed = np.asarray(points, dtype=float)
            if displayed.ndim != 2 or displayed.shape[1] != 2 or len(displayed) < 3:
                raise ValueError("invalid points")
            inverse = np.linalg.inv(np.vstack([self._edit_matrix, [0., 0., 1.]]))[:2]
            canonical = displayed @ inverse[:, :2].T + inverse[:, 2]
            # Affine roundtrips can place an exact 0/1 corner a few ulps
            # outside. Accept numerical noise only, never a real overflow.
            edge_epsilon = 1e-9
            if (not np.isfinite(displayed).all() or not np.isfinite(canonical).all()
                    or np.any(displayed < 0) or np.any(displayed > 1)
                    or np.any(canonical < -edge_epsilon) or np.any(canonical > 1 + edge_epsilon)):
                raise ValueError("outside reference")
            canonical[canonical < 0] = 0.
            canonical[canonical > 1] = 1.
        except (ValueError, TypeError, np.linalg.LinAlgError):
            self.guide_label.setText("區域轉回原始參考座標後無效或超出影像；原設定已保留，請重新調整。")
            return None
        return [tuple(point) for point in canonical]

    def _show_reference(self, enabled):
        self._capture_target = False
        self.target_button.setText("框選參考工件")
        self.draw_button.setChecked(False)
        self.video.set_draw_mode(False)
        if enabled:
            if self._production_running or self._reference_frame is None:
                self.reference_button.setChecked(False)
                return
            self._invalidate_validation("跟隨區域編輯中，請返回即時後重新試跑")
            self.freeze_check.setChecked(False)
            self.freeze_check.setEnabled(False)
            self.following_frame_tools.show()
            # Release live result/detection owners while editing; only the
            # lightweight raw/preview/pose cache continues to receive frames.
            self._packet = self._preview_result = self._packet_arrived_at = None
            self._focus_image()
            if not self._use_latest_following_frame():
                self._retake_following_frame()
        else:
            self._edit_frame = self._edit_matrix = None
            self._edit_frame_kind = self._edit_key = ""
            self.following_frame_tools.hide()
            self.freeze_check.setEnabled(True)
            self.freeze_check.setChecked(False)
            if self._latest_preview is not None:
                self.video.set_frame(self._latest_preview)
            else:
                self.video.clear_frame("等待即時影像")
        self._refresh_video_rois()
        self._update_live()

    def _return_live(self):
        self._last_live_preview_at = None
        self.reference_button.setChecked(False)
        self.freeze_check.setEnabled(True)
        self.freeze_check.setChecked(False)
        self._capture_target = False
        self.draw_button.setChecked(False)
        self.video.set_draw_mode(False)
        self.video.set_editable([])
        if self._latest_preview is not None:
            self.video.set_frame(self._latest_preview)
        self._refresh_video_rois()
        self.target_button.setText("框選參考工件")
        self._release_auto_focus()
        self.guide_label.setText("即時影像：可框選參考工件與區域，所有工序的判定預覽同時更新。")

    # ---- trial -------------------------------------------------------------
    def _validation_blocker(self):
        if self._production_running:
            return "請先停止正式監控，再開始編輯器試跑。"
        if self._current_step() is None:
            return "請先新增工序。"
        if self.freeze_check.isChecked() or self.reference_button.isChecked():
            return "目前為凍結／參考影像；請返回即時畫面。"
        if self._packet is None or self._packet_arrived_at is None or time.monotonic() - self._packet_arrived_at > 3:
            return "等待即時影像；請先連線影像來源。"
        steps = self.sop.steps if self.validation_mode.currentData() == "flow" else [self._current_step()]
        needs_model = any(step.conditions or step.forbidden for step in steps)
        if self.validation_mode.currentData() == "flow":
            needs_model |= bool(self.sop.cycle.reset_conditions)
        if needs_model and (self._model_state != "available" or self._preview_result is None or not self._preview_result.detection_valid):
            return "模型尚未就緒，無法開始自動條件試跑。"
        return ""

    def _start_validation(self, _checked=False):
        blocker = self._validation_blocker()
        if blocker:
            self._validation.message = blocker
            self._update_validation_view()
            return
        candidate = copy.deepcopy(self.sop)
        if self.validation_mode.currentData() == "single":
            candidate.steps = [candidate.steps[self.validation_step.currentIndex()]]
            candidate.cycle.reset_conditions = []  # Single-step trials do not exercise cycle reset.
        try:
            errors, _warnings = candidate.validate(self.classes or None)
            if errors:
                raise ValueError("；".join(errors))
            self._validation.start(self.sop, self.validation_mode.currentData(),
                                   self.validation_step.currentIndex(), self._preview_result.timestamp)
        except Exception as exc:
            self._validation.invalidate(f"無法開始試跑：{exc}")
            self._validation.message = f"無法開始試跑：{exc}"
        self._update_validation_view()

    def _stop_validation(self, _checked=False):
        self._validation.stop()
        self._update_validation_view()

    def _confirm_validation(self):
        self._validation.confirm()
        self._update_validation_view()

    def _invalidate_validation(self, reason="設定已變更，請重新試跑"):
        self._validation.invalidate(reason)
        if hasattr(self, "validation_status"):
            self._update_validation_view()

    def _validation_scope_changed(self, _index):
        self.validation_step.setEnabled(self.validation_mode.currentData() == "single" and not self._production_running)
        self._invalidate_validation("驗證範圍已變更，請重新試跑")

    def _check_validation_feed(self):
        if self._validation.running and (self._packet_arrived_at is None or time.monotonic() - self._packet_arrived_at > 3):
            self._invalidate_validation("影像停止更新，請重新試跑")
        self._update_validation_view()

    def _update_validation_view(self):
        if not hasattr(self, "validation_status"):
            return
        trial = self._validation
        blocker = self._validation_blocker()
        self.validation_status.setText(trial.message if trial.engine is not None or trial.message.startswith("無法") else blocker or trial.message)
        set_state(self.validation_status, "warning" if trial.stale or blocker else "running" if trial.running else "neutral")
        metrics = trial.metrics()
        self.validation_target.setText(metrics["target"])
        self.validation_count.setText("條件 " + metrics["count"])
        self.validation_elapsed.setText("觀察 " + metrics["elapsed"])
        self.validation_ratio.setText("比例 " + metrics["ratio"].replace("≥", "至少"))
        self.validation_start.setEnabled(not trial.running and not blocker)
        self.validation_stop.setEnabled(trial.running)
        manual = trial.running and trial.engine is not None and trial.engine.phase.value == "running" and not trial.engine.sop.steps[trial.engine.index].conditions
        self.validation_confirm.setEnabled(manual)
        lines = [trial.message]
        lines.extend(event.message for event in trial.events)
        detail = "\n".join(lines)
        if self.validation_details.toPlainText() != detail:
            scrollbar = self.validation_details.verticalScrollBar()
            position, at_end = scrollbar.value(), scrollbar.value() == scrollbar.maximum()
            self.validation_details.setPlainText(detail)
            scrollbar.setValue(scrollbar.maximum() if at_end else position)

    def hideEvent(self, event):
        if self._validation.running:
            self._validation.stop("已離開編輯器，試跑已停止")
        super().hideEvent(event)

    def closeEvent(self, event):
        self._validation.stop()
        self._validation_watchdog.stop()
        super().closeEvent(event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange) and hasattr(self, "_theme_dark"):
            self._apply_workbench_theme()

    def _tracked(self, result: FrameResult) -> FrameResult:
        """只採用與目前參考工件相符的背景定位結果，避免套用舊設定的位置。"""
        if self.sop.workpiece is None or self._locator is None:
            return replace(result, workpiece_transform=None)
        if result.workpiece_key != self._workpiece_key:
            return replace(result, workpiece_transform=None, tracking_status="定位設定同步中…")
        return result

    def clear_video(self, text: str):
        self._last_live_preview_at = None
        self._invalidate_validation("影像來源已變更或中斷，請重新試跑")
        self._clear_following_frames()
        self._return_live()
        self._packet = None
        self._preview_result = None
        self._packet_arrived_at = None
        self.video.clear_frame(text)
        self._refresh_video_rois()
        self._update_live()
        self._update_validation_view()

    def _step_issues(self, step: Step) -> list[str]:
        issues = []
        if step.completion_mode not in ("all", "any"):
            issues.append("完成方式無效")
        if step.forbidden_mode not in ("all", "any"):
            issues.append("違規方式無效")
        if not 0 < step.ratio <= 1:
            issues.append("成立比例無效")
        rois = self.sop.roi_map()
        for condition in step.conditions + step.forbidden:
            if condition.type not in CONDITION_TYPES:
                issues.append("條件類型無效")
            if not condition.label:
                issues.append("未選物件")
            elif self.classes and condition.label not in self.classes:
                issues.append(f"模型無類別 {condition.label}")
            if condition.roi and condition.roi not in rois:
                issues.append(f"區域不存在：{condition.roi}")
            elif condition.roi:
                roi = rois[condition.roi]
                if roi.anchor == "workpiece" and self.sop.workpiece is None:
                    issues.append(f"區域缺少參考工件：{roi.name}")
                elif roi.anchor not in ("fixed", "workpiece"):
                    issues.append(f"區域定位基準無效：{roi.name}")
        return list(dict.fromkeys(issues))

    def _mark_modified(self):
        if not self._loading:
            self._invalidate_validation()
            self.modified.emit()

    def _set_model_label(self, path: str | None):
        """畫面顯示檔名，完整路徑放在提示中，避免 ZIP 路徑擠壓表單。"""
        if not path:
            self.model_label.setText("（未選擇）")
            self.model_label.setToolTip("")
            return
        self.model_label.setText(breakable(Path(path).name))
        self.model_label.setToolTip(str(path) + (f"\n{self._model_detail}" if self._model_detail else ""))

    def _on_sop_name(self, text: str):
        self.sop.name = text
        self._mark_modified()

    def _choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇模型", str(self._data_root),
                                              "模型包 (*.zip);;模型設定 (model.json)")
        if path:
            self.sop.model_path = self._relativize_path(path)
            self.set_model_state("selected")
            self.model_selected.emit(path)
            self._mark_modified()

    def _on_cycle_form(self, *_args):
        if self._loading:
            return
        self._clear_roi_history()
        self.sop.cycle.enabled = self.cycle_enabled_switch.isChecked()
        self.sop.cycle.auto_restart_sec = self.auto_restart_spin.value()
        self.sop.cycle.reset_hold_sec = self.reset_hold_spin.value()
        self.sop.cycle.reset_ratio = self.reset_ratio_spin.value() / 100
        self.sop.cycle.detect_sequence = self.sequence_check.isChecked()
        self.sop.cycle.reset_conditions = self.reset_table.conditions()
        self._update_live()
        self._update_cycle_summary()
        self._mark_modified()

    def _update_cycle_summary(self):
        self.cycle_settings.setEnabled(self.sop.cycle.enabled)
        uses_conditions = bool(self.sop.cycle.reset_conditions)
        self.auto_restart_spin.setEnabled(not uses_conditions)
        self.reset_hold_spin.setEnabled(uses_conditions)
        self.cycle_rule_summary.setText(
            f"有重置條件：以 {self.sop.cycle.reset_hold_sec:g} 秒觀察窗、{self.sop.cycle.reset_ratio:.0%} "
            f"影格成立比例判定；延遲秒數不生效。"
            if uses_conditions else f"無重置條件：本輪完成後延遲 {self.sop.cycle.auto_restart_sec:g} 秒開始下一輪。")
        self.reset_ratio_spin.setEnabled(uses_conditions)
        if not self.sop.cycle.enabled:
            self.cycle_rule_summary.setText("已停用自動開始下一輪；本輪完成後停止，設定會保留。")

    def _all_conditions(self):
        for step in self.sop.steps:
            yield from step.conditions
            yield from step.forbidden
        yield from self.sop.cycle.reset_conditions
