"""Camera preview and staged controls with explicit driver readback."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QGridLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
                               QScrollArea, QSlider, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from ..camera_controls import AUTO_PROPERTIES, PROPERTIES
from .video_widget import VideoWidget


class CameraDialog(QDialog):
    requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("攝影機設定")
        self.resize(960, 780)
        self._loading = True
        self._waiting = True
        self._running = False
        self._dirty = set()
        self._submitted = {}
        self._last_values = {}
        self.rows, self.sliders = {}, {}
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        preview_row = QHBoxLayout()
        self.preview = VideoWidget()
        self.preview.setFixedHeight(200)
        self.preview.clear_frame("等待攝影機畫面…")
        preview_row.addWidget(self.preview, 2)
        summary = QVBoxLayout()
        summary.addWidget(QLabel("即時預覽 · 原始影像"))
        self.metrics = QLabel("擷取速度：量測中\n程式處理：量測中")
        summary.addWidget(self.metrics)
        self.format_status = QLabel("正在讀取設備格式…")
        self.format_status.setWordWrap(True)
        summary.addWidget(self.format_status)
        help_text = QLabel("調整後按「套用變更」，在左側查看效果。\n已套用的設定不會因關閉視窗而復原。")
        help_text.setWordWrap(True)
        summary.addWidget(help_text)
        summary.addStretch()
        preview_row.addLayout(summary, 1)
        layout.addLayout(preview_row)
        tabs = QTabWidget()
        layout.addWidget(tabs, 1)
        format_page = QWidget()
        format_layout = QVBoxLayout(format_page)
        self.format_hint = QLabel("解析度決定影像細節；FPS 決定每秒擷取張數。變更時畫面可能短暫停頓。")
        self.format_hint.setWordWrap(True)
        format_layout.addWidget(self.format_hint)
        grid = QGridLayout()
        self.resolution_preset = QComboBox()
        self.resolution_preset.addItem("自訂", None)
        for width, height in ((640, 480), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)):
            self.resolution_preset.addItem(f"{width} × {height}", f"{width}x{height}")
        self.width_input, self.height_input = QSpinBox(), QSpinBox()
        for control, name in ((self.width_input, "影像寬度"), (self.height_input, "影像高度")):
            control.setRange(1, 16384)
            control.setSuffix(" px")
            control.setKeyboardTracking(False)
            control.setAccessibleName(name)
            control.valueChanged.connect(self._resolution_edited)
        self.width_input.setValue(1280)
        self.height_input.setValue(720)
        self.resolution_current = QLabel("讀取中…")
        for col, widget in enumerate((QLabel("解析度"), self.resolution_preset, QLabel("寬"), self.width_input,
                                     QLabel("高"), self.height_input)):
            grid.addWidget(widget, 0, col)
        grid.addWidget(self.resolution_current, 1, 1, 1, 5)
        self.resolution_preset.currentIndexChanged.connect(self._choose_resolution)
        format_layout.addLayout(grid)
        fps_grid = QGridLayout()
        self._headers(fps_grid)
        self._add_control("fps", fps_grid, 1)
        format_layout.addLayout(fps_grid)
        presets = QHBoxLayout()
        presets.addWidget(QLabel("常用 FPS"))
        self.fps_presets = []
        for fps in (15, 24, 30, 60):
            button = QPushButton(str(fps))
            button.clicked.connect(lambda _, n=fps: self._choose_fps(n))
            self.fps_presets.append(button)
            presets.addWidget(button)
        presets.addStretch()
        format_layout.addLayout(presets)
        explanation = QLabel("常用值不代表設備支援清單。套用後請確認「設備回報」及實際畫面。\n"
                             "攝影機擷取 FPS 與程式處理 FPS 不同；提高擷取速度不保證 AI 辨識變快。")
        explanation.setWordWrap(True)
        format_layout.addWidget(explanation)
        format_layout.addStretch()
        tabs.addTab(format_page, "解析度與 FPS")
        image_page = QWidget()
        image_layout = QVBoxLayout(image_page)
        image_hint = QLabel("直接拖曳拉桿或輸入數值即可調整；曝光、白平衡與對焦可切換自動模式。")
        image_hint.setWordWrap(True)
        image_layout.addWidget(image_hint)
        grid = QGridLayout()
        self._headers(grid)
        for row, key in enumerate((key for key in PROPERTIES if key != "fps"), 1):
            self._add_control(key, grid, row)
        image_layout.addLayout(grid)
        hint = QLabel("拉桿範圍僅供操作，實際範圍依攝影機而異；可輸入數值延伸範圍。\n"
                      "設備回報 0 或 -1 也可能表示不支援；曝光負值不一定是錯誤。")
        hint.setWordWrap(True)
        image_layout.addWidget(hint)
        image_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(image_page)
        tabs.addTab(scroll, "影像調整")
        self.state_label = QLabel("正在讀取攝影機設定…")
        layout.addWidget(self.state_label)
        self.feedback = QPlainTextEdit()
        self.feedback.setReadOnly(True)
        self.feedback.setMaximumHeight(90)
        self.feedback.setPlaceholderText("套用結果會顯示在這裡。參數只作用於目前連線，本程式不保存設定。")
        layout.addWidget(self.feedback)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("關閉")
        buttons.rejected.connect(self.reject)
        self.refresh = QPushButton("重新讀取")
        self.discard = QPushButton("捨棄未套用變更")
        self.apply = QPushButton("套用變更")
        self.apply.setDefault(True)
        for button in (self.refresh, self.discard, self.apply):
            buttons.addButton(button, QDialogButtonBox.ButtonRole.ActionRole)
        self.refresh.clicked.connect(lambda: self._request({}))
        self.discard.clicked.connect(self._discard)
        self.apply.clicked.connect(self._apply)
        layout.addWidget(buttons)
        self._loading = False
        self._update_state()

    @staticmethod
    def _headers(grid):
        for col, text in enumerate(("參數", "設備回報", "模式", "拉桿調整", "設定值")):
            grid.addWidget(QLabel(text), 0, col)
        grid.setColumnStretch(3, 1)

    def _add_control(self, key, grid, row):
        label, _ = PROPERTIES[key]
        current = QLabel("讀取中…")
        mode = QComboBox()
        mode.addItem("依設備設定", "keep")
        mode.addItem("手動", "manual")
        if key in AUTO_PROPERTIES:
            mode.addItem("自動", "auto")
        value = QDoubleSpinBox()
        value.setRange(1 if key == "fps" else -1000000, 240 if key == "fps" else 1000000)
        value.setDecimals(2)
        value.setKeyboardTracking(False)
        value.setAccessibleName(f"{label}設定值")
        if key == "fps":
            value.setSuffix(" FPS")
        slider = QSlider(Qt.Orientation.Horizontal)
        low, high = {"fps": (1, 120), "exposure": (-13, 0), "wb_temperature": (2000, 10000)}.get(key, (0, 255))
        slider.setRange(low * 100, high * 100)
        slider.setSingleStep(100)
        slider.setPageStep(1000)
        slider.setMinimumWidth(180)
        slider.setAccessibleName(f"{label}拉桿")
        value.setValue(30 if key == "fps" else max(low, min(high, value.value())))
        slider.valueChanged.connect(lambda position, v=value: v.setValue(position / 100))
        value.valueChanged.connect(lambda number, k=key: self._value_edited(k, number))
        mode.currentIndexChanged.connect(lambda _, k=key: self._mode_edited(k))
        for col, widget in enumerate((QLabel(label), current, mode, slider, value)):
            grid.addWidget(widget, row, col)
        self.rows[key] = (current, mode, value)
        self.sliders[key] = slider
        self._sync_slider(key, value.value())

    def _sync_slider(self, key, number):
        slider = self.sliders[key]
        position = round(number * 100)
        blocked = slider.blockSignals(True)
        try:
            slider.setRange(min(slider.minimum(), position), max(slider.maximum(), position))
            slider.setValue(position)
            slider.setToolTip(f"操作範圍：{slider.minimum() / 100:g} ～ {slider.maximum() / 100:g}")
        finally:
            slider.blockSignals(blocked)

    def _value_edited(self, key, number):
        self._sync_slider(key, number)
        if self._loading:
            return
        mode = self.rows[key][1]
        blocked = mode.blockSignals(True)
        mode.setCurrentIndex(mode.findData("manual"))
        mode.blockSignals(blocked)
        self._dirty.add(key)
        self._update_state()

    def _mode_edited(self, key):
        if self._loading:
            return
        if self.rows[key][1].currentData() == "keep":
            self._dirty.discard(key)
            self._restore_value(key)
        else:
            self._dirty.add(key)
        self._update_state()

    def _choose_resolution(self):
        size = self.resolution_preset.currentData()
        if size is None or self._loading:
            return
        size = tuple(map(int, size.split("x")))
        self._loading = True
        self.width_input.setValue(size[0])
        self.height_input.setValue(size[1])
        self._loading = False
        self._dirty.add("resolution")
        self._update_state()

    def _resolution_edited(self):
        if self._loading:
            return
        self._match_resolution()
        self._dirty.add("resolution")
        self._update_state()

    def _match_resolution(self):
        size = f"{self.width_input.value()}x{self.height_input.value()}"
        blocked = self.resolution_preset.blockSignals(True)
        self.resolution_preset.setCurrentIndex(max(0, self.resolution_preset.findData(size)))
        self.resolution_preset.blockSignals(blocked)

    def _choose_fps(self, fps):
        self.rows["fps"][2].setValue(fps)
        self._value_edited("fps", fps)

    def set_running(self, running):
        self._running = running
        self.format_hint.setText("作業進行中：請先停止作業，再調整解析度與 FPS。" if running else
                                 "解析度決定影像細節；FPS 決定每秒擷取張數。變更時畫面可能短暫停頓。")
        self._update_state()

    def _update_state(self):
        if self._loading:
            return
        self.refresh.setEnabled(not self._waiting)
        self.discard.setEnabled(not self._waiting and bool(self._dirty))
        blocked_format = self._running and bool(self._dirty & {"fps", "resolution"})
        self.apply.setEnabled(not self._waiting and bool(self._dirty) and not blocked_format)
        self.apply.setText(f"套用 {len(self._dirty)} 項變更" if self._dirty else "套用變更")
        self.state_label.setText("正在與攝影機通訊…" if self._waiting else
                                 f"{len(self._dirty)} 項變更尚未套用" if self._dirty else "設定已讀取，可直接調整")
        format_enabled = not self._waiting and not self._running
        for control in (self.resolution_preset, self.width_input, self.height_input, *self.fps_presets):
            control.setEnabled(format_enabled)
        for key, (_, mode, value) in self.rows.items():
            enabled = not self._waiting and (key != "fps" or not self._running)
            mode.setEnabled(enabled)
            manual = enabled and mode.currentData() != "auto"
            value.setEnabled(manual)
            self.sliders[key].setEnabled(manual)

    def _request(self, changes):
        self._submitted = dict(changes)
        self._waiting = True
        self._update_state()
        self.requested.emit(changes)

    def _apply(self):
        changes = {}
        if "resolution" in self._dirty:
            changes["resolution"] = ("manual", (self.width_input.value(), self.height_input.value()))
        for key, (_, mode, value) in self.rows.items():
            if key in self._dirty:
                changes[key] = (mode.currentData(), value.value())
        if changes:
            self._request(changes)

    def _restore_value(self, key):
        self._loading = True
        try:
            actual = self._last_values.get(key)
            if key == "resolution":
                if actual:
                    self.width_input.setValue(actual[0])
                    self.height_input.setValue(actual[1])
                    self._match_resolution()
            else:
                _, mode, value = self.rows[key]
                mode.setCurrentIndex(0)
                if actual is not None:
                    value.setValue(actual)
        finally:
            self._loading = False

    def _discard(self):
        for key in self._dirty:
            self._restore_value(key)
        self._dirty.clear()
        self._update_state()

    def update_controls(self, values, messages):
        self._last_values.update(values)
        for key in self._submitted:
            label = "解析度" if key == "resolution" else PROPERTIES[key][0]
            if any(message.startswith(label + "：") and "未完成" not in message for message in messages):
                self._dirty.discard(key)
        self._submitted = {}
        for key, (current, _, _) in self.rows.items():
            actual = values.get(key)
            current.setText(f"{actual:g}" if actual is not None else "無法讀取")
            if key not in self._dirty:
                self._restore_value(key)
        size = values.get("resolution")
        size_text = f"{size[0]} × {size[1]}" if size else "無法讀取"
        fps = values.get("fps")
        fps_text = f"{fps:g} FPS" if fps is not None else "無法讀取 FPS"
        self.resolution_current.setText(f"設備回報：{size_text}")
        self.format_status.setText(f"設備設定回報\n{size_text} · {fps_text}")
        if "resolution" not in self._dirty:
            self._restore_value("resolution")
        if messages:
            self.feedback.setPlainText("\n".join(messages))
        self._waiting = False
        self._update_state()

    def on_packet(self, packet):
        self.preview.set_frame(packet.frame)
        capture = f"{packet.camera_fps:.1f} FPS" if packet.camera_fps is not None else "量測中"
        height, width = packet.frame.shape[:2]
        self.metrics.setText(f"目前影像：{width} × {height}\n實際擷取：{capture}\n程式處理：{packet.fps:.1f} FPS")
