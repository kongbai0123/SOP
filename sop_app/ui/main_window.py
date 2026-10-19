"""主視窗：影像來源、SOP 檔案管理，並把背景執行緒的結果分派給各頁面。"""
from __future__ import annotations

import copy
from datetime import date
import json
import socket
import time
import logging
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QGridLayout, QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QToolBar, QWidget)

from ..model_bundle import ModelBundleError, ModelInfo, read_model_info
from ..model_eta import ModelETA, ModelLoadProgress, model_key, stage_key
from ..native_runtime import cv2, NativeDependencyError
from ..paths import APP_SETTINGS, PROJECT_ROOT, RECORDS_DIR, SOPS_DIR, resolve
from ..pipeline import FramePacket, VideoPipeline
from ..recorder import query
from ..sop_schema import SOPDefinition, Step, load_sop, save_sop
from ..startup import initial_sop_path
from .editor_page import EditorPage
from .records_page import RecordsPage
from .compact_layout import StatusText
from .run_page import RunPage
from .workbench_widgets import PageStack, application_styles, install_readability_guard

VIDEO_FILTER = "影片 (*.mp4 *.avi *.mov *.mkv *.wmv);;所有檔案 (*)"


class MainWindow(QMainWindow):
    save_finished = Signal(str)

    def __init__(self, defer_initial_sop=False, inference_worker=None, station_store=None):
        super().__init__()
        self.setObjectName("sopMainWindow")
        if QApplication.instance().style().objectName().lower() in ("windows", "windowsvista"):
            QApplication.instance().setStyle("Fusion")
        font = QFont(self.font())
        font.setPointSizeF(10)
        self.setFont(font)
        self._chrome_dark = None
        self.station_store = station_store
        self._station_log_handler = None
        self.current_station = station_store.active if station_store else None
        self._set_station_paths(self.current_station)
        self._bind_station_logging()
        self._pending_station = None
        self.settings = self._load_settings()
        self.sop_path: Path | None = None
        self.dirty = False
        self._save_pending = False
        self.model_info: ModelInfo | None = None
        self.model_ready = False
        self._requested_model: str | None = None
        self._engine_running = False
        self._model_started = None
        self._model_stage = ""
        self._model_eta = None
        self._model_work = ModelLoadProgress()
        self._source_label = ""
        self._error_box: QMessageBox | None = None
        self._error_counts: dict[str, int] = {}
        self._vision_error = ""
        try:
            cv2.require()
        except NativeDependencyError as exc:
            self._vision_error = str(exc)
            logging.getLogger("sop.launcher").warning("%s", exc)
        self.resize(1500, 900)

        self.pipeline = VideoPipeline(self._records_dir, self, inference_worker=inference_worker)
        self.pipeline.packet_ready.connect(self._on_packet)
        self.pipeline.events_ready.connect(self._on_events)
        self.pipeline.model_loaded.connect(self._on_model_loaded)
        self.pipeline.source_changed.connect(self._on_source_changed)
        self.pipeline.engine_state.connect(self._on_engine_state)
        self.pipeline.status.connect(lambda text: self.statusBar().showMessage(text, 6000))
        self.pipeline.error.connect(self._notify_error)
        self.pipeline.records_changed.connect(self._on_records_changed)

        self.run_page = RunPage()
        self.run_page.start_requested.connect(self._start_run)
        self.run_page.stop_requested.connect(self.pipeline.stop_engine)
        self.run_page.confirm_requested.connect(self.pipeline.confirm_step)
        self.run_page.skip_requested.connect(self.pipeline.skip_step)
        self.run_page.restart_requested.connect(self.pipeline.restart_cycle)
        self.run_page.alarm_ack_requested.connect(self.pipeline.acknowledge_alarms)
        self.run_page.context_changed.connect(self._on_context_changed)
        self.run_page.set_context(self.settings.get("run_context", {}))
        self.editor = EditorPage()
        self._set_editor_station_paths()
        self.editor.workpiece_changed.connect(self.pipeline.set_workpiece)
        self.editor.modified.connect(self._on_modified)
        self.editor.model_selected.connect(self._load_model)
        self.save_finished.connect(self.editor.on_save_result)
        self.records_page = RecordsPage(self._records_dir)
        self._maintenance_busy = False
        self.records_page.backup_requested.connect(self._backup_station)
        self.records_page.restore_requested.connect(self._restore_station)
        self.pipeline.maintenance_finished.connect(self._on_maintenance_finished)
        self.pipeline.recording_status.connect(self.records_page.set_recording_status)
        self.pipeline.station_switched.connect(self._on_station_switched)

        self.tabs = PageStack()
        self.tabs.setObjectName("mainPages")
        self.tabs.addTab(self.run_page, "執行監控")
        self.tabs.addTab(self.editor, "工序編輯")
        self.tabs.addTab(self.records_page, "作業紀錄")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        self._build_header()
        self._build_toolbar()
        self.pipeline.camera_available.connect(self.camera_button.setEnabled)
        # One compact line: progress, processing rate, then model and compute device.
        self.statusBar().setSizeGripEnabled(False)
        self.model_status = StatusText("模型：未載入")
        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 100)
        self.model_progress.setFixedWidth(170)
        self.model_progress.setValue(0)
        self.model_progress.hide()
        self.statusBar().addPermanentWidget(self.model_progress)
        self.model_timer = QTimer(self)
        self.model_timer.setInterval(500)
        self.model_timer.timeout.connect(self._update_model_progress)
        self.pipeline.model_progress.connect(self._on_model_progress)
        self.pipeline.model_timing.connect(self._on_model_timing)
        self.fps_status = QLabel("")
        self.statusBar().addPermanentWidget(self.fps_status)
        self.statusBar().addPermanentWidget(self.model_status)
        if self._vision_error:
            self.runtime_warning = QLabel("影像功能不可用：OpenCV 載入失敗。仍可編輯 SOP 與查看紀錄。")
            self.runtime_warning.setStyleSheet("color:#b71c1c; font-weight:bold")
            self.runtime_warning.setToolTip(self._vision_error)
            self.statusBar().addPermanentWidget(self.runtime_warning)

        QApplication.instance().paletteChanged.connect(self._apply_chrome_theme)
        self._apply_chrome_theme()
        install_readability_guard(self)
        self._push_context(self.run_page.context())   # 啟動時只同步給背景執行緒，不寫入設定檔
        self._refresh_today_stats()
        self.pipeline.start()
        if not defer_initial_sop:
            self._open_initial_sop()

    # ---- 版面 ------------------------------------------------------------------
    def _apply_chrome_theme(self, _palette=None):
        dark = QApplication.instance().palette().window().color().lightness() < 128
        if self._chrome_dark != dark:
            self._chrome_dark = dark
            self.setStyleSheet(application_styles(dark))

    def _build_header(self):
        """Page switcher in the centre and file actions on the right of the single top header.

        The SOP name and version are edited in the editor's project section; the
        window title shows the file and an asterisk for unsaved changes.
        """
        header = QWidget()
        header.setObjectName("appHeader")
        header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.nav_buttons = QButtonGroup(self)
        self.nav_buttons.setExclusive(True)
        navigation = QHBoxLayout()
        navigation.setSpacing(0)
        for index in range(self.tabs.count()):
            button = QPushButton(self.tabs.tabText(index))
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setProperty("segment", "first" if index == 0 else
                               "last" if index == self.tabs.count() - 1 else "middle")
            self.nav_buttons.addButton(button, index)
            navigation.addWidget(button)
        self.nav_buttons.idClicked.connect(self.tabs.setCurrentIndex)
        self.nav_buttons.button(self.tabs.currentIndex()).setChecked(True)
        files = QHBoxLayout()
        files.setSpacing(4)
        files.addStretch(1)
        self.new_button, self.open_button = QPushButton("新增"), QPushButton("開啟…")
        self.save_button, self.save_as_button = QPushButton("儲存"), QPushButton("另存新檔…")
        self.save_button.setObjectName("primaryButton")
        for button, handler in ((self.new_button, self._new_sop), (self.open_button, self._open_sop),
                                (self.save_button, self._request_save), (self.save_as_button, self._request_save_as)):
            button.clicked.connect(lambda _checked=False, h=handler: h())
            files.addWidget(button)
        layout = QGridLayout(header)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setHorizontalSpacing(12)
        layout.addLayout(navigation, 0, 1)
        layout.addLayout(files, 0, 2)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(2, 1)
        self.station_combo = QComboBox()
        self.station_combo.setAccessibleName("目前工站")
        self.station_combo.setMinimumWidth(180)
        self.station_combo.setMaximumWidth(260)
        self.station_combo.setMinimumContentsLength(12)
        self.station_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.new_station_button = QPushButton("新增工站")
        self.new_station_button.clicked.connect(self._new_station)
        self.station_combo.activated.connect(self._select_station)
        self.station_controls = QWidget()
        station_line = QHBoxLayout(self.station_controls)
        station_line.setContentsMargins(0, 0, 0, 0)
        station_line.addWidget(QLabel("工站"))
        station_line.addWidget(self.station_combo)
        station_line.addWidget(self.new_station_button)
        station_line.addStretch(1)
        layout.addWidget(self.station_controls, 0, 0)
        self.station_controls.setVisible(self.station_store is not None)
        self._refresh_station_choices()
        self.setMenuWidget(header)
        # Keyboard shortcuts stay available without a separate menu bar.
        for shortcut, handler in ((QKeySequence.StandardKey.New, self._new_sop),
                                  (QKeySequence.StandardKey.Open, self._open_sop),
                                  (QKeySequence.StandardKey.Save, self._request_save),
                                  (QKeySequence.StandardKey.SaveAs, self._request_save_as)):
            action = QAction(self)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda _checked=False, h=handler: h())
            self.addAction(action)

    def _build_toolbar(self):
        toolbar = QToolBar("影像來源")
        toolbar.setObjectName("sourceToolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addWidget(QLabel("影像來源"))
        self.source_combo = QComboBox()
        for index in range(4):
            self.source_combo.addItem(f"攝影機 {index}", f"camera:{index}")
        self.source_combo.addItem("影片檔…", "file")
        self.source_combo.setCurrentIndex(max(0, self.source_combo.findData(self.settings.get("source", "camera:0"))))
        toolbar.addWidget(self.source_combo)
        connect_button = QPushButton("連線")
        connect_button.clicked.connect(self._connect_source)
        toolbar.addWidget(connect_button)
        disconnect_button = QPushButton("中斷")
        disconnect_button.clicked.connect(self.pipeline.close_source)
        toolbar.addWidget(disconnect_button)
        self.camera_button = QPushButton("攝影機參數…")
        self.camera_button.setEnabled(False)
        self.camera_button.clicked.connect(self._camera_settings)
        toolbar.addWidget(self.camera_button)
        self.source_status = QLabel("未連線")
        toolbar.addWidget(self.source_status)
        toolbar.addSeparator()
        self.mask_check = QCheckBox("顯示遮罩")
        self.mask_check.setChecked(True)
        self.mask_check.toggled.connect(self._apply_display)
        toolbar.addWidget(self.mask_check)
        toolbar.addWidget(QLabel("最低顯示信心度"))
        self.display_score = QDoubleSpinBox()
        self.display_score.setRange(0.3, 1.0)
        self.display_score.setSingleStep(0.05)
        self.display_score.setValue(0.5)
        self.display_score.valueChanged.connect(self._apply_display)
        toolbar.addWidget(self.display_score)

    # ---- 設定檔 ----------------------------------------------------------------
    def _set_station_paths(self, station):
        self._data_root = station.root if station else PROJECT_ROOT
        self._sops_dir = station.sops if station else SOPS_DIR
        self._records_dir = station.records if station else RECORDS_DIR
        self._settings_path = station.settings if station else APP_SETTINGS
        self._resolve_path = station.resolve if station else resolve

    def _set_editor_station_paths(self):
        self.editor._data_root = self._data_root
        self.editor._resolve_path = self._resolve_path
        if self.current_station:
            self.editor._relativize_path = self.current_station.relativize

    def _bind_station_logging(self):
        logger = logging.getLogger("sop")
        if self._station_log_handler:
            logger.removeHandler(self._station_log_handler)
            self._station_log_handler.close()
            self._station_log_handler = None
        if self.current_station:
            from logging.handlers import RotatingFileHandler
            directory = self._data_root / "logs"
            try:
                directory.mkdir(parents=True, exist_ok=True)
                handler = RotatingFileHandler(directory / "workstation.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            except OSError:
                logger.exception("無法建立工站日誌")
                return
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
            self._station_log_handler = handler

    def _atomic_settings(self):
        from ..station_backup import atomic_json
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self._settings_path, self.settings)

    def _refresh_station_choices(self):
        self.station_combo.blockSignals(True)
        self.station_combo.clear()
        if self.station_store:
            for station in self.station_store.list():
                self.station_combo.addItem(f"{station.id} · {station.name}", station.id)
            self.station_combo.setCurrentIndex(self.station_combo.findData(self.current_station.id))
            self.station_combo.setToolTip(str(self.current_station.root))
        self.station_combo.blockSignals(False)

    def _new_station(self):
        if not self.station_store or self._engine_running or self._maintenance_busy or self._save_pending:
            return
        name, accepted = QInputDialog.getText(self, "新增工站", "工站名稱：")
        if not accepted:
            return
        try:
            station = self.station_store.create(name)
            self._refresh_station_choices()
            self._switch_to_station(station)
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "無法新增工站", str(exc))

    def _select_station(self, index):
        if not self.station_store:
            return
        self._switch_to_station(self.station_store.get(self.station_combo.itemData(index)))

    def _switch_to_station(self, station):
        if station is None or station.id == self.current_station.id:
            return
        if self._engine_running or self._maintenance_busy or self._save_pending:
            self._refresh_station_choices()
            QMessageBox.information(self, "無法切換工站", "請先停止作業或等待儲存／資料維護完成。")
            return
        if not self._confirm_discard():
            self._refresh_station_choices()
            return
        if not self.pipeline.isRunning():
            self._refresh_station_choices()
            QMessageBox.warning(self, "無法切換工站", "紀錄系統未就緒，請重新啟動程式。")
            return
        self.editor._stop_validation()
        self._on_context_changed(self.run_page.context())
        self._pending_station = station
        self._maintenance_busy = True
        self.setEnabled(False)
        self.statusBar().showMessage("正在保存目前工站紀錄並切換工站…")
        self.pipeline.switch_station(station.records)

    def _on_station_switched(self, directory, error):
        station = self._pending_station
        if station is None:
            return
        self._pending_station = None
        self._maintenance_busy = False
        self.setEnabled(True)
        if error:
            self._refresh_station_choices()
            QMessageBox.warning(self, "切換工站未完成", error)
            return
        self.current_station = station
        self._set_station_paths(station)
        self._bind_station_logging()
        try:
            self.station_store.activate(station.id)
        except (OSError, ValueError, RuntimeError) as exc:
            self._notify_error(f"已切換工站，但未能記住選擇；下次啟動請重新選擇：{exc}")
        self._set_editor_station_paths()
        self.settings = self._load_settings()
        self.settings["station"] = station.name
        self.run_page.reset_station()
        self.run_page.blockSignals(True)
        self.run_page.set_context(self.settings.get("run_context", {}))
        self.run_page.blockSignals(False)
        self.source_combo.setCurrentIndex(max(0, self.source_combo.findData(self.settings.get("source", "camera:0"))))
        self.records_page.set_records_directory(station.records)
        self._refresh_station_choices()
        self._requested_model = None
        self._open_initial_sop()
        self._push_context(self.run_page.context())
        self._refresh_today_stats()
        self.statusBar().showMessage(f"已切換至 {station.id} · {station.name}；請連線此工站的影像來源。", 10000)

    def _load_settings(self) -> dict:
        try:
            data = json.loads(self._settings_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_settings(self):
        try:
            self._atomic_settings()
        except OSError:
            pass

    @property
    def station(self) -> str:
        """工站名稱寫入每一輪紀錄；未設定時使用電腦名稱。"""
        return self.current_station.name if self.current_station else str(self.settings.get("station") or socket.gethostname())

    # ---- 通知 ------------------------------------------------------------------
    def _notify_error(self, text: str):
        """背景錯誤不使用阻擋式對話框：同一則訊息只開一個視窗並累計次數，產線操作不被打斷。"""
        logging.getLogger("sop.ui").warning("%s", text)
        self.statusBar().showMessage(f"⚠ {text.splitlines()[0]}", 15000)
        count = self._error_counts[text] = self._error_counts.get(text, 0) + 1
        box = self._error_box
        if box is None:
            box = self._error_box = QMessageBox(QMessageBox.Icon.Warning, "錯誤", "", QMessageBox.StandardButton.Ok, self)
            box.setWindowModality(Qt.WindowModality.NonModal)
            box.finished.connect(lambda _result: self._error_counts.clear())
        box.setText(text)
        box.setInformativeText(f"此訊息已出現 {count} 次" if count > 1 else "")
        box.show()

    def _on_context_changed(self, context: dict):
        remembered = {key: context.get(key, "") for key in ("operator", "work_order")}
        if self.settings.get("run_context", {"operator": "", "work_order": ""}) != remembered:
            self.settings["run_context"] = remembered
            self._save_settings()
        self._push_context(context)

    def _push_context(self, context: dict):
        self.pipeline.set_run_context({**context, "station": self.station})

    def _on_records_changed(self):
        self.records_page.mark_stale()
        self._refresh_today_stats()

    def _refresh_today_stats(self):
        try:
            rows = query(self._records_dir / "records.db", "SELECT result, COUNT(*) FROM cycles "
                         "WHERE started_at >= ? AND result IN ('OK','NG') GROUP BY result",
                         (f"{date.today().isoformat()} 00:00:00",))
        except Exception as exc:      # 紀錄庫暫時無法讀取時不影響監控
            logging.getLogger("sop.ui").warning("讀取今日統計失敗：%s", exc)
            return
        self.run_page.set_today_stats(dict(rows))

    # ---- 影像來源 --------------------------------------------------------------
    def _connect_source(self):
        if self._vision_error:
            QMessageBox.warning(self, "影像功能不可用", self._vision_error)
            return
        source = self.source_combo.currentData()
        if source.startswith("camera:"):
            self.pipeline.open_camera(int(source.split(":")[1]))
        else:
            start_dir = self.settings.get("video_path") or str(self._data_root)
            path, _ = QFileDialog.getOpenFileName(self, "選擇影片", start_dir, VIDEO_FILTER)
            if not path:
                return
            self.pipeline.open_file(path, loop=True)
            self.settings["video_path"] = path
        self.settings["source"] = source
        self._save_settings()

    def _apply_display(self, *_args):
        self.pipeline.set_display(self.mask_check.isChecked(), self.display_score.value())

    def _camera_settings(self):
        if self._vision_error:
            QMessageBox.warning(self, "影像功能不可用", self._vision_error)
            return
        from .camera_dialog import CameraDialog
        dialog = CameraDialog(self)
        dialog.set_running(self._engine_running)
        dialog.requested.connect(self.pipeline.camera_controls)
        self.pipeline.camera_controls_ready.connect(dialog.update_controls)
        self.pipeline.packet_ready.connect(dialog.on_packet)
        self.pipeline.engine_state.connect(dialog.set_running)
        def source_changed(_label):
            dialog.reject()
        self.pipeline.source_changed.connect(source_changed)
        try:
            self.pipeline.camera_controls()
            dialog.exec()
        finally:
            self.pipeline.camera_controls_ready.disconnect(dialog.update_controls)
            self.pipeline.packet_ready.disconnect(dialog.on_packet)
            self.pipeline.engine_state.disconnect(dialog.set_running)
            self.pipeline.source_changed.disconnect(source_changed)
            dialog.deleteLater()

    def _on_source_changed(self, label: str):
        self._source_label = label
        self.run_page.clear_video("影像來源已變更，等待影像" if label else "影像來源已中斷")
        self.editor.clear_video("影像來源已變更，等待影像")
        self.source_status.setText(label or "未連線")
        if not label:
            self.fps_status.setText("")
            self.editor.clear_video("影像來源已中斷")

    # ---- 背景執行緒結果 ----------------------------------------------------------
    def _on_packet(self, packet: FramePacket):
        try:
            if not self.model_ready:
                # Do not preview stale detections while clear/load commands are queued.
                # Trials read every pending result, so clear those the same way as the preview frame.
                cleared = replace(packet.result, detections=[], detection_valid=False)
                packet = replace(packet, result=cleared, display=packet.frame, classes=(),
                                 results=tuple(replace(result, detections=[], detection_valid=False)
                                               for result in packet.results))
            page = self.tabs.currentWidget()
            if page is self.run_page:
                self.run_page.on_packet(packet)
            elif page is self.editor:
                self.editor.on_packet(packet)
            count = len([d for d in packet.result.detections if d.score >= self.display_score.value()])
            capture = f"擷取 {packet.camera_fps:.1f} FPS · " if packet.camera_fps is not None else ""
            self.fps_status.setText(f"{capture}處理 {packet.fps:.1f} FPS · 推論 {packet.inference_ms:.0f} ms · 偵測 {count} 個  ")
        finally:
            self.pipeline.acknowledge_packet()

    def _on_events(self, events):
        self.run_page.on_events(events)

    def _on_engine_state(self, running: bool):
        self._engine_running = running
        self.editor.set_production_running(running)
        self.station_combo.setEnabled(not running and self.station_store is not None)
        self.new_station_button.setEnabled(not running and self.station_store is not None)
        self.run_page.set_running(running)
        if not running:
            self.run_page.set_sop(self.editor.sop)

    def _on_tab_changed(self, index: int):
        button = self.nav_buttons.button(index) if hasattr(self, "nav_buttons") else None
        if button is not None and not button.isChecked():
            button.setChecked(True)
        if self.tabs.currentWidget() is self.run_page:
            self.run_page.set_sop(self.editor.sop)     # 未執行時同步編輯後的內容

    # ---- 模型 ------------------------------------------------------------------
    def _load_model(self, path: str):
        # Invalidate previous readiness before reading a newly selected bundle.
        self.model_ready = False
        self.model_info = None
        self._requested_model = None
        self.pipeline.clear_model()
        self.model_timer.stop()
        self.model_progress.hide()
        self._model_started = None
        self.editor.set_model_state("selected")
        try:
            info = read_model_info(path)
        except (ModelBundleError, OSError, ValueError, TypeError) as exc:
            self.model_status.setText("模型：失敗")
            self.editor.set_model_state("failed", str(exc))
            QMessageBox.warning(self, "模型", str(exc))
            self.editor.set_classes([])
            return
        self.model_info = info
        self.model_ready = False
        self.editor.set_classes(info.classes)
        self.editor.sop.model_version_id = info.model_version_id
        if self._vision_error:
            self.model_status.setText("模型：影像環境不可用，未載入")
            self.editor.set_model_state("failed", self._vision_error)
            return
        self._requested_model = str(Path(path).resolve())
        self._model_started = time.monotonic()
        self._model_eta = ModelETA(self._data_root / "logs" / "model-load-times.json", model_key(path))
        self._model_work = ModelLoadProgress()
        self._model_stage = "正在準備模型…"
        self.model_progress.show()
        self.model_timer.start()
        self.editor.set_model_state("loading", self._model_stage)
        self._update_model_progress()
        self.pipeline.load_model(path)

    def _is_selected_model(self, path: str | Path) -> bool:
        if not self.editor.sop.model_path:
            return False
        def normalized(value):
            result = self._resolve_path(value).resolve()
            return result.parent if result.name.lower() == "model.json" else result
        return normalized(path) == normalized(self.editor.sop.model_path)

    def _on_model_progress(self, stage, path):
        if not self._is_selected_model(path):
            return
        self._model_stage = stage_key(stage)
        if self._model_stage == "載入失敗":
            self._model_work.failed = True
        if self._requested_model:
            self.editor.set_model_state("failed" if stage == "載入失敗" else "loading", stage)
        self._update_model_progress()

    def _on_model_timing(self, event):
        if self._model_eta is not None:
            self._model_eta.observe(event)
            self._model_work.observe(event)

    def _update_model_progress(self):
        if self._model_started is not None:
            work = self._model_work
            self.model_progress.setValue(work.percent)
            self.model_progress.setFormat("%p%")
            self.model_progress.setToolTip("完成度依已完成工作計算；模型就緒後才達到 100%。")
            if self._model_stage in ("載入完成", "載入失敗"):
                self.model_status.setText(f"模型：{self._model_stage}")
                return
            estimate = self._model_eta.remaining_text() if self._model_eta else "剩餘時間估算中"
            self.model_status.setText(f"模型載入 · {estimate}")

    def _on_model_loaded(self, info: ModelInfo | None, device: str, path: str):
        if not self._is_selected_model(path):
            return  # Stale successes/failures must not overwrite the latest selection's state.
        self.model_timer.stop()
        if info is not None:
            self._model_work.completed = self._model_work.total
            self._model_stage = "載入完成"
        self._update_model_progress()
        self.model_progress.hide()
        self._model_started = None
        self._model_eta = None
        self.model_ready = info is not None
        if info is None:
            self._requested_model = None
            self.model_info = None
            self.model_status.setText("模型：失敗，未載入")
            self.editor.set_model_state("failed", "載入或推論失敗，請查看錯誤訊息並重新選擇模型。")
        else:
            self.model_info = info
            self._requested_model = str(self._resolve_path(self.editor.sop.model_path).resolve())
            self.editor.set_classes(info.classes)
            self.editor.set_model_state("available", f"{info.model_version_id} · {device}")
            self.model_status.setText(f"模型：{info.project_name} {info.model_version_id} · "
                                      f"{len(info.classes)} 類 · {device}")

    # ---- SOP 檔案 --------------------------------------------------------------
    def _open_initial_sop(self):
        path = initial_sop_path(self.settings, root=self._data_root) if self.current_station else initial_sop_path(self.settings)
        if path is not None:
            return self._open_path(path)
        self._apply_sop(SOPDefinition(steps=[Step("工序 1")]), None)

    def _open_path(self, path: Path):
        try:
            sop = load_sop(path)
            if self.current_station and not path.resolve().is_relative_to(self._sops_dir.resolve()):
                # Import a copy; never edit another station's original SOP in place.
                import uuid
                if sop.model_path and not Path(sop.model_path).is_absolute():
                    base = path.parent.parent if path.parent.name == "sops" else path.parent
                    sop.model_path = str((base / sop.model_path).resolve())
                target = self._sops_dir / path.name
                if target.exists():
                    target = target.with_name(f"{target.stem}-{uuid.uuid4().hex[:8]}{target.suffix}")
                save_sop(sop, target)
                path = target
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "開啟失敗", f"無法讀取 {path.name}：{exc}")
            return
        self._apply_sop(sop, path)

    def _apply_sop(self, sop: SOPDefinition, path: Path | None):
        self.sop_path = path
        self.editor.set_sop(sop)
        self.run_page.set_sop(sop)
        self.dirty = False
        self._update_title()
        if path is not None:
            self.settings["last_sop"] = self.current_station.relativize(path) if self.current_station else str(path)
            self._save_settings()
        if sop.model_path:
            model_path = self._resolve_path(sop.model_path)
            if str(model_path.resolve()) != self._requested_model:
                self._load_model(str(model_path))
            elif self.model_info:
                self.editor.set_classes(self.model_info.classes)
                self.editor.set_model_state("available" if self.model_ready else "loading")
        else:
            self.pipeline.clear_model()
            self.model_ready = False
            self.model_info = None
            self._requested_model = None
            self.model_timer.stop()
            self.model_progress.hide()
            self._model_started = None
            self.model_status.setText("模型：未選擇")
            self.editor.set_classes([])
            self.editor.set_model_state("unselected")

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        answer = QMessageBox.question(
            self, "尚未儲存", "SOP 有尚未儲存的變更，要先儲存嗎？",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Save:
            return self._request_save()
        return answer == QMessageBox.StandardButton.Discard

    def _new_sop(self):
        if not self._confirm_discard():
            return
        current = self.editor.sop
        sop = SOPDefinition(name="新 SOP", model_path=current.model_path, model_version_id=current.model_version_id,
                            rois=copy.deepcopy(current.rois), steps=[Step("工序 1")],
                            workpiece=copy.deepcopy(current.workpiece))
        self._apply_sop(sop, None)

    def _open_sop(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "開啟 SOP", str(self._sops_dir), "SOP 設定檔 (*.json)")
        if path:
            self._open_path(Path(path))

    def _request_save(self, save_as=False) -> bool:
        """A modal file dialog runs a nested event loop; ignore repeated clicks meanwhile."""
        if self._save_pending:
            return False
        self._save_pending = True
        for button in (self.save_button, self.save_as_button):
            button.setEnabled(False)
        try:
            return self._save_as() if save_as else self._save()
        finally:
            self._save_pending = False
            for button in (self.save_button, self.save_as_button):
                button.setEnabled(True)

    def _request_save_as(self) -> bool:
        return self._request_save(save_as=True)

    def _save(self) -> bool:
        if self.sop_path is None:
            return self._save_as()
        return self._write(self.sop_path)

    def _save_as(self) -> bool:
        default = self._sops_dir / f"{self.editor.sop.name or 'sop'}.json"
        path, _ = QFileDialog.getSaveFileName(self, "另存 SOP", str(default), "SOP 設定檔 (*.json)")
        if not path:
            self.save_finished.emit("cancelled")
            return False
        return self._write(Path(path), new_file=True)

    def _write(self, path: Path, new_file: bool = False) -> bool:
        if self.current_station and not path.resolve().is_relative_to(self._sops_dir.resolve()):
            QMessageBox.warning(self, "請儲存於目前工站", f"SOP 與工序請儲存於：\n{self._sops_dir}")
            self.save_finished.emit("failed")
            return False
        sop = self.editor.sop
        candidate = copy.deepcopy(sop)
        if self.dirty and not new_file and path.exists():
            candidate.version += 1   # Commit the in-memory version only after atomic save succeeds.
        try:
            save_sop(candidate, path)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "儲存失敗", str(exc))
            self.save_finished.emit("failed")
            return False
        sop.version = candidate.version
        self.sop_path = path
        self.dirty = False
        self.editor.version_label.setText(str(sop.version))
        self.settings["last_sop"] = self.current_station.relativize(path) if self.current_station else str(path)
        self._save_settings()
        self._update_title()
        self.statusBar().showMessage(f"已儲存 {path.name}（版本 {sop.version}）", 5000)
        self.save_finished.emit("saved")
        return True

    def _on_modified(self):
        self.dirty = True
        self._update_title()

    def _update_title(self):
        name = self.sop_path.name if self.sop_path else "未儲存"
        sop = self.editor.sop
        station_prefix = f"[{self.current_station.id} {self.current_station.name}] " if self.current_station else ""
        self.setWindowTitle(f"SOP 工序監控 — {station_prefix}{sop.name}（{name}）{' *' if self.dirty else ''}")

    # ---- 開始作業 --------------------------------------------------------------
    def _start_run(self):
        if self._maintenance_busy or not self.pipeline.isRunning():
            QMessageBox.warning(self, "無法開始作業", "資料維護中或紀錄系統未就緒，請稍後或重新啟動程式。")
            return
        if self._vision_error:
            QMessageBox.warning(self, "無法開始監控", self._vision_error)
            return
        if self.dirty:
            answer = QMessageBox.question(
                self, "尚未儲存", "SOP 有尚未儲存的變更。要先儲存再開始嗎？\n（選「否」會直接用目前內容開始）",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Cancel or (answer == QMessageBox.StandardButton.Yes
                                                               and not self._request_save()):
                return
        sop = self.editor.sop
        errors, warnings = sop.validate(self.model_info.classes if self.model_info else None)
        if errors:
            QMessageBox.critical(self, "SOP 設定有誤", "無法開始作業：\n\n" + "\n".join(f"• {e}" for e in errors))
            return
        if not self.model_ready:
            warnings.insert(0, "模型尚未載入完成，在載入前不會有任何偵測結果")
        if not self._source_label:
            warnings.insert(0, "尚未連線影像來源")
        if not self.run_page.context()["operator"]:
            warnings.append("尚未填寫作業員，紀錄將無法追溯人員")
        if warnings and QMessageBox.question(
                self, "確認開始", "\n".join(f"• {w}" for w in warnings) + "\n\n仍要開始作業嗎？"
        ) != QMessageBox.StandardButton.Yes:
            return
        self.run_page.set_sop(sop)
        self.pipeline.start_engine(sop)
        self.tabs.setCurrentWidget(self.run_page)

    def _backup_station(self):
        self._station_maintenance("backup")

    def _restore_station(self):
        self._station_maintenance("restore")

    def _station_maintenance(self, action):
        if self._maintenance_busy or self._engine_running:
            QMessageBox.information(self, "請先停止作業", "停止作業後才能備份或還原工站資料。")
            return
        if not self.pipeline.isRunning():
            QMessageBox.warning(self, "紀錄系統未就緒", "請重新啟動程式後再進行資料維護。")
            return
        if action == "restore":
            if not self._confirm_discard():
                return
            path, _ = QFileDialog.getOpenFileName(self, "選擇工站備份", str(self._data_root / "backups"), "工站備份 (*.zip)")
            if not path:
                return
            if QMessageBox.question(self, "還原工站資料", "將以選定備份取代目前的 SOP、作業紀錄、截圖及應用程式設定。\n"
                    "程式會先驗證內容並建立目前資料的安全備份；外部模型與外部 SOP 不會變更。\n是否繼續？") != QMessageBox.StandardButton.Yes:
                return
        else:
            if self.dirty:
                QMessageBox.information(self, "請先儲存 SOP", "目前有未儲存的編輯，請先儲存後再備份。")
                return
            from datetime import datetime
            path, _ = QFileDialog.getSaveFileName(self, "備份工站資料（不含外部 SOP 與模型）",
                str(self._data_root / "backups" / f"station-{datetime.now():%Y%m%d-%H%M%S}.zip"), "工站備份 (*.zip)")
            if not path:
                return
        self._maintenance_busy = True
        self.setEnabled(False)
        self.statusBar().showMessage("正在備份工站資料…" if action == "backup" else "正在驗證並還原工站資料…")
        self.pipeline.maintain_records(action, path)

    def _on_maintenance_finished(self, action, result, error):
        self._maintenance_busy = False
        self.setEnabled(True)
        if error:
            QMessageBox.warning(self, "資料維護未完成", error)
        else:
            if action == "restore":
                self.settings = self._load_settings()
                if self.current_station:
                    self.settings["station"] = self.current_station.name
                previous = Path(self.settings.get("last_sop") or "example_sop.json").name
                candidates = sorted(self._sops_dir.glob("*.json"))
                restored = self._sops_dir / previous
                self.settings["last_sop"] = str(restored if restored.exists() else candidates[0]) if candidates else ""
                self.run_page.set_context(self.settings.get("run_context", {}))
                self._open_initial_sop()
            QMessageBox.information(self, "還原完成" if action == "restore" else "備份完成",
                f"還原完成；還原前的資料保存於：\n{result}" if action == "restore" else
                f"已建立並校驗工站備份：\n{result}\n\n包含資料目錄內的 SOP、紀錄、截圖及設定；不含外部 SOP、模型與日誌。")
        self.records_page.mark_stale()
        self._refresh_today_stats()
        self.statusBar().showMessage("資料維護未完成" if error else "資料維護完成", 6000)

    def closeEvent(self, event):
        if self._maintenance_busy:
            event.ignore()
            return
        if not self._confirm_discard():
            event.ignore()
            return
        self.editor._stop_validation()
        if not self.pipeline.shutdown():
            self.statusBar().showMessage("影像來源正在關閉，請稍候再關閉視窗", 6000)
            event.ignore()
            return
        if self._error_box is not None:
            self._error_box.close()
        if self._station_log_handler:
            logging.getLogger("sop").removeHandler(self._station_log_handler)
            self._station_log_handler.close()
            self._station_log_handler = None
        event.accept()
