"""主視窗：影像來源、SOP 檔案管理，並把背景執行緒的結果分派給各頁面。"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QLabel, QMainWindow, QMessageBox,
                               QPushButton, QTabWidget, QToolBar)

from ..model_bundle import ModelBundleError, ModelInfo, read_model_info
from ..paths import APP_SETTINGS, PROJECT_ROOT, RECORDS_DIR, SOPS_DIR, resolve
from ..pipeline import FramePacket, VideoPipeline
from ..sop_schema import SOPDefinition, Step, load_sop, save_sop
from .editor_page import EditorPage
from .records_page import RecordsPage
from .run_page import RunPage
from .camera_dialog import CameraDialog

VIDEO_FILTER = "影片 (*.mp4 *.avi *.mov *.mkv *.wmv);;所有檔案 (*)"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = self._load_settings()
        self.sop_path: Path | None = None
        self.dirty = False
        self.model_info: ModelInfo | None = None
        self.model_ready = False
        self._requested_model: str | None = None
        self._engine_running = False
        self.resize(1500, 900)

        self.pipeline = VideoPipeline(RECORDS_DIR, self)
        self.pipeline.packet_ready.connect(self._on_packet)
        self.pipeline.events_ready.connect(self._on_events)
        self.pipeline.model_loaded.connect(self._on_model_loaded)
        self.pipeline.source_changed.connect(self._on_source_changed)
        self.pipeline.engine_state.connect(self._on_engine_state)
        self.pipeline.status.connect(lambda text: self.statusBar().showMessage(text, 6000))
        self.pipeline.error.connect(lambda text: QMessageBox.warning(self, "錯誤", text))

        self.run_page = RunPage()
        self.run_page.start_requested.connect(self._start_run)
        self.run_page.stop_requested.connect(self.pipeline.stop_engine)
        self.run_page.confirm_requested.connect(self.pipeline.confirm_step)
        self.run_page.restart_requested.connect(self.pipeline.restart_cycle)
        self.editor = EditorPage()
        self.editor.modified.connect(self._on_modified)
        self.editor.model_selected.connect(self._load_model)
        self.records_page = RecordsPage(RECORDS_DIR)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.run_page, "執行監控")
        self.tabs.addTab(self.editor, "工序編輯")
        self.tabs.addTab(self.records_page, "作業紀錄")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        self._build_menu()
        self._build_toolbar()
        self.pipeline.camera_available.connect(self.camera_button.setEnabled)
        self.model_status = QLabel("模型：未載入")
        self.fps_status = QLabel("")
        self.statusBar().addPermanentWidget(self.fps_status)
        self.statusBar().addPermanentWidget(self.model_status)

        self.pipeline.start()
        self._open_initial_sop()

    # ---- 版面 ------------------------------------------------------------------
    def _build_menu(self):
        menu = self.menuBar().addMenu("檔案(&F)")
        for text, shortcut, handler in [("新增 SOP", QKeySequence.StandardKey.New, self._new_sop),
                                        ("開啟 SOP…", QKeySequence.StandardKey.Open, self._open_sop),
                                        ("儲存", QKeySequence.StandardKey.Save, self._save),
                                        ("另存新檔…", QKeySequence.StandardKey.SaveAs, self._save_as)]:
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.triggered.connect(handler)
            menu.addAction(action)
        menu.addSeparator()
        exit_action = QAction("離開", self)
        exit_action.triggered.connect(self.close)
        menu.addAction(exit_action)

    def _build_toolbar(self):
        toolbar = QToolBar("影像來源")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addWidget(QLabel(" 影像來源 "))
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
        self.source_status = QLabel("  未連線")
        toolbar.addWidget(self.source_status)
        toolbar.addSeparator()
        self.mask_check = QCheckBox("顯示遮罩")
        self.mask_check.setChecked(True)
        self.mask_check.toggled.connect(self._apply_display)
        toolbar.addWidget(self.mask_check)
        toolbar.addWidget(QLabel("  顯示信心度≥ "))
        self.display_score = QDoubleSpinBox()
        self.display_score.setRange(0.3, 1.0)
        self.display_score.setSingleStep(0.05)
        self.display_score.setValue(0.5)
        self.display_score.valueChanged.connect(self._apply_display)
        toolbar.addWidget(self.display_score)

    # ---- 設定檔 ----------------------------------------------------------------
    @staticmethod
    def _load_settings() -> dict:
        try:
            return json.loads(APP_SETTINGS.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_settings(self):
        try:
            APP_SETTINGS.write_text(json.dumps(self.settings, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    # ---- 影像來源 --------------------------------------------------------------
    def _connect_source(self):
        source = self.source_combo.currentData()
        if source.startswith("camera:"):
            self.pipeline.open_camera(int(source.split(":")[1]))
        else:
            start_dir = self.settings.get("video_path") or str(PROJECT_ROOT)
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
        self.editor.clear_video("影像來源已變更，等待影像")
        self.source_status.setText(f"  {label}" if label else "  未連線")
        if not label:
            self.fps_status.setText("")
            self.run_page.clear_video("影像來源已中斷")
            self.editor.clear_video("影像來源已中斷")

    # ---- 背景執行緒結果 ----------------------------------------------------------
    def _on_packet(self, packet: FramePacket):
        try:
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
        if any(e.kind in ("cycle_completed", "cycle_aborted") for e in events):
            self.records_page.mark_stale()

    def _on_engine_state(self, running: bool):
        self._engine_running = running
        self.run_page.set_running(running)
        if not running:
            self.run_page.set_sop(self.editor.sop)

    def _on_tab_changed(self, _index: int):
        if self.tabs.currentWidget() is self.run_page:
            self.run_page.set_sop(self.editor.sop)     # 未執行時同步編輯後的內容

    # ---- 模型 ------------------------------------------------------------------
    def _load_model(self, path: str):
        try:
            info = read_model_info(path)
        except ModelBundleError as exc:
            QMessageBox.warning(self, "模型", str(exc))
            self.editor.set_classes([])
            return
        self.model_info = info
        self.model_ready = False
        self.editor.set_classes(info.classes)
        self.editor.sop.model_version_id = info.model_version_id
        self._requested_model = str(Path(path).resolve())
        self.model_status.setText(f"模型：{info.model_version_id} 載入中…")
        self.pipeline.load_model(path)

    def _on_model_loaded(self, info: ModelInfo | None, device: str):
        self.model_ready = info is not None
        if info is None:
            self.model_status.setText("模型：未載入")
        else:
            self.model_status.setText(f"模型：{info.project_name} {info.model_version_id} · "
                                      f"{len(info.classes)} 類 · {device}")

    # ---- SOP 檔案 --------------------------------------------------------------
    def _open_initial_sop(self):
        last = self.settings.get("last_sop")
        if last and resolve(last).exists():
            return self._open_path(resolve(last))
        example = SOPS_DIR / "example_sop.json"
        if example.exists():
            return self._open_path(example)
        self._apply_sop(SOPDefinition(steps=[Step("工序 1")]), None)

    def _open_path(self, path: Path):
        try:
            sop = load_sop(path)
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
            self.settings["last_sop"] = str(path)
            self._save_settings()
        if sop.model_path:
            model_path = resolve(sop.model_path)
            if str(model_path.resolve()) != self._requested_model:
                self._load_model(str(model_path))
            elif self.model_info:
                self.editor.set_classes(self.model_info.classes)

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        answer = QMessageBox.question(
            self, "尚未儲存", "SOP 有尚未儲存的變更，要先儲存嗎？",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Save:
            return self._save()
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
        path, _ = QFileDialog.getOpenFileName(self, "開啟 SOP", str(SOPS_DIR), "SOP 設定檔 (*.json)")
        if path:
            self._open_path(Path(path))

    def _save(self) -> bool:
        if self.sop_path is None:
            return self._save_as()
        return self._write(self.sop_path)

    def _save_as(self) -> bool:
        default = SOPS_DIR / f"{self.editor.sop.name or 'sop'}.json"
        path, _ = QFileDialog.getSaveFileName(self, "另存 SOP", str(default), "SOP 設定檔 (*.json)")
        return self._write(Path(path), new_file=True) if path else False

    def _write(self, path: Path, new_file: bool = False) -> bool:
        sop = self.editor.sop
        if self.dirty and not new_file and path.exists():
            sop.version += 1          # 覆寫既有檔案時版本 +1，紀錄可追溯是哪個版本的 SOP
        try:
            save_sop(sop, path)
        except OSError as exc:
            QMessageBox.warning(self, "儲存失敗", str(exc))
            return False
        self.sop_path = path
        self.dirty = False
        self.editor.version_label.setText(str(sop.version))
        self.settings["last_sop"] = str(path)
        self._save_settings()
        self._update_title()
        self.statusBar().showMessage(f"已儲存 {path.name}（版本 {sop.version}）", 5000)
        return True

    def _on_modified(self):
        if not self.dirty:
            self.dirty = True
            self._update_title()

    def _update_title(self):
        name = self.sop_path.name if self.sop_path else "未儲存"
        self.setWindowTitle(f"SOP 工序監控 — {self.editor.sop.name}（{name}）{' *' if self.dirty else ''}")

    # ---- 開始作業 --------------------------------------------------------------
    def _start_run(self):
        if self.dirty:
            answer = QMessageBox.question(
                self, "尚未儲存", "SOP 有尚未儲存的變更。要先儲存再開始嗎？\n（選「否」會直接用目前內容開始）",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Cancel or (answer == QMessageBox.StandardButton.Yes
                                                               and not self._save()):
                return
        sop = self.editor.sop
        errors, warnings = sop.validate(self.model_info.classes if self.model_info else None)
        if errors:
            QMessageBox.critical(self, "SOP 設定有誤", "無法開始作業：\n\n" + "\n".join(f"• {e}" for e in errors))
            return
        if not self.model_ready:
            warnings.insert(0, "模型尚未載入完成，在載入前不會有任何偵測結果")
        if self.source_status.text().strip() == "未連線":
            warnings.insert(0, "尚未連線影像來源")
        if warnings and QMessageBox.question(
                self, "確認開始", "\n".join(f"• {w}" for w in warnings) + "\n\n仍要開始作業嗎？"
        ) != QMessageBox.StandardButton.Yes:
            return
        self.run_page.set_sop(sop)
        self.pipeline.start_engine(sop)
        self.tabs.setCurrentWidget(self.run_page)

    def closeEvent(self, event):
        if not self._confirm_discard():
            event.ignore()
            return
        self.pipeline.shutdown()
        event.accept()
