"""定位、遺失保護、SOP 保存與編輯工作流程的回歸測試。"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from sop_app.tracking import WorkpieceLocator, capture_workpiece, transform_roi, workpiece_key
from sop_app.sop_schema import ROI, Condition, Step, SOPDefinition, load_sop, save_sop
from sop_app.detection import Detection, FrameResult
from sop_app.conditions import evaluate_condition
from sop_app.engine import SOPEngine, Phase
from sop_app.pipeline import FramePacket, VideoPipeline
from sop_app.recorder import Recorder, query
from sop_app.ui.editor_page import EditorPage
from sop_app.ui.run_page import RunPage
from sop_app.ui.records_page import RecordsPage


def scene():
    frame = np.zeros((480, 640, 3), np.uint8)
    rng = np.random.default_rng(42)
    for _ in range(400):
        x, y = rng.integers([180, 140], [430, 330])
        cv2.circle(frame, (int(x), int(y)), int(rng.integers(2, 6)),
                   tuple(int(v) for v in rng.integers(70, 255, 3)), -1)
    return frame


TARGET = [(0.25, 0.25), (0.7, 0.25), (0.7, 0.72), (0.25, 0.72)]
IDENTITY = np.array([[1., 0., 0.], [0., 1., 0.]])


class PipelineTracking:
    """模擬背景管線：逐幀定位並把結果與工件識別碼放進畫面封包（介面不再自行定位）。"""

    def __init__(self, workpiece):
        self.locator = WorkpieceLocator(workpiece)
        self.key = workpiece_key(workpiece)

    def packet(self, image, timestamp):
        transform, status = self.locator.locate(image)
        result = FrameResult(image.shape[1], image.shape[0], timestamp, workpiece_transform=transform,
                             tracking_status=status, workpiece_key=self.key)
        return FramePacket(image, image.copy(), result, None, 10, 5)


class TrackingTests(unittest.TestCase):
    def test_continuous_motion_and_bounded_flow_fallback(self):
        reference = scene()
        locator = WorkpieceLocator(capture_workpiece(reference, TARGET))
        for _ in range(3):
            locator.locate(reference)
        for i in range(1, 16):
            affine = cv2.getRotationMatrix2D((320,240), i * .5, 1)
            affine[:,2] += [i*2,i]
            moved = cv2.warpAffine(reference, affine, (640,480))
            actual, _ = locator.locate(moved)
            self.assertIsNotNone(actual)
            expected = np.diag([1/640,1/480]) @ affine @ np.diag([640,480,1])
            np.testing.assert_allclose(actual, expected, atol=.015)
        # 特徵配對暫時失敗時使用實際光流；超過上限必須失效，不能永遠漂移。
        with patch.object(locator, '_estimate_features', return_value=(None,0)), \
             patch.object(locator, '_estimate_sift', return_value=(None,0)), \
             patch.object(locator, '_estimate_template', return_value=(None,0)):
            outputs = [locator.locate(moved)[0] for _ in range(locator.MAX_FLOW_FRAMES + 3)]
        self.assertTrue(any(m is not None for m in outputs))
        self.assertIsNone(outputs[-1])
        self.assertIsNone(locator._flow_gray)

    def test_edge_template_recovers_low_feature_frame_translation(self):
        reference = scene()
        locator = WorkpieceLocator(capture_workpiece(reference, TARGET))
        shifted = cv2.warpAffine(reference, np.float32([[1,0,18],[0,1,9]]), (640,480))
        with patch.object(locator, '_estimate_features', return_value=(None,0)):
            self.assertIsNotNone(locator.locate(shifted)[0])
            self.assertIsNotNone(locator.locate(shifted)[0])
            actual, status = locator.locate(shifted)
        self.assertIsNotNone(actual)
        self.assertIsNotNone(locator._flow_gray)
        expected = IDENTITY.copy()
        expected[:,2] = [18/640, 9/480]
        np.testing.assert_allclose(actual, expected, atol=.012)

    def test_translation_rotation_scale_and_loss_recovery(self):
        reference = scene()
        locator = WorkpieceLocator(capture_workpiece(reference, TARGET))
        affine = cv2.getRotationMatrix2D((320, 240), 18, 0.85)
        affine[:, 2] += [55, 25]
        moved = cv2.warpAffine(reference, affine, (640, 480))
        self.assertIsNotNone(locator.locate(moved)[0])
        self.assertIsNotNone(locator.locate(moved)[0])
        actual, _ = locator.locate(moved)
        self.assertIsNotNone(actual)
        expected = np.diag([1/640, 1/480]) @ affine @ np.diag([640, 480, 1])
        np.testing.assert_allclose(actual, expected, atol=0.01)
        self.assertIsNone(locator.locate(np.zeros_like(reference))[0])
        # 單一失效幀後若位置一致，應立即恢復，不再重新等待三幀。
        self.assertIsNotNone(locator.locate(moved)[0])

    def test_contour_only_reference_can_follow_translation(self):
        reference = scene()
        with patch('sop_app.tracking.cv2.ORB_create') as create:
            create.return_value.detectAndCompute.return_value = ([], None)
            locator = WorkpieceLocator(capture_workpiece(reference, TARGET))
            moved = cv2.warpAffine(reference, np.float32([[1,0,18],[0,1,9]]), (640,480))
            for _ in range(3):
                actual, status = locator.locate(moved)
        self.assertIsNotNone(actual)
        self.assertIsNotNone(locator._flow_gray)
        expected = IDENTITY.copy()
        expected[:, 2] = [18/640, 9/480]
        np.testing.assert_allclose(actual, expected, atol=.012)
        self.assertIsNone(locator.locate(np.zeros_like(reference))[0])

    def test_template_handles_resolution_change(self):
        reference = scene()
        locator = WorkpieceLocator(capture_workpiece(reference, TARGET))
        with patch.object(locator, '_estimate_features', return_value=(None, 0)):
            for _ in range(3):
                actual, _ = locator.locate(cv2.resize(reference, (960, 720)))
        self.assertIsNotNone(actual)
        np.testing.assert_allclose(actual, IDENTITY, atol=.015)

    def test_template_rotation_seeds_fast_flow_and_rechecks_reference(self):
        reference = scene()
        locator = WorkpieceLocator(capture_workpiece(reference, TARGET))
        affine = cv2.getRotationMatrix2D((320, 240), 6, 1)
        affine[:, 2] += [18, 9]
        moved = cv2.warpAffine(reference, affine, (640, 480))
        expected = np.diag([1/640, 1/480]) @ affine @ np.diag([640, 480, 1])
        with patch.object(locator, '_estimate_features', return_value=(None, 0)), \
             patch.object(locator, '_estimate_sift', return_value=(None, 0)), \
             patch.object(locator, '_estimate_template', wraps=locator._estimate_template) as template:
            for _ in range(5):
                actual, _ = locator.locate(moved)
            self.assertEqual(template.call_count, 1)
            np.testing.assert_allclose(actual, expected, atol=.015)
            for _ in range(locator.MAX_RECHECK_FRAMES - 4):
                locator.locate(moved)
            self.assertEqual(template.call_count, 2)
        self.assertIsNone(locator.locate(np.zeros_like(reference))[0])

    def test_sift_relocalizes_scale_rotation_and_rejects_blank(self):
        reference = scene()
        locator = WorkpieceLocator(capture_workpiece(reference, TARGET))
        affine = cv2.getRotationMatrix2D((320, 240), 25, .8)
        affine[:, 2] += [30, 15]
        moved = cv2.warpAffine(reference, affine, (640, 480))
        actual, count = locator._estimate_sift(cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY))
        self.assertIsNotNone(actual)
        self.assertGreaterEqual(count, 12)
        expected = np.diag([1/640, 1/480]) @ affine @ np.diag([640, 480, 1])
        np.testing.assert_allclose(actual, expected, atol=.015)
        self.assertIsNone(locator._estimate_sift(np.zeros((480,640), np.uint8))[0])

    def test_two_percent_compares_estimates_and_corrects_large_error(self):
        locator = WorkpieceLocator(capture_workpiece(scene(), TARGET))
        diagonal = np.linalg.norm(np.ptp(np.array(TARGET) * [640, 480], axis=0))
        for ratio in (.01, .03):
            verified = IDENTITY.copy()
            verified[0, 2] = diagonal * ratio / 640
            self.assertAlmostEqual(locator._deviation_ratio(IDENTITY, verified, 640, 480), ratio)
            locator._flow_gray = np.zeros((480,640), np.uint8)
            locator._flow_age = locator.MAX_RECHECK_FRAMES
            with patch.object(locator, '_estimate_flow', return_value=(IDENTITY.copy(), 30)), \
                 patch.object(locator, '_estimate_features', return_value=(verified, 30)):
                matrix, _ = locator._estimate(scene())
            expected = verified
            np.testing.assert_allclose(matrix, expected)

    def test_first_frame_and_recovery_have_no_confirmation_delay(self):
        reference = scene()
        locator = WorkpieceLocator(capture_workpiece(reference, TARGET))
        self.assertIsNotNone(locator.locate(reference)[0])
        for _ in range(4):
            self.assertIsNone(locator.locate(np.zeros_like(reference))[0])
        moved = cv2.warpAffine(reference, np.float32([[1,0,40],[0,1,20]]), (640,480))
        matrix, _ = locator.locate(moved)
        self.assertIsNotNone(matrix)
        self.assertAlmostEqual(matrix[0, 2], 40/640, delta=.01)
        self.assertAlmostEqual(matrix[1, 2], 20/480, delta=.01)

    def test_low_texture_rejected(self):
        with self.assertRaises(ValueError):
            capture_workpiece(np.zeros((480, 640, 3), np.uint8), TARGET)

    def test_condition_follows_region_and_never_disappears_when_lost(self):
        roi = ROI("作業點", [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)], "workpiece")
        matrix = IDENTITY.copy()
        matrix[0, 2] = 0.4
        frame = FrameResult(100, 100, 0, [Detection("tool", 0.9, (55, 15, 65, 25))], matrix)
        self.assertTrue(evaluate_condition(Condition(label="tool", roi=roi.name), frame, {roi.name: roi}).met)
        frame.workpiece_transform = None
        frame.detections = []
        result = evaluate_condition(Condition("disappear", "tool", roi.name), frame, {roi.name: roi})
        self.assertFalse(result.met)
        self.assertTrue(result.error)
        matrix[0, 2] = 0.8
        self.assertIsNone(transform_roi(roi, matrix))

    def test_loss_clears_hold_window_and_logs_once(self):
        roi = ROI("作業點", [(0, 0), (1, 0), (1, 1), (0, 1)], "workpiece")
        sop = SOPDefinition(rois=[roi], steps=[Step("移除工具", conditions=[Condition("disappear", "tool", roi.name)])])
        engine = SOPEngine(sop)
        events = engine.start(0)
        for t in (0, 0.5, 0.9):
            events += engine.update(FrameResult(100, 100, t, [], IDENTITY))
        for t in (1, 2, 3):
            events += engine.update(FrameResult(100, 100, t))
        self.assertEqual(sum(e.kind == "position_lost" for e in events), 1)
        events += engine.update(FrameResult(100, 100, 3.1, [], IDENTITY))
        self.assertEqual(engine.phase, Phase.RUNNING)
        events += engine.update(FrameResult(100, 100, 4.2, [], IDENTITY))
        self.assertEqual(engine.phase, Phase.CYCLE_DONE)
        with tempfile.TemporaryDirectory() as folder:
            recorder = Recorder(Path(folder))
            recorder.handle(events, "測試", 1, "model")
            recorder.close()
            self.assertEqual(len(query(Path(folder) / "records.db", "SELECT * FROM positioning_events")), 2)

    def test_schema_roundtrip_and_legacy(self):
        sop = SOPDefinition(workpiece=capture_workpiece(scene(), TARGET),
                            rois=[ROI("點", TARGET, "workpiece")], steps=[Step("人工")])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sop.json"
            save_sop(sop, path)
            restored = load_sop(path)
            self.assertEqual(sop, restored)
            self.assertEqual(restored.validate()[0], [])
        legacy = SOPDefinition.from_dict({"rois": [{"name": "old", "points": TARGET}]})
        self.assertEqual(legacy.rois[0].anchor, "fixed")
        self.assertIsNone(legacy.workpiece)

    def test_model_failure_cannot_satisfy_disappear(self):
        result = evaluate_condition(Condition("disappear", "tool"),
                                    FrameResult(100, 100, 1, detection_valid=False), {})
        self.assertFalse(result.met)

    def test_reset_and_forbidden_do_not_treat_loss_as_removal(self):
        roi = ROI("作業點", TARGET, "workpiece")
        disappear = Condition("disappear", "tool", roi.name)
        sop = SOPDefinition(rois=[roi], steps=[Step("人工", forbidden=[disappear], hold_sec=0)])
        sop.cycle.reset_conditions = [disappear]
        sop.cycle.reset_hold_sec = 0
        engine = SOPEngine(sop)
        engine.start(0)
        events = engine.update(FrameResult(640, 480, 2))
        self.assertFalse(any(e.kind == "alarm" for e in events))
        engine.confirm_step(3)
        engine.update(FrameResult(640, 480, 10))
        self.assertEqual(engine.cycle, 1)
        engine.update(FrameResult(640, 480, 11, [], IDENTITY))
        self.assertEqual(engine.cycle, 2)


class EditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_editor_expands_all_steps_and_regions_in_one_scrollable_page(self):
        editor = EditorPage()
        self.addCleanup(editor.close)
        editor.set_classes(['grip'])
        editor.set_sop(SOPDefinition(rois=[ROI('左側', TARGET)], steps=[
            Step('安裝左握把', conditions=[Condition(label='grip', roi='左側')]),
            Step('確認裝配')]))
        editor.resize(1100, 850)
        editor.show()
        self.app.processEvents()
        self.assertTrue(editor.page_scroll.isVisible())
        self.assertTrue(editor.sop_box.isVisible())
        self.assertEqual(len(editor.step_editors), 2)
        self.assertEqual(len(editor.roi_rows), 1)
        self.assertTrue(editor.roi_rows[0].name_edit.isVisible())
        self.assertTrue(editor.validation_box.isVisible())
        # A loaded SOP is a compact list; each step still shows its name and full rule summary.
        for form in editor.step_editors:
            self.assertTrue(form.step_name_edit.isVisible())
            self.assertIn(form.step.name, form.header_row.toolTip())
            self.assertFalse(form.condition_table.isVisible())
        self.assertIn('grip＠左側 出現', editor.step_editors[0].header_row.toolTip())
        packet = FramePacket(scene(), scene(), FrameResult(640, 480, 0, detection_valid=True), None, 10, 5)
        editor.on_packet(packet)
        self.assertIn('左側（固定畫面）', editor.roi_rows[0].toolTip())
        self.assertIn('條件本幀成立', editor.roi_rows[0].toolTip())
        editor.expand_all_button.click()
        self.app.processEvents()
        self.assertGreater(editor.page_scroll.verticalScrollBar().maximum(), 0)
        for form in editor.step_editors:
            for field in (form.step_name_edit, form.condition_table, form.forbidden_table,
                          form.hold_spin, form.forbidden_hold_spin):
                self.assertTrue(field.isVisible())
        # Choosing a trial target only changes the trial; every form remains
        # available and edits apply to the step being edited directly.
        editor.validation_step.setCurrentIndex(1)
        name = editor.step_editors[1].step_name_edit
        name.setFocus()
        QTest.keyClick(name, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClicks(name, 'assembly-check')
        self.app.processEvents()
        self.assertEqual(editor.sop.steps[0].name, '安裝左握把')
        self.assertEqual(editor.sop.steps[1].name, 'assembly-check')
        self.assertTrue(all(form.isVisible() for form in editor.step_editors))

    def test_monitor_idle_previews_regions_and_clears_lost_positions(self):
        frame = scene()
        roi = ROI("安裝位置", [(.4,.4),(.5,.4),(.5,.5),(.4,.5)], "workpiece")
        fixed = ROI("桌面", [(0,0),(.1,0),(.1,.1),(0,.1)])
        sop = SOPDefinition(workpiece=capture_workpiece(frame, TARGET), rois=[fixed,roi], steps=[Step("人工")])
        run = RunPage()
        self.addCleanup(run.close)
        run.set_sop(sop)
        self.assertTrue(run.start_button.isEnabled())
        self.assertEqual(run.start_button.text(), '開始作業')
        self.assertFalse(run.start_button.icon().isNull())
        started = []
        run.start_requested.connect(lambda: started.append(True))
        run.start_button.click()
        self.assertEqual(started, [True])
        run.display_mode.setCurrentIndex(run.display_mode.findData('all'))
        self.assertEqual([r.name for r in run.video._rois], ['桌面','安裝位置'])
        self.assertEqual(run.video._stale, {'安裝位置'})
        packet = FramePacket(frame, frame.copy(), FrameResult(640,480,0), None,10,5)
        pipeline = PipelineTracking(sop.workpiece)
        for _ in range(3):
            run.on_packet(pipeline.packet(frame, 0))
        self.assertTrue(run.start_button.isEnabled())
        self.assertEqual(run.start_button.text(), '開始作業')
        self.assertEqual([r.name for r in run.video._rois], ['桌面','安裝位置'])
        self.assertIn('2/2', run.tracking_label.text())
        # 監控待機沒有 snapshot，也應持續更新座標。
        moved = cv2.warpAffine(frame, np.float32([[1,0,20],[0,1,10]]), (640,480))
        run.on_packet(pipeline.packet(moved, 1))
        self.assertAlmostEqual(run.video._rois[1].points[0][0], .4+20/640, places=2)
        blank = np.zeros_like(frame)
        run.on_packet(pipeline.packet(blank, 2))
        self.assertEqual([r.name for r in run.video._rois], ['桌面','安裝位置'])
        self.assertEqual(run.video._stale, set())
        run.on_packet(pipeline.packet(blank, 3))
        self.assertEqual(run.video._stale, {'安裝位置'})
        self.assertIn('位置跟隨暫停', run.tracking_label.text())
        self.assertIn('停止累積', run.tracking_label.text())
        # 正式執行不能以本地預覽覆蓋引擎遺失結果。
        run.set_running(True)
        for _ in range(3):
            run.on_packet(packet)
        self.assertEqual([r.name for r in run.video._rois], ['桌面','安裝位置'])
        self.assertEqual(run.video._stale, {'安裝位置'})
        run.clear_video('中斷')
        self.assertEqual(run.video._rois, [])
        self.assertIsNone(run._transform)

    def test_monitor_hides_frame_by_frame_tracking_details_when_step_has_no_region(self):
        frame = scene()
        sop = SOPDefinition(
            workpiece=capture_workpiece(frame, TARGET),
            rois=[ROI('安裝位置', TARGET, 'workpiece')],
            steps=[Step('取料')])
        run = RunPage()
        self.addCleanup(run.close)
        run.set_sop(sop)
        texts = {run.tracking_label.text()}
        for index, image in enumerate([frame, frame, frame, np.zeros_like(frame), frame]):
            run.on_packet(FramePacket(image, image, FrameResult(640,480,index), None,10,5))
            texts.add(run.tracking_label.text())
        self.assertEqual(texts, {'目前工序不使用工作區域'})
        self.assertNotIn('特徵', run.tracking_label.text())

    def test_direct_handlebar_setup_binds_visible_condition_and_saves(self):
        editor = EditorPage()
        self.addCleanup(editor.close)
        editor.set_classes(['grip'])
        editor.set_sop(SOPDefinition())
        frame = scene()
        packet = FramePacket(frame,frame.copy(),FrameResult(640,480,0),None,10,5)
        editor.on_packet(packet)
        editor.target_button.click()
        editor._on_roi_drawn(TARGET)
        editor.draw_button.click()
        position = [(.4,.4),(.5,.4),(.5,.5),(.4,.5)]
        editor._on_roi_drawn(position)
        editor.roi_rows[-1].name_edit.setText('左側安裝位置')
        editor.roi_rows[-1].name_edit.editingFinished.emit()
        editor.add_step_button.click()
        conditions = editor.step_editors[0].condition_table
        conditions.add_button.click()
        region = conditions.field_widget(0, 'roi')
        region.setCurrentIndex(region.findData('左側安裝位置'))
        self.assertEqual(editor.sop.rois[0].anchor,'workpiece')
        self.assertEqual(editor.sop.steps[0].conditions[0].roi,'左側安裝位置')
        self.assertEqual(editor.sop.steps[0].conditions[0].label,'grip')
        self.assertEqual(editor.roi_rows[0].name_edit.text(), '左側安裝位置')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'handlebar.json'
            save_sop(editor.sop, path)
            restored = load_sop(path)
        editor.on_save_result('saved')
        self.assertIn('已儲存', editor.guide_label.text())
        self.assertEqual(restored.steps[0].conditions[0].roi, '左側安裝位置')
        self.assertEqual(restored.rois[0].anchor, 'workpiece')

    def test_clear_workpiece_removes_dependent_positions_and_conditions(self):
        frame = scene()
        editor = EditorPage()
        self.addCleanup(editor.close)
        moving = ROI('左側安裝位置', [(.3,.3),(.4,.3),(.4,.4),(.3,.4)], 'workpiece')
        fixed = ROI('桌面', [(0,0),(.2,0),(.2,.2),(0,.2)], 'fixed')
        sop = SOPDefinition(
            workpiece=capture_workpiece(frame, TARGET),
            rois=[moving, fixed],
            steps=[Step('安裝', conditions=[
                Condition(label='grip', roi='左側安裝位置'),
                Condition(label='part', roi='桌面')])])
        editor.set_sop(sop)
        with patch('sop_app.ui.editor_page.QMessageBox.question',
                   return_value=QMessageBox.StandardButton.Yes):
            editor._clear_workpiece()
        self.assertIsNone(editor.sop.workpiece)
        self.assertEqual([roi.name for roi in editor.sop.rois], ['桌面'])
        self.assertEqual([c.roi for c in editor.sop.steps[0].conditions], ['桌面'])
        self.assertFalse(editor.reference_button.isEnabled())
        self.assertIn('作業區域 1 個', editor.setup_summary.text())
        self.assertEqual([row.roi.name for row in editor.roi_rows], ['桌面'])
        self.assertEqual(editor.step_editors[0].condition_table.conditions()[0].roi, '桌面')

    def test_following_position_can_be_outside_hidden_tracking_reference(self):
        frame = scene()
        editor = EditorPage()
        self.addCleanup(editor.close)
        editor.set_classes(['grip'])
        editor.set_sop(SOPDefinition(workpiece=capture_workpiece(frame, TARGET), steps=[Step('安裝')]))
        editor.reference_button.setChecked(True)
        editor.draw_button.click()
        outside = [(.05,.05),(.15,.05),(.15,.15),(.05,.15)]
        editor._on_roi_drawn(outside)
        editor.roi_rows[-1].name_edit.setText('左側位置')
        editor.roi_rows[-1].name_edit.editingFinished.emit()
        self.assertEqual(editor.sop.rois[0].points, outside)
        self.assertEqual(editor.sop.rois[0].anchor, 'workpiece')
        self.assertNotIn('車把手主工件', [roi.name for roi in editor.video._rois])

    def test_monitor_step_and_all_modes_filter_objects(self):
        left = ROI('左',[(.1,.1),(.3,.1),(.3,.4),(.1,.4)])
        right = ROI('右',[(.6,.1),(.8,.1),(.8,.4),(.6,.4)])
        sop = SOPDefinition(rois=[left,right], steps=[
            Step('左步驟',conditions=[Condition(label='left_part',roi='左')],hold_sec=0),
            Step('右步驟',conditions=[Condition(label='right_part',roi='右')],hold_sec=0)])
        run = RunPage()
        self.addCleanup(run.close)
        run.set_sop(sop)
        run.set_running(True)
        engine = SOPEngine(sop)
        engine.start(0)
        detections = [Detection('left_part',.9,(10,10,20,20)),Detection('right_part',.9,(60,10,70,20)),
                      Detection('noise',.9,(80,70,90,80))]
        raw = np.zeros((100,100,3),np.uint8)
        result = FrameResult(100,100,0,detections)
        packet = FramePacket(raw, np.full_like(raw,255), result, engine.snapshot(),10,5,None,
                             ('left_part','right_part','noise'),.5,False)
        run.on_packet(packet)
        self.assertEqual([r.name for r in run.video._rois],['左'])
        self.assertIn('目前流程 1：左步驟',run.display_summary.text())
        self.assertIn('left_part',run.display_summary.text())
        self.assertGreater(run.video._image.pixelColor(15,15).red(), 0)
        self.assertEqual(run.video._image.pixelColor(85,75).red(), 0)
        run.display_mode.setCurrentIndex(run.display_mode.findData('all'))
        self.assertEqual([r.name for r in run.video._rois],['左','右'])
        self.assertIn('目前偵測 3 個物件',run.display_summary.text())
        # 左側完成後，同一個 snapshot 切到右側，舊框立即消失。
        engine.update(FrameResult(100,100,1,[detections[0]]))
        packet.snapshot = engine.snapshot()
        run.display_mode.setCurrentIndex(run.display_mode.findData('step'))
        run.on_packet(packet)
        self.assertEqual([r.name for r in run.video._rois],['右'])
        self.assertIn('目前流程 2：右步驟',run.display_summary.text())

    def test_monitor_new_sop_cannot_reuse_old_transform(self):
        run = RunPage()
        self.addCleanup(run.close)
        run._transform = IDENTITY.copy()
        run.set_sop(SOPDefinition(rois=[ROI('新位置',TARGET,'workpiece')]))
        run.display_mode.setCurrentIndex(run.display_mode.findData('all'))
        self.assertIsNone(run._transform)
        self.assertEqual(run.video._rois, [])
        self.assertIn('區域超出畫面', run.tracking_label.text())

    def test_capture_bind_preview_and_reload(self):
        editor = EditorPage()
        editor.set_classes(["tool"])
        editor.set_sop(SOPDefinition(steps=[Step("壓合")]))
        frame = scene()
        packet = FramePacket(frame, frame.copy(), FrameResult(640, 480, 0), None, 10, 5)
        editor.on_packet(packet)
        editor._begin_target()
        editor._on_roi_drawn(TARGET)
        self.assertTrue(editor.reference_button.isChecked())
        self.assertFalse(editor.draw_button.isChecked())
        editor.draw_button.click()
        self.assertTrue(editor.draw_button.isChecked())
        self.assertTrue(editor.video._draw_mode)
        editor._on_roi_drawn([(0.4, 0.4), (0.5, 0.4), (0.5, 0.5), (0.4, 0.5)])
        editor.roi_rows[-1].name_edit.setText('壓合位置')
        editor.roi_rows[-1].name_edit.editingFinished.emit()
        self.assertEqual(editor.sop.rois[0].anchor, "workpiece")
        conditions = editor.step_editors[0].condition_table
        conditions.add_button.click()
        region = conditions.field_widget(0, 'roi')
        region.setCurrentIndex(region.findData('壓合位置'))
        self.assertEqual(editor.sop.steps[0].conditions[0].roi, '壓合位置')
        editor.reference_button.setChecked(False)
        pipeline = PipelineTracking(editor.sop.workpiece)
        for _ in range(3):
            editor.on_packet(pipeline.packet(frame, 0))
        self.assertIsNotNone(editor._preview_result.workpiece_transform)
        # 舊參考工件的定位結果不能套用到目前設定。
        stale = pipeline.packet(frame, 1)
        stale.result.workpiece_key = "old-reference"
        editor.on_packet(stale)
        self.assertIsNone(editor._preview_result.workpiece_transform)
        saved = SOPDefinition.from_dict(editor.sop.to_dict())
        editor.set_sop(saved)
        self.assertTrue(editor.reference_button.isEnabled())
        run = RunPage()
        run.set_sop(saved)
        run.set_running(True)
        engine = SOPEngine(saved)
        engine.start(0)
        engine.update(packet.result)
        packet.snapshot = engine.snapshot()
        run.on_packet(packet)
        self.assertEqual([roi.name for roi in run.video._rois], ['壓合位置'])
        self.assertEqual(run.video._stale, {'壓合位置'})
        self.assertIn('位置跟隨暫停', run.tracking_label.text())
        editor.close()
        run.close()

    def test_failed_reference_can_be_redrawn_and_cancelled(self):
        editor = EditorPage()
        frame = np.zeros((480, 640, 3), np.uint8)
        packet = FramePacket(frame, frame.copy(), FrameResult(640, 480, 0), None, 10, 5)
        editor.on_packet(packet)
        editor._begin_target()
        with patch("sop_app.ui.editor_page.QMessageBox.warning"):
            editor._on_roi_drawn(TARGET)
        self.assertTrue(editor._capture_target)
        self.assertTrue(editor.video._draw_mode)
        self.assertIsNone(editor.sop.workpiece)
        editor.live_button.click()
        self.assertFalse(editor.freeze_check.isChecked())
        self.assertFalse(editor._capture_target)
        self.assertFalse(editor.video._draw_mode)
        editor.on_packet(packet)
        self.assertIs(editor._packet, packet)
        editor.close()

    def test_mouse_drag_reference_then_explicit_region_action(self):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        editor = EditorPage()
        editor.resize(1500, 900)
        editor.show()
        self.app.processEvents()
        frame = scene()
        packet = FramePacket(frame, frame.copy(), FrameResult(640, 480, 0), None, 10, 5)
        editor.on_packet(packet)
        editor.target_button.click()

        def drag(x1, y1, x2, y2):
            rect = editor.video._image_rect()
            start = QPoint(round(rect.x() + x1 * rect.width()), round(rect.y() + y1 * rect.height()))
            end = QPoint(round(rect.x() + x2 * rect.width()), round(rect.y() + y2 * rect.height()))
            QTest.mousePress(editor.video, Qt.MouseButton.LeftButton, pos=start)
            QTest.mouseMove(editor.video, end)
            QTest.mouseRelease(editor.video, Qt.MouseButton.LeftButton, pos=end)

        drag(.25, .25, .7, .72)
        self.assertTrue(editor.reference_button.isChecked())
        self.assertFalse(editor.video._draw_mode)
        editor.draw_button.click()
        self.assertTrue(editor.video._draw_mode)
        drag(.4, .4, .5, .5)
        editor.roi_rows[-1].name_edit.setText('安裝位置')
        editor.roi_rows[-1].name_edit.editingFinished.emit()
        self.assertEqual(editor.sop.rois[0].name, "安裝位置")
        self.assertEqual(editor.sop.rois[0].anchor, "workpiece")
        editor.live_button.click()
        self.assertFalse(editor.reference_button.isChecked())
        self.assertFalse(editor.freeze_check.isChecked())
        editor.on_packet(packet)
        self.assertIs(editor._packet, packet)
        editor.close()

    def test_pipeline_locates_and_records_current_region(self):
        from types import SimpleNamespace
        frame = scene()
        sop = SOPDefinition(workpiece=capture_workpiece(frame, TARGET), rois=[ROI("點", TARGET, "workpiece")],
                            steps=[Step("放置", conditions=[Condition(label="tool", roi="點")], hold_sec=0)])
        detector = SimpleNamespace(classes=["tool"], info=SimpleNamespace(model_version_id="test"),
                                   detect=lambda image: [Detection("tool", 0.9, (250, 200, 280, 230))])
        with tempfile.TemporaryDirectory() as folder:
            recorder = Recorder(Path(folder))
            pipe = VideoPipeline(Path(folder))
            pipe._source = None
            pipe._engine = None
            pipe._locator = None
            pipe._detector = detector
            pipe._now = 0
            pipe._last_display = None
            pipe._show_masks, pipe._min_score = False, 0.5
            pipe._fps, pipe._last_frame_at = 0, None
            packets = []
            pipe.packet_ready.connect(packets.append)
            pipe._handle_command("start", sop, recorder)
            for t in (0, 0.1, 0.2):
                pipe.acknowledge_packet()
                pipe._process(np.zeros_like(frame) if t == 0 else frame, t, recorder)
            self.assertEqual(pipe._engine.phase, Phase.CYCLE_DONE)
            self.assertIsNotNone(packets[-1].result.workpiece_transform)
            self.assertTrue(list((Path(folder) / "snapshots").glob("*.jpg")))
            page = RecordsPage(Path(folder))
            page.refresh()
            page.cycles.selectRow(0)
            self.assertEqual(len(query(Path(folder) / 'records.db', 'SELECT * FROM positioning_events')), 2)
            self.assertEqual(page.positioning.rowCount(), 1)
            self.assertIn('判定暫停 1 次', page.positioning_summary.text())
            self.assertIn('已恢復 1 次', page.positioning_summary.text())
            page.close()
            pipe._handle_command("stop", None, recorder)
            self.assertIsNone(pipe._locator)
            recorder.close()
