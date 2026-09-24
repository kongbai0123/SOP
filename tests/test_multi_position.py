import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from unittest.mock import patch
import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox
from sop_app.sop_schema import ROI, Condition, SOPDefinition, Step, Workpiece
from sop_app.conditions import evaluate_conditions
from sop_app.detection import Detection, FrameResult
from sop_app.engine import SOPEngine, Phase
from sop_app.pipeline import FramePacket
from sop_app.ui.editor_page import EditorPage


def region(name, x):
    return ROI(name, [(x,.2),(x+.2,.2),(x+.2,.6),(x,.6)])


class MultiPositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_editor(self):
        editor = EditorPage()
        self.addCleanup(editor.close)
        editor.set_sop(SOPDefinition(rois=[region('左', .1), region('右', .6)],
                                    steps=[Step('安裝', conditions=[Condition(label='grip', roi='左')])]))
        editor.set_classes(['grip'])
        editor.resize(1500,900)
        editor.show()
        self.app.processEvents()
        frame = np.zeros((480,640,3), np.uint8)
        editor.on_packet(FramePacket(frame, frame, FrameResult(640,480,0), None, 10,5))
        editor.roi_list.setCurrentRow(0)
        editor._begin_roi_edit()
        return editor

    def test_drag_resize_cancel_and_undo(self):
        editor = self.make_editor()
        video = editor.video
        def point(x,y):
            return video._to_widget(video._image_rect(), x,y).toPoint()
        def drag(start,end):
            QTest.mousePress(video, Qt.MouseButton.LeftButton, pos=point(*start))
            QTest.mouseMove(video, point(*end))
            QTest.mouseRelease(video, Qt.MouseButton.LeftButton, pos=point(*end))
        original = list(editor.sop.rois[0].points)
        drag((.2,.4),(.3,.4))
        self.assertAlmostEqual(editor.sop.rois[0].points[0][0], .2, places=2)
        editor._undo_roi()
        self.assertEqual(editor.sop.rois[0].points, original)
        editor._redo_roi()
        self.assertAlmostEqual(editor.sop.rois[0].points[0][0], .2, places=2)
        drag((.4,.6),(.5,.7))
        self.assertAlmostEqual(editor.sop.rois[0].points[2][1], .7, places=2)
        resized = list(editor.sop.rois[0].points)
        QTest.mousePress(video, Qt.MouseButton.LeftButton, pos=point(.3,.4))
        QTest.mouseMove(video, point(.4,.5))
        QTest.keyClick(video, Qt.Key.Key_Escape)
        QTest.mouseRelease(video, Qt.MouseButton.LeftButton, pos=point(.4,.5))
        self.assertEqual(editor.sop.rois[0].points, resized)
        editor._return_live()
        self.assertFalse(video._editable)

    def test_duplicate_rename_apply_and_history(self):
        editor = self.make_editor()
        editor._duplicate_roi()
        self.assertEqual(len(editor.sop.rois), 3)
        editor.roi_list.currentItem().setText('第三位置')
        with patch('sop_app.ui.editor_page.QInputDialog.getItem', return_value=('grip', True)):
            editor._apply_roi()
            editor._apply_roi()
        self.assertEqual(len(editor.sop.steps[0].conditions), 2)
        editor.roi_list.currentItem().setText('新名稱')
        self.assertEqual(editor.sop.steps[0].conditions[-1].roi, '新名稱')
        editor._undo_roi()
        self.assertEqual(editor.sop.steps[0].conditions[-1].roi, '第三位置')
        with patch('sop_app.ui.editor_page.QMessageBox.information') as info:
            editor._delete_roi()
        self.assertIn('安裝', info.call_args.args[2])
        self.assertEqual(len(editor.sop.rois), 3)
        restored = SOPDefinition.from_dict(editor.sop.to_dict())
        self.assertEqual(restored, editor.sop)

    def test_all_any_and_missing_location(self):
        rois = [region('左', .1), region('右', .6)]
        conditions = [Condition(label='grip', roi=r.name) for r in rois]
        frame = FrameResult(100,100,0,[Detection('grip',.9,(12,25,25,45))])
        for mode, expected in [('all',False),('any',True)]:
            sop = SOPDefinition(rois=rois, steps=[Step(conditions=conditions, hold_sec=0, completion_mode=mode)])
            loaded = SOPDefinition.from_dict(sop.to_dict())
            engine = SOPEngine(loaded)
            engine.start(0)
            engine.update(frame)
            self.assertEqual(engine.phase == Phase.CYCLE_DONE, expected)
        met, _ = evaluate_conditions(conditions, frame, {'左':rois[0]}, 'any')
        self.assertFalse(met)
        frame.detections.append(Detection('grip',.9,(62,25,75,45)))
        self.assertTrue(evaluate_conditions(conditions,frame,{r.name:r for r in rois})[0])
        old = SOPDefinition.from_dict({'schema_version':2,'steps':[{'name':'舊設定'}]})
        self.assertEqual(old.steps[0].completion_mode, 'all')

    def test_convert_fixed_to_workpiece_preserves_current_position(self):
        editor = self.make_editor()
        editor._return_live()
        editor.sop.workpiece = Workpiece(points=[(0,0),(1,0),(1,1),(0,1)])
        matrix = np.array([[1.,0.,.05],[0.,1.,.1]])
        editor._preview_result = FrameResult(640,480,1,workpiece_transform=matrix)
        original = np.array(editor.sop.rois[0].points)
        editor._anchor_selected_roi()
        self.assertEqual(editor.sop.rois[0].anchor,'workpiece')
        transformed = np.array(editor.sop.rois[0].points) @ matrix[:,:2].T + matrix[:,2]
        np.testing.assert_allclose(transformed,original)
        editor._undo_roi()
        self.assertEqual(editor.sop.rois[0].anchor,'fixed')
        editor._preview_result = FrameResult(640,480,2)
        editor._anchor_selected_roi()
        self.assertEqual(editor.sop.rois[0].anchor,'fixed')

    def test_sequential_positions_do_not_skip(self):
        sop = SOPDefinition(rois=[region('左',.1),region('右',.6)],
                            steps=[Step(name=r, conditions=[Condition(label='grip',roi=r)],hold_sec=0)
                                   for r in ['左','右']])
        engine = SOPEngine(sop)
        engine.start(0)
        engine.update(FrameResult(100,100,1,[Detection('grip',.9,(62,25,75,45))]))
        self.assertEqual(engine.index,0)
        engine.update(FrameResult(100,100,2,[Detection('grip',.9,(12,25,25,45))]))
        self.assertEqual(engine.index,1)
        self.assertEqual(engine.phase,Phase.RUNNING)
