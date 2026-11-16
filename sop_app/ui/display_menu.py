"""A checkable menu that stays open while choosing several preview categories."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMenu


class DisplayCategoriesMenu(QMenu):
    def mouseReleaseEvent(self, event):
        action = self.actionAt(event.position().toPoint())
        if (event.button() == Qt.MouseButton.LeftButton and action is not None
                and action.isEnabled() and action.isCheckable()):
            action.trigger()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        action = self.activeAction()
        if (event.key() == Qt.Key.Key_Space and action is not None
                and action.isEnabled() and action.isCheckable()):
            action.trigger()
            event.accept()
            return
        super().keyPressEvent(event)
