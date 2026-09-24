"""車把手快速設定：一次完成安裝位置命名、物件配對與流程建立。"""
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
                               QLineEdit, QVBoxLayout)


class QuickPositionDialog(QDialog):
    def __init__(self, default_name: str, classes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新增車把手安裝位置")
        self.setMinimumWidth(430)
        title = QLabel("建立安裝位置並配對辨識物件")
        title.setStyleSheet("font-size:16px; font-weight:bold; color:#163f67")
        text = QLabel("儲存後會自動建立一道安裝流程；仍可在進階設定中調整信心度與持續時間。")
        text.setWordWrap(True)
        text.setStyleSheet("color:#607d8b")
        self.name_edit = QLineEdit(default_name)
        self.name_edit.selectAll()
        self._has_classes = bool(classes)
        self.object_box = QComboBox()
        self.object_box.addItem("請選擇到位物件…" if classes else "模型未載入，稍後在進階設定選擇", "")
        for name in classes:
            self.object_box.addItem(name, name)
        form = QFormLayout()
        form.addRow("位置名稱", self.name_edit)
        form.addRow("到位物件", self.object_box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setText("新增並繼續")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(text)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.name_edit.textChanged.connect(self._update_save)
        self.object_box.currentIndexChanged.connect(self._update_save)
        self._update_save()

    def _update_save(self):
        self.save_button.setEnabled(bool(self.name_edit.text().strip()) and
                                    (not self._has_classes or bool(self.object_box.currentData())))

    @classmethod
    def get_values(cls, parent, default_name, classes):
        dialog = cls(default_name, classes, parent)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        return dialog.name_edit.text().strip(), dialog.object_box.currentData() or "", accepted
