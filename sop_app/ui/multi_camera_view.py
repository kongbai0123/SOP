"""Preview tiles for cameras that do not participate in SOP decisions."""
from __future__ import annotations

from math import ceil, isfinite
from typing import Iterable

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QSplitter, QSplitterHandle, QVBoxLayout, QWidget

from .compact_layout import StatusText
from .video_widget import VideoWidget


MIN_PREVIEW_WIDTH = 110


class _PreviewHandle(QSplitterHandle):
    def __init__(self, orientation, splitter):
        super().__init__(orientation, splitter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("調整主畫面與副畫面比例")
        self.setToolTip("拖曳或用方向鍵調整畫面比例")

    def keyPressEvent(self, event):
        horizontal = self.orientation() == Qt.Orientation.Horizontal
        start, end = (Qt.Key.Key_Left, Qt.Key.Key_Right) if horizontal else (Qt.Key.Key_Up, Qt.Key.Key_Down)
        if event.key() in (start, end):
            position = self.x() if horizontal else self.y()
            self.moveSplitter(position + (-16 if event.key() == start else 16))
            event.accept()
        else:
            super().keyPressEvent(event)


class _PreviewSplitter(QSplitter):
    resized = Signal()

    def createHandle(self):
        return _PreviewHandle(self.orientation(), self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()


class CameraPreviewPane(QWidget):
    """Let operators resize the primary and secondary camera views."""

    split_ratio_changed = Signal(float)

    def __init__(self, primary: VideoWidget):
        super().__init__()
        self._primary = primary
        self._ratio = .5
        self.auxiliary = AuxiliaryCameraGrid()
        self._splitter = _PreviewSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.setObjectName("cameraPreviewSplitter")
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(8)
        self._splitter.setStyleSheet("""
            QSplitter#cameraPreviewSplitter::handle { background: palette(mid); border-radius: 2px; margin: 2px; }
            QSplitter#cameraPreviewSplitter::handle:hover { background: palette(highlight); }
        """)
        self._splitter.addWidget(primary)
        self._splitter.addWidget(self.auxiliary)
        self._splitter.splitterMoved.connect(self._on_splitter_moved)
        self._splitter.resized.connect(self._apply_split_ratio)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._splitter)

    @property
    def minimum_preview_height(self):
        primary = max(self._primary.minimumHeight(), self._primary.minimumSizeHint().height())
        if self.auxiliary.isHidden():
            return primary
        auxiliary = max(self.auxiliary.minimumHeight(), self.auxiliary.minimumSizeHint().height())
        if self._splitter.orientation() == Qt.Orientation.Vertical:
            return primary + auxiliary + self._splitter.handle(1).sizeHint().height()
        return max(primary, auxiliary)

    def set_split_layout(self, orientation: str, ratio: float):
        direction = Qt.Orientation.Vertical if orientation == "vertical" else Qt.Orientation.Horizontal
        ratio = min(1., max(0., ratio)) if isfinite(ratio) else .5
        if direction == self._splitter.orientation() and ratio == self._ratio:
            return
        self._ratio = ratio
        self._splitter.setOrientation(direction)
        self._apply_split_ratio()

    def set_devices(self, devices: Iterable[tuple[int, str]]):
        was_hidden = self.auxiliary.isHidden()
        self.auxiliary.set_devices(devices)
        if was_hidden and not self.auxiliary.isHidden():
            self._apply_split_ratio()

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_split_ratio()

    def _apply_split_ratio(self):
        if self.auxiliary.isHidden():
            return
        horizontal = self._splitter.orientation() == Qt.Orientation.Horizontal
        handle_size = self._splitter.handle(1).sizeHint()
        length = (self._splitter.width() - handle_size.width() if horizontal
                  else self._splitter.height() - handle_size.height())
        if length > 0:
            primary = round(length * self._ratio)
            self._splitter.setSizes([primary, length - primary])

    def _on_splitter_moved(self, _position, _index):
        primary, auxiliary = self._splitter.sizes()
        if primary and auxiliary:
            ratio = primary / (primary + auxiliary)
            if ratio != self._ratio:
                self._ratio = ratio
                self.split_ratio_changed.emit(ratio)


class _CameraTile(QWidget):
    def __init__(self, index: int, name: str, parent: QWidget):
        super().__init__(parent)
        self.title = StatusText()
        self.video = VideoWidget()
        self.video.setMinimumSize(MIN_PREVIEW_WIDTH, 64)
        self.video.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.title)
        layout.addWidget(self.video, 1)
        self.set_name(index, name)
        self.video.clear_frame("等待影像")

    def set_name(self, index: int, name: str):
        source = f"攝影機 {index}"
        title = f"{source} · {name}" if name and name != source else source
        self.title.setText(title)
        self.title.setAccessibleName(title)
        self.video.setAccessibleName(f"{title}預覽")


class AuxiliaryCameraGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAccessibleName("其他攝影機預覽")
        self._tiles: dict[int, _CameraTile] = {}
        self._columns = 0
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(6)
        self.hide()

    def set_devices(self, devices: Iterable[tuple[int, str]]):
        names = dict(devices)
        for index in self._tiles.keys() - names.keys():
            tile = self._tiles.pop(index)
            self._grid.removeWidget(tile)
            tile.hide()
            tile.deleteLater()
        ordered = {}
        for index, name in names.items():
            tile = self._tiles.get(index)
            if tile is None:
                tile = _CameraTile(index, name, self)
            else:
                tile.set_name(index, name)
            ordered[index] = tile
        if list(self._tiles) != list(ordered):
            self._columns = 0
        self._tiles = ordered
        self.setVisible(bool(self._tiles))
        self._arrange_tiles()

    def set_frame(self, index: int, frame: np.ndarray):
        tile = self._tiles.get(index)
        if tile is not None:
            if tile.video.toolTip():
                tile.video.setToolTip("")
            tile.video.set_frame(frame)

    def set_status(self, index: int, text: str):
        tile = self._tiles.get(index)
        if tile is not None and text:
            tile.video.setToolTip(text)
            tile.video.clear_frame(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._arrange_tiles()

    def _arrange_tiles(self):
        count = len(self._tiles)
        if not count:
            return

        # Leave as much useful space as possible for a whole, uncut preview.
        label_height = next(iter(self._tiles.values())).title.sizeHint().height() + 2
        gap = self._grid.spacing()

        def preview_width(columns):
            rows = ceil(count / columns)
            width = (self.width() - (columns - 1) * gap) / columns
            height = (self.height() - (rows - 1) * gap) / rows - label_height
            return min(width, height * 16 / 9)

        limit = min(count, max(1, (self.width() + gap) // (MIN_PREVIEW_WIDTH + gap)))
        columns = max(range(1, limit + 1), key=preview_width)
        if columns == self._columns:
            return
        self._columns = columns
        while self._grid.count():
            self._grid.takeAt(0)
        for column in range(self._grid.columnCount()):
            self._grid.setColumnStretch(column, int(column < columns))
        rows = ceil(count / columns)
        for row in range(self._grid.rowCount()):
            self._grid.setRowStretch(row, int(row < rows))
        for position, tile in enumerate(self._tiles.values()):
            row, column = divmod(position, columns)
            self._grid.addWidget(tile, row, column)
            self._grid.setColumnStretch(column, 1)
            self._grid.setRowStretch(row, 1)
            tile.show()
