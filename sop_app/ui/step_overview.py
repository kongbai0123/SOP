"""One wrapped SOP list, packed into two columns without splitting entries."""
from __future__ import annotations

from dataclasses import dataclass
import math

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (QColor, QFont, QPainter, QPainterPath, QPen,
                          QTextLayout, QTextOption)
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QScrollBar,
                               QSizePolicy, QStyle, QToolTip, QVBoxLayout, QWidget)

from ..engine import EngineSnapshot, Phase, StepStatus


@dataclass(frozen=True)
class _Card:
    index: int
    name: str
    detail: str
    kind: str
    active: bool

    @property
    def description(self):
        return f"{self.index + 1}. {self.name}；{self.detail}"


@dataclass
class _Entry:
    index: int
    layout: QTextLayout
    line_count: int
    natural_height: float
    oversized: bool


@dataclass(frozen=True)
class _Placement:
    page: int
    column: int
    row: int
    top: float
    height: float


class _ColumnList(QWidget):
    GAP = 0
    COLUMN_GAP = 12
    MARGIN = 4
    TOP_PADDING = 4
    BOTTOM_PADDING = 4
    MARKER_WIDTH = 18
    TEXT_INSET = 25
    RIGHT_PADDING = 5

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.cards = ()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("工序清單，左欄由上到下，再接右欄")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.owner._resize_cards()

    def entry_rect(self, index):
        placement = self.owner._placements.get(index)
        if placement is None or placement.page != self.owner.page_index:
            return QRectF()
        width = self.owner._column_width
        return QRectF(self.MARGIN + placement.column * (width + self.COLUMN_GAP),
                      self.MARGIN + placement.top, width, placement.height)

    def card_rect(self, index):
        return self.entry_rect(index)

    def paintEvent(self, event):
        painter = QPainter(self)
        colors = self.owner._colors
        painter.fillRect(self.rect(), QColor(colors["panel"]))
        painter.setPen(QPen(QColor(colors["border"]), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        middle = self.width() // 2
        painter.drawLine(middle, self.MARGIN, middle, self.height() - self.MARGIN - 1)
        for card in self.cards:
            rect = self.entry_rect(card.index)
            if rect.isEmpty():
                continue
            entry = self.owner._entries[card.index]
            color = QColor(colors.get(card.kind, colors["pending"]))
            painter.save()
            painter.setClipRect(rect)
            if card.active:
                tint = QColor(colors["active"])
                tint.setAlpha(28)
                painter.fillRect(rect, tint)
            marker = QRectF(rect.left() + 3., rect.top() + self.TOP_PADDING,
                            self.MARKER_WIDTH, self.owner._line_height)
            self._draw_marker(painter, marker, card, color)
            painter.setPen(color if card.kind != "pending" else QColor(colors["fg"]))
            scrollbar = self.owner.entry_scrollbar(card.index)
            right = self.owner._scrollbar_extent if entry.oversized else 0
            text_clip = rect.adjusted(self.TEXT_INSET, 0, -right - self.RIGHT_PADDING, 0)
            painter.setClipRect(text_clip, Qt.ClipOperation.IntersectClip)
            offset = scrollbar.value() if scrollbar is not None else 0
            # Measurement and painting share this exact layout and fixed font.
            entry.layout.draw(painter, QPointF(rect.left() + self.TEXT_INSET,
                                              rect.top() + self.TOP_PADDING - offset))
            painter.restore()

    @staticmethod
    def _draw_marker(painter, rect, entry, color):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(color, 1.6, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        x, y = rect.center().x(), rect.center().y()
        if entry.kind == "ng":
            painter.drawLine(QPointF(x, y - 5.), QPointF(x, y + 1.))
            painter.drawEllipse(QRectF(x - .8, y + 4., 1.6, 1.6))
        elif entry.active:
            path = QPainterPath()
            path.moveTo(x - 4, y - 5)
            path.lineTo(x + 5, y)
            path.lineTo(x - 4, y + 5)
            path.closeSubpath()
            painter.fillPath(path, color)
        elif entry.kind == "ok":
            path = QPainterPath()
            path.moveTo(x - 5, y)
            path.lineTo(x - 1, y + 4)
            path.lineTo(x + 6, y - 5)
            painter.drawPath(path)
        elif entry.kind == "warn":
            path = QPainterPath()
            path.moveTo(x - 6, y)
            path.lineTo(x + 5, y)
            path.moveTo(x + 1, y - 4)
            path.lineTo(x + 5, y)
            path.lineTo(x + 1, y + 4)
            painter.drawPath(path)
        else:
            painter.drawEllipse(QRectF(x - 4, y - 4, 8, 8))
        painter.restore()

    def wheelEvent(self, event):
        for index, scrollbar in self.owner._visible_scrollbars.items():
            if self.entry_rect(index).contains(event.position()):
                delta = event.pixelDelta().y()
                if not delta:
                    delta = event.angleDelta().y() / 120. * scrollbar.singleStep() * 3
                scrollbar.setValue(scrollbar.value() - round(delta))
                event.accept()
                return
        event.ignore()

    def event(self, event):
        if event.type() == QEvent.Type.ToolTip:
            for card in self.cards:
                if self.entry_rect(card.index).contains(event.pos()):
                    QToolTip.showText(event.globalPos(), f"{card.index + 1}. {card.name}\n{card.detail}", self)
                    return True
            QToolTip.hideText()
            return True
        return super().event(event)


class StepOverview(QWidget):
    """Prelayout names, then repaint runtime status without repacking on frames.

    Each entry stays in one column. Oversized entries have their own viewport
    and local scrollbar; neither the page nor the whole SOP has a scrollbar.
    All public indices/pages are zero based. An invalid page/index yields an
    empty tuple/rectangle, zero metrics or None from page/scrollbar lookups.
    """
    DEFAULT_COLORS = {"fg": "#202124", "muted": "#5f6368", "panel": "#ffffff",
                      "border": "#dadce0", "pending": "#78909c", "active": "#1565c0",
                      "ok": "#2e7d32", "warn": "#ef6c00", "ng": "#c62828"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._names = []
        self._snapshot = None
        self._running = False
        self._follow = True
        self._new_run_pending = False
        self._page = 0
        self._entries = []
        self._placements = {}
        self._pages = []
        self._column_width = 1.
        self._available_height = 1.
        self._card_height = 26  # legacy probe: baseline single-line height only
        self._line_height = 16.
        self._scrollbar_extent = 14
        self._colors = dict(self.DEFAULT_COLORS)
        self._built = False
        self._resizing = False
        self._layout_signature = None
        self._layout_generation = 0
        self._scrollbar_page = None
        self._scrollbar_pool = []
        self._visible_scrollbars = {}
        self._scroll_offsets = {}
        self._grid = _ColumnList(self)
        self.previous_button = QPushButton("上一頁")
        self.next_button = QPushButton("下一頁")
        self.follow_button = QPushButton("回目前工序")
        self.page_label = QLabel("0 / 0")
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_label.setTextFormat(Qt.TextFormat.PlainText)
        self.previous_button.clicked.connect(lambda _checked=False: self.show_page(self._page - 1))
        self.next_button.clicked.connect(lambda _checked=False: self.show_page(self._page + 1))
        self.follow_button.clicked.connect(self.follow_current)
        self._footer = QWidget()
        footer = QHBoxLayout(self._footer)
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(6)
        footer.addWidget(self.previous_button)
        footer.addWidget(self.page_label, 1)
        footer.addWidget(self.next_button)
        footer.addWidget(self.follow_button)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        outer.addWidget(self._grid, 1)
        outer.addWidget(self._footer)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("雙欄工序清單")
        self._built = True
        self._update_font_geometry()

    @property
    def rows_per_column(self):
        counts = [0, 0]
        for index in self.visible_indices:
            counts[self._placements[index].column] += 1
        return max(counts)

    @property
    def capacity(self):
        return len(self.visible_indices)

    @property
    def page_index(self):
        return self._page

    @property
    def page_count(self):
        return len(self._pages)

    @property
    def visible_indices(self):
        return self.page_indices(self._page)

    @property
    def visible_slots(self):
        return {index: (self._placements[index].row, self._placements[index].column)
                for index in self.visible_indices}

    @property
    def is_following(self):
        return self._follow

    def page_indices(self, page):
        return self._pages[page] if type(page) is int and 0 <= page < len(self._pages) else ()

    def page_for_index(self, index):
        placement = self._placements.get(index)
        return placement.page if placement is not None else None

    def entry_line_count(self, index):
        return self._entries[index].line_count if type(index) is int and 0 <= index < len(self._entries) else 0

    def entry_natural_height(self, index):
        return self._entries[index].natural_height if type(index) is int and 0 <= index < len(self._entries) else 0.

    def entry_is_oversized(self, index):
        return bool(type(index) is int and 0 <= index < len(self._entries) and self._entries[index].oversized)

    def entry_scrollbar(self, index):
        return self._visible_scrollbars.get(index)

    def entry_rect(self, index):
        """The current-page entry viewport, in StepOverview coordinates."""
        rect = self._grid.entry_rect(index)
        if not rect.isEmpty():
            rect.translate(self._grid.pos())
        return rect

    def card_rect(self, index):
        return self.entry_rect(index)

    def sizeHint(self):
        return QSize(480, self._card_height * 8 + 2 * self._grid.MARGIN +
                     self._footer.sizeHint().height() + 6)

    def set_steps(self, names: list[str]):
        self._names = [str(name) for name in names]
        self._snapshot = None
        self._follow = True
        self._page = 0
        self._scroll_offsets.clear()
        self._scrollbar_page = None
        self._resize_cards()
        self._render()

    def set_snapshot(self, snapshot: EngineSnapshot | None):
        previous = self._snapshot
        new_cycle = (previous is not None and snapshot is not None and
                     (previous.cycle != snapshot.cycle or
                      (previous.phase == Phase.CYCLE_DONE and snapshot.phase == Phase.RUNNING)))
        if new_cycle:
            self._follow = True
        self._snapshot = snapshot
        if self._follow and snapshot is not None:
            if snapshot.phase == Phase.CYCLE_DONE:
                self._page = max(0, self.page_count - 1)
            elif snapshot.phase == Phase.RUNNING and self._running:
                if self._new_run_pending or new_cycle:
                    self._locate_current()
                else:
                    target = self.page_for_index(self._current_index())
                    if target is not None and target > self._page and self._completed_pages_before(target):
                        self._page = target
        # The first runtime sample consumes the new-run exception even when
        # the operator is browsing manually. Returning to follow later must
        # still honor the normal completed-page boundary check.
        if snapshot is not None and snapshot.phase == Phase.RUNNING and self._running:
            self._new_run_pending = False
        self._render()

    def set_running(self, running: bool):
        if running and not self._running:
            self._follow = True
            self._new_run_pending = True
            self._locate_current()
        self._running = bool(running)
        self._render()

    def set_colors(self, colors: dict):
        values = {**self.DEFAULT_COLORS, **colors}
        if values != self._colors:
            self._colors = values
            self._grid.update()

    def show_page(self, index):
        self._follow = False
        self._page = min(max(0, int(index)), max(0, self.page_count - 1))
        self._render()

    def follow_current(self):
        self._follow = True
        self._locate_current()
        self._render()

    def _current_index(self):
        if not self._names or self._snapshot is None:
            return 0
        if self._snapshot.phase == Phase.CYCLE_DONE:
            return len(self._names) - 1
        if self._snapshot.phase == Phase.RUNNING:
            return min(max(0, self._snapshot.current_index), len(self._names) - 1)
        return 0

    def _locate_current(self):
        target = self.page_for_index(self._current_index())
        self._page = target if target is not None else 0

    def _completed_pages_before(self, target):
        snapshot = self._snapshot
        for page in range(self._page, target):
            indices = self.page_indices(page)
            if not indices:
                return False
            last = indices[-1]
            if last >= len(snapshot.steps) or snapshot.steps[last].status not in (StepStatus.DONE, StepStatus.SKIPPED):
                return False
        return True

    def _update_font_geometry(self):
        self._card_height = max(26, self.fontMetrics().height() + 10)
        self._grid.setMinimumHeight(self._card_height + 2 * self._grid.MARGIN)
        self.setMinimumHeight(self._card_height + 2 * self._grid.MARGIN +
                              self._footer.sizeHint().height() + 6)
        self._resize_cards()

    def changeEvent(self, event):
        super().changeEvent(event)
        if getattr(self, "_built", False) and event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._update_font_geometry()

    def _resize_cards(self):
        if not self._built or self._resizing:
            return
        self._resizing = True
        try:
            font = QFont(self._grid.font())
            font.setWeight(QFont.Weight.Normal)
            extent = max(1, self._grid.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent))
            signature = (tuple(self._names), font.toString(), self._grid.width(), self._grid.height(), extent)
            if signature != self._layout_signature:
                anchor = self.visible_indices[0] if self.visible_indices else 0
                self._rebuild_layout(font, extent)
                self._layout_signature = signature
                self._layout_generation += 1
                if (self._follow and self._snapshot is not None and
                        self._snapshot.phase in (Phase.RUNNING, Phase.CYCLE_DONE)):
                    self._locate_current()
                else:
                    self._page = self.page_for_index(anchor) or 0
            self._render()
        finally:
            self._resizing = False

    def _text_layout(self, index, width, font):
        # U+2028 is a forced line break within QTextLayout's single paragraph.
        # Keep original names unchanged for tooltips/accessibility.
        name = self._names[index].replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\u2028")
        text = f"{index + 1}. {name}"
        layout = QTextLayout(text, font)
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout.setTextOption(option)
        layout.setCacheEnabled(True)
        layout.beginLayout()
        top = 0.
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(max(1., width))
            line.setPosition(QPointF(0., top))
            top += line.height()
        layout.endLayout()
        height = max(float(self._card_height), math.ceil(top) + self._grid.TOP_PADDING + self._grid.BOTTOM_PADDING)
        return layout, height

    def _rebuild_layout(self, font, scrollbar_extent):
        """Measure and pack only when names, font or available geometry change."""
        self._scrollbar_extent = scrollbar_extent
        self._column_width = max(1., (self._grid.width() - 2 * self._grid.MARGIN - self._grid.COLUMN_GAP) / 2.)
        self._available_height = max(1., self._grid.height() - 2 * self._grid.MARGIN)
        self._line_height = float(self.fontMetrics().height())
        text_width = max(1., self._column_width - self._grid.TEXT_INSET - self._grid.RIGHT_PADDING)
        entries, pages, placements = [], [], {}
        page, column, row, top = 0, 0, 0, 0.
        current = []
        for index in range(len(self._names)):
            layout, natural = self._text_layout(index, text_width, font)
            oversized = natural > self._available_height
            if oversized:
                layout, natural = self._text_layout(index, max(1., text_width - scrollbar_extent), font)
            entries.append(_Entry(index, layout, layout.lineCount(), natural, oversized))
            height = self._available_height if oversized else natural
            if top > 0 and top + height > self._available_height + .001:
                if column == 0:
                    column, row, top = 1, 0, 0.
                else:
                    pages.append(tuple(current))
                    current = []
                    page, column, row, top = page + 1, 0, 0, 0.
            placements[index] = _Placement(page, column, row, top, height)
            current.append(index)
            row += 1
            top += height
        if current:
            pages.append(tuple(current))
        self._entries, self._pages, self._placements = entries, pages, placements
        self._scrollbar_page = None

    def _scroll_changed(self, scrollbar, value):
        index = scrollbar.property("entry_index")
        if type(index) is int:
            self._scroll_offsets[index] = value
            self._grid.update(self._grid.entry_rect(index).toAlignedRect())

    def _sync_scrollbars(self):
        key = (self._layout_generation, self._page)
        if self._scrollbar_page == key:
            return
        self._scrollbar_page = key
        oversized = [index for index in self.visible_indices if self.entry_is_oversized(index)]
        self._visible_scrollbars = {}
        while len(self._scrollbar_pool) < len(oversized):
            scrollbar = QScrollBar(Qt.Orientation.Vertical, self._grid)
            scrollbar.valueChanged.connect(lambda value, bar=scrollbar: self._scroll_changed(bar, value))
            self._scrollbar_pool.append(scrollbar)
        for slot, scrollbar in enumerate(self._scrollbar_pool):
            if slot >= len(oversized):
                scrollbar.hide()
                scrollbar.setProperty("entry_index", None)
                continue
            index = oversized[slot]
            rect = self._grid.entry_rect(index)
            blocked = scrollbar.blockSignals(True)
            try:
                scrollbar.setProperty("entry_index", index)
                scrollbar.setGeometry(math.floor(rect.right() - self._scrollbar_extent),
                                      math.ceil(rect.top()) + 1, self._scrollbar_extent,
                                      max(1, math.floor(rect.height()) - 2))
                scrollbar.setRange(0, max(0, math.ceil(self.entry_natural_height(index) - rect.height())))
                scrollbar.setPageStep(max(1, math.floor(rect.height())))
                scrollbar.setSingleStep(max(1, round(self._line_height)))
                scrollbar.setValue(min(self._scroll_offsets.get(index, 0), scrollbar.maximum()))
                scrollbar.setAccessibleName(f"工序 {index + 1} 文字捲動")
                scrollbar.setAccessibleDescription(self._names[index])
                scrollbar.setToolTip("此工序文字超過一欄高度，可在本項內捲動閱讀全文。")
            finally:
                scrollbar.blockSignals(blocked)
            self._visible_scrollbars[index] = scrollbar
            scrollbar.show()
            scrollbar.raise_()

    def _card(self, index):
        runtime = (self._snapshot.steps[index] if self._snapshot is not None and
                   index < len(self._snapshot.steps) else None)
        status = runtime.status if runtime is not None else StepStatus.PENDING
        active = (self._snapshot is not None and self._snapshot.phase == Phase.RUNNING and
                  index == self._snapshot.current_index)
        labels = {StepStatus.PENDING: "等待", StepStatus.ACTIVE: "執行中",
                  StepStatus.DONE: "已完成", StepStatus.SKIPPED: "已跳過"}
        detail = "人工完成" if runtime and runtime.manual and status == StepStatus.DONE else labels.get(status, "等待")
        stopped = status == StepStatus.ACTIVE and self._snapshot is not None and self._snapshot.phase == Phase.IDLE
        if stopped:
            detail = "已停止"
        if runtime is not None and runtime.started_at is not None:
            end = runtime.ended_at if runtime.ended_at is not None else self._snapshot.now
            detail += f" · {max(0., end - runtime.started_at):.1f} 秒"
        alarm = runtime is not None and runtime.has_alarm
        if alarm:
            detail += " · 有警報"
        kind = ("ng" if alarm else "pending" if stopped else
                {StepStatus.PENDING: "pending", StepStatus.ACTIVE: "active",
                 StepStatus.DONE: "ok", StepStatus.SKIPPED: "warn"}.get(status, "pending"))
        return _Card(index, self._names[index], detail, kind, active)

    def _render(self):
        self._page = min(max(0, self._page), max(0, self.page_count - 1))
        cards = tuple(self._card(index) for index in self.visible_indices)
        changed = cards != self._grid.cards
        self._grid.cards = cards
        self._sync_scrollbars()
        if changed:
            self._grid.update()
        label = f"{self._page + 1 if self.page_count else 0} / {self.page_count}"
        if self.page_label.text() != label:
            self.page_label.setText(label)
        self.previous_button.setEnabled(self._page > 0)
        self.next_button.setEnabled(self._page + 1 < self.page_count)
        self.follow_button.setVisible(not self._follow and bool(self._names))
        self.follow_button.setToolTip("返回目前工序並恢復自動跟隨" if self._snapshot else "返回第一頁並恢復自動跟隨")
        description = f"第 {label} 頁；" + "；".join(card.description for card in cards)
        if self.accessibleDescription() != description:
            self.setAccessibleDescription(description)
            self._grid.setAccessibleDescription(description)
