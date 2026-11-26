"""清點 DirectShow 影像來源，不開啟擷取裝置。"""

from __future__ import annotations

from dataclasses import dataclass
import sys
import threading

from PySide6.QtCore import QObject, Qt, Signal, Slot


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str


@dataclass(frozen=True)
class CameraDiscoveryResult:
    devices: tuple[CameraDevice, ...] = ()
    error: str = ""


_com_import_lock = threading.Lock()


def enumerate_cameras() -> tuple[CameraDevice, ...]:
    """Use the same device order as OpenCV's Windows DirectShow backend."""
    if sys.platform != "win32":
        raise OSError("此平台不支援 DirectShow 攝影機搜尋")

    with _com_import_lock:
        already_imported = "comtypes" in sys.modules
        import comtypes

        # The first import initializes COM itself on the importing thread.
        if already_imported:
            comtypes.CoInitializeEx()
    try:
        from pygrabber.dshow_graph import SystemDeviceEnum
        from pygrabber.dshow_ids import DeviceCategories

        names = SystemDeviceEnum().get_available_filters(DeviceCategories.VideoInputDevice)
        return tuple(CameraDevice(index, name) for index, name in enumerate(names))
    finally:
        comtypes.CoUninitialize()


class CameraDiscovery(QObject):
    """Enumerate in the background and deliver one result on the GUI thread."""

    finished = Signal(object)
    _result_ready = Signal(object)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._running = False
        self._result_ready.connect(self._publish, Qt.ConnectionType.QueuedConnection)

    def request(self) -> bool:
        """Start a search from the GUI thread, coalescing concurrent requests."""
        if self._running:
            return False
        self._running = True
        try:
            threading.Thread(target=self._search, name="sop-camera-discovery", daemon=True).start()
        except RuntimeError as exc:
            self._result_ready.emit(CameraDiscoveryResult(error=str(exc)))
        return True

    def _search(self):
        try:
            result = CameraDiscoveryResult(enumerate_cameras())
        except Exception as exc:
            result = CameraDiscoveryResult(error=str(exc) or type(exc).__name__)
        try:
            self._result_ready.emit(result)
        except RuntimeError:
            # The parent window can be closed while DirectShow is enumerating.
            return

    @Slot(object)
    def _publish(self, result: CameraDiscoveryResult):
        self._running = False
        self.finished.emit(result)
