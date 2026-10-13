"""Native dependencies load only when needed; policy failures must not crash the editor."""
import importlib
import threading


class NativeDependencyError(ValueError):
    pass


class LazyOpenCV:
    def __init__(self):
        self._module = None
        self._failure = None
        self._lock = threading.RLock()

    def require(self):
        with self._lock:
            if self._failure:
                raise NativeDependencyError(self._failure)
            if self._module is None:
                try:
                    self._module = importlib.import_module("cv2")
                except (ImportError, OSError) as exc:
                    self._failure = (
                        "OpenCV 無法載入，攝影機、影片與工件定位暫不可用。\n"
                        "若詳細資訊顯示應用程式控制封鎖，需由電腦管理者確認 DLL 的信任與原則。\n"
                        f"詳細資訊：{exc}"
                    )
                    raise NativeDependencyError(self._failure) from exc
            return self._module

    def __getattr__(self, name):
        return getattr(self.require(), name)


cv2 = LazyOpenCV()
