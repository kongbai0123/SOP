"""Camera controls in native driver units; no guessed hardware ranges."""
import math

import cv2


PROPERTIES = {
    "fps": ("攝影機 FPS", cv2.CAP_PROP_FPS),
    "brightness": ("亮度", cv2.CAP_PROP_BRIGHTNESS),
    "contrast": ("對比", cv2.CAP_PROP_CONTRAST),
    "saturation": ("飽和度", cv2.CAP_PROP_SATURATION),
    "sharpness": ("銳利度", cv2.CAP_PROP_SHARPNESS),
    "gain": ("增益", cv2.CAP_PROP_GAIN),
    "exposure": ("曝光", cv2.CAP_PROP_EXPOSURE),
    "wb_temperature": ("白平衡色溫", cv2.CAP_PROP_WB_TEMPERATURE),
    "focus": ("對焦", cv2.CAP_PROP_FOCUS),
}
AUTO_PROPERTIES = {
    "exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "wb_temperature": cv2.CAP_PROP_AUTO_WB,
    "focus": cv2.CAP_PROP_AUTOFOCUS,
}


def read_controls(capture):
    values = {}
    for key, (_, prop) in PROPERTIES.items():
        try:
            value = capture.get(prop)
            values[key] = value if math.isfinite(value) and (key != "fps" or value > 0) else None
        except cv2.error:
            values[key] = None
    try:
        width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        values["resolution"] = (int(width), int(height)) if width > 0 and height > 0 else None
    except (cv2.error, ValueError, OverflowError):
        values["resolution"] = None
    return values


def apply_controls(capture, changes):
    """Apply only explicitly selected controls and report each driver's response."""
    messages = []
    changes = dict(changes)
    resolution = changes.pop("resolution", None)
    if resolution is not None:
        try:
            mode, size = resolution
            width, height = size
            if mode != "manual" or any(type(n) is not int or not 1 <= n <= 16384 for n in size):
                raise ValueError("請輸入 1～16384 的整數寬高")
            # DirectShow stages width and height separately; always send both.
            width_ok = capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            height_ok = capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            result = f"要求 {width} × {height}，驅動回報 {actual_w:g} × {actual_h:g}"
            if not width_ok or not height_ok:
                messages.append(f"解析度：未完成（驅動拒絕設定；{result}）")
            else:
                adjusted = "（設備採用不同解析度）" if (actual_w, actual_h) != (width, height) else ""
                messages.append(f"解析度：{result}{adjusted}")
        except (cv2.error, ValueError, TypeError) as exc:
            messages.append(f"解析度：未完成（{exc}）")
    for key, (mode, value) in changes.items():
        label, prop = PROPERTIES[key]
        try:
            if mode not in ("manual", "auto") or not math.isfinite(value):
                raise ValueError("參數無效")
            if key == "fps" and value <= 0:
                raise ValueError("FPS 必須大於 0")
            if key in AUTO_PROPERTIES:
                # These 0/1 mode values are specific to the DirectShow backend.
                if capture.get(cv2.CAP_PROP_BACKEND) != cv2.CAP_DSHOW:
                    raise ValueError("目前影像後端不支援此自動／手動切換")
                if not capture.set(AUTO_PROPERTIES[key], 1 if mode == "auto" else 0):
                    raise ValueError("攝影機不支援此模式，或驅動拒絕設定")
            elif mode == "auto":
                raise ValueError("此項目沒有自動模式")
            if mode == "auto":
                messages.append(f"{label}：驅動已接受自動模式")
                continue
            if not capture.set(prop, value):
                raise ValueError("攝影機不支援，或數值超出驅動允許範圍")
            actual = capture.get(prop)
            adjusted = "（設備採用不同數值）" if not math.isclose(value, actual, abs_tol=0.01) else ""
            messages.append(f"{label}：要求 {value:g}，驅動回報 {actual:g}{adjusted}")
        except (cv2.error, ValueError) as exc:
            messages.append(f"{label}：未完成（{exc}）")
    return messages
