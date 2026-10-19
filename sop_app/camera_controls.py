"""Camera controls in native driver units, final readback and JSON preferences."""
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
FORMAT_KEYS = {"resolution", "fps", "pixel_format"}


def control_label(key):
    return {"resolution": "解析度", "pixel_format": "像素格式"}.get(key, PROPERTIES.get(key, (key,))[0])


def _read(capture, prop):
    try:
        value = float(capture.get(prop))
        return value if math.isfinite(value) else None
    except (cv2.error, ValueError, TypeError, OverflowError):
        return None


def fourcc_text(value):
    """Decode FOURCC without inventing a format for unavailable values."""
    if type(value) not in (int, float) or not 0 < value <= 0xFFFFFFFF or not math.isfinite(value) or int(value) != value:
        return None
    chars = [(int(value) >> (8 * index)) & 0xFF for index in range(4)]
    return "".join(map(chr, chars)) if all(32 <= char <= 126 for char in chars) else None


def _pixel_format(value):
    if isinstance(value, str) and len(value) == 4 and all(32 <= ord(char) <= 126 for char in value):
        return value.upper()
    raise ValueError("像素格式必須是 4 字元代碼，例如 MJPG 或 YUY2")


def _size(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
            type(number) is not int or not 1 <= number <= 16384 for number in value):
        raise ValueError("請輸入 1～16384 的整數寬高")
    return tuple(value)


def _number(value, *, positive=False):
    try:
        number = float(value) if type(value) in (int, float) else float("nan")
    except OverflowError:
        number = float("nan")
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError("參數無效" if not positive else "FPS 必須大於 0")
    return number


def read_controls(capture):
    values = {key: _read(capture, prop) for key, (_, prop) in PROPERTIES.items()}
    if values["fps"] is not None and values["fps"] <= 0:
        values["fps"] = None
    width = _read(capture, cv2.CAP_PROP_FRAME_WIDTH)
    height = _read(capture, cv2.CAP_PROP_FRAME_HEIGHT)
    values["resolution"] = ((int(width), int(height)) if width and height and width >= 1 and height >= 1
                            and int(width) == width and int(height) == height else None)
    values["pixel_format"] = fourcc_text(_read(capture, cv2.CAP_PROP_FOURCC))
    modes = {}
    if _read(capture, cv2.CAP_PROP_BACKEND) == cv2.CAP_DSHOW:
        for key, prop in AUTO_PROPERTIES.items():
            raw = _read(capture, prop)
            # DirectShow autofocus returns CameraControl flags (1/2), WB is
            # normalized to 1/0, and auto-exposure readback may be unavailable.
            manual = 2 if key == "focus" else 0
            modes[key] = "auto" if raw == 1 else "manual" if raw == manual else None
    values["control_modes"] = {key: modes.get(key) for key in AUTO_PROPERTIES}
    return values


def apply_controls(capture, changes):
    """Apply selected settings, then report the final negotiated mode.

    DirectShow stages width/height. FPS can rebuild the graph with a default
    format, so set FOURCC last. A True setter alone does not prove 60 FPS.
    """
    changes = dict(changes)
    requested, accepted, failures = {}, set(), {}
    for key, setting in changes.items():
        try:
            mode, value = setting
            if mode not in ("manual", "auto"):
                raise ValueError("參數模式無效")
            if key == "resolution":
                value = _size(value)
            elif key == "pixel_format":
                value = _pixel_format(value)
            elif key in PROPERTIES:
                value = _number(value, positive=key == "fps")
            else:
                raise ValueError("不支援的參數")
            if mode == "auto" and key not in AUTO_PROPERTIES:
                raise ValueError("此項目沒有自動模式")
            requested[key] = (mode, value)
        except (ValueError, TypeError) as exc:
            failures[key] = str(exc)

    format_selected = bool(FORMAT_KEYS.intersection(requested))
    before = read_controls(capture) if format_selected else {}
    targets = {key: requested[key][1] if key in requested else before.get(key) for key in FORMAT_KEYS}
    for key in ("resolution", "fps", "pixel_format"):
        if not format_selected or targets[key] is None:
            continue
        try:
            value = targets[key]
            if key == "resolution":
                width_ok = capture.set(cv2.CAP_PROP_FRAME_WIDTH, value[0])
                height_ok = capture.set(cv2.CAP_PROP_FRAME_HEIGHT, value[1])
                ok = width_ok and height_ok
            elif key == "fps":
                ok = capture.set(cv2.CAP_PROP_FPS, value)
            else:
                ok = capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*value))
            if not ok:
                raise ValueError("驅動拒絕設定")
            if key in requested:
                accepted.add(key)
        except (cv2.error, ValueError) as exc:
            if key in requested:
                failures[key] = str(exc)

    for key, (mode, value) in requested.items():
        if key in FORMAT_KEYS:
            continue
        try:
            if key in AUTO_PROPERTIES:
                if _read(capture, cv2.CAP_PROP_BACKEND) != cv2.CAP_DSHOW:
                    raise ValueError("目前影像後端不支援此自動／手動切換")
                if not capture.set(AUTO_PROPERTIES[key], 1 if mode == "auto" else 0):
                    raise ValueError("攝影機不支援此模式，或驅動拒絕設定")
            if mode == "manual" and not capture.set(PROPERTIES[key][1], value):
                raise ValueError("攝影機不支援，或數值超出驅動允許範圍")
            accepted.add(key)
        except (cv2.error, ValueError) as exc:
            failures[key] = str(exc)

    actual = read_controls(capture)
    # Format controls are coupled. One bounded repair handles drivers that reset
    # FOURCC during FPS negotiation or reset FPS while selecting FOURCC. Never
    # probe a guessed format, and never loop until a device happens to agree.
    if format_selected and targets["pixel_format"] is not None and (
            actual.get("pixel_format") != targets["pixel_format"] or
            (targets["fps"] is not None and (actual.get("fps") is None or
             not math.isclose(actual["fps"], targets["fps"], abs_tol=.01)))):
        repaired = set()
        for key, prop in (("fps", cv2.CAP_PROP_FPS), ("pixel_format", cv2.CAP_PROP_FOURCC)):
            if targets[key] is None:
                continue
            value = targets[key] if key == "fps" else cv2.VideoWriter_fourcc(*targets[key])
            try:
                ok = capture.set(prop, value)
                if key in requested and not ok:
                    failures[key] = "驅動拒絕設定"
                elif key in requested and ok:
                    repaired.add(key)
            except cv2.error as exc:
                if key in requested:
                    failures[key] = str(exc)
        actual = read_controls(capture)
        for key in repaired:
            current = actual.get(key)
            matches = (current is not None and math.isclose(current, targets[key], abs_tol=.01)
                       if key == "fps" else current == targets[key])
            if matches:
                failures.pop(key, None)
                accepted.add(key)
    messages = []
    keys = [key for key in ("resolution", "pixel_format", "fps") if key in changes]
    keys.extend(key for key in changes if key not in FORMAT_KEYS)
    for key in keys:
        label = control_label(key)
        if key in failures:
            messages.append(f"{label}：未完成（{failures[key]}）")
            continue
        if key not in accepted:
            continue
        mode, target = requested[key]
        actual_mode = actual["control_modes"].get(key)
        if actual_mode is not None and actual_mode != mode:
            messages.append(f"{label}：未完成（要求{'自動' if mode == 'auto' else '手動'}模式，驅動回報{'自動' if actual_mode == 'auto' else '手動'}）")
        elif mode == "auto":
            suffix = "；模式無法讀取，未宣稱已驗證" if actual_mode is None else "；驅動回報自動"
            messages.append(f"{label}：驅動已接受自動模式{suffix}")
        elif actual.get(key) is None:
            messages.append(f"{label}：未完成（驅動接受要求，但無法讀取實際設定）")
        elif key == "resolution":
            width, height = actual[key]
            adjusted = "（設備採用不同解析度）" if tuple(actual[key]) != target else ""
            messages.append(f"{label}：要求 {target[0]} × {target[1]}，驅動回報 {width} × {height}{adjusted}")
        elif key == "pixel_format":
            adjusted = "（設備採用不同格式）" if actual[key] != target else ""
            messages.append(f"{label}：要求 {target}，驅動回報 {actual[key]}{adjusted}")
        else:
            adjusted = "（設備採用不同數值）" if not math.isclose(target, actual[key], abs_tol=.01) else ""
            messages.append(f"{label}：要求 {target:g}，驅動回報 {actual[key]:g}{adjusted}")
    for key in ("resolution", "pixel_format", "fps"):
        if not format_selected or key in changes or targets[key] is None:
            continue
        expected, current = targets[key], actual.get(key)
        matched = (current is not None and math.isclose(current, expected, abs_tol=.01)
                   if key == "fps" else current == expected)
        if not matched:
            messages.append(f"{control_label(key)}：未完成（維持原設定 {expected} 失敗；驅動最終回報 {current}）")
    return messages


def saved_camera_changes(preferences):
    """Normalize canonical/legacy JSON preferences to explicit changes."""
    if not isinstance(preferences, dict):
        return {}
    changes = {}
    size = preferences.get("resolution", (preferences.get("width"), preferences.get("height")))
    try:
        changes["resolution"] = ("manual", _size(size))
    except (ValueError, TypeError):
        pass
    try:
        changes["fps"] = ("manual", _number(preferences.get("fps"), positive=True))
    except (ValueError, TypeError):
        pass
    pixel = preferences.get("pixel_format", preferences.get("fourcc"))
    if isinstance(pixel, (float, int)):
        pixel = fourcc_text(pixel)
    try:
        changes["pixel_format"] = ("manual", _pixel_format(pixel))
    except ValueError:
        pass
    controls = preferences.get("controls", {})
    if not isinstance(controls, dict):
        return changes
    for original, setting in controls.items():
        key = "wb_temperature" if original == "white_balance" else original
        if key not in PROPERTIES or key == "fps":
            continue
        if isinstance(setting, dict):
            mode = setting.get("mode", "auto" if setting.get("auto") is True else "manual")
            value = setting.get("value")
        else:
            mode, value = "manual", setting
        try:
            if mode == "auto" and key in AUTO_PROPERTIES:
                changes[key] = ("auto", 0.)  # current sample is not a preference
            elif mode == "manual":
                changes[key] = ("manual", _number(value))
        except (ValueError, TypeError):
            pass
    return changes


def _save_setting(preferences, key, mode, value):
    if key == "resolution":
        preferences[key] = list(value)
    elif key in ("pixel_format", "fps"):
        preferences[key] = value
    else:
        preferences.setdefault("controls", {})[key] = ({"mode": "auto"} if mode == "auto" else
                                                       {"mode": "manual", "value": value})


def accepted_camera_preferences(previous, changes, values, messages):
    """Merge acknowledged settings using final readback; rejected items stay.

    Auto settings retain only the acknowledged mode, including modes for which
    DirectShow accepts the setting but cannot supply a mode readback.
    """
    result = {}
    for key, (mode, value) in saved_camera_changes(previous).items():
        _save_setting(result, key, mode, value)
    if not isinstance(changes, dict) or not isinstance(values, dict):
        return result
    modes = values.get("control_modes")
    modes = modes if isinstance(modes, dict) else {}
    for key, setting in changes.items():
        label = control_label(key)
        if not any(isinstance(message, str) and message.startswith(label + "：") and
                   "未完成" not in message for message in (messages or [])):
            continue
        try:
            mode, _requested = setting
            if mode == "auto" and key in AUTO_PROPERTIES:
                if modes.get(key) not in (None, "auto"):
                    continue
                _save_setting(result, key, mode, None)
            elif mode == "manual":
                actual = values.get(key)
                if key == "resolution":
                    actual = _size(actual)
                elif key == "pixel_format":
                    actual = _pixel_format(actual)
                elif key in PROPERTIES:
                    if key in AUTO_PROPERTIES and modes.get(key) not in (None, "manual"):
                        continue
                    actual = _number(actual, positive=key == "fps")
                else:
                    continue
                _save_setting(result, key, mode, actual)
        except (ValueError, TypeError):
            continue
    # A format change may also negotiate different dimensions/FPS. Save the
    # observed group, except explicit rejected items and failed preservation.
    changed_format = any(key in FORMAT_KEYS and any(
        isinstance(message, str) and message.startswith(control_label(key) + "：") and
        "未完成" not in message for message in (messages or [])) for key in changes)
    if changed_format:
        for key in FORMAT_KEYS.difference(changes):
            label = control_label(key)
            if any(isinstance(message, str) and message.startswith(label + "：") and
                   "未完成" in message for message in (messages or [])):
                continue
            value = values.get(key)
            try:
                actual = _size(value) if key == "resolution" else _pixel_format(value) if key == "pixel_format" else _number(value, positive=True)
                _save_setting(result, key, "manual", actual)
            except (ValueError, TypeError):
                continue
    return result
