"""SOP 定義格式：工序數量與內容全部由 JSON 設定檔決定，程式不寫死。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Sequence

SCHEMA_VERSION = 3
CONDITION_TYPES = {"appear": "出現", "disappear": "消失"}


@dataclass
class Condition:
    type: str = "appear"          # appear：數量 ≥ min_count；disappear：數量 = 0
    label: str = ""
    roi: str = ""                 # 空字串 = 整個畫面
    min_count: int = 1
    min_score: float = 0.5
    roi_overlap: float = 0.5      # 物件有多少比例落在區域內才算「在區域內」

    def describe(self) -> str:
        where = f"＠{self.roi}" if self.roi else ""
        if self.type == "disappear":
            return f"{self.label}{where} 消失"
        count = f" ×{self.min_count}" if self.min_count > 1 else ""
        return f"{self.label}{where} 出現{count}"


@dataclass
class ROI:
    name: str
    points: list[tuple[float, float]]   # 正規化座標 0~1，換攝影機解析度也不用重畫
    anchor: str = "fixed"              # workpiece：參考影像座標，由定位矩陣轉到當前畫面


@dataclass
class Workpiece:
    name: str = "主要工件"
    reference_png: str = ""            # base64 PNG，隨 SOP 儲存、另存與搬移
    points: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class Step:
    name: str = "新工序"
    instruction: str = ""
    conditions: list[Condition] = field(default_factory=list)   # 全部成立才算完成
    forbidden: list[Condition] = field(default_factory=list)    # 全部成立 = 違規（例如跳步）
    hold_sec: float = 1.0         # 條件需持續多久
    ratio: float = 0.8            # 持續期間內至少多少比例的幀成立（抗辨識閃爍）
    timeout_sec: float = 0.0      # 0 = 不限時
    completion_mode: str = "all"  # all / any；工序之間仍依序執行


@dataclass
class CycleSettings:
    reset_conditions: list[Condition] = field(default_factory=list)
    reset_hold_sec: float = 2.0
    auto_restart_sec: float = 3.0   # 沒有重置條件時，本輪完成後幾秒自動開始下一輪


@dataclass
class SOPDefinition:
    name: str = "未命名 SOP"
    version: int = 1
    model_path: str = ""
    model_version_id: str = ""
    rois: list[ROI] = field(default_factory=list)
    cycle: CycleSettings = field(default_factory=CycleSettings)
    steps: list[Step] = field(default_factory=list)
    workpiece: Workpiece | None = None

    def roi_map(self) -> dict[str, ROI]:
        return {roi.name: roi for roi in self.rois}

    def to_dict(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "SOPDefinition":
        if int(data.get("schema_version", 1)) > SCHEMA_VERSION:
            raise ValueError("不支援較新版的 SOP 格式")
        cycle = data.get("cycle") or {}
        workpiece = data.get("workpiece")
        return cls(
            name=str(data.get("name", "未命名 SOP")),
            version=int(data.get("version", 1)),
            model_path=str(data.get("model_path", "")),
            model_version_id=str(data.get("model_version_id", "")),
            workpiece=Workpiece(str(workpiece.get("name", "主要工件")),
                                str(workpiece.get("reference_png", "")),
                                [(float(x), float(y)) for x, y in workpiece.get("points", [])])
                      if workpiece else None,
            rois=[ROI(str(r["name"]), [(float(x), float(y)) for x, y in r["points"]],
                      str(r.get("anchor", "fixed")))
                  for r in data.get("rois", [])],
            cycle=CycleSettings(
                reset_conditions=_conditions(cycle.get("reset_conditions")),
                reset_hold_sec=float(cycle.get("reset_hold_sec", 2.0)),
                auto_restart_sec=float(cycle.get("auto_restart_sec", 3.0)),
            ),
            steps=[Step(
                name=str(s.get("name", "新工序")),
                instruction=str(s.get("instruction", "")),
                conditions=_conditions(s.get("conditions")),
                forbidden=_conditions(s.get("forbidden")),
                hold_sec=float(s.get("hold_sec", 1.0)),
                ratio=float(s.get("ratio", 0.8)),
                timeout_sec=float(s.get("timeout_sec", 0.0)),
                completion_mode=str(s.get("completion_mode", "all")),
            ) for s in data.get("steps", [])],
        )

    def validate(self, classes: Sequence[str] | None = None) -> tuple[list[str], list[str]]:
        """回傳 (錯誤, 提醒)。有錯誤時不應開始作業。"""
        errors: list[str] = []
        warnings: list[str] = []
        roi_names = [roi.name for roi in self.rois]
        if len(roi_names) != len(set(roi_names)):
            errors.append("偵測區域名稱重複")
        if not self.steps:
            errors.append("SOP 沒有任何工序")
        for roi in self.rois:
            if roi.anchor not in ("fixed", "workpiece"):
                errors.append(f"區域「{roi.name}」的定位基準無效")
            if roi.anchor == "workpiece" and self.workpiece is None:
                errors.append(f"區域「{roi.name}」尚未設定參考工件")
        if self.workpiece is not None:
            from .tracking import WorkpieceLocator
            try:
                WorkpieceLocator(self.workpiece)
            except (ValueError, TypeError) as exc:
                errors.append(f"工件定位設定：{exc}")

        def check(conditions: list[Condition], where: str):
            for cond in conditions:
                if cond.type not in CONDITION_TYPES:
                    errors.append(f"{where}：未知的條件類型 {cond.type}")
                if not cond.label:
                    errors.append(f"{where}：有條件尚未選擇物件")
                elif classes is not None and cond.label not in classes:
                    errors.append(f"{where}：模型沒有「{cond.label}」這個類別")
                if cond.roi and cond.roi not in roi_names:
                    errors.append(f"{where}：找不到偵測區域「{cond.roi}」")

        for index, step in enumerate(self.steps, 1):
            where = f"工序 {index}「{step.name}」"
            if step.completion_mode not in ("all", "any"):
                errors.append(f"{where}：完成方式無效")
            if not step.conditions:
                warnings.append(f"{where} 沒有完成條件，執行時需手動確認")
            if not 0 < step.ratio <= 1:
                errors.append(f"{where}：成立比例需介於 1%～100%")
            check(step.conditions, where)
            check(step.forbidden, where + "的違規條件")
        check(self.cycle.reset_conditions, "循環重置條件")
        return errors, warnings


def _conditions(rows) -> list[Condition]:
    return [Condition(
        type=str(r.get("type", "appear")),
        label=str(r.get("label", "")),
        roi=str(r.get("roi", "") or ""),
        min_count=max(1, int(r.get("min_count", 1))),
        min_score=float(r.get("min_score", 0.5)),
        roi_overlap=float(r.get("roi_overlap", 0.5)),
    ) for r in rows or []]


def load_sop(path: str | Path) -> SOPDefinition:
    return SOPDefinition.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_sop(sop: SOPDefinition, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(sop.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)
