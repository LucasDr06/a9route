# -*- coding: utf-8 -*-
"""train.modelcard —— **模型清单**：让"换一个模型"变成一件安全的事。

## 为什么需要它

模型不是只有权重。要正确用起来，还得知道：

* **类别顺序**（`CLASSES` 的第 0 个是刹车还是氮气）—— 顺序错了，模型会把刹车
  当氮气报出来，而且**不会报任何错** ✗✗；
* **推理分辨率**（`imgsz`）—— 导出时用 640、推理时用 416，框会整体偏；
* 它是在**哪个数据集/什么时候**训的、当时的指标是多少（出问题时要能追）。

这些东西如果只写在人的脑子里，那么每次换模型都是一次"猜"。
所以 `a9route train export` 会在 ONNX 旁边写一个**同名 `.json` 清单**，
运行时 `vision.keys` 会读它并**核对**——对不上就直接报错，不静默跑歪的模型。

## 形状

```
models/keys.onnx
models/keys.json          <-- 这个清单
```

```json
{
  "format": 1,
  "name": "keys",
  "onnx": "keys.onnx",
  "source_weights": "models/runs/keys/weights/best.pt",
  "classes": ["brake_pressed", "nitro_pressed"],
  "nc": 2,
  "imgsz": 640,
  "conf_default": 0.40,
  "iou_default": 0.50,
  "trained_at": "2026-09-15 15:20",
  "dataset": "keys",
  "dataset_frames": 1200,
  "dataset_reviewed": 572,
  "metrics": {"box_map50": 0.970, "box_map": 0.970},
  "notes": ""
}
```

**没有清单也能用**（老模型、别人给的模型）—— 这时运行时只做"能查到的那点校验"
（输出通道数 vs 类别数），并给出提示。清单只是把"能自动查的"变多。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: 清单格式版本（以后加字段时用它做兼容分支）
FORMAT = 1
#: 清单后缀：`keys.onnx` -> `keys.json`
SUFFIX = ".json"


def card_path(model: str | Path) -> Path:
    """模型文件对应的清单路径（`x.onnx` -> `x.json`）。"""
    p = Path(model)
    return p.with_suffix(SUFFIX)


@dataclass
class ModelCard:
    """一个模型的清单。字段全都可缺（缺就用运行时的默认值）。"""

    path: str = ""
    name: str = ""
    onnx: str = ""
    source_weights: str = ""
    classes: list = field(default_factory=list)
    nc: int = 0
    imgsz: int = 0
    conf_default: float = 0.0
    iou_default: float = 0.0
    trained_at: str = ""
    dataset: str = ""
    dataset_frames: int = 0
    dataset_reviewed: int = 0
    metrics: dict = field(default_factory=dict)
    notes: str = ""
    loaded: bool = False          #: 真的读到了清单文件（False = 没有/坏了）

    def describe(self) -> str:
        if not self.loaded:
            return "(没有清单 .json —— 用默认值，类序无法自动核对)"
        bits = [f"classes={','.join(self.classes) or '?'}",
                f"imgsz={self.imgsz or '?'}",
                f"conf={self.conf_default or '?'}"]
        if self.trained_at:
            bits.append(f"训于 {self.trained_at}")
        if self.dataset:
            bits.append(f"数据集 {self.dataset}"
                        f"（{self.dataset_frames} 帧，复核 {self.dataset_reviewed}）")
        if self.metrics:
            m = self.metrics
            for k in ("box_map50", "box_map", "mAP50", "mAP50-95"):
                if k in m:
                    bits.append(f"{k}={m[k]:.4f}" if isinstance(m[k], float) else str(m[k]))
                    break
        return "；".join(bits)

    def as_dict(self) -> dict:
        return {
            "format": FORMAT, "name": self.name, "onnx": self.onnx,
            "source_weights": self.source_weights, "classes": list(self.classes),
            "nc": self.nc, "imgsz": self.imgsz,
            "conf_default": self.conf_default, "iou_default": self.iou_default,
            "trained_at": self.trained_at, "dataset": self.dataset,
            "dataset_frames": self.dataset_frames,
            "dataset_reviewed": self.dataset_reviewed,
            "metrics": dict(self.metrics), "notes": self.notes,
        }


def load(model: str | Path) -> ModelCard:
    """读模型清单。**读不到/坏了都返回"空清单"而不是抛异常** ——

    模型本身还能用，只是少了自动校验；这时由调用方决定怎么提示。
    但**如果清单在、只是类序对不上，那是另一回事**：调用方必须报错
    （见 `vision.keys.check_card`）。
    """
    p = card_path(model)
    card = ModelCard(path=str(p))
    if not p.is_file():
        return card
    try:
        raw = json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return card
    card.loaded = True
    card.name = str(raw.get("name") or Path(model).stem)
    card.onnx = str(raw.get("onnx") or Path(model).name)
    card.source_weights = str(raw.get("source_weights") or "")
    card.classes = [str(c) for c in (raw.get("classes") or [])]
    card.nc = int(raw.get("nc") or len(card.classes) or 0)
    card.imgsz = int(raw.get("imgsz") or 0)
    card.conf_default = float(raw.get("conf_default") or 0.0)
    card.iou_default = float(raw.get("iou_default") or 0.0)
    card.trained_at = str(raw.get("trained_at") or "")
    card.dataset = str(raw.get("dataset") or "")
    card.dataset_frames = int(raw.get("dataset_frames") or 0)
    card.dataset_reviewed = int(raw.get("dataset_reviewed") or 0)
    card.metrics = dict(raw.get("metrics") or {})
    card.notes = str(raw.get("notes") or "")
    return card


def write(model: str | Path, *, card: ModelCard) -> Path:
    """写清单（`train export` 调它）。"""
    p = card_path(model)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(card.as_dict(), ensure_ascii=False, indent=2,
                            sort_keys=True) + "\n", encoding="utf-8")
    return p


def list_models(models_dir: str | Path) -> list:
    """列出一个目录下所有模型（`.onnx` / `.pt`），带上它们的清单信息。

    给 `a9route train models` 用 —— **换模型前先看有什么**。
    """
    root = Path(models_dir)
    out = []
    if not root.is_dir():
        return out
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in (".onnx", ".pt"):
            continue
        card = load(p)
        out.append({"path": str(p), "name": p.name,
                    "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
                    "mtime": _mtime(p), "card": card,
                    "description": card.describe()})
    return out


def _mtime(p: Path) -> str:
    import time
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
    except OSError:
        return ""


def known_class_sets() -> dict:
    """项目里所有"合法类别表"：`{任务名: 类别表}`。

    换模型时**必须知道这个模型属于哪个任务** —— 按键的模型不能塞进选路的槽
    （两套类别不同，塞错了两边都会错，而且错得很隐蔽）。
    """
    from a9route.train.choice import CHOICE_CLASSES
    from a9route.train.labels import CLASSES
    return {"keys": tuple(CLASSES), "choice": tuple(CHOICE_CLASSES)}


def identify(card: "ModelCard") -> str:
    """按清单里的类别表判断这是**哪个任务**的模型（`keys` / `choice` / `""`）。

    没有清单、或类序谁都不匹配 -> 返回 `""`（调用方据此拒绝切换）。
    """
    if not card.loaded or not card.classes:
        return ""
    for task, names in known_class_sets().items():
        if list(card.classes) == list(names):
            return task
    return ""


def check_card(card: ModelCard, expected_classes) -> str:
    """核对清单里的类序 vs 运行时的类序。

    返回**错误说明**（空字符串 = 没问题）。这是整个"换模型"流程里最要紧的一道闸：

    * 清单里有类序、且和运行时不一致 -> **必须报错**。
      否则模型输出的 `class 0` 会被当成刹车，而它可能训练时是氮气 ——
      结果就是"刹车和氮气反了"，而且**一路都不报错** ✗✗。
    * 清单里没写类序（没有清单）-> 不报错，让调用方提示"无法核对"。
    """
    expected = [str(c) for c in expected_classes]
    if not card.loaded or not card.classes:
        return ""
    if list(card.classes) != expected:
        return ("模型的类别顺序和本项目不一致：\n"
                "     模型: {0}\n"
                "     项目: {1}\n"
                "   顺序不同会让「刹车」和「氮气」整体错位，而且不会报错。\n"
                "   -> 用本项目的数据集重新导出，或确认这个模型是按同一套类别训的。"
                .format(" / ".join(card.classes), " / ".join(expected)))
    if card.nc and card.nc != len(expected):
        return ("清单里 nc={0}，但项目有 {1} 个类别。"
                .format(card.nc, len(expected)))
    return ""
