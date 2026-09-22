# -*- coding: utf-8 -*-
"""a9route.formats —— **和模型/标注之间的格式契约**（运行时这一侧）。

## 为什么这个模块在运行时项目里

拆项目的时候，数据集与训练搬去了 `a9lab`，但这个项目**必须自己知道两件事**，
否则它连"模型输出里哪个通道是刹车"都答不出来：

1. **类别名**（`brake_pressed` / `nitro_pressed` / `choice_icon` / `choice_selected`）
   —— 模型吐的是 class id，运行时得把它翻译成"刹车按下了"；
2. **模型清单**（`models/<名字>.json`）怎么读、**类序不对必须拦住** ——
   类序错了会让"刹车"和"氮气"整体错位，而且**一路都不报错** ✗✗。

所以这里是**契约的消费端**，`a9lab`（生产端）写出符合这份契约的模型；
两边都有校验，装错任务/类序反了在**两边各拦一次**。

## ⚠️ 改这里要考虑对面

* 加类别：只能**往后追加**（插在中间会让所有旧标注错位）；
* 加清单字段：`a9lab` 的 `core/modelcard.py` 写、这里读，缺字段时用默认值
  （没有清单的老模型也得能用 —— 只是少了自动校验）。

`a9lab` 那边通过 `pip install -e <a9route路径>`（或并排放两个仓库）来用这一份，
**不复制第二份**（复制出来的两份必然漂移 —— 这个项目里已经因为"两套口径"栽过一次）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 按键（两个键）
#: 类别名（顺序 = YOLO 的 class id，**只能往后追加**）
CLASSES: tuple[str, ...] = ("brake_pressed", "nitro_pressed")
#: 名字 -> id
CLASS_IDS: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
#: 中文短名（日志/界面里给人看的）
CLASS_ZH: dict[str, str] = {
    "brake_pressed": "刹车按下",
    "nitro_pressed": "氮气按下",
}
#: 每类的主色（BGR）——标注/复核页画框用
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "brake_pressed": (0, 220, 255),    # 黄
    "nitro_pressed": (60, 60, 245),    # 红
}

# ---------------------------------------------------------------- 选路（岔路口）
#: 选路是两个类：路标本身 + **被选中的那个**。
#: 「选中的那个」做成独立类别是刻意的 —— 模型直接学"哪个是选中的"，
#: 不用再靠"圆里蓝像素占比 > 0.35"这种像素判据。
CHOICE_CLASSES: tuple[str, ...] = ("choice_icon", "choice_selected")
CHOICE_IDS: dict[str, int] = {n: i for i, n in enumerate(CHOICE_CLASSES)}
CHOICE_CLASS_ZH: dict[str, str] = {
    "choice_icon": "路标（未选中）",
    "choice_selected": "路标（选中）",
}
#: 别名：老代码里叫 `CHOICE_ZH`
CHOICE_ZH = CHOICE_CLASS_ZH

#: 一屏最多/最少几条路（用户规则：至少 2 条才算岔路口；一屏只有 2/3/4 条）
MIN_OPTIONS = 2
MAX_OPTIONS = 4


def option_range(cfg: dict | None = None) -> tuple:
    """「有几个选项」的合法范围 —— **配置里的那个来源**。

    标注页的按钮、训练侧写回、运行时夹取都用它，所以只在这里取一次 ——
    各写各的常量必然漂移。

    ⚠️ 配置里被写进**非数字**（手改 `config.json` 打错字、或某个界面校验漏了）时
    **退回内置默认**，绝不抛：调用它的是逐帧浏览和每次写回 ——
    "配置文件里打错一个字"不该变成"页面整页打不开"。
    （这条是搬文件时**丢过一次**的：原来 `train/choice.py` 里有 try/except，
    搬进 `formats.py` 时漏了，于是 `int("啊")` 直接把运行时那条路也炸了 ✗
    —— 是 a9lab 的测试跑出来的。）
    """
    if cfg is None:
        # ⚠️ 这里**必须自己兜住**：`current()` 前要先 `apply()`，而 apply 本身
        # 就可能踩到坏值；抛出去的话调用方连"退默认"的机会都没有
        try:
            from a9route import config as cfgmod

            cfg = cfgmod.current()
        except Exception:                              # noqa: BLE001
            cfg = {}
    v = (cfg or {}).get("vision", {}) if isinstance(cfg, dict) else {}
    try:
        lo = int(v.get("choice_min_options", MIN_OPTIONS))
    except (TypeError, ValueError):
        lo = MIN_OPTIONS
    try:
        hi = int(v.get("choice_max_options", MAX_OPTIONS))
    except (TypeError, ValueError):
        hi = MAX_OPTIONS
    lo = max(1, min(8, lo))
    hi = max(lo, min(8, hi))              # hi 永远不小于 lo（否则"选第几个"没法答）
    return lo, hi


def clamp_answers(count: int, selected: int, *, lo: int | None = None,
                  hi: int | None = None) -> tuple:
    """把"几个选项 / 选第几个"夹到规则允许的范围里。"""
    if lo is None or hi is None:
        r_lo, r_hi = option_range()
        lo = r_lo if lo is None else lo
        hi = r_hi if hi is None else hi
    c = max(int(lo), min(int(hi), int(count or 0)))
    s = int(selected or 0)
    if s < 0 or s > c:
        s = 0                       # 0 = 没认出选中的是哪个（合法状态，不是错误）
    return c, s


# ---------------------------------------------------------------- 模型清单
#: 清单格式版本（以后加字段时用它做兼容分支）
FORMAT = 1
#: 清单后缀：`keys.onnx` -> `keys.json`
SUFFIX = ".json"


def card_path(model: str | Path) -> Path:
    """模型文件对应的清单路径（`x.onnx` -> `x.json`）。"""
    return Path(model).with_suffix(SUFFIX)


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
    #: 真的读到了清单文件（False = 没有清单 / 坏了）
    loaded: bool = False

    def describe(self) -> str:
        if not self.loaded:
            return "(没有清单 .json —— 用默认值，类序无法自动核对)"
        bits = ["classes={0}".format(",".join(self.classes) or "?"),
                "imgsz={0}".format(self.imgsz or "?"),
                "conf={0}".format(self.conf_default or "?")]
        if self.trained_at:
            bits.append("训于 {0}".format(self.trained_at))
        if self.dataset:
            bits.append("数据集 {0}（{1} 帧，复核 {2}）".format(
                self.dataset, self.dataset_frames, self.dataset_reviewed))
        if self.metrics:
            for k in ("box_map50", "box_map", "mAP50", "mAP50-95"):
                if k in self.metrics:
                    v = self.metrics[k]
                    bits.append("{0}={1:.4f}".format(k, v) if isinstance(v, float)
                                else "{0}={1}".format(k, v))
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
    """读模型清单。**读不到 / 坏了都返回"空清单"，不抛异常** ——

    模型本身还能用，只是少了自动校验，由调用方决定怎么提示。
    但**清单在、类序对不上**是另一回事：那是必须报错的（见 `check_card`）。
    """
    p = card_path(model)
    card = ModelCard(path=str(p))
    if not p.is_file():
        return card
    try:
        raw = json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:                                  # noqa: BLE001
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


def check_card(card: ModelCard, expected_classes) -> str:
    """核对清单里的类序 vs 运行时的类序。返回**错误说明**（空串 = 没问题）。

    这是整个"换模型"流程里最要紧的一道闸：

    * 清单里有类序、且和运行时不一致 -> **必须报错**。否则模型输出的 class 0
      会被当成刹车，而它训练时可能是氮气 ——
      结果是"刹车和氮气反了"，而且**一路都不报错** ✗✗；
    * 没有清单（老模型/别人给的模型）-> 不报错，只提示"无法核对"。
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
        return "清单里 nc={0}，但项目有 {1} 个类别。".format(card.nc, len(expected))
    return ""


def known_class_sets() -> dict:
    """本项目认得的类别表：`{任务名: 类别表}`。

    用途只有一个：判断"某个模型属于哪个任务"（按键的模型不能塞进选路的槽 ——
    两套类别不同，塞错了两边都错，而且错得很隐蔽）。
    """
    return {"keys": tuple(CLASSES), "choice": tuple(CHOICE_CLASSES)}


def identify(card: ModelCard) -> str:
    """按清单里的类别表判断这是**哪个任务**的模型（`keys` / `choice` / `""`）。"""
    if not card.loaded or not card.classes:
        return ""
    for task, names in known_class_sets().items():
        if list(card.classes) == list(names):
            return task
    return ""


def list_models(models_dir: str | Path) -> list:
    """列出一个目录下的模型（`.onnx` / `.pt`）及清单信息。"""
    import time

    root = Path(models_dir)
    out = []
    if not root.is_dir():
        return out
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in (".onnx", ".pt"):
            continue
        card = load(p)
        try:
            mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
        except OSError:
            mt = ""
        out.append({"path": str(p), "name": p.name,
                    "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
                    "mtime": mt, "card": card, "description": card.describe()})
    return out
