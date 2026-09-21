# -*- coding: utf-8 -*-
"""vision.choice —— **选路的三个后端**（启发式 / ONNX / ultralytics）。

和 `vision.keys` 是同一套做法（见那边的模块文档），但**类别不同**：

| | 类别 | 一帧的输出 |
|---|---|---|
| `vision.keys` | `brake_pressed` / `nitro_pressed` | 两个布尔 |
| **`vision.choice`** | `choice_icon` / **`choice_selected`** | **三个答案**（是否有选路 / 几个选项 / 选第几个） |

## 启发式后端就是原来那套（HoughCircles）

`HeuristicChoice` 直接调 `hud.choice_icons()`（圆检测 + 蓝色占比）——
**行为和以前逐字一致**，所以不切模型时什么都不变。
已知的两个毛病（NOTES §5 记的）：

* 偶尔**多认一个圆**（用户报过"实际 2 个却判成 3 选 3"）；
* 蓝色高亮偶尔**一个都认不出**（`selected = 0`）→ 只能猜第 1 个。

这两条正是模型要修的：`choice_selected` 是一个**独立类别**，
模型直接学"哪个是选中的"，不用再靠"圆里蓝像素占比 > 0.35"。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from a9route.train.choice import (CHOICE_CLASSES, CHOICE_ZH, MAX_OPTIONS,
                                  MIN_OPTIONS, option_range)

#: 可用的后端名（`config.vision.choice_backend` 的取值）
BACKENDS = ("heuristic", "auto", "onnx", "ultralytics")

#: 推理分辨率 / NMS（默认值和按键那边一致）
DEFAULT_IMGSZ = 640
#: ⚠️ 选路的默认置信度是 **0.30**，不是按键那边的 0.40 ——
#: 这是 `train eval` 在 val 上扫出来的最佳档，也写在模型清单里
#: （`models/choice.json` 的 `conf_default`）。两边默认值**必须一致**：
#: 以前配置默认 0.40、清单记 0.30，于是 clone 下来不带 `config.json` 的人
#: 拿到的是**没调过的那一档**，而文档里写的是 0.30 ✗（2026-09-15 修）。
DEFAULT_CONF = 0.30
DEFAULT_IOU = 0.50


@dataclass
class ChoiceRead:
    """一帧的选路读数 —— **三个答案**就在这儿。"""

    has: bool = False
    count: int = 0
    selected: int = 0            #: **1 起**的下标（0 = 没认出选中的是哪个）
    #: 兼容旧形状：`[{"xy": (x, y), "radius": r, "blue": bool, "blue_ratio": f}]`
    icons: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def as_tuple(self) -> tuple:
        return (self.has, self.count, self.selected)

    def describe(self) -> str:
        if not self.has:
            return "没有选路"
        tail = "" if self.selected else "（**没认出选中的是哪个**）"
        return "{0} 个选项，选第 {1} 个{2}".format(self.count, self.selected, tail)


def _icons_to_choice(icons: list) -> ChoiceRead:
    """`hud.choice_icons()` 的输出 -> `ChoiceRead`（启发式和 ultralytics 共用）。"""
    if not icons:
        return ChoiceRead(False, 0, 0, [])
    ordered = sorted(icons, key=lambda s: s["xy"][0])
    sel = next((i + 1 for i, s in enumerate(ordered) if s.get("blue")), 0)
    return ChoiceRead(True, len(ordered), sel, ordered)


#: 带区闸的容差（像素）。路标偶尔会略微超出带区，所以放一点余量 ——
#: 和 `train.audit._in_band(pad=24)`、`choice.BAND_PAD` 是同一个量级。
BAND_TOL = 24


def _band_gate(dets: list, frame_w: int) -> tuple:
    """**带区闸**：只收"框中心落在选路带里"的检出，返回 `(留下的, 被挡掉几条)`。

    为什么需要它（2026-09-15 实测）：拿选路模型推 `v2` 那两段录像时，
    模型在**带区外**报了几个圆（一个深色半圆、一个条纹圆，都在带下方 30~60px）——
    那些不是路标。运行时的折叠规则是"≥2 个路标 = 岔路口"，
    所以**两三个这种误报就能凭空造出一个岔路口** ✗✗。

    这不违反"判定交给模型"：带区是**游戏 HUD 的固定布局**（和"两个按键框"一样是
    标定量），模型负责"这里有没有路标、选中的是哪一个"，闸只负责
    "只看该看的那块屏幕"。旧启发式其实天生只在带区里找圆，模型没有这个先验，
    所以这个闸得补上。
    """
    from a9route.vision.hud import CHOICE_BAND
    bx, by, bw, bh = CHOICE_BAND
    keep, blocked = [], 0
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if (bx - BAND_TOL <= cx <= bx + bw + BAND_TOL
                and by - BAND_TOL <= cy <= by + bh + BAND_TOL):
            keep.append(d)
        else:
            blocked += 1
    return keep, blocked


class HeuristicChoice:
    """**默认后端**：原来那套 HoughCircles（`hud.choice_icons`）。"""

    name = "heuristic"
    detail = "HoughCircles + 蓝色占比（和以前逐字一致）"

    def __init__(self, reader=None):
        self._reader = reader

    def _reader_obj(self):
        if self._reader is None:
            from a9route.vision.hud import RaceReader
            self._reader = RaceReader()
        return self._reader

    def read(self, frame) -> ChoiceRead:
        try:
            icons = self._reader_obj().choice_icons(frame)
        except Exception as exc:                       # noqa: BLE001
            return ChoiceRead(False, 0, 0, [], {"error": str(exc)})
        return _icons_to_choice(icons)

    def close(self) -> None:
        pass


class OnnxChoice:
    """用 ONNX 模型判选路（**只要 onnxruntime**）。

    检出按 x 排序：`chocie_selected`（选中那个）**排第几**就是"选第几个"。
    """

    name = "onnx"

    def __init__(self, model: str | Path, *, imgsz: int = DEFAULT_IMGSZ,
                 conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU,
                 provider: str = "auto", fmt: str = "auto"):
        from a9route.vision.keys import OnnxDetector
        # 复用按键那边那套 letterbox/解码/NMS —— 只换类别表。
        # `anchor_tol=0`：选路的位置每帧都不同，**没有固定锚点可用**。
        self._det = OnnxDetector(
            model, classes=CHOICE_CLASSES, imgsz=imgsz, conf=conf, iou=iou,
            provider=provider, anchor_tol=0.0, fmt=fmt, what="选路",
            hint="  先分类 + 复核 + 训练：\n"
                 "    a9route train choice build      # 把现有数据分类\n"
                 "    a9route train choice review     # 三个选择题（/choice 页面）\n"
                 "    a9route train run --name choice\n"
                 "    a9route train export --name choice\n"
                 "  或者退回启发式：--set vision__choice_backend=heuristic")
        self.detail = self._det.detail

    def read(self, frame) -> ChoiceRead:
        dets = [d for d in self._det.detections(frame) if d["conf"] >= self._det.conf]
        dets, gated_out = _band_gate(dets, frame.shape[1])
        if not dets:
            return ChoiceRead(False, 0, 0, [],
                              {"backend": self.name, "conf": self._det.conf,
                               "gated_out": gated_out})
        icons = []
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            r = max(x2 - x1, y2 - y1) / 2.0
            icons.append({
                "xy": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "radius": r,
                "blue": d["name"] == "choice_selected",
                "blue_ratio": round(d["conf"], 3),
                "conf": round(d["conf"], 3),
            })
        out = _icons_to_choice(icons)
        out.info = {"backend": self.name, "conf": self._det.conf,
                    "gated_out": gated_out,
                    "raw": [(d["name"], round(d["conf"], 3)) for d in dets]}
        return out

    def close(self) -> None:
        self._det.close()


class UltralyticsChoice:
    """`ultralytics` 后端（要 torch）—— 适合边训边试。"""

    name = "ultralytics"

    def __init__(self, model: str | Path, *, imgsz: int = DEFAULT_IMGSZ,
                 conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU,
                 device: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:                     # pragma: no cover
            raise RuntimeError(
                "这个环境里没有 ultralytics/torch —— 想用 torch 那条路请在"
                "训练环境里跑，或导出 ONNX 后用 "
                "--set vision__choice_backend=onnx") from exc
        p = Path(model)
        if not p.is_file():
            raise RuntimeError(f"找不到模型文件：{p}")
        # 类序闸和 ONNX 那条路一样要过（顺序错了会把刹车/氮气式错位搬到选路上）
        from a9route.train import modelcard as MC
        err = MC.check_card(MC.load(p), CHOICE_CLASSES)
        if err:
            raise RuntimeError("模型不能用：\n  " + err)
        self.model = YOLO(str(p))
        self.imgsz, self.conf, self.iou = int(imgsz), float(conf), float(iou)
        self.device = device
        names = {int(k): str(v) for k, v in dict(getattr(self.model, "names", {}) or {}).items()}
        self.names = names
        self.detail = "{0} @ {1} conf={2}".format(p.name, self.imgsz, self.conf)

    def read(self, frame) -> ChoiceRead:
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                 iou=self.iou, device=self.device, verbose=False)[0]
        boxes = getattr(res, "boxes", None)
        icons = []
        gated_out = 0
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.tolist()
            clss = boxes.cls.tolist()
            confs = boxes.conf.tolist()
            for (x1, y1, x2, y2), cid, cf in zip(xyxy, clss, confs):
                nm = self.names.get(int(cid), "")
                # 同 ONNX 那条路：**过带区闸**（见 `_band_gate` 的说明）
                keep, blocked = _band_gate(
                    [{"xyxy": (x1, y1, x2, y2), "name": nm, "conf": float(cf)}],
                    frame.shape[1])
                gated_out += blocked
                if not keep:
                    continue
                icons.append({"xy": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                              "radius": max(x2 - x1, y2 - y1) / 2.0,
                              "blue": nm == "choice_selected",
                              "blue_ratio": round(float(cf), 3),
                              "conf": round(float(cf), 3)})
        out = _icons_to_choice(icons)
        out.info = {"backend": self.name, "conf": self.conf, "gated_out": gated_out}
        return out


# ---------------------------------------------------------------- 选择与缓存
_lock = threading.Lock()
_cache: dict = {}


def resolve_backend(cfg=None) -> tuple:
    """算出"这次用哪个后端、哪个模型"。

    `auto` = **模型文件在就用模型**；模型不在**报错**（见下面 274 行的注释 ——
    用户明确要求"判定全部交给 YOLO"，所以 `auto` **不静默退回**启发式）。

    ⚠️ 不传 `cfg` 时用的是 `config.current()`（**当前生效**的那份），
    **不是** `load_config()`（只读磁盘）。以前用后者，于是
    `analyze --set vision__choice_backend=heuristic` 这类**本次覆盖**会被漏掉 ✗
    """
    from a9route import config as cfgmod
    from a9route import paths

    cfg = cfg if cfg is not None else cfgmod.current()
    v = (cfg or {}).get("vision", {})
    backend = str(v.get("choice_backend", "heuristic") or "heuristic").strip().lower()
    if backend not in BACKENDS:
        raise ValueError("vision.choice_backend={0!r} 不认识；可选：{1}"
                         .format(backend, ", ".join(BACKENDS)))
    model = str(v.get("choice_model") or "").strip()
    default = paths.DEFAULT_CHOICE_MODEL
    model_path = Path(model) if model else default
    params = {
        "imgsz": int(v.get("choice_imgsz", DEFAULT_IMGSZ)),
        "conf": float(v.get("choice_conf", DEFAULT_CONF)),
        "iou": float(v.get("choice_iou", DEFAULT_IOU)),
        "provider": str(v.get("choice_provider", "auto") or "auto"),
        "fmt": str(v.get("choice_fmt", "auto") or "auto"),
    }
    if backend == "auto":
        # 同按键那边：**不静默退回启发式**，模型不在就报错（用户要求：判定全交给模型）
        if not model_path.is_file():
            raise RuntimeError(
                "vision.choice_backend=auto，但选路模型不存在：{0}\n"
                "  现在是「判定必须交给模型」的配置 —— **不会**悄悄退回 HoughCircles。\n"
                "  修法一（推荐）：a9route train export --name choice"
                "  或  a9route train use choice\n"
                "  修法二（只用于调试/预标注）："
                "a9route config set vision__choice_backend=heuristic".format(model_path))
        backend = "onnx"
    return backend, str(model_path), params


def build_choice_detector(cfg=None, *, backend: str | None = None,
                          model: str | None = None, use_cache: bool = True):
    """造一个选路后端（**带缓存** —— 逐帧调用时不能每帧重建 session）。"""
    from a9route import config as cfgmod

    cfg = cfg if cfg is not None else cfgmod.current()
    be, mp, params = resolve_backend(cfg)
    if backend:
        be = backend
    if model:
        mp = str(model)
    if be == "ultralytics" and mp and Path(mp).suffix.lower() == ".onnx":
        be = "onnx"
    key = (be, mp, params["imgsz"], params["conf"], params["iou"],
           params["provider"], params["fmt"])
    if use_cache:
        with _lock:
            hit = _cache.get(key)
            if hit is not None:
                return hit
    if be == "heuristic":
        det = HeuristicChoice()
    elif be == "onnx":
        det = OnnxChoice(mp, imgsz=params["imgsz"], conf=params["conf"],
                         iou=params["iou"], provider=params["provider"],
                         fmt=params["fmt"])
    elif be == "ultralytics":
        det = UltralyticsChoice(mp, imgsz=params["imgsz"], conf=params["conf"],
                                iou=params["iou"])
    else:                                              # pragma: no cover
        raise ValueError("未知后端 {0!r}".format(be))
    det.detail = getattr(det, "detail", "") or "model={0}".format(mp or "(无)")
    if use_cache:
        with _lock:
            _cache[key] = det
    return det


def reset_cache() -> None:
    with _lock:
        for d in _cache.values():
            try:
                d.close()
            except Exception:
                pass
        _cache.clear()


def describe_choice_backend(cfg=None) -> str:
    """一行话说明选路用的是哪个后端（**别让人猜**）。"""
    def _raw() -> str:
        c = cfg
        if c is None:
            from a9route import config as cfgmod
            c = cfgmod.current()
        return str(((c or {}).get("vision", {}) or {})
                   .get("choice_backend") or "heuristic").lower()

    try:
        be, mp, params = resolve_backend(cfg)
        raw = _raw()
    except Exception as exc:                           # noqa: BLE001
        if _raw() == "auto":
            return ("选路后端: **配置为 auto，但模型不能用** —— 这次分析会直接失败，"
                    "不会退回 HoughCircles。\n    "
                    + str(exc).replace("\n", "\n    "))
        return "选路后端: 配置有问题（{0}: {1}）".format(type(exc).__name__, exc)
    if be == "heuristic":
        return ("选路后端: heuristic HoughCircles（**像素判据**，"
                "正式跑图请用模型：choice_backend={0}）".format(raw))
    zh = "、".join(CHOICE_ZH.get(c, c) for c in CHOICE_CLASSES)
    return ("选路后端: {0}（{1}）  {2}  imgsz={3} conf={4}"
            .format(be, zh, mp, params["imgsz"], params["conf"]))


def clamp_answers(count: int, selected: int, *, lo: int | None = None,
                  hi: int | None = None) -> tuple:
    """把"几个选项/选第几个"夹到规则允许的范围（默认 2~4，选中必须在范围内）。

    范围来自 `train.choice.option_range()` —— 和界面按钮、训练侧写回**同一个来源**，
    所以 `vision.choice_min_options` 改了以后三处一起变（不是各写各的常量）。
    """
    if lo is None or hi is None:
        r_lo, r_hi = option_range()
        lo = r_lo if lo is None else lo
        hi = r_hi if hi is None else hi
    c = max(int(lo), min(int(hi), int(count or 0)))
    s = max(1, min(c, int(selected or 0) or 1))
    return c, s
