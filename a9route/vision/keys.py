# -*- coding: utf-8 -*-
"""vision.keys —— **按键检测的三个后端**（启发式 / ONNX / ultralytics），一个接口。

    from a9route.vision import keys
    det = keys.build_key_detector()        # 按 config.vision.key_backend 选
    r = det.read(frame)                    # -> KeyRead(brake=?, nitro=?)

## 为什么要有三个后端（而不是直接换成模型）

| 后端 | 依赖 | 什么时候用 |
|---|---|---|
| `heuristic` | 只要 cv2/numpy | 旧的像素判据。**只在调试/预标注时显式打开** —— 正式跑图不用它 |
| `onnx` | `onnxruntime`（约 15 MB） | **当前默认走这条**：训练在 `ai_joy`（有 torch cu128），运行时不必为了推理再装 2.5 GB torch |
| `ultralytics` | `torch` + `ultralytics` | 手里只有 `.pt`、或者想边训边试的时候 |

**默认是 `auto`**：`models/keys.onnx` 在就用模型（解析成 `onnx`）。
但 ⚠️ `auto` **不是**"模型没了就悄悄退回启发式"（2026-09-15 按用户要求改）——
模型文件不在会**直接报错**，因为"看起来正常、其实是像素判据"的路线比报错更坏 ✗。
要显式用判定旧那一套：`a9route config set vision__key_backend=heuristic`。

## 三个后端必须**同语义**

`KeyRead.brake` 的语义**永远是**："这一帧刹车键是不是按下外观"。
之前 `scan_fine` 用的是 `brake_pressed() > BRAKE_WHITE_THR`（**只看绝对白度**），
而 `cues.key_pressed()` 用的是"圈内−圈外 **或** 绝对白度"（**并集**）——
这两处口径本来就不一致。这里统一成"和 `cues.key_pressed()` 一致"，
因为超参（`brake_ring_thr`/`nitro_red_thr`）都是照它调出来的。
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from a9route.train import modelcard
from a9route.train.labels import CLASS_IDS, CLASSES

#: 可用的后端名（`config.vision.key_backend` 的取值）
BACKENDS = ("heuristic", "auto", "onnx", "ultralytics")

#: 默认推理分辨率（和训练时一致；改这里必须和 `train export --imgsz` 对齐）
DEFAULT_IMGSZ = 640
#: 默认置信度阈值。**不要调得太低** —— 这个项目的已知问题是"氮气误报偏多"，
#: 0.35~0.5 之间才是想要的取舍
DEFAULT_CONF = 0.40
#: 默认 NMS IoU
DEFAULT_IOU = 0.50
#: letterbox 的填充灰度（**必须和 ultralytics 一致**，否则框会整体偏）
PAD_VALUE = 114


@dataclass
class KeyRead:
    """一帧的按键读数。`info` 里放后端自己的明细（调试/复核用）。"""

    brake: bool = False
    nitro: bool = False
    info: dict = field(default_factory=dict)

    def as_tuple(self) -> tuple[bool, bool]:
        return (self.brake, self.nitro)


# ---------------------------------------------------------------- 几何（ONNX 用）
def letterbox(frame, imgsz: int = DEFAULT_IMGSZ) -> tuple:
    """把整帧缩放到 `imgsz×imgsz`（保持比例，居中填边）。

    返回 `(画布, scale, left, top)`；反算坐标用 `unletterbox()`。
    **这里的取整方式照抄 ultralytics**（`round(dw - 0.1)`）——
    差 1 个像素就会让框整体偏 1 像素，和训练时的坐标对不上。
    """
    import cv2

    h, w = frame.shape[:2]
    r = min(imgsz / float(w), imgsz / float(h))
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = None
    dw = (imgsz - new_w) / 2.0
    dh = (imgsz - new_h) / 2.0
    left, top = int(round(dw - 0.1)), int(round(dh - 0.1))
    import numpy as np
    canvas = np.full((imgsz, imgsz, 3), PAD_VALUE, dtype=frame.dtype)
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, r, left, top


def unletterbox(box_xyxy, scale: float, left: int, top: int) -> tuple[float, float, float, float]:
    """letterbox 坐标 -> 原图坐标。"""
    x1, y1, x2, y2 = box_xyxy
    return ((x1 - left) / scale, (y1 - top) / scale,
            (x2 - left) / scale, (y2 - top) / scale)


def to_blob(canvas, imgsz: int = DEFAULT_IMGSZ):
    """画布 -> ONNX 输入张量 `(1,3,imgsz,imgsz)`（**RGB**、0~1、NCHW）。

    ultralytics 走的是 BGR→RGB（`im[:, :, ::-1]`），这里必须一样。
    """
    import numpy as np

    rgb = canvas[:, :, ::-1]
    x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return x


def _nms(boxes, scores, iou_thr: float) -> list[int]:
    """按类做一次 NMS，返回留下的下标。

    用 `cv2.dnn.NMSBoxes`（两边环境都有 OpenCV），不自己写 ——
    自己写的 NMS 是这类脚本最容易出错的地方。
    """
    import cv2
    import numpy as np

    if not len(boxes):
        return []
    rects = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
             for b in boxes]
    idx = cv2.dnn.NMSBoxes(rects, [float(s) for s in scores], 0.0, float(iou_thr))
    if idx is None or len(idx) == 0:
        return []
    return [int(i) for i in np.array(idx).reshape(-1)]


# ---------------------------------------------------------------- 后端
class KeyDetector:
    """后端接口。`name` 用来打日志，`read()` 是唯一要实现的。"""

    name = "base"
    #: 后端加载失败时的原因（doctor/CLI 用来给出"怎么修"）
    detail = ""

    def read(self, frame) -> KeyRead:      # pragma: no cover - 抽象
        raise NotImplementedError

    def close(self) -> None:
        pass


class HeuristicKeys(KeyDetector):
    """**默认后端**：原样调用现成的启发式（`cues` 里那套实测判据）。

    它同时是"标注的源头"和"模型的对照组" —— `train.runner.evaluate_frames`
    拿它和模型逐帧比，才知道模型到底有没有变好。

    ## ⚠️ `mode`：项目里其实有**两套**启发式，它们不一样

    接线时发现的一个真问题（**没有改它，只是把它摊开**）：

    | | 刹车 | 氮气 |
    |---|---|---|
    | `mode="fine"`（`scan_fine` 一直在用的） | `bright_ratio > 0.35`（**只看绝对白度**） | `max(red, ring) > 0.15`（**红 ∪ 圈内−圈外**） |
    | `mode="cues"`（`cues.key_pressed()`，粗扫在用） | `ring > 0.35 or white > 0.35`（**并集**） | `red > 0.15`（**只认红**） |

    两边都不一致，而且 `mode="fine"` 的氮气把**圈内−圈外**也并进来了 ——
    可 NOTES §1 实测的结论恰恰是"**圈内判据对氮气没信号**
    （测试3 按下 0.004），并进来只会把误报抬高 ✗"，
    `cues.py` 里的注释也写着"氮气只认红 >0.15"。
    所以 README 里那条已知问题「**氮气偏多**」，至少有一部分是这儿来的：
    真正产生路线里 `N:` 条目的是 `scan_fine`，而它用的是被 NOTES 否掉的那个并集。

    * 默认 `mode="fine"` —— **逐字保持现状**，路线输出一个字节都不变；
    * 想试"按 NOTES 的结论统一"就设 `--set vision__key_heuristic_mode=cues`，
      然后用 `a9route train pulses` 对比两边的动作数。
    """

    name = "heuristic"

    def __init__(self, mode: str = "fine"):
        self.mode = mode if mode in ("fine", "cues") else "fine"
        self.detail = "mode={0}".format(self.mode)

    def read(self, frame) -> KeyRead:
        from a9route.vision import cues

        if self.mode == "cues":
            hit_b, info_b = cues.key_pressed(frame, cues.BRAKE_KEY_BOX, "brake")
            hit_n, info_n = cues.key_pressed(frame, cues.NITRO_KEY_BOX, "nitro")
            return KeyRead(
                brake=bool(hit_b), nitro=bool(hit_n),
                info={"ring": info_b["ring"], "brake_white": info_b["white"],
                      "red": info_n["red"], "nitro_white": info_n["white"],
                      "mode": self.mode})
        # mode="fine" —— `scan_fine()` 原来的两行，逐字搬过来
        brake = cues.brake_pressed(frame, cues.BRAKE_KEY_BOX) > cues.BRAKE_WHITE_THR
        nitro = cues.nitro_pressed(frame, cues.NITRO_KEY_BOX) > cues.NITRO_RED_THR
        return KeyRead(
            brake=bool(brake), nitro=bool(nitro),
            info={"brake_white": round(cues.bright_ratio(frame, cues.BRAKE_KEY_BOX), 3),
                  "nitro_max": round(float(cues.nitro_pressed(frame, cues.NITRO_KEY_BOX)), 3),
                  "mode": self.mode})


class OnnxDetector:
    """**通用的 ONNX 目标检测器**（不限类别）—— 按键和选路共用这一份。

    它只做一件事：`frame -> [{"name": 类名, "cls_id": i, "conf": c, "xyxy": (...)},
    ...]`（**原图坐标**，已做 NMS）。"这些框表示什么"由调用方决定：

    * `OnnxKeys`（按键）-> 看两类有没有出现 -> 两个布尔；
    * `OnnxChoice`（选路）-> 数一共几个、哪个是 `choice_selected` -> 三个答案。

    把 letterbox / 解码 / NMS 这套**最容易写错**的东西只留一份 ——
    选路模型直接复用它，不用再抄一遍（抄一遍就多一处"框整体偏"的机会）。

    ## ⚠️ 输出形态的**歧义**（这个坑很隐蔽）

    YOLOv8 导出 ONNX 有两种输出：

    * **未做 NMS**（`yolo export` 的默认）：`(1, 4+nc, N)` —— 行是 `cx,cy,w,h,类分数...`；
    * **已做 NMS**（`nms=True`）：`(1, N, 6)` —— 行是 `x1,y1,x2,y2,conf,cls`。

    我们只有 **2 类**，于是 `4+nc == 6` —— **两种形态的行宽一模一样**，
    光看形状分不出来 ✗。这时候"猜错"不会报错，只会**静默把坐标当分数用**，
    结果就是框全乱、检出全错，而且看起来像"模型训废了"。

    处理办法（**明确 + 可覆盖**）：

    * `fmt="auto"`（默认）：按 `列宽 == 4+nc` 判成**未做 NMS** ——
      因为 `a9route train export` 导出的就是这种（**我们不传 `nms=True`**）；
    * 如果你自己用 `nms=True` 导过，就把 `vision.key_fmt` 设成 `nms`
      （或给 `OnnxDetector(fmt="nms")`），别指望它自己认出来。
    """

    name = "onnx"

    def __init__(self, model: str | Path, *, classes=None,
                 imgsz: int = DEFAULT_IMGSZ,
                 conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU,
                 provider: str = "auto", anchor_tol: float = 0.0,
                 fmt: str = "auto", what: str = "按键",
                 hint: str = ""):
        try:
            import onnxruntime as ort
        except ImportError as exc:                       # pragma: no cover
            raise RuntimeError(
                "没装 onnxruntime —— 这是用 ONNX 模型推理必需的（约 15 MB）：\n"
                "    python -m pip install onnxruntime\n"
                "  （或者改用 torch 那套：--set vision__key_backend=ultralytics）"
            ) from exc
        self.classes = tuple(classes) if classes else CLASSES
        p = Path(model)
        if not p.is_file():
            raise RuntimeError(
                "找不到{0}模型：{1}\n{2}".format(
                    what, p, hint or "  先训练并导出：a9route train run / "
                                     "a9route train export"))
        self.model_path = p
        # ---- 模型清单：核对类序 + 跟随它自己的 imgsz（**换模型时最要紧的那道闸**）----
        self.card = modelcard.load(p)
        self.card_error = modelcard.check_card(self.card, self.classes)
        if self.card_error:
            # **宁可起不来，也不能静默跑一个类序反了的模型**
            raise RuntimeError("模型不能用（换模型时最怕的就是这种情况）：\n  "
                               + self.card_error)
        if self.card.loaded and self.card.imgsz and int(imgsz) == DEFAULT_IMGSZ:
            # 用户没显式改过 imgsz（还是默认值）-> 跟随模型清单里记的。
            # 否则"导出时 640、推理时写了个别的值"会让框整体偏，还不报错。
            imgsz = int(self.card.imgsz)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.anchor_tol = float(anchor_tol)
        self.fmt = fmt if fmt in ("auto", "raw", "nms") else "auto"
        avail = ort.get_available_providers()
        want = ([provider] if provider and provider != "auto"
                else ["CUDAExecutionProvider", "CPUExecutionProvider"])
        use = [p_ for p_ in want if p_ in avail] or ["CPUExecutionProvider"]
        self.providers = use
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(p), sess_options=so, providers=use)
        self.input_name = self.session.get_inputs()[0].name
        self.detail = "{0} @ {1} conf={2} {3}".format(
            p.name, self.imgsz, self.conf, ",".join(use))
        if self.card.loaded:
            self.detail += "  清单[{0}]".format(self.card.name or p.stem)

    # ---------------------------------------------------------------- 解码
    def _decode(self, out, w: int, h: int) -> list:
        """ONNX 输出 -> `[(clsid, conf, (x1,y1,x2,y2)), ...]`（原图坐标，已 NMS）。

        兼容三种导出形态（见类文档里的歧义说明）：
          1. `(1, 4+nc, N)` 未做 NMS —— 要转置 + 自己 NMS（**默认导出**）
          2. `(1, N, 4+nc)` 未做 NMS —— 只是转置过的
          3. `(1, N, 6)`    已做 NMS —— `[x1,y1,x2,y2,conf,cls]`（**要显式指定**）
        """
        import numpy as np

        a = np.asarray(out)
        a = a[0] if a.ndim == 3 else a
        nc = len(self.classes)
        found: list = []
        if a.ndim != 2:
            return found
        fmt = self.fmt
        if fmt == "auto":
            fmt = "raw" if a.shape[1] == 4 + nc else ("nms" if a.shape[1] == 6 else "?")
        # ---- 形态 3：已经 NMS 过（列数 = 6）
        if fmt == "nms":
            if a.shape[1] != 6:
                raise RuntimeError(
                    f"fmt=nms 但输出列数是 {a.shape[1]}（应为 6）")
            for row in a:
                c = float(row[4])
                cid = int(round(float(row[5])))
                if c < self.conf or not 0 <= cid < nc:
                    continue
                if self._gated(cid, row[:4], w):
                    found.append((cid, c, tuple(
                        unletterbox([float(v) for v in row[:4]], self._scale,
                                    self._left, self._top))))
            return found
        # ---- 形态 1/2：`4+nc` 那一维在行还是列
        if a.shape[0] == 4 + nc and a.shape[1] != 4 + nc:
            a = a.T
        if a.shape[1] != 4 + nc:
            raise RuntimeError(
                f"看不懂的模型输出形状 {np.asarray(out).shape} —— "
                f"期望 (1,{4 + nc},N) / (1,N,{4 + nc}) / (1,N,6)"
                f"（已做 NMS 的导出要设 key_fmt=nms）")
        cls_scores = a[:, 4:4 + nc]
        cls_ids = cls_scores.argmax(axis=1)
        confs = cls_scores.max(axis=1)
        keep = confs >= self.conf
        if not keep.any():
            return found
        cx, cy, bw, bh = a[keep, 0], a[keep, 1], a[keep, 2], a[keep, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        confs = confs[keep]
        cls_ids = cls_ids[keep]
        for cid in range(nc):
            m = cls_ids == cid
            if not m.any():
                continue
            sub_boxes, sub_conf = boxes[m], confs[m]
            for k in _nms(sub_boxes, sub_conf, self.iou):
                xyxy = sub_boxes[k]
                if not self._gated(cid, xyxy, w):
                    continue
                found.append((cid, float(sub_conf[k]), tuple(
                    unletterbox([float(v) for v in xyxy], self._scale,
                                self._left, self._top))))
        return found

    def _gated(self, cid: int, xyxy, w: int) -> bool:
        """定位闸（`anchor_tol=0` 时整个闸关掉，默认关）。

        只对**按键**有意义（"框中心必须离标定框不远"）；选路的目标位置每帧都不同，
        所以子类把 `anchor_tol` 置 0 即可。基类这里只做通用判断，
        具体"期望位置"由 `_expected_center()` 给（默认 None = 不过闸）。
        """
        if self.anchor_tol <= 0:
            return True
        exp = self._expected_center(cid)
        if exp is None:
            return True
        cx = (float(xyxy[0]) + float(xyxy[2])) / 2.0
        cy = (float(xyxy[1]) + float(xyxy[3])) / 2.0
        tol = self.anchor_tol * w
        return ((cx - exp[0]) ** 2 + (cy - exp[1]) ** 2) ** 0.5 <= tol

    def _expected_center(self, cid: int):
        return None

    # ---------------------------------------------------------------- 推理
    def detections(self, frame) -> list:
        """跑一帧，返回**所有**检出：`[{"name", "cls_id", "conf", "xyxy"}, ...]`。"""
        h, w = frame.shape[:2]
        canvas, self._scale, self._left, self._top = letterbox(frame, self.imgsz)
        out = self.session.run(None, {self.input_name: to_blob(canvas, self.imgsz)})
        return [{"name": self.classes[cid] if 0 <= cid < len(self.classes) else "?",
                 "cls_id": int(cid), "conf": float(c),
                 "xyxy": (float(b[0]), float(b[1]), float(b[2]), float(b[3]))}
                for cid, c, b in self._decode(out[0], w, h)]

    def detect(self, frame) -> dict:
        """每类的**最高置信度** `{类名: conf}`（按键只需要这个）。"""
        best: dict = {}
        for d in self.detections(frame):
            if d["conf"] > best.get(d["name"], 0.0):
                best[d["name"]] = d["conf"]
        return best

    def close(self) -> None:
        self.session = None


class OnnxKeys(OnnxDetector):
    """按键专用的 ONNX 后端（在 `OnnxDetector` 之上只加"两个布尔"的解释）。"""

    name = "onnx"

    def _expected_center(self, cid: int):
        """按键的定位闸：期望位置就是**标定框的中心**（见基类 `_gated`）。"""
        from a9route.vision import cues
        want = "brake_pressed" if cid == 0 else "nitro_pressed"
        box = cues.BRAKE_KEY_BOX if want == "brake_pressed" else cues.NITRO_KEY_BOX
        return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)

    def read(self, frame) -> KeyRead:
        best = self.detect(frame)
        brake = best.get("brake_pressed", 0.0)
        nitro = best.get("nitro_pressed", 0.0)
        return KeyRead(brake=brake >= self.conf, nitro=nitro >= self.conf,
                       info={"brake": round(brake, 3), "nitro": round(nitro, 3),
                             "backend": self.name})


class UltralyticsKeys(KeyDetector):
    """`ultralytics` 后端（要 torch）。适合**边训边试**，不适合日常跑全片。"""

    name = "ultralytics"

    def __init__(self, model: str | Path, *, imgsz: int = DEFAULT_IMGSZ,
                 conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU,
                 device: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:                       # pragma: no cover
            raise RuntimeError(
                "这个环境里没有 ultralytics/torch ——\n"
                "  · 想用 torch 那条路：在训练环境里跑（见 TRAINING.md）\n"
                "  · 日常跑全片建议导出 ONNX，然后 "
                "--set vision__key_backend=onnx（只用 onnxruntime）"
            ) from exc
        p = Path(model)
        if not p.is_file():
            raise RuntimeError(f"找不到模型文件：{p}")
        self.model = YOLO(str(p))
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        names = getattr(self.model, "names", {}) or {}
        self.names = {int(k): str(v) for k, v in dict(names).items()}
        self.detail = f"{p.name} @ {self.imgsz} conf={self.conf}"

    def read(self, frame) -> KeyRead:
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                 iou=self.iou, device=self.device, verbose=False)[0]
        best: dict[str, float] = {}
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes):
            clss = boxes.cls.tolist()
            confs = boxes.conf.tolist()
            for cid, cf in zip(clss, confs):
                nm = self.names.get(int(cid), "")
                if nm in CLASS_IDS and cf > best.get(nm, 0.0):
                    best[nm] = float(cf)
        brake = best.get("brake_pressed", 0.0)
        nitro = best.get("nitro_pressed", 0.0)
        return KeyRead(brake=brake >= self.conf, nitro=nitro >= self.conf,
                       info={"brake": round(brake, 3), "nitro": round(nitro, 3),
                             "backend": self.name})


# ---------------------------------------------------------------- 选择与缓存
_lock = threading.Lock()
_cache: dict[tuple, KeyDetector] = {}


def resolve_backend(cfg=None) -> tuple[str, str, dict]:
    """算出"这次到底用哪个后端、哪个模型文件"。

    返回 `(后端名, 模型路径, 参数)`。`auto` 的解析规则就是文档里说的那一条：
    **模型文件存在就用模型；不存在就报错**（不静默退回启发式）。

    ⚠️ 不传 `cfg` 时用 `config.current()`（**当前生效**那份），不是 `load_config()`
    （只读磁盘）—— 后者会把 `analyze --set vision__key_backend=...` 这类本次覆盖漏掉。
    """
    from a9route import config as cfgmod
    from a9route import paths

    cfg = cfg if cfg is not None else cfgmod.current()
    v = (cfg or {}).get("vision", {})
    backend = str(v.get("key_backend", "heuristic") or "heuristic").strip().lower()
    if backend not in BACKENDS:
        raise ValueError(f"vision.key_backend={backend!r} 不认识；可选：{', '.join(BACKENDS)}")
    model = str(v.get("key_model") or "").strip()
    model_path = Path(model) if model else paths.DEFAULT_KEY_MODEL
    params = {
        "imgsz": int(v.get("key_imgsz", DEFAULT_IMGSZ)),
        "conf": float(v.get("key_conf", DEFAULT_CONF)),
        "iou": float(v.get("key_iou", DEFAULT_IOU)),
        "provider": str(v.get("key_provider", "auto") or "auto"),
        "anchor_tol": float(v.get("key_anchor_tol", 0.0)),
        "mode": str(v.get("key_heuristic_mode", "fine") or "fine"),
        "fmt": str(v.get("key_fmt", "auto") or "auto"),
    }
    if backend == "auto":
        # ⚠️ **`auto` 不再静默退回启发式**（2026-09-15 按用户要求改）：
        # 用户要的是"所有判定都交给 YOLO"，那么"模型不在"就**必须响亮地报错**，
        # 而不是悄悄换成像素判据跑出一条看起来正常的路线 ✗
        #（以前是 `backend = "onnx" if model_path.is_file() else "heuristic"`，
        #  于是删掉模型文件之后，路线的判据会**无声无息**地退化成启发式。）
        # 想用启发式请**显式**设 `vision.key_backend=heuristic`（调试/预标注用）。
        if not model_path.is_file():
            raise RuntimeError(
                "vision.key_backend=auto，但按键模型不存在：{0}\n"
                "  现在是「判定必须交给模型」的配置 —— **不会**悄悄退回启发式。\n"
                "  修法一（推荐）：a9route train export --name keys"
                "  或  a9route train use <模型名>\n"
                "  修法二（只用于调试/预标注）："
                "a9route config set vision__key_backend=heuristic".format(model_path))
        backend = "onnx"
    return backend, str(model_path), params


def build_key_detector(cfg=None, *, backend: str | None = None,
                       model: str | None = None, use_cache: bool = True) -> KeyDetector:
    """按配置造一个后端（**带缓存** —— 逐帧调用时不能每帧重建 session）。"""
    from a9route import config as cfgmod
    from a9route import paths

    cfg = cfg if cfg is not None else cfgmod.current()
    be, mp, params = resolve_backend(cfg)
    if backend:
        be = backend
    if model:
        mp = str(model)
    if be == "ultralytics" and mp and Path(mp).suffix.lower() == ".onnx":
        be = "onnx"                      # 给了 onnx 却选 ultralytics —— 别硬来
    key = (be, mp, params["imgsz"], params["conf"], params["iou"],
           params["provider"], params["anchor_tol"], params["mode"], params["fmt"])
    if use_cache:
        with _lock:
            hit = _cache.get(key)
            if hit is not None:
                return hit
    if be == "heuristic":
        det: KeyDetector = HeuristicKeys(mode=params["mode"])
    elif be == "onnx":
        det = OnnxKeys(mp, imgsz=params["imgsz"], conf=params["conf"],
                       iou=params["iou"], provider=params["provider"],
                       anchor_tol=params["anchor_tol"], fmt=params["fmt"])
    elif be == "ultralytics":
        det = UltralyticsKeys(mp, imgsz=params["imgsz"], conf=params["conf"],
                              iou=params["iou"])
    else:                                                # pragma: no cover
        raise ValueError(f"未知后端 {be!r}")
    det.detail = getattr(det, "detail", "") or f"model={mp or '(无)'}"
    if use_cache:
        with _lock:
            _cache[key] = det
    return det


def reset_cache() -> None:
    """清缓存（测试/切换模型时用；`config.apply()` 之后建议清一次）。"""
    with _lock:
        for d in _cache.values():
            try:
                d.close()
            except Exception:
                pass
        _cache.clear()


def describe_backend(cfg=None) -> str:
    """一行话说明当前后端（CLI 启动时打出来，**别让人猜现在用的是哪套判据**）。

    `auto` 会**说清它到底选了哪个**；模型文件不存在时 `resolve_backend` 会抛
    （`auto` 不静默退回启发式），这里把那句话原样报出来，而不是含糊过去。
    """
    from a9route.train.labels import CLASS_ZH

    def _raw() -> str:
        """当前配置里那个**原始**取值（不传 cfg 时看 `config.current()` ——
        否则 `describe_backend()` 会看不见 `--set`/环境变量造成的本次覆盖 ✗）。"""
        c = cfg
        if c is None:
            from a9route import config as cfgmod
            c = cfgmod.current()
        return str(((c or {}).get("vision", {}) or {}).get("key_backend") or "heuristic").lower()

    try:
        be, mp, params = resolve_backend(cfg)
        raw = _raw()
    except Exception as exc:
        if _raw() == "auto":
            return ("按键后端: **配置为 auto，但模型不能用** —— 这次分析会直接失败，"
                    "不会退回像素判据。\n    " + str(exc).replace("\n", "\n    "))
        return f"按键后端: 配置有问题（{type(exc).__name__}: {exc}）"
    if be == "heuristic":
        return ("按键后端: heuristic 启发式判据（**像素判据**，"
                "正式跑图请用模型：vision.key_backend 现在是 " + raw + "）")
    exists = Path(mp).is_file()
    zh = "、".join(CLASS_ZH.get(c, c) for c in CLASSES)
    line = (f"按键后端: {be}（{zh}）  {mp}"
            f"{'' if exists else '  ⚠️ 文件不存在'}"
            f"  imgsz={params['imgsz']} conf={params['conf']}")
    if be == "onnx" and exists:
        card = modelcard.load(mp)
        line += "  |  " + card.describe()
    return line


def _debug_main(argv: list[str] | None = None) -> int:   # pragma: no cover
    """`python -m a9route.vision.keys 模型.onnx 图.jpg` —— 单帧试模型用。"""
    import sys

    from a9route import config as cfgmod
    cfgmod.apply()
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(describe_backend())
        return 0
    import cv2
    model = args[0]
    det = build_key_detector(backend="onnx", model=model, use_cache=False)
    print(f"后端 {det.name}  {det.detail}")
    for img in args[1:]:
        frame = cv2.imread(img)
        if frame is None:
            print(f"  读不到 {img}")
            continue
        r = det.read(frame)
        print(f"  {Path(img).name}: 刹车={r.brake} 氮气={r.nitro}  {r.info}")
    return 0


if __name__ == "__main__":                               # pragma: no cover
    raise SystemExit(_debug_main())
