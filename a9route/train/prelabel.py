# -*- coding: utf-8 -*-
"""train.prelabel —— **用现有启发式给每一帧打初始标签**（人工只需要改错的那部分）。

## 为什么要预标注

`vision/cues.py` 那套判据是**在真录像上量出来的**（氮气红 0.36/0.09、
刹车圈内−圈外 0.37/0.11），它的**召回**其实不差 —— README 记的两个问题是
"氮气偏多"（假阳）和边界帧不稳，不是"什么都看不出来"。
所以从零手标几千帧是浪费：**先把启发式的判断写成标注，人工只复核/纠正**，
工作量差一个数量级。

## 但预标注**不能直接当训练集**（这是这个模块最要紧的一句话）

拿启发式当训练标签、再用它训模型，是**自己教自己**：模型只会学会复现
同一套阈值，一个误报都修不掉。真正让模型比启发式强的是**复核**这一步 ——
所以：

* 预标注出来的每一帧都带 `reason`（为什么被抽出来）和 `conf`（有多确信）；
* `train.review` 会**优先把"最可能标错"的帧排在最前面**：
  翻转边界帧、两通道吵架的帧、离阈值很近的帧（见 `frames.plausible_mistakes`）；
* 复核完的标注才是训练集。

## 信号怎么算的（和运行时**完全同一套代码**）

直接调 `cues.key_pressed()` —— 这样"预标注的判据"和"运行时启发式后端"
永远是同一个东西，不会出现"标注用的阈值和推理用的阈值不一样"这种鬼。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from a9route.train.labels import Box, CLASS_IDS


@dataclass
class KeySignals:
    """一帧上两个按键的**全部证据**（判据明细 + 最终布尔 + 复核用的置信度）。

    四个通道值都是 `cues.key_pressed()` 原样给的，没有二次加工 ——
    这样复核页上显示的数和 `a9route analyze` 打印的数是同一个数。
    """

    brake_ring: float = 0.0     #: 刹车「圈内 − 圈外」白度（实测最准的刹车判据）
    brake_white: float = 0.0    #: 刹车圈内**绝对**白度（兜底判据）
    nitro_red: float = 0.0      #: 氮气**红**占比（实测最准的氮气判据）
    nitro_white: float = 0.0    #: 氮气圈内绝对白度（**不用它判定**，只当"吵架检测"）
    brake_hit: bool = False
    nitro_hit: bool = False
    #: 归一化余量：`得分 / 阈值`。**>= 1 = 判按下**，越接近 1 越可疑。
    #: 这是复核排序的关键量（离 1 近的帧最可能标错）。
    brake_margin: float = 0.0
    nitro_margin: float = 0.0
    #: 两个通道判断不一致（已知误报的来源：「瓶子满面红但没按」这类）
    brake_disagree: bool = False
    nitro_disagree: bool = False
    #: **这一帧的标签是谁给的**：`heuristic` 或 `onnx:keys.onnx` 这种 ——
    #: 用户 2026-09-15 起要求"用模型推训练集"，所以来源必须记下来：
    #: 拿模型预标注出来的一批帧，和启发式预标注的**不能混为一谈**
    #: （前者是"模型的判断"，复核之后才变成"人的判断"）。
    by: str = "heuristic"
    #: 模型给的置信度（启发式后端是 0）
    brake_conf: float = 0.0
    nitro_conf: float = 0.0

    def boxes(self, brake_box, nitro_box) -> list[Box]:
        """这一帧的预标注框。**框就是配置里那个按键框本身**（位置固定已知）。

        `conf` 存的是归一化余量（不是模型置信度）—— 复核页上显示成
        "确信度 1.42" 这种，用来一眼看出哪些帧悬在阈值上。
        """
        out: list[Box] = []
        if self.brake_hit and brake_box:
            out.append(Box(CLASS_IDS["brake_pressed"], *[float(v) for v in brake_box],
                           conf=round(self.brake_margin, 3)))
        if self.nitro_hit and nitro_box:
            out.append(Box(CLASS_IDS["nitro_pressed"], *[float(v) for v in nitro_box],
                           conf=round(self.nitro_margin, 3)))
        return out

    def as_dict(self) -> dict:
        return {
            "brake_ring": round(self.brake_ring, 4),
            "brake_white": round(self.brake_white, 4),
            "nitro_red": round(self.nitro_red, 4),
            "nitro_white": round(self.nitro_white, 4),
            "brake_hit": self.brake_hit,
            "nitro_hit": self.nitro_hit,
            "brake_margin": round(self.brake_margin, 3),
            "nitro_margin": round(self.nitro_margin, 3),
            "brake_disagree": self.brake_disagree,
            "nitro_disagree": self.nitro_disagree,
            "by": self.by,
            "brake_conf": round(self.brake_conf, 3),
            "nitro_conf": round(self.nitro_conf, 3),
        }


def _ratio(num: float, den: float) -> float:
    """归一化余量（阈值 <=0 时退回 0，不让它除爆）。"""
    return float(num) / float(den) if den else 0.0


def read_signals(frame, *, coords: dict | None = None, detector=None) -> KeySignals:
    """一帧 -> `KeySignals`。**必须在 `config.apply()` 之后调用**
    （判据框/阈值是模块级常量，由 config 灌进去）。

    `detector`：给了就用**模型后端**（`vision.keys`：OnnxKeys / UltralyticsKeys）判
    "按下/没按"，并把来源记进 `sig.by`；没给就走 `cues.key_pressed()`（像素启发式）。

    ⚠️ 两种模式的**像素明细都照算**（ring/white/red）—— 它们**不参与标签**，
    只用于复核页显示和"哪一帧最可疑"的排序；标签只认 `brake_hit/nitro_hit`。
    """
    from a9route.vision import cues

    brake_box = (coords or {}).get("brake_key_box") or cues.BRAKE_KEY_BOX
    nitro_box = (coords or {}).get("nitro_key_box") or cues.NITRO_KEY_BOX

    hit_b, info_b = cues.key_pressed(frame, brake_box, "brake")
    hit_n, info_n = cues.key_pressed(frame, nitro_box, "nitro")

    sig = KeySignals(
        brake_ring=info_b["ring"], brake_white=info_b["white"],
        nitro_red=info_n["red"], nitro_white=info_n["white"],
        brake_hit=bool(hit_b), nitro_hit=bool(hit_n),
    )
    # 归一化余量：刹车取两个通道里**更靠得住**的那个（和 key_pressed 的
    # "任一超阈值即算按下"一致 —— 余量也必须按同一个并集算，否则排序会错）
    sig.brake_margin = max(_ratio(sig.brake_ring, cues.BRAKE_RING_THR),
                           _ratio(sig.brake_white, cues.BRAKE_WHITE_THR))
    sig.nitro_margin = _ratio(sig.nitro_red, cues.NITRO_RED_THR)
    # 吵架检测：两个通道给出不同结论 -> 这一帧最值得人看一眼
    sig.brake_disagree = ((sig.brake_ring > cues.BRAKE_RING_THR)
                          != (sig.brake_white > cues.BRAKE_WHITE_THR))
    sig.nitro_disagree = ((sig.nitro_red > cues.NITRO_RED_THR)
                          != (sig.nitro_white > cues.NITRO_RED_THR))
    if detector is not None:
        # ---- 用模型给标签（用户 2026-09-15 的用法：拿已训好的模型推新数据集）----
        r = detector.read(frame)
        sig.brake_hit = bool(r.brake)
        sig.nitro_hit = bool(r.nitro)
        sig.by = "{0}:{1}".format(getattr(detector, "name", "?"),
                                  Path(getattr(detector, "path", "") or "").name)
        ic = r.info or {}
        sig.brake_conf = float(ic.get("brake") or 0.0)
        sig.nitro_conf = float(ic.get("nitro") or 0.0)
        # **余量照样归一化到"1 = 阈值"**：模型没有"阈值余量"这个概念，用
        # `置信度 / 判定阈值` 代替 —— 于是"离 1 越近越可疑"这条排序规则不用改，
        # 复核页仍然是"先看模型没把握的帧"✓
        thr = float(getattr(detector, "conf", 0.0) or 0.0)
        sig.brake_margin = _ratio(sig.brake_conf, thr)
        sig.nitro_margin = _ratio(sig.nitro_conf, thr)
    return sig


def heuristic_boxes(frame, *, coords: dict | None = None) -> list[Box]:
    """只要框（不需要信号明细时用）。"""
    return read_signals(frame, coords=coords).boxes(
        (coords or {}).get("brake_key_box") or _cur("brake"),
        (coords or {}).get("nitro_key_box") or _cur("nitro"))


def _cur(which: str):
    from a9route.vision import cues
    return cues.BRAKE_KEY_BOX if which == "brake" else cues.NITRO_KEY_BOX


@dataclass
class KeyReaderConfig:
    """预标注用的框（从 `config.vision` 读，一般不用手填）。"""

    brake_box: tuple = ()
    nitro_box: tuple = ()
    extra: dict = field(default_factory=dict)
