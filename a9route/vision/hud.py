# -*- coding: utf-8 -*-
"""hud.py —— 从**一帧画面**里读比赛 HUD（**只读，不做任何操作**）。

> 这个文件是从原项目 `asphalt9auto` 的 `race/progress.py` 搬过来的，
> 只删掉了"去设备截屏"那条路径（本项目只处理视频文件），**读数逻辑一行没改**。

## 界面事实（用户提供的三张比赛内截图，1280×720）

```
┌───────────────────────────────────────────────────────────────┐
│ ┌──────────┐        ┌────进度条(氮气槽)────┐        KM/H 202   │
│ │ 排名 1/1 │                                       00:53.270   │
│ │ 路程 37% │   ← 这个百分比就是路线脚本里的"路程"              │
│ ├──────────┤                                                   │
│ │TOUCHDRIVE 开│      ◎ ◎ ◎   ← 选路图标（第 2 张图是 3 个）    │
│ └──────────┘                                                   │
│                                                    漂移78米    │
│   (小地图)                                                     │
│                                                                │
│                          (车)                                  │
│        ◉ 漂移/刹车                    🍶 氮气                  │
└───────────────────────────────────────────────────────────────┘
```

* **路程 `NN%`** 在左上角「路程」那一行 —— 这是路线脚本用的百分比来源；
* 「排名」在它上面；`TOUCHDRIVE 开/关` 在下面；
* **选路图标**出现在屏幕上方中间（1~4 个，当前选中的那个是蓝色高亮的）；
* 左边小地图、右边「漂移NN米 / 完美驾驶」只是状态提示，不参与判断。

## 为什么不用顶部那根进度条

三张图里那根条（黄/橙）在 22% / 34% / 37% 时长度并不成比例 —— 它更像**氮气槽**
（漂移时会被点亮填满）。所以百分比只认左上角的数字读数，那根条不用。

## 框的位置怎么改

下面这些框都是 **1280×720** 实测值，现在由 `config.json` 的 `hud` 段覆盖
（`a9route config set hud__progress_box=140,58,260,44`）。
所有录像必须是**同一分辨率**：换分辨率要重量一遍，别指望自适应。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

#: 「路程 NN%」的裁剪框（1280×720 实测：左上角那一行）
PROGRESS_BOX = (140, 58, 260, 44)
#: 「排名 N/M」的裁剪框（顺带读出来，方便日志/判断是否在比赛中）
RANK_BOX = (140, 20, 260, 40)
#: TOUCHDRIVE 开关文字（关自动驾驶操作要用）
TOUCHDRIVE_BOX = (20, 112, 300, 44)
#: 选路图标所在的一条横带（图标中心大致 y=130）
CHOICE_BAND = (400, 92, 480, 80)

#: 圆检测参数（与 cues.CHOOICE_* 分开：一个是"路标"、一个是"按键圆"）
CHOICE_PARAM1 = 120
CHOICE_PARAM2 = 28
CHOICE_PARAM2_FALLBACK = 22
CHOICE_MIN_DIST = 40
CHOICE_MIN_R = 22
CHOICE_MAX_R = 45
CHOICE_BLUE_RATIO = 0.35
#: 选"像一排路标"的容差：高度差 <= N px、半径在 0.6~1.7 倍中位数之间
CHOICE_ROW_TOL = 22
CHOICE_R_MIN_K = 0.6
CHOICE_R_MAX_K = 1.7
CHOICE_MAX_COUNT = 4

#: 比赛结束的判据：完赛界面上会有这些词
FINISH_WORDS = ("成绩", "继续", "再来一次", "重新开始", "奖励", "完成比赛")

_RE_PERCENT = re.compile(r"(\d{1,3})\s*%")
_RE_RANK = re.compile(r"(\d{1,2})\s*/\s*(\d{1,2})")


@dataclass
class RaceState:
    """一帧读到的东西。任何一个字段读不到就是 None。"""

    percent: float | None = None
    rank: str = ""
    touchdrive_on: bool | None = None
    choice_icons: int = 0            # 选路图标数量（0 = 现在没有选路）
    #: 选路图标的详细信息：[{"xy": (x, y), "blue": bool, ...}]（选路操作要照着点）
    icons: list = field(default_factory=list)
    raw: tuple[str, ...] = ()        # 原始 OCR 文本（排查用）

    @property
    def racing(self) -> bool:
        return self.percent is not None

    def describe(self) -> str:
        p = f"{self.percent:g}%" if self.percent is not None else "?"
        td = {True: "开", False: "关", None: "?"}[self.touchdrive_on]
        return (f"路程 {p}  排名 {self.rank or '?'}  TouchDrive {td}"
                f"  选路图标 {self.choice_icons or '无'}")


class RaceReader:
    """从一帧里读比赛状态。依赖可注入（离线可测）。"""

    def __init__(self, *, ocr=None, reader=None, grab=None, coords: dict | None = None,
                 choice=None):
        from a9route.ocr.reader import OcrReader
        self.ocr = ocr if ocr is not None else OcrReader()
        self._reader = reader            # (frame, box) -> list[str]，测试注入
        self._grab = grab
        #: 选路的**后端**（`vision.choice`：启发式 / ONNX 模型）。
        #: 给了就用它判路标 —— 这一处注入让**粗扫和细扫**都同时换了后端
        #: （两边都走 `choice_icons()`），不会出现"一处用模型、一处用圆检测"的分裂。
        self._choice = choice
        self.coords = dict(coords) if coords is not None else {}
        #: 用户/标定可以覆盖这几个框
        self.progress_box = tuple(self.coords.get("progress_box") or PROGRESS_BOX)
        self.rank_box = tuple(self.coords.get("rank_box") or RANK_BOX)
        self.touchdrive_box = tuple(self.coords.get("touchdrive_box") or TOUCHDRIVE_BOX)
        self.choice_band = tuple(self.coords.get("choice_band") or CHOICE_BAND)

    # ---------------------------------------------------------------- 基础
    def texts(self, frame, box) -> list[str]:
        if self._reader is not None:
            return list(self._reader(frame, box))
        x, y, w, h = box
        try:
            return [t for t, _ in self.ocr.read(frame[y:y + h, x:x + w])]
        except Exception:
            return []

    def grab(self):
        """取一帧画面。

        **本项目里不存在"抓帧"这回事** —— 视频分析的每一帧都由调用方传进来
        （`read(frame)`），所以这里只在**测试注入** `grab=` 时才有意义。
        原项目里这个方法会去 adb 截屏；独立出来后那条路径已被删除，
        忘了传 `frame` 会直接报错而不是偷偷去连手机。
        """
        if self._grab is not None:
            return self._grab()
        raise RuntimeError(
            "RaceReader.read(frame) 需要传一帧画面；"
            "本项目只做视频分析，不连设备截屏（测试可用 grab= 注入）")

    # ---------------------------------------------------------------- 解析
    @staticmethod
    def parse_percent(texts: list[str]) -> float | None:
        """从「路程 37%」这类文本里取百分比。**只认带 % 的数字**，避免把排名当路程。"""
        for t in texts:
            m = _RE_PERCENT.search(t)
            if m:
                return float(m.group(1))
        return None

    @staticmethod
    def parse_rank(texts: list[str]) -> str:
        for t in texts:
            m = _RE_RANK.search(t)
            if m:
                return f"{m.group(1)}/{m.group(2)}"
        return ""

    @staticmethod
    def parse_touchdrive(texts: list[str]) -> bool | None:
        joined = " ".join(texts)
        if "关" in joined:
            return False
        if "开" in joined:
            return True
        return None

    # ---------------------------------------------------------------- 选路图标
    def choice_icons(self, frame) -> list[dict]:
        """找屏幕上方那排**圆形选路路标**，返回按 x 排序的列表。

        每个元素：`{"xy": (中心x, 中心y), "radius": r, "blue": 是否蓝色高亮,
        "blue_ratio": 蓝色占比}`。

        用户给的规则（2026-09-12）：
        * 选路靠**点击屏幕上的圆形路标**（不是方向键 ✗）；
        * **蓝色高亮的那个 = 当前选中的路**；
        * 一屏可能有 **2 / 3 / 4 条**路可选 → 逐帧检测数量与位置，不写死。

        ## 两个后端

        `self._choice`（构造时注入）非空时**整段交给它** —— 那是 `vision.choice`
        的后端（启发式 / ONNX 模型 / ultralytics）。返回的形状要一致：
        `[{"xy": (x, y), "radius": r, "blue": bool, ...}, ...]`，
        这样调用方（`read()` / `count_choice_icons()` / 精细扫描）一行都不用改。
        **没注入就是原来那套 HoughCircles，逐字未改** ✓。

        ## 为什么用圆检测（HoughCircles）而不是"亮块分组"

        实测（真帧 `a9route/tests/fixtures/choice_band_2icons.png`，75% 处两个路标）：

        * 两个路标**几乎贴在一起**（左右各 ~70px 直径、间隔只有几像素）；
        * 未选中的那个是**黑底 + 细白圈 + 白箭头** —— 按"亮像素分组"会把它切成
          零件、还会把背景里的亮块当成路标（实测误报出 5 个 ✗）；
        * 换成 HoughCircles（param1=120 / param2=28 / 半径 22~45）后，
          在这一帧上**正好**找到 2 个圆：`(601,127) r=29 蓝占比 0.00`（未选中）与
          `(677,127) r=34 蓝占比 0.74`（当前选中）—— 与人眼判断完全一致 ✓。

        **参数都在 `config.json` 的 `vision` 段**（`choice_param2` 等），
        所以调这几个值不用改这个文件。
        """
        import cv2
        import numpy as np
        if self._choice is not None:
            # 后端接管（模型或启发式后端）：它返回的就是同一个形状
            try:
                return list(self._choice.read(frame).icons)
            except Exception:
                return []
        x, y, w, h = self.choice_band
        band = frame[y:y + h, x:x + w]
        if band.size == 0:
            return []
        gray = cv2.medianBlur(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY), 5)
        # 两遍：先严（28，实测 12 张采集帧零误报），找不到再用 22 兜一次
        # （实测有一帧两个路标都在、但 28 一个都没找到 ✗ —— 那一帧路标上叠了蓝光特效）。
        p2_main = int(self.coords.get("choice_param2") or CHOICE_PARAM2)
        p2_back = int(self.coords.get("choice_param2_fallback")
                      or CHOICE_PARAM2_FALLBACK)
        min_r = int(self.coords.get("choice_min_r") or CHOICE_MIN_R)
        max_r = int(self.coords.get("choice_max_r") or CHOICE_MAX_R)
        blue_ratio_thr = float(self.coords.get("choice_blue_ratio") or CHOICE_BLUE_RATIO)
        circles = None
        for p2 in (p2_main, p2_back):
            try:
                circles = cv2.HoughCircles(
                    gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=CHOICE_MIN_DIST,
                    param1=int(self.coords.get("choice_param1") or CHOICE_PARAM1),
                    param2=p2, minRadius=min_r, maxRadius=max_r)
            except Exception:
                circles = None
            if circles is not None and len(circles[0]):
                break
        if circles is None:
            return []
        c = band.astype(int)
        b, _g, r_ = c[:, :, 0], c[:, :, 1], c[:, :, 2]
        blue_mask = ((b - r_) > 30) & (b > 110)
        yy, xx = np.ogrid[:band.shape[0], :band.shape[1]]
        out: list[dict] = []
        for cx, cy, rad in np.round(circles[0]).astype(int):
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= (rad * 0.85) ** 2
            ratio = float(blue_mask[inside].mean()) if inside.any() else 0.0
            out.append({"xy": (x + int(cx), y + int(cy)), "radius": int(rad),
                        "blue": ratio > blue_ratio_thr, "blue_ratio": round(ratio, 2)})
        out.sort(key=lambda s: s["xy"][0])
        # ---- 只留"像一排路标"的那些圆（用户 2026-09-13 报过"实际只有 2 个却判成 3 个" ✗）----
        # 真路标是一排**大小相近、高度一致**的圆（实测 (601,127) r=29 与 (677,127) r=34）。
        # 用"最大的一组同高同大小的圆"当结果：先按 y 排序取中位，再筛掉半径差太多的。
        if len(out) >= 2:
            ys = sorted(s["xy"][1] for s in out)
            med_y = ys[len(ys) // 2]
            same_row = [s for s in out if abs(s["xy"][1] - med_y) <= CHOICE_ROW_TOL]
            if len(same_row) >= 2:
                med_r = sorted(s["radius"] for s in same_row)[len(same_row) // 2]
                same_row = [s for s in same_row
                            if CHOICE_R_MIN_K * med_r <= s["radius"] <= CHOICE_R_MAX_K * med_r]
                out = same_row
            # 剩下还超过 4 个 -> 只留最靠近中位 y 的那几个（一屏最多 4 条路 ✓）
            if len(out) > CHOICE_MAX_COUNT:
                out = sorted(out, key=lambda s: abs(s["xy"][1] - med_y))[:CHOICE_MAX_COUNT]
            out.sort(key=lambda s: s["xy"][0])
        return out

    def count_choice_icons(self, frame) -> int:
        """选路图标的**个数**（= `choice_icons()` 的长度；只在比赛中有意义）。"""
        try:
            return len(self.choice_icons(frame))
        except Exception:
            return 0

    # ---------------------------------------------------------------- 一帧
    def read(self, frame=None, *, with_percent: bool = True,
             with_icons: bool = True) -> RaceState:
        """读一帧。

        * `with_percent=False`：跳过「路程 NN%」那一次 OCR（省时间）；
        * `with_icons=False`：跳过选路路标那一次检测 —— 给**根本不用图标**的
          调用方用（比如 `video.sample_video` 只要时间轴），
          否则会白跑一遍检测（还可能是启发式的 HoughCircles）。
        """
        frame = self.grab() if frame is None else frame
        st = RaceState()
        raw: list[str] = []
        if with_percent:
            t = self.texts(frame, self.progress_box)
            raw += t
            st.percent = self.parse_percent(t)
        r = self.texts(frame, self.rank_box)
        raw += r
        st.rank = self.parse_rank(r)
        td = self.texts(frame, self.touchdrive_box)
        raw += td
        st.touchdrive_on = self.parse_touchdrive(td)
        try:
            # **只在"确认在比赛中"时数图标**（读得到路程百分比才算）——
            # 完赛界面/菜单上的一条亮横幅会被误计成 1 个（实测踩过）。
            # 所以 with_percent=False 时也不数（那种情况下我们不知道在不在比赛里）。
            if with_icons and st.percent is not None:
                st.icons = self.choice_icons(frame)
                st.choice_icons = len(st.icons)
            else:
                st.icons, st.choice_icons = [], 0
        except Exception:
            st.icons, st.choice_icons = [], 0
        st.raw = tuple(raw)
        return st
