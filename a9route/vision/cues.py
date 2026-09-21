# -*- coding: utf-8 -*-
"""cues.py —— 从**一帧画面**判断"现在正在做什么操作"（视频转路线用的判据）。

## 判据怎么来的（都在真实录像上量过，`output/跑图转路线测试.mp4`）

| 操作 | 判据 | 实测 |
|---|---|---|
| **氮气** | 顶部氮气槽左段是**青色**（喷氮）还是**黄色**（不喷） | 两者**互斥**：青色 0.14~0.43 时黄色 0.00，反之亦然 —— 干脆用"青色占比 > 0.10" ✓ |
| **漂移** | 右侧状态区出现「**漂移NN米**」 | 用户截图 + 录像里都能读到 ✓（车尾烟雾/刹车图标都不如它稳） |
| **选路** | 屏幕上方那排**圆形路标**（HoughCircles） | 录像里 27-28s 出 2 个、39-40s 出 3 个 ✓，蓝色高亮那个=当前选中 ✓ |
| **360** | 右侧状态区出现「**完成360度旋转**」 | 游戏自己的提示，最硬 ✓（车身姿态检测太难） |
| **关自动驾驶** | 左上角「TOUCHDRIVE **关**」 | 实时流程里已经在用同一个读数 ✓ |

颜色判据都在**固定 UI 位置**上量（不依赖车身位置）—— 车会跑、UI 不动，这是最稳的做法。
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: 顶部氮气槽左段（喷氮时青色，平时黄色）
GAUGE_BOX = (355, 22, 300, 34)
#: 右侧状态区（「漂移NN米」/「完成360度旋转」在这里）
STATUS_BOX = (1000, 230, 280, 120)
#: 左上角 TouchDrive 那行
TOUCHDRIVE_BOX = (24, 96, 260, 40)
#: 青色判定的阈值（实测喷氮 0.14~0.43、不喷 0.00）
CYAN_THR = 0.10
#: 右下**氮气键**：一个**圆**（瓶子只是画在圆里的图形，本身半透明 ✗）。
#: 用户 2026-09-13："你最好框一个圆，实在不行就把整个圆用正方形框进去" ✓。
#: 圆心用圆检测量出来 = **(1080, 552)**、半径 ≈50 —— 与刹车键 (200,552) 关于屏幕中线
#: (x=640) **完全对称** ✓（HUD 布局如此）。
NITRO_KEY_BOX = (1033, 497, 110, 110)
#: 左下**刹车键**：整个圆盘（含里面的图形）。圆心实测 (200,552)、半径 ≈50；
#: 原来 (170,540,70,70) 只压到左上弧 ✗。左右两个框严格对称：(137+110)=247，1280-247=1033 ✓
BRAKE_KEY_BOX = (137, 497, 110, 110)
#: 刹车键：**圈内相对白度** > 0.35（实测 按下 0.37 / 没按 0.11 ✓✓）；或绝对白度 > 0.35
#: （绝对白度实测：截图 按下 0.643 / 没按 0.175；标定视频窗口内 0.268、氮气段 0.049）
BRAKE_RING_THR = 0.35
BRAKE_WHITE_THR = 0.35
#: 氮气键：**红** > 0.15（实测 0.36 / 0.09 ✓✓）；或圈内相对白度 > 0.30
NITRO_RED_THR = 0.15
NITRO_WHITE_THR = 0.25
NITRO_RING_THR = 0.30
#: 「圈内 − 圈外」那个参照环的宽度（像素）与"白"的判定阈值
RING_PAD = 26
WHITE_THR = 165
#: 绝对白度（bright_ratio）的默认阈值
BRIGHT_THR = 165
#: `CueDetector` 的默认框/阈值 —— 由 `a9route.config.apply()` 灌进来，
#: 这样 config.json 改了立刻就生效（不用改这个文件）。
COORDS: dict = {}


def red_ratio(frame, box=None) -> float:
    """框里"红/粉"像素占比 —— 两个按键**按下时都会变红**（瓶子 / 刹车环）。"""
    x, y, w, h = box or NITRO_KEY_BOX
    c = frame[y:y + h, x:x + w].astype(int)
    r, g, b = c[:, :, 2], c[:, :, 1], c[:, :, 0]
    return float(((r > 130) & (r - g > 40) & (r - b > 35)).mean())


def brake_pressed(frame, box=None) -> float:
    """刹车键是否按下：**亮白占比**（按下时图标变"部分白色不透明"）。"""
    return bright_ratio(frame, box or BRAKE_KEY_BOX)


def ring_score(frame, box=None, *, pad: int | None = None, thr: int | None = None) -> float:
    """**圈内白度 − 圈外参照白度** —— 差分掉背景的相对判据。

    为什么需要它：图标是**半透明叠加**在场景上的（用户提醒），所以"圈内白度"的绝对值
    会被赛道明暗带着走 ✗。圈外那一圈背景和圈内几乎一样，减掉之后剩下的
    "图标自己变亮了没有"就与场景无关 ✓ —— 实测这是刹车段最准的判据
    （阈值 0.53: 按下 0.37 / 没按 0.11 ✓✓）。
    """
    x, y, w, h = box or BRAKE_KEY_BOX
    pad = RING_PAD if pad is None else int(pad)
    thr = WHITE_THR if thr is None else int(thr)
    inner = float((frame[y:y + h, x:x + w].min(axis=2) > thr).mean())
    ox, oy = x - pad, y - pad
    ow, oh = w + 2 * pad, h + 2 * pad
    band = frame[oy:oy + oh, ox:ox + ow]
    inside = frame[y:y + h, x:x + w]
    band_white = float((band.min(axis=2) > thr).sum()) - float((inside.min(axis=2) > thr).sum())
    band_area = ow * oh - w * h
    return inner - band_white / max(1, band_area)


def key_pressed(frame, box, kind: str) -> tuple[bool, dict]:
    """按键是否被按下 —— **三种证据取并集**（用户 2026-09-13 让三种都试）。

    实测对比（三条正样本：标定视频刹车段/氮气段 + 测试3 的 17%）：

    | 判据 | 刹车段 | 氮气段 | 测试3@17% |
    |---|---|---|---|
    | 模板匹配（点亮模板） | -0.106 ✗ | -0.171 ✗ | +0.039 |
    | 模板差分（点亮−未点亮） | +0.012 ✗ | +0.021 ✗ | -0.083 ✗ |
    | 亮白不透明 | +0.115 ✓ | +0.067 | -0.061 ✗ |
    | 红 | +0.053 | **+0.150** ✓✓ | -0.012 ✗ |
    | **圈内−圈外** | **+0.123** ✓✓ | +0.063 | -0.039 ✗ |

    -> 模板法最差（半透明叠加，模板主要在匹配背景 ✗）；
       **刹车以"圈内−圈外"最准**、**氮气以"红"最准**，所以两个都收，任一超阈值即算按下 ✓。
    返回 (是否按下, 各指标明细)。
    """
    ring = ring_score(frame, box)
    red = red_ratio(frame, box)
    white = bright_ratio(frame, box)
    info = {"ring": round(ring, 3), "red": round(red, 3), "white": round(white, 3)}
    if kind == "brake":
        # 刹车：**圈内−圈外**最准（0.37 / 0.11 ✓✓），保留绝对白度当兜底
        hit = ring > BRAKE_RING_THR or white > BRAKE_WHITE_THR
    else:
        # 氮气：**红**最准（0.36 / 0.09 ✓✓）。
        # 实测（三条样本）：
        #  * 圈内−圈外**对氮气没信号**（测试3 按下 0.004 ✗），并进来只会增加误报 ✗；
        #  * 而用户报的"17% 那两次点击"红占比正好 **0.151 > 0.15** ✓ ——
        #    当初漏掉是因为**去抖 hold=2 把只有 1 帧的点击抹了** ✗，不是阈值 ✗；
        #  * 所以氮气只认红 >0.15，去抖放到 1 帧（见 video.scan_fine 的 hold/hold_brake）✓。
        hit = red > NITRO_RED_THR
    info["hit"] = hit
    return hit, info


def nitro_pressed(frame, box=None) -> float:
    """氮气键"按下强度"（红 或 圈内相对白度，取较大者）—— 供旧代码/调试用。"""
    b = box or NITRO_KEY_BOX
    return max(red_ratio(frame, b), ring_score(frame, b))


def bright_ratio(frame, box=None, thr: int | None = None) -> float:
    """框里"亮"像素占比 —— 按下时图标里的图形也会变亮（备用判据）。"""
    x, y, w, h = box or BRAKE_KEY_BOX
    c = frame[y:y + h, x:x + w].astype(int)
    return float((c.min(axis=2) > (BRIGHT_THR if thr is None else thr)).mean())


def cyan_ratio(frame, box=GAUGE_BOX) -> float:
    """框里"亮青色"像素占比（喷氮时氮气槽就是青的）。"""
    x, y, w, h = box
    c = frame[y:y + h, x:x + w].astype(int)
    b, g, r = c[:, :, 0], c[:, :, 1], c[:, :, 2]
    m = (b > 140) & (g > 140) & (r < 140) & (b - r > 40)
    return float(m.mean())


@dataclass
class Cue:
    """一帧里检测到的状态。

    用户 2026-09-13 定的规则：**看按键亮没亮**（而不是看尾焰/文字这类"效果"）。
    所以这里两个主信号是：
      * `nitro_lit`：右下**氮气键被按下**（红 或 亮白不透明，见 `nitro_pressed()`）；
      * `brake_lit`：左下**刹车键变亮**。
    """

    nitro: bool = False          # 氮气**正在喷**（顶部槽变青）
    nitro_lit: bool = False      # 同上（语义别名）
    nitro_pressed: bool = False  # **玩家正在点氮气键**（瓶子变红；没氮气时也会点）
    brake_lit: bool = False      # **玩家正在按住刹车键**（图标变红 —— 用户两张图标定）
    drifting: bool = False       # 漂移（= 刹车键按下，兜底看「漂移NN米」）
    spin360: bool = False
    autopilot_off: bool = False
    #: 选路路标（`{"xy": (x,y), "blue": bool}`），空 = 现在没有岔路口
    icons: list = field(default_factory=list)
    #: 调试用
    cyan: float = 0.0
    nitro_red: float = 0.0
    nitro_ring: float = 0.0
    brake_red: float = 0.0
    brake_ring: float = 0.0
    brake_bright: float = 0.0
    status_text: str = ""

    @property
    def choice(self) -> tuple[int, int] | None:
        """选路：返回 (路标数量, 当前蓝色高亮是第几个 1 起)；没有路标返回 None。"""
        if not self.icons:
            return None
        ordered = sorted(self.icons, key=lambda s: s["xy"][0])
        idx = next((i + 1 for i, s in enumerate(ordered) if s.get("blue")), 0)
        return (len(ordered), idx or 1)

    def describe(self) -> str:
        bits = []
        if self.nitro_pressed:
            bits.append("氮气键按")
        elif self.nitro:
            # 顶部槽变青只说明"氮气在喷"，**不等于玩家在按键** ✗
            #（用户 2026-09-13 报过：70% 之后连续误判"氮气键按下"，其实只是槽是青的）
            bits.append("氮气槽青(在喷)")
        if self.brake_lit:
            bits.append("刹车键按")
        elif self.drifting:
            bits.append("漂移提示")
        if self.spin360:
            bits.append("360")
        if self.autopilot_off:
            bits.append("关自动驾驶")
        if self.icons:
            n, idx = self.choice or (0, 0)
            bits.append(f"选路{n}选{idx}")
        return " + ".join(bits) or "（无）"


class CueDetector:
    """把一帧变成 `Cue`。依赖可注入（离线测试用假 reader/reader_texts）。

    `key_detector`：可选的**按键后端**（`vision.keys`）。给了就用它判
    `brake_lit` / `nitro_pressed`，没给就沿用下面那套内置判据 ——
    这样"换 YOLOv8 模型"这件事在**粗扫和精细扫描两边是同一个开关**
    （`config.vision.key_backend`），不会出现"粗扫用模型、细扫用阈值"的分裂。

    `choice`：可选的**选路后端**（`vision.choice`），同理 —— 给了就用它数路标。
    ⚠️ 这个口子是**补上的**（2026-09-15）：以前这里自己造一个裸 `RaceReader()`，
    于是粗扫的 `cue.icons` **一直是 HoughCircles**，而细扫早就换成模型了 ✗✗。
    后果不只是"白跑一遍启发式"：`cue.icons` 还被用来做**交叉校验**
    （粗扫也看到 ≥2 个路标才认这次选路），**启发式因此能一票否掉模型判出来的选路**。
    """

    def __init__(self, *, reader=None, reader_texts=None, ocr=None,
                 coords: dict | None = None, drift_from_key: bool = False,
                 key_detector=None, choice=None):
        self.reader = reader
        self._reader_texts = reader_texts
        self._ocr = ocr
        self.coords = dict(coords or {})
        #: True = 漂移完全按"刹车键亮不亮"判（需要先标定刹车键；见类文档）
        self.drift_from_key = bool(drift_from_key)
        #: 按键后端（None = 用内置启发式；见类文档）
        self.key_detector = key_detector
        #: 选路后端（None = 内置 HoughCircles；见类文档）
        self.choice = choice
        if reader is not None and choice is not None:
            # 注入的 reader 自带后端，这里再给一个就会**两边不一致** —— 说出来，
            # 别让"以为注入了、其实没生效"再发生一次
            raise ValueError("CueDetector 同时给了 reader 和 choice："
                             "reader 自带后端，两者会打架；只用其中一个")

    def _percent_and_icons(self, frame):
        if self.reader is None:
            from a9route.vision.hud import RaceReader
            # **选路后端一起注入**：粗扫和细扫必须是同一个后端（见类文档）
            self.reader = RaceReader(choice=self.choice)
        st = self.reader.read(frame)
        pct = getattr(st, "percent", None)
        icons = list(getattr(st, "icons", []) or [])
        return pct, icons, st

    def _texts(self, frame, box):
        """读一块区域的文字（注入的优先，其次用 OCR）。"""
        if self._reader_texts is not None:
            return list(self._reader_texts(frame, box))
        if self._ocr is None:
            from a9route.ocr.reader import OcrReader
            self._ocr = OcrReader()
        x, y, w, h = box
        try:
            return [t for t, _ in self._ocr.read(frame[y:y + h, x:x + w])]
        except Exception:
            return []

    def _c(self, key: str, fallback):
        """取一个框/阈值：**构造时注入的** > `config.apply()` 灌进 `COORDS` 的** > 内置默认。

        （这样 `config.json` 改了就生效，同时老的"显式 coords="用法和离线测试都不受影响。）
        """
        val = self.coords.get(key)
        if val is None:
            val = COORDS.get(key)
        return fallback if val is None else val

    def detect(self, frame) -> tuple[Cue, float | None]:
        """返回 (状态, 这一帧读到的路程百分比)。"""
        pct, icons, st = self._percent_and_icons(frame)
        cue = Cue(icons=icons)
        cue.cyan = round(cyan_ratio(frame, self._c("gauge_box", GAUGE_BOX)), 3)
        cue.nitro = cue.cyan > float(self._c("cyan_thr", CYAN_THR))
        # ---- 两个**按键**：三种证据取并集（详见 key_pressed() 的实测对比表）----
        cue.brake_bright = round(bright_ratio(frame, self._c("brake_key_box",
                                                             BRAKE_KEY_BOX)), 3)
        if self.key_detector is not None:
            # 外部后端（YOLOv8 模型）—— 粗扫/细扫共用同一个判断
            read = self.key_detector.read(frame)
            cue.brake_lit = bool(read.brake)
            cue.nitro_pressed = bool(read.nitro)
            cue.brake_red = float(read.info.get("red", 0.0) or 0.0)
            cue.brake_ring = float(read.info.get("ring", 0.0) or 0.0)
            cue.nitro_red = float(read.info.get("red", 0.0) or 0.0)
            cue.nitro_ring = float(read.info.get("ring", 0.0) or 0.0)
        else:
            hit_b, info_b = key_pressed(frame, self._c("brake_key_box", BRAKE_KEY_BOX),
                                       "brake")
            cue.brake_lit = hit_b
            cue.brake_red = info_b["red"]
            cue.brake_ring = info_b["ring"]
            hit_n, info_n = key_pressed(frame, self._c("nitro_key_box", NITRO_KEY_BOX),
                                       "nitro")
            cue.nitro_pressed = hit_n
            cue.nitro_red = info_n["red"]
            cue.nitro_ring = info_n["ring"]
        cue.nitro_lit = cue.nitro_pressed
        status = " ".join(self._texts(frame, self._c("status_box", STATUS_BOX)))
        cue.status_text = status
        # 漂移：**看刹车键有没有按下**（用户标定过的判据）；文字提示只当兜底
        #（实测「漂移NN米」是累计里程、会滞留，所以不能只看它 ✗）
        # ⚠️ 有按键模型时**只认模型**：用户的规则是"漂移 = 按住刹车"，
        #    文字那一路（OCR 到「漂移」）会额外加进来，等于在模型之外多一条判定 ✗
        #    （2026-09-15 按"判定全交给模型"的要求收紧；没模型时才用文字兜底。）
        if self.key_detector is not None:
            cue.drifting = bool(cue.brake_lit)
        else:
            cue.drifting = bool(cue.brake_lit) or (
                ("漂移" in status) and self.drift_from_key is not True)
        # 只认完整的「完成360度旋转」提示 —— 裸 "360" 会误命中（实测把 9%~19% 连报 11 次 ✗）
        cue.spin360 = "360度" in status or "完成360" in status
        on = getattr(st, "touchdrive_on", None)
        # 自动驾驶状态**自己查一遍文字**：只看 reader 的结论会误报（实测 3%/18% 假阳 ✗），
        # 必须"确实读到 TOUCHDRIVE 这一行、且有关无开"才算关。
        td = " ".join(self._texts(frame, self._c("touchdrive_box", TOUCHDRIVE_BOX)))
        if "TOUCHDRIVE" in td.upper() or "TOUCHDRIVE" in status.upper():
            if "关" in td and "开" not in td:
                on = False
            elif "开" in td:
                on = True
        cue.autopilot_off = (on is False) and ("TOUCHDRIVE" in (td + status).upper())
        return cue, pct
