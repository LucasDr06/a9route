# -*- coding: utf-8 -*-
"""config.py —— **所有可调参数集中在这一个文件（+ config.json）里**。

## 为什么要有这一层

这条线原来是靠"改代码里的常量、然后跑整个手机自动化项目的回归"来调参的，
太慢。现在所有阈值/框位置/采样间隔都能从**一个 JSON** 覆盖：

* 优先级：**环境变量** > `config.json` > 下面的 `DEFAULTS`（代码内置值）；
* 环境变量写成 `A9ROUTE_<段>__<键>`，双下划线分段、单下划线是键名的一部分，
  例如 `A9ROUTE_vision__nitro_red_thr=0.22`、`A9ROUTE_scan__every=0.2`；
* `a9route config list / set / reset` 直接读写 `config.json`；
* `a9route analyze --set vision__nitro_red_thr=0.22` 只影响这一次运行。

**默认值全部是原项目实测出来的**（三条正样本 + 用户标定视频），
每个都标了实测数据，**不要凭感觉改**。已知不准的两处见 README「已知不准」。

## 谁在用这些值

`apply()` 会把配置灌进 `a9route.vision` 的模块级常量
（`cues.BRAKE_RING_THR` / `progress.PROGRESS_BOX` / `scan_fine` 的 `hold` 等），
所以**任何入口（CLI / Web / 测试）都必须先调一次 `apply()`**（`a9route.cli` 已代劳）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from a9route import paths

ENV_PREFIX = "A9ROUTE_"

#: 最近一次 `apply(cfg)` 用的那份配置（见 `current()`）。
#: ⚠️ 别拿 `load_config()` 当"当前配置"用 —— 它**只读磁盘**，
#: 会把 `analyze --set ...` / `analyze(overrides=...)` 这类本次覆盖漏掉。
_CURRENT: dict | None = None

DEFAULTS: dict[str, dict] = {
    # ---------------------------------------------------------------- 视频分析
    "scan": {
        #: 粗扫抽帧间隔（秒）。0.25 = 每秒 4 帧，实测能覆盖 100 个整百分点且不漏
        "every": 0.25,
        #: 只分析前 N 秒（试跑用）；0 = 不限
        "max_seconds": 0.0,
        #: 并行度：>1 时给粗扫加"提前解码线程"，并把按键细扫放另一个线程（实测 -26% 耗时）
        "workers": 2,
        #: 是否做"按键细扫"（逐帧看刹车键/氮气键 + 选路路标）。
        #: **关掉会退回"窗口汇总"那一路估算**（整数时长，不准），默认必须开
        "with_buttons": True,
        #: 精细扫描浮点比较用的容差
        "gap": 0.12,
        #: 去抖**分通道**：刹车是长按、要连续 2 帧；氮气**允许单帧**
        #: （实测用户 17% 那两次点击在 59 帧里只亮 2 帧 = 每次 1 帧 33ms，
        #:   写成"连续 2 帧"会把真实点击整段抹掉 ✗✗）
        "hold": 1,
        "hold_brake": 2,
        #: 精细扫描里每几帧看一次选路路标
        "icon_every": 2,
        #: 选路稳定性闸：同一个值连续这么多帧才认（真岔路口持续好几秒，
        #: 景色闪光只闪一两帧 —— 不加这道闸，10 秒视频能报出 16 次选路 ✗）
        "choice_min_hold": 4,
        #: **「选路结束」的阈值**：连续这么多个采样点都没检测到路标，
        #: 就认为这一段选路结束了（回到"选路结束状态"，下一次检测到就开新的一段）。
        #: 它决定"挨得近的两个岔路口会不会被并成一段"：
        #: 太短 -> 路标闪一下就把一段拆成两段（多报选路）；太长 -> 并成一段（少报选路）。
        #: 采样间隔由 `every`/`icon_every` 决定（默认 ≈67ms 一次），所以 8 ≈ 0.53 秒。
        "choice_idle_hold": 8,
        #: 选路**交叉校验**的容差（个百分点）：精细扫描说这里是岔路口，
        #: 粗扫也必须在 ±这个范围内看到 >=2 个路标才算（两边都说是，才认）✓
        "choice_cross_tol": 1,
    },

    # ---------------------------------------------------------------- 判据：按键
    "vision": {
        #: 右下**氮气键**：整个圆（圆心实测 (1080,552) r≈50）。
        #: 只框瓶子左上角会大半是背景 ✗（画出来才发现）
        "nitro_key_box": [1033, 497, 110, 110],
        #: 左下**刹车键**：整个圆盘（圆心实测 (200,552) r≈50）；
        #: 与氮气键关于屏幕中线 x=640 完全对称：1280-(137+110)=1033
        "brake_key_box": [137, 497, 110, 110],
        #: 刹车按下：**圈内白度 − 圈外参照白度** > 这个值
        #: （实测 按下 0.37 / 没按 0.11 ✓✓ —— 半透明图标必须用相对量，绝对值会被赛道明暗带走）
        "brake_ring_thr": 0.35,
        #: 刹车备用的**绝对白度**阈值（按下 0.643 / 没按 0.175）
        "brake_white_thr": 0.35,
        #: 氮气按下：框内**红占比** > 这个值（实测 0.36 / 0.09 ✓✓）
        "nitro_red_thr": 0.15,
        #: 圈外参照环的宽度（像素）与"白"的判定阈值
        "ring_pad": 26,
        "white_thr": 165,
        "bright_thr": 150,

        #: 顶部氮气**槽**左段（喷氮时青色、平时黄色）—— 只用来描述"在喷"，
        #: **不能**当"玩家在按键"用（用户 2026-09-13 报过 70% 之后连续误判）
        "gauge_box": [355, 22, 300, 34],
        "cyan_thr": 0.10,
        #: 右侧状态区（「漂移NN米」/「完成360度旋转」/「完美氮气」都在这里）
        "status_box": [1000, 230, 280, 120],
        #: 左上角 TouchDrive 那行
        "touchdrive_box": [24, 96, 260, 40],

        #: 选路模型的运行后端（和按键那套同样的规则，只是**另一套类别**）：
        #:   auto         **默认**：`models/choice.onnx` 在就用模型，不在就退回启发式
        #:   onnx / ultralytics / heuristic
        #: 换模型：`a9route train models` / `a9route train use <名字>`
        "choice_backend": "auto",
        #: 选路模型文件（空 = 用 models/choice.onnx）
        "choice_model": "",
        #: 选路模型的推理分辨率（留默认值时会跟随模型清单里记的 imgsz）
        "choice_imgsz": 640,
        #: 选路模型的置信度阈值
        #: （0.30 = `train eval` 在 val 上扫出来的最佳档，和 `models/choice.json`
        #:   里记的 `conf_default` 是同一个数 —— 两边必须一致，否则 clone 下来的
        #:   人用的是没调过的 0.40，而文档写 0.30 ✗）
        "choice_conf": 0.30,
        #: 选路模型的 NMS IoU
        "choice_iou": 0.50,
        #: 选路模型的 onnxruntime 执行后端：auto / CPUExecutionProvider / …
        "choice_provider": "auto",
        #: 选路模型的 ONNX 输出形态（同 key_fmt，2 类时行宽有歧义）
        "choice_fmt": "auto",
        #: 选路：**最少几个路标才算岔路口**（用户规则：至少 2 个）
        "choice_min_options": 2,
        #: 选路：最多几条路（用户规则：一屏只有 2/3/4 条）
        "choice_max_options": 4,

        #: 选路路标检测（HoughCircles）：两遍，先严后松
        "choice_param2": 28,
        "choice_param2_fallback": 22,
        "choice_param1": 120,
        "choice_min_r": 22,
        "choice_max_r": 45,
        #: 蓝色高亮判定：圆内蓝色占比 > 这个值
        "choice_blue_ratio": 0.35,

        # ---------------------------------------------------------- 按键模型（YOLOv8）
        #: 按键识别后端：
        #:   auto         **默认**：`models/keys.onnx` 在就用模型，不在就退回启发式
        #:                （退回时在 analyze 的日志里**明说**一句话，不静默）
        #:   onnx         强制用模型（文件不在就报错，不偷偷降级）
        #:   ultralytics  直接用 .pt（要 torch；适合边训边试）
        #:   heuristic    只用 cv2/numpy 的启发式（和最初逐字一致）
        #: 换模型：`a9route train models` 看有什么，`a9route train use <名字>` 切过去
        #: （切换时会**核对模型清单里的类别顺序**，类序反了直接拒绝启动）。
        "key_backend": "auto",
        #: 模型文件（空 = 用 models/keys.onnx）。`.onnx` 走 onnxruntime、`.pt` 走 ultralytics
        "key_model": "",
        #: 推理分辨率。**留默认值(640)时会跟随模型清单里记的 imgsz** ——
        #: 只有你显式改过这个值才强制用它（避免"导出 640、推理别的值"的静默错位）
        "key_imgsz": 640,
        #: 置信度阈值。**别调太低** —— 这个项目的已知问题是"氮气误报偏多"
        "key_conf": 0.40,
        #: NMS 的 IoU 阈值
        "key_iou": 0.50,
        #: onnxruntime 的执行后端：auto / CPUExecutionProvider / CUDAExecutionProvider
        "key_provider": "auto",
        #: 定位闸（0 = 关）：检测框中心离"配置里那个按键位置"超过
        #: 这个值 × 画面宽度 就当没看见 —— 防模型在加载/结算画面乱响
        "key_anchor_tol": 0.0,
        #: **ONNX 输出形态**：auto / raw / nms
        #: 我们只有 2 类 -> `4+nc == 6`，而"已做 NMS"的导出行宽**也是 6**，
        #: 光看形状分不出来 ✗（猜错不报错，只会静默把坐标当分数用）。
        #: `auto` 按"未做 NMS"解析（`a9route train export` 导出的就是这种，
        #: 我们**不传 nms=True**）；你要用 `nms=True` 导过，就设成 `nms`。
        "key_fmt": "auto",
        #: **启发式后端**用哪套口径（只有在 `key_backend=heuristic` 时有意义）：
        #:   fine  逐字保持现状 —— `scan_fine` 一直在用的那两行
        #:         （刹车 = 绝对白度 > brake_white_thr；氮气 = max(红, 圈内外) > nitro_red_thr）
        #:   cues  和 `cues.key_pressed()` 统一（刹车 = 圈内外 ∪ 白度；氮气 = **只认红**）
        #: ⚠️ 两者**不一样**，而且默认的 fine 口径把"圈内−圈外"并进了氮气 ——
        #: 可 NOTES §1 实测的结论是"圈内判据对氮气没信号（按下 0.004），
        #: 并进来只会把误报抬高 ✗"。README 里「氮气偏多」那条已知问题，
        #: 至少有一部分是这儿来的。**默认不改**（路线输出一个字节不变），
        #: 想验证就把这个设成 cues，再用 `a9route train pulses` 对比动作数。
        "key_heuristic_mode": "fine",
    },

    # ---------------------------------------------------------------- 比赛内 HUD 读数
    "hud": {
        #: 「路程 NN%」—— **路线百分比只认它**（不要用顶部那根条：
        #: 三张图里 22% / 34% / 37% 时长度不成比例，那更像氮气槽）
        "progress_box": [140, 58, 260, 44],
        #: 「排名 N/M」（顺带记录）
        "rank_box": [140, 20, 260, 40],
        #: TOUCHDRIVE 开/关
        "touchdrive_box": [20, 112, 300, 44],
        #: 选路图标所在的一条横带（图标中心大致 y=130）
        "choice_band": [400, 92, 480, 80],
    },

    # ---------------------------------------------------------------- 判据：操作目的
    "intent": {
        #: 刹车**成对脉冲**间隔 < 这个值 -> 判 360
        #: （用户 2026-09-13 从 0.20 放宽到 0.30：实测双击偶尔慢一点，
        #:   0.20 太紧会把 360 判成"独立脉冲（打断氮气）"✗）
        "tap_360_gap": 0.30,
        #: 刹车单发持续 >= 这个值 -> 按住漂移；否则算"独立短脉冲（打断氮气）"
        "drift_min": 0.30,
        #: 氮气脉冲聚成"一次连点"的时间窗
        "nitro_gap": 0.25,
        #: 氮气长按判定（>= 这个时长按"快速连点"写 N:0:k:100）
        "nitro_hold": 0.60,
    },
}


# ---------------------------------------------------------------- 读写
def load_config(path: Path | None = None) -> dict:
    """读 config.json 并盖在 DEFAULTS 上（深一层合并）。坏文件当空配置，不崩。"""
    out = {section: dict(vals) for section, vals in DEFAULTS.items()}
    f = Path(path) if path else paths.CONFIG_FILE
    if f.is_file():
        try:
            data = json.loads(f.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
        for section, vals in (data or {}).items():
            if isinstance(vals, dict):
                out.setdefault(section, {}).update(vals)
    return out


def save_config(cfg: dict, path: Path | None = None) -> Path:
    f = Path(path) if path else paths.CONFIG_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
    return f


def flat(cfg: dict, *, env: bool = True) -> dict[str, object]:
    """`{段__键: 值}` 的扁平视图（环境变量也并进来）。"""
    out: dict[str, object] = {}
    for section, vals in cfg.items():
        for key, val in vals.items():
            out[f"{section}__{key}"] = val
    if env:
        for name, raw in os.environ.items():
            if not name.startswith(ENV_PREFIX):
                continue
            key = name[len(ENV_PREFIX):]
            if key.lower() in ("no_console_setup", "video_dir", "config", "profile"):
                continue
            cur = out.get(key)
            out[key] = _coerce(raw, cur)
    return out


def get(cfg: dict, key: str, default=None):
    """取一个扁平键（`段__键`，也接受 `段.键`）。支持环境变量覆盖。"""
    key = key.replace(".", "__")
    env = os.environ.get(ENV_PREFIX + key)
    if env is not None and env != "":
        cur = default
        section, _, name = key.partition("__")
        if isinstance(cfg, dict) and section in cfg and name in cfg[section]:
            cur = cfg[section][name]
        return _coerce(env, cur)
    section, _, name = key.partition("__")
    if isinstance(cfg, dict) and section in cfg and name in cfg[section]:
        return cfg[section][name]
    return default


def coerce(raw: str, like):
    """把字符串按"参照值的类型"转换（没有参照就按内容猜）。

    给 CLI / Web 用：`--set vision__nitro_red_thr=0.22` 要变成 float，
    `--set scan__with_buttons=false` 要变成 bool。
    """
    return _coerce(raw, like)


def _coerce(raw: str, like):
    """按"参照值的类型"转换字符串（没有参照就按内容猜）。"""
    if isinstance(like, bool):
        return str(raw).strip().lower() in ("1", "true", "yes", "on", "开")
    if isinstance(like, int) and not isinstance(like, bool):
        try:
            return int(float(raw))
        except ValueError:
            return like
    if isinstance(like, float):
        try:
            return float(raw)
        except ValueError:
            return like
    if isinstance(like, (list, tuple)):
        parts = [p for p in str(raw).replace("，", ",").split(",") if p.strip()]
        try:
            return [int(p) if str(p).strip().lstrip("-").isdigit() else float(p)
                    for p in parts]
        except ValueError:
            return like
    return raw


def _as_box(val, fallback):
    if val is None:
        return tuple(fallback)
    if isinstance(val, (list, tuple)) and len(val) == 4:
        return tuple(int(float(v)) for v in val)
    return tuple(fallback)


def apply(cfg: dict | None = None) -> dict:
    """把配置灌进各模块的模块级常量。**入口都要调它**（返回实际生效的配置）。"""
    from a9route.core import intent
    from a9route.vision import cues, hud

    cfg = cfg or load_config()
    v = cfg.get("vision", {})
    h = cfg.get("hud", {})
    it = cfg.get("intent", {})

    # ---- 按键框与阈值（cues 里的函数都按模块级常量取默认值）----
    cues.NITRO_KEY_BOX = _as_box(v.get("nitro_key_box"), cues.NITRO_KEY_BOX)
    cues.BRAKE_KEY_BOX = _as_box(v.get("brake_key_box"), cues.BRAKE_KEY_BOX)
    cues.BRAKE_RING_THR = float(v.get("brake_ring_thr", cues.BRAKE_RING_THR))
    cues.BRAKE_WHITE_THR = float(v.get("brake_white_thr", cues.BRAKE_WHITE_THR))
    cues.NITRO_RED_THR = float(v.get("nitro_red_thr", cues.NITRO_RED_THR))
    cues.GAUGE_BOX = _as_box(v.get("gauge_box"), cues.GAUGE_BOX)
    cues.CYAN_THR = float(v.get("cyan_thr", cues.CYAN_THR))
    cues.STATUS_BOX = _as_box(v.get("status_box"), cues.STATUS_BOX)
    cues.TOUCHDRIVE_BOX = _as_box(v.get("touchdrive_box"), cues.TOUCHDRIVE_BOX)
    cues.RING_PAD = int(v.get("ring_pad", cues.RING_PAD))
    cues.WHITE_THR = int(v.get("white_thr", cues.WHITE_THR))
    cues.BRIGHT_THR = int(v.get("bright_thr", cues.BRIGHT_THR))

    # ---- 比赛内 HUD 读数框 ----
    hud.PROGRESS_BOX = _as_box(h.get("progress_box"), hud.PROGRESS_BOX)
    hud.RANK_BOX = _as_box(h.get("rank_box"), hud.RANK_BOX)
    hud.TOUCHDRIVE_BOX = _as_box(h.get("touchdrive_box"), hud.TOUCHDRIVE_BOX)
    hud.CHOICE_BAND = _as_box(h.get("choice_band"), hud.CHOICE_BAND)

    # ---- 选路路标检测（HoughCircles）----
    hud.CHOICE_PARAM1 = int(v.get("choice_param1", hud.CHOICE_PARAM1))
    hud.CHOICE_PARAM2 = int(v.get("choice_param2", hud.CHOICE_PARAM2))
    hud.CHOICE_PARAM2_FALLBACK = int(v.get("choice_param2_fallback",
                                          hud.CHOICE_PARAM2_FALLBACK))
    hud.CHOICE_MIN_R = int(v.get("choice_min_r", hud.CHOICE_MIN_R))
    hud.CHOICE_MAX_R = int(v.get("choice_max_r", hud.CHOICE_MAX_R))
    hud.CHOICE_BLUE_RATIO = float(v.get("choice_blue_ratio", hud.CHOICE_BLUE_RATIO))

    # ---- 操作目的（信号形状 -> 目的）----
    intent.TAP_360_GAP = float(it.get("tap_360_gap", intent.TAP_360_GAP))
    intent.DRIFT_MIN = float(it.get("drift_min", intent.DRIFT_MIN))
    intent.NITRO_GAP = float(it.get("nitro_gap", intent.NITRO_GAP))
    intent.NITRO_HOLD = float(it.get("nitro_hold", intent.NITRO_HOLD))
    scan = cfg.get("scan", {})
    intent.CHOICE_MIN_HOLD = int(scan.get("choice_min_hold", intent.CHOICE_MIN_HOLD))
    intent.CHOICE_IDLE_HOLD = int(scan.get("choice_idle_hold",
                                           intent.CHOICE_IDLE_HOLD))

    # ---- CueDetector 用的那几路框/阈值（它从 coords 里取，没给才用默认）----
    cues.COORDS = {
        "gauge_box": cues.GAUGE_BOX,
        "status_box": cues.STATUS_BOX,
        "touchdrive_box": cues.TOUCHDRIVE_BOX,
        "brake_key_box": cues.BRAKE_KEY_BOX,
        "nitro_key_box": cues.NITRO_KEY_BOX,
        "cyan_thr": cues.CYAN_THR,
    }

    # ---- 按键模型后端（YOLOv8）----
    # 配置可能变了（尤其是 `key_model`/`key_backend`），所以**必须清掉后端缓存** ——
    # 不然一次进程里先后跑两种后端会拿到旧的那个（缓存按参数做 key，其实也安全，
    # 但 ONNX session 不释放会白占显存，所以这里是显式清）。
    try:
        from a9route.vision import keys as _keys
        _keys.reset_cache()
    except Exception:
        pass
    try:
        from a9route.vision import choice as _choice
        _choice.reset_cache()
    except Exception:
        pass
    # ⚠️ **记住"这次生效的是哪份配置"**：`vision.keys.resolve_backend()` /
    # `vision.choice.resolve_backend()` 以前是各自去 `load_config()` 读**磁盘上**那份 ——
    # 于是 `analyze --set vision__choice_backend=heuristic`（以及
    # `analyze(overrides=...)`）**对被覆盖的那几项是无效的**：模块常量确实被灌进去了，
    # 但"用哪个后端/哪个模型"是重建时另读磁盘决定的 ✗✗（实测：两次跑明明给了不同
    # 的 backend，日志里打印的都是同一个模型，整片对比直接失效）。
    # 现在后端解析走 `current()`（= 最近一次 apply 用的那份配置），
    # 覆盖项才真的覆盖得住。
    global _CURRENT
    _CURRENT = cfg
    return cfg


def current() -> dict:
    """**当前生效**的配置（最近一次 `apply(cfg)` 用的那份）。

    没 `apply` 过就退回读磁盘 —— 和以前的行为一致，`load_config()` 仍然只读文件。
    """
    return _CURRENT if _CURRENT is not None else load_config()


def describe(cfg: dict | None = None, changed_only: bool = False) -> str:
    """打印当前生效的配置（标出与环境变量/默认值不同的项）。"""
    cfg = cfg or load_config()
    flat_now = flat(cfg)
    lines = [f"配置文件: {paths.CONFIG_FILE}"
            f"{'' if paths.CONFIG_FILE.is_file() else '（不存在 —— 全用内置默认值）'}",
             ""]
    for section, vals in sorted(cfg.items()):
        rows = []
        for key in sorted(vals):
            cur = flat_now.get(f"{section}__{key}")
            dflt = DEFAULTS.get(section, {}).get(key)
            mark = ""
            if cur != dflt:
                mark = f"   <- 覆盖（内置默认 {dflt!r}）"
            if changed_only and not mark:
                continue
            rows.append(f"  {key:<22} = {cur!r}{mark}")
        if rows:
            lines.append(f"[{section}]")
            lines.extend(rows)
            lines.append("")
    if changed_only and len(lines) == 2:
        return "没有任何覆盖：全部是内置默认值。"
    return "\n".join(lines).rstrip()
