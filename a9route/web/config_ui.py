# -*- coding: utf-8 -*-
"""web.config_ui —— **判据参数的编辑表**（给 Web 端的滑块/输入框用）。

为什么要有这么一张"表"（而不是把 config.json 整个丢给前端）：

1. **范围**：`choice_idle_hold = -1` 这种值不该被写进配置文件 ——
   写进去之后行为会很怪（甚至是"永远不结束"），而且不报错 ✗。
   表里给每项定了 `min/max/step`，后端**照样再校验一遍**（前端不能当闸门）；
2. **中文名 + 说明**：`scan.choice_idle_hold` 这个名字对人没有意义，
   真正要知道的是"连续多少个采样点没路标才算这一段选路结束"；
3. **⚠️ 这一项在当前后端下到底生效吗**：最要命的一类误导是
   "调了没反应" —— 比如 `vision.choice_param2` 只影响 HoughCircles 那条启发式路，
   而现在跑的是模型。表里用 `backend` 标出依赖哪个后端，
   接口会把**当前实际生效的后端**对照进去（`applies` / `note`），
   于是页面能直接写"当前用的是模型，这一项不生效"✓

只读这里不够：改值、校验、落盘在 `web/app.py` 的 `/api/config`。
"""
from __future__ import annotations

from a9route import config as cfgmod
from a9route import paths

#: 哪些后端算"模型"（用来判断"只在启发式生效"的项当前是否管用）
MODEL_BACKENDS = ("onnx", "ultralytics")

#: 可编辑项。
#:
#: 字段说明：
#:   key       —— `段.键`（和 `a9route config set 段__键=值` 一致）
#:   group/zh  —— 分组与中文名
#:   kind      —— int / float / box（4 个整数）/ bool / str
#:   min/max/step —— 滑块的取值范围（box 用它限定每个数）
#:   backend   —— `keys` 或 `choice`：这一项只在**那个后端的启发式**下生效；
#:                空 = 与后端无关（总是生效）
#:   hint      —— 一句话说明（会显示在输入框下面）
FIELDS: list[dict] = [
    # ---------------------------------------------------------------- 选路
    {"key": "scan.choice_idle_hold", "group": "选路（岔路口）",
     "zh": "选路结束闸：连续多少采样点没路标 = 这一段选路结束", "kind": "int",
     "min": 1, "max": 300, "step": 1, "backend": None,
     "hint": "调小 -> 容易把一段岔路口拆成两段（多报）；"
             "调大 -> 相邻两个岔路口会被并成一段（少报）"},
    {"key": "scan.choice_min_hold", "group": "选路（岔路口）",
     "zh": "闪烁闸：同一个值连续多少采样点才认", "kind": "int",
     "min": 1, "max": 60, "step": 1, "backend": None,
     "hint": "挡「只闪一两帧」的误检（实测不加它，10 秒视频能报出 16 次选路）"},
    {"key": "scan.choice_cross_tol", "group": "选路（岔路口）",
     "zh": "交叉校验容差（个百分点）", "kind": "int",
     "min": 0, "max": 10, "step": 1, "backend": None,
     "hint": "精细扫描判出的选路，粗扫也要在 ±这个范围内看到 ≥2 个路标才认"},
    {"key": "scan.icon_every", "group": "选路（岔路口）",
     "zh": "精细扫描里每几帧看一次路标", "kind": "int",
     "min": 1, "max": 30, "step": 1, "backend": None,
     "hint": "调大 = 更快、但更可能漏掉一闪而过的选路"},
    {"key": "vision.choice_conf", "group": "选路（岔路口）",
     "zh": "选路模型置信度阈值", "kind": "float",
     "min": 0.05, "max": 0.95, "step": 0.05, "backend": "choice",
     "hint": "只在**模型后端**下生效（实测 0.30 那一档最好）"},
    {"key": "vision.choice_iou", "group": "选路（岔路口）",
     "zh": "选路模型 NMS IoU", "kind": "float",
     "min": 0.1, "max": 0.95, "step": 0.05, "backend": "choice", "hint": ""},
    {"key": "vision.choice_min_options", "group": "选路（岔路口）",
     "zh": "最少几个选项才算岔路口", "kind": "int",
     "min": 1, "max": 8, "step": 1, "backend": None,
     "hint": "用户口径：至少 2 个才算岔路口"},
    {"key": "vision.choice_max_options", "group": "选路（岔路口）",
     "zh": "最多几个选项", "kind": "int",
     "min": 1, "max": 8, "step": 1, "backend": None, "hint": ""},
    {"key": "vision.choice_param2", "group": "选路（岔路口）",
     "zh": "HoughCircles 圆检测灵敏度（param2）", "kind": "int",
     "min": 10, "max": 80, "step": 1, "backend": "choice",
     "hint": "只在**启发式后端**下生效"},
    {"key": "vision.choice_blue_ratio", "group": "选路（岔路口）",
     "zh": "判「蓝色高亮（选中）」的蓝占比阈值", "kind": "float",
     "min": 0.05, "max": 0.95, "step": 0.05, "backend": "choice",
     "hint": "只在**启发式后端**下生效"},

    # ---------------------------------------------------------------- 按键
    {"key": "vision.brake_ring_thr", "group": "按键（刹车 / 氮气）",
     "zh": "刹车：圈内−圈外白度阈值", "kind": "float",
     "min": 0.02, "max": 0.95, "step": 0.01, "backend": "keys",
     "hint": "只在**启发式后端**下生效"},
    {"key": "vision.brake_white_thr", "group": "按键（刹车 / 氮气）",
     "zh": "刹车：圈内绝对白度阈值", "kind": "float",
     "min": 0.05, "max": 0.99, "step": 0.01, "backend": "keys",
     "hint": "只在**启发式后端**下生效（实测这一路误报最多）"},
    {"key": "vision.nitro_red_thr", "group": "按键（刹车 / 氮气）",
     "zh": "氮气：红占比阈值", "kind": "float",
     "min": 0.02, "max": 0.95, "step": 0.01, "backend": "keys",
     "hint": "只在**启发式后端**下生效"},
    {"key": "vision.key_conf", "group": "按键（刹车 / 氮气）",
     "zh": "按键模型置信度阈值", "kind": "float",
     "min": 0.05, "max": 0.95, "step": 0.05, "backend": "keys",
     "hint": "只在**模型后端**下生效"},
    {"key": "vision.key_iou", "group": "按键（刹车 / 氮气）",
     "zh": "按键模型 NMS IoU", "kind": "float",
     "min": 0.1, "max": 0.95, "step": 0.05, "backend": "keys", "hint": ""},
    {"key": "vision.bright_thr", "group": "按键（刹车 / 氮气）",
     "zh": "亮度阈值（0~255）", "kind": "int",
     "min": 40, "max": 255, "step": 1, "backend": "keys", "hint": ""},
    {"key": "vision.white_thr", "group": "按键（刹车 / 氮气）",
     "zh": "白度阈值（0~255）", "kind": "int",
     "min": 40, "max": 255, "step": 1, "backend": "keys", "hint": ""},

    # ---------------------------------------------------------------- 信号形状
    {"key": "intent.tap_360_gap", "group": "信号形状 → 操作",
     "zh": "两下刹车间隔 < 这个秒数 = 360", "kind": "float",
     "min": 0.05, "max": 0.6, "step": 0.01, "backend": None, "hint": ""},
    {"key": "intent.drift_min", "group": "信号形状 → 操作",
     "zh": "刹车按住超过这个秒数 = 漂移", "kind": "float",
     "min": 0.05, "max": 3.0, "step": 0.05, "backend": None, "hint": ""},
    {"key": "intent.nitro_gap", "group": "信号形状 → 操作",
     "zh": "氮气脉冲聚成「一次连点」的时间窗（秒）", "kind": "float",
     "min": 0.05, "max": 1.0, "step": 0.01, "backend": None, "hint": ""},
    {"key": "intent.nitro_hold", "group": "信号形状 → 操作",
     "zh": "氮气按住超过这个秒数 = 长按连点", "kind": "float",
     "min": 0.1, "max": 3.0, "step": 0.05, "backend": None, "hint": ""},

    # ---------------------------------------------------------------- 扫描
    {"key": "scan.every", "group": "扫描节奏",
     "zh": "粗扫抽帧间隔（秒）", "kind": "float",
     "min": 0.05, "max": 2.0, "step": 0.05, "backend": None,
     "hint": "调大 = 快、但可能漏掉某个百分点"},
    {"key": "scan.hold", "group": "扫描节奏",
     "zh": "氮气去抖：连续几帧才算按下", "kind": "int",
     "min": 1, "max": 30, "step": 1, "backend": None, "hint": ""},
    {"key": "scan.hold_brake", "group": "扫描节奏",
     "zh": "刹车去抖：连续几帧才算按下", "kind": "int",
     "min": 1, "max": 30, "step": 1, "backend": None, "hint": ""},
    {"key": "scan.gap", "group": "扫描节奏",
     "zh": "按键分段间隔（秒）", "kind": "float",
     "min": 0.02, "max": 1.0, "step": 0.01, "backend": None, "hint": ""},
    {"key": "scan.max_seconds", "group": "扫描节奏",
     "zh": "只分析前 N 秒（0 = 不限）", "kind": "float",
     "min": 0.0, "max": 3600.0, "step": 5.0, "backend": None,
     "hint": "试跑用"},

    # ---------------------------------------------------------------- 标定（框）
    {"key": "vision.brake_key_box", "group": "标定（位置框）",
     "zh": "刹车键框 x,y,w,h", "kind": "box",
     "min": 0, "max": 1280, "step": 1, "backend": None,
     "hint": "改完跑 `a9route train audit` 看标注还对不对得上"},
    {"key": "vision.nitro_key_box", "group": "标定（位置框）",
     "zh": "氮气键框 x,y,w,h", "kind": "box",
     "min": 0, "max": 1280, "step": 1, "backend": None, "hint": ""},
    {"key": "hud.choice_band", "group": "标定（位置框）",
     "zh": "选路带 x,y,w,h（屏幕上方那条）", "kind": "box",
     "min": 0, "max": 1280, "step": 1, "backend": None,
     "hint": "模型出的路标只在带内才收（带区闸）；标注页也是按它裁图"},
]

#: 字段名的索引（校验时用）
BY_KEY = {f["key"]: f for f in FIELDS}


def _get(cfg: dict, key: str):
    sec, _, name = key.partition(".")
    return (cfg.get(sec) or {}).get(name)


def _default_of(key: str):
    return _get(cfgmod.DEFAULTS, key)


def coerce(field: dict, raw) -> tuple:
    """把前端来的值转成该字段的类型，并**校验范围**。

    返回 `(值, 错误信息)`：错误信息非空时，值不可用。
    这里的校验**不是**给前端用的门面 —— 前端可以绕过，所以后端必须自己拦。
    """
    key, kind = field["key"], field["kind"]
    try:
        if kind == "int":
            v = int(float(raw))
        elif kind == "float":
            v = float(raw)
        elif kind == "bool":
            v = bool(raw)
        elif kind == "box":
            if isinstance(raw, str):
                parts = [p for p in raw.replace("，", ",").split(",") if p.strip()]
            else:
                parts = list(raw or [])
            if len(parts) != 4:
                return None, "{0}：位置框要 4 个数（x,y,w,h），收到 {1} 个" .format(
                    key, len(parts))
            v = [int(float(p)) for p in parts]
            lo, hi = int(field.get("min", 0)), int(field.get("max", 10000))
            if not all(lo <= x <= hi for x in v):
                return None, "{0}：每个数都要在 {1}~{2} 之间，收到 {3}".format(
                    key, lo, hi, v)
            if v[2] <= 0 or v[3] <= 0:
                return None, "{0}：宽高必须 > 0（收到 {1}×{2}）".format(key, v[2], v[3])
            return v, ""
        else:
            return str(raw), ""
    except (TypeError, ValueError):
        return None, "{0}：要一个数，收到 {1!r}".format(key, raw)
    if kind in ("int", "float"):
        lo = field.get("min")
        hi = field.get("max")
        if lo is not None and v < lo:
            return None, "{0}：不能小于 {1}（收到 {2}）".format(key, lo, v)
        if hi is not None and v > hi:
            return None, "{0}：不能大于 {1}（收到 {2}）".format(key, hi, v)
    return v, ""


def payload(cfg: dict | None = None, backends: dict | None = None) -> dict:
    """给前端的完整参数表：字段定义 + 当前值 + 默认值 + 改过的项 + 生效提示。"""
    cfg = cfg if cfg is not None else cfgmod.current()
    backends = backends or {}
    out = []
    for f in FIELDS:
        cur = _get(cfg, f["key"])
        dflt = _default_of(f["key"])
        item = dict(f)
        item["value"] = cur
        item["default"] = dflt
        item["changed"] = (cur != dflt)
        # ⚠️ 这一项在当前后端下到底生效吗
        task = f.get("backend")
        be = (backends.get(task) if task else None) or ""
        if not task:
            item["applies"] = True
            item["note"] = ""
        elif be in MODEL_BACKENDS:
            item["applies"] = (f["key"].split(".")[-1].endswith("conf")
                               or f["key"].split(".")[-1].endswith("iou"))
            item["note"] = ("" if item["applies"] else
                            "当前 {0} 用的是**模型**，这一项（启发式判据）不生效".format(
                                task))
        else:
            item["applies"] = True
            item["note"] = "当前 {0} 用的是**启发式**（{1}）".format(task, be or "?")
        out.append(item)
    return {"fields": out, "config_file": str(paths.CONFIG_FILE),
            "config_exists": paths.CONFIG_FILE.is_file(),
            "changed": [f["key"] for f in out if f["changed"]]}


def save(set_map: dict) -> dict:
    """校验并落盘。**先整体校验，有一个不合格就一个都不写**（不留半套配置）。

    返回 `{ok, errors, written, values}`；`errors` 非空时 `written` 为空。
    """
    if not isinstance(set_map, dict) or not set_map:
        return {"ok": False, "errors": ["没给要改的项"], "written": []}
    errors: list[str] = []
    clean: dict = {}
    for raw_key, raw_val in set_map.items():
        key = str(raw_key).replace("__", ".").strip()
        f = BY_KEY.get(key)
        if f is None:
            errors.append("不认识的参数：{0}".format(raw_key))
            continue
        v, err = coerce(f, raw_val)
        if err:
            errors.append(err)
            continue
        clean[key] = v
    if errors:
        return {"ok": False, "errors": errors, "written": []}
    cfgf = cfgmod.load_config()
    dropped: list[str] = []
    for key, v in clean.items():
        sec, _, name = key.partition(".")
        if v == _default_of(key):
            # ⚠️ 和内置默认值相同的项**不写进文件**（把已有的键删掉）——
            # config.json 只记**真正的覆盖**，人打开它看到的才是有意义的差异。
            # （否则"恢复默认"会把默认值一个个写进去，文件越来越长、
            #   看 `config list --changed` 也全是噪音 ✗）
            if name in (cfgf.get(sec) or {}):
                del cfgf[sec][name]
                dropped.append(key)
            continue
        cfgf.setdefault(sec, {})[name] = v
    cfgmod.save_config(cfgf)
    cfgmod.apply()                      # 立刻生效（下一次分析就读到新值）
    return {"ok": True, "errors": [], "written": sorted(clean),
            "dropped_to_default": sorted(dropped),
            "values": {k: _get(cfgmod.current(), k) for k in clean}}


def reset(keys: list[str] | None = None) -> dict:
    """恢复内置默认值（给"默认"按钮用）。`keys=None` = 全部恢复。"""
    keys = [str(k).replace("__", ".") for k in (keys or list(BY_KEY))]
    bad = [k for k in keys if k not in BY_KEY]
    if bad:
        return {"ok": False, "errors": ["不认识的参数：" + "，".join(bad)],
                "written": []}
    want = {k: _default_of(k) for k in keys}
    return save(want)
