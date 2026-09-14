# -*- coding: utf-8 -*-
"""video.py —— 把跑图**视频**半自动转成百分比路线脚本（用户提的可选项）。

## 老实说清楚能做到什么

视频里**没有"按键"这个真值**（画面里看不到你按了什么），所以只能靠画面线索**推断**，
误报难免。因此这里的定位是 **"自动生成草稿 + 人工确认"**，不是全自动：

1. **抽帧 + 读「路程 NN%」**：这是**可靠**的那一半 —— 路程百分比在画面左上角，
   和实时跑图用的是同一个读数（`RaceReader`）。于是能得到
   `时间 -> 百分比` 的映射（比赛里百分比单调递增，可以插值补空档）。
2. **生成草稿**：按整百分点输出一张"检查表"：每个百分比对应视频里的时间戳。
   你自己看视频、在对应百分比后面填操作（`N:...` / `D:...` / `360` / 选路两位数字）——
   比对着视频手写一遍快得多，而且**时间点已经被自动算好了**。
3. （可选、后续）**画面线索辅助建议**：车尾蓝/紫火焰≈氮气、右下角「漂移NN米」在涨≈漂移、
   上方出现圆形路标≈选路 —— 这些作为**建议**列在草稿里，你确认后才写进路线。

## 用法

    python -m a9route video 录像.mp4                 # 打印时间轴 + 草稿
    python -m a9route video 录像.mp4 --out routes/draft.txt --every 0.4
    python -m a9route analyze 录像.mp4               # 逐百分点截图 + 推断操作 + 生成路线
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Sample:
    """视频里的一帧采样：时刻（秒）+ 读到的路程百分比（读不到就是 None）。"""

    t: float
    percent: float | None = None


@dataclass
class Timeline:
    """时间轴。`samples` 按时间升序；`points` 是"百分比 -> 时间"的已插值映射。"""

    samples: list[Sample] = field(default_factory=list)
    points: list[tuple[float, float]] = field(default_factory=list)   # (percent, t)
    warnings: list[str] = field(default_factory=list)

    def time_at(self, percent: float) -> float | None:
        """线性插值：百分比 -> 秒。超出范围返回 None。"""
        if not self.points:
            return None
        pts = self.points
        if percent < pts[0][0] or percent > pts[-1][0]:
            return None
        for (p0, t0), (p1, t1) in zip(pts, pts[1:]):
            if p0 <= percent <= p1:
                if p1 == p0:
                    return t0
                k = (percent - p0) / (p1 - p0)
                return t0 + k * (t1 - t0)
        return pts[-1][1]

    def percent_at(self, t: float) -> float | None:
        """反插值：秒 -> 百分比（按键细扫出来的时刻要落回百分比）。"""
        if not self.points:
            return None
        pts = sorted(self.points, key=lambda x: x[1])
        if t < pts[0][1] or t > pts[-1][1]:
            return None
        for (p0, t0), (p1, t1) in zip(pts, pts[1:]):
            if t0 <= t <= t1:
                if t1 == t0:
                    return p0
                k = (t - t0) / (t1 - t0)
                return p0 + k * (p1 - p0)
        return pts[-1][0]

    def describe(self) -> str:
        n_ok = sum(1 for s in self.samples if s.percent is not None)
        lines = [f"采样 {len(self.samples)} 帧，其中 {n_ok} 帧读到了路程百分比",
                 f"时间轴覆盖 {self.points[0][0]:g}% ~ {self.points[-1][0]:g}%"
                 if self.points else "（没读到任何百分比）"]
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def build_timeline(samples: list[Sample]) -> Timeline:
    """把采样整理成"百分比 -> 时间"的时间轴。

    * 丢掉没读到百分比的帧；
    * **百分比必须单调不减**（比赛里只会往前走）：读到倒退的点视为 OCR 抖动丢掉；
    * 同一个百分比只留**第一次**出现的时间（后面再读到同一个值是转场/抖动）。
    """
    tl = Timeline(samples=list(samples))
    last_p = -1.0
    for s in samples:
        if s.percent is None:
            continue
        if s.percent + 0.01 < last_p:
            tl.warnings.append(f"{s.t:.1f}s 读到 {s.percent:g}%（比上一个 {last_p:g}% 小）"
                               "—— 当作 OCR 抖动丢掉")
            continue
        if tl.points and abs(s.percent - tl.points[-1][0]) < 0.01:
            last_p = s.percent
            continue
        tl.points.append((float(s.percent), float(s.t)))
        last_p = float(s.percent)
    return tl


def sample_video(path: str | Path, *, every: float = 0.5, reader=None,
                 max_seconds: float | None = None) -> list[Sample]:
    """按固定间隔抽帧并读路程百分比，返回采样列表（有序）。

    * `reader` 必须提供 `.read(frame) -> RaceState`（默认用 `RaceReader`）；
    * 抽帧用 OpenCV（见 `iter_frames`），不依赖 ffmpeg 可执行文件。
    """
    if reader is None:
        from a9route.vision.hud import RaceReader
        reader = RaceReader()
    out: list[Sample] = []
    for t, frame in iter_frames(path, every=every, max_seconds=max_seconds):
        percent = None
        try:
            st = reader.read(frame)
            percent = getattr(st, "percent", None)
            if percent is None and isinstance(st, dict):
                percent = st.get("percent")
        except Exception:
            percent = None
        out.append(Sample(t=t, percent=percent))
    return out


def iter_frames(path: str | Path, *, every: float = 0.5,
                max_seconds: float | None = None, ahead: int = 0):
    """按间隔抽帧，**产出 (时刻秒, 帧)**（需要帧本身时用这个）。

    `ahead>0`：另起一个**解码线程提前取帧**塞进小队列 —— 抽帧是"seek + 解码"，
    通常比逐帧检测（OCR/圆检测）慢或相当，重叠起来能省掉等待时间 ✓
    （只读画面、不改状态，所以线程安全）。
    """
    import cv2
    import queue
    import threading
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total / fps if fps else 0.0

    def produce(put, stop):
        try:
            t = 0.0
            while True:
                if stop.is_set():
                    break
                if max_seconds is not None and t > max_seconds:
                    break
                if duration and t > duration:
                    break
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    break
                # ⚠️ **用解码出来的那一帧的真实时间**，不是我们请求的时间 ✗
                # （`set(POS_MSEC)` 之后解码器常常落在附近的帧上，请求时间会偏；
                #   精细扫描是顺序解码、时间轴是真实 PTS，两套时间基不一致就会出现
                #   "精细识别说 1%、百分比识别说 5%"这种错位 —— 用户 2026-09-13 报的 ✓）
                real = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                put((round(real if real > 0 else t, 3), frame))
                t += every
        finally:
            put(None)

    if ahead and ahead > 0:
        q: queue.Queue = queue.Queue(maxsize=max(2, ahead))
        stop = threading.Event()
        th = threading.Thread(target=produce, args=(q.put, stop),
                              name="video-decode", daemon=True)
        th.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                yield item
        finally:
            stop.set()
            try:
                while True:
                    if q.get_nowait() is None:
                        break
            except Exception:
                pass
        return
    try:
        t = 0.0
        while True:
            if max_seconds is not None and t > max_seconds:
                break
            if duration and t > duration:
                break
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            # 同 produce()：报告**真实解码时间**，保证和精细扫描同一套时间基 ✓
            real = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield (round(real if real > 0 else t, 3), frame)
            t += every
    finally:
        cap.release()


@dataclass
class PercentShot:
    """一个整百分点上的"截图 + 当时在做什么"。"""

    percent: float
    t: float
    path: str = ""              # 截好的图（PNG）
    cue: object = None          # cues.Cue
    op: str = ""                # 推断出来的操作（路线脚本里的写法，可含 |）
    #: 该百分点检测到的操作**类别**（`choice/drift/nitro/spin/auto`）——
    #: 用户 2026-09-13 的规则：同一段里多个类别就写成 `|`（同时执行）
    kinds: list = field(default_factory=list)
    #: 类别 -> 操作写法（选路是 `32` 这种；漂移/氮气只有时长，段合并时再算）
    values: dict = field(default_factory=dict)
    #: 这个百分点上的**按键动作**（细扫出来的 `ButtonEvent`，含单击/双击/时长）
    buttons: list = field(default_factory=list)
    #: 这个百分点上的**操作意图**（`intent.Intent`：带目的 + 实测间隔 + 判据说明）
    intents: list = field(default_factory=list)

    def describe(self) -> str:
        c = self.cue
        desc = c.describe() if c is not None else ""
        return (f"{self.percent:>5.0f}%  @ {self.t:6.2f}s  "
                f"{'（无操作）' if not self.op else self.op:<20} {desc}")


def extract_per_percent(path: str | Path, *, out_dir: str | Path,
                        every: float = 0.25, detector=None,
                        max_seconds: float | None = None,
                        with_buttons: bool = True,
                        workers: int = 2,
                        fine_gap: float | None = None,
                        fine_hold: int | None = None,
                        fine_hold_brake: int | None = None,
                        icon_every: int | None = None,
                        choice_cross_tol: int | None = None,
                        progress=print) -> list[PercentShot]:
    """每个整百分点截一张图 + 判断当时在做什么操作。

    `with_buttons=True`（默认）：**逐帧**再看一遍两个按键（刹车=亮白、氮气=红），
    这样能分辨**单击 / 双击 / 长按**三种氮气（用户 2026-09-13 的要求）。

    `workers>1`：解码与逐帧检测并行（解码线程提前取帧，主线程做检测/OCR/圆检测）——
    抽帧受解码限制，重叠起来能省掉大部分等待时间（见 `iter_frames`）。

    `fine_*` / `icon_every` / `choice_cross_tol`：精细扫描与交叉校验的参数，
    **不传就用 `config.json` 的 `scan` 段**（`a9route.analyze()` 负责传）。

    做法（补充）：按 `every` 秒扫一遍视频，对每帧算 `(百分比, Cue)`；
    同一个整百分点只留**第一次读到的**那一帧（写成 PNG），
    同时把该百分点附近的 cue 汇总起来判断操作 —— 单帧可能有闪烁，窗口汇总才稳。
    """
    import cv2
    from a9route.vision.cues import CueDetector
    from a9route import config as _cfgmod
    det = detector or CueDetector()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # 精细扫描/交叉校验的参数：显式传参 > config.json（含环境变量）> 内置默认
    scan_cfg = (_cfgmod.load_config() or {}).get("scan", {})
    fine_kw = {
        "gap": scan_cfg.get("gap", 0.12) if fine_gap is None else fine_gap,
        "hold": scan_cfg.get("hold", 1) if fine_hold is None else fine_hold,
        "hold_brake": (scan_cfg.get("hold_brake", 2) if fine_hold_brake is None
                       else fine_hold_brake),
        "icon_every": (scan_cfg.get("icon_every", 2) if icon_every is None
                       else icon_every),
    }
    cross_tol = int(scan_cfg.get("choice_cross_tol", 1) if choice_cross_tol is None
                    else choice_cross_tol)

    seen: dict[int, PercentShot] = {}
    window: dict[int, list] = {}
    n_frames = 0
    # 按键细扫跑在**另一个线程**（它自己开一路解码，和这一路互不干扰）——
    # 两条流并行能省掉整段细扫时间 ✓
    btn_result: dict = {}

    def _button_worker():
        try:
            s, ic = scan_fine(path, progress=None, **fine_kw)
            btn_result["samples"] = s
            btn_result["icons"] = ic
        except Exception as exc:
            btn_result["error"] = f"{type(exc).__name__}: {exc}"

    btn_thread = None
    if with_buttons:
        import threading
        btn_thread = threading.Thread(target=_button_worker, name="race-buttons", daemon=True)
        btn_thread.start()

    for t, frame in iter_frames(path, every=every, max_seconds=max_seconds,
                               ahead=(2 if workers > 1 else 0)):
        cue, pct = det.detect(frame)
        n_frames += 1
        if pct is None:
            if progress and n_frames % 200 == 0:
                progress(f"已扫 {n_frames} 帧（还没读到路程百分比 —— 前面可能是加载/倒计时）")
            continue
        p = int(round(float(pct)))
        window.setdefault(p, []).append((t, cue))
        if p not in seen:
            fname = out / f"{p:03d}pct_t{t:06.2f}s.png"
            cv2.imwrite(str(fname), frame)
            seen[p] = PercentShot(percent=float(p), t=float(t), path=str(fname), cue=cue)
            # **每覆盖几个百分点报一次**：前端进度条就靠这句话
            # （早期只在每 200 帧报一次，界面上会长时间停在"准备中" ✗）
            if progress and len(seen) % 3 == 0:
                progress(f"已扫 {n_frames} 帧，覆盖 {len(seen)} 个百分点…")
    shots = [seen[p] for p in sorted(seen)]
    # 用"窗口汇总"决定每个百分点的操作（单帧会闪）
    for s in shots:
        ops = _ops_from_window(window.get(int(s.percent), []))
        s.kinds = [k for k in OP_ORDER if k in ops]
        s.values = dict(ops)
        s.op = "|".join(ops[k] for k in OP_ORDER if k in ops)
    # **精细扫描**：逐帧看两个按键 + 每 2 帧看一次选路路标。
    # 它**只产时间戳**（不产百分比）—— 目的由形状判定（见 intent.py），
    # 百分比最后由这里的「时间 -> 百分比」关系填 ✓（用户 2026-09-13 的规则）。
    if with_buttons:
        try:
            from a9route.core import intent as IT
            if btn_thread is not None:
                btn_thread.join(timeout=900)
            if progress:
                progress("  精细扫描（逐帧看刹车键/氮气键 + 选路路标）…")
            samples = btn_result.get("samples") or []
            icons_tl = btn_result.get("icons") or []
            if not samples and btn_result.get("error"):
                raise RuntimeError(btn_result["error"])
            # OCR 见到「完美氮气」的时间窗：用来确认那是"双击"（用户规则 ✓）
            perfect: list[tuple[float, float]] = []
            for s in shots:
                if s.cue is not None and "完美氮气" in (getattr(s.cue, "status_text", "") or ""):
                    perfect.append((s.t - 0.2, s.t + 0.6))
            intents = IT.classify_buttons(samples, perfect_nitro=perfect)
            intents += IT.classify_choices(icons_tl)
            intents.sort(key=lambda e: e.t0)
            # 时刻 -> 百分比（用粗扫的映射）
            tl = build_timeline([Sample(t=s.t, percent=s.percent) for s in shots])
            kept = []
            for it in intents:
                p = tl.percent_at(it.t0)
                if p is None:
                    continue
                it.percent = float(int(round(p)))
                kept.append(it)
            # **交叉校验**：选路必须"粗扫也看到 ≥2 个路标"才算（允许 ±`cross_tol` 个百分点）——
            # 用户 2026-09-13 报过"开头多出一次选路、实际只有 5 次" ✗，
            # 原因是精细扫描的圆检测偶发多认一个圆；两边都说是岔路口才认，误报几乎清零 ✓
            coarse_choice = set()
            for s in shots:
                icons = getattr(getattr(s, "cue", None), "icons", None) or []
                if len(icons) >= 2:
                    coarse_choice.add(int(s.percent))
            if coarse_choice:
                kept = [it for it in kept
                        if not (it.kind == "choice"
                                and not any(abs(int(round(it.percent)) - p) <= cross_tol
                                            for p in coarse_choice))]
            kept = IT.merge_by_percent(kept)
            for s in shots:
                s.intents = [it for it in kept if int(round(it.percent)) == int(s.percent)]
        except Exception as exc:
            if progress:
                progress(f"  精细扫描失败（不影响其余判据）：{type(exc).__name__}: {exc}")
    if progress:
        progress(f"  扫完 {n_frames} 帧，得到 {len(shots)} 张百分比截图")
        if with_buttons:
            n_it = sum(len(getattr(s, "intents", []) or []) for s in shots)
            progress(f"  精细扫描判出 {n_it} 个操作（按目的：360/漂移/打断氮气/氮气/选路）")
    return shots


def _ops_from_window(frames: list) -> dict:
    """把一个百分点附近的若干帧汇总成 `{类别: 操作写法}`（空字典 = 没检测到操作）。

    这是用户 2026-09-13 要的规则：**看按键亮没亮**，亮着就记当前百分比。

    * `nitro`：右下**氮气键变红**（实测这一段录像：喷氮时瓶子区域 3141 个红像素，
      不喷时几乎为 0 —— 判据很干净 ✓）；
    * `drift`：左下**刹车键亮**（漂移时图标内的图形变亮）。⚠️ 这段录像上路面的
      明暗变化太大，纯像素判据与游戏自己的「漂移NN米」提示对不上（扫描最好只有 0.61
      准确率），所以这里**同时**接受"刹车图标变亮"或"右侧出现漂移提示"，
      待用户给两张对比截图后再收紧；
    * `choice`：上方圆形路标（数量 + 当前蓝色高亮是第几个，**取最后一次**）；
    * `spin`：右侧「完成360度旋转」；
    * `auto`：左上角 TOUCHDRIVE 关。
    """
    if not frames:
        return {}
    n = len(frames)
    out: dict = {}
    # 选路：至少 2 个路标 且 至少两帧都看到（单个圆多半是背景物；闪光只出现一帧）
    icon_frames = [(t, c) for t, c in frames if len(c.icons) >= 2]
    if len(icon_frames) >= 2:
        cnt, idx = icon_frames[-1][1].choice or (len(icon_frames[-1][1].icons), 1)
        out["choice"] = f"{cnt}{idx}"
    if any(c.spin360 for _t, c in frames):
        out["spin"] = "360"
    if any(c.autopilot_off for _t, c in frames):
        out["auto"] = "S:2000"
    # 按键类：窗口里**亮着的帧**占比够就算这一段按住了（单帧闪烁不算）
    # ⚠️ 氮气**只看"氮气键被按下"**（`nitro_pressed`），**不看顶部槽是否青** ✗ ——
    #   槽青只说明"在喷"，70% 之后长段青会被误判成"一直在点氮气键"（用户 2026-09-13 报的）
    if sum(1 for _t, c in frames if getattr(c, "nitro_pressed", False)) >= max(1, n // 4):
        out["nitro"] = "N"          # 具体次数/间隔等合并成段之后再算
    if sum(1 for _t, c in frames if c.drifting) >= max(1, n // 4):
        out["drift"] = "D"
    return out


#: 输出顺序（同一条目里用 | 连接时的先后顺序，固定住便于对比）
OP_ORDER = ("choice", "drift", "nitro", "spin", "auto")


def _run_op(kind: str, seconds: float, value: str) -> str:
    """把一段（连续若干百分点都在按同一个键）写成一个操作写法。"""
    if kind == "drift":
        ms = max(200, min(8000, int(round(seconds * 1000))))
        return f"D:{ms}"
    if kind == "nitro":
        # 喷氮写成"快速连点"：次数按这段时长估，间隔 100ms（用户脚本里 2~20 次都出现过）
        k = max(1, min(20, int(round(seconds / 0.5))))
        return f"N:0:{k}:100"
    return value


def _op_kind(op: str) -> str:
    """操作写法 -> 类别（'choice' / 'nitro' / 'drift' / 'spin' / 'auto' / ''）。"""
    if not op:
        return ""
    if op == "360":
        return "spin"
    if op.startswith("N:"):
        return "nitro"
    if op.startswith("D:"):
        return "drift"
    if op.startswith("S:"):
        return "auto"
    if op.isdigit() and len(op) == 2:
        return "choice"
    return ""


def collapse_runs(shots: list[PercentShot], *, sample_seconds: float = 0.25) -> list[PercentShot]:
    """把**连续同类**的检测合并成一条（一个岔路口/一段氮气/一段漂移写成一条）。

    为什么必须合并：判据是逐帧的，而用户脚本的写法是"**一个事件一条**"
    （一个岔路口写 `40,22` 一条，而不是 40/41/42/43 各写一条 ✗）。
    合并规则（用户 2026-09-13 定）：

    * 相邻百分点、**按键状态相同**（同一组类别）的算同一段，段内只留一条；
    * **同一段里多个操作用 `|` 连起来**（= 同时执行，用户脚本里就是这么写的）；
    * 时长按这一段覆盖的秒数换算：漂移 -> `D:毫秒`、氮气 -> `N:0:k:100`；
    * 选路 / 360 / 关自动驾驶：取**段内最后一个**检测到的值
      （选路尤其重要：玩家可能在图标显示期间改选）。
    """
    out: list[PercentShot] = []
    i = 0
    while i < len(shots):
        cur = shots[i]
        kinds = cur.kinds or ([_op_kind(cur.op)] if cur.op else [])
        kinds = [k for k in kinds if k]
        if not kinds:
            out.append(cur)
            i += 1
            continue
        j = i
        while (j + 1 < len(shots) and (shots[j + 1].kinds or []) == kinds
               and shots[j + 1].percent - shots[j].percent <= 1.5):
            j += 1
        span = max(0.0, shots[j].t - cur.t) + sample_seconds
        merged = PercentShot(percent=cur.percent, t=cur.t, path=cur.path, cue=cur.cue,
                             kinds=list(kinds))
        parts = []
        for k in OP_ORDER:
            if k not in kinds:
                continue
            # 段内最后一个值（选路要最后选的那条；漂移/氮气只看时长）
            value = ""
            for s in shots[i:j + 1]:
                if k in (s.kinds or []) and s.values and s.values.get(k):
                    value = s.values[k]
            parts.append(_run_op(k, span, value))
        merged.op = "|".join(parts)
        out.append(merged)
        i = j + 1
    return out


@dataclass
class ButtonEvent:
    """一次按键动作（细扫出来的）。"""

    kind: str            # "drift" | "nitro"
    t0: float            # 开始时刻
    duration: float      # 持续秒数
    pulses: int = 1      # 里面点了几个"脉冲"（双击=2）
    op: str = ""         # 写进路线的操作
    percent: float = 0.0  # 落在哪个百分比

    def describe(self) -> str:
        return (f"{self.percent:>5.0f}%  {self.kind:<5} 起 {self.t0:6.2f}s "
                f"持续 {self.duration:.2f}s 脉冲 {self.pulses} -> {self.op}")


def scan_fine(path: str | Path, *, gap: float = 0.12, hold: int = 1,
              hold_brake: int = 2, icon_every: int = 2, progress=None
              ) -> tuple[list[tuple[float, bool, bool]], list[tuple[float, tuple]]]:
    """**精细扫描**：逐帧看两个按键 + 每 `icon_every` 帧看一次选路路标。

    产出（**都不带百分比** —— 用户 2026-09-13 明确要求：精细扫描只产时间戳，
    百分比由粗扫的「时间 -> 百分比」关系事后填 ✓）：

    * `samples`：`(时刻, 刹车按下, 氮气按下)`；
    * `icons`：`(时刻, (图标数, 蓝色高亮第几个) 或 None)` —— 选路也在精细扫描里识别 ✓。

    为什么选路也要进精细扫描：路标出现/消失/改选的**时刻**很关键（用户要求
    "一直不变取第一次识别到、变了取第一次改变时"），粗扫 0.25s 一拍会漏掉改选 ✗。

    ⚠️ 这里用的是 `cues.BRAKE_KEY_BOX` / `NITRO_KEY_BOX` 等**模块级常量**，
    它们由 `a9route.config.apply()` 按 `config.json` 灌好 —— 所以本函数
    **必须在 `apply()` 之后调用**（CLI / Web 入口都已代劳）。
    """
    import cv2
    from a9route.vision.cues import (BRAKE_KEY_BOX, BRAKE_WHITE_THR, NITRO_KEY_BOX,
                                  NITRO_RED_THR, brake_pressed, nitro_pressed)
    from a9route.vision.hud import RaceReader
    reader = RaceReader()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{path}")
    samples: list[tuple[float, bool, bool]] = []
    icons: list[tuple[float, tuple]] = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dt = 1.0 / fps if fps else 1 / 30.0
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i * dt
            samples.append((round(t, 3),
                            brake_pressed(frame, BRAKE_KEY_BOX) > BRAKE_WHITE_THR,
                            nitro_pressed(frame, NITRO_KEY_BOX) > NITRO_RED_THR))
            if icon_every and i % icon_every == 0:
                try:
                    found = reader.choice_icons(frame)
                except Exception:
                    found = []
                cur = None
                if len(found) >= 2:
                    ordered = sorted(found, key=lambda s: s["xy"][0])
                    idx = next((k + 1 for k, s in enumerate(ordered)
                                if s.get("blue")), 0)
                    cur = (len(ordered), idx or 1)
                icons.append((round(t, 3), cur))
            i += 1
            if progress and i % 900 == 0:
                progress(f"  精细扫描 {i} 帧（{t:.0f}s）…")
    finally:
        cap.release()
    # 去抖**分通道**（用户 2026-09-13 报"17% 处两次氮气点击没识别到"后定的）：
    #  * 刹车：`hold_brake=2` —— 漂移是"按住"，要求连续两帧更稳 ✓；
    #  * 氮气：`hold=1` —— 实测那两次点击在 59 帧里只亮 2 帧（每次约 1 帧 = 33ms ✗），
    #    "连续两帧"会把它整段抹掉 ✗✗；改成靠**阈值够高**（红>0.15 或 亮白>0.25）挡噪声 ✓。
    out: list[tuple[float, bool, bool]] = []
    for k, (t, _b, _n) in enumerate(samples):
        vals = []
        for idx in (0, 1):
            need = hold_brake if idx == 0 else hold
            w = [samples[j][1 + idx] for j in range(max(0, k - need + 1), k + 1)]
            vals.append(len(w) >= need and all(w))
        out.append((t, vals[0], vals[1]))
    return out, icons


def scan_buttons(path: str | Path, *, gap: float = 0.12, hold: int = 2,
                 t_from: float | None = None, t_to: float | None = None,
                 progress=None) -> list[tuple[float, bool, bool]]:
    """兼容旧接口：只要按键那一路（内部就是 `scan_fine`）。"""
    s, _icons = scan_fine(path, gap=gap, hold=hold, icon_every=0, progress=progress)
    if t_from is not None or t_to is not None:
        s = [x for x in s
             if (t_from is None or x[0] >= t_from - 1.0)
             and (t_to is None or x[0] <= t_to + 1.0)]
    return s


def _episodes(samples: list[tuple[float, bool, bool]], idx: int, *, gap: float
              ) -> list[tuple[float, float, int]]:
    """把"按下"的帧聚成一次按键动作，返回 [(起点, 时长秒, 脉冲数)]。

    `samples` 每项是 `(t, 刹车按下, 氮气按下)`，`idx` 0=刹车 1=氮气。

    ## 踩过的坑（2026-09-13）

    原来是拿"**最后一次松开**的帧时刻"算间隔：`t - last_off` —— 那永远是"上一帧"
    （约 33ms），**永远不会超过 gap**，于是用户 0.5s 一次的点松被并成一个 8 秒长按 ✗✗。
    必须用"**这一次松开从哪一帧开始**"来算松开时长：

        off_len = 本次按下帧时刻 - 第一帧松开时刻

    实测波形（用户标定视频）：按下时 0.64~0.66、松开时 0.00，中间只有 5~7 帧过渡，
    所以 `gap=0.12s`（约 3.6 帧）能把每次点松干净切开 ✓。
    """
    runs: list[tuple[float, float, int]] = []      # (起点, 时长, 脉冲数)
    cur_start: float | None = None
    cur_last: float = 0.0
    pulse_start: float | None = None
    pulses = 0
    first_off: float | None = None
    prev_on = False

    def close_run():
        nonlocal cur_start, cur_last, pulses, pulse_start, first_off
        if cur_start is None:
            return
        runs.append((cur_start, cur_last - cur_start, max(1, pulses)))
        cur_start = None
        pulse_start = None
        pulses = 0
        first_off = None

    for s in samples:
        t = float(s[0])
        on = bool(s[1 + idx])
        if on:
            if cur_start is None:
                cur_start, cur_last, pulses = t, t, 1
                pulse_start = t
                first_off = None
            else:
                # 松开时长 = 本次按下 - 这一次松开的**第一帧**
                if first_off is not None and (t - first_off) > gap + 1e-6:
                    close_run()
                    cur_start, cur_last, pulses = t, t, 1
                    pulse_start = t
                    first_off = None
                else:
                    if first_off is not None and prev_on is False:
                        pulses += 1          # 短暂断开又按下 -> 双击的第二个脉冲
                    cur_last = t
                    first_off = None
        else:
            if cur_start is not None:
                if first_off is None:
                    first_off = t
                cur_last = cur_last          # 松开期间不动"最后按下"时刻
        prev_on = on
    close_run()
    return runs


def button_events(samples: list[tuple[float, bool, bool]], *,
                  gap: float = 0.12) -> list[ButtonEvent]:
    """按键帧序列 -> 按键动作列表，并**按用户的规则写出操作**。

    用户的规则（2026-09-13）：

    * **氮气爆发 / 完美氮气**：都是**双击**（两下 100ms 内；完美氮气还会让氮气条变蓝）
      -> 按双击写：`N:0:2:100`（= 延时 0、2 次、间隔 100ms）；
    * **普通氮气**：单击 -> `N:100:1:100`（他自己的脚本里就是这个写法）；
    * 长按（≥0.6s）仍然按"快速连点"写 `N:0:k:100`（他脚本里有 `N:0:10:100`/`N:0:20:100`）；
    * **漂移**：按住刹车 -> `D:毫秒`（时长细扫量出来的，比原来猜的准）。
    """
    events: list[ButtonEvent] = []
    for (t0, dur, pulses) in _episodes(samples, 0, gap=gap):        # 刹车（第 2 个字段）
        ms = max(200, min(8000, int(round((dur + 0.05) * 1000))))
        events.append(ButtonEvent(kind="drift", t0=t0, duration=dur, pulses=pulses,
                                  op=f"D:{ms}"))
    for (t0, dur, pulses) in _episodes(samples, 1, gap=gap):        # 氮气（第 3 个字段）
        if pulses >= 2 and dur <= 0.45:
            op = "N:0:2:100"                     # 双击（氮气爆发/完美氮气）
        elif dur >= 0.6:
            k = max(2, min(20, int(round((dur + 0.05) / 0.35))))
            op = f"N:0:{k}:100"                  # 长按 -> 快速连点
        else:
            op = "N:100:1:100"                   # 单击（普通氮气）
        events.append(ButtonEvent(kind="nitro", t0=t0, duration=dur, pulses=pulses, op=op))
    events.sort(key=lambda e: e.t0)
    return events


def attach_events(shots: list[PercentShot], events: list[ButtonEvent],
                  *, max_shift: float = 1.2) -> list[ButtonEvent]:
    """把按键动作落到百分比上（用 shots 的 时间->百分比 关系插值）。"""
    tl = build_timeline([Sample(t=s.t, percent=s.percent) for s in shots])
    kept: list[ButtonEvent] = []
    for e in events:
        p = tl.percent_at(e.t0)
        if p is None:
            continue
        e.percent = float(int(round(p)))
        kept.append(e)
    # 同一个百分比上的同类动作只留一条（细扫可能把一个长按切成两段）
    out: list[ButtonEvent] = []
    for e in kept:
        dup = next((o for o in out if o.kind == e.kind and o.percent == e.percent), None)
        if dup:
            if e.duration > dup.duration:
                out.remove(dup)
                out.append(e)
            continue
        out.append(e)
    out.sort(key=lambda e: e.percent)
    return out


def flat_route_from_events(events: list[ButtonEvent], *,
                           extra_ops: dict | None = None) -> str:
    """把**按键动作**（细扫）+ 其他线索（选路/360/关自动驾驶）写成扁平路线。

    同一个百分比上有多个操作 -> 用 `|` 连起来（用户 2026-09-13 的规则）。
    `extra_ops` 形如 `{40: ["21"], 67: ["360"]}`。
    """
    by_pct: dict[int, list[str]] = {}
    for e in events:
        by_pct.setdefault(int(round(e.percent)), []).append(e.op)
    for pct, ops in (extra_ops or {}).items():
        by_pct.setdefault(int(pct), []).extend(ops)
    parts: list[str] = []
    for pct in sorted(by_pct):
        ops = [o for o in by_pct[pct] if o]
        # **360 时屏蔽漂移**（用户 2026-09-13）：360 要**双击刹车键**，而漂移是**长按同一个键**，
        # 长按会把双击吃掉 —— 所以这个百分点上有 360 就不写漂移（运行时也会再挡一道）。
        if any(o == "360" for o in ops):
            ops = [o for o in ops if not o.startswith("D:")]
        if not ops:
            continue
        parts.append(str(pct))
        parts.append("|".join(ops))
    return ",".join(parts)


def flat_route(ops: list[PercentShot]) -> str:
    """把操作写成**扁平逗号流**（用户 2026-09-13 指定的格式）。

    一整行：`1,31,2,N:0:2:750,4,360,…` —— 和现成脚本一个写法，
    直接粘过去就能用，不用再逐行整理。
    """
    parts: list[str] = []
    for s in ops:
        if s.op:
            parts.append(f"{s.percent:.0f}")
            parts.append(s.op)
    return ",".join(parts)


def suggest_route(shots: list[PercentShot], *, timeline: Timeline | None = None,
                  collapsed: bool = True) -> str:
    """把"每个百分点检测到的操作"写成**可以跑的路线脚本**。

    输出格式：

    * 第一段（**第一行**）：**扁平逗号流**的路线 —— `1,31,2,N:0:2:750,4,360,…`，
      可以直接粘进脚本 / `race run` 直接用；
    * 后面：`#` 开头的说明与"每个百分点的检测明细"，方便核对（解析器会忽略注释）。

    判据优先级：**精细扫描判出的"操作意图"**（`shots[].intents`，能区分
    360 / 漂移 / 打断氮气的短脉冲 / 单击长按双击氮气 / 选路，且带**实测间隔**）优先；
    没有精细扫描结果时退回"窗口汇总"的估算（那种整数时长的结果）。

    用户 2026-09-13 的规则要点：
    * 精细扫描**不产百分点**，只产时间戳；百分比由"时间 -> 百分比"关系事后填；
    * 刹车脉冲**间隔 <200ms 的成对脉冲 = 360**；
    * **独立短刹车脉冲 = 打断氮气**（写成很短的 `D:`）；
    * 氮气多次快速点击 -> **每百分比最多取两次**，并把**两次实测间隔**写进操作
      （`N:0:2:<间隔>`）；OCR 见到「完美氮气」时同样用实测间隔 ✓；
    * 选路：**一直不变取第一次识别到**，**变了取第一次改变时**（不反复选）。
    """
    intents = [it for s in shots for it in getattr(s, "intents", [])]
    if intents:
        kinds: dict[str, int] = {}
        for it in intents:
            kinds[it.kind] = kinds.get(it.kind, 0) + 1
        head_scan = [
            f"# 精细扫描判出 {len(intents)} 个操作："
            f"360 {kinds.get('360', 0)} / 漂移 {kinds.get('drift', 0)} / "
            f"打断氮气 {kinds.get('brake_tap', 0)} / 氮气 {kinds.get('nitro', 0)} / "
            f"选路 {kinds.get('choice', 0)}",
            "#   —— 精细扫描只出时间戳，百分比是事后按「时间->百分比」对齐的；",
            "#      氮气写法里的间隔是**逐帧量出来的实测值**（不是估算）✓",
        ]
        # 360 所在百分比：漂移让位（360 要双击刹车，长按会吃掉双击）
        spin_pcts = {int(round(it.percent)) for it in intents if it.kind == "360"}
        by_pct: dict[int, list[str]] = {}
        for it in intents:
            p = int(round(it.percent))
            if it.kind == "drift" and p in spin_pcts:
                continue
            by_pct.setdefault(p, []).append(it.op)
        extra: dict[int, list[str]] = {}
        for s in shots:
            ops = [v for k, v in (s.values or {}).items() if k in ("auto",)]
            if ops:
                extra.setdefault(int(s.percent), []).extend(ops)
        for p, ops in extra.items():
            by_pct.setdefault(p, []).extend(ops)
        data = ",".join(f"{p}," + "|".join(o for o in ops if o)
                        for p, ops in sorted(by_pct.items()) if any(ops))
        lines = [
            "# 由跑图视频**自动推断**的路线（操作是判据猜的，请核对后使用）",
            *head_scan,
            "# 判据：刹车键**成对脉冲(<200ms)=360**、**独立短脉冲=打断氮气**、按住=漂移；",
            "#       氮气键=正在点氮气（多次连点每百分比最多取两次 + 实测间隔）；",
            "#       选路=上方圆形路标（不变取第一次、变了取第一次改变时）。",
            "# 同一个百分比上的多个操作用 | 连起来（= 同时执行）。",
            "# 用法：核对 -> python -m a9route check --text \"<上面那一行>\"",
            "",
            data if data else "# （没检测到任何操作 —— 确认视频里是比赛画面、且 HUD 完整）",
            "",
            "# ---- 精细扫描判出的操作（时刻 + 判据）----",
        ]
        for it in intents:
            lines.append(f"# {it.describe()}")
        lines.append("")
        lines.append("# ---- 每个百分点的检测明细（核对用）----")
        for s in shots:
            lines.append(f"# {s.describe()}")
        return "\n".join(lines) + "\n"

    events = [e for s in shots for e in getattr(s, "buttons", [])]
    if events:
        # 先把"用没用上细扫"写在头部，避免和粗判据的结果混淆
        # （用户 2026-09-13 就因为这个看错了：整数时长的 `D:4000`/`N:0:12:100`
        #   是"窗口汇总"估出来的，不是逐帧按键量出来的 ✗）
        kinds = {}
        for e in events:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        head_scan = [f"# 按键细扫：{len(events)} 个动作"
                     f"（刹车 {kinds.get('drift', 0)} / 氮气 {kinds.get('nitro', 0)}）",
                     "#   —— 时长与单击/双击都来自逐帧按键，不是估算 ✓"]

        # **360 时屏蔽漂移**：360 要双击刹车键、漂移长按同一个键会吃掉双击 ✗
        spin_pcts = {int(round(s.percent)) for s in shots
                     if any(op.kind == "spin" for op in getattr(s, "buttons", []))} | \
                    {int(round(s.percent)) for s in shots
                     if "spin" in (s.kinds or [])}
        if spin_pcts:
            events = [e for e in events
                      if not (e.kind == "drift" and int(round(e.percent)) in spin_pcts)]
        extra: dict[int, list[str]] = {}
        for s in shots:
            ops = [v for k, v in (s.values or {}).items() if k in ("choice", "spin", "auto")]
            if ops:
                extra[int(s.percent)] = ops
        data = flat_route_from_events(events, extra_ops=extra)
        lines = [
            "# 由跑图视频**自动推断**的路线（操作是判据猜的，请核对后使用）",
            *head_scan,
            "# 判据：**刹车键变红/变白=按住漂移**、**氮气键变红=正在点氮气**（用户截图标的），",
            "#       选路=上方圆形路标（取最后一次高亮）、360=「完成360度旋转」、",
            "#       关自动驾驶=TOUCHDRIVE 关。",
            "# 氮气写法：双击（氮气爆发/完美氮气，100ms 内两下）= N:0:2:100；",
            "#           单击（普通氮气）= N:100:1:100；长按 = N:0:k:100。",
            "# 同一个百分比上的多个操作用 | 连起来（= 同时执行）。",
            "# 用法：核对 -> python -m a9route check --text \"<上面那一行>\"",
            "",
            data if data else "# （没检测到任何操作 —— 确认视频里是比赛画面、且 HUD 完整）",
            "",
            "# ---- 按键动作（细扫，逐帧）----",
        ]
        for e in events:
            lines.append(f"# {e.describe()}")
        lines.append("")
        lines.append("# ---- 每个百分点的检测明细（核对用）----")
        for s in shots:
            lines.append(f"# {s.describe()}")
        return "\n".join(lines) + "\n"

    ops = collapse_runs(shots) if collapsed else shots
    data = flat_route(ops)
    lines = [
        "# 由跑图视频**自动推断**的路线（操作是判据猜的，请核对后使用）",
        "# ⚠️ **本次没用上「按键细扫」** —— 下面是「窗口汇总」判据估出来的，",
        "#    所以漂移/氮气的时长是**整数估算值**（D:4000 / N:0:12:100 这种）✗，",
        "#    不是逐帧按键量出来的真值。想要真值请确认细扫没被跳过（见上面的进度日志）",
        "# 判据：顶部氮气槽青色=氮气 / 右侧「漂移NN米」=漂移 / 上方圆形路标=选路 /",
        "#       「完成360度旋转」=360 / TOUCHDRIVE 关=关自动驾驶",
        "# 说明：连续同类检测已合并成一条（一个岔路口一条、一段氮气一条）",
        "# 用法：核对 -> python -m a9auto race check 本文件 -> race run 本文件",
        "",
        data if data else "# （没检测到任何操作 —— 确认视频里是比赛画面、且 HUD 完整）",
        "",
        "# ---- 每个百分点的检测明细（核对用）----",
    ]
    for s in shots:
        lines.append(f"# {s.describe()}")
    return "\n".join(lines) + "\n"


def draft_route(tl: Timeline, *, step: float = 1.0, start: float | None = None,
                end: float | None = None) -> str:
    """生成**空草稿**：每个整百分点一行，操作留空等你填（时间写在注释里）。

    与 `suggest_route` 的区别：这个不猜操作（只对齐时间），
    适合"我自己看视频写"；`suggest_route` 会把检测到的操作写出来给你核对。
    """
    if not tl.points:
        return "# （视频里没读到任何「路程 NN%」，无法生成草稿）\n"
    p0 = tl.points[0][0] if start is None else start
    p1 = tl.points[-1][0] if end is None else end
    lines = [
        "# 由视频自动生成的路线草稿（百分比 + 时间已对齐；操作请对着视频填）",
        "# 格式：<百分比>,<操作>   —— 操作写法见 routes/demo.txt",
        "#   NN 选路 / N:延时:次数:间隔 氮气 / D:毫秒 漂移 / S:毫秒 关自动驾驶 / 360",
        f"# 视频里覆盖到 {tl.points[0][0]:g}% ~ {tl.points[-1][0]:g}%"
        f"（共 {len(tl.points)} 个采样点）",
        "",
    ]
    p = p0
    while p <= p1 + 1e-6:
        t = tl.time_at(p)
        ts = f"{t:8.2f}s" if t is not None else "     ?  "
        lines.append(f"# {p:5.0f}%  @ {ts}   <- 在这里填操作，然后把这行改成："
                     f"{p:.0f},<操作>")
        p += step
    return "\n".join(lines) + "\n"
