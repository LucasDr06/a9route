# -*- coding: utf-8 -*-
"""analysis.py —— 一次「视频 → 路线」分析的全部编排（CLI 和 Web 共用这一份）。

    from a9route import analysis

    rep = analysis.analyze("录像.mp4", out_dir="worktmp/analysis/demo", progress=print)
    print(rep.route_text)     # 扁平逗号流那一行（在 rep.flat 里）
    for s in rep.shots:       # 每个百分点的截图 + 判据 + 操作
        ...

## 一条分析具体做了三件事（顺序很重要）

1. **粗扫**：按 `scan.every` 抽帧，逐帧算「在做什么」+ 读「路程 NN%」，
   每个整百分点**存一张 PNG** -> 这是给你**人工核对**用的（`rep.shots`）。
2. **精细扫描**（并行另一路解码）：逐帧只看两个按键 + 每 2 帧看一次选路路标，
   **只产时间戳**；用 `core.intent` 按信号形状判目的（成对脉冲=360、按住=漂移…）。
3. **时间 -> 百分比**：用粗扫得到的映射把第 2 步的时间戳落回百分比，
   再用 `vision.video.suggest_route()` 写成**扁平逗号流**（可粘贴的一条）+ 注释明细。

> 为什么第 2 步不直接产百分比：路线脚本的精细度只有 **100 次**，
> 同一段里可能连做好几个动作，光有"这个百分点按键亮着"分不出**这次按键是为了什么**。
> 这是用户 2026-09-13 明确给的规则。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from a9route import config as cfgmod
from a9route import paths


@dataclass
class AnalysisReport:
    """一次分析的结果（`to_json()` 给 Web 前端用）。"""

    video: str = ""
    out_dir: str = ""
    shots: list = field(default_factory=list)      # list[video.PercentShot]
    route_text: str = ""                           # 带注释的完整路线文件内容
    flat: str = ""                                 # **正文那一行**（扁平逗号流）
    button_events: int = 0                         # 细扫判出的按键动作数
    intents: int = 0                               # 细扫判出的"操作目的"数
    frames_scanned: int = 0
    seconds: float = 0.0
    timeline: object = None                        # video.Timeline
    config: dict = field(default_factory=dict)     # 本次生效的配置（排查用）
    warnings: list = field(default_factory=list)
    #: **这次分析无效的原因**（非空 = 结果不可用，而且**不是视频的问题**）。
    #: 典型：OCR 读不到「路程 NN%」 -> 一张图都没扫到 -> 路线必然是空的。
    #: 有这个字段，界面/CLI 才能把它当**错误**报出来，而不是给一条空路线让人以为
    #: 「这视频里没有操作」—— **实测就是这么被误导的**。
    blocker: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.shots) and not self.blocker

    def percent_rows(self) -> list[dict]:
        """每个百分点的精简信息（Web 的截图网格用）。"""
        rows = []
        for s in self.shots:
            rows.append({
                "percent": s.percent,
                "t": round(s.t, 2),
                "op": s.op,
                "cue": (s.cue.describe() if s.cue is not None else ""),
                "kinds": list(s.kinds or []),
                "image": Path(s.path).name if s.path else "",
                "intents": [
                    {"kind": it.kind, "op": it.op, "t0": round(it.t0, 2),
                     "dur": round(it.dur, 2), "pulses": it.pulses,
                     "interval_ms": it.interval_ms, "note": it.note}
                    for it in (getattr(s, "intents", []) or [])
                ],
            })
        return rows

    def to_json(self) -> dict:
        return {
            "video": self.video,
            "out_dir": self.out_dir,
            "route": self.route_text,
            "flat": self.flat,
            "shots": self.percent_rows(),
            "button_events": self.button_events,
            "intents": self.intents,
            "frames_scanned": self.frames_scanned,
            "seconds": round(self.seconds, 2),
            "warnings": list(self.warnings),
            "blocker": self.blocker,
        }

    def describe(self) -> str:
        n_op = len([s for s in self.shots if s.op])
        return (f"扫了 {self.frames_scanned} 帧 / {self.seconds:.1f}s，"
                f"得到 {len(self.shots)} 张百分点截图，"
                f"其中 {n_op} 个百分点了有操作；"
                f"按键细扫 {self.button_events} 个动作, {self.intents} 个操作目的")


def _with_blocker(route_text: str, blocker: str) -> str:
    """把"这次分析为什么无效"**写进 route.txt**（追加成注释块）。

    为什么要这么做（2026-09-24 真踩）：用户从网页拖进来一段录像，扫出 0 张截图，
    而报告里写着「这一局模型没判出任何操作 —— 所以正文是空的（不是「没扫描成功」）」；
    真实原因是**百分比一个都没读到**。事后只能靠"文件里没有明细"倒推 ——
    因为真正的原因（blocker）只在控制台和界面上闪过一次。

    文件是最容易被翻出来、也最容易被发给别人的东西 —— 它必须能**自己解释自己**。
    """
    if not blocker:
        return route_text
    lines = ["", "# ---- ⚠️ 这次分析无效（**不是**「这一局没有操作」）----"]
    lines += ["#   " + ln for ln in blocker.splitlines()]
    lines.append("#   （上面这段也会原样出现在命令行输出和网页的红框里）")
    return route_text.rstrip("\n") + "\n" + "\n".join(lines) + "\n"


def _diagnose_no_shots(video: Path) -> str:
    """**一张百分点截图都没扫到**时，查清楚到底为什么，并给一句能照做的话。

    为什么要专门做这件事（2026-09-15 真踩）：

        用户："现在识别路线识别不到任何操作"

    实际根因跟"检测"毫无关系：`~/.paddlex` 在工作区外，受限环境**读不到 OCR 模型**
    → `RaceReader.percent` 恒为 `None` → 粗扫一张图都存不下 → 路线空。
    而当时只打了两句温和的 warning，看起来就像"这视频里没有操作" ✗✗。

    ⚠️ **只看"中点那一帧"是不行的**（2026-09-24 又踩一次）：用户那段录像在中点那一帧
    恰好读不到字，于是诊断断言"这一帧可能正好在加载/回放画面 —— 换一段比赛中的录像"。
    可**同一份文件重跑完全正常**（99 张截图）—— 等于把"这一次运行的问题"
    说成了"你的视频有问题"，用户会白折腾 ✗。现在改成**抽 5 帧**看整体：
    文件本身读得到，就明说"不是视频的问题，重跑就行"。

    **永远不抛异常**（它本身是错误路径上的诊断）。
    """
    from a9route.ocr import reader as OR

    try:
        local, default = OR.cache_dir(), OR.default_cache_dir()
        local_state, default_state = OR._probe(local), OR._probe(default)
    except Exception as exc:                           # noqa: BLE001
        return "读不到「路程 NN%」，而且连 OCR 状态都查不出来："\
               f"{type(exc).__name__}: {exc}"

    if local_state != "ok" and default_state != "ok":
        return (
            "**OCR 读不到模型，所以一张百分点截图都没扫到 —— 不是视频的问题。**\n"
            f"     工作区内缓存 {local}：{local_state}\n"
            f"     默认缓存   {default}：{default_state}\n"
            f"     修法（在**普通终端**里跑一次即可）：a9route ocr cache\n"
            f"     之后 PaddleOCR 会把模型从工作区内那一份加载，受限环境也能用。")

    # 缓存没问题 -> 在视频上**抽 5 帧**试，把"到底能不能读"和 OCR 自己的报错摊开
    samples = []                                       # [(位置比例, 文本, 错误, 秒)]
    try:
        import cv2

        from a9route import config as _cfg
        from a9route.ocr.reader import OcrReader

        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if n <= 0:
            cap.release()
            return "视频解不出帧（文件坏了，或编码不支持）。"
        h = (_cfg.load_config() or {}).get("hud", {})
        box = h.get("progress_box") or [140, 58, 260, 44]
        x, y, w, hh = (int(v) for v in box)
        o = OcrReader()
        for frac in (0.05, 0.25, 0.45, 0.65, 0.85):
            idx = int(n * frac)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                samples.append((frac, [], "解不出这一帧", idx / fps if fps else 0.0))
                continue
            texts = o.read(frame[y:y + hh, x:x + w])
            samples.append((frac, [t for t, _ in texts], o.last_error or "",
                            idx / fps if fps else 0.0))
        cap.release()
    except Exception as exc:                           # noqa: BLE001
        return f"诊断时出错：{type(exc).__name__}: {exc}"

    ok_samples = [s for s in samples if s[1]]
    detail = "；".join(
        "{0:.0%}处(t={1:.1f}s) {2}".format(
            s[0], s[3],
            ("读到 " + repr(s[1][:2])) if s[1] else ("没读到（" + (s[2] or "空") + "）"))
        for s in samples)

    if ok_samples:
        # ★ 文件本身读得到 —— **别把锅甩给视频**
        return (
            "**这一次运行没读到任何「路程 NN%」，但视频本身是读得到的**"
            "（抽查 {0}/{1} 帧有字）。\n"
            "     抽查：{2}\n"
            "     结论：**不是视频的问题** —— 更像这一次运行出了问题"
            "（例如 OCR 刚加载/并发状态下返回了空结果）。\n"
            "     **直接重跑一次**通常就好；如果每次都这样，把这段文字连同"
            " `worktmp/analysis/` 里那个任务目录一起留下来。"
            .format(len(ok_samples), len(samples), detail))
    return (
        "**抽查的 {0} 帧里，「路程 NN%」那一块一个字都读不到。**\n"
        "     抽查：{1}\n"
        "     可能的原因：① 这一段录像基本都在加载/回放/结算画面（没有比赛 HUD）；\n"
        "               ② 录像不是 1280×720，或「路上 NN%」不在 progress_box 里。\n"
        "     怎么查：`a9route config list hud` 看 progress_box；"
        "或先用 `--max-seconds 30` 只跑一小段。"
        .format(len(samples), detail))


def analyze(video: str | Path, *, out_dir: str | Path | None = None,
            every: float | None = None, max_seconds: float | None = None,
            workers: int | None = None, with_buttons: bool | None = None,
            detector=None, overrides: dict | None = None,
            progress=None) -> AnalysisReport:
    """跑一次完整分析。**所有参数不传就用 config.json 的值。**

    `overrides`：形如 `{"vision__nitro_red_thr": 0.22}` —— **只影响这一次运行**
    （CLI 的 `--set` 就是这个），不会写进 config.json。
    """
    import time

    from a9route.vision import video as V

    t0 = time.perf_counter()
    # ---- 配置：config.json（含环境变量） -> 本次覆盖 -> 灌进各模块 ----
    cfg = cfgmod.load_config()
    for key, val in (overrides or {}).items():
        section, _, name = key.replace(".", "__").partition("__")
        if not name:
            raise ValueError(f"覆盖项要写成 段__键（例如 vision__nitro_red_thr），收到 {key!r}")
        cfg.setdefault(section, {})[name] = val
    cfgmod.apply(cfg)
    scan = cfg.get("scan", {})

    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"找不到视频：{video}")
    every = float(scan.get("every", 0.25) if every is None else every)
    if every <= 0:
        raise ValueError("--every 必须是正数（秒）")
    if max_seconds is None:
        ms = float(scan.get("max_seconds") or 0)
        max_seconds = ms if ms > 0 else None
    workers = int(scan.get("workers", 2) if workers is None else workers)
    with_buttons = bool(scan.get("with_buttons", True) if with_buttons is None
                        else with_buttons)

    out_dir = Path(out_dir) if out_dir else (paths.ANALYSIS_DIR / video.stem)
    shots_dir = out_dir / "shots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        progress(f"开始分析 {video.name}（每 {every:g}s 一帧"
                 f"{'' if max_seconds is None else f'，只看前 {max_seconds:g}s'}）…")

    # ⚠️ **模型起不来 = 这次分析无效，而不是"退回启发式接着跑"。**
    # 本项目现在的口径是「所有判定都交给模型」：按键框/选路条数都必须来自
    # `models/*.onnx`。起不来时 `extract_per_percent` 会抛错，这里把它变成
    # 一个**blocker**（CLI 红字 + 退出码 1、界面弹红框），
    # 而不是让它变成一个看起来正常、其实是像素判据的路线 ✗
    try:
        shots = V.extract_per_percent(
            video, out_dir=shots_dir, every=every, detector=detector,
            max_seconds=max_seconds, with_buttons=with_buttons, workers=workers,
            progress=progress,
        )
    except RuntimeError as exc:
        rep = AnalysisReport(
            video=str(video), out_dir=str(out_dir), shots=[],
            route_text="", flat="", frames_scanned=0,
            seconds=time.perf_counter() - t0,
            config={"scan": scan, "vision": cfg.get("vision", {}),
                    "intent": cfg.get("intent", {}), "hud": cfg.get("hud", {})},
        )
        rep.blocker = str(exc)
        rep.warnings.append("这次分析**无效**：判定必须来自模型，而模型起不来")
        (out_dir / "route.txt").write_text(_with_blocker("", rep.blocker),
                                           encoding="utf-8")
        if progress:
            progress("  ⚠️ " + rep.blocker.replace("\n", "\n  "))
        return rep
    # ⚠️ `suggest_route` 在没有"模型判出的操作"时会**拒绝出路线**（不再用窗口汇总
    # 那套像素/文字判据估一条）。这里把它变成 blocker（红字 + 退出码 1 + 界面红框），
    # 而不是变成一条看着像路线、其实是另一套判据的东西 ✗
    try:
        # `fine_scan_ran=with_buttons`：模型那一遍跑过了的话，"没有操作"就是一个
        # 合法结论（给空路线）；没跑的话 `suggest_route` 会拒绝出路线（不估算）
        route_text = V.suggest_route(shots, fine_scan_ran=with_buttons)
    except RuntimeError as exc:
        rep = AnalysisReport(
            video=str(video), out_dir=str(out_dir), shots=shots,
            route_text="", flat="", frames_scanned=len(shots),
            seconds=time.perf_counter() - t0,
            config={"scan": scan, "vision": cfg.get("vision", {}),
                    "intent": cfg.get("intent", {}), "hud": cfg.get("hud", {})},
        )
        rep.blocker = str(exc)
        rep.warnings.append("这次分析**无效**（没有模型判出的操作，不给路线）")
        # ⚠️ **不要写空文件**：空 route.txt 是最难查的一种状态 ——
        #    "文件在、但是空的"，看不出是"没操作"还是"根本没跑成"。
        #    把原因写成注释放进去（`_with_blocker` 干这个），文件自己就能解释自己。
        (out_dir / "route.txt").write_text(_with_blocker("", rep.blocker),
                                           encoding="utf-8")
        if progress:
            progress("  ⚠️ " + rep.blocker.replace("\n", "\n  "))
        return rep
    flat = _flat_line(route_text)
    n_btn = sum(len(getattr(s, "buttons", []) or []) for s in shots)
    n_it = sum(len(getattr(s, "intents", []) or []) for s in shots)

    rep = AnalysisReport(
        video=str(video), out_dir=str(out_dir), shots=shots,
        route_text=route_text, flat=flat, button_events=n_btn, intents=n_it,
        frames_scanned=len(shots), seconds=time.perf_counter() - t0,
        config={"scan": scan, "vision": cfg.get("vision", {}),
                "intent": cfg.get("intent", {}), "hud": cfg.get("hud", {})},
    )
    # ⚠️ **一张图都没扫到 = 这次分析无效，必须响亮地说清原因。**
    # 不能只丢一句温和的 warning —— 用户看到的就是"识别不到任何操作"，
    # 会去查检测/阈值，而真凶往往是 OCR 读不到模型（见 `_diagnose_no_shots`）。
    if not shots:
        rep.blocker = _diagnose_no_shots(video)
        rep.warnings.append("这次分析**无效**（没扫到任何百分比）：见下面的原因")
        # ⚠️ **把原因写进 route.txt**：文件是会被人翻出来看的（这次排查就是靠它还原现场）。
        #    只把 blocker 放在 API 响应/控制台里的话，磁盘上就留下一份
        #    **声称自己没事的报告** —— 那比没有报告更坏 ✗
        route_text = _with_blocker(route_text, rep.blocker)
    # ⚠️ 这里原来只看 `n_btn`（`s.buttons`），而 `s.buttons` **只有旧的
    # `vision.attach_events()` 那条路会填**；现在的主线（`extract_per_percent`）
    # 填的是 `s.intents`。于是 `n_btn` 恒为 0 —— 哪怕精细扫描明明判出了几十个操作，
    # 也会打一句"按键细扫没有结果" ✗✗。**误导性的警告正是用户最怕的东西**
    # （README/NOTES 里记着：他就因为看错"用没用上按键细扫"把估算值当成真值），
    # 所以两个都算：**真的两边都没有**才是没有结果。
    if with_buttons and shots and n_btn == 0 and n_it == 0:
        rep.warnings.append("按键细扫没有结果（阈值可能不合适，或视频里没有按键动作）")
    (out_dir / "route.txt").write_text(route_text, encoding="utf-8")
    if progress:
        if rep.blocker:
            progress("⚠️ 这次分析无效：" + rep.blocker.splitlines()[0])
        progress("完成：" + rep.describe())
        progress(f"路线文件：{out_dir / 'route.txt'}")
    return rep


def _flat_line(route_text: str) -> str:
    """从生成的路线文本里取出**正文那一行**（扁平逗号流；`#` 是注释）。"""
    for line in route_text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return s
    return ""


def analyze_to_route(video: str | Path, out_file: str | Path | None = None,
                     **kw) -> AnalysisReport:
    """便捷入口：分析并把路线写成文件（默认 `<视频名>.route.txt`）。"""
    rep = analyze(video, **kw)
    target = Path(out_file) if out_file else Path(video).with_suffix(".route.txt")
    target.write_text(rep.route_text, encoding="utf-8")
    return rep
