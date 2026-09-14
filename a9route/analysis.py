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

    @property
    def ok(self) -> bool:
        return bool(self.shots)

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
        }

    def describe(self) -> str:
        n_op = len([s for s in self.shots if s.op])
        return (f"扫了 {self.frames_scanned} 帧 / {self.seconds:.1f}s，"
                f"得到 {len(self.shots)} 张百分点截图，"
                f"其中 {n_op} 个百分点了有操作；"
                f"按键细扫 {self.button_events} 个动作, {self.intents} 个操作目的")


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

    shots = V.extract_per_percent(
        video, out_dir=shots_dir, every=every, detector=detector,
        max_seconds=max_seconds, with_buttons=with_buttons, workers=workers,
        progress=progress,
    )
    route_text = V.suggest_route(shots)
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
    if with_buttons and not shots:
        rep.warnings.append("没扫到任何百分比 —— 确认视频里是比赛画面、HUD 完整")
    if with_buttons and n_btn == 0:
        rep.warnings.append("按键细扫没有结果（阈值可能不合适，或视频里没有按键动作）")
    (out_dir / "route.txt").write_text(route_text, encoding="utf-8")
    if progress:
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
