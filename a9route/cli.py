# -*- coding: utf-8 -*-
"""cli.py —— 命令行入口（`python -m a9route <子命令>` 或 `a9route <子命令>`）。

    a9route video   录像.mp4                 # 只算「时间 -> 路程百分比」的时间轴 + 空草稿
    a9route analyze 录像.mp4                 # 【主命令】逐百分点截图 + 推断操作 + 生成路线
    a9route check   --text "1,31,2,N:0:2:750"  # 校验路线（不碰视频、不碰设备）
    a9route show    routes/demo.txt          # 打印解析结果 + 警告
    a9route serve                            # 起本地 Web 窗口（默认 http://127.0.0.1:8790/）
    a9route routes                           # 列出能分析的视频
    a9route config  list|set|reset           # 调参（**不用改代码**）
    a9route ocr     status|cache             # OCR 模型缓存体检 / 镜像进工作区
    a9route test    [all|video|intent|web…]  # 离线回归

所有子命令都会先 `config.apply()`，所以 `config.json` 改了什么立刻生效。

⚠️ **数据集与训练不在这个项目里**（2026-09-15 拆出去了）：
抽帧 / 标注 / 体检 / 训练 / 评估 / 导出都是另一个项目 **`a9lab`** 的活：

    a9lab serve                     # 网页工作台：概览 / 数据集 / 标注 / 体检 / 训练 / 模型
    a9lab build 录像.mp4 --name v3  # 抽帧 + 预标注
    a9lab run --name v3             # 训练（要在有 torch 的环境里）
    a9lab install v3.onnx           # 装回本项目（会核对类别顺序）

本项目只留**运行时**：录像 -> 路线，外加它需要的判据、HUD 读数、ONNX 检测器，
以及和模型之间的**格式契约**（`a9route/formats.py`）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from a9route import __version__, config as cfgmod, paths
from a9route.bootstrap import configure_console
from a9route.core import route as R

VIDEO_SUFFIXES = (".mp4", ".mkv", ".avi", ".mov", ".webm")


# ---------------------------------------------------------------- 路线文本来源
def _route_text(args) -> str:
    """取路线：`--text` 直接粘的正文 或 位置参数（含逗号当文本、否则当文件）。

    与工程里原来的约定一致：**粘贴文本优先**（用户要求过不再强制读文件），
    文件路径继续可用，老脚本不用改。
    """
    text = (getattr(args, "text", None) or "").strip()
    if text:
        return text
    pos = (getattr(args, "file", None) or "").strip()
    if not pos:
        raise R.RouteError('没给路线：用 --text "<路线文本>" 直接粘贴，或给一个路线文件路径')
    if "," in pos:
        return pos
    p = Path(pos)
    if not p.is_file():
        raise FileNotFoundError(f"找不到路线文件：{p}")
    return p.read_text(encoding="utf-8")


def _report_route(route: R.Route, verbose: bool = False) -> None:
    if verbose:
        print(route.describe())
    else:
        print(f"路线共 {len(route.entries)} 条操作：")
        for e in route.entries:
            print(f"  {e.percent:>5g}%  " + " + ".join(op.describe() for op in e.ops))
    if route.warnings:
        print("⚠️ 警告：")
        for w in route.warnings:
            print(f"  ! {w}")


# ---------------------------------------------------------------- 子命令
def cmd_video(args) -> int:
    """只做**可靠的那一半**：抽帧读「路程 NN%」-> 时间轴 -> 空草稿（操作留给你填）。"""
    from a9route.vision import video as V

    every = args.every if args.every is not None else float(
        cfgmod.load_config()["scan"]["every"])
    print(f"抽帧读「路程 NN%」（每 {every:g}s 一帧）…")
    try:
        samples = V.sample_video(args.video, every=every, max_seconds=args.max_seconds)
    except Exception as exc:
        print(f"读视频失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    tl = V.build_timeline(samples)
    print(tl.describe())
    if not tl.points:
        print("视频里没读到「路程 NN%」—— 确认画面里有比赛 HUD（左上角路程）。\n"
              "（这不是错误：加载/赛道介绍那几屏本来就什么都读不到）", file=sys.stderr)
        return 1
    draft = V.draft_route(tl)
    if args.out:
        Path(args.out).write_text(draft, encoding="utf-8")
        print(f"\n草稿已写到 {args.out}\n"
              f"下一步：对着视频在每个百分比后面填操作，然后 a9route check {args.out}")
    else:
        print("\n" + draft)
    return 0


def cmd_analyze(args) -> int:
    from a9route import analysis

    overrides = _parse_sets(args.set or [])
    try:
        rep = analysis.analyze(
            args.video, out_dir=args.out_dir, every=args.every,
            max_seconds=args.max_seconds, workers=args.workers,
            with_buttons=(None if args.with_buttons is None
                          else str(args.with_buttons).lower() in ("1", "true", "on", "yes", "开")),
            overrides=overrides, progress=print,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"分析失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\n" + "=" * 60)
    if rep.blocker:
        # **这次分析无效** —— 说清楚原因，别让人以为"视频里没有操作"
        print("⚠️ 这次分析无效（不是视频的问题）：")
        print("   " + rep.blocker.replace("\n", "\n   "))
        print("=" * 60)
        return 1
    print(f"路线正文（{len(rep.flat.split(',')) // 2 if rep.flat else 0} 条操作）：")
    print(rep.flat or "# （没检测到任何操作）")
    print("=" * 60)
    for w in rep.warnings:
        print(f"⚠️ {w}")
    if args.write:
        target = Path(args.write)
        target.write_text(rep.route_text, encoding="utf-8")
        print(f"完整路线（含逐条判据注释）已写到 {target}")
    # 逐百分点明细（核对用）
    if args.verbose:
        for s in rep.shots:
            print("  " + s.describe())
    print(f"\n截图目录：{Path(rep.out_dir) / 'shots'}")
    if rep.flat:
        print("核对完就可以直接拿上面那一行去跑图脚本；要校验格式：")
        print(f'  python -m a9route check --text "{rep.flat}"')
    else:
        print("这次没识别出任何操作 —— 先确认视频里是比赛画面且 HUD 完整"
              "（左上角有「路程 NN%」），或试试 --set vision__nitro_red_thr=0.12 放宽判据")
    return 0 if rep.ok else 1


def cmd_check(args) -> int:
    try:
        route = R.parse_route(_route_text(args))
    except R.RouteError as exc:
        print(f"路线脚本有问题：{exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    _report_route(route, verbose=args.verbose)
    return 0


def cmd_serve(args) -> int:
    """起本地 Web 窗口（**在 `main()` 这个进程里跑**，Ctrl+C 结束）。

    ⚠️ 它是个**阻塞**调用：`web.app.main()` 里 `app.run()` 一直占着，
    所以 `python main.py` / `python -m a9route serve` 跑起来之后，
    这个终端窗口就是服务的宿主 —— 关掉终端（或被别人 kill 掉）服务就没了，
    浏览器随即变成"拒绝连接"。
    """
    from a9route.web.app import main as web_main

    return web_main(["--host", args.host, "--port", str(args.port)]
                    + (["--no-browser"] if args.no_browser else []))


def cmd_routes(args) -> int:
    """列出「能分析的视频」：默认扫 `output/`，另有 `A9ROUTE_VIDEO_DIR` 里的目录。"""
    found = 0
    for d in paths.video_dirs():
        print(f"[{d}]{'' if d.is_dir() else '  （不存在）'}")
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in VIDEO_SUFFIXES:
                mb = p.stat().st_size / 1024 / 1024
                print(f"  {p.name:<40} {mb:8.1f} MB")
                found += 1
    if not found:
        print("\n没有找到视频。把录像放到 output/ 下，或设 A9ROUTE_VIDEO_DIR 指到别的目录。")
    return 0


def _parse_sets(items: list[str]) -> dict:
    """`段__键=值` -> dict（值按内置默认的类型转换）。"""
    flat_defaults = cfgmod.flat(cfgmod.load_config(), env=False)
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set 要写成 段__键=值，收到 {item!r}")
        key, raw = item.split("=", 1)
        key = key.replace(".", "__")
        out[key] = cfgmod.coerce(raw, flat_defaults.get(key))
    return out


def cmd_config(args) -> int:
    action = args.action or "list"
    if action == "list":
        print(cfgmod.describe(changed_only=args.changed))
        print("\n改法：a9route config set vision__nitro_red_thr=0.22"
              "   （或直接编 config.json；环境变量 A9ROUTE_vision__nitro_red_thr 优先级最高）")
        return 0
    if action == "get":
        flat = cfgmod.flat(cfgmod.load_config())
        if not args.keys:
            for k in sorted(flat):
                print(f"{k} = {flat[k]!r}")
            return 0
        rc = 0
        for k in args.keys:
            k = k.replace(".", "__")
            if k not in flat:
                print(f"没有这个键：{k}", file=sys.stderr)
                rc = 2
                continue
            print(f"{k} = {flat[k]!r}")
        return rc
    if action in ("set", "reset"):
        cfg = cfgmod.load_config()
        if action == "reset":
            keys = [k.replace(".", "__") for k in (args.keys or [])]
            if not keys:
                f = paths.CONFIG_FILE
                if f.is_file():
                    f.unlink()
                print(f"已恢复全部内置默认值（删掉了 {f}）")
                return 0
            # 只重置指定的键：把它从 config.json 里删掉（回到内置默认）
            raw = {}
            if paths.CONFIG_FILE.is_file():
                import json
                try:
                    raw = json.loads(paths.CONFIG_FILE.read_text(encoding="utf-8")) or {}
                except Exception:
                    raw = {}
            for k in keys:
                section, _, name = k.partition("__")
                raw.get(section, {}).pop(name, None)
            cfgmod.save_config(raw)
            print(f"已重置 {len(keys)} 个键（回到内置默认）")
            return 0
        if not args.sets and not args.keys:
            print("用法：a9route config set vision__nitro_red_thr=0.22 scan__every=0.2",
                  file=sys.stderr)
            return 2
        sets = _parse_sets(args.sets or args.keys)
        for k, v in sets.items():
            section, _, name = k.partition("__")
            if not name or section not in cfgmod.DEFAULTS:
                print(f"未知配置段：{k}（可用段：{', '.join(cfgmod.DEFAULTS)}）", file=sys.stderr)
                return 2
            if name not in cfgmod.DEFAULTS[section]:
                print(f"未知配置键：{k}"
                      f"（{section} 段可用：{', '.join(sorted(cfgmod.DEFAULTS[section]))}）",
                      file=sys.stderr)
                return 2
            cfg[section][name] = v
        f = cfgmod.save_config(cfg)
        print(f"已写入 {f}")
        for k, v in sets.items():
            print(f"  {k} = {v!r}")
        return 0
    if action == "edit":
        print(f"配置文件：{paths.CONFIG_FILE}")
        return 0
    print(__doc__)
    return 2


# ---------------------------------------------------------------- train（YOLOv8 训练框架）
def _merge_overrides(overrides: dict) -> dict:
    """把扁平覆盖项并进 config（`train` 子命令只有 `--set`，没有完整 CLI 参数）。"""
    cfg = cfgmod.load_config()
    for key, val in overrides.items():
        section, _, keyname = key.replace(".", "__").partition("__")
        if not keyname:
            raise SystemExit(f"--set 要写成 段__键=值，收到 {key!r}")
        cfg.setdefault(section, {})[keyname] = val
    return cfg


def _backend_line() -> str:
    try:
        from a9route.vision import keys as keymod
        return keymod.describe_backend()
    except Exception as exc:
        return f"（读不出来：{type(exc).__name__}: {exc}）"


def _resolve_backend_safe():
    """`keys.resolve_backend()` 的安全版（拿不到就返回空，不抛）。"""
    from a9route.vision import keys as keymod
    return keymod.resolve_backend()


def cmd_ocr(args) -> int:
    """`a9route ocr status|cache` —— OCR 模型缓存的体检与**一次性镜像**。

    为什么要这个命令：`~/.paddlex` 在**工作区外**。受限环境里读不到它 →
    `OcrReader` 全程返回 `[]` → 读不到「路程 NN%」→ 粗扫一张图都存不下 →
    **路线是空的**，看起来像"检测不出操作"，其实跟检测毫无关系 ✗✗。

    跑一次 `a9route ocr cache` 把模型抄进工作区之后，受限环境也能用。
    """
    from a9route.ocr import reader as R

    action = args.action or "status"
    local, default = R.cache_dir(), R.default_cache_dir()
    print("OCR 模型缓存")
    print(f"  工作区内: {local}")
    print(f"     状态:  {R._probe(local)}")
    print(f"  默认位置: {default}")
    print(f"     状态:  {R._probe(default)}")
    if action == "status":
        if R._probe(local) == "ok":
            print("\n工作区里已有一份可用的模型 —— 受限环境也能跑 OCR ✓")
        elif R._probe(default) == "ok":
            print("\n默认那份读得到，正常环境直接可用（不需要镜像）。")
        else:
            print("\n⚠️ 读不到 OCR 模型：现在跑 analyze 会**读不到「路程 NN%」**，")
            print("   路线会是空的（看起来像「没检出操作」，其实跟检测无关）。")
            print("   修法（在**普通终端**里跑一次即可）：a9route ocr cache")
        return 0
    if action == "cache":
        try:
            chosen = R.ensure_cache(progress=print, force=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"镜像失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(f"\n完成：工作区里已有一份模型 -> {R.cache_dir()}")
        print("之后**受限环境也能跑 OCR** 了（不用再抄）。")
        print(f"（想让 PaddleOCR 用它：PADDLE_PDX_CACHE_HOME={R.cache_dir()}）")
        return 0
    print(__doc__)
    return 2


def cmd_test(args) -> int:
    import os
    import subprocess

    name = args.name or "all"
    #: 别名 -> 真实模块名。**注意 `web` 指的是 `test_webvideo`**（历史上叫 test_web，
    #: 后来加了视频端到端就叫 webvideo 了；CLI 上仍然叫 `web`，因为人是这么记的）。
    #: 数据集/训练那几个模块（曾经的 `train`）已经搬去 a9lab 了。
    mods = {"video": "a9route.tests.test_video",
            "intent": "a9route.tests.test_intent",
            "route": "a9route.tests.test_route",
            "web": "a9route.tests.test_webvideo",
            "config_ui": "a9route.tests.test_config_ui",
            "ocr": "a9route.tests.test_ocr"}
    if name == "all":
        names = ["route", "intent", "video", "web", "config_ui", "ocr"]
    elif name in mods:
        names = [name]
    else:
        print(f"未知测试 {name!r}；可用：all, {', '.join(mods)}", file=sys.stderr)
        return 2
    env = dict(os.environ)
    # 子进程要能 import a9route：把项目根放进 PYTHONPATH（cwd 已经在项目根，
    # 但显式给一份更稳 —— 在别的目录下跑也不会突然 import 失败）
    env["PYTHONPATH"] = str(paths.ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    worst = 0
    for n in names:
        print(f"\n{'=' * 62}\n>>> 测试: {n}\n{'=' * 62}")
        rc = subprocess.run([sys.executable, "-m", mods[n]],
                            cwd=str(paths.ROOT), env=env).returncode
        worst = worst or rc
    return worst


# ---------------------------------------------------------------- 主入口
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="a9route",
        description="跑图视频 → 路线：从比赛录像里推断操作，生成百分比路线脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("-V", "--version", action="version",
                   version=f"a9route {__version__}")
    sub = p.add_subparsers(dest="cmd")

    a = sub.add_parser("video", help="只算时间轴 + 空草稿（可靠的那一半）")
    a.add_argument("video")
    a.add_argument("--every", type=float, default=None, help="抽帧间隔秒（默认取 config）")
    a.add_argument("--max-seconds", type=float, default=None, dest="max_seconds")
    a.add_argument("--out", default="", help="把草稿写到文件")
    a.set_defaults(func=cmd_video)

    a = sub.add_parser("analyze", help="【主命令】逐百分点截图 + 推断操作 + 生成路线")
    a.add_argument("video")
    a.add_argument("--out-dir", default=None, dest="out_dir",
                   help="产物目录（默认 worktmp/analysis/<视频名>）")
    a.add_argument("--every", type=float, default=None, help="粗扫抽帧间隔秒")
    a.add_argument("--max-seconds", type=float, default=None, dest="max_seconds",
                   help="只分析前 N 秒（试跑用）")
    a.add_argument("--workers", type=int, default=None, help="并行度（默认 2）")
    a.add_argument("--with-buttons", default=None, dest="with_buttons",
                   help="是否做按键细扫（默认开；关掉会退回估算，不准）")
    a.add_argument("--set", action="append", default=[],
                   help="临时覆盖配置，可重复：--set vision__nitro_red_thr=0.22")
    a.add_argument("--write", default="", help="把完整路线（含注释）写到文件")
    a.add_argument("--verbose", action="store_true", help="打印每个百分点的明细")
    a.set_defaults(func=cmd_analyze)

    a = sub.add_parser("check", help="校验路线脚本（解析 + 警告）")
    a.add_argument("file", nargs="?", default="", help="路线文件（含逗号时当文本）")
    a.add_argument("--text", default="", help="直接粘贴路线文本")
    a.add_argument("--verbose", action="store_true", help="打印每条操作的完整说明")
    a.set_defaults(func=cmd_check, verbose=False)

    a = sub.add_parser("show", help="打印路线解析结果（每条明细 + 警告）")
    a.add_argument("file", nargs="?", default="", help="路线文件（含逗号时当文本）")
    a.add_argument("--text", default="", help="直接粘贴路线文本")
    a.set_defaults(func=cmd_check, verbose=True)

    a = sub.add_parser("serve", help="起本地 Web 窗口")
    a.add_argument("--host", default="127.0.0.1")
    a.add_argument("--port", type=int, default=8790)
    a.add_argument("--no-browser", action="store_true", dest="no_browser")
    a.set_defaults(func=cmd_serve)

    a = sub.add_parser("routes", help="列出能分析的视频")
    a.set_defaults(func=cmd_routes)

    a = sub.add_parser("config", help="调参：list / get / set / reset")
    a.add_argument("action", nargs="?", default="list",
                   choices=["list", "get", "set", "reset", "edit"])
    a.add_argument("keys", nargs="*", help="get/reset 用的键")
    a.add_argument("--set", action="append", default=[], dest="sets",
                   help="set 用：段__键=值")
    a.add_argument("--changed", action="store_true", help="list 只看被覆盖的项")
    a.set_defaults(func=cmd_config)

    a = sub.add_parser("test", help="离线回归测试")
    a.add_argument("name", nargs="?", default="all")
    a.set_defaults(func=cmd_test)

    a = sub.add_parser("ocr", help="OCR 模型缓存：体检 / 一次性镜像到工作区")
    a.add_argument("action", nargs="?", default="status",
                   choices=["status", "cache"],
                   help="status=看能不能读到；cache=把模型抄进工作区（受限环境用）")
    a.set_defaults(func=cmd_ocr)

    return p


def main(argv: list[str] | None = None) -> int:
    configure_console()
    raw = list(sys.argv[1:] if argv is None else argv)
    # ⚠️ **不带参数 = 起本地 Web 窗口**（`serve`）。
    #
    # 为什么：`main.py` 顶部的用法说明、README 里那句"`python main.py` 与
    # `python -m a9route` 等价"，一直承诺的是"直接跑就是那个 Web 窗口"
    # （双击/直接跑的用法）。但实际上这里以前只 `print_help()` —— 用户按文档跑，
    # 得到一屏帮助，然后去浏览器发现"拒绝连接" ✗✗（文档骗人比报错更坏）。
    # 想看帮助用 `--help` / `-h`，语义没丢。
    if not raw:
        raw = ["serve"]
    parser = build_parser()
    args = parser.parse_args(raw)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
