# -*- coding: utf-8 -*-
"""cli.py —— 命令行入口（`python -m a9route <子命令>` 或 `a9route <子命令>`）。

    a9route video   录像.mp4                 # 只算「时间 -> 路程百分比」的时间轴 + 空草稿
    a9route analyze 录像.mp4                 # 【主命令】逐百分点截图 + 推断操作 + 生成路线
    a9route check   --text "1,31,2,N:0:2:750"  # 校验路线（不碰视频、不碰设备）
    a9route show    routes/demo.txt          # 打印解析结果 + 警告
    a9route serve                            # 起本地 Web 窗口（默认 http://127.0.0.1:8790/）
    a9route routes                           # 列出能分析的视频
    a9route config  list|set|reset|edit      # 调参（**不用改代码**）
    a9route train   doctor|build|review|run… # 【YOLOv8】按键识别的训练框架
    a9route test    [all|video|intent|web]   # 离线回归

所有子命令都会先 `config.apply()`，所以 `config.json` 改了什么立刻生效。

按键识别默认还是**启发式**（和以前逐字一致）；换成模型是显式的：

    a9route train build output\\录像.mp4      # 抽帧 + 启发式预标注
    a9route train review                      # 人工复核（**模型的增量在这一步**）
    a9route train run                         # 训练（要在有 torch 的环境里）
    a9route train export                      # 导出 ONNX
    a9route analyze 录像.mp4 --set vision__key_backend=onnx
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
def _dataset_task(name: str) -> str:
    """这个数据集是哪个任务（`keys` = 刹车/氮气，`choice` = 选路）。

    从 `dataset.json` 的 `task` 字段读（`train choice-build` 会写）。
    这决定 **eval 用哪套指标、export 用哪套类别** —— 选路是按"三个答案"评的，
    塞进按键那套 per-class F1 里毫无意义。
    """
    try:
        from a9route.train.dataset import dataset_dir, load_meta
        return str((load_meta(dataset_dir(name)) or {}).get("task") or "keys").lower()
    except Exception:                                  # noqa: BLE001
        return "keys"


def _run_name(raw, name: str) -> str:
    """训练产物目录名（`models/runs/<这个名字>/`）。

    显式给了 `--run-name` 就听显式的，否则**跟数据集同名**。

    ⚠️ 这里曾经写死默认 `"keys"`，`run`/`eval`/`export` 三处都用它 ——
    于是 `train run --name choice` 把选路的权重写进 `models/runs/keys/`，
    而 `train export --name choice` 又去同一个地方拿权重，
    等于**把按键的模型导出成选路模型**（2026-09-15 实测踩到，见 NOTES 10.27）。
    """
    return (raw or "").strip() or name


def cmd_train(args) -> int:
    """`a9route train <子命令>` —— 见 `a9route.train` 的总览。

    所有子命令都会先 `config.apply()`（和别的入口一样）。数据集构建/复核只要
    cv2/numpy，训练/导出才需要 torch —— 缺哪个环境由 `train doctor` 和
    `runner` 自己的报错说清楚，CLI 层不假装。
    """
    sub = args.action or "doctor"
    # 数据集名的默认值**跟着子命令走**：选路那几条默认 `choice`，其余默认 `keys`。
    # （`choice-stats` 曾经读到按键数据集 —— 因为 `--name` 的默认值是 `keys`，
    #   于是报出"有岔路口 0"这种看着像 bug 的数 ✗）
    _raw_name = (getattr(args, "name", None) or "").strip()
    name = _raw_name or ("choice" if sub.startswith("choice") else "keys")
    # 训练产物目录也跟着数据集走（见 `--run-name` 的说明：写死 keys 会把两个任务的
    # 权重混在一个目录里，导出时会拿错）
    if hasattr(args, "run_name"):
        args.run_name = _run_name(getattr(args, "run_name", None), name)
    overrides = _parse_sets(getattr(args, "set", None) or [])
    # ⚠️ **必须无条件 apply()**，不能只在有 --set 时才调 ——
    # 这条是真踩出来的：`cues.BRIGHT_THR` 的**模块默认值是 165**，而
    # `DEFAULTS` 里是 **150**。预标注（`dataset.build`）调了 apply()，所以标注是按
    # 150 打的；而 `train eval` 当初没调 apply()，于是拿 165 去复算启发式，
    # **刹车召回凭空掉了 35%**，看着像"启发式很弱、模型大胜"，其实是
    # **两个入口的口径不一致** ✗✗（`config.py` 的模块文档早就写了
    # "任何入口都必须先调一次 apply()"，这里当初漏了）。
    cfgmod.apply(_merge_overrides(overrides) if overrides else None)

    if sub == "doctor":
        from a9route.train import doctor
        rep = doctor.run(name=name)
        print(rep.describe())
        print("\n" + _backend_line())
        # 退出码只反映**关键项**（cv2/numpy 这类"跑都跑不起来"的）。
        # "这个环境没有 torch"在运行环境里是**预期状态**，不该算失败。
        n_missing = len(rep.missing)
        if rep.ok and n_missing:
            print("\n（上面标「缺」的 {0} 项是**这个环境的正常状态** —— "
                  "训练相关的依赖在另一个环境里，见 TRAINING.md §3）".format(n_missing))
        return 0 if rep.ok else 1

    if sub == "fetch":
        from a9route.train import runner
        print(f"权重目录：{paths.WEIGHTS_DIR}")
        got = runner.ensure_weights(args.weight or runner.DEFAULT_WEIGHT, progress=print)
        ok = got.is_file()
        print(f"{'已就绪' if ok else '没下到'}：{got}")
        if not ok:
            print("下不动也不致命：`a9route train run --weight yolov8n.yaml` "
                  "可以从零训练（ultralytics 内置配置，不需联网）")
        return 0 if ok else 1

    if sub == "build":
        from a9route.train import dataset
        if not args.videos:
            print("要给它录像：a9route train build output\\录像.mp4 [更多…]",
                  file=sys.stderr)
            return 2
        try:
            rep = dataset.build(
                args.videos, name=name, budget=args.budget,
                per_video_budget=args.per_video,
                max_seconds=args.max_seconds, every=args.every,
                edge_pad=args.edge_pad, margin_band=args.margin_band,
                edge_share=args.edge_share,
                jpeg_quality=args.jpeg_quality, seed=args.seed,
                progress=print)
        except Exception as exc:
            print(f"构建失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print("\n" + rep.describe())
        print("\n下一步（**这一步才是模型的增量**）：")
        print(f"  python -m a9route train review --name {name}   # 生成复核页，改掉标错的")
        return 0

    if sub == "stats":
        from a9route.train import dataset
        try:
            print(dataset.stats(name=name).describe())
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0

    if sub == "verify":
        from a9route.train import dataset
        problems = dataset.verify(name=name)
        if not problems:
            print("体检通过：没发现问题。")
            print(dataset.stats(name=name).describe())
            return 0
        print(f"发现 {len(problems)} 个问题：")
        for p in problems:
            print(f"  ⚠️ {p}")
        return 1

    if sub == "audit":
        # `verify` 只看数据集自己（快、只读元数据）；
        # `audit` 还要回答"标注和**当前标定**对不对得上"——会抽样解图重算判据。
        from a9route.train import audit as A
        try:
            rep = A.audit(name=name, recheck=not args.no_recheck,
                          limit=args.limit or 300)
        except Exception as exc:
            print(f"体检失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(rep.describe())
        print(f"\n看图形界面：a9route serve  然后打开 http://127.0.0.1:8790/dataset")
        return 0 if rep.ok else 1

    if sub == "review":
        from a9route.train import review
        try:
            r = review.make_review(name=name, page_size=args.page_size,
                                   limit=args.limit, include_reviewed=args.all,
                                   progress=print)
        except Exception as exc:
            print(f"生成复核页失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(f"\n复核页：{r['index']}（{r['pages']} 页，待复核 {r['queued']} 帧）")
        for k, v in sorted(r["by_reason"].items()):
            print(f"  {k:<9} {v:>6}")
        print("\n在浏览器里改完 -> 点「导出修正 JSON」-> 写回：")
        print(f"  python -m a9route train apply 下载的_corrections.json --name {name}")
        if args.sheets:
            for p in review.contact_sheets(name=name, progress=print):
                print(f"  拼图: {p}")
        return 0

    if sub == "apply":
        from a9route.train import review
        if not args.file:
            print("要给复核页导出的 JSON：a9route train apply corrections.json",
                  file=sys.stderr)
            return 2
        try:
            r = review.apply_corrections(args.file, name=name, progress=print)
        except Exception as exc:
            print(f"写回失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(f"写出 {r['reviewed']} 帧（其中 {r['changed']} 帧真的改了，"
              f"{r['skipped']} 帧对不上被跳过）")
        print("改完标注记得重新划分再训：")
        print(f"  python -m a9route train split --name {name}")
        return 0

    if sub == "split":
        from a9route.train import dataset
        try:
            rep = dataset.split(name=name, val_ratio=args.val_ratio, seed=args.seed,
                                progress=print)
        except Exception as exc:
            print(f"划分失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print("\n" + rep.describe())
        print(f"data.yaml -> {dataset.dataset_dir(name) / 'data.yaml'}")
        return 0

    if sub == "choice-build":
        # **把现有数据分类**：复用按键数据集的帧，贴一套选路标注（三个答案的初始值
        # 来自启发式圆检测）。图片走**硬链接**，几乎不占额外空间。
        from a9route.train import choice as C
        try:
            r = C.build_from(name or C.DEFAULT_NAME, source=args.source,
                             progress=print)
        except Exception as exc:
            print(f"分类失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print("\n选路数据集：{0}".format(r["dataset"]))
        print("  帧 {0}（其中有岔路口 {1}）".format(r["frames"], r["with_choice"]))
        print("  图片：硬链接 {0} / 复制 {1}（硬链接不占额外空间）".format(
            r["linked"], r["copied"]))
        print("  「几个选项」分布：" + "，".join(
            "{0} 个×{1}".format(k, v) for k, v in sorted(r["by_count"].items())))
        print("  「选第几个」分布：" + "，".join(
            "{0}×{1}".format("没认出" if k == 0 else "第{0}个".format(k), v)
            for k, v in sorted(r["selected"].items())))
        print("\n下一步（**三个选择题，单独一页**）：")
        print("  a9route serve  然后打开 http://127.0.0.1:8790/choice")
        print("  （或命令行看统计：a9route train choice-stats）")
        return 0

    if sub == "choice-stats":
        from a9route.train import choice as C
        try:
            st = C.stats(name or C.DEFAULT_NAME)
        except Exception as exc:
            print(f"读不到：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print("选路数据集 {0}".format(name or C.DEFAULT_NAME))
        print("  帧 {0}，已复核 {1}（待复核 {2}）".format(
            st["frames"], st["reviewed"], st["frames"] - st["reviewed"]))
        print("  有岔路口 {0}；其中「没认出选中的是哪个」{1} 帧（最该人看）".format(
            st["with_choice"], st["no_selected"]))
        print("  位置是猜的（检测不到位置、外推补的）{0} 帧".format(st["guessed"]))
        print("  「几个选项」：" + "，".join(
            "{0}×{1}".format(k, v) for k, v in sorted(st["by_count"].items())))
        print("  「选第几个」：" + "，".join(
            "{0}×{1}".format("没认出" if k == 0 else k, v)
            for k, v in sorted(st["selected"].items())))
        print("\n标注界面：a9route serve -> http://127.0.0.1:8790/choice")
        return 0

    if sub == "choice-review":
        from a9route.train import choice as C
        st = C.stats(name or C.DEFAULT_NAME)
        print("选路标注页（**和刹车/氮气那页分开**）："
              "http://127.0.0.1:8790/choice")
        print("  数据集 {0}：{1} 帧，已复核 {2}".format(
            name or C.DEFAULT_NAME, st["frames"], st["reviewed"]))
        print("\n先起服务：a9route serve")
        return 0

    if sub == "run":
        from a9route.train import dataset, runner
        ds = dataset.dataset_dir(name)
        yml = Path(args.data) if args.data else (ds / "data.yaml")
        if not yml.is_file():
            print(f"找不到 {yml} —— 先 `a9route train split --name {name}`",
                  file=sys.stderr)
            return 2
        try:
            rep = runner.train(
                data=yml, weight=args.weight or runner.DEFAULT_WEIGHT,
                epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, run_name=args.run_name, workers=args.workers,
                patience=args.patience, lr0=args.lr0, seed=args.seed,
                progress=print)
        except Exception as exc:
            print(f"训练失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print("\n" + rep.describe())
        print("\n下一步：a9route train eval")
        return 0

    if sub == "eval":
        from a9route.train import dataset, runner
        ds = dataset.dataset_dir(name)
        weights = args.weights or str(paths.MODELS_DIR / "runs" / args.run_name
                                      / "weights" / "best.pt")
        if not Path(weights).is_file():
            print(f"找不到权重 {weights}；用 --weights 指定，或先 `train run`",
                  file=sys.stderr)
            return 2
        # 置信度默认**跟着运行时那份配置走**（`vision.key_conf` / `vision.choice_conf`）——
        # 不然评估用的是 argparse 里写死的 0.40，而运行时用的是配置里的值，
        # 评出来的数字**不是运行时真会得到的数字** ✗（实测：配置 0.30 时
        # 评估仍按 0.40 报 0.9875，而运行时其实是 0.990 —— 差的那一档白丢了）
        task = _dataset_task(name)
        conf = args.conf
        if conf is None:
            conf = float(((cfgmod.current().get("vision") or {})
                          .get("choice_conf" if task == "choice" else "key_conf",
                               0.40)))
        try:
            if task == "choice":
                rep = runner.evaluate_choice(
                    weights, dataset=ds, split=args.split, imgsz=args.imgsz,
                    conf=conf, limit=args.limit,
                    compare_heuristic=not args.no_baseline)
                print("\n" + runner.format_choice_eval(rep))
            else:
                rep = runner.evaluate_frames(
                    weights, dataset=ds, split=args.split, imgsz=args.imgsz,
                    conf=conf, device=args.device, limit=args.limit,
                    compare_heuristic=not args.no_baseline)
                print("\n" + runner.format_frame_eval(rep))
        except Exception as exc:
            print(f"评估失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if args.json:
            p = runner.save_report(args.json, rep)
            print(f"\n明细已写到 {p}")
        return 0

    if sub == "pulses":
        from a9route.train import runner
        from a9route.vision import keys as keymod
        # 录像可以写成位置参数（`train pulses 录像.mp4`）或 `--video` —— 两种都收，
        # 位置参数更顺手（和 `train build` 一致）
        video = args.video or (args.videos[0] if args.videos else "")
        if not video:
            print("要给一段录像：a9route train pulses 录像.mp4 [--weights X]",
                  file=sys.stderr)
            return 2
        try:
            if args.weights:
                model = keymod.build_key_detector(
                    backend=("onnx" if str(args.weights).endswith(".onnx")
                             else "ultralytics"),
                    model=args.weights, use_cache=False)
                got = runner.pulse_report(model, video, gap=args.gap,
                                          max_seconds=args.max_seconds, progress=print)
                base = None
                if not args.no_baseline:
                    from a9route.vision.keys import HeuristicKeys
                    base = runner.pulse_report(HeuristicKeys(), video,
                                               gap=args.gap,
                                               max_seconds=args.max_seconds)
            else:
                got = runner.pulse_report(keymod.build_key_detector(), video,
                                          gap=args.gap,
                                          max_seconds=args.max_seconds, progress=print)
                base = None
        except Exception as exc:
            print(f"跑失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print("\n" + runner.format_pulse_report(got, base))
        return 0

    if sub == "export":
        from a9route.train import runner
        weights = args.weights or str(paths.MODELS_DIR / "runs" / args.run_name
                                      / "weights" / "best.pt")
        if not Path(weights).is_file():
            print(f"找不到权重 {weights}；用 --weights 指定，或先 `train run`",
                  file=sys.stderr)
            return 2
        try:
            if _dataset_task(name) == "choice":
                # 选路模型：**另一套类别**，导出到 models/choice.onnx
                from a9route.train import choice as C
                out = runner.export_onnx(weights, imgsz=args.imgsz, out=args.out,
                                         opset=args.opset, progress=print,
                                         name="choice", classes=C.CHOICE_CLASSES)
            else:
                out = runner.export_onnx(weights, imgsz=args.imgsz, out=args.out,
                                         opset=args.opset, progress=print,
                                         name="keys")
        except Exception as exc:
            print(f"导出失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print("\n切到模型后端（一条命令，会核对类别顺序）：")
        print(f"  python -m a9route train use \"{out}\"")
        print(f"  python -m a9route train models          # 看有哪些模型")
        return 0

    if sub == "models":
        # 换模型之前先看有什么（含清单里的类序/imgsz/指标）
        from a9route.train import modelcard as MC
        rows = MC.list_models(paths.MODELS_DIR)
        print(f"模型目录：{paths.MODELS_DIR}")
        print("\n当前生效：")
        print("  " + _backend_line())
        if not rows:
            print("\n（还没有模型 —— 训练并导出：train run / train export）")
            return 0
        try:
            be_now, mp_now, _ = _resolve_backend_safe()
        except Exception:
            be_now, mp_now = "", ""
        print("\n可用模型：")
        for r in rows:
            mark = ""
            if be_now == "onnx" and mp_now:
                try:
                    if Path(mp_now).resolve() == Path(r["path"]).resolve():
                        mark = "   <- **正在用**"
                except OSError:
                    pass
            print("  {0}  ({1} MB, {2}){3}".format(r["name"], r["size_mb"],
                                                  r["mtime"], mark))
            print("     {0}".format(r["description"]))
        print("\n切换：a9route train use <名字或路径>")
        print("      a9route config set vision__key_backend=heuristic   # 退回启发式")
        return 0

    if sub == "use":
        # **换模型**：一条命令把"后端 + 模型文件"写进 config.json，并**核对类序**。
        # 关键：**先判这个模型属于哪个任务**（按键 / 选路），再写对应的那组配置键 ——
        # 把选路模型塞进按键的槽会两边都错，而且错得很隐蔽。
        from a9route.train import modelcard as MC
        want = args.file or (args.videos[0] if args.videos else "")
        if not want:
            print("要给模型名字或路径：a9route train use keys.onnx", file=sys.stderr)
            return 2
        cand = Path(want)
        if not cand.is_absolute():
            hits = [m for m in MC.list_models(paths.MODELS_DIR)
                    if m["name"] == want or Path(m["name"]).stem == want
                    or m["name"] == want + ".onnx" or m["name"] == want + ".pt"]
            if len(hits) == 1:
                cand = Path(hits[0]["path"])
            elif len(hits) > 1:
                print("models/ 里有多个匹配的模型，给全名："
                      + "，".join(m["name"] for m in hits), file=sys.stderr)
                return 2
        if not cand.is_file():
            print(f"找不到模型：{cand}\n  先看看有什么：a9route train models",
                  file=sys.stderr)
            return 2
        card = MC.load(cand)
        if not card.loaded:
            print(f"这个模型没有清单（{MC.card_path(cand).name} 不存在）——\n"
                  "  没有清单就**无法确认它属于按键还是选路**，切错了会让两套识别"
                  "整体错位而且不报错。\n"
                  "  请用 `a9route train export` 重新导出（它会写清单）。",
                  file=sys.stderr)
            return 1
        task = MC.identify(card)
        if not task:
            print("这个模型的类别表谁也不匹配（既不是按键也不是选路）：\n  "
                  + " / ".join(card.classes), file=sys.stderr)
            print("  可用类别表：" + "；".join(
                "{0}={1}".format(k, "/".join(v))
                for k, v in MC.known_class_sets().items()), file=sys.stderr)
            return 1
        backend = "onnx" if cand.suffix.lower() == ".onnx" else "ultralytics"
        cfgf = cfgmod.load_config()
        cfgf.setdefault("vision", {})["{0}_backend".format(task)] = backend
        cfgf["vision"]["{0}_model".format(task)] = str(cand)
        if card.imgsz:
            # 让 imgsz 跟着模型走 —— 少了这一步就会出现"导出 640、推理别的值"的静默错位
            cfgf["vision"]["{0}_imgsz".format(task)] = int(card.imgsz)
        f = cfgmod.save_config(cfgf)
        zh = {"keys": "按键（刹车/氮气）", "choice": "选路（岔路口）"}.get(task, task)
        print(f"已写入 {f}")
        print(f"  任务 = {zh}")
        print(f"  后端 = {backend}（配置键 vision.{task}_backend）")
        print(f"  模型 = {cand}")
        print(f"  清单 = {card.describe()}")
        print("\n下次 analyze / serve 就用它。退回启发式：")
        print(f"  a9route config set vision__{task}_backend=heuristic")
        return 0

    print(__doc__)
    return 2


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
    mods = {"video": "a9route.tests.test_video",
            "intent": "a9route.tests.test_intent",
            "route": "a9route.tests.test_route",
            "web": "a9route.tests.test_webvideo",
            "train": "a9route.tests.test_train"}
    if name == "all":
        names = ["route", "intent", "video", "web", "train"]
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

    a = sub.add_parser("train", help="【YOLOv8】按键识别的训练框架（build/review/run/eval…）")
    a.add_argument("action", nargs="?", default="doctor",
                   choices=["doctor", "fetch", "build", "review", "apply", "split",
                            "verify", "audit", "stats", "run", "eval", "pulses",
                            "export", "models", "use",
                            "choice-build", "choice-stats", "choice-review"],
                   help="doctor=环境自检；build=抽帧+预标注；review=复核页；"
                        "apply=写回修正；split=划分；verify=数据集自检；"
                        "audit=**和标定对照的体检**；run=训练；eval=逐帧评估；"
                        "pulses=整片脉冲对比；export=导出 ONNX；"
                        "models=列出模型；use=**切换模型**；"
                        "choice-*=**选路**（分类现有数据/统计/标注页）")
    a.add_argument("--source", default="keys",
                   help="choice-build 用：从哪个数据集复用帧（默认 keys）")
    a.add_argument("videos", nargs="*", help="build 用：录像路径（可多段）")
    a.add_argument("--name", default="",
                   help="数据集名（默认：选路那几条子命令用 choice，其余用 keys）")
    a.add_argument("--set", action="append", default=[],
                   help="临时覆盖配置，可重复（会 config.apply）")
    # build
    a.add_argument("--budget", type=int, default=3000, help="抽帧总预算（默认 3000）")
    a.add_argument("--per-video", type=int, default=None, dest="per_video",
                   help="每段录像的预算（默认按段数均分 --budget）")
    a.add_argument("--max-seconds", type=float, default=None, dest="max_seconds")
    a.add_argument("--every", type=int, default=1,
                   help="每隔几帧算一次信号（**>1 会漏掉单帧点击**，默认 1）")
    a.add_argument("--edge-pad", type=int, default=3, dest="edge_pad",
                   help="翻转帧前后各多抽几帧（默认 3）")
    a.add_argument("--margin-band", type=float, default=0.25, dest="margin_band",
                   help="离阈值多近算「可疑」（默认 0.25）")
    a.add_argument("--edge-share", type=float, default=0.6, dest="edge_share",
                   help="翻转帧最多占预算的比例（默认 0.6）；"
                        "给多了会抽不到背景帧")
    a.add_argument("--jpeg-quality", type=int, default=92, dest="jpeg_quality")
    a.add_argument("--seed", type=int, default=0)
    # review
    a.add_argument("--page-size", type=int, default=120, dest="page_size")
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--no-recheck", action="store_true", dest="no_recheck",
                   help="audit 用：跳过「用当前判据重算」那一步"
                        "（快，但查不出阈值漂移）")
    a.add_argument("--all", action="store_true", help="review 时连已复核的也列出来")
    a.add_argument("--sheets", action="store_true", help="review 时额外生成拼图")
    # apply
    a.add_argument("--file", default="", help="apply 用：修正 JSON 路径")
    # split
    a.add_argument("--val-ratio", type=float, default=0.2, dest="val_ratio")
    # run
    a.add_argument("--weight", default="", help="起点权重（默认 yolov8n.pt；"
                                                "给 .yaml 就是从零训练）")
    a.add_argument("--epochs", type=int, default=100)
    a.add_argument("--imgsz", type=int, default=640)
    a.add_argument("--batch", type=int, default=16)
    a.add_argument("--workers", type=int, default=4,
                   help="DataLoader 进程数。**受限沙箱/Windows 上报 "
                        "「DataLoader worker exited unexpectedly」就给 0** —— "
                        "worker 之间走命名管道，沙箱会拦")
    a.add_argument("--patience", type=int, default=30)
    a.add_argument("--lr0", type=float, default=None)
    a.add_argument("--device", default=None, help="0 / cpu（不给就自动）")
    # ⚠️ 默认值**不能写死 "keys"**：它是训练产物目录名，`run`/`eval`/`export` 三处都用它。
    # 写死 "keys" 的后果实测过（2026-09-15）：`train run --name choice` 把**选路的训练
    # 日志/权重**写进 `models/runs/keys/`，而 `train export --name choice` 又去
    # `models/runs/keys/weights/best.pt` 拿权重 —— 等于**把按键的模型导出成选路模型** ✗✗
    # （和 §10.25 那个 `/dataset` 事故是同一类：多任务共用一处写死的默认值。）
    # 现在默认跟数据集同名；显式给了 `--run-name` 就听显式的。
    a.add_argument("--run-name", default=None, dest="run_name",
                   help="训练产物目录名（默认**跟数据集同名**：keys/choice）")
    a.add_argument("--data", default="", help="run 用：data.yaml（默认数据集里那份）")
    # eval / export
    a.add_argument("--weights", default="", help="eval/export 用：权重路径")
    a.add_argument("--out", default="", help="export 用：输出 ONNX 路径")
    a.add_argument("--opset", type=int, default=12)
    a.add_argument("--split", default="val", help="eval 用：val / train")
    a.add_argument("--conf", type=float, default=None,
                   help="eval 用：置信度阈值（默认**跟着运行时配置**"
                        " vision.key_conf / choice_conf）")
    a.add_argument("--no-baseline", action="store_true", dest="no_baseline",
                   help="eval/pulses 时不跑启发式对照组")
    a.add_argument("--json", default="", help="eval 用：把明细写到文件")
    # pulses
    a.add_argument("--video", default="", help="pulses 用：录像路径")
    a.add_argument("--gap", type=float, default=0.12, help="pulses 用：分段间隔秒")
    a.set_defaults(func=cmd_train)

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
