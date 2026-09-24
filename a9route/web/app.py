# -*- coding: utf-8 -*-
"""本地 Web 窗口（Flask）—— **两个页面，各干一件事**。

* `/` —— 拖视频进去，出来一条路线；
* `/dataset` —— **数据集体检 / 逐帧对照**（标注 ↔ 标定 ↔ 判据 对不对得上）。

只在 `127.0.0.1` 上监听：这个窗口会把你上传的录像写到磁盘、跑几十秒的分析，
不属于"对外服务"。端口默认 **8790**（换端口用 `--port`）。

## JSON API（前端只依赖这些）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET  | `/` | 跑图页面 |
| GET  | `/dataset` | **数据集体检页面** |
| GET  | `/api/version` | 版本、项目根、配置文件、本次生效的配置 |
| GET  | `/api/videos` | 可分析的视频（`output/` + `A9ROUTE_VIDEO_DIR` 里的目录） |
| POST | `/api/analyze` | 启动分析：上传文件（multipart 字段 `video`）**或** JSON `{path}`；可选 `every` / `max_seconds` / `set` |
| GET  | `/api/jobs` | 最近的任务列表（刷新页面不丢，仍在内存里的） |
| GET  | `/api/status?job=` | 进度 / 路线 / 逐百分点明细（前端轮询） |
| GET  | `/media/<job>/<name>` | 某百分点截图（PNG） |
| GET  | `/api/route?job=` | 把该任务的完整路线文本下载下来 |

数据集体检那几个接口在 `web/dataset.py`（Flask Blueprint），
业务逻辑在 `train/audit.py`（CLI 的 `a9lab audit` 用同一份）。

分析要跑几十秒，所以放**后台线程**、前端轮询 —— 进度靠"覆盖 N 个百分点"这句。
"""
from __future__ import annotations

import argparse
import threading
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from a9route import __version__, config as cfgmod, paths
from a9route.bootstrap import configure_console

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8790


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(paths.WEB_TEMPLATES_DIR),
        static_folder=str(paths.WEB_STATIC_DIR),
        static_url_path="/static",
    )
    app.config["JSON_AS_ASCII"] = False
    app.json.ensure_ascii = False
    paths.ensure_dirs()

    # ⚠️ 数据集/标注那两个页面（`/dataset`、`/choice`）**已经搬去 a9lab** 了 ——
    # 它们属于"造数据集/标数据"那条线，不属于运行时。这个窗口只留跑图页 + 参数面板。

    jobs: dict[str, dict] = {}
    lock = threading.Lock()

    # ------------------------------------------------------------ 页面
    @app.get("/")
    def index():
        return send_from_directory(str(paths.WEB_TEMPLATES_DIR), "index.html")

    @app.get("/favicon.ico")
    def favicon():
        return ("", 204)

    @app.get("/api/version")
    def api_version():
        cfg = cfgmod.load_config()
        return jsonify(version=__version__, root=str(paths.ROOT),
                       config_file=str(paths.CONFIG_FILE),
                       config_exists=paths.CONFIG_FILE.is_file(),
                       scan=cfg.get("scan", {}), vision=cfg.get("vision", {}))

    # ------------------------------------------------------------ 视频列表
    @app.get("/api/videos")
    def api_videos():
        out: list[dict] = []
        for d in paths.video_dirs():
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() not in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
                    continue
                out.append({"path": str(p), "name": p.name,
                            "size_mb": round(p.stat().st_size / 1024 / 1024, 1)})
        with lock:
            for j in jobs.values():
                if j.get("uploaded"):
                    out.append({"path": j["uploaded"], "name": f"{j['name']}（上传）",
                                "size_mb": 0})
        return jsonify(videos=out,
                       dirs=[str(d) for d in paths.video_dirs()])

    # ------------------------------------------------------------ 分析任务
    def _new_job(name: str, video: Path) -> str:
        jid = uuid.uuid4().hex[:12]
        d = paths.ANALYSIS_DIR / jid
        (d / "shots").mkdir(parents=True, exist_ok=True)
        with lock:
            jobs[jid] = {"id": jid, "name": name, "state": "running",
                         "message": "准备中…", "route": "", "flat": "",
                         "shots": [], "error": "", "dir": str(d),
                         "queue": ["准备中…"], "video": str(video)}
        return jid

    def _push(job: dict, msg: str) -> None:
        """只留最近 N 条：日志条数会很多（每个百分点一条）。"""
        q = job.setdefault("queue", [])
        q.append(str(msg).strip())
        if len(q) > 40:
            del q[:-40]
        job["message"] = str(msg).strip()
        app.logger.info("[analyze] %s", msg)

    def _run(jid: str, kwargs: dict) -> None:
        from a9route import analysis

        job = jobs.get(jid)
        if job is None:
            return
        try:
            rep = analysis.analyze(progress=lambda m: _push(job, m), **kwargs)
            job["route"] = rep.route_text
            job["flat"] = rep.flat
            job["shot_rows"] = rep.percent_rows()
            job["button_events"] = rep.button_events
            job["intents"] = rep.intents
            job["seconds"] = round(rep.seconds, 2)
            # **这次分析无效的原因**（例如 OCR 读不到模型）—— 前端要当**错误**显示，
            # 否则用户看到的只是一条空路线，会跑去查检测/阈值
            job["blocker"] = rep.blocker
            job["warnings"] = list(rep.warnings)
            job["state"] = "done"
            job["message"] = (
                f"完成：{len(rep.shots)} 张截图，"
                f"{len([r for r in job['shot_rows'] if r['op']])} 条操作，"
                f"耗时 {rep.seconds:.1f}s")
        except Exception as exc:
            job["state"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["message"] = "分析失败"
            app.logger.exception("analyse job failed")

    @app.post("/api/analyze")
    def api_analyze():
        """上传文件（multipart 字段 `video`）**或**给服务器上的路径（JSON `path`）。

        可选：`every`（抽帧间隔秒）、`max_seconds`（只分析前 N 秒，试跑用）、
        `set`（临时改配置，形如 `{"vision__nitro_red_thr": 0.22}`，只影响这一次）。
        """
        # JSON 解不出来要把话说清楚（否则前端只看到一个**空 400**，很难查）——
        # 常见原因：Content-Type 没写 application/json，或 body 不是 UTF-8
        #（实测：PowerShell 的 Invoke-WebRequest 发中文 JSON 时会踩这个坑）。
        data = request.get_json(silent=True)
        if data is None and request.files.get("video") is None:
            if (request.data or b"").strip():
                return jsonify(error="请求体不是合法 JSON（要 UTF-8 + "
                                     "Content-Type: application/json）"), 400
            data = {}
        data = data or {}
        every = data.get("every") or request.form.get("every")
        every = float(every) if every else None
        max_seconds = data.get("max_seconds") or request.form.get("max_seconds")
        max_seconds = float(max_seconds) if max_seconds else None
        sets = data.get("set") or {}
        if isinstance(sets, str):                      # 也接受 "a__b=1,c__d=2"
            parsed: dict = {}
            flat_defaults = cfgmod.flat(cfgmod.load_config(), env=False)
            for item in sets.split(","):
                if "=" in item:
                    k, v = item.split("=", 1)
                    k = k.strip().replace(".", "__")
                    parsed[k] = cfgmod.coerce(v.strip(), flat_defaults.get(k))
            sets = parsed

        f = request.files.get("video")
        # ⚠️ **同一个进程里只允许跑一个分析**（2026-09-24 加的）。
        #    为什么：每个分析自己一份 PaddleOCR + 两份 ONNX，还各带一个解码线程和一个
        #    按键细扫线程；同时跑两个会互相抢（OCR 返回空结果、显存/线程争用），
        #    而表现是**"这次分析没扫到任何百分比"** —— 看起来像视频有问题 ✗。
        #    这种"资源争用导致结果莫名其妙"最难查，所以宁可**明说**：
        #    "已经有一个在跑，等它结束"（界面上一句话，比一堆玄学结果强）。
        with lock:
            running = [j for j in jobs.values() if j.get("state") == "running"]
        if running:
            return jsonify(
                error="已经有一个分析在跑了（{0}）—— 同一个进程里同时跑两个会互相抢 "
                      "OCR/模型，结果会莫名其妙。等它结束再拖，或另开一个窗口。"
                      .format(running[0].get("name") or "未命名")), 409
        if f is not None:
            name = f.filename or "upload.mp4"
            jid = _new_job(name, Path(name))
            target = paths.ANALYSIS_DIR / jid / f"input{Path(name).suffix or '.mp4'}"
            f.save(str(target))
            jobs[jid]["uploaded"] = str(target)
            jobs[jid]["video"] = str(target)
        else:
            raw = str(data.get("path") or request.args.get("path") or "").strip()
            if not raw:
                return jsonify(error="没给视频（拖文件进来，或从列表里选一个）"), 400
            target = Path(raw)
            if not target.is_absolute():
                target = paths.ROOT / raw
            if not target.is_file():
                return jsonify(error=f"找不到视频：{target}"), 400
            jid = _new_job(target.name, target)

        kwargs = {"video": target, "out_dir": Path(jobs[jid]["dir"]),
                  "every": every, "max_seconds": max_seconds,
                  "overrides": sets or None}
        threading.Thread(target=_run, args=(jid, kwargs),
                         name=f"analyze-{jid}", daemon=True).start()
        return jsonify(job=jid, name=jobs[jid]["name"], every=every,
                       max_seconds=max_seconds, set=sets or {})

    @app.get("/api/config")
    def api_config_get():
        """**可编辑的判据参数表**（给前端画滑块/输入框用）。

        ⚠️ 返回的不只是值：每项还带 `min/max/step`、中文名、内置默认值、
        以及"**这一项在当前后端下是否真的生效**"（`applies` / `note`）——
        最坑的一类误导就是"调了没反应"（比如当前用模型，
        而调的是只对 HoughCircles 生效的参数）。
        """
        from a9route.web import config_ui as CU
        from a9route.vision import choice as VC
        from a9route.vision import keys as VK

        backends = {}
        for task, mod in (("keys", VK), ("choice", VC)):
            try:
                backends[task] = mod.resolve_backend()[0]
            except Exception as exc:                   # noqa: BLE001
                backends[task] = "起不来（{0}）".format(type(exc).__name__)
        data = CU.payload(backends=backends)
        data["backends"] = backends
        return jsonify(data)

    @app.post("/api/config")
    def api_config_post():
        """改参数并**落盘到 config.json**（下一次分析立刻生效）。

        请求体：`{"set": {"scan.choice_idle_hold": 12, ...}}`
        或 `{"reset": ["scan.choice_idle_hold"]}` / `{"reset": "all"}`。

        ⚠️ 校验在**后端**（前端能绕过）；有一项不合格就**一个都不写** ——
        不留半套配置（那种状态最难查）。
        """
        from a9route.web import config_ui as CU

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="要 POST 一个 JSON：{set:{...}} 或 {reset:[...]}"), 400
        try:
            if "reset" in data:
                r = data.get("reset")
                keys = None if r in ("all", "*", True) else list(r or [])
                res = CU.reset(keys)
            else:
                res = CU.save(data.get("set") or {})
        except Exception as exc:                       # noqa: BLE001
            return jsonify(error="{0}: {1}".format(type(exc).__name__, exc)), 400
        if not res.get("ok"):
            return jsonify(error="；".join(res.get("errors") or ["改参数失败"]),
                           errors=res.get("errors")), 400
        return jsonify(result=res)

    @app.get("/api/jobs")
    def api_jobs():
        with lock:
            return jsonify(jobs=[{"id": j["id"], "name": j["name"],
                                  "state": j["state"], "message": j["message"]}
                                 for j in jobs.values()])

    @app.get("/api/status")
    def api_status():
        jid = request.args.get("job", "")
        job = jobs.get(jid)
        if not job:
            return jsonify(error="没有这个分析任务"), 404
        out = {k: job.get(k) for k in ("id", "name", "state", "message", "error",
                                       "route", "flat", "video", "blocker",
                                       "warnings")}
        out["shots"] = job.get("shot_rows") or []
        out["button_events"] = job.get("button_events", 0)
        out["intents"] = job.get("intents", 0)
        out["seconds"] = job.get("seconds", 0)
        out["log"] = list(job.get("queue") or [])
        return jsonify(out)

    # ------------------------------------------------------------ 产物
    @app.get("/media/<jid>/<path:name>")
    def api_media(jid: str, name: str):
        """某百分点截图。

        目录穿越由 `send_from_directory` 自己挡（它会做 safe_join），
        **不要**自己拼路径 —— 原项目里那几个接口就是这么写的。
        """
        job = jobs.get(jid)
        if not job:
            return ("", 404)
        return send_from_directory(str(Path(job["dir"]) / "shots"), name)

    @app.get("/api/route")
    def api_route():
        jid = request.args.get("job", "")
        job = jobs.get(jid)
        if not job or not job.get("route"):
            return ("", 404)
        return (job["route"], 200, {"Content-Type": "text/plain; charset=utf-8",
                                    "Content-Disposition":
                                        f'attachment; filename="{jid}_route.txt"'})

    return app


def main(argv: list[str] | None = None) -> int:
    configure_console()
    p = argparse.ArgumentParser(prog="a9route serve",
                               description="起本地 Web 窗口（拖视频进去 -> 出路线）")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--no-browser", action="store_true", dest="no_browser")
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    app = create_app()
    url = f"http://{args.host}:{args.port}/"
    print(f"a9route Web 窗口: {url}")
    print(f"  项目根: {paths.ROOT}")
    print(f"  配置:   {paths.CONFIG_FILE}"
          f"{'' if paths.CONFIG_FILE.is_file() else '（不存在 —— 用内置默认值）'}")
    print(f"  视频目录: {', '.join(str(d) for d in paths.video_dirs())}")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
