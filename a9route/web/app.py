# -*- coding: utf-8 -*-
"""本地 Web 窗口（Flask）—— **一页就干一件事：拖视频进去，出来一条路线。**

只在 `127.0.0.1` 上监听：这个窗口会把你上传的录像写到磁盘、跑几十秒的分析，
不属于"对外服务"。端口默认 **8790**（换端口用 `--port`）。

## JSON API（前端只依赖这些）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET  | `/` | 页面 |
| GET  | `/api/version` | 版本、项目根、配置文件、本次生效的配置 |
| GET  | `/api/videos` | 可分析的视频（`output/` + `A9ROUTE_VIDEO_DIR` 里的目录） |
| POST | `/api/analyze` | 启动分析：上传文件（multipart 字段 `video`）**或** JSON `{path}`；可选 `every` / `max_seconds` / `set` |
| GET  | `/api/jobs` | 最近的任务列表（刷新页面不丢，仍在内存里的） |
| GET  | `/api/status?job=` | 进度 / 路线 / 逐百分点明细（前端轮询） |
| GET  | `/media/<job>/<name>` | 某百分点截图（PNG） |
| GET  | `/api/route?job=` | 把该任务的完整路线文本下载下来 |

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
        data = request.get_json(silent=True) or {}
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
                                       "route", "flat", "video")}
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
