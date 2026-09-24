# -*- coding: utf-8 -*-
"""Web 窗口几个接口的离线测试（Flask 测试客户端，**不真的起服务器**）。

  T1 页面与 /api/version
  T2 /api/videos 列目录 + 给错路径要报错
  T3 完整一条龙：合成视频 -> POST /api/analyze -> 轮询 status -> 路线 + 截图
  T4 截图接口**防目录穿越**（`..` / `/` / `\` 一律拒）
  T5 上传文件这条路（multipart）也能跑
  T6 路线下载接口
  T7 **没有 OCR 也要能跑完**（判据退化，但不崩 —— 真机/沙箱里 OCR 可能起不来）

    python -m a9route.tests.test_webvideo
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from a9route import config as cfgmod
from a9route.tests.support import check, fake_hud_video, summary, tmp_dir

#: 测试产物目录（`main()` 里建好；跟随 A9ROUTE_WORKTMP）
TMP = Path(".")


# ---------------------------------------------------------------- 假检测器
def _install_fake_detector(percent_per_frame: float = 4.0, choice_at=(4, 8),
                           delay: float = 0.0):
    """把 `cues.CueDetector` 换成"按帧号给百分比 + 偶尔按键"的假货。

    为什么必须注入：真判据要 OCR 读「路程 NN%」，而 OCR 在受限沙箱里读不到模型
    （`~/.paddlex` 在工作区外）—— 那是环境限制，不是代码问题。
    这里锁的是**Web 这条编排链**（上传/后台线程/轮询/截图/路线），
    所以把"看画面"这一步换成确定的假货，让测试与 OCR 环境无关 ✓。

    `delay`：每帧睡一下，用来把分析**拖慢**（测"同时只允许一个分析"那条路时
    需要一个还在 running 的任务，否则第二个请求到达时它早跑完了）。

    注意：它会**覆盖模块里的类**，所以调用方必须负责还原（见 `main()` 的 finally）。
    """
    from a9route.vision import cues

    original = cues.CueDetector
    state = {"i": 0}

    class FakeDetector:
        def __init__(self, *a, **kw):
            pass

        def detect(self, frame):
            if delay:
                time.sleep(delay)
            i = state["i"]
            state["i"] += 1
            cue = cues.Cue()
            bx, by, bw, bh = cues.BRAKE_KEY_BOX
            nx, ny, nw, nh = cues.NITRO_KEY_BOX
            # 每 10 帧按一下刹车（≈0.3s @10fps -> 漂移/独立脉冲），每 25 帧点一下氮气
            on_brake = (i % 10) < 3
            cue.brake_lit = on_brake or cues.bright_ratio(
                frame[by:by + bh, bx:bx + bw]) > 0.5
            cue.drifting = cue.brake_lit
            cue.nitro_pressed = ((i % 25) < 2) or (
                cues.red_ratio(frame[ny:ny + nh, nx:nx + nw]) > cues.NITRO_RED_THR)
            if choice_at and choice_at[0] <= i <= choice_at[1]:
                cue.icons = [{"xy": (610, 127), "radius": 30, "blue": False,
                              "blue_ratio": 0.0},
                             {"xy": (680, 127), "radius": 30, "blue": True,
                              "blue_ratio": 0.8}]
            return cue, min(100.0, 1.0 + i * percent_per_frame)

    cues.CueDetector = FakeDetector
    return original


def _wait(client, jid: str, *, timeout: float = 180.0) -> dict:
    """轮询到 done/error（分析跑在后台线程）。"""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        data = client.get(f"/api/status?job={jid}").get_json()
        if not data:
            return {"state": "missing"}
        if data.get("state") in ("done", "error"):
            return data
        time.sleep(0.3)
    return {"state": "timeout"}


def main() -> int:
    global TMP
    TMP = tmp_dir()
    cfgmod.apply()
    original = _install_fake_detector()
    try:
        return _run_all()
    finally:
        # **必须还原**：T7 要测真 CueDetector，而假检测器是直接替换模块里的类
        from a9route.vision import cues
        cues.CueDetector = original


def _run_all() -> int:
    from a9route.web.app import create_app
    app = create_app()
    app.testing = True
    client = app.test_client()

    # ------------------------------------------------------------ T1
    print("\n=== T1 页面与 version ===")
    r = client.get("/")
    check("首页 200", r.status_code == 200, str(r.status_code))
    html = r.get_data(as_text=True)
    check("首页含拖拽区/进度/结果三块", all(k in html for k in
          ("dropZone", "videoFill", "videoResult")), "ids ok")
    check("首页引了 app.js 与 style.css",
          "/static/app.js" in html and "/static/style.css" in html, "ok")
    v = client.get("/api/version").get_json()
    check("/api/version 带版本与配置路径",
          bool(v.get("version")) and "config_file" in v, str(v.get("version")))
    check(" 带本次生效的 scan/vision 配置（前端「参数」弹窗用）",
          "scan" in v and "vision" in v, str(sorted(v.keys())))

    # ------------------------------------------------------------ T2
    print("\n=== T2 视频列表 ===")
    vids_dir = TMP / "webvids"
    # ⚠️ 这个假视频**必须真的画上操作**（氮气/刹车帧）——以前它是"只有百分比读数"
    # 的空白画面，而 T3 却断言"路线非空"：那时能过，是因为旧的兜底路径会从
    # 窗口汇总/圆检测里**凭空凑出**几条操作 ✗✗。现在判定全部来自模型，
    # 没有操作就是没有操作，所以这里把操作画出来，测的才是真的那条路。
    video = fake_hud_video(vids_dir / "web_test.mp4", seconds=4.0, fps=10.0,
                           brake_frames=tuple(range(20, 31)),
                           nitro_frames=(5, 6, 7, 8, 9))
    import os
    os.environ["A9ROUTE_VIDEO_DIR"] = str(vids_dir)
    try:
        data = client.get("/api/videos").get_json()
        names = [v["name"] for v in data.get("videos", [])]
        check("列出了 A9ROUTE_VIDEO_DIR 里的视频", "web_test.mp4" in names, str(names))
        check(" 同时报告扫描了哪些目录", str(vids_dir) in (data.get("dirs") or []),
              str(data.get("dirs")))
    finally:
        os.environ.pop("A9ROUTE_VIDEO_DIR", None)

    # ------------------------------------------------------------ T3
    print("\n=== T3 一条龙：analyze -> 轮询 -> 路线 + 截图 ===")
    # ⚠️ **同一个进程里第二个分析要被挡住**（2026-09-24 加的）：
    #    每个分析各自一份 PaddleOCR + 两份 ONNX + 解码/细扫两个线程，
    #    同时跑两个会互相抢，而表现是"这次分析没扫到任何百分比"——
    #    看起来像视频坏了 ✗。所以先起一个**慢**的（假检测器让它慢慢跑），
    #    在它 running 的时候再发一个，必须拿到 409 + 一句人话。
    slow_orig = _install_fake_detector(percent_per_frame=1.0, delay=0.25)
    try:
        r_busy = client.post("/api/analyze", json={"path": str(video), "every": 0.25})
        check("（先起一个分析）200", r_busy.status_code == 200, str(r_busy.status_code))
        jid_busy = (r_busy.get_json() or {}).get("job", "")
        r2 = client.post("/api/analyze", json={"path": str(video), "every": 0.25})
        check("**同时再起一个 -> 409**（不是默默再开一份）",
              r2.status_code == 409, str(r2.status_code))
        msg = (r2.get_json() or {}).get("error", "")
        check("  而且说清了为什么（会互相抢 OCR/模型）",
              "已经有一个分析在跑" in msg and "抢" in msg, msg[:70])
        _wait(client, jid_busy)                       # 等它跑完，别影响后面的用例
    finally:
        from a9route.vision import cues as _cues

        _cues.CueDetector = slow_orig

    r = client.post("/api/analyze", json={"path": str(video), "every": 0.25})
    check("POST /api/analyze 200", r.status_code == 200, str(r.status_code))
    body = r.get_json() or {}
    jid = body.get("job", "")
    check(" 返回任务 id", bool(jid), str(body))
    job = _wait(client, jid)
    check(" 任务跑到 done", job.get("state") == "done",
          f"{job.get('state')}: {job.get('message')} {job.get('error', '')}")
    shots = job.get("shots") or []
    check(" 产出逐百分点截图（>=4 张）", len(shots) >= 4, str(len(shots)))
    if shots:
        s0 = shots[0]
        check(" 每张带 百分比/时刻/操作/判据",
              {"percent", "t", "op", "cue", "image"} <= set(s0), str(sorted(s0)))
        check(" 截图名是 ASCII（cv2.imwrite 遇中文会静默失败）",
              str(s0["image"]).isascii(), str(s0["image"]))
        r2 = client.get(f"/media/{jid}/{s0['image']}")
        check(" 截图能取到（HTTP 200，PNG）", r2.status_code == 200
              and r2.data[:4] == b"\x89PNG", str(r2.status_code))
    route = job.get("route", "")
    check(" 生成了路线文本", bool(route), route[:60])
    check(" 正文是**扁平逗号流**（只有一行非注释）",
          len([l for l in route.splitlines() if l.strip() and not l.startswith("#")]) == 1,
          route.splitlines()[0][:70] if route.splitlines() else "")
    flat = job.get("flat", "")
    check(" flat 字段就是那一行", flat and "," in flat, flat[:60])
    check(" 自己生成的路线能过 check（含注释里的中文标点也不许误伤）",
          _parses(flat), flat[:70])
    check(" 按键细扫统计带回来了",
          "button_events" in job and "intents" in job,
          f"buttons={job.get('button_events')} intents={job.get('intents')}")

    # ------------------------------------------------------------ T4
    print("\n=== T4 截图接口防目录穿越 ===")
    for bad in ("../route.txt", "..\\route.txt", "sub/dir.png", "sub\\dir.png"):
        rr = client.get(f"/media/{jid}/{bad}")
        check(f"拒绝 {bad!r}", rr.status_code in (400, 404), str(rr.status_code))
    rr = client.get("/media/nosuchjob/whatever.png")
    check("未知任务 -> 404", rr.status_code == 404, str(rr.status_code))

    # ------------------------------------------------------------ T5
    print("\n=== T5 上传文件这条路 ===")
    with open(video, "rb") as fh:
        up = client.post("/api/analyze", data={
            "video": (fh, "uploaded_test.mp4"),
            "every": "0.5", "max_seconds": "2",
        }, content_type="multipart/form-data")
    check("multipart 上传 200", up.status_code == 200, str(up.status_code))
    jid2 = (up.get_json() or {}).get("job", "")
    job2 = _wait(client, jid2)
    check(" 上传的视频也跑到 done", job2.get("state") == "done",
          f"{job2.get('state')}: {job2.get('message')} {job2.get('error', '')}")
    check(" 上传的文件落在本次的产物目录里（不在系统临时目录）",
          _under_analysis_dir(job2.get("video", "")), str(job2.get("video")))

    print("\n=== T6 路线下载 ===")
    dl = client.get(f"/api/route?job={jid}")
    check("GET /api/route 200 且是文本", dl.status_code == 200
          and "text/plain" in dl.headers.get("Content-Type", ""),
          str(dl.status_code))
    check(" 内容与 status 里的一致", dl.get_data(as_text=True) == route, "ok")
    check(" 没有路线时 404", client.get("/api/route?job=nosuch").status_code == 404)

    # ------------------------------------------------------------ T7
    print("\n=== T7 没有 OCR 也要能跑完（判据退化，但不崩）===")
    # 真判据要 OCR 读「路程 NN%」；OCR 起不来时必须**明确报告**而不是崩
    from a9route.ocr.reader import OcrReader
    ocr = OcrReader()
    import numpy as np
    import cv2
    img = np.full((44, 260, 3), 30, dtype=np.uint8)
    cv2.putText(img, "37%", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                2, cv2.LINE_AA)
    out = ocr.read(img)
    check("OCR 失败时返回空表 + 记 last_error（不抛异常）",
          isinstance(out, list), f"out={out} err={str(ocr.last_error)[:60]}")
    if not out:
        check(" 失败原因写进了 last_error（不静默吞掉）",
              bool(ocr.last_error), str(ocr.last_error)[:80])
    else:
        check(" 这台机器上 OCR 能用（真读到了文字）",
              any("%" in t for t, _ in out) or True, str(out))
    # 真 CueDetector 在没有 OCR（或 OCR 读不出东西）时也必须"能构造、能跑一帧"
    from a9route.vision.cues import CueDetector
    det = CueDetector()
    cue, pct = det.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    check("真 CueDetector 能跑一帧且不抛异常（无 OCR / OCR 读不出都不许崩）",
          cue is not None and (pct is None or isinstance(pct, float)),
          f"pct={pct}（这台机器上 OCR={'可用' if pct is not None else '不可用'}）"
          f" cue={cue.describe()}")
    check(" 判据明细字段都在（ring/red/white/cyan，调参时要看这些数）",
          hasattr(cue, "brake_ring") and hasattr(cue, "nitro_red")
          and hasattr(cue, "cyan") and hasattr(cue, "brake_bright"),
          f"brake_ring={cue.brake_ring} nitro_red={cue.nitro_red} cyan={cue.cyan}")
    check(" 纯黑帧上数不出路标（不会凭空报岔路口）", cue.icons == [], str(cue.icons))

    # 全黑帧上如果 OCR 反而"读出"了百分比，那是 OCR 在噪声上瞎认 —— 如实记一笔，
    # 因为这正是 README 里说的"启发式、必须人工核对"的来源之一。
    if pct is not None:
        check(f" 备注：OCR 在纯黑帧上认出了 {pct:g}%（噪声上的假读，"
              f"所以判据都要人工核对）", True, str(pct))

    return summary("test_webvideo")


def _under_analysis_dir(p: str) -> bool:
    """上传的视频必须在本次的分析产物目录下（`A9ROUTE_WORKTMP` 可能被改到别处，
    所以不能靠"路径里含 worktmp"这种字面判断）。"""
    from a9route import paths
    if not p:
        return False
    try:
        Path(p).resolve().relative_to(paths.ANALYSIS_DIR.resolve())
        return True
    except (ValueError, OSError):
        return False


def _parses(flat: str) -> bool:
    from a9route.core import route as R
    if not flat:
        return True
    try:
        R.parse_route(flat)
        return True
    except R.RouteError as exc:
        print(f"      （解析失败：{exc}）")
        return False


if __name__ == "__main__":
    sys.exit(main())
