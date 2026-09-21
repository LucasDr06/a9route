# -*- coding: utf-8 -*-
"""Web 界面：**选路标注页**（`/choice`）—— 和刹车/氮气那页**完全分开**。

业务逻辑全在 `train/choice.py`（可离线测），这个模块只做 HTTP 包装。

## 为什么单独一页而不是加在 `/dataset` 里

用户明确要求"给一个单独的界面，不要和氮气和刹车放一起"，
而且两者确实不该混：

* **标的对象不同**：刹车/氮气是"两个固定位置的键亮没亮"，选路是"一排圆形路标里有几个、选中哪个"；
* **标注方式不同**：刹车/氮气是**逐个开关**，选路是**三个选择题**；
* **类别方案不同**：`brake_pressed/nitro_pressed` vs `choice_icon/choice_selected`。

放一起会让"一帧上同时有两套类别的框"变得没法解释（点哪个框算谁的？）。

## 接口（前端只依赖这些）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET  | `/choice` | 页面 |
| GET  | `/api/choice/list` | 选路数据集列表 + 进度 |
| GET  | `/api/choice/frames` | 一页帧（三个答案 + 启发式检测 + 框） |
| GET  | `/api/choice/image` | band（路标带放大）/ full（整帧缩略图） |
| POST | `/api/choice/answer` | 写回三个选择题的答案 |
| POST | `/api/choice/undo` | 撤销上一次写回 |
"""
from __future__ import annotations

from flask import Blueprint, Response, jsonify, request, send_from_directory

from a9route import paths
from a9route.train import choice as C

bp = Blueprint("choice", __name__)


def _int(name, default, lo, hi):
    try:
        v = int(float(request.args.get(name) or default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


@bp.get("/choice")
def page():
    return send_from_directory(str(paths.WEB_TEMPLATES_DIR), "choice.html")


@bp.get("/api/choice/list")
def api_list():
    return jsonify(C.dataset_list())


@bp.get("/api/choice/frames")
def api_frames():
    name = request.args.get("name") or C.DEFAULT_NAME
    try:
        data = C.frame_rows(name,
                            offset=_int("offset", 0, 0, 10 ** 7),
                            limit=_int("limit", 24, 1, 200),
                            flt=(request.args.get("filter") or "all").strip(),
                            search=(request.args.get("search") or "").strip())
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 404
    except Exception as exc:                          # noqa: BLE001
        return jsonify(error="{0}: {1}".format(type(exc).__name__, exc)), 400
    return jsonify(data)


@bp.get("/api/choice/image")
def api_image():
    """`mode=band` 是**路标带**（屏幕上方那条，`CHOICE_BAND`）放大图；
    `mode=full` 是整帧缩略图。

    band 是标注时真正要盯着看的地方（路标是圆、还带蓝色高亮，
    缩放看才分得清"选中没有"），所以默认给 band。
    """
    import cv2

    from a9route.train import audit as A

    name = request.args.get("name") or C.DEFAULT_NAME
    file = request.args.get("file") or ""
    mode = (request.args.get("mode") or "band").strip().lower()
    ds = A.dataset_dir(name)
    p = A.resolve_image(ds, file)
    if p is None:
        return ("", 404)
    frame = cv2.imread(str(p))
    if frame is None:
        return ("", 404)
    if mode == "full":
        w = _int("w", 480, 80, 1280)
        if frame.shape[1] > w:
            sc = w / float(frame.shape[1])
            frame = cv2.resize(frame, (w, max(1, int(frame.shape[0] * sc))),
                               interpolation=cv2.INTER_AREA)
    else:
        # 裁**和 `train.choice.band_crop_rect()` 完全同一块**（含 10px 边距）——
        # 那个函数同时被 `/api/choice/frames` 返回给前端做坐标换算，
        # 所以两边的边界**不可能漂移**（各自写一遍常量必然漂移 ✗）
        bx, by, bw, bh = C.band_crop_rect(frame.shape[1], frame.shape[0])
        frame = frame[by:by + bh, bx:bx + bw]
        z = _int("zoom", 2, 1, 4)
        if z > 1:
            frame = cv2.resize(frame, (frame.shape[1] * z, frame.shape[0] * z),
                               interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return ("", 500)
    resp = Response(buf.tobytes(), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "public, max-age=600"
    return resp


@bp.post("/api/choice/answer")
def api_answer():
    """写回三个选择题的答案。

    请求体：`{"dataset": "choice", "items": [{"file", "has", "count", "selected",
    "boxes"?, "status"?}, ...]}`（`boxes` 是**整帧像素**的 `[x,y,w,h]` 列表，
    只在手工点过位置时才给）。
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data.get("items"):
        return jsonify(error='要 POST 一个 {"items":[{file,has,count,selected}]} 的 JSON'), 400
    name = data.get("dataset") or request.args.get("name") or C.DEFAULT_NAME
    try:
        res = C.apply_answers(data["items"], name=name, progress=None)
    except Exception as exc:                          # noqa: BLE001
        return jsonify(error="{0}: {1}".format(type(exc).__name__, exc)), 400
    return jsonify(result=res, dataset=name)


@bp.post("/api/choice/undo")
def api_undo():
    data = request.get_json(silent=True) or {}
    name = data.get("dataset") or request.args.get("name") or C.DEFAULT_NAME
    try:
        res = C.undo_answers(name=name)
    except Exception as exc:                          # noqa: BLE001
        return jsonify(error="{0}: {1}".format(type(exc).__name__, exc)), 400
    return jsonify(result=res, dataset=name)
