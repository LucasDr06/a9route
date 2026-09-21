# -*- coding: utf-8 -*-
"""Web 界面：**数据集体检 / 逐帧对照**（`a9route serve` 的第二个页面）。

页面在 `/dataset`，回答一个问题：

> **数据集里的标注，和"标定"（`config.json` 的按键框 + 阈值）对得上吗？**

它和 `a9route train review` 生成的静态复核页是**两个用途**，别混：

| | `train review` 的静态页 | 这个页面 |
|---|---|---|
| 目的 | **改标注**（导出修正 JSON -> `train apply`） | **查对应关系**（数据集 vs 标定 vs 判据） |
| 看什么 | 两张放大图 + 当前标注 | 整帧上的**标注框 vs 标定框**、**存的信号 vs 现在重算的信号**、体检结论 |
| 什么时候用 | 准备训练之前 | 训练结果不对、或改了 `config.json` 之后 |
| 数据来源 | `pool/` 的原图 | 划分后的 `images/{train,val}` + `meta.jsonl` |

后端逻辑全在 `train.audit` 里（CLI 的 `a9route train audit` 用的是同一份），
这个模块只做 HTTP 包装。

## ⚠️ 安全：图片接口不能让路径跑出去

`/api/dataset/image?file=...` 直接吃用户给的字符串，所以**一律走
`train.audit.resolve_image()`** —— 它只接受纯文件名（拒绝 `/`、`\\`、`..`、`:`）。
**不要**在这里自己拼路径（原项目那套接口就是自己拼的，靠 `send_from_directory`
兜着；这里干脆连机会都不给）。
"""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, jsonify, request, send_from_directory

from a9route import paths
from a9route.train import audit as A

bp = Blueprint("dataset", __name__)


def _flag(name: str, default: bool = False) -> bool:
    raw = (request.args.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on", "开")


def _int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(float(request.args.get(name) or default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


@bp.get("/dataset")
def page():
    return send_from_directory(str(paths.WEB_TEMPLATES_DIR), "dataset.html")


@bp.get("/api/dataset/list")
def api_list():
    ds = A.datasets()
    return jsonify(datasets=ds, root=str(paths.DATASETS_DIR),
                   models_dir=str(paths.MODELS_DIR))


@bp.get("/api/dataset/audit")
def api_audit():
    name = request.args.get("name") or "keys"
    try:
        rep = A.audit(name, recheck=_flag("recheck", True),
                      limit=_int("limit", 300, 1, 20000))
    except Exception as exc:                          # noqa: BLE001
        return jsonify(error="{0}: {1}".format(type(exc).__name__, exc)), 400
    return jsonify(rep.as_dict())


@bp.get("/api/dataset/frames")
def api_frames():
    name = request.args.get("name") or "keys"
    try:
        data = A.frame_rows(
            name,
            split=request.args.get("split") or "",
            reason=request.args.get("reason") or "",
            cls=request.args.get("cls") or "",
            status=request.args.get("status") or "",
            only=request.args.get("only") or "",
            search=request.args.get("search") or "",
            offset=_int("offset", 0, 0, 10 ** 7),
            limit=_int("limit", 24, 1, 200),
            recheck=_flag("recheck", False),
        )
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 404
    except Exception as exc:                          # noqa: BLE001
        return jsonify(error="{0}: {1}".format(type(exc).__name__, exc)), 400
    return jsonify(data)


@bp.get("/api/dataset/image")
def api_image():
    """整帧缩略图（`mode=full`）或按键框放大图（`mode=crop`）。

    **路径校验全在 `audit.resolve_image()` 里**；这里只挑要返回哪种图。
    """
    import cv2

    name = request.args.get("name") or "keys"
    file = request.args.get("file") or ""
    mode = (request.args.get("mode") or "full").strip().lower()
    if mode == "crop":
        which = "nitro" if (request.args.get("which") or "").strip() == "nitro" else "brake"
        img = A.crop_preview(name, file, which=which,
                             zoom=_int("zoom", 2, 1, 6))
    else:
        img = A.full_preview(name, file, width=_int("w", 480, 80, 1280))
    if img is None:
        return ("", 404)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return ("", 500)
    resp = Response(buf.tobytes(), mimetype="image/jpeg")
    # 图片内容只跟数据集文件有关（改标注不会改图），可以放心让浏览器缓存
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


@bp.post("/api/dataset/corrections")
def api_corrections():
    """把界面上改的标注写回磁盘（**和 `train apply` 是同一份逻辑**）。

    请求体就是复核页「导出修正」的那个 JSON 形状：
    `{"items": [{"file": "...", "boxes": ["brake_pressed"], "status": "reviewed"}]}`
    —— 这样界面导出的文件和命令行 `a9route train apply` 完全通用。
    """
    from a9route.train import review as RV

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data.get("items"):
        return jsonify(error="要 POST 一个 {items:[{file, boxes}]} 的 JSON"), 400
    name = data.get("dataset") or request.args.get("name") or "keys"
    try:
        res = RV.apply_corrections(data, name=name, progress=None)
    except Exception as exc:                          # noqa: BLE001
        return jsonify(error="{0}: {1}".format(type(exc).__name__, exc)), 400
    return jsonify(result=res, dataset=name)


@bp.post("/api/dataset/mark_reviewed")
def api_mark_reviewed():
    """把若干帧标成「已复核」，**一个字节的标注文件都不改**。

    界面在**翻页**时自动调它（用户原话："如果我翻过这一页就把当前页的数据
    作为已复核"）—— 没错的帧不必逐个点「没问题」。

    ⚠️ 和 `/corrections` 的区别很要紧：那个会**按当前标定重写标注框**，
    这个只改 `meta.jsonl` 的 `status`。翻页这种批量动作必须用后者 ——
    否则会把"数据集和标定对不上"这个体检结论悄悄抹平
    （详见 `train.review.mark_reviewed` 的说明）。
    """
    from a9route.train import review as RV

    data = request.get_json(silent=True) or {}
    files = data.get("files") or []
    if not isinstance(files, list) or not files:
        return jsonify(error='要 POST {"files": ["xxx.jpg", …]}'), 400
    name = data.get("dataset") or request.args.get("name") or "keys"
    try:
        res = RV.mark_reviewed(files, name=name, progress=None)
    except Exception as exc:                          # noqa: BLE001
        return jsonify(error="{0}: {1}".format(type(exc).__name__, exc)), 400
    return jsonify(result=res, dataset=name)


@bp.post("/api/dataset/undo_mark")
def api_undo_mark():
    """撤销上一次「标记已复核」（手快连着翻页时用）。只认最近一次。"""
    from a9route.train import review as RV

    data = request.get_json(silent=True) or {}
    name = data.get("dataset") or request.args.get("name") or "keys"
    try:
        res = RV.undo_last_mark(name=name)
    except Exception as exc:                          # noqa: BLE001
        return jsonify(error="{0}: {1}".format(type(exc).__name__, exc)), 400
    return jsonify(result=res, dataset=name)
