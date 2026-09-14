# -*- coding: utf-8 -*-
"""reader.py —— PaddleOCR 的懒加载封装（**只读小图，不做任何设备操作**）。

这段代码整体来自原项目 `asphalt9auto/a9auto/core/decision.py` 的 `OcrReader`，
判据/静音处理**一字未改**，只去掉了同文件里的模板匹配与点击部分。

## 为什么这么写（都是实测踩出来的）

* **懒加载 + 实例复用**：`PaddleOCR(...)` 初始化要几秒，而一次视频分析要读上千次小图，
  每次重建会慢到不可用。
* **初始化必须静音**，否则控制台被刷屏：

  | 刷屏内容 | 来源 | 怎么静音 |
  |---|---|---|
  | `Connectivity check ... has been skipped ...` | **import 时**就打了（paddlex logger INFO） | logger 抬到 ERROR，且 **import 也要包进上下文** |
  | `Creating model: (...)` / `Model files already exist...` | paddlex logger INFO | 同上 |
  | `UserWarning: No ccache found ...` | warnings | `warnings.simplefilter("ignore")` |
  | `INFO: Could not find files for the given pattern(s).` | paddle 在**子进程**里探测 ccache，`where` 打出来的 | Python 层拦不住 → 临时 dup 进程级 fd 1/2 到 devnull |

* **失败不要静默吞掉**：`read()` 出错返回 `[]` 并把原因记进 `self.last_error`
  —— 否则「OCR 没命中」和「OCR 根本没跑起来」分不清（这条吃过亏）。

## 模型与缓存

* 显式指定 `PP-OCRv5_mobile_det/rec`；此时 PaddleOCR 会**忽略** `lang`，
  中文识别靠模型自带字符集（实测能识别中文界面文字）。
* 默认缓存在 `~/.paddlex`。若项目里有 `worktmp/paddlex` 就用它
  （原项目是在受限沙箱里跑，工作区外的模型读不到）。
* 没有 GPU 也能跑（`paddle` 的 CPU 版）；有 CUDA 的 paddle 会**自动走 GPU**，
  实测 OCR 本来就在 GPU 上。
"""
from __future__ import annotations

import contextlib
import os
import time

import numpy as np

from a9route.paths import ROOT


@contextlib.contextmanager
def _silenced_output():
    """临时把**进程级** stdout/stderr（fd 1/2）+ warnings + paddlex 日志压掉。

    为什么连 fd 都要动：`PaddleOCR(...)` 初始化时 Paddle 会在**子进程**里探测 ccache，
    `where` 找不到就打一行 `INFO: Could not find files for the given pattern(s).` ——
    那行是子进程直接写到控制台的，`contextlib.redirect_stdout` 之类**拦不住**。
    所以这里 dup 出 fd 1/2、指向 `os.devnull`，出来时再恢复。

    只用来包住"创建 OCR 引擎"这一段：真出错照旧会抛（异常走 Python 层，不受影响）。
    """
    import logging
    import warnings

    @contextlib.contextmanager
    def _null_fds():
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            saved = (os.dup(1), os.dup(2))
        except OSError:            # 某些环境拿不到 fd，退回"只静音 Python 层"
            yield
            return
        try:
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            yield
        finally:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            for f in (devnull, *saved):
                try:
                    os.close(f)
                except OSError:
                    pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prev = {}
        for name in ("paddlex", "paddle", "paddleocr", "ppocr"):
            lg = logging.getLogger(name)
            prev[name] = lg.level
            lg.setLevel(logging.ERROR)
            for handler in list(lg.handlers):
                prev[f"{name}:{id(handler)}"] = handler.level
                handler.setLevel(logging.ERROR)
        try:
            with _null_fds():
                yield
        finally:
            for name in ("paddlex", "paddle", "paddleocr", "ppocr"):
                logging.getLogger(name).setLevel(prev.get(name, logging.NOTSET))
                for handler in list(logging.getLogger(name).handlers):
                    k = f"{name}:{id(handler)}"
                    if k in prev:
                        handler.setLevel(prev[k])


class OcrReader:
    """PaddleOCR 懒加载封装：首次调用才初始化（初始化要几秒），之后复用。

    只应传入**裁剪后的小图**（例如左上角那块「路程 37%」）。
    每次新建 PaddleOCR 实例会重复几秒开销，所以这里缓存实例。
    """

    def __init__(self):
        self._ocr = None
        self.load_seconds = 0.0
        self.last_error: str | None = None
        self.calls = 0

    def _ensure(self):
        if self._ocr is None:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            # 项目内有一份模型缓存就优先用它 —— 受限沙箱里读不到 ~/.paddlex
            local_cache = ROOT / "worktmp" / "paddlex"
            if local_cache.is_dir():
                os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(local_cache))
            # **import 和创建都要包进静音上下文**（只包创建会漏掉 import 时那一行 —— 实测踩过）
            with _silenced_output():
                from paddleocr import PaddleOCR
                t0 = time.perf_counter()
                self._ocr = PaddleOCR(
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            self.load_seconds = time.perf_counter() - t0
        return self._ocr

    def available(self) -> bool:
        """能不能用（装没装 paddleocr）。**懒加载成功前不会真的去初始化**。"""
        try:
            import paddleocr  # noqa: F401
        except Exception as exc:               # pragma: no cover - 取决于环境
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        return True

    def read(self, img: np.ndarray) -> list[tuple[str, float]]:
        """对裁剪图做识别，返回 [(text, score), ...]。

        失败时返回 []，并把原因记到 `self.last_error`：**不要静默吞掉**，
        否则「OCR 没命中」和「OCR 根本没跑起来」分不清。
        """
        if img is None or img.size == 0:
            self.last_error = "empty crop"
            return []
        try:
            ocr = self._ensure()
            self.calls += 1
            res = ocr.predict(img)
            out: list[tuple[str, float]] = []
            for r in res or []:
                d = _as_dict(r)
                texts = d.get("rec_texts") or []
                scores = d.get("rec_scores") or [1.0] * len(texts)
                for i, t in enumerate(texts):
                    out.append((t, float(scores[i]) if i < len(scores) else 1.0))
            if not out:
                self.last_error = "no text in crop"
            else:
                self.last_error = None
            return out
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return []

    def read_located(self, img: np.ndarray
                     ) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        """同 `read()`，但**带上每段文字在裁剪图里的位置** (x, y, w, h)。

        （原项目里那些"页面上的控件位置会变、要照 OCR 位置点"的流程用它；
        视频分析目前只用 `read()`，留着是为了以后"照读数位置取证"方便。）
        """
        if img is None or img.size == 0:
            self.last_error = "empty crop"
            return []
        try:
            ocr = self._ensure()
            self.calls += 1
            res = ocr.predict(img)
            out: list[tuple[str, float, tuple[int, int, int, int]]] = []
            for r in res or []:
                d = _as_dict(r)
                texts = d.get("rec_texts") or []
                scores = d.get("rec_scores") or [1.0] * len(texts)
                polys = d.get("rec_polys")
                if polys is None:
                    polys = d.get("dt_polys")
                boxes = d.get("rec_boxes")
                for i, t in enumerate(texts):
                    rect = None
                    if polys is not None and i < len(polys):
                        rect = _poly_bbox(polys[i])
                    elif boxes is not None and i < len(boxes):
                        b = boxes[i]
                        try:
                            x0, y0, x1, y1 = (int(v) for v in b)
                            rect = (x0, y0, x1 - x0, y1 - y0)
                        except Exception:
                            rect = None
                    if rect is None:
                        rect = (0, 0, 0, 0)
                    out.append((t, float(scores[i]) if i < len(scores) else 1.0, rect))
            if not out:
                self.last_error = "no text in crop"
            return out
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return []


def _poly_bbox(poly) -> tuple[int, int, int, int]:
    """四点多边形 -> 外接矩形 (x, y, w, h)。"""
    try:
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
    except Exception:
        return (0, 0, 0, 0)
    x0, y0 = int(min(xs)), int(min(ys))
    return (x0, y0, int(max(xs)) - x0, int(max(ys)) - y0)


def _as_dict(res) -> dict:
    """PaddleOCR 3.x 的结果对象 → dict（兼容 dict / 带 json 属性的对象）。"""
    if isinstance(res, dict):
        return res
    for attr in ("json", "res"):
        v = getattr(res, attr, None)
        if isinstance(v, dict):
            return v.get("res", v) if "res" in v else v
        if callable(v):
            try:
                r = v()
                if isinstance(r, dict):
                    return r.get("res", r) if "res" in r else r
            except Exception:
                pass
    return {}
