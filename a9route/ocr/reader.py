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
* 默认缓存在 `~/.paddlex`。**若读不到它（受限环境/只读 HOME），就自动把
  真正需要的那两个模型镜像到 `WORKTMP/paddlex`**，再把
  `PADDLE_PDX_CACHE_HOME` 指过去 —— 见 `ensure_cache()`。
* 没有 GPU 也能跑（`paddle` 的 CPU 版）；有 CUDA 的 paddle 会**自动走 GPU**，
  实测 OCR 本来就在 GPU 上。

## ⚠️ 这条踩过一个大坑（2026-09-15）

`~/.paddlex` 在工作区外。受限环境里**读不到模型文件** →
`OcrReader.read()` 全程返回 `[]` → `RaceReader.percent` 恒为 `None` →
粗扫一张图都存不下 → **路线是空的**。

表现是「**识别不到任何操作**」，看起来像"检测模型没检出东西"，
实际根因跟检测毫无关系 ✗✗。所以这里做了两件事：

1. **自动镜像**（下面 `ensure_cache()`）：只抄真正要用的两个模型（约 21 MB；
   默认缓存里另外 220 MB 是 server 版/方向分类/UVDoc，本模块已显式关掉）；
2. **失败要响亮**：`analysis.analyze()` 在"一张图都没扫到"时会把
   `OcrReader.last_error` 一起报出来并判这次分析**无效**，而不是给一条空路线。
"""
from __future__ import annotations

import contextlib
import os
import shutil
import time
from pathlib import Path

import numpy as np

#: OCR 真正需要的两个模型（就是 `_ensure()` 里显式指定的那两个）。
#: **只镜像它们** —— 默认缓存 240 MB，其中 220 MB 是 `server` 版 det/rec、
#: 方向分类、UVDoc 解畸变，而本模块把后面三个都关掉了，抄过来纯属浪费。
NEEDED_MODELS = ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec")


def cache_dir() -> Path:
    """本项目自己的 OCR 缓存目录（**工作区内**，受限环境也读得到）。"""
    from a9route import paths
    return paths.WORKTMP_DIR / "paddlex"


def default_cache_dir() -> Path:
    """paddlex 的默认缓存（`PADDLE_PDX_CACHE_HOME` 或 `~/.paddlex`）。"""
    env = os.environ.get("PADDLE_PDX_CACHE_HOME")
    if env:
        return Path(env)
    return Path.home() / ".paddlex"


def _models_present(root: Path | None) -> bool:
    """这个目录里那两个模型齐不齐。**只回答是/否，永远不抛。**"""
    return _probe(root) == "ok"


def _probe(root: Path | None) -> str:
    """模型缓存的状态：`"ok"` / `"missing"`（真没有）/ `"denied"`（有但读不到）。

    ⚠️ **必须把"没有"和"读不到"分开** —— 这两种情况的处置完全相反：

    * `missing` -> 首次运行会联网下载（或者用户得把模型放进去）；
    * `denied`  -> 模型**就在那儿**，只是当前环境读不到，**镜像一份到工作区**即可。

    第一版没分，于是受限环境（`is_file()` 直接抛 `PermissionError`）被误判成
    "找不到 OCR 模型"，给出的提示完全是错的 ✗✗。
    """
    if root is None:
        return "missing"
    try:
        names = set(os.listdir(root / "official_models"))
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "denied"            # 目录在，但列不出来
    if not all(m in names for m in NEEDED_MODELS):
        return "missing"
    return "ok" if _readable(root) else "denied"


def _readable(root: Path) -> bool:
    """真的**试读**一下关键文件。

    ⚠️ 不要用 `os.access()` 判断 —— 受限沙箱下它可能说"可读"，
    真 open 才抛 `PermissionError`（实测过）。**能不能读，只有读一次才知道。**
    """
    try:
        with open(root / "official_models" / NEEDED_MODELS[0] / "inference.yml",
                  "rb") as fh:
            fh.read(16)
        return True
    except OSError:
        return False


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


def ensure_cache(*, progress=None, force: bool = False) -> Path | None:
    """保证 OCR 读得到模型；读不到就把需要的两个模型**镜像**到工作区内。

    返回**要设给 `PADDLE_PDX_CACHE_HOME` 的目录**，或 `None`（= 用默认的就行）。

    决策顺序（**正常环境一分钱不花**）：

    1. 用户显式设过 `PADDLE_PDX_CACHE_HOME` -> 完全听他的，不镜像；
    2. 工作区内的 `WORKTMP/paddlex` 已经齐了 -> 用它；
    3. 默认缓存（`~/.paddlex`）**真的读得到** -> 什么都不做，用默认的；
    4. 默认缓存**读不到**（受限环境）-> 把两个模型抄进工作区，用它。

    `force=True`：**不管读不读得到，都抄一份到工作区** ——
    `a9route ocr cache` 用它：在能读到的环境（普通终端）里先抄好，
    之后受限环境（比如被沙箱限制的服务进程）就也能用了。

    ⚠️ 第 4 步在"读不到"的环境里**本身也会失败**（连复制源都读不了）。
    这时给的是**能直接照做**的话：在普通终端里跑一次 `a9route ocr cache`。
    """
    explicit = os.environ.get("PADDLE_PDX_CACHE_HOME")
    if explicit and not force:
        return Path(explicit)

    local = cache_dir()
    if _models_present(local) and not force:
        return local

    default = default_cache_dir()
    state = _probe(default)
    if not force:
        if state == "ok":
            return None                 # 默认那份能用，不动它
        if state == "missing":
            raise RuntimeError(
                "找不到 OCR 模型：`{0}` 里没有 {1}。\n"
                "  · 首次运行会自动联网下载；\n"
                "  · 离线/受限环境请手工把这两个模型目录放进去。"
                .format(default, " / ".join(NEEDED_MODELS)))
    elif state == "missing":
        raise RuntimeError(
            "找不到 OCR 模型：`{0}` 里没有 {1} —— 没有可抄的源。"
            .format(default, " / ".join(NEEDED_MODELS)))

    # 镜像到工作区
    if progress:
        progress("  镜像 OCR 模型 {0} -> {1} …".format(default, local))
    local.mkdir(parents=True, exist_ok=True)
    try:
        for m in NEEDED_MODELS:
            src = default / "official_models" / m
            dst = local / "official_models" / m
            if (dst / "inference.yml").is_file() and not force:
                continue
            if dst.is_dir():
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
    except OSError as exc:
        raise RuntimeError(
            "OCR 模型在 {0}，但**当前环境读不到**，镜像也失败：{1}: {2}\n"
            "  照这个做一次就能修好（只需做一次，抄好之后哪儿都能用）：\n"
            "    在**普通终端**里跑： a9route ocr cache\n"
            "  或手工把那两个目录复制过去：\n"
            "    {3}\\official_models\\{4}\n"
            "      -> {5}\\official_models\\\n"
            .format(default, type(exc).__name__, exc, default,
                    " 和 ".join(NEEDED_MODELS), local)) from exc
    if progress:
        progress("  OCR 模型已在工作区里（{0}，约 21 MB）".format(
            " / ".join(NEEDED_MODELS)))
    return local


class OcrReader:
    """PaddleOCR 懒加载封装：首次调用才初始化（初始化要几秒），之后复用。

    只应传入**裁剪后的小图**（例如左上角那块「路程 37%」）。
    每次新建 PaddleOCR 实例会重复几秒开销，所以这里缓存实例。
    """

    def __init__(self, progress=None):
        self._ocr = None
        self.load_seconds = 0.0
        self.last_error: str | None = None
        self.calls = 0
        #: 镜像模型缓存时给人看的一句话（可选）
        self._progress = progress

    def _ensure(self):
        if self._ocr is None:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            # 模型缓存：工作区内有一份就用它；默认那份读不到就自动镜像一份进来。
            # （受限环境里读不到 ~/.paddlex —— 那会让整个粗扫静默地一张图都存不下）
            chosen = ensure_cache(progress=self._progress)
            if chosen is not None:
                os.environ["PADDLE_PDX_CACHE_HOME"] = str(chosen)
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
