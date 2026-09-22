# -*- coding: utf-8 -*-
"""OCR 模型缓存的三态判定 + 「一张图都没扫到」时的**响亮报错**。

这个 bug 真踩过：`~/.paddlex` 在工作区外、受限环境读不到 ->
`percent` 恒为 None -> 一张图都存不下 -> **路线是空的**。
用户看到的是「识别不到任何操作」，看着像检测的问题 ✗✗（NOTES §10.17）。

**为什么在 a9route 这边**：OCR 是**运行时**的事（读「路程 NN%」）。
数据集/训练那半已经拆到 `a9lab`，但"分析到底有没有扫到东西"属于本项目的健康检查。
"""
from __future__ import annotations

import sys
from pathlib import Path

from a9route import analysis, paths
from a9route.ocr import reader as OR
from a9route.tests.support import check, fake_hud_video, summary, tmp_dir

TMP = Path(".")


class _NoShotsDetector:
    """永远读不到百分比的假检测器（用来测「一张图都没扫到」这条路）。"""

    def __init__(self, *a, **kw):
        pass

    def detect(self, frame):
        from a9route.vision.cues import Cue

        return Cue(), None


def t1_cache_probe():
    """缓存目录的三种状态：ok / missing / denied（**不许抛**）。"""
    print("\n=== T1 OCR 缓存的三态判定 ===")
    import os

    empty = TMP / "ocr_empty"
    empty.mkdir(parents=True, exist_ok=True)
    check("目录不存在 -> missing", OR._probe(TMP / "没有这个目录") == "missing", "")
    check("目录在但模型不齐 -> missing", OR._probe(empty) == "missing", OR._probe(empty))

    good = TMP / "ocr_good"
    for m in OR.NEEDED_MODELS:
        (good / "official_models" / m).mkdir(parents=True, exist_ok=True)
        (good / "official_models" / m / "inference.yml").write_text(
            "x: 1\n", encoding="utf-8")
    check("两个模型都齐 -> ok", OR._probe(good) == "ok", OR._probe(good))

    real_listdir = os.listdir

    def _denied(_path):
        raise PermissionError(13, "Access is denied")

    try:
        os.listdir = _denied
        check("**列不出来（受限环境）-> denied 而不是抛异常**",
              OR._probe(good) == "denied", OR._probe(good))
    finally:
        os.listdir = real_listdir

    # `is_file()` 抛异常时也不能崩（**真踩过：它会抛 PermissionError**）
    real_is_file = Path.is_file

    def _boom(self):
        raise PermissionError(5, "Access is denied")

    try:
        Path.is_file = _boom
        check("  is_file() 抛异常也不崩", OR._probe(good) in ("ok", "denied"),
              OR._probe(good))
        check("  _models_present 永远只回是/否",
              OR._models_present(good) in (True, False), "")
    finally:
        Path.is_file = real_is_file


def t2_cache_mirror():
    """`ensure_cache()`：听用户的显式设置、force 时的两种结果、自动选工作区那份。"""
    print("\n=== T2 把 OCR 模型镜像进工作区 ===")
    import os

    empty = TMP / "ocr_empty"
    good = TMP / "ocr_good"
    real_wt = paths.WORKTMP_DIR
    real_env = os.environ.get("PADDLE_PDX_CACHE_HOME")
    try:
        paths.WORKTMP_DIR = TMP / "ocr_wt"
        # 1) **用户显式设了 PADDLE_PDX_CACHE_HOME 就完全听他的**（不镜像、不校验）——
        #    因为 paddlex 可能会往那儿下载模型，管太宽反而坏事。
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(empty)
        check("显式设了 PADDLE_PDX_CACHE_HOME -> 原样照用（听用户的）",
              OR.ensure_cache() == empty, str(OR.ensure_cache()))
        # 2) force 且源里什么都没有 -> 明确报错
        paths.WORKTMP_DIR = TMP / "ocr_wt3"
        try:
            OR.ensure_cache(force=True)
            check("force 且源里没有模型 -> 明确报错（不静默）", False, "居然没报错")
        except RuntimeError as exc:
            check("force 且源里没有模型 -> 明确报错（不静默）",
                  "找不到 OCR 模型" in str(exc), str(exc)[:50])
        # 3) force 且源里有 -> 真的镜像进工作区
        paths.WORKTMP_DIR = TMP / "ocr_wt"
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(good)
        chosen = OR.ensure_cache(force=True)
        check("force=True 真的把模型镜像进工作区",
              chosen == paths.WORKTMP_DIR / "paddlex" and OR._probe(chosen) == "ok",
              str(chosen))
        check("  只抄需要的两个模型（不抄 server 版/UVDoc 那些）",
              sorted(x.name for x in (chosen / "official_models").iterdir())
              == sorted(OR.NEEDED_MODELS),
              sorted(x.name for x in (chosen / "official_models").iterdir()))
        # 4) 把显式环境变量撤掉之后，**自动**用工作区那份（这正是受限环境的走法）
        os.environ.pop("PADDLE_PDX_CACHE_HOME", None)
        check("  撤掉显式设置后，自动选工作区那份（受限环境就靠它）",
              OR.ensure_cache() == chosen, str(OR.ensure_cache()))
    finally:
        paths.WORKTMP_DIR = real_wt
        if real_env is None:
            os.environ.pop("PADDLE_PDX_CACHE_HOME", None)
        else:
            os.environ["PADDLE_PDX_CACHE_HOME"] = real_env


def t3_no_shots_blocker():
    """一张图都没扫到 -> **blocker**（这次结果不可用），而不是给一条空路线。"""
    print("\n=== T3 「一张图都没扫到」要当错误报出来 ===")
    real_diag = analysis._diagnose_no_shots
    try:
        analysis._diagnose_no_shots = lambda v: "测试原因：OCR 读不到模型"
        rep = analysis.analyze(fake_hud_video(TMP / "noblank.mp4", seconds=1.0,
                                              fps=10.0),
                               out_dir=TMP / "no_shots", with_buttons=False,
                               detector=_NoShotsDetector(), progress=None)
        check("**一张图都没扫到 -> blocker 有值**", bool(rep.blocker), rep.blocker)
        check("  ok 变成 False（这次结果不可用）", rep.ok is False, rep.ok)
        check("  to_json 带上 blocker（界面要当错误显示）",
              rep.to_json().get("blocker") == rep.blocker, "")
        check("  warnings 里也说了「这次分析无效」",
              any("无效" in w for w in rep.warnings), rep.warnings)
    finally:
        analysis._diagnose_no_shots = real_diag


def t4_diagnosis_text():
    """诊断那句话必须**能照做**：点明是 OCR、给出命令、列出两个缓存位置。"""
    print("\n=== T4 诊断文本（能直接照着修）===")
    import os

    empty = TMP / "ocr_empty"
    real_wt = paths.WORKTMP_DIR
    real_env = os.environ.get("PADDLE_PDX_CACHE_HOME")
    try:
        paths.WORKTMP_DIR = TMP / "ocr_wt2"
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(empty)
        msg = analysis._diagnose_no_shots(TMP / "noblank.mp4")
        check("诊断第一句就点明是 OCR 的问题（不是视频没操作）",
              "OCR" in msg and "不是视频的问题" in msg, msg.splitlines()[0][:70])
        check("  并给出能直接抄的命令 a9route ocr cache",
              "a9route ocr cache" in msg, "")
        check("  还把两个缓存位置都列出来（好对照）", "paddlex" in msg, "")
    finally:
        paths.WORKTMP_DIR = real_wt
        if real_env is None:
            os.environ.pop("PADDLE_PDX_CACHE_HOME", None)
        else:
            os.environ["PADDLE_PDX_CACHE_HOME"] = real_env


def main() -> int:
    global TMP
    TMP = tmp_dir()
    t1_cache_probe()
    t2_cache_mirror()
    t3_no_shots_blocker()
    t4_diagnosis_text()
    return summary("test_ocr")


if __name__ == "__main__":
    sys.exit(main())
