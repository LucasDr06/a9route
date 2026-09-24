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


def t5_no_shots_is_not_no_ops():
    """**0 张截图 ≠ 「这一局没有操作」**（2026-09-24 用户拖进来一段录像真踩）。

    那次现场：`route.txt` 里写着「这一局模型没判出任何操作 —— 所以正文是空的
    （**不是**「没扫描成功」）」，而文件里一条明细行都没有 —— 那是**扫描失败**
    （「路程 NN%」一个都没读到）。那句话**恰好说反了**，用户会跑去查按键/阈值 ✗✗。

    这里锁三件事：
      ① 空截图时不许说"没判出任何操作"；
      ② **原因要写进 route.txt**（文件得能自己解释自己）；
      ③ 诊断**不许只凭一两帧就冤枉视频** —— 抽 5 帧，读得到就明说"不是视频的问题"。
    """
    print("\n=== T5 0 张截图时的报告与诊断 ===")
    from a9route.vision import video as V
    from a9route.vision.video import PercentShot

    txt = V.suggest_route([], fine_scan_ran=True)
    check("**空截图 -> 说「扫描失败」**，不说「模型没判出任何操作」",
          "没扫到" in txt and "模型没判出任何操作" not in txt,
          txt.splitlines()[0][:60])
    check("  也不再有那句说反了的话（不是「没扫描成功」）",
          "不是「没扫描成功」" not in txt, "")
    check("  正文是空的（不给路线）",
          not [ln for ln in txt.splitlines() if ln.strip() and not ln.startswith("#")], "")

    # 对照：**有截图但没操作**时，原来那句话仍然是对的（别把这条一起改坏）
    shot = PercentShot(percent=5.0, t=1.0, path="", op="", cue=None)
    txt2 = V.suggest_route([shot], fine_scan_ran=True)
    check("  对照：有截图、没操作 -> 仍说「模型没判出任何操作」",
          "模型没判出任何操作" in txt2, txt2.splitlines()[1][:60])

    # 真跑一次：blocker 必须落进 route.txt（而不是一个空文件）
    work = TMP / "no_shots_fix"
    work.mkdir(parents=True, exist_ok=True)
    saved_diag = analysis._diagnose_no_shots
    analysis._diagnose_no_shots = lambda v: "测试原因：OCR 读不到模型"
    try:
        rep = analysis.analyze(fake_hud_video(work / "v.mp4", seconds=1.0, fps=10.0),
                               out_dir=work / "out", with_buttons=True,
                               detector=_NoShotsDetector(), progress=None)
    finally:
        analysis._diagnose_no_shots = saved_diag
    rt = (work / "out" / "route.txt").read_text(encoding="utf-8")
    check("这次分析带 blocker", bool(rep.blocker), rep.blocker.splitlines()[0][:40])
    first = rep.blocker.splitlines()[0].strip()
    check("**原因写进了 route.txt**（文件能自己解释自己）", bool(first) and first in rt,
          "找 {0!r}".format(first[:40]))
    check("  文件里明说「这次分析无效」", "这次分析无效" in rt, "")
    check("  文件不是空的（空文件最难查）", len(rt.strip()) > 40, str(len(rt)))
    check("  也没有「这一局没操作」那种误导话", "不是「没扫描成功」" not in rt, "")


def t6_diagnosis_samples_many_frames():
    """诊断要**抽 5 帧**，而不是只看一帧就下结论。

    真踩过：用户那段录像在中点那一帧恰好读不到字，诊断就说
    「这一帧可能正好在加载/回放画面 —— 换一段比赛中的录像」，
    可**同一份文件重跑完全正常**（99 张截图）—— 把"这一次运行的问题"
    说成了"你的视频有问题" ✗。现在抽查多帧；读得到就明说"不是视频的问题"。
    """
    print("\n=== T6 诊断抽 5 帧 ===")
    import inspect

    src = inspect.getsource(analysis._diagnose_no_shots)
    # ⚠️ 只看**函数体**（docstring 里会引用旧文案，拿全文断言会误报 —— 真踩过）
    body = src.split('"""')[-1]
    check("源码里写的是**多帧**抽样（不是只看 `n // 2`）",
          "n // 2" not in body and "0.05" in body and "0.85" in body, "")
    check("  读得到时明说「不是视频的问题」", "不是视频的问题" in body, "")
    check("  读得到时给出的建议是「重跑」，不是「换一段录像」",
          "重跑" in body and "换一段比赛中的录像" not in body, "")
    check("  读不到时才提 progress_box / 分辨率",
          "progress_box" in body and "1280" in body, "")


def main() -> int:
    global TMP
    TMP = tmp_dir()
    t1_cache_probe()
    t2_cache_mirror()
    t3_no_shots_blocker()
    t4_diagnosis_text()
    t5_no_shots_is_not_no_ops()
    t6_diagnosis_samples_many_frames()
    return summary("test_ocr")


if __name__ == "__main__":
    sys.exit(main())
