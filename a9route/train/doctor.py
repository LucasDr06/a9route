# -*- coding: utf-8 -*-
"""train.doctor —— **环境与素材自检**：`a9route train doctor`。

这个项目天然要跨两个 Python 环境：

* **训练环境**（本机是 `ai_joy`）：有 torch cu128 + ultralytics，负责训模型、导出 ONNX；
* **运行环境**（本机是 `alphash9auto`）：跑 `a9route analyze`，只要
  cv2/numpy（+ OCR）；**推理走 onnxruntime**，不必装 torch。

跨环境最容易出的错不是代码，是"在错的环境里跑对的命令"：
在 `alphash9auto` 里敲 `a9route train run` 会报"没有 torch"，
而在 `ai_joy` 里敲 `a9route analyze` 会报"读不到路程百分比"（没装 OCR）。
所以 `doctor` 的作用就一个：**一句话告诉你"现在这个环境能干什么、缺什么、怎么补"**。
"""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from a9route import paths
from a9route.train.labels import CLASSES, CLASS_ZH
from a9route.train.runner import DEFAULT_WEIGHT, find_weight

#: 本机已知的训练环境（没有就自己去装一个 —— 见 TRAINING.md）
KNOWN_TRAIN_PYTHONS = [
    r"C:\Users\Admin\miniconda3\envs\ai_joy\python.exe",
    r"C:\Users\Admin\miniconda3\envs\alphash9auto\python.exe",
]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    hint: str = ""            #: 缺了怎么办（**只写能直接抄的命令**）
    #: True = "缺了就算失败"。**默认 False** —— 因为"这个环境没有 torch"
    #: 在**运行环境**里是**预期状态**、不是错误；把预期状态算成失败会让
    #: `train doctor` 永远退 1，人就学会无视它了。
    critical: bool = False


@dataclass
class DoctorReport:
    checks: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", hint: str = "",
            critical: bool = False) -> None:
        self.checks.append(Check(name, bool(ok), detail, hint, bool(critical)))

    @property
    def ok(self) -> bool:
        """只有**关键项**（cv2/numpy 这种"跑都跑不起来"的）缺了才算不 ok。"""
        return all(c.ok for c in self.checks if c.critical)

    @property
    def missing(self) -> list:
        return [c for c in self.checks if not c.ok]

    def describe(self) -> str:
        L = ["=" * 64, "a9route 训练环境自检", "=" * 64]
        for c in self.checks:
            L.append("  [{0}] {1:<22} {2}".format("OK  " if c.ok else "缺  ", c.name, c.detail))
            if not c.ok and c.hint:
                for line in c.hint.splitlines():
                    L.append("        → " + line)
        if self.notes:
            L.append("")
            L.extend("  " + n for n in self.notes)
        return "\n".join(L)


def _mod(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def check_environment(*, train_hint: str = "") -> DoctorReport:
    """查当前 Python 环境能干什么。"""
    rep = DoctorReport()
    rep.add("python", True, "{0}（{1}）".format(sys.version.split()[0], sys.executable))
    for name, why in (("cv2", "读视频/存帧"), ("numpy", "数值"),
                      ("yaml", "写 data.yaml")):
        m = _mod(name)
        rep.add(name, m is not None, getattr(m, "__version__", "") if m else why,
                hint="python -m pip install {0}".format(
                    {"cv2": "opencv-python", "yaml": "pyyaml"}.get(name, name)),
                # 这三个缺了**什么都干不了**（连数据集都建不出来）—— 算关键项
                critical=True)

    torch = _mod("torch")
    if torch is None:
        rep.add("torch", False, "没装（训练需要；**推理不需要**）",
                hint=train_hint or (
                    "训练要在带 CUDA 的 torch 环境里跑，见 TRAINING.md\n"
                    "本机 `ai_joy` 已有 torch 2.8.0+cu128（正好支持 RTX 5070 Ti）"))
        rep.add("CUDA", False, "不知道（没有 torch）",
                hint="没有 torch 就没有 CUDA 这一项；"
                     "只想推理的话不需要 CUDA（onnxruntime 用 CPU 也够）")
    else:
        rep.add("torch", True, torch.__version__)
        try:
            avail = bool(torch.cuda.is_available())
            detail = "不可用"
            if avail:
                detail = "{0}（cap {1}）".format(
                    torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
            rep.add("CUDA", avail, detail,
                    hint="没 CUDA 也能训，就是慢（加 --device cpu）")
        except Exception as exc:
            rep.add("CUDA", False, str(exc))

    ul = _mod("ultralytics")
    rep.add("ultralytics", ul is not None,
            getattr(ul, "__version__", "") if ul else "没装（训练/导出要用）",
            hint="python -m pip install -i "
                 "https://pypi.tuna.tsinghua.edu.cn/simple ultralytics")

    ort = _mod("onnxruntime")
    rep.add("onnxruntime", ort is not None,
            (getattr(ort, "__version__", "") + "  " +
             ",".join(ort.get_available_providers())) if ort
            else "没装（**运行时用 ONNX 推理需要它**，约 15 MB）",
            hint="python -m pip install onnxruntime")
    if ort is None and torch is not None:
        rep.notes.append("这个环境有 torch 但没 onnxruntime —— 也可以直接"
                         " --set vision__key_backend=ultralytics 先用起来")
    return rep


def check_assets(*, name: str = "keys") -> DoctorReport:
    """查权重 / 数据集 / 素材。"""
    rep = DoctorReport()
    paths.ensure_model_dirs()
    hit = find_weight(DEFAULT_WEIGHT)
    rep.add("预训练权重", hit is not None,
            str(hit) if hit else "models/weights/ 里没有 {0}".format(DEFAULT_WEIGHT),
            hint="a9route train fetch          # 用 urllib 下（**不是 curl**，"
                 "本机 curl 的 schannel 是坏的）")
    for nm in ("keys.onnx", "keys.pt"):
        p = paths.MODELS_DIR / nm
        rep.add(nm, p.is_file(),
                "{0}（{1:.1f} MB）".format(p, p.stat().st_size / 1e6) if p.is_file()
                else "还没有（训完/导出后才有）")
    # 训练产物在 runs/<name>/weights/best.pt —— 别因为"根目录没有 keys.pt"
    # 就说"还没训过"（第一版就是这么误报的）
    runs = paths.MODELS_DIR / "runs"
    bests = sorted(runs.glob("*/weights/best.pt")) if runs.is_dir() else []
    rep.add("训练产物 best.pt", bool(bests),
            "，".join("{0}（{1:.1f} MB）".format(p.parent.parent.name,
                                                p.stat().st_size / 1e6)
                      for p in bests[:3]) if bests else "还没训过",
            hint="& $tr -m a9route train run --name keys   # 训练环境")

    from a9route.train.dataset import DatasetError, dataset_dir, load_meta, stats
    ds = dataset_dir(name)
    if not ds.is_dir():
        rep.add("数据集", False, str(ds),
                hint="a9route train build <录像.mp4> --name {0}".format(name))
    else:
        try:
            info = stats(name=name)
            rep.add("数据集", info.images > 0,
                    "{0} 帧 / {1} 框 / 背景 {2} / 来源 {3} 段".format(
                        info.images, info.positives, info.background,
                        len(info.sources)))
            meta = load_meta(ds)
            reviewed = meta.get("reviewed_frames", 0)
            rep.add("人工复核", reviewed > 0,
                    "已复核 {0} 帧".format(reviewed),
                    hint="a9route train review --name {0}   # 生成复核页，"
                         "改完 apply 回去".format(name))
            for c in CLASSES:
                rep.add("框·" + CLASS_ZH.get(c, c), info.boxes_by_class.get(c, 0) > 0,
                        "{0} 个".format(info.boxes_by_class.get(c, 0)))
        except DatasetError as exc:
            rep.add("数据集", False, str(exc))

    videos = []
    for d in paths.video_dirs():
        if d.is_dir():
            videos += [p for p in d.iterdir()
                       if p.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov", ".webm")]
    rep.add("录像素材", bool(videos),
            "，".join(p.name for p in videos[:4]) if videos
            else "output/ 和 A9ROUTE_VIDEO_DIR 里都没有录像",
            hint="把跑图录像放进 output/，或 $env:A9ROUTE_VIDEO_DIR = \"D:\\录像\"")
    rep.notes.append("数据集: {0}".format(paths.DATASETS_DIR))
    rep.notes.append("模型:   {0}".format(paths.MODELS_DIR))
    rep.notes.append("限流说明：本机 github 走的是本地代理（慢但能通）；"
                     "curl / Invoke-WebRequest 的 schannel 是坏的，"
                     "所以下载一律走 Python urllib。")
    return rep


def check_other_envs() -> DoctorReport:
    """看一眼**别的** conda 环境里有没有 torch（训练环境通常不是当前这个）。"""
    rep = DoctorReport()
    extra = os.environ.get("A9ROUTE_TRAIN_PYTHON")
    cands = ([extra] if extra else []) + KNOWN_TRAIN_PYTHONS
    for p in cands:
        if not p or not Path(p).is_file():
            continue
        name = Path(p).parent.name
        sp = Path(p).parent / "Lib" / "site-packages"
        torch = (sp / "torch").is_dir()
        ul = any(sp.glob("ultralytics-*.dist-info"))
        ort = any(sp.glob("onnxruntime*.dist-info"))
        tags = []
        if torch:
            tags.append("torch")
        if ul:
            tags.append("ultralytics")
        if ort:
            tags.append("onnxruntime")
        rep.add(name, torch or ul or ort,
                "、".join(tags) if tags else "（都没装）",
                hint="" if (torch and ul) else
                     "{0} -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
                     "ultralytics".format(p))
    if not rep.checks:
        rep.notes.append("没找到已知的 conda 环境 —— 用 A9ROUTE_TRAIN_PYTHON 指定训练用的 python")
    return rep


def run(name: str = "keys") -> DoctorReport:
    """跑全部自检并合并成一份报告。"""
    rep = check_environment()
    other = check_other_envs()
    rep.checks.extend(other.checks)
    rep.notes.extend(other.notes)
    assets = check_assets(name=name)
    rep.checks.extend(assets.checks)
    rep.notes.extend(assets.notes)
    # 一句话结论
    have_torch = any(c.name == "torch" and c.ok for c in rep.checks)
    have_ul = any(c.name == "ultralytics" and c.ok for c in rep.checks)
    if have_torch and have_ul:
        rep.notes.append("结论：**这个环境能训练**（torch + ultralytics 都有）。")
    else:
        rep.notes.append("结论：**当前环境只能做数据集/复核/推理** —— "
                         "训练请用有 torch+ultralytics 的那个环境"
                         "（a9route train run 会检查并给出提示）。")
    return rep
