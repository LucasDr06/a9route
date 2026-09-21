# -*- coding: utf-8 -*-
"""train.runner —— **训练 / 评估 / 导出 ONNX**（真正碰 ultralytics 的地方）。

只有这个模块 import torch/ultralytics，而且都是**函数内 import** ——
这样 `a9route` 的运行时（`alphash9auto` 环境）永远不需要装 torch 也能跑，
`a9route train --help`、数据集构建、复核页也都不受影响。

## 三件必须知道的事（都在本机实测过）

1. **ultralytics 自己下权重会失败**。它下载权重是**shell 调 curl**，
   而本机 curl 的 schannel 挂了（`SEC_E_NO_CREDENTIALS`），于是报
   `Download failure ... Curl return value 35`。
   → `ensure_weights()` 用 **Python 自己的 urllib**（OpenSSL，实测能通）下载。
2. **权重放 `models/weights/`**。放好了 `train run` 直接找得到，
   完全不联网也能训。
3. **`YOLO_CONFIG_DIR` 指到项目内**。ultralytics 默认往 `~/.config/Ultralytics`
   写 settings/字体缓存，受限目录下会炸；这里统一改到 `models/.ultralytics/`。
"""
from __future__ import annotations

import json
import os
import shutil
import ssl
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from a9route import paths
from a9route.train.labels import CLASSES, CLASS_ZH, CLASS_IDS

#: 预训练权重的下载地址。**本机实测**：official 能通（走本地代理，约 9s），
#: ghfast 更快，modelscope 最快 —— 按顺序试，第一个成功的就用。
WEIGHT_URLS: dict[str, list[str]] = {
    "yolov8n.pt": [
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
        "https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
        "https://modelscope.cn/models/Ultralytics/YOLOv8/resolve/master/yolov8n.pt",
    ],
    "yolov8s.pt": [
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt",
        "https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt",
    ],
}

DEFAULT_WEIGHT = "yolov8n.pt"


def prepare_env() -> None:
    """把 ultralytics/matplotlib 的缓存目录挪到项目内（**必须在 import ultralytics 之前调**）。

    ⚠️ 这一步在**受限沙箱 / 只读 HOME** 里是**必需**的，不是"优化"：

    * ultralytics 往 `~/.config/Ultralytics` 写 settings —— 写不进去直接抛
      `PermissionError`；
    * matplotlib（ultralytics 画训练曲线要用）往 `~/.matplotlib` 写字体缓存 ——
      写不进去会退到系统临时目录，那里**同样**可能只读，于是训练在
      "Fast image access" 之后直接 `训练失败：PermissionError: [WinError 5]`。

    实测过：不改这两个环境变量，训练在受限环境下**跑不起来**。
    这和 NOTES §9 里「OCR 缓存默认在 ~/.paddlex，项目里优先用 worktmp/paddlex」
    是同一类问题、同一个解法。
    """
    d = paths.MODELS_DIR / ".ultralytics"
    d.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(d))
    mpl = paths.WORKTMP_DIR / "matplotlib"
    mpl.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl))
    # ultralytics 启动会去查更新/下载字体，离线或慢网时会把训练卡在开头几秒
    os.environ.setdefault("YOLO_OFFLINE", "1")


def _require_ultralytics():
    prepare_env()
    try:
        from ultralytics import YOLO
    except ImportError as exc:                            # pragma: no cover
        raise RuntimeError(
            "这个 Python 环境里没有 ultralytics/torch —— 训练要在**带 torch 的环境**里跑。\n"
            "  本机情况：`ai_joy` 环境有 torch 2.8.0+cu128（正好支持 RTX 5070 Ti）\n"
            "    C:\\Users\\Admin\\miniconda3\\envs\\ai_joy\\python.exe -m pip install "
            "-i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics\n"
            "  详见 TRAINING.md"
        ) from exc
    _pin_output_dirs()
    return YOLO


def _pin_output_dirs() -> None:
    """把 ultralytics 的输出目录钉在项目内。

    ⚠️ 不钉的话它会**相对 CWD** 建 `runs/` 和 `Ultralytics/` ——
    实测跑完 `train run` + `train eval` 之后项目根目录冒出：

        runs/detect/val/BoxF1_curve.png …
        Ultralytics/settings.json

    这些是"跑一次就多一堆、还找不着是谁建的"的典型垃圾。
    （`prepare_env()` 只把 *settings* 指到项目内，**产物目录是另几个设置项**。）
    """
    try:
        from ultralytics import settings as ul_settings
        ul_settings.update({
            "runs_dir": str(paths.MODELS_DIR / "runs"),
            "datasets_dir": str(paths.DATASETS_DIR),
            "weights_dir": str(paths.WEIGHTS_DIR),
        })
    except Exception:
        pass


def find_weight(name: str = DEFAULT_WEIGHT) -> Path | None:
    """在 `models/weights/`、项目根、CWD 里找一个权重文件。"""
    cands = [paths.WEIGHTS_DIR / name, paths.ROOT / name, Path.cwd() / name]
    for c in cands:
        if c.is_file() and c.stat().st_size > 100_000:
            return c
    return None


def ensure_weights(name: str = DEFAULT_WEIGHT, *, progress=print,
                   allow_download: bool = True) -> Path:
    """拿到预训练权重；本地没有就用 **urllib**（不是 curl）下到 `models/weights/`。

    下不动也不致命 —— 返回的路径不存在时，`train()` 会自动退回
    **从零训练**（`yolov8n.yaml`，ultralytics 包内置，不需要联网）。
    从零训在这个任务上完全可行（只有 2 类、位置固定），只是收敛慢一点。
    """
    hit = find_weight(name)
    if hit:
        return hit
    if not allow_download:
        return paths.WEIGHTS_DIR / name
    urls = WEIGHT_URLS.get(name, [])
    if not urls:
        return paths.WEIGHTS_DIR / name
    paths.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.WEIGHTS_DIR / name
    ctx = ssl.create_default_context()
    for url in urls:
        try:
            progress(f"  下载 {name} <- {url.split('/')[2]} …")
            t0 = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": "a9route-train"})
            with urllib.request.urlopen(req, timeout=300, context=ctx) as resp:
                tmp = out.with_suffix(out.suffix + ".part")
                with open(tmp, "wb") as fh:
                    shutil.copyfileobj(resp, fh, 1 << 20)
            if tmp.stat().st_size < 100_000:
                tmp.unlink(missing_ok=True)
                raise RuntimeError("下下来的文件太小，多半是错误页")
            shutil.move(str(tmp), str(out))
            progress(f"  OK {out.name} {out.stat().st_size / 1e6:.1f} MB "
                     f"（{time.time() - t0:.1f}s）")
            return out
        except Exception as exc:
            progress(f"  失败（{type(exc).__name__}: {str(exc)[:90]}）")
    progress("  ⚠️ 所有镜像都下不动 —— 会退回**从零训练**（yolov8n.yaml）")
    return out


# ---------------------------------------------------------------- 训练
@dataclass
class TrainReport:
    weights: str = ""
    from_scratch: bool = False
    run_dir: str = ""
    best: str = ""
    epochs: int = 0
    imgsz: int = 0
    seconds: float = 0.0
    metrics: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"训练完成（{self.seconds / 60:.1f} 分钟）",
                 f"  起点权重：{self.weights}"
                 + ("   ⚠️ 从零训练（没有预训练权重）" if self.from_scratch else ""),
                 f"  产物：{self.run_dir}",
                 f"  最好权重：{self.best}"]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
        lines.extend("  " + n for n in self.notes)
        return "\n".join(lines)


def train(*, data: str | Path, weight: str = DEFAULT_WEIGHT, epochs: int = 100,
          imgsz: int = 640, batch: int = 16, device: str | int | None = None,
          run_name: str = "keys", project: str | Path | None = None,
          workers: int = 4, patience: int = 30, lr0: float | None = None,
          resume: bool = False, seed: int = 0, progress=print,
          **extra) -> TrainReport:
    """跑训练的**唯一入口**（CLI 和 Python 都走这里）。

    `weight` 可以是：`models/weights/yolov8n.pt` 里的名字、一个本地 `.pt` 路径、
    或者 `yolov8n.yaml`（= **从零训练**）。
    """
    YOLO = _require_ultralytics()
    t0 = time.perf_counter()
    data = Path(data).resolve()
    if not data.is_file():
        raise RuntimeError(f"找不到 data.yaml：{data}（先 `a9route train split`）")
    rep = TrainReport(epochs=int(epochs), imgsz=int(imgsz))

    w = Path(weight)
    if w.suffix == ".yaml":
        rep.weights, rep.from_scratch = str(w), True
    else:
        found = w if w.is_file() else find_weight(w.name)
        if found is None:
            found = ensure_weights(w.name, progress=progress)
        if found.is_file():
            rep.weights = str(found)
        else:
            # 下载失败 -> 从零训练，并且**明说**（别让人以为在微调预训练权重）
            rep.weights = "yolov8n.yaml"
            rep.from_scratch = True
            rep.notes.append("没有预训练权重 -> 从零训练（yolov8.yaml）；"
                             "把 yolov8n.pt 放到 models/weights/ 后会好很多")
    project = Path(project) if project else (paths.MODELS_DIR / "runs")
    project.mkdir(parents=True, exist_ok=True)

    model = YOLO(rep.weights)
    kwargs = dict(data=str(data), epochs=int(epochs), imgsz=int(imgsz),
                  batch=int(batch), project=str(project), name=run_name,
                  workers=int(workers), patience=int(patience), seed=int(seed),
                  exist_ok=True, plots=True, val=True, resume=bool(resume),
                  **extra)
    if device is not None:
        kwargs["device"] = device
    if lr0 is not None:
        kwargs["lr0"] = float(lr0)
    progress(f"开始训练：{rep.weights} -> {project / run_name}"
             f"  ({epochs} epochs, imgsz={imgsz}, batch={batch})")
    model.train(**kwargs)

    rep.run_dir = str(project / run_name)
    best = Path(rep.run_dir) / "weights" / "best.pt"
    rep.best = str(best) if best.is_file() else ""
    rep.seconds = time.perf_counter() - t0
    # 训练完顺手在 val 上验一次，把指标抄进报告（省得再去翻 results.csv）
    if best.is_file():
        try:
            m = validate_weights(best, data=data, imgsz=imgsz, device=device)
            rep.metrics = m
        except Exception as exc:
            rep.notes.append(f"训练后验证失败（不影响已训好的权重）：{exc}")
    return rep


# ---------------------------------------------------------------- 评估
def validate_weights(weights: str | Path, *, data: str | Path,
                     imgsz: int = 640, device=None, split: str = "val",
                     conf: float | None = None) -> dict:
    """ultralytics 官方的 mAP 评估（**它衡量的是"框画得准不准"**）。

    ⚠️ 这个数字**不代表路线质量**：两个按键位置固定、框永远一样大，
    mAP 很容易 0.99，但"某一帧到底按没按"照样可能错。
    真正要看的指标在 `evaluate_frames()`（逐帧 P/R 和误报数）。
    """
    YOLO = _require_ultralytics()
    model = YOLO(str(weights))
    kw = dict(data=str(data), imgsz=int(imgsz), split=split, verbose=False,
              # 显式给定，别让它落到相对 CWD 的 `runs/detect/val`
              project=str(paths.MODELS_DIR / "runs"), name="_val", exist_ok=True)
    if device is not None:
        kw["device"] = device
    if conf is not None:
        kw["conf"] = float(conf)
    metrics = model.val(**kw)
    out: dict[str, float] = {}
    box = getattr(metrics, "box", None)
    if box is not None:
        for k in ("map", "map50", "map75", "mp", "mr"):
            v = getattr(box, k, None)
            if v is not None:
                out[f"box_{k}"] = float(v)
    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        out["ms_per_image"] = float(speed.get("inference", 0.0))
    return out


def evaluate_frames(weights: str | Path, *, dataset: str | Path,
                    split: str = "val", imgsz: int = 640, conf: float = 0.40,
                    device=None, limit: int | None = None,
                    compare_heuristic: bool = True, classes=None,
                    compare_choice: bool = False) -> dict:
    """**逐帧**评估：模型 vs 人工复核过的标注（**这才是路线质量相关的指标**）。

    指标：
      * 每类 `precision / recall / f1`（按"这一帧这个键有没有按下"算，不是 mAP）；
      * `fp_per_1k_bg`：每 1000 张背景帧里误报了几个 —— **"氮气偏多"就是这个数**；
      * `edge_acc`：标注发生翻转的那些帧上的正确率（最难的那批）；
      * `heuristic_*`：同一批帧上启发式的同款指标（**对照组**，模型必须比它好）。

    为什么不用 mAP 当主指标：路线脚本只关心两个布尔序列，
    框的位置固定且已知 —— 定位精度没有任何价值，**假阳/假阴才有**。

    ## 权重给 `.onnx` 就在**运行环境**里评估（不需要 torch）

    `weights` 是 `.onnx` 时走 `vision.keys.OnnxKeys`（只要 onnxruntime），
    所以"训练在 `ai_joy`、评估+部署在 `alphash9auto`"这条路是通的 ——
    而且**评估的就是运行时真正加载的那个文件**，不是另一个 .pt，
    免得出现"评估的和部署的不是同一个模型"。
    """
    import cv2

    from a9route.train.labels import read_label

    wpath = Path(weights)
    if not wpath.is_file():
        raise RuntimeError(f"找不到权重：{wpath}")
    ds = Path(dataset)
    img_dir = ds / "images" / split
    if not img_dir.is_dir():
        raise RuntimeError(f"没有 {img_dir}（先 `a9route train split`）")
    imgs = sorted([p for p in img_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if limit:
        imgs = imgs[:int(limit)]
    if not imgs:
        raise RuntimeError(f"{img_dir} 里没有图片")

    names = tuple(classes) if classes else CLASSES
    n_cls = len(names)

    predict = _make_predictor(wpath, imgsz=imgsz, conf=conf, device=device)
    # ⚠️ 对照组必须给**两套口径**，否则会得出错误结论：
    #   * 标注是 `cues.key_pressed()` 打的（预标注走的就是它）——
    #     要和它比就得用 mode="cues"，这才是**公平对照**；
    #   * 但**线上真正在跑**的是 `scan_fine` 的 mode="fine"（刹车只看绝对白度、
    #     氮气取 max(红, 圈内外)）—— 这是**实际部署**的那套。
    # 只给一套就会出现"拿 A 口径去比用 B 口径打的标注"：看着像模型大胜，
    # 其实测出来的是**两套启发式之间的差**，不是"模型比启发式强多少"。
    modes = ["cues", "fine"] if compare_heuristic else []
    dets = {}
    if modes:
        from a9route.vision.keys import HeuristicKeys
        dets = {m: HeuristicKeys(m) for m in modes}

    # 每帧的复核状态 —— 后面要拿它把"人工答案"那部分单独统计出来
    meta_by_file: dict = {}
    try:
        mp = ds / "meta.jsonl"
        if mp.is_file():
            for line in mp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                if m.get("file"):
                    meta_by_file[m["file"]] = m
    except Exception:
        meta_by_file = {}

    def _blank():
        return {k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for k in names}

    stat = _blank()
    hstat = {m: _blank() for m in modes}
    #: **只在人工复核过的帧上**统计（见下面 `reviewed_only` 的说明）
    rstat = _blank()
    rhstat = {m: _blank() for m in modes}
    n_reviewed = 0
    bg = 0
    edge_tot = edge_ok = 0
    prev_truth: dict[str, bool] = {}
    n = 0
    for img in imgs:
        frame = cv2.imread(str(img))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        truth = {k: False for k in names}
        lbl = img.parent.parent.parent / "labels" / split / f"{img.stem}.txt"
        if not lbl.is_file():       # pool 布局（images/ 与 labels/ 同级）兜底
            lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
        for b in read_label(lbl, w, h):
            truth[b.name] = True
        pred = predict(frame)
        hpred = {}
        for m, det in dets.items():
            r = det.read(frame)
            hpred[m] = {"brake_pressed": r.brake, "nitro_pressed": r.nitro}
        is_bg = not any(truth.values())
        bg += 1 if is_bg else 0
        is_reviewed = (meta_by_file.get(img.name) or {}).get("status") == "reviewed"
        if is_reviewed:
            n_reviewed += 1
        for k in names:
            _tally(stat[k], pred[k], truth[k])
            if is_reviewed:
                _tally(rstat[k], pred[k], truth[k])
            for m in modes:
                _tally(hstat[m][k], hpred[m][k], truth[k])
                if is_reviewed:
                    _tally(rhstat[m][k], hpred[m][k], truth[k])
            flipped = (k in prev_truth) and (prev_truth[k] != truth[k])
            if flipped:
                edge_tot += 1
                edge_ok += 1 if pred[k] == truth[k] else 0
            prev_truth[k] = truth[k]
        n += 1

    out = {"frames": n, "images_dir": str(img_dir), "conf": float(conf),
           "imgsz": int(imgsz), "weights": str(weights),
           "backend": "onnx" if wpath.suffix.lower() == ".onnx" else "ultralytics"}
    for k in names:
        st = stat[k]
        out[k] = {**_prf(st), "tp": st["tp"], "fp": st["fp"], "fn": st["fn"],
                  "tn": st["tn"], "zh": CLASS_ZH.get(k, "")}
        out[k]["fp_per_1k_bg"] = round(1000.0 * st["fp"] / bg, 2) if bg else 0.0
    out["edge_acc"] = round(edge_ok / edge_tot, 4) if edge_tot else None
    out["edge_frames"] = edge_tot
    # ---- 标注到底是不是"真值"？这决定了报告该怎么读 ----
    # 预标注是**启发式打的**。没经过人工复核的话，`cues` 那套对照就是
    # **自己跟自己比**（F1 必然 1.000），"模型没有更好"这句话毫无信息量。
    #
    # ⚠️ **必须按"正在评的这个 split"来数，不能数整个数据集** ——
    # 这条踩过：用户把修正全做在 `train` 里、`val` 一帧没复核；
    # 若按数据集总数判断，就会亮起绿灯给出"模型更好"的结论，
    # 而那个 val 的答案其实还是启发式自己打的 ✗✗。
    # **报告必须描述它真正评的那批帧。**
    out["reviewed_frames"] = n_reviewed
    out["eval_split"] = split
    out["labels_are_truth"] = n_reviewed > 0

    # ---- **只在人工复核过的帧上**再算一遍（这才是唯一"答案可信"的子集）----
    # 另外那 (总数 - 已复核) 帧的"答案"是启发式自己打的 —— 拿它评分，
    # 启发式必然满分，模型必然"没赢"。所以结论应当**只看这一块**。
    def _block(stats: dict) -> dict:
        b = {}
        for k in names:
            st = stats[k]
            b[k] = {**_prf(st), "tp": st["tp"], "fp": st["fp"], "fn": st["fn"],
                    "tn": st["tn"]}
        return b

    if n_reviewed:
        out["reviewed_only"] = {
            "frames": n_reviewed,
            "model": _block(rstat),
            "heuristics": {m: _block(rhstat[m]) for m in modes},
        }
    if modes:
        out["heuristics"] = {}
        for m in modes:
            block = {}
            for k in names:
                st = hstat[m][k]
                block[k] = {**_prf(st), "tp": st["tp"], "fp": st["fp"], "fn": st["fn"]}
                block[k]["fp_per_1k_bg"] = (
                    round(1000.0 * st["fp"] / bg, 2) if bg else 0.0)
            out["heuristics"][m] = block
        # `heuristic` 保留成"和标注同口径"那一套（老的报告读法仍然能用）
        out["heuristic"] = out["heuristics"].get("cues", out["heuristics"][modes[0]])
    return out


def evaluate_choice(weights: str | Path, *, dataset: str | Path,
                    split: str = "val", imgsz: int = 640, conf: float = 0.40,
                    limit: int | None = None, compare_heuristic: bool = True) -> dict:
    """**选路**的评估：按**三个答案**算，而不是按框的 mAP。

    选路真正在乎的是三个答案对不对（用户就是这么标的）：

      * **是否有选路** —— 二分类；
      * **有几个选项** —— 2/3/4；
      * **选第几个** —— 1..N。

    框画得多准其实无所谓（路标是圆、位置每帧都不同），**答案对不对才有用** ——
    路线脚本里写的是 `31`/`42` 这种"几个选第几"，不是框坐标。

    所以这里对每一帧把模型检出**按运行时同一套规则**折成三个答案
    （`vision.choice._icons_to_choice`），再和标注反推的答案比；
    同时把**启发式（HoughCircles）**当对照组跑一遍 —— 模型要赢的是它。
    """
    import cv2

    from a9route.train import choice as C
    from a9route.train.labels import read_label

    wpath = Path(weights)
    if not wpath.is_file():
        raise RuntimeError(f"找不到权重：{wpath}")
    ds = Path(dataset)
    img_dir = ds / "images" / split
    if not img_dir.is_dir():
        raise RuntimeError(f"没有 {img_dir}（先 `a9route train choice-build`）")
    imgs = sorted([p for p in img_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if limit:
        imgs = imgs[:int(limit)]
    if not imgs:
        raise RuntimeError(f"{img_dir} 里没有图片")

    if wpath.suffix.lower() == ".onnx":
        from a9route.vision.choice import OnnxChoice
        det = OnnxChoice(wpath, imgsz=imgsz, conf=conf)
    else:
        from a9route.vision.choice import UltralyticsChoice
        det = UltralyticsChoice(wpath, imgsz=imgsz, conf=conf)
    heur = None
    if compare_heuristic:
        from a9route.vision.choice import HeuristicChoice
        heur = HeuristicChoice()

    keys = ("has", "count", "selected")
    stat = {k: {"ok": 0, "bad": 0} for k in keys}
    hstat = {k: {"ok": 0, "bad": 0} for k in keys}
    count_conf: dict = {}
    n = n_pos = ok_all = h_ok_all = 0
    for img in imgs:
        frame = cv2.imread(str(img))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        lbl = img.parent.parent.parent / "labels" / split / f"{img.stem}.txt"
        if not lbl.is_file():
            lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
        truth = C.answers_from_boxes(
            read_label(lbl, w, h, classes=C.CHOICE_CLASSES))
        pred = det.read(frame)
        pred_a = C.Answers(pred.has, pred.count, pred.selected)
        hp = heur.read(frame) if heur is not None else None
        hp_a = C.Answers(hp.has, hp.count, hp.selected) if hp else None
        n += 1
        if truth.has:
            n_pos += 1
        all_ok = True
        h_all_ok = True
        for k in keys:
            tv = getattr(truth, k)
            # "没有选路"时 count/selected 都是 0，那几个数没意义 —— 只算 has
            if k != "has" and not truth.has:
                continue
            pv = getattr(pred_a, k)
            stat[k]["ok" if pv == tv else "bad"] += 1
            if k == "count":
                count_conf[(tv, pv)] = count_conf.get((tv, pv), 0) + 1
            if pv != tv:
                all_ok = False
            if hp_a is not None:
                hv = getattr(hp_a, k)
                hstat[k]["ok" if hv == tv else "bad"] += 1
                if hv != tv:
                    h_all_ok = False
        ok_all += 1 if all_ok else 0
        h_ok_all += 1 if (hp_a is not None and h_all_ok) else 0

    def block(st):
        out = {}
        for k in keys:
            tot = st[k]["ok"] + st[k]["bad"]
            out[k] = {"accuracy": round(st[k]["ok"] / tot, 4) if tot else None,
                      "ok": st[k]["ok"], "bad": st[k]["bad"], "n": tot}
        return out

    rep = {"frames": n, "with_choice": n_pos, "weights": str(weights),
           "backend": "onnx" if wpath.suffix.lower() == ".onnx" else "ultralytics",
           "imgsz": int(imgsz), "conf": float(conf), "split": split,
           "model": block(stat), "all_three_ok": round(ok_all / n, 4) if n else None,
           "count_confusion": {"{0}->{1}".format(a, b): c
                               for (a, b), c in sorted(count_conf.items())}}
    if compare_heuristic:
        rep["heuristic"] = block(hstat)
        #: ⚠️ 启发式的"三个答案同时都对"**也要算** —— 不然"模型到底赢没赢"
        #: 只能靠三个单项自己脑补（本项目的老规矩：模型没赢就别切，
        #: 那就必须把两边的**同一个数**摆在一起）
        rep["heuristic_all_three_ok"] = round(h_ok_all / n, 4) if n else None
    det.close()
    return rep


def format_choice_eval(rep: dict, *, baseline: bool = True) -> str:
    """把 `evaluate_choice()` 打成人能看的一段话。"""
    ZH = {"has": "是否有选路", "count": "有几个选项", "selected": "选第几个"}
    L = ["选路评估  split={0}  {1} 帧（其中 {2} 帧真有岔路口）".format(
        rep["split"], rep["frames"], rep["with_choice"]),
        "  权重 {0}  [{1}]".format(rep["weights"], rep["backend"]),
        "  {0:<14}{1:>12}{2:>10}".format(
            "三个答案", "模型正确率",
            "启发式正确率" if rep.get("heuristic") else "样本")]
    for k in ("has", "count", "selected"):
        m = rep["model"][k]
        row = "  {0:<14}{1:>12}".format(
            ZH[k], "—" if m["accuracy"] is None else "{0:.3f}".format(m["accuracy"]))
        if rep.get("heuristic"):
            h = rep["heuristic"][k]
            row += "{0:>10}".format(
                "—" if h["accuracy"] is None else "{0:.3f}".format(h["accuracy"]))
        else:
            row += "{0:>10}".format("")
        L.append(row + "{0:>8}".format(m["n"]))
    L.append("  三个答案**同时**都对：模型 {0}    启发式 {1}".format(
        rep.get("all_three_ok"),
        rep.get("heuristic_all_three_ok")
        if rep.get("heuristic_all_three_ok") is not None else "—"))
    if baseline and rep.get("heuristic"):
        better, worse = [], []
        for k in ("has", "count", "selected"):
            m, h = rep["model"][k]["accuracy"], rep["heuristic"][k]["accuracy"]
            if m is None or h is None:
                continue
            if m > h + 1e-6:
                better.append("{0} {1:.3f}->{2:.3f}".format(ZH[k], h, m))
            elif m < h - 1e-6:
                worse.append("{0} {1:.3f}->{2:.3f}".format(ZH[k], h, m))
        a, b = rep.get("all_three_ok"), rep.get("heuristic_all_three_ok")
        # 结论**以"三个答案同时都对"为准**：单项互有胜负时，那个合成数才是路线真正吃的东西
        if a is not None and b is not None:
            if a > b + 1e-6:
                L.append("  ✅ 三个答案同时都对：{0:.4f} > 启发式 {1:.4f}".format(a, b))
            elif a < b - 1e-6:
                L.append("  ⚠️ 三个答案同时都对：{0:.4f} **不如**启发式 {1:.4f}"
                         " —— 按本项目的规矩，这种模型**先别切**".format(a, b))
            else:
                L.append("  ⚠️ 三个答案同时都对：模型和启发式**打平**（{0:.4f}）".format(a))
        if better and not worse:
            L.append("  ✅ 模型更好：" + "；".join(better))
        elif better:
            L.append("  ⚠️ 有进步也有退步：" + "；".join(better + worse))
        elif worse:
            L.append("  ⚠️ 模型**没有比启发式更好**：" + "；".join(worse))
    if rep.get("count_confusion"):
        L.append("  「几个选项」的错法（真->预测）：" + "，".join(
            "{0}×{1}".format(k, v) for k, v in rep["count_confusion"].items()))
    return "\n".join(L)


def _make_predictor(weights: Path, *, imgsz: int, conf: float, device=None):
    """造一个 `frame -> {类别名: bool}` 的函数（`.pt` 走 ultralytics，`.onnx` 走 onnxruntime）。

    两条件事都返回**同一套键名**（`CLASSES`），所以下游统计完全一样。
    """
    if weights.suffix.lower() == ".onnx":
        from a9route.vision.keys import OnnxKeys
        det = OnnxKeys(weights, imgsz=imgsz, conf=conf)

        def predict(frame):
            r = det.read(frame)
            return {"brake_pressed": r.brake, "nitro_pressed": r.nitro}
        return predict

    YOLO = _require_ultralytics()
    model = YOLO(str(weights))
    names = {int(k): str(v) for k, v in dict(getattr(model, "names", {}) or {}).items()}

    def predict(frame):
        res = model.predict(frame, imgsz=int(imgsz), conf=float(conf),
                            verbose=False, device=device)[0]
        out = {k: False for k in CLASSES}
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes):
            for cid, cf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                nm = names.get(int(cid), "")
                if nm in out and cf >= conf:
                    out[nm] = True
        return out
    return predict


def _tally(st: dict, pred: bool, truth: bool) -> None:
    if pred and truth:
        st["tp"] += 1
    elif pred and not truth:
        st["fp"] += 1
    elif truth and not pred:
        st["fn"] += 1
    else:
        st["tn"] += 1


def _prf(st: dict) -> dict:
    tp, fp, fn = st["tp"], st["fp"], st["fn"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def format_frame_eval(rep: dict, *, baseline: bool = True) -> str:
    """把 `evaluate_frames()` 的结果打成人能看的一段话。

    ## 分两块打，而且**结论只看第二块**

    * **第一块：全部帧**。其中没被人工复核过的帧，"答案"是**启发式自己打的** ——
      拿它评分，启发式必然满分、模型必然"没赢"，**没有信息量**。
    * **第二块：只有人工复核过的帧**。这里的答案是**人给的**，
      是唯一能回答"模型到底有没有比启发式强"的数据。

    所以：`✅/⚠️` 那句结论**只基于第二块**；第一块只用来对照总体的行为有没有跑偏。
    （这条是用户把修正做在 train、val 一帧没复核时暴露的：
    按数据集总数判断会亮起绿灯给出假的"模型更好"。）
    """
    heurs = rep.get("heuristics") or {}
    rv = rep.get("reviewed_only") or {}

    def table(title, model_block, heur_block, extra=""):
        L = [title]
        L.append(f"  {'类别':<16}{'精确率':>9}{'召回':>9}{'F1':>9}"
                 f"{'误报/千帧背景':>15}")
        for k in CLASSES:
            d = model_block.get(k) or {}
            if not d:
                continue
            L.append(f"  {k:<16}{d['precision']:>9.3f}{d['recall']:>9.3f}"
                     f"{d['f1']:>9.3f}{d.get('fp_per_1k_bg', 0.0):>15.2f}")
        labels = {"cues": "启发式 cues（和标注同口径）",
                  "fine": "启发式 fine（**线上真正在跑的**）"}
        for m in ("cues", "fine"):
            if not heur_block.get(m):
                continue
            L.append(f"  --- 对照：{labels.get(m, '启发式 ' + m)} ---")
            for k in CLASSES:
                d = heur_block[m].get(k) or {}
                if not d:
                    continue
                L.append(f"  {k:<16}{d['precision']:>9.3f}{d['recall']:>9.3f}"
                         f"{d['f1']:>9.3f}{d.get('fp_per_1k_bg', 0.0):>15.2f}")
        if extra:
            L.append(extra)
        return L

    L = [f"逐帧评估  split={rep.get('eval_split', 'val')}  "
         f"{rep['frames']} 帧  conf={rep['conf']}  imgsz={rep['imgsz']}",
         f"  权重 {rep['weights']}"
         + (f"  [{rep['backend']}]" if rep.get("backend") else "")]

    if rv:
        L += table(f"\n【一】**只有人工复核过的 {rv['frames']} 帧**"
                   f"（答案是**人给的** —— 结论只看这块）",
                   rv.get("model") or {}, rv.get("heuristics") or {})
        verdict = _verdict(rv.get("model") or {}, (rv.get("heuristics") or {}).get("cues")
                           or {}, total=rv["frames"])
        L.append(verdict)
    else:
        L.append("\n【一】没有任何帧被人工复核过 —— **下面这块的答案是启发式自己打的**，")
        L.append("     拿它评分等于自己考自己，模型必然「没赢」，结论没有信息量。")
        L.append("     先 `a9route train review` 复核一批（重点是 val！）再回来看。")

    L += table(f"\n【二】全部 {rep['frames']} 帧（含未复核的 —— 只作总体对照，不作结论）",
               rep, heurs,
               extra=(f"  翻转帧（最难的那批）正确率：{rep['edge_acc']:.3f}"
                      f"（{rep['edge_frames']} 帧）"
                      if rep.get("edge_acc") is not None else ""))
    return "\n".join(L)


def _verdict(model: dict, cues: dict, *, total: int) -> str:
    """拿**人工复核过的子集**比模型 vs cues，给出结论。"""
    if not model or not cues:
        return "  （没法比较）"
    better, worse = [], []
    for k in CLASSES:
        dm, dh = model.get(k) or {}, cues.get(k) or {}
        if not dm or not dh:
            continue
        if dm["f1"] > dh["f1"] + 1e-6:
            better.append(f"{CLASS_ZH.get(k, k)} F1 {dh['f1']:.3f}->{dm['f1']:.3f}")
        elif dm["f1"] < dh["f1"] - 1e-6:
            worse.append(f"{CLASS_ZH.get(k, k)} F1 {dh['f1']:.3f}->{dm['f1']:.3f}")
    if total < 50:
        return (f"  ⚠️ 只有 {total} 帧人工复核过 —— 样本太少，"
                f"再复核一些（val 上做到 100~200 帧）结论才稳")
    if better and not worse:
        return "  ✅ **在人给的答案上，模型比启发式更好**：" + "；".join(better)
    if better and worse:
        return "  ⚠️ 有进步也有退步：" + "；".join(better + worse)
    if worse:
        return ("  ⚠️ **模型没有比启发式更好** —— 先查这两件事："
                "① val 的标注是不是真的人给答案（不是就别下结论）；"
                "② 复核量够不够（每条类别的正样本最好 50+ 帧）")
    return "  · 两者打平"


# ---------------------------------------------------------------- 导出
def export_onnx(weights: str | Path, *, imgsz: int = 640, out: str | Path | None = None,
                opset: int = 12, half: bool = False, progress=print,
                name: str = "", notes: str = "", classes=None,
                dataset_name: str = "") -> Path:
    """导出 ONNX（**运行时不装 torch 的关键一步**），并拷到 `models/<name>.onnx`。

    同时写一份**模型清单** `<输出名>.json`（见 `train/modelcard.py`）——
    里面记着类别顺序、imgsz、训练时间、数据集与指标。
    运行时 `vision.keys` / `vision.choice` 会读它并**核对类序**：换模型时类序错了
    就直接报错，而不是"刹车和氮气反了还不报错"。

    `classes`：这套模型用哪套类别（默认按键那套；**选路传 `CHOICE_CLASSES`**）。
    """
    from a9route.train import modelcard as MC
    from a9route.train.labels import CLASS_ZH

    names = tuple(classes) if classes else CLASSES
    zh = dict(CLASS_ZH) if classes is None else dict(
        __import__("a9route.train.choice", fromlist=["CHOICE_ZH"]).CHOICE_ZH)
    YOLO = _require_ultralytics()
    src = Path(weights)
    if not src.is_file():
        raise RuntimeError(f"找不到权重：{src}")
    model = YOLO(str(src))
    progress(f"导出 ONNX（imgsz={imgsz}, opset={opset}）…")
    path = model.export(format="onnx", imgsz=int(imgsz), opset=int(opset),
                        half=bool(half), dynamic=False, simplify=False)
    produced = Path(path)
    if out:
        target = Path(out)
    else:
        target = paths.MODELS_DIR / "{0}.onnx".format(name or "keys")
    target.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != target.resolve():
        shutil.copy2(produced, target)
    progress(f"ONNX -> {target}（{target.stat().st_size / 1e6:.1f} MB）")

    # ---- 写清单（换模型/排查都靠它）----
    card = MC.ModelCard(
        loaded=True,            # 这份就是要写下去的，别让 describe() 说"没有清单"
        name=name or target.stem,
        onnx=target.name,
        source_weights=str(src),
        classes=[str(c) for c in names],
        nc=len(names),
        imgsz=int(imgsz),
        conf_default=0.40,
        iou_default=0.50,
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        dataset="", dataset_frames=0, dataset_reviewed=0,
        metrics={}, notes=notes,
    )
    # 数据集信息 / 指标：能查到就填进去（查不到不影响模型本身）
    try:
        card.dataset = card.dataset or target.parent.name
        # 训练产物同目录常有 runs/<name>/args.yaml 指到的数据集
        ds_dir = paths.DATASETS_DIR / target.stem
        if (ds_dir / "meta.jsonl").is_file():
            card.dataset = ds_dir.name
            card.dataset_frames = sum(1 for _ in (ds_dir / "meta.jsonl")
                                      .open(encoding="utf-8"))
            info = json.loads((ds_dir / "dataset.json").read_text(encoding="utf-8")
                              ) if (ds_dir / "dataset.json").is_file() else {}
            card.dataset_reviewed = int(info.get("reviewed_frames") or 0)
        run_csv = src.parent.parent / "results.csv"
        if run_csv.is_file():
            rows = [r for r in run_csv.read_text(encoding="utf-8").splitlines() if r.strip()]
            if len(rows) > 1:
                head = [h.strip() for h in rows[0].split(",")]
                best = None
                for line in rows[1:]:
                    vals = [v.strip() for v in line.split(",")]
                    d = dict(zip(head, vals))
                    try:
                        m = float(d.get("metrics/mAP50-95(B)") or 0)
                    except ValueError:
                        continue
                    if best is None or m > best[1]:
                        best = (d, m)
                if best:
                    d = best[0]
                    card.metrics = {
                        "box_map50": _safe_float(d.get("metrics/mAP50(B)")),
                        "box_map": _safe_float(d.get("metrics/mAP50-95(B)")),
                        "precision": _safe_float(d.get("metrics/precision(B)")),
                        "recall": _safe_float(d.get("metrics/recall(B)")),
                        "best_epoch": int(float(d.get("epoch") or 0)),
                    }
    except Exception as exc:                           # noqa: BLE001
        progress(f"  （清单里没写全数据集/指标信息：{type(exc).__name__}）")
    cardp = MC.write(target, card=card)
    progress(f"模型清单 -> {cardp}")
    progress(f"  {card.describe()}")
    return target


# ---------------------------------------------------------------- 整片脉冲对比
def _safe_float(v):
    """CSV 里可能缺列/空值 —— 转不动就记 0，不要让整份清单写不出来。"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def pulse_report(det, video: str | Path, *, gap: float = 0.12,
                 max_seconds: float | None = None, progress=None) -> dict:
    """整片跑一遍，把**按键脉冲段**列出来（`det` 可以是模型，也可以是启发式）。

    这是"路线会不会变"的直观检查：同一段录像，模型和启发式各出多少个
    漂移/氮气动作、时刻差多少。**它不是准确率**（视频里没有真值），
    但"氮气动作从 31 个降到 20 个"这种变化一眼就能看出方向对不对。
    """
    import cv2

    from a9route.vision.video import _episodes

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps
    limit = int(max_seconds * fps) if max_seconds else None
    samples: list[tuple[float, bool, bool]] = []
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if limit is not None and i >= limit:
                break
            r = det.read(frame)
            samples.append((round(i * dt, 3), r.brake, r.nitro))
            i += 1
            if progress and i % 900 == 0:
                progress(f"    {i} 帧（{i * dt:.0f}s）…")
    finally:
        cap.release()
    if not samples:
        raise RuntimeError("一帧都没读到")
    out = {"backend": getattr(det, "name", "?"), "frames": len(samples),
           "fps": round(fps, 4), "seconds": round(samples[-1][0], 2)}
    for idx, key in ((0, "brake"), (1, "nitro")):
        eps = _episodes(samples, idx, gap=gap)
        out[key] = {
            "episodes": len(eps),
            "total_s": round(sum(e[1] for e in eps), 3),
            "longest_s": round(max((e[1] for e in eps), default=0.0), 3),
            "pulses": sum(e[2] for e in eps),
            "list": [(round(e[0], 3), round(e[1], 3), e[2]) for e in eps],
        }
    return out


def format_pulse_report(model: dict, base: dict | None = None) -> str:
    """把 `pulse_report()` 的两个结果并排打出来。"""
    L = [f"整片按键脉冲（{model['backend']}，{model['frames']} 帧 / "
         f"{model['seconds']}s）"]
    head = f"  {'通道':<8}{'动作数':>8}{'按下总时长':>12}{'最长':>9}{'脉冲数':>9}"
    if base is not None:
        head += f"{'（启发式动作数）':>18}{'差':>7}"
    L.append(head)
    for key, zh in (("brake", "刹车"), ("nitro", "氮气")):
        d = model[key]
        line = (f"  {zh:<8}{d['episodes']:>8}{d['total_s']:>12.2f}"
                f"{d['longest_s']:>9.2f}{d['pulses']:>9}")
        if base is not None:
            b = base[key]
            line += f"{b['episodes']:>18}{d['episodes'] - b['episodes']:>+7}"
        L.append(line)
    return "\n".join(L)


def save_report(path: str | Path, payload: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")
    return p


def class_summary() -> str:
    return "、".join(f"{i}={n}({CLASS_ZH.get(n, '')})"
                     for n, i in CLASS_IDS.items())
