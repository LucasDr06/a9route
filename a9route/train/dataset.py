# -*- coding: utf-8 -*-
"""train.dataset —— 数据集目录的**布局、构建、划分、校验、统计**。

## 目录长什么样

```
datasets/keys/                     # 一个"数据集"（--name，默认 keys）
├── dataset.json                   # 元信息：源视频、抽帧参数、类别、时间
├── meta.jsonl                      # **每帧一行**：文件、帧号、时刻、抽样原因、信号明细
├── pool/                           # ① build 的产物（抽出来 + 预标注好，还没划分）
│   ├── images/*.jpg
│   └── labels/*.txt
├── images/{train,val}/             # ② split 把 pool 里的帧**搬**到这里
├── labels/{train,val}/
├── data.yaml                       # ultralytics 直接吃这个
└── review/                         # ③ review 的产物（复核页 + 清单）
```

`build` 和 `split` 分成两步是**故意的**：人要在 `pool/` 阶段复核标注，
复核完再划分 —— 否则改一版标注就得重新划一次 train/val，指标没法比较。

## ⚠️ 划分必须**按视频**分（这是最容易自欺欺人的一步）

相邻帧只差 33ms，画面几乎一模一样。如果**按帧随机**划分 train/val，
验证集里的每一帧在训练集里都有一个"几乎相同"的兄弟 —— val mAP 会漂亮得
不像话，而模型其实什么都没学会（典型的**数据泄漏**）。

所以：
* 有**多段视频** -> 整段整段地分（`by_video=True`，默认）；
* **只有一段视频** -> 按**连续时间块**分（每 5 块留 1 块），并在
  `data.yaml` 和报告里**明说"只有一段视频，val 偏乐观"**。
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from a9route import paths
from a9route.train import frames as F
from a9route.train.labels import (CLASSES, CLASS_ZH, Box, DatasetInfo,
                                  LabelError, read_label, write_data_yaml,
                                  write_label)

#: 数据集元信息文件名
META_FILE = "dataset.json"
#: 每帧一行的明细（复核页/统计/划分都读它）
FRAME_META = "meta.jsonl"
#: 抽帧 + 预标注的暂存区（划分前的 stage）
POOL_DIR = "pool"

#: 标注是不是要**按存下来的 JPEG 重新算一遍**（见 `build()` 里的说明）。
#: 留成开关是为了出问题时能一键回到"按视频帧标"的老行为做对照。
_RELABEL_FROM_JPEG = True


class DatasetError(RuntimeError):
    """数据集层面的错误（路径不对、没帧、划分不合法……）。"""


# ---------------------------------------------------------------- 路径
def dataset_dir(name: str = "keys") -> Path:
    """`datasets/<name>/`（可用 `A9ROUTE_DATASETS` 指到别的盘）。"""
    return paths.DATASETS_DIR / name


def _require(ds: Path) -> Path:
    if not ds.is_dir():
        raise DatasetError(f"数据集不存在：{ds}\n先用 `a9route train build <录像…>` 建一个")
    return ds


# ---------------------------------------------------------------- 元信息
def save_meta(ds: Path, info: dict) -> Path:
    p = Path(ds) / META_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
    return p


def load_meta(ds: Path) -> dict:
    p = Path(ds) / META_FILE
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def read_frames_meta(ds: Path) -> list[dict]:
    """读 `meta.jsonl`（每帧一行）。**坏行报错**（不静默跳过）。"""
    p = Path(ds) / FRAME_META
    if not p.is_file():
        return []
    out = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        try:
            out.append(json.loads(s))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{p.name}:{i} 不是合法 JSON：{exc}") from exc
    return out


# ---------------------------------------------------------------- ① build
@dataclass
class BuildReport:
    dataset: str = ""
    frames: int = 0
    by_reason: dict = field(default_factory=dict)
    by_class: dict = field(default_factory=dict)
    videos: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    bytes: int = 0
    #: 有多少帧的标注在"按存下来的 JPEG 重算"之后**结论变了**
    #: （JPEG 压缩把贴着阈值的帧推过了判定线；实测 1200 帧里 1 帧）
    jpeg_flips: int = 0

    def describe(self) -> str:
        lines = [f"数据集 {self.dataset}：抽了 {self.frames} 帧"
                 f"（{self.bytes / 1024 / 1024:.0f} MB）"]
        for r in F.REASONS:
            if self.by_reason.get(r):
                lines.append(f"  {r:<9} {self.by_reason[r]:>6}   {F.REASON_ZH.get(r, '')}")
        for c in CLASSES:
            lines.append(f"  {c:<16} {self.by_class.get(c, 0):>6} 个框")
        if self.jpeg_flips:
            lines.append(
                f"  ⚠️ 其中 {self.jpeg_flips} 帧的标注在按存下来的 JPEG 重算后**结论变了**"
                f"（JPEG 压缩把贴着阈值的帧推过了判定线，已按重算结果标注 ✓）")
        for s in self.skipped:
            lines.append(f"  ⚠️ 跳过：{s}")
        return "\n".join(lines)


def _prelabel_detector(cfgmod) -> tuple:
    """预标注用哪个后端：`(detector 或 None, 人话的来源说明)`。

    没给 detector = 走 `cues.key_pressed()` 像素启发式。

    ⚠️ 和运行时的口径**一致**：`auto` 时模型不在就**报错**（`resolve_backend` 抛），
    不会静默退回启发式 —— 否则"我明明用模型推的数据集"这句话就不成立了。
    """
    from pathlib import Path

    try:
        from a9route.vision import keys as K
    except Exception as exc:                          # noqa: BLE001
        return None, "heuristic（vision.keys 起不来：{0}）".format(exc)
    be, mp, _p = K.resolve_backend()
    if be == "heuristic":
        return None, "heuristic(cues.key_pressed) —— 配置里显式指定了启发式"
    det = K.build_key_detector()
    return det, "{0}:{1}".format(be, Path(mp).name)


def build(videos: list[str | Path], *, name: str = "keys",
          budget: int = 3000, per_video_budget: int | None = None,
          max_seconds: float | None = None, every: int = 1,
          edge_pad: int = 3, margin_band: float = 0.25, edge_share: float = 0.6,
          jpeg_quality: int = 92, append: bool = False,
          seed: int = 0, progress=None) -> BuildReport:
    """抽帧 + 启发式预标注 -> `datasets/<name>/pool/`。

    `budget` 是**总预算**；`per_video_budget` 不给时按视频段数均分
    （多段视频各自出一份，免得一段长录像把预算全吃掉）。
    """
    from a9route import config as cfgmod
    from a9route.train.prelabel import read_signals
    from a9route.vision import cues
    import cv2

    cfgmod.apply()                     # 判据框/阈值必须先灌好，下面才读得对
    # ---- 预标注用哪个判据：**默认跟运行时后端走**（`auto` -> 有模型就用模型）----
    # 用户 2026-09-15 的用法："用先前练的两个模型用这两个视频推一个新的训练集出来"。
    # 所以这里不再是"永远启发式"：配置说 auto/onnx 就用模型（模型不在会**报错**，
    # 和运行时一个口径 —— 不静默退回启发式），显式 `heuristic` 才走像素判据。
    # 来源会记进 `dataset.json` 和每帧的 `signals.by`，免得以后分不清
    # "这批标签是模型给的"还是"启发式给的"。
    pre_det, prelabel_by = _prelabel_detector(cfgmod)
    if progress:
        progress(f"预标注判据：{prelabel_by}")
    ds = dataset_dir(name)
    pool = ds / POOL_DIR
    img_dir = pool / "images"
    lbl_dir = pool / "labels"
    if not append and ds.exists():
        # **不删数据集**（里面可能有人工复核过的心血），只清 pool 重建
        for d in (img_dir, lbl_dir):
            if d.exists():
                shutil.rmtree(d)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    reps = [Path(v) for v in videos]
    missing = [p for p in reps if not p.is_file()]
    if missing:
        raise DatasetError("找不到录像：" + "，".join(str(p) for p in missing))
    if not reps:
        raise DatasetError("没给录像。用法：a9route train build <录像.mp4> [更多…]")

    per = int(per_video_budget) if per_video_budget else max(
        50, int(budget) // max(1, len(reps)))
    rep = BuildReport(dataset=str(ds))
    meta_lines: list[dict] = []
    prev = read_frames_meta(ds) if append else []
    if prev:
        meta_lines.extend(prev)

    for video in reps:
        prefix = F.slug(video.stem)
        if progress:
            progress(f"[{video.name}] 第一遍：逐帧算判据信号…")
        sigs, info = F.scan_signals(video, max_seconds=max_seconds, every=every,
                                   progress=progress, detector=pre_det)
        if not sigs:
            rep.skipped.append(f"{video.name}（解不出帧）")
            continue
        plan = F.plan_sampling(sigs, budget=per, edge_pad=edge_pad,
                               margin_band=margin_band, edge_share=edge_share,
                               seed=seed)
        if progress:
            progress(f"[{video.name}] 第二遍：存 {len(plan.picks)} 帧 / {plan.summary()}")
        saved = F.save_frames(video, plan.picks, img_dir, prefix=prefix,
                              jpeg_quality=jpeg_quality, progress=progress)

        n_box: dict[str, int] = {}
        for fsig, path in saved:
            # ⚠️ **用存下来的那张 JPEG 重新算一遍判据**，而不是直接沿用解码视频帧上的结果。
            #
            # 为什么：标注必须描述"训练**真正读到**的那张图"。JPEG q92 是有损的，
            # 实测抓到一帧（`3_001402.jpg`）：视频帧上 `brake_white=0.366`（> 0.35，
            # 判按下），存成 JPEG 再读回来变成 `0.338`（< 0.35，判没按）——
            # 那一帧离阈值只有 **4.6%**，压缩就把它推过去了。
            # 1200 帧里出现 1 帧（**0.08%**）。
            #
            # 不改的后果：数据集体检**永远**报"1 帧漂移"，而且模型在学一个
            # 和它实际看到的像素不一致的标签。改了之后体检是干净的。
            sig_used = fsig.sig
            flipped = False
            if _RELABEL_FROM_JPEG:
                again = cv2.imread(str(path))
                if again is not None:
                    sig_used = read_signals(again, detector=pre_det)
                    flipped = (sig_used.brake_hit != fsig.sig.brake_hit
                               or sig_used.nitro_hit != fsig.sig.nitro_hit)
            boxes = sig_used.boxes(cues.BRAKE_KEY_BOX, cues.NITRO_KEY_BOX)
            write_label(lbl_dir / f"{path.stem}.txt", boxes,
                        info["width"], info["height"])
            for b in boxes:
                n_box[b.name] = n_box.get(b.name, 0) + 1
            if flipped:
                rep.jpeg_flips += 1
            meta_lines.append({
                "file": path.name,
                "video": video.name,
                "video_slug": prefix,
                "frame": fsig.idx,
                "t": fsig.t,
                "reason": fsig.reason,
                "width": info["width"],
                "height": info["height"],
                "fps": round(info["fps"], 4),
                "boxes": [{"cls": b.cls_id, "name": b.name,
                           "xywh": [round(b.x, 1), round(b.y, 1),
                                    round(b.w, 1), round(b.h, 1)],
                           "conf": b.conf} for b in boxes],
                "signals": sig_used.as_dict(),
                #: 这一帧的标注是"按存下来的 JPEG 重算、且结论和视频帧上不同"的
                #: （极少；界面会单独标出来，免得人以为是标注坏了）
                "jpeg_flip": flipped,
                # 复核状态：`prelabel` = 机器标的还没人看过；复核后改成 `reviewed`
                "status": "prelabel",
            })
            rep.bytes += path.stat().st_size
        rep.frames += len(saved)
        rep.videos.append(f"{video.name}（{prefix}）")
        for r, n in plan.counts.items():
            rep.by_reason[r] = rep.by_reason.get(r, 0) + n
        for c, n in n_box.items():
            rep.by_class[c] = rep.by_class.get(c, 0) + n

    if not rep.frames:
        raise DatasetError("一帧都没抽出来 —— 检查录像能不能解码")

    (ds / FRAME_META).write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta_lines),
        encoding="utf-8")
    info = load_meta(ds)
    info.update({
        "name": name,
        "classes": list(CLASSES),
        "classes_zh": CLASS_ZH,
        "created": info.get("created") or time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "videos": sorted({m.get("video") for m in meta_lines if m.get("video")}),
        "sampling": {"budget_total": budget, "budget_per_video": per,
                     "edge_pad": edge_pad, "margin_band": margin_band,
                     "edge_share": edge_share,
                     "jpeg_quality": jpeg_quality, "every": every,
                     "max_seconds": max_seconds, "seed": seed},
        # ⚠️ 预标注来源必须记下来：`heuristic(cues.key_pressed)` 还是
        # `onnx:keys.onnx` —— 这决定了"这批标签是模型的判断"还是"像素判据的判断"，
        # 复核之后才都变成"人的判断"（体检/评估里要能分开看）
        "prelabel": prelabel_by,
        "frames": len(meta_lines),
    })
    save_meta(ds, info)
    return rep


# ---------------------------------------------------------------- ② split
@dataclass
class SplitReport:
    train: int = 0
    val: int = 0
    val_videos: list = field(default_factory=list)
    note: str = ""

    def describe(self) -> str:
        lines = [f"划分完成：train {self.train} 帧 / val {self.val} 帧"]
        if self.val_videos:
            lines.append("  val 来自：" + "，".join(self.val_videos))
        if self.note:
            lines.append("  ⚠️ " + self.note)
        return "\n".join(lines)


def plan_split(metas: list[dict], *, val_ratio: float = 0.2, seed: int = 0
               ) -> tuple[dict[str, str], list[str], str]:
    """决定每帧去 train 还是 val。返回 `({file: split}, val 的视频名, 提醒)`。

    **按视频分**（多段时）；只有一段时按连续时间块分。理由见模块文档。
    """
    if not metas:
        return {}, [], ""
    val_ratio = min(0.9, max(0.05, float(val_ratio)))
    videos = sorted({m.get("video") or "?" for m in metas})
    note = ""
    if len(videos) >= 2:
        # 按视频分：留出整段（取"帧多的排前面"的顺序，让 val 尽量有代表性）
        by_video: dict[str, int] = {}
        for m in metas:
            v = m.get("video") or "?"
            by_video[v] = by_video.get(v, 0) + 1
        order = sorted(videos, key=lambda v: -by_video.get(v, 0))
        total = len(metas)
        want = int(round(total * val_ratio))
        val_videos: list[str] = []
        got = 0
        # 从**帧最少**的那几段开始留给 val：大段留给训练更划算
        for v in reversed(order):
            if got >= want or len(val_videos) >= len(videos) - 1:
                break
            val_videos.append(v)
            got += by_video.get(v, 0)
        assigned = {m["file"]: ("val" if (m.get("video") or "?") in val_videos
                                else "train") for m in metas if m.get("file")}
        return assigned, val_videos, note

    # 只有一段视频：按**连续时间块**分（每 5 块留 1 块）
    note = ("只有一段源视频 —— val 与 train 来自同一段录像，"
            "指标会偏乐观；想拿可信指标请再补几段别的录像")
    ordered = sorted([m for m in metas if m.get("file")],
                     key=lambda m: m.get("frame", 0))
    blocks = 5
    assigned: dict[str, str] = {}
    for i, m in enumerate(ordered):
        blk = i * blocks // max(1, len(ordered))
        assigned[m["file"]] = "val" if blk == blocks - 1 else "train"
    return assigned, [metas[0].get("video") or "?"], note


def split(*, name: str = "keys", val_ratio: float = 0.2, seed: int = 0,
          progress=None) -> SplitReport:
    """把 `pool/` 按 `plan_split` 搬到 `images/{train,val}` + `labels/{train,val}`，
    然后写 `data.yaml`。**会先清空已有的 train/val**（pool 是唯一事实来源）。
    """
    ds = _require(dataset_dir(name))
    pool = ds / POOL_DIR
    metas = read_frames_meta(ds)
    if not metas:
        raise DatasetError(f"{ds} 里没有 {FRAME_META} —— 先 `a9route train build`")
    assigned, val_videos, note = plan_split(metas, val_ratio=val_ratio, seed=seed)

    for sub in ("images", "labels"):
        for sp in ("train", "val"):
            d = ds / sub / sp
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    rep = SplitReport(val_videos=val_videos, note=note)
    moved = 0
    for m in metas:
        fn = m.get("file")
        sp = assigned.get(fn)
        if not sp:
            continue
        src_img = pool / "images" / fn
        src_lbl = pool / "labels" / f"{Path(fn).stem}.txt"
        if not src_img.is_file():
            continue
        shutil.copy2(src_img, ds / "images" / sp / fn)
        # 标注文件可能不存在（= 空标注的背景帧），补一个空文件让 YOLO 认
        dst_lbl = ds / "labels" / sp / f"{Path(fn).stem}.txt"
        if src_lbl.is_file():
            shutil.copy2(src_lbl, dst_lbl)
        else:
            dst_lbl.write_text("", encoding="utf-8")
        setattr(rep, sp, getattr(rep, sp) + 1)
        moved += 1
        if progress and moved % 500 == 0:
            progress(f"    划分 {moved}/{len(metas)}…")
    if not moved:
        raise DatasetError("pool 里的帧一个都没找到（标注 meta 和图片对不上？）")

    # `pool/` 留着不删：复核要对着原图看；磁盘紧张时手工删掉即可
    write_data_yaml(ds / "data.yaml", ds)
    info = load_meta(ds)
    info.update({"split": {"val_ratio": val_ratio, "val_videos": val_videos,
                           "train": rep.train, "val": rep.val,
                           "note": note, "at": time.strftime("%Y-%m-%d %H:%M:%S")}})
    save_meta(ds, info)
    return rep


# ---------------------------------------------------------------- ③ 统计 / 校验
def iter_pairs(root: str | Path) -> list[tuple[Path, Path]]:
    """递归找出 `(图片, 标注文件)` 对。标注路径按 YOLO 约定推：
    `.../images/xxx.jpg` -> `.../labels/xxx.txt`（`pool/images` 一样适用）。
    """
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue
        parts = list(img.parts)
        if "images" not in parts:
            continue
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
        out.append((img, Path(*parts).with_suffix(".txt")))
    return out


def stats(name: str = "keys", *, root: str | Path | None = None,
          read_images: bool = False) -> DatasetInfo:
    """统计数据集（`DatasetInfo`）。默认读 `meta.jsonl` 拿画面尺寸，
    没有 meta 的"外来数据集"才去读图片头。"""
    ds = Path(root) if root else dataset_dir(name)
    if not ds.is_dir():
        raise DatasetError(f"数据集不存在：{ds}")
    info = DatasetInfo(root=ds)
    metas = read_frames_meta(ds) if not root else []
    size_of = {m.get("file"): (m.get("width"), m.get("height"))
               for m in metas if m.get("file")}
    src_of = {m.get("file"): (m.get("video") or "?") for m in metas if m.get("file")}

    pairs = iter_pairs(ds)
    # pool 和 train/val 可能同时存在 -> 同一个文件名只算一次（以 train/val 优先）
    seen: dict[str, tuple[Path, Path]] = {}
    for img, lbl in pairs:
        key = img.name
        cur = seen.get(key)
        if cur is None or "pool" in cur[0].parts and "pool" not in img.parts:
            seen[key] = (img, lbl)
    for img, lbl in seen.values():
        info.images += 1
        wh = size_of.get(img.name)
        boxes: list[Box] = []
        if lbl.is_file():
            info.labels += 1
            w, h = wh if (wh and wh[0]) else _img_size(img, read_images)
            if w and h:
                try:
                    boxes = read_label(lbl, w, h)
                except LabelError as exc:
                    info.problems.append(str(exc))
                    boxes = []
        w, h = wh if (wh and wh[0]) else _img_size(img, read_images)
        if w and h:
            info.sizes[f"{w}x{h}"] = info.sizes.get(f"{w}x{h}", 0) + 1
        if not boxes:
            info.background += 1
        if len(boxes) >= 2:
            info.multi += 1
        for b in boxes:
            info.boxes_by_class[b.name] = info.boxes_by_class.get(b.name, 0) + 1
            info.frames_by_class[b.name] = info.frames_by_class.get(b.name, 0) + 1
        v = src_of.get(img.name)
        if v:
            info.sources[v] = info.sources.get(v, 0) + 1
    if info.images and info.labels < info.images:
        info.problems.append(
            f"{info.images - info.labels} 张图没有对应标注文件"
            f"（背景帧的标注文件应当是**空文件**，不是不存在）")
    return info


def _img_size(img: Path, allow_read: bool) -> tuple[int, int]:
    """画面尺寸。`read_images=False` 时**不去解图**（几千张图很慢）。"""
    if not allow_read:
        return (0, 0)
    import cv2
    im = cv2.imread(str(img), cv2.IMREAD_REDUCED_COLOR_8)
    return (0, 0) if im is None else (im.shape[1] * 8, im.shape[0] * 8)


def verify(name: str = "keys", *, root: str | Path | None = None) -> list[str]:
    """训练前的体检：返回问题清单（空 = 没问题）。

    查的是**会让训练静默变差**的东西：图片/标注对不上、类别越界、
    坐标越界、val 空、类别严重不平衡。
    """
    ds = Path(root) if root else dataset_dir(name)
    problems: list[str] = []
    if not ds.is_dir():
        return [f"数据集不存在：{ds}"]
    for sub in ("train", "val"):
        d = ds / "images" / sub
        if not d.is_dir() or not any(d.iterdir()):
            # pool 还在（没划分）时这条不算错
            if (ds / POOL_DIR).is_dir():
                continue
            problems.append(f"images/{sub} 是空的 —— 先 `a9route train split`")
    info = stats(root=ds if root else None, name=name, read_images=False)
    problems.extend(info.problems)
    if info.images == 0:
        problems.append("一张图都没有")
    if info.images and info.background == 0:
        problems.append("**全是正样本、一张背景帧都没有** —— "
                        "模型会把「总能找到按键」学进去（先跑 build，别手工筛帧）")
    if info.positives == 0:
        problems.append("一个标注框都没有")
    counts = [info.boxes_by_class.get(c, 0) for c in CLASSES]
    if counts and max(counts) > 8 * max(1, min(counts)):
        lo = CLASSES[counts.index(min(counts))]
        hi = CLASSES[counts.index(max(counts))]
        problems.append(f"类别不平衡：{hi} {max(counts)} 个框 vs {lo} {min(counts)} 个 —— "
                        f"训练时给 {lo} 更大的 `--class-weight`，或补它的样本")
    return problems


def summary_line(name: str = "keys") -> str:
    """一行话描述数据集（CLI 收尾用）。"""
    try:
        info = stats(name=name)
    except DatasetError as exc:
        return str(exc)
    return (f"{name}: {info.images} 帧 / {info.positives} 框 / "
            f"背景 {info.background} / 来源 {len(info.sources)} 段")
