# -*- coding: utf-8 -*-
"""train.choice —— **选路（岔路口）标注**：另一套类别 + 三个选择题。

## 为什么选路要单独一套

刹车/氮气是"两个固定位置上的按键有没有亮"，选路是"屏幕上出现了一排**圆形路标**，
其中**蓝色的那个**是当前选中的路"。两者除了都用 YOLO 之外没有共同点：

| | 刹车 / 氮气 | **选路** |
|---|---|---|
| 类别 | `brake_pressed` / `nitro_pressed` | `choice_icon` / **`choice_selected`** |
| 位置 | 固定（标定框） | 每帧不同（2~4 个圆形） |
| 一帧几个目标 | 0~2 | 0~4 |
| 人怎么标 | 逐个"按下/没按" | **三个选择题**（见下） |
| 界面 | `/dataset` | **`/choice`（单独一页，不混在一起）** |

## 三个选择题 ↔ 标注怎么对应

用户定的标注方式（2026-09-15）：

1. **是否有选路？** 有 / 没有
2. **有几个选项？** 2 / 3 / 4（用户规则：一屏只有 2/3/4 条路；**至少 2 个路标**才算岔路口）
3. **选第几个？** 1..N（按**从左到右**编号；用户在游戏里点哪个）

落成 YOLO 标注就是：

* "没有" -> **空标注文件**（背景帧）；
* "有，N 个选项，选第 k 个" -> **N 个框**，全部按 x 排序，
  第 k 个（1 起）的类别是 `choice_selected`，其余是 `choice_icon`。

于是**三个答案全都能从标注反推回来**（`answers_from_boxes()`）——
界面上重新打开一帧时，三个选择题的位置就是上次的答案 ✓。

> **位置从哪来**：人只答三个选择题，不画框。框的位置来自启发式
> （`hud.choice_icons()` 的 HoughCircles）的预标注；
> 界面上**点 band 图**可以加/删框来修位置（数量对不上时最常用）。
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from a9route import paths
from a9route.train.dataset import (FRAME_META, POOL_DIR, dataset_dir,
                                   load_meta, read_frames_meta, save_meta)
from a9route.train.labels import (CLASS_ZH, Box, read_label, write_label,
                                  write_label_if_changed)

#: 选路模型的类别（顺序就是 YOLO 的 class id，**只能往后追加**）
CHOICE_CLASSES: tuple = ("choice_icon", "choice_selected")
CHOICE_IDS: dict = {n: i for i, n in enumerate(CHOICE_CLASSES)}
CHOICE_ZH: dict = {"choice_icon": "路标（未选中）",
                   "choice_selected": "路标（选中）"}
#: 复核页/画框用的颜色（BGR）
CHOICE_COLORS: dict = {"choice_icon": (0, 215, 255), "choice_selected": (60, 60, 245)}

#: 用户规则：一屏只有 2/3/4 条路，**至少 2 个**才算岔路口
MIN_OPTIONS = 2
MAX_OPTIONS = 4


def option_range(cfg: dict | None = None) -> tuple[int, int]:
    """第②题「有几个选项」允许的范围 `(lo, hi)`。

    默认是用户口径 **2~4**（1 个图标是普通路标，不算岔路口），
    但**允许配置改**：`vision.choice_min_options` / `vision.choice_max_options`。

    ⚠️ 这两个配置项一开始是**写了没人读**的（`train/choice.py` 里写死 2/4，
    页面也写死 2/3/4）—— 那就是个"看起来能调、其实调了没用"的假旋钮。
    现在三处（训练侧写回、运行侧折叠、界面按钮）**都走这一个函数**：
    谁改都能生效，而且页面上显示的范围和后端夹取的范围不可能不一致。
    """
    vis = {}
    if cfg is not None:
        vis = (cfg.get("vision") or {}) if isinstance(cfg, dict) else {}
    else:
        try:
            from a9route import config as cfgmod
            vis = (cfgmod.load_config() or {}).get("vision") or {}
        except Exception:                              # noqa: BLE001
            vis = {}
    try:
        lo = int(vis.get("choice_min_options", MIN_OPTIONS))
    except (TypeError, ValueError):
        lo = MIN_OPTIONS
    try:
        hi = int(vis.get("choice_max_options", MAX_OPTIONS))
    except (TypeError, ValueError):
        hi = MAX_OPTIONS
    lo = max(1, min(8, lo))
    hi = max(lo, min(8, hi))                           # hi 永远不小于 lo（否则没法答）
    return lo, hi

#: 默认数据集名（和刹车/氮气的 `keys` 分开）
DEFAULT_NAME = "choice"

#: 裁 band 图时**上下左右各留的边**（路标偶尔会略微超出带区）。
#: ⚠️ 这个值**必须和 `web/choice.py` 里用的一致** —— 所以那边改成调
#: `band_crop_rect()`，而不是自己写一遍常量（两处各写一遍必然漂移）。
BAND_PAD = 10


def band_crop_rect(frame_w: int = 1280, frame_h: int = 720) -> list:
    """band 图**实际覆盖的整帧区域** `[x, y, w, h]`。

    ⚠️ 这不是 `CHOICE_BAND` 本身 —— 它比带区**各边多 10px**。
    界面必须拿**这个**把点击位置换算回整帧坐标：

        整帧x = crop[0] + 图内像素x * (crop[2] / 图的自然宽度)

    第一版前端按 `CHOICE_BAND × zoom` 换算，于是整体偏了约 4% ✗
    （越靠右越偏，最多三百多像素 —— 一眼可见）。
    """
    from a9route.vision.hud import CHOICE_BAND
    bx, by, bw, bh = CHOICE_BAND
    x0, y0 = max(0, int(bx) - BAND_PAD), max(0, int(by) - BAND_PAD)
    x1 = min(int(frame_w), int(bx) + int(bw) + BAND_PAD)
    y1 = min(int(frame_h), int(by) + int(bh) + BAND_PAD)
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


# ---------------------------------------------------------------- 三个答案 ↔ 框
@dataclass
class Answers:
    """三个选择题的答案。`selected` 是 **1 起**的下标（0 = 没选/没有选路）。"""

    has: bool = False
    count: int = 0
    selected: int = 0

    def as_dict(self) -> dict:
        return {"has": bool(self.has), "count": int(self.count),
                "selected": int(self.selected)}

    def describe(self) -> str:
        if not self.has:
            return "没有选路"
        return "{0} 个选项，选第 {1} 个".format(self.count, self.selected)


def answers_from_boxes(boxes: list) -> Answers:
    """N 个框 -> 三个答案（**按 x 从左到右**编号）。

    * 一个框都没有 -> "没有选路"；
    * 有框 -> `count` = 框数，`selected` = `choice_selected` 那个排第几（1 起）；
      如果一个 `choice_selected` 都没有（旧的/启发式标的），`selected` 记 0，
      界面上会提示"没标出选中的是哪个"。
    """
    if not boxes:
        return Answers(False, 0, 0)
    ordered = sorted(boxes, key=lambda b: b.center[0])
    sel = 0
    for i, b in enumerate(ordered, 1):
        if b.name == "choice_selected":
            sel = i
            break
    return Answers(True, len(ordered), sel)


def boxes_from_answers(ans: Answers, positions: list) -> list[Box]:
    """三个答案 + 位置 -> `Box` 列表。

    `positions`：`[(x, y, w, h), ...]`（**整帧像素**，按 x 已排序或未排序都行，
    这里会重新按中心 x 排序）。选第 k 个 -> 第 k 个框的类别是 `choice_selected`。
    """
    if not ans.has or ans.count <= 0:
        return []
    pos = sorted(positions, key=lambda p: float(p[0]) + float(p[2]) / 2.0)
    pos = pos[:ans.count]
    out = []
    for i, p in enumerate(pos, 1):
        cls = (CHOICE_IDS["choice_selected"] if i == ans.selected
               else CHOICE_IDS["choice_icon"])
        out.append(Box(cls, float(p[0]), float(p[1]), float(p[2]), float(p[3]),
                       names=CHOICE_CLASSES))
    return out


def icons_to_boxes(icons: list) -> list[Box]:
    """`hud.choice_icons()` 的结果 -> `Box`（圆的**外接正方框**）。

    `icons`：`[{"xy": (cx, cy), "radius": r, "blue": bool, ...}, ...]`
    """
    out = []
    for s in icons:
        cx, cy = s["xy"]
        r = float(s.get("radius") or 28)
        cls = (CHOICE_IDS["choice_selected"] if s.get("blue")
               else CHOICE_IDS["choice_icon"])
        out.append(Box(cls, cx - r, cy - r, 2 * r, 2 * r,
                       conf=float(s.get("blue_ratio") or 0.0) or None,
                       names=CHOICE_CLASSES))
    return sorted(out, key=lambda b: b.center[0])


def icons_payload(icons: list) -> list:
    """`hud.choice_icons()` 的结果 -> 给界面用的 JSON（整帧像素坐标）。"""
    return [{"x": int(s["xy"][0]), "y": int(s["xy"][1]),
             "r": int(s.get("radius") or 28),
             "blue": bool(s.get("blue")),
             "blue_ratio": float(s.get("blue_ratio") or 0.0)}
            for s in icons]


def prelabel(frame, reader=None, detector=None) -> tuple[list, list]:
    """一帧 -> `(Box 列表, icons 原始列表)`。

    默认用启发式 `hud.choice_icons`（HoughCircles）；给了 `detector`
    （`vision.choice` 的后端：OnnxChoice / UltralyticsChoice）就用**模型** ——
    用户 2026-09-15 的用法："用先前练的两个模型推一个新的训练集出来"。
    """
    if detector is not None:
        r = detector.read(frame)
        return icons_to_boxes(r.icons), list(r.icons)
    from a9route.vision.hud import RaceReader
    r = reader or RaceReader()
    icons = r.choice_icons(frame)
    return icons_to_boxes(icons), icons


def _prelabel_detector(cfgmod) -> tuple:
    """选路预标注用哪个后端：`(detector 或 None, 人话的来源说明)`。

    和按键那边同一个口径：`auto` 时模型不在就**报错**，不静默退回 HoughCircles。
    """
    from pathlib import Path

    try:
        from a9route.vision import choice as VC
    except Exception as exc:                          # noqa: BLE001
        return None, "heuristic(hud.choice_icons)（vision.choice 起不来：{0}）".format(exc)
    be, mp, _p = VC.resolve_backend()
    if be == "heuristic":
        return None, "heuristic(hud.choice_icons) —— 配置里显式指定了启发式"
    det = VC.build_choice_detector()
    return det, "{0}:{1}".format(be, Path(mp).name)


# ---------------------------------------------------------------- 数据集
def choice_dataset_dir(name: str = DEFAULT_NAME) -> Path:
    return dataset_dir(name)


def build_from(name: str = DEFAULT_NAME, *, source: str = "keys",
               progress=print) -> dict:
    """**把现有数据集分类**：复用 `source`（默认 `keys`）的帧，贴一套选路标注。

    为什么复用而不是重扫录像：用户要的就是"把现有的数据分类"；
    而且帧是同一批（选路路标和按键出现在同样的画面上），
    重扫录像既慢又会得到另一套抽样。

    图片用**硬链接**（同一卷上几乎不占空间），失败就退回复制。
    train/val 划分**照抄 `source` 的** —— 同一段录像不能既在 train 又在 val，
    否则又是数据泄漏（见 `dataset.py` 开头那段）。
    """
    import cv2

    from a9route import config as cfgmod
    cfgmod.apply()

    src = dataset_dir(source)
    if not (src / FRAME_META).is_file():
        raise RuntimeError("源数据集不存在或没有 {0}：{1}".format(FRAME_META, src))
    metas = read_frames_meta(src)
    if not metas:
        raise RuntimeError("源数据集里没有帧：{0}".format(src))

    dst = dataset_dir(name)
    img_root, lbl_root = dst / "images", dst / "labels"
    # 重建（**不删 dataset.json 的历史**，只清图/标注）
    for d in (img_root, lbl_root):
        if d.exists():
            shutil.rmtree(d)

    reader = None
    # ---- 预标注用哪个后端：**默认跟运行时后端走**（用户要求用模型推数据集）----
    pre_det, prelabel_by = _prelabel_detector(cfgmod)
    if progress:
        progress("  选路预标注判据：{0}".format(prelabel_by))
    out_meta: list = []
    stats = {"frames": 0, "with_choice": 0, "linked": 0, "copied": 0,
             "no_source_image": 0, "by_count": {}, "selected": {}}

    # 源数据集按 split 分好了，照抄它
    from a9route.train.audit import split_dirs
    for split, s_img_dir, _s_lbl_dir in split_dirs(src):
        if split == POOL_DIR:
            continue                        # pool 里是同一批帧的副本，跳过
        d_imgs = img_root / split
        d_lbls = lbl_root / split
        d_imgs.mkdir(parents=True, exist_ok=True)
        d_lbls.mkdir(parents=True, exist_ok=True)
        names = {p.name for p in s_img_dir.iterdir()
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        for m in metas:
            fn = m.get("file")
            if fn not in names:
                continue
            s_img = s_img_dir / fn
            if not s_img.is_file():
                stats["no_source_image"] += 1
                continue
            frame = cv2.imread(str(s_img))
            if frame is None:
                stats["no_source_image"] += 1
                continue
            boxes, icons = prelabel(frame, detector=pre_det)
            write_label(d_lbls / (Path(fn).stem + ".txt"), boxes,
                        int(m.get("width") or frame.shape[1]),
                        int(m.get("height") or frame.shape[0]))
            _link_or_copy(s_img, d_imgs / fn, stats)
            ans = answers_from_boxes(boxes)
            out_meta.append({
                "file": fn, "video": m.get("video") or "", "t": m.get("t"),
                "frame": m.get("frame"), "split": split,
                "width": int(m.get("width") or frame.shape[1]),
                "height": int(m.get("height") or frame.shape[0]),
                "answers": ans.as_dict(),
                "detected": {"count": len(icons),
                             "selected": ans.selected,
                             "icons": icons_payload(icons)},
                "reason": "choice" if ans.has else "bg",
                "manual": False,
                "status": "prelabel",
            })
            stats["frames"] += 1
            if ans.has:
                stats["with_choice"] += 1
                stats["by_count"][ans.count] = stats["by_count"].get(ans.count, 0) + 1
                stats["selected"][ans.selected] = (
                    stats["selected"].get(ans.selected, 0) + 1)
            if progress and stats["frames"] % 200 == 0:
                progress("    分类 {0} 帧…".format(stats["frames"]))

    if not stats["frames"]:
        raise RuntimeError("一帧都没处理（源数据集的 images/ 是空的？）")

    (dst / FRAME_META).write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in out_meta),
        encoding="utf-8")
    info = load_meta(dst)
    info.update({
        "name": name,
        "task": "choice",
        "classes": list(CHOICE_CLASSES),
        "classes_zh": CHOICE_ZH,
        "created": info.get("created") or time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_dataset": source,
        "videos": sorted({m.get("video") for m in out_meta if m.get("video")}),
        "prelabel": prelabel_by,
        "frames": len(out_meta),
        "reviewed_frames": 0,
    })
    save_meta(dst, info)
    from a9route.train.labels import write_data_yaml
    write_data_yaml(dst / "data.yaml", dst, classes=CHOICE_CLASSES,
                    zh=CHOICE_ZH)
    return {"dataset": str(dst), **stats}


def _link_or_copy(src: Path, dst: Path, stats: dict) -> None:
    """硬链接（同卷上几乎不占空间）；不行就复制。

    硬链接是这里的关键：选路复用**同一批帧**，复制一份要多占 ~175 MB，
    而硬链接 0 额外空间、`cv2.imread` 也完全当普通文件读。
    """
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
        stats["linked"] += 1
    except OSError:
        shutil.copy2(str(src), str(dst))
        stats["copied"] += 1


# ---------------------------------------------------------------- 读写答案
def label_path(name: str, split: str, file: str) -> Path:
    return dataset_dir(name) / "labels" / split / (Path(file).stem + ".txt")


def _split_of(ds: Path, file: str) -> str:
    for sp in ("train", "val", POOL_DIR):
        base = (ds / "images" / sp) if sp != POOL_DIR else (ds / POOL_DIR / "images")
        if (base / file).is_file():
            return sp
    return "train"


def read_answers(name: str, split: str, file: str, w: int, h: int) -> tuple:
    """读一帧的标注 -> `(Answers, Box 列表)`。

    ⚠️ **答案从标注文件反推**（`answers_from_boxes`），不是从 meta 里读 ——
    这样手改过 `labels/*.txt` 之后界面显示的就是改过的样子。
    """
    p = label_path(name, split, file)
    boxes = read_label(p, w, h, classes=CHOICE_CLASSES) if p.is_file() else []
    return answers_from_boxes(boxes), boxes


def apply_answers(items: list, *, name: str = DEFAULT_NAME, progress=None) -> dict:
    """写回三个选择题的答案。

    `items`：`[{"file", "has", "count", "selected", "boxes"?, "status"?}, ...]`

    **位置从哪来**（按优先级）：

    1. `item["boxes"]`（界面上手工点出来的）—— 最准；
    2. meta 里存的 `detected.icons`（启发式预标注的位置）。

    若 2 的位置**不够** `count` 个（检测少了），就用最后一个框的间距**向右外推**补足，
    并把这一帧标上 `position_guessed` —— 老实说"位置是猜的"，
    而不是偷偷少写几个框（那样答案和标注就对不上了）。
    """
    ds = dataset_dir(name)
    metas = read_frames_meta(ds)
    if not metas:
        raise RuntimeError("数据集不存在或没有 {0}：{1}".format(FRAME_META, ds))
    by_file = {m.get("file"): m for m in metas if m.get("file")}
    prev: dict = {}
    changed = guessed = skipped = same = processed = 0
    lo, hi = option_range()
    for it in items:
        fn = it.get("file")
        m = by_file.get(fn)
        if m is None:
            skipped += 1
            continue
        split = m.get("split") or _split_of(ds, fn)
        w = int(m.get("width") or 1280)
        h = int(m.get("height") or 720)
        ans = Answers(bool(it.get("has")), int(it.get("count") or 0),
                      int(it.get("selected") or 0))
        if not ans.has:
            ans = Answers(False, 0, 0)
        else:
            # 兜底：答案不合法就修正成合理值（宁可能用，也别写出坏标注）
            ans.count = max(lo, min(hi, ans.count))
            ans.selected = max(1, min(ans.count, ans.selected or 1))
        p = label_path(name, split, fn)
        old_label = p.read_text(encoding="utf-8") if p.is_file() else None
        old_ans = m.get("answers")
        old_status = m.get("status")
        old_manual = bool(m.get("manual"))
        old_guess = bool(m.get("position_guessed"))

        manual = bool(it.get("boxes"))
        guessed_here = False
        if ans.has:
            if manual:
                positions = [[float(v) for v in b] for b in it["boxes"]]
            else:
                icons = ((m.get("detected") or {}).get("icons") or [])
                positions = [[ic["x"] - ic["r"], ic["y"] - ic["r"],
                              2 * ic["r"], 2 * ic["r"]] for ic in icons]
                positions.sort(key=lambda b: b[0] + b[2] / 2.0)
            positions = positions[:ans.count]
            if len(positions) < ans.count:
                positions = _pad_positions(positions, ans.count)
                guessed_here = True
            boxes = boxes_from_answers(ans, positions)
        else:
            boxes = []
        p, wrote_label = write_label_if_changed(p, boxes, w, h)
        m["answers"] = ans.as_dict()
        m["manual"] = manual
        # ⚠️ **每次都要重新赋值**（true/false 都写）——
        # 只写 `= True` 的话，人后来手工把位置摆正了，这个标记还挂着，
        # 界面会一直提示"位置是猜的"（实测抓到）✗
        m["position_guessed"] = guessed_here
        m["status"] = str(it.get("status") or "reviewed")
        processed += 1
        # ⚠️ `changed` 必须是"**真的改了**"，不能是"处理了几帧"：
        # 界面上报给人的是「改动 N 帧」（说假话就是 bug），
        # 而且它同时决定撤销点要不要挪 —— 空提交把撤销点顶掉，
        # 人就回不到自己上一次真正的改动了（NOTES 10.15/10.22）。
        if (wrote_label or m["answers"] != old_ans or m["status"] != old_status
                or m["manual"] != old_manual
                or m["position_guessed"] != old_guess):
            changed += 1
            prev[str(p)] = old_label
            if guessed_here:
                guessed += 1
        else:
            same += 1
        if progress and processed % 50 == 0:
            progress("    写回 {0} 帧…".format(processed))

    (ds / FRAME_META).write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in metas),
        encoding="utf-8")
    info = load_meta(ds)
    info["reviewed_frames"] = sum(1 for x in metas if x.get("status") == "reviewed")
    if changed:
        info["last_answers"] = {"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "files": list(prev), "prev": prev,
                                "changed": changed,
                                "same": same, "processed": processed}
    save_meta(ds, info)
    return {"items": len(items), "changed": changed, "same": same,
            "processed": processed, "skipped": skipped,
            "guessed": guessed, "reviewed_total": info["reviewed_frames"]}


def _pad_positions(positions: list, count: int) -> list:
    """位置不够时**向右外推**补足（并会让调用方标上 `position_guessed`）。

    为什么不用"随便填中间"：路标是一排等距的圆，用最后两个的间距外推
    至少方向和量级是对的，人一眼就能看出偏了没有（界面上也会提示"位置是猜的"）。
    """
    out = [list(p) for p in positions]
    if not out:                                  # 一个都没有：给一排默认位置
        from a9route.vision.hud import CHOICE_BAND
        bx, by, bw, bh = CHOICE_BAND
        step = max(40.0, bw / float(count + 1))
        w = hh = 58.0
        cy = by + bh / 2.0 - hh / 2.0
        for i in range(count):
            out.append([bx + step * (i + 1) - w / 2.0, cy, w, hh])
        return out
    step = out[-1][2] if len(out) == 1 else (out[-1][0] - out[-2][0])
    step = step if abs(step) > 1 else 60.0
    while len(out) < count:
        last = out[-1]
        out.append([last[0] + step, last[1], last[2], last[3]])
    return out


def undo_answers(*, name: str = DEFAULT_NAME) -> dict:
    """撤销**上一次** `apply_answers()`（把标注文件按原样写回去）。"""
    ds = dataset_dir(name)
    info = load_meta(ds)
    last = info.get("last_answers")
    if not last or not last.get("prev"):
        raise RuntimeError("没有可撤销的记录（没写过，或者已经撤过了）")
    restored = 0
    metas = read_frames_meta(ds)
    by_stem = {Path(m.get("file") or "").stem: m for m in metas}
    for path_str, content in last["prev"].items():
        p = Path(path_str)
        if content is None:
            if p.is_file():
                p.unlink()
                restored += 1
        else:
            if not p.is_file() or p.read_text(encoding="utf-8") != content:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
                restored += 1
        stem = p.stem
        m = by_stem.get(stem)
        if m is not None:
            # 答案跟着标注一起回滚（标注是事实来源，这里重新推一遍）
            ans, _b = read_answers(name, m.get("split") or "train",
                                   m.get("file") or "", int(m.get("width") or 1280),
                                   int(m.get("height") or 720))
            m["answers"] = ans.as_dict()
            m["status"] = "prelabel"
    (ds / FRAME_META).write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in metas),
        encoding="utf-8")
    info["reviewed_frames"] = sum(1 for x in metas if x.get("status") == "reviewed")
    info.pop("last_answers", None)
    save_meta(ds, info)
    return {"restored": restored, "files": len(last["prev"]),
            "reviewed_total": info["reviewed_frames"], "at": last.get("at")}


# ---------------------------------------------------------------- 浏览 / 统计
def stats(name: str = DEFAULT_NAME) -> dict:
    """数据集的现状（界面上那行统计）。"""
    ds = dataset_dir(name)
    metas = read_frames_meta(ds)
    out = {"frames": len(metas), "reviewed": 0, "with_choice": 0,
           "by_count": {}, "selected": {}, "guessed": 0, "no_selected": 0}
    for m in metas:
        if m.get("status") == "reviewed":
            out["reviewed"] += 1
        if m.get("position_guessed"):
            out["guessed"] += 1
        a = m.get("answers") or {}
        if a.get("has"):
            out["with_choice"] += 1
            c = int(a.get("count") or 0)
            out["by_count"][c] = out["by_count"].get(c, 0) + 1
            s = int(a.get("selected") or 0)
            out["selected"][s] = out["selected"].get(s, 0) + 1
            if s == 0:
                out["no_selected"] += 1
    return out


def frame_rows(name: str = DEFAULT_NAME, *, offset: int = 0, limit: int = 24,
               flt: str = "all", search: str = "") -> dict:
    """界面用的一页数据。

    `flt`：`all` / `pending`（没标过）/ `reviewed` / `has_choice` /
    `no_choice` / `mismatch`（**答案和启发式检测对不上** —— 最该人看的）。

    `search`：文件名 / 录像名的子串匹配（界面上的搜索框）。
    ⚠️ 这个参数是**补上的**：第一版接口漏了它，而界面有搜索框 ——
    于是"搜索"静默失效（后端忽略未知查询参数），看起来像前端坏了 ✗。
    """
    ds = dataset_dir(name)
    if not ds.is_dir():
        raise FileNotFoundError("数据集不存在：{0}".format(ds))
    metas = read_frames_meta(ds)
    lo, hi = option_range()
    from a9route.vision.hud import CHOICE_BAND
    band = list(CHOICE_BAND)
    crop = band_crop_rect()

    rows = []
    for m in metas:
        fn = m.get("file")
        if not fn:
            continue
        split = m.get("split") or _split_of(ds, fn)
        w = int(m.get("width") or 1280)
        h = int(m.get("height") or 720)
        ans, boxes = read_answers(name, split, fn, w, h)
        det = m.get("detected") or {}
        flags = []
        if m.get("position_guessed"):
            flags.append("position_guessed")
        if ans.has and int(det.get("count") or 0) != ans.count:
            flags.append("detected_mismatch")
        if ans.has and ans.selected == 0:
            flags.append("no_selected")
        if ans.has and ans.count < lo:
            flags.append("too_few")
        # ⚠️ 反过来的那一半：**检测到图标、但不够 min 个**（默认 2）——
        # 按规则它不算岔路口，答案会是「没有选路」。可是那个图标是**真存在**的，
        # 要是不留个标记，翻页自动复核之后这一帧就再也看不出"这里有 1 个图标"了
        # （答案=没有、检测数=1 在界面上完全正常）→ 口径问题被悄悄抹平 ✗
        elif not ans.has and 0 < int(det.get("count") or 0) < lo:
            flags.append("below_min")
        rows.append({
            "file": fn, "video": m.get("video") or "", "t": float(m.get("t") or 0),
            "split": split, "status": m.get("status") or "prelabel",
            "answers": ans.as_dict(),
            "detected": {"count": int(det.get("count") or 0),
                         "selected": int(det.get("selected") or 0),
                         "icons": list(det.get("icons") or [])},
            "manual": bool(m.get("manual")),
            "boxes": [[round(b.x, 1), round(b.y, 1), round(b.w, 1), round(b.h, 1)]
                      for b in boxes],
            "band_rect": band,
            #: band 图**实际覆盖的整帧区域**（比带区各边多 10px）。
            #: 界面必须用**这个**做点击↔整帧坐标的换算，不能用 band_rect × zoom。
            "crop_rect": crop,
            "flags": flags,
        })

    if flt == "pending":
        rows = [r for r in rows if r["status"] != "reviewed"]
    elif flt == "reviewed":
        rows = [r for r in rows if r["status"] == "reviewed"]
    elif flt == "has_choice":
        rows = [r for r in rows if r["answers"]["has"]]
    elif flt == "no_choice":
        rows = [r for r in rows if not r["answers"]["has"]]
    elif flt == "mismatch":
        rows = [r for r in rows if r["flags"]]
    if search:
        q = search.strip().lower()
        rows = [r for r in rows
                if q in r["file"].lower() or q in (r["video"] or "").lower()]

    total = len(rows)
    page = rows[max(0, offset):max(0, offset) + max(1, limit)]
    return {"dataset": name, "root": str(ds), "all_frames": len(metas),
            #: 第②题允许的范围 —— 界面**按这个渲染按钮**，不许自己写死 2/3/4
            "options": {"min": lo, "max": hi},
            "total": total, "total_known": True, "offset": offset,
            "next_offset": offset + len(page), "limit": limit,
            "rows": page, "band_rect": band, "stats": stats(name),
            "meta": load_meta(ds)}


def dataset_list() -> dict:
    """`/api/choice/list`：列出所有**选路**数据集（task=choice 的）。"""
    root = paths.DATASETS_DIR
    out = []
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            info = load_meta(d)
            if (info.get("task") or "").lower() != "choice":
                continue
            st = stats(d.name)
            out.append({"name": d.name, "frames": st["frames"],
                        "reviewed": st["reviewed"],
                        "with_choice": st["with_choice"],
                        "has_split": (d / "data.yaml").is_file(),
                        "source_dataset": info.get("source_dataset") or ""})
    return {"datasets": out, "root": str(root)}

