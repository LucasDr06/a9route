# -*- coding: utf-8 -*-
"""train.audit —— **数据集体检 + 逐帧浏览**（只读，不改任何东西）。

这个模块回答一个很具体的问题：

> **数据集里的标注，和"标定"（`config.json` 里的按键框 + 阈值）对得上吗？**

对不上是最阴的一类问题：数据集是用**当时的**标定建的，之后你改了
`vision__brake_key_box` 或调了 `nitro_red_thr`，于是

* 标注框还落在**旧位置** → 训练出来的模型学的是错的框；
* 标注的"按下/没按"还是**旧阈值**的判断 → 模型在学一个已经不成立的判据；
* 而训练、评估、导出**都不会报错**，只会静默变差 ✗。

## 三层检查

| 层 | 查什么 | 为什么值得单独一层 |
|---|---|---|
| `layout` | 图片↔标注↔`meta.jsonl` 三者的对应（谁多谁少、谁指向不存在的文件） | 少一个 `labels/*.txt` 会让 YOLO 把那张图当"没标"而不是"背景帧" |
| `calibration` | 标注框 vs **当前** `config.json` 的按键框；标定自身的左右对称；分辨率 | 这就是"标定数据是否对应" |
| `signals` | 拿**当前**判据重算一遍，和 `meta.jsonl` 里存的信号/标注比 | 抓"阈值被改过"这种看不见的漂移 |

`signals` 那一层要真的解图、真的跑判据，所以默认**抽样**（`limit`）；
Web 界面里则只对**当前这一页**的帧重算，看到哪查到哪。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from a9route import paths
from a9route.train import frames as F
from a9route.train.dataset import (FRAME_META, POOL_DIR, dataset_dir,
                                   load_meta, read_frames_meta)
from a9route.train.labels import CLASSES, CLASS_ZH, LabelError, read_label

#: 体检结论的级别
LEVELS = ("ok", "warn", "error")

#: 框坐标比较容差（像素）。YOLO 存的是归一化坐标，来回换算会有小数误差；
#: 1.5px 足够容下它，又远小于"框挪了位置"的幅度（几十像素）
BOX_TOL = 1.5

#: 一个体检项里最多带几个"例子文件名"（界面要显示，太多没意义）
MAX_SAMPLES = 8

#: 逐帧 `only="problem"` 认哪些标记。
#:
#: ⚠️ **`human_fix` 故意不在里面** —— 它的意思是"人复核过、标注和启发式不同"，
#: 那是**成果**、不是问题。要是把它也算成"有问题"，用户越认真复核，
#: "只看有问题的"就越会刷出他自己的修正记录，意思完全反了 ✗
#: （这条是复核了 30 帧之后立刻暴露出来的）。
PROBLEM_FLAGS = ("label_drift", "signal_drift", "calib_mismatch",
                 "missing_file", "undecodable")


@dataclass
class Finding:
    """一条体检结论。`kind` 是分组键，`level` 是严重程度。"""

    kind: str
    level: str
    title: str
    detail: str = ""
    samples: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "level": self.level, "title": self.title,
                "detail": self.detail, "samples": list(self.samples)}


@dataclass
class AuditReport:
    """一次体检的全部结论。"""

    dataset: str = ""
    root: str = ""
    #: `keys`（刹车/氮气）或 `choice`（选路）—— 判据完全不同，见 `_task_of()`
    task: str = "keys"
    findings: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    rechecked: int = 0
    drift: dict = field(default_factory=dict)

    @property
    def problems(self) -> list:
        return [f for f in self.findings if f.level in ("warn", "error")]

    @property
    def ok(self) -> bool:
        return not any(f.level == "error" for f in self.findings)

    def add(self, kind: str, level: str, title: str, detail: str = "",
            samples=None) -> None:
        self.findings.append(Finding(kind, level, title, detail,
                                     list(samples or [])[:MAX_SAMPLES]))

    def as_dict(self) -> dict:
        return {"dataset": self.dataset, "root": self.root, "task": self.task,
                "findings": [f.as_dict() for f in self.findings],
                "counts": self.counts, "calibration": self.calibration,
                "rechecked": self.rechecked, "drift": self.drift,
                "problems": len(self.problems), "ok": self.ok}

    def describe(self) -> str:
        L = ["=" * 64,
             "数据集体检：{0}（{1}）".format(
                 self.dataset, "选路" if self.task == "choice" else "刹车/氮气按键"),
             "=" * 64,
             "  目录: {0}".format(self.root)]
        for k, v in sorted(self.counts.items()):
            L.append("  {0:<16} {1}".format(k, v))
        if self.calibration:
            L.append("")
            L.append("  当前标定（config.json）:")
            for k, v in self.calibration.items():
                L.append("    {0:<24} {1}".format(k, v))
        L.append("")
        mark = {"ok": "[OK  ]", "warn": "[注意]", "error": "[问题]"}
        for f in self.findings:
            L.append("  {0} {1}".format(mark.get(f.level, "[?]"), f.title))
            if f.detail:
                L.append("          {0}".format(f.detail))
            if f.samples:
                L.append("          例: {0}{1}".format(
                    ", ".join(f.samples),
                    " …" if len(f.samples) >= MAX_SAMPLES else ""))
        L.append("")
        n_err = sum(1 for f in self.findings if f.level == "error")
        n_warn = sum(1 for f in self.findings if f.level == "warn")
        L.append("结论: {0} 个问题 / {1} 个注意".format(n_err, n_warn))
        return "\n".join(L)


# ---------------------------------------------------------------- 布局
def split_dirs(ds: Path) -> list:
    """数据集里所有 `(split 名, 图片目录, 标注目录)`。

    ⚠️ **顺序 = 优先级**：`train` / `val` 排在 `pool` 前面。
    同一帧在 `pool/` 和 `images/train/` 下都有一份（`split` 是拷过去的），
    按"先到先得"去重时必须让**划分后的那份**赢 —— 否则统计出来永远显示
    "全在 pool"，界面上就看不出 train/val 的分布（第一版就是这么错的）。
    """
    out = []
    for sp in ("train", "val"):
        img, lbl = ds / "images" / sp, ds / "labels" / sp
        if img.is_dir():
            out.append((sp, img, lbl))
    pool_img, pool_lbl = ds / POOL_DIR / "images", ds / POOL_DIR / "labels"
    if pool_img.is_dir():
        out.append(("pool", pool_img, pool_lbl))
    return out


def resolve_image(ds: Path, file: str, split: str = "") -> Path | None:
    """按文件名找图片（`split` 为空就依次找 train/val/pool）。

    ⚠️ 只接受**纯文件名**：带路径分隔符/`..` 一律拒绝 —— 这个函数会被 HTTP
    接口直接调用，不能让 `file=../../config.json` 把项目文件读出去。
    """
    if not file or "/" in file or "\\" in file or file in (".", "..") \
            or ":" in file:
        return None
    cands = []
    for name, img_dir, _lbl in split_dirs(ds):
        if split and name != split:
            continue
        cands.append(img_dir / file)
    for c in cands:
        if c.is_file():
            return c
    return None


# ---------------------------------------------------------------- 体检
def _calibration_now() -> dict:
    """当前生效的标定（**必须在 `config.apply()` 之后读**）。"""
    from a9route.vision import cues
    bb = tuple(int(v) for v in cues.BRAKE_KEY_BOX)
    nb = tuple(int(v) for v in cues.NITRO_KEY_BOX)
    return {
        "brake_key_box": list(bb),
        "nitro_key_box": list(nb),
        "brake_ring_thr": float(cues.BRAKE_RING_THR),
        "brake_white_thr": float(cues.BRAKE_WHITE_THR),
        "nitro_red_thr": float(cues.NITRO_RED_THR),
        "white_thr": int(cues.WHITE_THR),
        "bright_thr": int(cues.BRIGHT_THR),
        "ring_pad": int(cues.RING_PAD),
    }


def _symmetry_note(cal: dict) -> tuple:
    """标定自身的自检：两个按键框应当关于屏幕中线**完全对称**（README 记的不变量）。"""
    bx, _by, bw, _bh = cal["brake_key_box"]
    nx = cal["nitro_key_box"][0]
    width = 1280
    expect = width - (bx + bw)
    if nx == expect:
        return ("ok", "两个按键框关于屏幕中线对称（{0} = 1280-({1}+{2})）".format(
            nx, bx, bw))
    return ("warn", "两个按键框**不对称**：氮气框 x={0}，按对称应当是 {1}"
                    "（= 1280-({2}+{3})）。框多半量歪了".format(nx, expect, bx, bw))


def _task_of(ds: Path, name: str = "") -> tuple:
    """这个数据集是**哪个任务**的：`(task, classes, zh)`。

    ⚠️ 这件事以前**没有做**，后果是体检把选路数据集当成按键数据集来判：
    类别表读的是 `labels.CLASSES`（刹车/氮气），于是 `choice_icon` 的第 0 类
    被读成"刹车按下"，551 个选路图标框**全被拿去和两个按键框比位置** ——
    报出一个红色的「551 个框的位置和当前标定对不上」，而数据本身完全没问题 ✗✗
    （实测：用户看到的第一条就是这个假警报）。

    判定顺序：`dataset.json` 的 `task` -> `data.yaml` 的 names -> 默认按键那套。
    """
    from a9route.train import labels as L
    from a9route.train.choice import CHOICE_CLASSES, CHOICE_ZH
    task = ""
    try:
        info = load_meta(ds) or {}
        task = str(info.get("task") or "").lower()
        got = info.get("classes")
        if not task and got and tuple(got) == tuple(CHOICE_CLASSES):
            task = "choice"
    except Exception:                                  # noqa: BLE001
        pass
    if not task:
        try:
            txt = (ds / "data.yaml").read_text(encoding="utf-8")
            if "choice_icon" in txt:
                task = "choice"
        except OSError:
            pass
    if task == "choice":
        return "choice", tuple(CHOICE_CLASSES), dict(CHOICE_ZH)
    return "keys", tuple(L.CLASSES), dict(L.CLASS_ZH)


def _in_band(b, pad: int = 0) -> bool:
    """框是不是落在**选路带**里（选路任务唯一有意义的位置不变量）。"""
    from a9route.vision.hud import CHOICE_BAND
    x, y, w, h = CHOICE_BAND
    cx, cy = b.center
    return (x - pad <= cx <= x + w + pad) and (y - pad <= cy <= y + h + pad)


def audit(name: str = "keys", *, root=None, recheck: bool = True,
          limit: int = 300, progress=None) -> AuditReport:
    """跑一遍体检。

    `recheck=True` 时抽样 `limit` 帧**用当前判据重算**，和 `meta.jsonl` 里存的
    信号/标注对比 —— 这是唯一能发现"阈值/框被改过"的检查。

    **按任务分岔**：按键数据集查"框对不对得上两个按键框"，选路数据集查
    "框在不在选路带里、圆框大小像不像" —— 两套判据完全不同，不能互相套用。
    """
    import cv2

    from a9route import config as cfgmod

    cfgmod.apply()                      # 标定必须按配置读，不能读模块默认值
    ds = Path(root) if root else dataset_dir(name)
    rep = AuditReport(dataset=name, root=str(ds))
    if not ds.is_dir():
        rep.add("layout", "error", "数据集不存在：{0}".format(ds))
        return rep

    task, classes, zh = _task_of(ds)
    rep.task = task
    cal = _calibration_now()
    rep.calibration = cal
    metas = read_frames_meta(ds)
    meta_by_file = {}
    for m in metas:
        if m.get("file"):
            meta_by_file[m["file"]] = m

    # ---- 1. 图片 / 标注 / meta 三者的对应 ----
    seen: dict = {}                     # 文件名 -> (split, 图片路径, 标注路径)
    for sp, img_dir, lbl_dir in split_dirs(ds):
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                continue
            if img.name in seen:
                continue                # train/val 优先于 pool（同名就是同一帧）
            seen[img.name] = (sp, img, lbl_dir / (img.stem + ".txt"))

    rep.counts = {"图片": len(seen), "meta 行": len(metas),
                  "已复核": sum(1 for m in metas if m.get("status") == "reviewed")}
    if not seen:
        rep.add("layout", "error", "一张图都没有 —— 先 `a9route train build`")
        return rep
    split_of = {}
    for fn, (sp, _i, _l) in seen.items():
        split_of[sp] = split_of.get(sp, 0) + 1
    rep.counts["分布"] = "，".join("{0} {1}".format(k, v)
                                  for k, v in sorted(split_of.items()))

    # 1a) 图片 ↔ 标注
    no_label = [fn for fn, (_sp, _i, lbl) in seen.items() if not lbl.is_file()]
    if no_label:
        rep.add("layout", "error",
                "{0} 张图**没有标注文件**".format(len(no_label)),
                "背景帧的标注应当是**空文件**、不是不存在 —— "
                "YOLO 对「没标」和「空标注」的处理不一样", no_label)
    else:
        rep.add("layout", "ok", "每张图都有对应的标注文件（背景帧是空文件）")

    # 1b) 标注 ↔ 图片
    orphan = []
    for sp, _img_dir, lbl_dir in split_dirs(ds):
        if not lbl_dir.is_dir():
            continue
        for lbl in sorted(lbl_dir.iterdir()):
            if lbl.suffix.lower() != ".txt" or lbl.name.endswith(".cache"):
                continue
            if (lbl.stem + ".jpg") not in seen and (lbl.stem + ".png") not in seen:
                orphan.append(lbl.name)
    if orphan:
        rep.add("layout", "warn",
                "{0} 个标注文件**没有对应的图**".format(len(orphan)),
                "改了抽样参数后重新 build 过？旧标注没清干净会两边对不上",
                orphan)

    # 1c) meta ↔ 磁盘
    if not metas:
        rep.add("layout", "warn", "没有 {0}（无法核对抽样来源与复核状态）"
                .format(FRAME_META),
                "重新 `a9route train build` 会生成它")
    else:
        missing = [fn for fn in meta_by_file if fn not in seen]
        extra = [fn for fn in seen if fn not in meta_by_file]
        if missing:
            rep.add("layout", "error",
                    "meta 里有 {0} 帧在磁盘上找不到".format(len(missing)),
                    "meta.jsonl 是划分与统计的依据，对不上说明数据集被手动改过",
                    missing)
        if extra:
            rep.add("layout", "warn",
                    "磁盘上有 {0} 张图不在 meta 里".format(len(extra)),
                    "这些帧**不会**被算进统计，也没有抽样原因/信号记录", extra)
        if not missing and not extra:
            rep.add("layout", "ok", "meta.jsonl 与磁盘一一对应")

    # ---- 2. 标定 ----
    #
    # 先查一件很容易踩的事：**同一帧在 pool 和 train/val 下各有一份标注**
    # （`split` 是拷过去的）。手改 `pool/labels/*.txt` 而不重跑 `split`，
    # 改动就**完全不生效**（训练读的是 train/val 那份）—— 而且没有任何提示 ✗✗。
    # 这是"用户以为改好了、其实没改"的典型场景，必须报出来。
    from a9route.vision import cues
    copies: dict = {}
    for sp, _img_dir, lbl_dir in split_dirs(ds):
        if not lbl_dir.is_dir():
            continue
        for lbl in lbl_dir.iterdir():
            if lbl.suffix.lower() != ".txt" or lbl.name.endswith(".cache"):
                continue
            copies.setdefault(lbl.stem + ".jpg", []).append((sp, lbl))
    diverged = []
    for fn, group in copies.items():
        if len(group) < 2:
            continue
        texts = set()
        for _sp, lbl in group:
            try:
                lines = [ln.split() for ln in
                         lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            except OSError:
                continue
            texts.add(tuple(sorted(tuple(ln) for ln in lines)))
        if len(texts) > 1:
            diverged.append("{0}（{1}）".format(
                fn, " vs ".join("{0}:{1}个框".format(sp, len(_count(lbl)))
                                for sp, lbl in group)))
    if diverged:
        rep.add("layout", "error",
                "{0} 帧的标注在 pool 和 train/val 下**不一致**".format(len(diverged)),
                "同一帧有两份标注（`split` 是拷过去的）。**训练读的是 train/val 那份** —— "
                "你手改 `pool/labels/*.txt` 之后必须重跑 `a9route train split`，"
                "否则改动完全不生效（还没有任何提示）", diverged)
    elif any(len(g) > 1 for g in copies.values()):
        rep.add("layout", "ok", "pool 与 train/val 的标注一致（没有改完忘了重划分）")

    # 按键任务：每个框都必须落在**两个按键框**上（这是标定给的硬不变量）。
    # 选路任务：**没有**"固定位置"这回事（路标在哪由画面决定），
    # 唯一的位置不变量是"落在选路带里、且是方的" —— 判据从这里分岔。
    box_of = {"brake_pressed": cal["brake_key_box"],
              "nitro_pressed": cal["nitro_key_box"]}
    if task == "choice":
        rep.add("calibration", "ok",
                "这是**选路**数据集：不按两个按键框判位置（路标位置每帧都不同）")
    else:
        lvl, msg = _symmetry_note(cal)
        rep.add("calibration", lvl, "标定自检：" + msg)

    sizes: dict = {}
    box_mismatch: list = []
    out_of_band: list = []
    not_square: list = []
    unknown_cls: list = []
    bad_label: list = []
    per_class = {c: 0 for c in classes}
    background = 0
    both = 0
    checked = 0
    for fn, (sp, img, lbl) in seen.items():
        m = meta_by_file.get(fn) or {}
        w = int(m.get("width") or 0)
        h = int(m.get("height") or 0)
        if not w or not h:
            im = cv2.imread(str(img), cv2.IMREAD_REDUCED_COLOR_8)
            if im is not None:
                w, h = im.shape[1] * 8, im.shape[0] * 8
        if w and h:
            sizes["{0}x{1}".format(w, h)] = sizes.get("{0}x{1}".format(w, h), 0) + 1
        boxes: list = []
        if lbl.is_file() and w and h:
            try:
                # ⚠️ 用**这个数据集自己的类别表**读 —— 否则选路标注（第 0 类
                # `choice_icon`）会被读成"刹车按下"，后面全盘错位 ✗
                boxes = read_label(lbl, w, h, classes=classes)
            except LabelError as exc:
                bad_label.append("{0}: {1}".format(fn, exc))
                boxes = []
        checked += 1
        names = []
        for b in boxes:
            if b.name not in per_class:
                unknown_cls.append(fn)
                continue
            names.append(b.name)
            per_class[b.name] = per_class.get(b.name, 0) + 1
            if task == "choice":
                # 选路：位置不固定，查"在不在选路带里""方不方"（图标是圆）
                if not _in_band(b, pad=24):
                    out_of_band.append("{0}（{1} 中心 ({2},{3})）".format(
                        fn, zh.get(b.name, b.name),
                        round(b.center[0]), round(b.center[1])))
                elif abs(b.w - b.h) > max(6.0, 0.25 * b.w):
                    not_square.append("{0}（{1} {2}×{3}）".format(
                        fn, zh.get(b.name, b.name), round(b.w), round(b.h)))
            else:
                want = box_of[b.name]
                if any(abs(a - c) > BOX_TOL for a, c in zip(b.xywh, want)):
                    box_mismatch.append("{0}（{1}: {2} != {3}）".format(
                        fn, zh.get(b.name, b.name),
                        [round(v) for v in b.xywh], list(want)))
        if not names:
            background += 1
        if len(names) >= 2:
            both += 1

    rep.counts["纯背景帧"] = background
    # 按键是"两个键能不能同时按下"；选路是"同帧有没有多个路标"——换个说法
    rep.counts["同帧多框" if task == "choice" else "双键同时按下"] = both
    for c in classes:
        rep.counts["框·" + zh.get(c, c)] = per_class.get(c, 0)
    rep.counts["画面尺寸"] = "，".join("{0}×{1} 帧".format(k, v)
                                     for k, v in sorted(sizes.items()))

    if bad_label:
        rep.add("calibration", "error",
                "{0} 个标注文件**格式不合法**".format(len(bad_label)),
                "训练会读不了（或读歪）；用复核页重新导出写回一遍",
                bad_label)
    if unknown_cls:
        rep.add("calibration", "error",
                "{0} 帧里有**越界的类别 id**".format(len(unknown_cls)),
                "类别表改过？旧标注会全部错位（本数据集按 {0} 读）"
                .format("/".join(classes)), unknown_cls)
    if task == "choice":
        if out_of_band:
            rep.add("calibration", "error",
                    "{0} 个框**不在选路带里**".format(len(out_of_band)),
                    "选路图标只出现在屏幕上方那条带区（CHOICE_BAND "
                    "= 上方 400,92 起 480×80，容差 24px）。框飘到带区外，"
                    "多半是**点 band 图加框时坐标换算错了**，或者标注不是这一版写的",
                    out_of_band)
        elif per_class:
            rep.add("calibration", "ok",
                    "所有标注框都落在选路带里（框都是路标的位置，不是固定位置）")
    if task != "choice":                # ⚠️ 选路**没有**"固定位置"这回事，不套按键判据
        if box_mismatch:
            rep.add("calibration", "error",
                    "{0} 个框的位置**和当前标定对不上**".format(len(box_mismatch)),
                    "数据集是用**另一套** `vision__brake_key_box` / `nitro_key_box` "
                    "建的（或者你改过 config.json）。训练出来的模型会学错位置 —— "
                    "要么把标定改回去，要么重新 build", box_mismatch)
        elif per_class:
            rep.add("calibration", "ok",
                    "所有标注框都落在当前标定的两个按键框上（容差 {0}px）"
                    .format(BOX_TOL))

    # 分辨率：标定是按 1280x720 量的，数据集是别的尺寸 -> 框必然对不上
    odd = {k: v for k, v in sizes.items() if k not in ("1280x720",)}
    if odd:
        rep.add("calibration", "warn",
                "有非 1280×720 的帧：{0}".format(
                    "，".join("{0}({1})".format(k, v) for k, v in odd.items())),
                ("按键框是在 **1280×720** 上量的；别的分辨率下这两个框位置不对，"
                 "模型学到的位置先验会失效" if task != "choice" else
                 "选路带区也是在 **1280×720** 上量的；别的分辨率下带区和框都会错位"))
    elif sizes:
        rep.add("calibration", "ok", "全部帧都是 1280×720（和标定一致）")

    # ---- 3. 用当前判据重算（抓"阈值被改过"）----
    #
    # ⚠️ **必须把"人改的"和"真漂移"分开** —— 这条是用户复核完之后立刻暴露的：
    # 他改了 30 帧（把启发式的误报改成"没按"），体检马上报出一个红色 error
    # 「判据已经和标注不一致」。可那 30 帧**正是人干的活**，是好事 ✗✗。
    #
    # 判据：标注和当前判据不一致时，看这一帧**有没有人复核过**：
    #   * `status == "reviewed"` -> **人工修正**（预期如此，而且正是我们要的增量）；
    #   * `status == "prelabel"`  -> **真漂移**（阈值/框被改过，或有人手改了却没重跑）。
    #
    # 不这么分的话，人越认真复核、页面越红，最后就学会无视红色了。
    if recheck and metas:
        # 抽检的"当前判据"**按任务选**：按键用 `prelabel.read_signals`，
        # 选路用 `choice.prelabel` ——
        # 拿按键判据去重算选路帧，只会得到一堆毫无意义的"漂移" ✗
        #
        # ⚠️ **还要按"这批标签当初是谁给的"选后端**（2026-09-15）：
        # 用户开始"拿模型推数据集"之后，`v2` 这类数据集的标签是**模型**给的，
        # 而这里原来一律用 `read_signals(frame)`（像素启发式）重算 ——
        # 于是每一帧都会被报成"标注和当前判据不一致" ✗✗（整批假漂移）。
        # 现在看 `dataset.json` 里记的 `prelabel`：写的是哪个后端，就用哪个后端重算。
        from a9route.train.prelabel import read_signals
        _det, judge = _recheck_judge(ds)
        if progress:
            progress(f"  抽检判据：{judge}")
        # 抽样要**铺满整段**（不要只取 meta 里的前 N 行 —— 那全是第一条视频）
        picks = _spread(metas, max(1, int(limit)))
        sig_drift = []
        label_drift = []
        human_fixed = []            #: 已复核帧上和启发式不同 = 人工修正（好事）
        n_sig = 0
        for m in picks:
            fn = m.get("file")
            hit = resolve_image(ds, fn) if fn else None
            if hit is None:
                continue
            frame = cv2.imread(str(hit))
            if frame is None:
                continue
            reviewed = (m.get("status") == "reviewed")
            if task == "choice":
                # ---- 选路：信号 = (几个图标, 蓝高亮是第几个) ----
                from a9route.train import choice as C

                _cdet, _cjudge = _recheck_choice_judge(ds)
                now_boxes, icons = C.prelabel(frame, detector=_cdet)
                n_sig += 1
                now_ans = C.answers_from_boxes(now_boxes)
                old_det = m.get("detected") or {}
                sel = next((i + 1 for i, ic in enumerate(
                    sorted(icons, key=lambda s: s["xy"][0])) if ic.get("blue")), 0)
                if (int(old_det.get("count") or 0) != len(icons)
                        or int(old_det.get("selected") or 0) != sel):
                    sig_drift.append(
                        "{0}（存: {1}个选{2} → 现在: {3}个选{4}）".format(
                            fn, old_det.get("count") or 0, old_det.get("selected") or 0,
                            len(icons), sel))
                got_a = m.get("answers") or {}
                now_t = (now_ans.has, now_ans.count, now_ans.selected)
                got_t = (bool(got_a.get("has")), int(got_a.get("count") or 0),
                         int(got_a.get("selected") or 0))
                if got_t != now_t:
                    line = "{0}（标注: {1} → 现在判: {2}）".format(
                        fn, C.Answers(*got_t).describe(), now_ans.describe())
                    (human_fixed if reviewed else label_drift).append(line)
                continue
            now = read_signals(frame, detector=_det)
            n_sig += 1
            old = m.get("signals") or {}
            # 信号对比**和"有没有人复核"无关**：`signals` 存的是"建数据集那一刻"
            # 启发式的输出（人工改标注不会改它），所以两边都是启发式 ——
            # 不一致只可能是**配置被改了**。
            if (bool(old.get("brake_hit")) != now.brake_hit
                    or bool(old.get("nitro_hit")) != now.nitro_hit):
                sig_drift.append("{0}（存: 刹车{1}/氮气{2} → 现在: 刹车{3}/氮气{4}）"
                                 .format(fn, old.get("brake_hit"), old.get("nitro_hit"),
                                         now.brake_hit, now.nitro_hit))
            now_boxes = now.boxes(cues.BRAKE_KEY_BOX, cues.NITRO_KEY_BOX)
            got = m.get("boxes") or []
            if sorted(b.get("name") for b in got) != sorted(b.name for b in now_boxes):
                line = "{0}（标注: {1} → 现在判: {2}）".format(
                    fn, ",".join(sorted(b.get("name") or "?" for b in got)) or "未按",
                    ",".join(sorted(b.name for b in now_boxes)) or "未按")
                (human_fixed if reviewed else label_drift).append(line)
        rep.rechecked = n_sig
        rep.drift = {"signal_drift": len(sig_drift),
                     "label_drift": len(label_drift),
                     "human_fixed": len(human_fixed), "sampled": n_sig}
        if n_sig:
            what = "config.json 里的阈值/选路参数" if task == "choice" else \
                "`config.json` 里的阈值/框"
            if sig_drift or label_drift:
                rep.add("signals", "error",
                        "抽检 {0} 帧：**未复核帧的标注和当前判据不一致**".format(n_sig),
                        "信号漂移 {0} 帧、标注漂移 {1} 帧 —— "
                        "{2}被改过，或有人手改了标注却没重跑。"
                        "这些帧的标注**不再代表当前判据**"
                        .format(len(sig_drift), len(label_drift), what),
                        (sig_drift + label_drift))
            else:
                rep.add("signals", "ok",
                        "抽检 {0} 帧：当前判据重算的结果和**未复核帧**的标注完全一致"
                        .format(n_sig))
            if human_fixed:
                rep.add("signals", "ok",
                        "另有 {0} 帧**人工复核过**，标注和启发式不同 —— "
                        "这是**预期**的（人正是在改启发式的错）✓"
                        .format(len(human_fixed)),
                        "这些帧才是「人比启发式多知道的东西」，也是模型能超过启发式的"
                        "唯一来源；`train eval` 会拿它们当真正的答案"
                        "（报告里的 `labels_are_truth` / `reviewed_frames`）",
                        human_fixed)
    return rep


def _recheck_judge(ds: Path) -> tuple:
    """抽检该用哪个后端重算：`(detector 或 None, 人话说明)`。

    看 `dataset.json` 里的 `prelabel`（`dataset.build` / `choice.build_from` 会写）：

    * `heuristic(...)` -> `None`（用 `cues.key_pressed` 那套像素判据）；
    * `onnx:keys.onnx`  这种 -> 建对应的模型后端。

    ⚠️ **只看数据集自己记录的来源**，不看当前 `config.json` —— 否则"用户临时把后端
    切成启发式"会让整批模型标注被误报成漂移。真要检查"换成另一个后端会怎样"，
    那是 `runner.evaluate_*` 的活（它本来就并排打模型和启发式）。
    """
    try:
        from a9route.train.dataset import load_meta
        by = str((load_meta(ds) or {}).get("prelabel") or "")
    except Exception:                                  # noqa: BLE001
        by = ""
    if not by or by.startswith("heuristic"):
        return None, by or "heuristic(默认)"
    # 模型标的：按当前配置把模型后端建起来（模型文件换了 = 真漂移，正是要报的）
    try:
        from a9route.vision import keys as K
        be, mp, _ = K.resolve_backend()
        if be == "heuristic":
            return None, ("{0}（但当前配置是 heuristic —— 两边不同源，"
                          "下面的不一致要按这个前提看）".format(by))
        return K.build_key_detector(), "{0}（数据集记录的来源：{1}）".format(be, by)
    except Exception as exc:                           # noqa: BLE001
        return None, "取不到模型后端（{0}: {1}）—— 退回启发式重算".format(
            type(exc).__name__, exc)


def _recheck_choice_judge(ds: Path) -> tuple:
    """选路版：按 `dataset.json.prelabel` 决定重算用模型还是 HoughCircles。"""
    try:
        from a9route.train.dataset import load_meta
        by = str((load_meta(ds) or {}).get("prelabel") or "")
    except Exception:                                  # noqa: BLE001
        by = ""
    if not by or by.startswith("heuristic"):
        return None, by or "heuristic(默认)"
    try:
        from a9route.vision import choice as VC
        be, mp, _ = VC.resolve_backend()
        if be == "heuristic":
            return None, "{0}（但当前配置是 heuristic）".format(by)
        return VC.build_choice_detector(), "{0}（数据集记录的来源：{1}）".format(be, by)
    except Exception as exc:                           # noqa: BLE001
        return None, "取不到选路后端（{0}: {1}）—— 退回 HoughCircles 重算".format(
            type(exc).__name__, exc)


def _count(lbl: Path) -> list:
    """一个标注文件里有几行（只用来在报告里说"两份差多少"）。"""
    try:
        return [ln for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return []


def _spread(items: list, n: int) -> list:
    """从列表里**均匀**取 n 个（首尾都取到）。"""
    if n >= len(items):
        return list(items)
    if n <= 1:
        return [items[len(items) // 2]]
    last = len(items) - 1
    return [items[min(last, int(round(i * last / float(n - 1))))] for i in range(n)]


# ---------------------------------------------------------------- 逐帧浏览
def frame_rows(name: str = "keys", *, root=None, split: str = "", reason: str = "",
               cls: str = "", status: str = "", only: str = "",
               search: str = "", offset: int = 0, limit: int = 24,
               recheck: bool = False, scan_cap: int = 4000,
               progress=None) -> dict:
    """逐帧浏览用的一页数据（给 Web 界面）。

    `only="problem"`：只看"值得人眼确认"的帧（标注和当前判据不一致、
    框和标定对不上）—— 这会**自动打开 `recheck`**，因为不重算就判不出来。

    `recheck=True` 时只对**这一页真正要返回的帧**重算当前判据
    （逐帧解图 + 跑判据，约 3~6 ms/帧），所以翻到哪算到哪，不会因为
    数据集大就卡住。`only` 过滤是在重算之后做的，所以这里**边扫描边收集**，
    凑够一页就停（`scan_cap` 是安全上限，防止整页都是问题帧时扫穿整个数据集）。
    """
    import cv2

    from a9route import config as cfgmod

    cfgmod.apply()
    ds = Path(root) if root else dataset_dir(name)
    if not ds.is_dir():
        raise FileNotFoundError("数据集不存在：{0}".format(ds))
    # ⚠️ 这个浏览页是**按键专用**的（左边裁两个按键框、右边画刹车/氮气的标注）。
    # 选路数据集套进来会整页错位（第 0 类会被读成"刹车按下"）—— 明说、
    # 指到正确的页面，而不是显示一堆看不懂的框 ✗
    if _task_of(ds, name)[0] == "choice":
        raise ValueError(
            "数据集 `{0}` 是**选路**数据集，不是刹车/氮气 —— 请用 "
            "http://127.0.0.1:8790/choice 标注它（那边的三个选择题和这里不是一套）"
            .format(name))
    metas = read_frames_meta(ds)
    cal = _calibration_now()
    box_of = {"brake_pressed": cal["brake_key_box"],
              "nitro_pressed": cal["nitro_key_box"]}

    def label_box_mismatch(labels: list) -> bool:
        """**不需要解图**就能判的：标注框是不是落在当前标定框上。"""
        for b in labels:
            want = box_of.get(b["name"])
            if want and any(abs(a - c) > BOX_TOL for a, c in zip(b["box"], want)):
                return True
        return False

    rows = []
    for m in metas:
        fn = m.get("file")
        if not fn:
            continue
        labels = [{"cls": int(b.get("cls") or 0), "name": b.get("name") or "",
                   "zh": CLASS_ZH.get(b.get("name") or "", ""),
                   "box": [float(v) for v in (b.get("xywh") or [0, 0, 0, 0])],
                   "conf": b.get("conf")}
                  for b in (m.get("boxes") or [])]
        rows.append({
            "file": fn, "split": _split_of_file(ds, fn) or "?",
            "video": m.get("video") or "", "t": float(m.get("t") or 0.0),
            "frame": int(m.get("frame") or 0),
            "reason": m.get("reason") or "bg",
            "status": m.get("status") or "prelabel",
            "width": int(m.get("width") or 0), "height": int(m.get("height") or 0),
            "labels": labels, "signals": m.get("signals") or {},
            "flags": (["calib_mismatch"] if label_box_mismatch(labels) else []),
        })

    # ---- 只用元数据就能做的过滤 ----
    if split and split != "all":
        rows = [r for r in rows if r["split"] == split]
    if reason and reason != "all":
        rows = [r for r in rows if r["reason"] == reason]
    if status and status != "all":
        rows = [r for r in rows if r["status"] == status]
    if cls and cls != "all":
        rows = [r for r in rows if any(b["name"] == cls for b in r["labels"])]
    if search:
        q = search.lower()
        rows = [r for r in rows if q in r["file"].lower()
                or q in (r["video"] or "").lower()]

    want_problem_only = (only == "problem")
    if want_problem_only:
        recheck = True          # 不重算就判不出"问题帧"

    start = max(0, int(offset))
    limit = max(1, int(limit))
    need = limit
    page: list = []
    consumed = 0
    scanned = 0
    #: 逐帧页的"重算"用哪个后端 —— 同样按数据集自己记录的来源（见 `_recheck_judge`），
    #: 否则模型标的帧会被像素判据判成"漂移"，整页都是假警报
    _recheck_det, _judge_name = _recheck_judge(ds)
    if progress and recheck:
        progress("  逐帧重算判据：{0}".format(_judge_name))

    def _recheck_row(r: dict) -> None:
        """对一帧重算当前判据，写进 `r["now"]` 并补 `flags`。

        ⚠️ 重算用的后端**必须和这批标签的来源一致**（见 `_recheck_judge`）：
        数据集的标签是模型给的（`prelabel: onnx:keys.onnx`），这里却用像素启发式
        去重算的话，**每一页都会标满"标注漂移"** ✗✗ —— 而人正对着这一页标定。
        """
        from a9route.train.prelabel import read_signals
        r["now"] = None
        p = resolve_image(ds, r["file"], r["split"])
        if p is None:
            if "missing_file" not in r["flags"]:
                r["flags"].append("missing_file")
            return
        frame = cv2.imread(str(p))
        if frame is None:
            if "undecodable" not in r["flags"]:
                r["flags"].append("undecodable")
            return
        now = read_signals(frame, detector=_recheck_det)
        now_boxes = now.boxes(cues_box("brake"), cues_box("nitro"))
        r["now"] = {
            "brake": now.brake_hit, "nitro": now.nitro_hit,
            "brake_margin": round(now.brake_margin, 3),
            "nitro_margin": round(now.nitro_margin, 3),
            "boxes": [{"name": b.name, "zh": CLASS_ZH.get(b.name, ""),
                       "box": [round(v, 1) for v in b.xywh]} for b in now_boxes],
            "signals": now.as_dict(),
        }
        # ⚠️ `now.boxes()` 返回的是 `Box` 对象（不是 dict）——
        # 用 `b["name"]` 会 TypeError。标注那边是 dict（来自 meta.jsonl），
        # 两边形式不一样，这里显式取属性。
        if {b.name for b in now_boxes} != {b["name"] for b in r["labels"]}:
            # **人复核过**的帧和启发式不同 = 正常（人就是在改它的错）→ 换个标记，
            # 别打成"漂移"的橙/红标，否则复核越多、页面越花 ✗
            r["flags"].append("human_fix" if r["status"] == "reviewed"
                              else "label_drift")
        old = r["signals"]
        if (bool(old.get("brake_hit")) != now.brake_hit
                or bool(old.get("nitro_hit")) != now.nitro_hit):
            r["flags"].append("signal_drift")

    if want_problem_only:
        # 边扫边收：凑够一页就停。**只认真问题**（PROBLEM_FLAGS），
        # `human_fix`（人工修正过）不算 —— 那是成果不是问题。
        i = start
        while i < len(rows) and len(page) < need and scanned < scan_cap:
            r = rows[i]
            i += 1
            scanned += 1
            consumed += 1
            _recheck_row(r)
            if any(fl in PROBLEM_FLAGS for fl in r["flags"]):
                page.append(r)
        next_offset = i
        total = None                    # 问题帧总数要扫完才知道，不谎报
    else:
        page = rows[start:start + need]
        consumed = len(page)
        next_offset = start + consumed
        total = len(rows)
        if recheck:
            for r in page:
                _recheck_row(r)

    reasons: dict = {}
    statuses: dict = {}
    for m in metas:
        reasons[m.get("reason") or "bg"] = reasons.get(m.get("reason") or "bg", 0) + 1
        statuses[m.get("status") or "prelabel"] = statuses.get(
            m.get("status") or "prelabel", 0) + 1
    return {
        "dataset": name, "root": str(ds), "all_frames": len(metas),
        "matched": len(rows), "total": total, "total_known": total is not None,
        "offset": start, "next_offset": next_offset, "limit": limit,
        "rows": page, "calibration": cal,
        "facet": {"reasons": reasons, "statuses": statuses,
                  "reasons_zh": F.REASON_ZH, "classes": list(CLASSES),
                  "classes_zh": CLASS_ZH},
        "recheck": bool(recheck), "scanned": scanned,
        "meta": load_meta(ds),
    }


def cues_box(which: str):
    from a9route.vision import cues
    return cues.BRAKE_KEY_BOX if which == "brake" else cues.NITRO_KEY_BOX


def _split_of_file(ds: Path, file: str) -> str:
    for sp, _img_dir, _lbl in split_dirs(ds):
        if resolve_image(ds, file, sp) is not None:
            return sp
    return ""


def datasets() -> list:
    """列出所有数据集（`datasets/<name>/`）。"""
    root = paths.DATASETS_DIR
    out = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        meta = load_meta(d)
        n = 0
        mp = d / FRAME_META
        if mp.is_file():
            try:
                n = sum(1 for _ in mp.open(encoding="utf-8"))
            except OSError:
                n = 0
        out.append({"name": d.name, "path": str(d), "frames": n,
                    #: `keys` / `choice` —— 界面上要标出来，还要挡住"在按键页打开选路数据集"
                    "task": _task_of(d, d.name)[0],
                    "videos": meta.get("videos") or [],
                    "created": meta.get("created") or "",
                    "updated": meta.get("updated") or "",
                    "reviewed": int(meta.get("reviewed_frames") or 0),
                    "has_pool": (d / POOL_DIR / "images").is_dir(),
                    "has_split": (d / "data.yaml").is_file()})
    return out


def crop_preview(name: str, file: str, *, which: str = "brake", zoom: int = 2,
                 pad: int = 10, root=None):
    """把某个按键框那一小块**放大**返回（BGR ndarray）—— 复核时真正要看的就是它。

    在**服务端**裁而不是前端用 CSS 摆：这样"看到的像素"和"判据读的像素"
    一定是同一块（前端算缩放/偏移很容易差几个像素，而差几个像素就看不出
    "透明 vs 半透明白"的差别了）。
    """
    import cv2

    ds = Path(root) if root else dataset_dir(name)
    p = resolve_image(ds, file)
    if p is None:
        return None
    frame = cv2.imread(str(p))
    if frame is None:
        return None
    box = cues_box(which)
    x, y, w, h = [int(v) for v in box]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    z = max(1, min(6, int(zoom)))
    return cv2.resize(patch, (patch.shape[1] * z, patch.shape[0] * z),
                      interpolation=cv2.INTER_NEAREST)


def full_preview(name: str, file: str, *, width: int = 480, root=None):
    """整帧缩略图（BGR ndarray）。"""
    import cv2

    ds = Path(root) if root else dataset_dir(name)
    p = resolve_image(ds, file)
    if p is None:
        return None
    frame = cv2.imread(str(p))
    if frame is None:
        return None
    if frame.shape[1] > width:
        sc = width / float(frame.shape[1])
        frame = cv2.resize(frame, (width, max(1, int(frame.shape[0] * sc))),
                           interpolation=cv2.INTER_AREA)
    return frame
