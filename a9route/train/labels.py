# -*- coding: utf-8 -*-
"""train.labels —— **类别定义 + YOLO 标注文件的读写**（训练框架的地基）。

## 类别怎么定的（只有两类）

| id | 名字 | 含义 |
|---|---|---|
| 0 | `brake_pressed` | 左下**刹车键正处于"按下"外观** |
| 1 | `nitro_pressed` | 右下**氮气键正处于"按下"外观** |

**没有"未按下"这个类** —— 框不存在 = 没按下。这样类别数最少、
正负样本天然平衡（绝大多数帧是空标注），而且和现有一切接口都对得上：
`scan_fine()` 要的就是每帧两个布尔值。

### ⚠️ 两个必须守住的语义（NOTES.md 里踩过两次的坑）

1. **"按下"看的是按键图标本身**，不是"氮气在喷"、也不是"瓶子红=已充满"。
   用户原话：**"图标没点是透明的，按下变成部分白色不透明"**。
   所以标注时只认那个圆盘的**点亮/不透明**外观 —— 顶部氮气槽变青、瓶子变红
   都**不算**（实测只有 65% 一致 ✗，见 NOTES §1）。
2. **刹车键和 360/漂移是同一个键**：360 = 双击、漂移 = 长按，三种都是"按下"。
   标注**不区分** —— 判"这次按下是为了什么"是下游 `core/intent.py` 的事
   （按信号形状判），不该让检测模型去猜目的。

### 想加类的话

`CLASSES` 是唯一的类别来源（`data.yaml`、标注文件、运行时后端都读它）。
最值得加的是第三类 `nitro_ready`（瓶子充满变红 = 可用但没按）——
它能把"氮气误报偏多"这个已知问题从根上分开。加的时候**必须**：
`CLASSES` 里顺序追加（**不能插在中间**，否则旧标注全部错位）+ 重跑
`a9route train prelabel --rewrite`。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: 类别名（顺序 = YOLO 的 class id，**只能往后追加**）
CLASSES: tuple[str, ...] = ("brake_pressed", "nitro_pressed")

#: 名字 -> id
CLASS_IDS: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

#: 中文短名（复核页/日志里给人看的）
CLASS_ZH: dict[str, str] = {
    "brake_pressed": "刹车按下",
    "nitro_pressed": "氮气按下",
}

#: 每类的主色（BGR，复核页画框用）—— 和项目别处的配色习惯一致
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "brake_pressed": (0, 220, 255),    # 黄
    "nitro_pressed": (60, 60, 245),    # 红
}


class LabelError(ValueError):
    """标注文件不合法（**带行号**，别只丢一个 ValueError 让人自己找）。"""


@dataclass
class Box:
    """一个目标框：**像素坐标**的左上角 + 宽高（训练/绘图时最顺手的形式）。

    YOLO 的文件格式是**归一化的中心点**，两者在这里来回换 ——
    像素坐标是"人能核对"的形式（和 `config.json` 里的
    `brake_key_box=(137,497,110,110)` 直接对得上），所以对外一律用像素。
    """

    cls_id: int
    x: float
    y: float
    w: float
    h: float
    #: 置信度（预标注/推理时有；人工标注的框是 None）
    conf: float | None = None
    #: 这个框属于**哪一套类别名**。空 = 用 `labels.CLASSES`（刹车/氮气那套）。
    #: 选路是**另一套**（`choice_icon` / `choice_selected`）—— 框必须自带它属于
    #: 哪一套，否则 `Box.name` 会拿错名字（选路的框被读成"刹车按下"）。
    names: tuple = ()

    @property
    def name(self) -> str:
        names = self.names or CLASSES
        return names[self.cls_id] if 0 <= self.cls_id < len(names) else f"?{self.cls_id}"

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        """`(x, y, w, h)`（和 `config.json` 里 `brake_key_box` 的写法一致）。"""
        return (self.x, self.y, self.w, self.h)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def describe(self) -> str:
        c = "" if self.conf is None else f" conf={self.conf:.2f}"
        return (f"{self.name} ({self.x:.0f},{self.y:.0f},{self.w:.0f},"
                f"{self.h:.0f}){c}")

    def clip(self, img_w: int, img_h: int) -> "Box":
        """裁到画面里（预标注的框就是配置里那个固定框，边缘可能正好压线）。"""
        x1 = min(max(0.0, self.x), float(img_w))
        y1 = min(max(0.0, self.y), float(img_h))
        x2 = min(max(0.0, self.x + self.w), float(img_w))
        y2 = min(max(0.0, self.y + self.h), float(img_h))
        return Box(self.cls_id, x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1),
                   self.conf, self.names)


# ---------------------------------------------------------------- YOLO txt 格式
def box_to_line(box: Box, img_w: int, img_h: int) -> str:
    """`Box` -> YOLO 标注的一行（`cls cx cy w h`，**归一化到 0~1**）。

    YOLO 要求 4 位小数够用（1 像素在 1280 宽上 = 0.0008，4 位不会丢信息）。
    """
    if img_w <= 0 or img_h <= 0:
        raise LabelError(f"画面尺寸不合法：{img_w}x{img_h}")
    cx, cy = box.center
    return (f"{int(box.cls_id)} {cx / img_w:.6f} {cy / img_h:.6f} "
            f"{box.w / img_w:.6f} {box.h / img_h:.6f}")


def line_to_box(line: str, img_w: int, img_h: int, *, where: str = "",
                classes=None) -> Box:
    """YOLO 标注的一行 -> `Box`（像素）。**坏行带行号报错，不静默跳过。**

    （静默跳过是这类脚本最容易埋的坑：标歪了一行，训练照跑，指标悄悄变差。）

    `classes`：这一行属于**哪套类别**（默认刹车/氮气那套；选路传 `CHOICE_CLASSES`）。
    类名会存进 `Box.names`，所以读回来之后 `box.name` 是对的。
    """
    names = tuple(classes) if classes else CLASSES
    parts = line.split()
    if len(parts) != 5:
        raise LabelError(f"{where}标注要有 5 个数（cls cx cy w h），"
                         f"收到 {len(parts)} 个：{line!r}")
    try:
        vals = [float(p) for p in parts]
    except ValueError as exc:
        raise LabelError(f"{where}标注里有非数字：{line!r}") from exc
    cls_f, cx, cy, w, h = vals
    cls_id = int(cls_f)
    if cls_id != cls_f:
        raise LabelError(f"{where}class id 必须是整数，收到 {cls_f!r}")
    if not 0 <= cls_id < len(names):
        raise LabelError(f"{where}class id {cls_id} 越界"
                         f"（当前只有 {len(names)} 类：{', '.join(names)}）")
    for nm, v in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
        if not 0.0 <= v <= 1.0:
            raise LabelError(f"{where}{nm}={v} 不在 0~1 之间（YOLO 是归一化坐标）")
    if w <= 0 or h <= 0:
        raise LabelError(f"{where}框的宽高必须 > 0（w={w}, h={h}）")
    return Box(cls_id, (cx - w / 2) * img_w, (cy - h / 2) * img_h,
               w * img_w, h * img_h, names=names)


def read_label(path: str | Path, img_w: int, img_h: int, *,
               classes=None) -> list[Box]:
    """读一个标注文件。**文件不存在 = 空列表**（YOLO 的"纯背景帧"就是这么表示的）。"""
    p = Path(path)
    if not p.is_file():
        return []
    out: list[Box] = []
    for i, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(line_to_box(s, img_w, img_h, where=f"{p.name}:{i} ",
                               classes=classes))
    return out


def label_text(boxes: list[Box], img_w: int, img_h: int) -> str:
    """标注文件**应该长什么样**（只算字符串，不落盘）。

    用来判断"这次写回到底改没改内容" —— 见 `write_label_if_changed`。
    """
    lines = [box_to_line(b.clip(img_w, img_h), img_w, img_h) for b in boxes]
    return "".join(ln + "\n" for ln in lines)


def write_label(path: str | Path, boxes: list[Box], img_w: int, img_h: int) -> Path:
    """写标注文件（**空框列表写空文件** —— 空文件才是 YOLO 认的"背景帧"，
    文件不存在会被当成"这张图没标"，两者语义不同）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(label_text(boxes, img_w, img_h), encoding="utf-8")
    return p


def write_label_if_changed(path: str | Path, boxes: list[Box], img_w: int,
                           img_h: int) -> tuple[Path, bool]:
    """内容真的不同才落盘，返回 `(路径, 到底写没写)`。

    ⚠️ 为什么不能无脑重写：界面上的「改动 N 帧」是按这个计数报给人看的 ——
    内容一个字节没变也报"改了 1 帧"，就是在**说假话**；
    更麻烦的是"翻页即复核"会把**整页没动过的帧**一起提交，
    无脑重写会让撤销点被这种空提交顶掉 —— 人明明什么都没改，
    「撤销」却回不到他上一次真正的改动 ✗（按键那边踩过同一个坑，见 NOTES 10.15）。
    """
    p = Path(path)
    new = label_text(boxes, img_w, img_h)
    old = p.read_text(encoding="utf-8") if p.is_file() else None
    if old == new:
        return p, False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(new, encoding="utf-8")
    return p, True


# ---------------------------------------------------------------- data.yaml
@dataclass
class DatasetInfo:
    """一个 YOLO 数据集的现况（`stats`/`verify`/`doctor` 都用它）。"""

    root: Path
    images: int = 0
    labels: int = 0
    boxes_by_class: dict[str, int] = field(default_factory=dict)
    frames_by_class: dict[str, int] = field(default_factory=dict)
    background: int = 0            #: 空标注帧（= 两个键都没按）
    multi: int = 0                 #: 同帧有 2 个框（刹车+氮气一起按）
    sizes: dict[str, int] = field(default_factory=dict)
    sources: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def positives(self) -> int:
        return sum(self.boxes_by_class.values())

    def describe(self) -> str:
        lines = [f"数据集: {self.root}",
                 f"  帧 {self.images}（标注文件 {self.labels}，"
                 f"纯背景 {self.background}，双键同时 {self.multi}）"]
        for name in CLASSES:
            n_box = self.boxes_by_class.get(name, 0)
            n_frame = self.frames_by_class.get(name, 0)
            lines.append(f"  {name:<16} {CLASS_ZH.get(name, ''):<8} "
                         f"框 {n_box:>6}  帧 {n_frame:>6}")
        if self.sizes:
            top = sorted(self.sizes.items(), key=lambda kv: -kv[1])[:4]
            lines.append("  画面尺寸 " + "，".join(f"{k}×{v} 帧" for k, v in top))
        if self.sources:
            lines.append(f"  来源视频 {len(self.sources)} 段：" +
                         "，".join(f"{k}({v})" for k, v in
                                   sorted(self.sources.items(), key=lambda kv: -kv[1])[:6]))
        if self.problems:
            lines.append(f"  ⚠️ {len(self.problems)} 个问题")
            for p in self.problems[:10]:
                lines.append(f"    - {p}")
        return "\n".join(lines)


def data_yaml(layout_root: Path, *, extra: dict | None = None,
              classes=None) -> dict:
    """生成 YOLO 的 `data.yaml` 内容（ultralytics 直接吃这个 dict）。"""
    root = Path(layout_root).resolve()
    names = tuple(classes) if classes else CLASSES
    out = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
    }
    if extra:
        out.update(extra)
    return out


def _yaml_scalar(v) -> str:
    """把简单值写成 YAML（**不 import pyyaml**）。

    为什么不用 `yaml.safe_dump`：pyyaml 在 `.[train]` 这个 extra 里，
    而 `a9route test all` 的设计目标是 **在只装了运行依赖的环境里也能全绿** ——
    这里以前在函数开头 `import yaml`，于是没装 pyyaml 的干净 clone 一跑测试
    就在这个函数上炸（2026-09-15 复查 clone 流程时发现）✗。
    真正需要的只有 str / 数 / bool / 数的列表，手写比引依赖划算。
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_yaml_scalar(x) for x in v) + "]"
    s = str(v)
    # 需要引号的情况：空串、含特殊字符、看起来像数/布尔
    if not s or s.strip() != s or any(c in s for c in ":#,[]{}&*!|>'\"%@`\n"):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def write_data_yaml(path: str | Path, layout_root: Path, *,
                    extra: dict | None = None, classes=None,
                    zh: dict | None = None) -> Path:
    """写 `data.yaml`。**不用 yaml.safe_dump 的默认排序** —— 手写一个稳定的格式，
    这样 git diff 干净、人也能直接读。

    `classes`：这套数据集用哪套类别（默认刹车/氮气；选路传 `CHOICE_CLASSES`）。
    """
    names = tuple(classes) if classes else CLASSES
    zh_map = dict(CLASS_ZH if zh is None else zh)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = data_yaml(layout_root, extra=extra, classes=names)
    body = [
        "# 由 `a9route train split` 自动生成 —— 不要手改（改 dataset 配置请用 CLI 参数）",
        f"path: {data['path']}",
        f"train: {data['train']}",
        f"val: {data['val']}",
        "",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, name in enumerate(names):
        z = zh_map.get(name, "")
        body.append(f"  {i}: {name}" + (f"    # {z}" if z else ""))
    for k, v in (extra or {}).items():
        if k in ("path", "train", "val", "names", "nc"):
            continue
        body.append(f"{k}: {_yaml_scalar(v)}")
    p.write_text("\n".join(body) + "\n", encoding="utf-8")
    return p
