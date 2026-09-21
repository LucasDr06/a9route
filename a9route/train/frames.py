# -*- coding: utf-8 -*-
"""train.frames —— **抽哪些帧来标**（不是"每隔 N 帧抽一张"那么简单）。

## 为什么不能均匀抽帧

按键的难点全在**边界**上，均匀抽帧恰好把边界全躲开了：

1. **单击可能只有 1 帧**（33ms）。NOTES §4 记着：用户 17% 那两次氮气点击
   在 59 帧里只亮 2 帧 —— 均匀每 30 帧抽一张，永远抽不到它。
2. **误报长在阈值附近**：`红占比 0.151 > 0.15` 就判"按下"，
   这种帧和"没按"（0.09）在人眼里长得差不多，**只有它才是要教模型的东西**。
3. **两个通道吵架的帧**才是"瓶子满面红但其实没按"这类已知错误的老家。

所以这里按**优先级**抽帧（`REASONS`），把预算砸在模型最需要学的帧上：

| 优先级 | reason | 抽什么 | 上限 | 为什么 |
|---|---|---|---|---|
| 1 | `edge` | 按键状态**翻转帧** ±`edge_pad` 帧 | ⚠️ `edge_share`（默认 60%） | 单击/双击/长按的分界全在这儿；翻转前那一帧是最难的负样本 |
| 2 | `disagree` | 两通道判断**不一致**的帧 | 25% | 已知误报（瓶子红≠按下）就在这里 |
| 3 | `near` | 归一化余量落在 `1 ± margin_band` 的帧 | 35% | 悬在阈值上的帧 —— 标对了模型才学得会"到底按没按" |
| 4 | `pos` | 其余**正样本**帧随机 | 剩余 | 不全是困难样本，否则数据分布偏了 |
| 5 | `bg` | 均匀背景帧（含加载画面/明暗剧变） | 剩余 | 负样本；模型得学会"没有按钮时什么都别输出" |

> ⚠️ **翻转帧也必须设上限** —— 这一条是真数据打脸后加的：启发式在阈值附近会抖，
> 一台机器上三段真录像 4766 帧里，"离某个翻转 ≤3 帧"的帧有 1253 个，
> 不设限就会把预算全吃光（第一版实测：1341 帧里 1253 帧是 edge、
> `pos`/`bg` 一个都没抽到）。上限之外的翻转帧**沿时间均匀取样**，
> 保住整段覆盖，而不是只留视频开头那批。

> 每张帧都记下 `reason` —— 复核页按它排序，**先让人看 `edge`/`disagree`**。

## 两遍解码

第一遍只算信号（每帧几个 float，一分钟录像也就几 MB），
第二遍才按选中的帧号存 JPEG。**不做单遍**是因为"预算"和"翻转帧"都要
看完整段才能定，单遍就得把帧全缓存下来（10 分钟 720p ≈ 50 GB ✗）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from a9route.train.prelabel import KeySignals, read_signals

#: 抽样原因（**顺序 = 优先级**，数字越小越先占预算）
REASONS: tuple[str, ...] = ("edge", "disagree", "near", "pos", "bg")

#: 复核时的排序权重（越小越该先看）—— 和 `REASONS` 顺序一致
REASON_ORDER: dict[str, int] = {r: i for i, r in enumerate(REASONS)}

REASON_ZH: dict[str, str] = {
    "edge": "翻转帧（按键状态变化处）",
    "disagree": "两通道吵架",
    "near": "贴着阈值",
    "pos": "普通正样本",
    "bg": "背景/负样本",
}


@dataclass
class FrameSignal:
    """一帧的信号（第一遍解码的产物）。"""

    idx: int
    t: float
    sig: KeySignals
    reason: str = ""

    @property
    def positive(self) -> bool:
        return self.sig.brake_hit or self.sig.nitro_hit


@dataclass
class SamplePlan:
    """抽样计划（第一遍解码的结论）。"""

    picks: list[FrameSignal] = field(default_factory=list)
    total_frames: int = 0
    fps: float = 30.0
    width: int = 0
    height: int = 0
    #: 各 reason 实际抽了多少帧
    counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        bits = "，".join(f"{r} {self.counts.get(r, 0)}" for r in REASONS
                         if self.counts.get(r))
        return (f"{self.total_frames} 帧里抽 {len(self.picks)} 帧"
                f"（{bits}）")


def slug(name: str) -> str:
    """把视频名变成**纯 ASCII** 的文件名前缀。

    ⚠️ 这不是"好看"的问题：`cv2.imwrite()` 遇到中文路径**静默失败**
    （不抛异常、只返回 False，NOTES §2 踩过）。视频名多半是中文
    （`跑图转路线测试.mp4`），所以**一律转成 ASCII**，中文原名记在 meta 里。
    """
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")
    if not s:
        # 全中文的名字（`跑图转路线测试`）会被清空 —— 用短哈希兜底，
        # 至少保证"不同视频不撞名"
        import hashlib
        s = "v" + hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:8]
    return s[:40]


# ---------------------------------------------------------------- 第一遍：算信号
def scan_signals(path: str | Path, *, max_seconds: float | None = None,
                 every: int = 1, progress=None, detector=None) -> tuple[list[FrameSignal], dict]:
    """逐帧算信号。**`every>1` 会漏掉单帧点击** —— 默认每帧都看（`every=1`）。

    `detector`：给了就用**模型**判"按下/没按"（用户 2026-09-15 的用法：
    拿训好的模型推新数据集）—— 抽样优先级（吵架/贴阈值）也随模型走，
    于是"模型最没把握的帧"会优先被抽出来给人看 ✓

    ⚠️ 时间用 `i / fps` 而不是 `CAP_PROP_POS_MSEC`：这里是顺序解码，
    两者等价；但**不要**在这里引入 seek（NOTES §3：请求时间 ≠ 解码时间）。
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    dt = 1.0 / fps if fps else 1 / 30.0
    limit = int(max_seconds * fps) if max_seconds else None
    out: list[FrameSignal] = []
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if limit is not None and i >= limit:
                break
            if i % max(1, every) == 0:
                out.append(FrameSignal(idx=i, t=round(i * dt, 3),
                                       sig=read_signals(frame, detector=detector)))
            i += 1
            if progress and i and i % 3000 == 0:
                progress(f"    读信号 {i} 帧（{i * dt:.0f}s）…")
    finally:
        cap.release()
    return out, {"fps": fps, "width": width, "height": height, "total": i}


# ---------------------------------------------------------------- 第二遍：选帧
def _edges(frames: list[FrameSignal], which: str) -> list[int]:
    """按键布尔翻转处的**帧下标**（在 `frames` 列表里的位置，不是帧号）。"""
    get = (lambda f: f.sig.brake_hit) if which == "brake" else (lambda f: f.sig.nitro_hit)
    out = []
    for k in range(1, len(frames)):
        if get(frames[k]) != get(frames[k - 1]):
            out.append(k)
    return out


def _worst_margin(sig: KeySignals) -> float:
    """这一帧离阈值最近的那个通道的余量（越接近 1 越可疑）。"""
    return min(abs(sig.brake_margin - 1.0), abs(sig.nitro_margin - 1.0))


def _balanced_pick(cands: list[FrameSignal], n: int, *, seed: int = 0
                   ) -> list[FrameSignal]:
    """从候选里**随机但确定**地取 n 个（同 seed 同参数 -> 同数据集，可复现）。

    随机而不是取前 n 个：候选按帧号排的，取前 n 个会全部集中在视频开头
    （背景/赛段分布就偏了）。
    """
    import random

    if n <= 0 or not cands:
        return []
    if len(cands) <= n:
        return list(cands)
    rng = random.Random(seed)
    return sorted(rng.sample(cands, n), key=lambda f: f.idx)


def _spread_pick(cands: list[FrameSignal], n: int) -> list[FrameSignal]:
    """均匀取 n 个（背景帧用：要的是**整段时间的覆盖面**，不是随机）。

    **首尾都取到**（用 `i*(len-1)/(n-1)` 而不是 `i*len/n`）——
    后者永远够不到最后一个候选帧，"整段覆盖"就变成了"覆盖到倒数第几个"。
    """
    if n <= 0 or not cands:
        return []
    if len(cands) <= n:
        return list(cands)
    if n == 1:
        return [cands[len(cands) // 2]]
    last = len(cands) - 1
    return [cands[min(last, int(round(i * last / float(n - 1))))] for i in range(n)]


def plan_sampling(frames: list[FrameSignal], *, budget: int = 3000,
                  edge_pad: int = 3, margin_band: float = 0.25,
                  edge_share: float = 0.6, bg_share: float = 0.2,
                  max_disagree: int | None = None,
                  max_near: int | None = None,
                  seed: int = 0) -> SamplePlan:
    """按优先级分配 `budget` 个帧位置。返回的 `picks` 按帧号排序。

    ## 为什么**翻转帧也要设上限**（真数据上量出来的）

    第一版让翻转帧不设限（"它最重要，随便抽"），结果在这台机器的三段真录像上：

        1146 帧里抽出 448 帧，**全是 edge**；另外两段同理（1341 帧里 edge 1253）

    原因很实在：**启发式在边界上会抖**（氮气红占比在阈值上下跳），
    一次按下可能产生好几个"翻转"，每个翻转 ±3 帧的窗口连起来就盖住了整段视频。
    这样 `reason` 就退化成"什么都是 edge"，`pos`/`bg` 一个都抽不到 ——
    数据集的分布反而偏了，而模型最后是要在**普通帧**上跑全片的。

    ## 为什么还要给 `bg` **预留**位置（`bg_share`）

    光限制 `edge` 还不够：`pos` 的候选是"还没被选中的正样本"，它会把剩下的预算
    全吃掉，`bg` 照样是 0（实测：抖动的信号 + 预算 40 -> edge 24 + pos 16 + bg 0）。
    所以给 `bg` 预留 `bg_share`（默认 20%）—— 负样本/背景帧是这个任务的另一半，
    抽不到的话模型就学不会"没有按钮时什么都别输出"。
    """
    if not frames:
        return SamplePlan()
    budget = max(1, int(budget))
    max_edge = max(1, int(budget * max(0.1, min(0.95, edge_share))))
    reserve_bg = int(budget * max(0.0, min(0.5, bg_share)))
    max_disagree = budget // 4 if max_disagree is None else int(max_disagree)
    max_near = int(budget * 0.35) if max_near is None else int(max_near)

    picked: dict[int, FrameSignal] = {}

    def take(f: FrameSignal, reason: str) -> None:
        """登记一个帧：**已有更高优先级的 reason 就不覆盖**。"""
        cur = picked.get(f.idx)
        if cur is not None:
            if REASON_ORDER[reason] < REASON_ORDER[cur.reason]:
                cur.reason = reason
            return
        picked[f.idx] = FrameSignal(idx=f.idx, t=f.t, sig=f.sig, reason=reason)

    def room(*, keep_bg: bool) -> int:
        """还剩多少位置（`keep_bg=True` 时扣掉给背景帧预留的那份）。"""
        left = budget - len(picked)
        return max(0, left - (reserve_bg if keep_bg else 0))

    # ---- 1. 翻转帧 ±edge_pad（最高优先级；超过 edge_share 就沿时间均匀取样）
    edge_idx: set[int] = set()
    for which in ("brake", "nitro"):
        for k in _edges(frames, which):
            for j in range(max(0, k - edge_pad), min(len(frames), k + edge_pad + 1)):
                edge_idx.add(j)
    edge_frames = [frames[j] for j in sorted(edge_idx)]
    for f in _spread_pick(edge_frames, min(max_edge, budget)):
        take(f, "edge")

    # ---- 2. 两通道吵架
    dis = [f for f in frames
           if f.sig.brake_disagree or f.sig.nitro_disagree
           if f.idx not in picked]
    for f in _balanced_pick(dis, min(max_disagree, room(keep_bg=True)), seed=seed):
        take(f, "disagree")

    # ---- 3. 贴着阈值（按"离 1 最近"排序取，不是随机 —— 越近越该看）
    near = [f for f in frames if _worst_margin(f.sig) <= margin_band
            and f.idx not in picked]
    near.sort(key=lambda f: _worst_margin(f.sig))
    for f in near[:max(0, min(max_near, room(keep_bg=True)))]:
        take(f, "near")

    # ---- 4. 其余正样本（随机；**给背景帧留够**）
    pos = [f for f in frames if f.positive and f.idx not in picked]
    for f in _balanced_pick(pos, room(keep_bg=True), seed=seed + 1):
        take(f, "pos")

    # ---- 5. 背景帧（均匀铺满整段，把剩下的位置全用掉）
    bg = [f for f in frames if f.idx not in picked]
    for f in _spread_pick(bg, room(keep_bg=False)):
        take(f, "bg")

    picks = sorted(picked.values(), key=lambda f: f.idx)
    counts: dict[str, int] = {}
    for f in picks:
        counts[f.reason] = counts.get(f.reason, 0) + 1
    return SamplePlan(picks=picks, counts=counts, total_frames=len(frames))


# ---------------------------------------------------------------- 第二遍解码：存帧
def save_frames(path: str | Path, picks: list[FrameSignal], out_dir: str | Path,
                *, prefix: str, jpeg_quality: int = 92, progress=None
                ) -> list[tuple[FrameSignal, Path]]:
    """按帧号存 JPEG（**顺序解码 + 跳过**，不做 seek）。返回 `[(信号, 路径)]`。

    文件名 `{prefix}_{idx:06d}.jpg` —— **纯 ASCII**（见 `slug()` 的说明）。
    """
    import cv2

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    want = {f.idx: f for f in picks}
    if not want:
        return []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{path}")
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    saved: list[tuple[FrameSignal, Path]] = []
    i = 0
    try:
        while want:
            ok, frame = cap.read()
            if not ok:
                break
            f = want.pop(i, None)
            if f is not None:
                p = out / f"{prefix}_{i:06d}.jpg"
                if not cv2.imwrite(str(p), frame, params):
                    raise RuntimeError(
                        f"cv2.imwrite 失败：{p}（路径必须纯 ASCII —— 见 frames.slug()）")
                saved.append((f, p))
            i += 1
            if progress and i and i % 3000 == 0:
                progress(f"    存帧 {i} 帧（还差 {len(want)} 张）…")
    finally:
        cap.release()
    missing = sorted(want)
    if missing:
        raise RuntimeError(f"有 {len(missing)} 个帧号没解码到（视频比预期短？）"
                           f"，例如 {missing[:5]}")
    return saved
