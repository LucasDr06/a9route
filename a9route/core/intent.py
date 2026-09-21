# -*- coding: utf-8 -*-
"""intent.py —— 把"精细扫描"的原始信号判成**操作目的**（用户 2026-09-13 的规则）。

## 为什么要这一步

路线脚本的精细度只有 **100 次**（0~100%）。同一段里可能连做好几个动作，
所以光有"百分比 + 按键亮着"分不出**这次按键是为了什么** ✗。
用户给的规则是：精细扫描**不直接产百分点**，而是产一串**带时刻的信号**，
再由这些信号的**形状**判定目的，最后才落到百分比上。

## 规则（用户口述 → 实现）

| 信号形状 | 判定 | 写成 |
|---|---|---|
| 刹车脉冲**间隔 < `TAP_360_GAP`**（默认 300ms）的成对脉冲 | **360** | `360` |
| 刹车**独立短脉冲**（单发、持续很短） | **打断氮气** | `D:<毫秒>`（很短） |
| 刹车**按住**（较长） | 漂移 | `D:<毫秒>` |
| 氮气**多次快速点击** | 每百分比**最多取两次** | `N:0:2:<两次实测间隔>` |
| OCR 看到「完美氮气」 | 同上（用实测的双击间隔） | `N:0:2:<间隔>` |
| 氮气**单击** | 普通氮气 | `N:100:1:100` |
| 氮气**长按** | 快速连点 | `N:0:k:100` |
| 上方圆形路标（精细扫描里一起看） | 选路：**按"选路段"取结果**（见 `classify_choices` 的规则表） | `NN` |

最后一步（时刻 -> 百分比）由 `video.attach_events()` 用粗扫得到的
「时间 -> 百分比」关系插值完成 ✓。
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: 刹车两下之间 < 这个值 -> 判为 360
#: （用户 2026-09-13 从 200ms 放宽到 **300ms**：实测双击偶尔会慢一点，
#:   200ms 太紧会把 360 判成"独立脉冲（打断氮气）"✗）
TAP_360_GAP = 0.30
#: 刹车单发持续超过这个值 -> 算"按住漂移"，否则算"独立脉冲（打断氮气）"
DRIFT_MIN = 0.30
#: 氮气脉冲聚成"一次连点"的时间窗
NITRO_GAP = 0.25
#: 氮气长按判定
NITRO_HOLD = 0.60

#: 选路稳定性闸：同一个值连续这么多帧才认（默认 4）。
#: 真岔路口会持续好几秒 ✓，把这些当成路标的景色闪光只闪一两帧 ✗
#: （不加这道闸，10 秒视频能报出 16 次选路）。
#: ⚠️ 这份**是 `a9route.config.apply()` 灌进来的运行值**，所以 `classify_choices()`
#: 在**调用时**读它（不能写成默认参数 —— 那会在 import 时就绑死）✓
CHOICE_MIN_HOLD = 4

#: **「选路结束」的判定**（用户 2026-09-15 的新口径）：
#: 连续这么多个采样点**都没检测到路标** = 这一段选路结束了，回到"选路结束状态"。
#:
#: 为什么需要一个**独立的**阈值（而不是复用 `CHOICE_MIN_HOLD`）：
#: 它决定"两个挨得近的岔路口会不会被并成一段" ——
#: 太短：路标闪一下就把一段拆成两段（多报一次选路）；
#: 太长：两个岔路口并成一段，后一个的"选中"会被当成前一段的"最后一次变化"，
#:       于是**少报一次选路**。
#: 采样间隔 = `scan.every` 与 `scan.icon_every` 决定的（默认每 2 帧一次 ≈ 67ms），
#: 所以 8 ≈ 0.53 秒。
CHOICE_IDLE_HOLD = 8

#: 这些写成"取值函数"而不是默认参数，理由同上：import 之后再改常量也能生效
def _tap_360_gap() -> float:
    return float(TAP_360_GAP)


def _drift_min() -> float:
    return float(DRIFT_MIN)


def _nitro_gap() -> float:
    return float(NITRO_GAP)


def _nitro_hold() -> float:
    return float(NITRO_HOLD)


def _choice_min_hold() -> int:
    return int(CHOICE_MIN_HOLD)


def _choice_idle_hold() -> int:
    return int(CHOICE_IDLE_HOLD)


@dataclass
class Intent:
    """一个有目的的操作（还没有百分比）。"""

    kind: str                    # 360 / drift / brake_tap / nitro / choice
    t0: float                    # 开始时刻（秒）
    dur: float = 0.0             # 持续（秒）
    pulses: int = 1              # 脉冲个数（双击=2）
    interval_ms: int = 0         # 两次点击之间的实测间隔（毫秒）
    op: str = ""                 # 路线里的写法
    percent: float = 0.0         # 事后由时间轴填
    note: str = ""               # 判据说明（写进注释，方便核对）

    def describe(self) -> str:
        return (f"{self.percent:>5.0f}%  {self.kind:<10} 起 {self.t0:6.2f}s "
                f"时长 {self.dur:5.2f}s 脉冲 {self.pulses} "
                f"{('间隔 %dms' % self.interval_ms) if self.interval_ms else '':<12} "
                f"-> {self.op:<14} {self.note}")


def _pulse_groups(samples: list[tuple], idx: int, *, gap: float) -> list[tuple[float, float, int, list[float]]]:
    """把"按下"的帧聚成一次动作，并把里面的**脉冲起始时刻**也带出来。

    返回 [(起点, 时长, 脉冲数, [各脉冲起始时刻])]。
    与 `video._episodes` 的区别：这里额外返回每个脉冲的起点 ——
    用户要求把"两次点击的间隔"填进氮气操作，没有脉冲时刻就算不出来 ✗。
    """
    runs: list[list[float]] = []
    cur: list[float] = []
    first_off: float | None = None
    for s in samples:
        t = float(s[0])
        on = bool(s[1 + idx])
        if on:
            if cur and first_off is not None and (t - first_off) > gap + 1e-6:
                runs.append(cur)
                cur = []
            cur.append(t)
            first_off = None
        elif cur:
            if first_off is None:
                first_off = t
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        starts = [r[0]]
        for prev, nxt in zip(r, r[1:]):
            if nxt - prev > 0.06:          # 中间断开了 -> 新脉冲
                starts.append(nxt)
        out.append((r[0], r[-1] - r[0], len(starts), starts))
    return out


def classify_buttons(samples: list[tuple[float, bool, bool]], *,
                     perfect_nitro: list[tuple[float, float]] | None = None,
                     ) -> list[Intent]:
    """按键信号 -> 操作（刹车 / 氮气），带目的与实测间隔。

    `samples` 每项 `(t, 刹车按下, 氮气按下)`（精细扫描逐帧产出，**不含百分比**）。
    `perfect_nitro` 是 OCR 看到「完美氮气」的时间窗 `[(起, 止)]`，用来确认双击。
    """
    out: list[Intent] = []
    for (t0, dur, pulses, starts) in _pulse_groups(samples, 0, gap=_tap_360_gap()):
        if pulses >= 2:
            # **两下间隔 < 300ms 的成对刹车脉冲 = 360**（用户 2026-09-13 放宽到 300ms）
            gaps = [int(round((starts[i + 1] - starts[i]) * 1000))
                    for i in range(len(starts) - 1)]
            out.append(Intent(kind="360", t0=t0, dur=dur, pulses=pulses,
                              interval_ms=gaps[0] if gaps else 0, op="360",
                              note=f"{pulses} 连击，间隔 {gaps[:3]}ms < 300ms -> 360"))
        elif dur >= _drift_min():
            ms = max(200, min(8000, int(round((dur + 0.03) * 1000))))
            out.append(Intent(kind="drift", t0=t0, dur=dur, pulses=1,
                              op=f"D:{ms}", note=f"按住 {dur:.2f}s -> 漂移"))
        else:
            # **独立短脉冲 = 用来打断氮气**（紫喷那种：喷一下氮气再点一下刹车）
            ms = max(100, min(300, int(round((dur + 0.03) * 1000))))
            out.append(Intent(kind="brake_tap", t0=t0, dur=dur, pulses=1,
                              op=f"D:{ms}",
                              note=f"单发短脉冲 {dur:.2f}s -> 打断氮气"))

    for (t0, dur, pulses, starts) in _pulse_groups(samples, 1, gap=_nitro_gap()):
        if pulses >= 2:
            # **多次快速点击：每百分比最多取两次，间隔填实测值**
            interval = int(round((starts[1] - starts[0]) * 1000))
            note = f"{pulses} 次连点 -> 取前两次，间隔 {interval}ms"
            if perfect_nitro and any(a - 0.2 <= t0 <= b + 0.2 for a, b in perfect_nitro):
                note += "（OCR 见到「完美氮气」，确认是双击）"
            out.append(Intent(kind="nitro", t0=t0, dur=dur, pulses=2,
                              interval_ms=interval, op=f"N:0:2:{interval}",
                              note=note))
        elif dur >= _nitro_hold():
            k = max(2, min(20, int(round((dur + 0.05) / 0.35))))
            out.append(Intent(kind="nitro", t0=t0, dur=dur, pulses=1,
                              op=f"N:0:{k}:100", note=f"按住 {dur:.2f}s -> 快速连点"))
        else:
            out.append(Intent(kind="nitro", t0=t0, dur=dur, pulses=1,
                              op="N:100:1:100", note=f"单击 {dur:.2f}s -> 普通氮气"))
    out.sort(key=lambda e: e.t0)
    return out


def _stable_segments(timeline: list, min_hold: int) -> list:
    """时间轴 -> **连续同值的段**（只留长度 >= `min_hold` 的段）。

    返回 `[(起始下标, 结束下标, 起始时刻, 值), ...]`；`值` 可能是 `None`
    （那一刻没有检测到路标）。

    带上**下标**是因为调用方要按"原始采样点"数空档：
    `min_hold` 这道闸会把短段整段丢掉，若只用"留下来的段"数空档，
    比 `min_hold` 短的空档就永远数不到，`idle_hold` 也就调不动了 ✗
    """
    segs: list = []
    sentinel = object()
    cur: object = sentinel
    cur_from = 0.0
    i0 = 0
    n = 0
    for i, (t, v) in enumerate(timeline):
        if n and v == cur:
            n += 1
            continue
        if n >= min_hold:
            segs.append((i0, i, cur_from, None if cur is sentinel else cur))
        cur, cur_from, i0, n = v, t, i, 1
    if n >= min_hold:
        segs.append((i0, len(timeline), cur_from,
                     None if cur is sentinel else cur))
    return segs


def classify_choices(icons_timeline: list[tuple[float, tuple[int, int] | None]],
                     *, min_hold: int | None = None,
                     idle_hold: int | None = None) -> list[Intent]:
    """选路信号 -> 操作。**按"选路段"取结果，而不是按每次变化取**（用户 2026-09-15 的新口径）。

    ## 两个状态

    * **选路结束状态**（没有路标，或持续一小段看不到路标）；
    * **选路开始状态**（从"结束状态里第一次检测到选路结果"进入）。

    ## 规则（用户原话的意思，逐条对应代码）

    1. **持续没有检测到选路 -> 进入选路结束状态**（`idle_hold` 个采样点都没路标）；
    2. 在结束状态下**第一次**检测到新结果 -> 进入开始状态，
       并以**这一次**的结果作为"这一段"的起点，之后拿它判断"有没有变化"；
    3. 在这一段结束（进入下一个结束状态）之前，
       **数量没变、选中也没变** -> 取**第一次**的结果和百分比；
    4. **数量没变、只有选中变了** -> 取**最后一次变化**时的结果和百分比；
    5. **数量变了** -> **立刻**结束当前这一段，并以新的结果进入**新的一段**
       （不等"持续没有检测到"）—— 因为"几条路可选"变了就是另一个岔路口了。

    ## 为什么还要 `min_hold` 这道闸

    它挡的是**闪烁**：只闪一两帧的值不算数（实测不加它，10 秒视频能报出 16 次选路 ✗）。
    状态机跑在**过了这道闸的段**上，所以"数量变化"和"选中变化"都是**站得住的变化**，
    规则 5 的"快速结束"也就不会因为一帧误检而拆段。

    `icons_timeline` 每项 `(时刻, (图标数, 蓝色高亮是第几个) 或 None)`。
    """
    min_hold = _choice_min_hold() if min_hold is None else int(min_hold)
    idle_hold = _choice_idle_hold() if idle_hold is None else int(idle_hold)
    idle_hold = max(1, int(idle_hold))

    segs = _stable_segments(icons_timeline, min_hold)
    out: list[Intent] = []
    # 当前这一段：起点值 + "最后一次变化"（没有变化时是 None）
    cur_val: tuple[int, int] | None = None
    cur_from = 0.0
    last_val: tuple[int, int] | None = None      # 仅"选中"变化时更新
    last_from = 0.0
    idle_run = 0
    prev_end = 0

    def close(note_tail: str = "") -> None:
        """收掉当前这一段：按规则 3/4 决定取哪个时刻/哪个值。"""
        nonlocal cur_val, last_val
        if cur_val is None:
            return
        if last_val is None:
            cnt, idx = cur_val
            out.append(Intent(kind="choice", t0=cur_from, op=f"{cnt}{idx}",
                              note=f"这一段一直没变（{cnt} 选 {idx}）"
                                   f" -> 取第一次识别到时{note_tail}"))
        else:
            cnt, idx = last_val
            out.append(Intent(kind="choice", t0=last_from, op=f"{cnt}{idx}",
                              note=f"这一段里只有「选中」变过，最后一次是 "
                                   f"{cnt} 选 {idx} -> 取最后一次变化时{note_tail}"))
        cur_val, last_val = None, None

    for i0, i1, t, val in segs:
        # 这一段之前、被稳定性闸丢掉的采样点，同样算"没确认到值" -> 计入空档
        idle_run += (i0 - prev_end)
        if cur_val is not None and idle_run >= idle_hold:
            close(note_tail="（之后持续没检测到路标 -> 这一段结束）")
        if val is None:
            # 确认"这一段没有路标"：整段都算进空档
            idle_run += (i1 - i0)
            if cur_val is not None and idle_run >= idle_hold:
                close(note_tail="（之后持续没检测到路标 -> 这一段结束）")
            prev_end = i1
            continue
        idle_run = 0
        cnt, sel = val
        if cur_val is None:
            # 规则 2：结束状态里第一次检测到 -> 开新的一段，以它为起点
            cur_val, cur_from, last_val = (cnt, sel), t, None
        elif cnt != cur_val[0]:
            # 规则 5：数量变了 -> **立刻**收掉这一段，用新结果开新的一段
            prev = cur_val[0]
            close(note_tail="（选项数变了：{0} -> {1}，这一段到此为止）"
                            .format(prev, cnt))
            cur_val, cur_from, last_val = (cnt, sel), t, None
        elif sel != cur_val[1]:
            # 规则 4：数量没变、只有选中变了 -> 记住"最后一次变化"
            cur_val = (cnt, sel)
            last_val, last_from = (cnt, sel), t
        prev_end = i1
    # 结尾剩下的采样点也算空档（"一直没再出现" = 这一段已经结束了）
    idle_run += (len(icons_timeline) - prev_end)
    close()
    return out


def merge_by_percent(intents: list[Intent], *, max_nitro: int = 2) -> list[Intent]:
    """把同一百分比上的操作整理好：**氮气每百分比最多取两次脉冲**（用户规则）。

    同一百分比上出现多次氮气动作时，只保留第一条（它的写法里已经是"两次点击 + 实测间隔"），
    其余丢掉 —— 这样"多次快速点击"不会写成一堆重复条目 ✓。
    """
    out: list[Intent] = []
    seen_nitro: set[int] = set()
    seen_choice: set[int] = set()
    for it in intents:
        p = int(round(it.percent))
        if it.kind == "nitro":
            if p in seen_nitro:
                continue
            seen_nitro.add(p)
        if it.kind == "choice":
            # 同一百分比上的选路只留一条（避免反复选）
            if p in seen_choice:
                continue
            seen_choice.add(p)
        out.append(it)
    return out
