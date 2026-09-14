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
| 刹车脉冲**间隔 < 200ms** 的成对脉冲 | **360** | `360` |
| 刹车**独立短脉冲**（单发、持续很短） | **打断氮气** | `D:<毫秒>`（很短） |
| 刹车**按住**（较长） | 漂移 | `D:<毫秒>` |
| 氮气**多次快速点击** | 每百分比**最多取两次** | `N:0:2:<两次实测间隔>` |
| OCR 看到「完美氮气」 | 同上（用实测的双击间隔） | `N:0:2:<间隔>` |
| 氮气**单击** | 普通氮气 | `N:100:1:100` |
| 氮气**长按** | 快速连点 | `N:0:k:100` |
| 上方圆形路标（精细扫描里一起看） | 选路：**一直不变就取第一次识别到的**；**变了就取第一次改变时的** | `NN` |

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


def classify_choices(icons_timeline: list[tuple[float, tuple[int, int] | None]],
                     *, min_hold: int | None = None) -> list[Intent]:
    """选路信号 -> 操作（**精细扫描里一起看**，用户要求）。

    规则（用户 2026-09-13）：

    * **一直不变**（例如从第一次识别到识别不到都是 `21`）-> 取**第一次识别到**的时刻；
    * **发生改变**（`21` -> `22`）-> 取**第一次改变**的时刻；
    * 这样一个岔路口只会选一次/每变一次选一次，**不会同一段反复选** ✓。

    `min_hold`：**同一个值必须连续出现这么多帧才认**（默认 3）——
    真岔路口的路标会持续好几秒 ✓，而把这些当成路标的景色闪光只闪一两帧 ✗
    （实测：不加这道闸，10 秒的视频能报出 16 次选路 ✗）。

    `icons_timeline` 每项 `(时刻, (图标数, 蓝色高亮是第几个) 或 None)`。
    """
    min_hold = _choice_min_hold() if min_hold is None else int(min_hold)
    # 先做"连续 min_hold 帧同值"的过滤
    stable: list[tuple[float, tuple[int, int]]] = []
    run_val: tuple[int, int] | None = None
    run_from: float | None = None
    run_len = 0
    for t, cur in icons_timeline:
        if cur is None:
            if run_val is not None and run_len >= min_hold and run_from is not None:
                stable.append((run_from, run_val))
            run_val, run_from, run_len = None, None, 0
            continue
        if cur == run_val:
            run_len += 1
        else:
            if run_val is not None and run_len >= min_hold and run_from is not None:
                stable.append((run_from, run_val))
            run_val, run_from, run_len = cur, t, 1
    if run_val is not None and run_len >= min_hold and run_from is not None:
        stable.append((run_from, run_val))

    out: list[Intent] = []
    last: tuple[int, int] | None = None
    for t, cur in stable:
        if last is None:
            cnt, idx = cur
            out.append(Intent(kind="choice", t0=t, op=f"{cnt}{idx}",
                              note=f"第一次识别到 {cnt} 选 {idx}"))
            last = cur
        elif cur != last:
            cnt, idx = cur
            out.append(Intent(kind="choice", t0=t, op=f"{cnt}{idx}",
                              note=f"从 {last[0]}选{last[1]} 变成 {cnt}选{idx}"
                                   f" -> 取第一次改变时"))
            last = cur
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
