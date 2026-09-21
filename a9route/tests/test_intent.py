# -*- coding: utf-8 -*-
"""`core/intent.py` 的离线测试（纯函数，不需要视频、不需要设备）。

这套规则是用户 2026-09-13 口述的：**精细扫描只产时间戳，
操作目的由信号的"形状"判定**。

  T1 刹车成对脉冲（间隔 <300ms）-> 360
  T2 刹车独立短脉冲 -> 打断氮气（很短的 D:）
  T3 刹车按住 -> 漂移（实测时长）
  T4 氮气多次快速点击 -> 每百分比最多两次 + 实测间隔
  T5 氮气单击 / 长按
  T6 选路：按「选路段」取结果（段内没变取第一次 / 只有选中变了取最后一次 /
     选项数变了立刻开新段 / 持续没检测到才算结束）
  T7 同一百分比上的氮气/选路各只留一条

    python -m a9route.tests.test_intent
"""
from __future__ import annotations

import sys

from a9route.core import intent as I
from a9route.tests.support import check, summary


def seq(items):
    return [(t, b, n) for t, b, n in items]


def main() -> int:
    print("=== T1 刹车成对脉冲（<300ms）-> 360 ===")
    # 两下刹车：0.00 按下、0.10 松开、0.12 再按下（间隔 120ms）
    s = seq([(0.00, True, False), (0.05, True, False), (0.10, False, False),
             (0.12, True, False), (0.20, True, False)] + [(0.30 + i * 0.03, False, False)
                                                          for i in range(10)])
    ev = I.classify_buttons(s)
    check("判成 360", len(ev) == 1 and ev[0].kind == "360" and ev[0].op == "360", str(ev))
    check(" 记下了两下的间隔（≈120ms）",
          bool(ev) and 80 <= ev[0].interval_ms <= 160, str(ev[0].interval_ms if ev else None))

    print("\n=== T2 刹车独立短脉冲 -> 打断氮气 ===")
    s = seq([(0.00, True, False), (0.03, True, False), (0.06, False, False)]
            + [(0.5 + i * 0.03, False, False) for i in range(20)])
    ev = I.classify_buttons(s)
    check("判成 brake_tap（很短）", len(ev) == 1 and ev[0].kind == "brake_tap", str(ev))
    check(" 写法是很短的 D:", bool(ev) and ev[0].op.startswith("D:")
          and int(ev[0].op[2:]) <= 300, str(ev[0].op if ev else None))

    print("\n=== T3 刹车按住 -> 漂移 ===")
    s = seq([(i * 0.03, True, False) for i in range(30)]
            + [(1.2 + i * 0.03, False, False) for i in range(10)])
    ev = I.classify_buttons(s)
    check("判成漂移（时长≈0.9s）", len(ev) == 1 and ev[0].kind == "drift"
          and 700 <= int(ev[0].op[2:]) <= 1200, str(ev))

    print("\n=== T4 氮气多次快速点击 -> 最多两次 + 实测间隔 ===")
    # 5 次快速点击（每次 0.05s，间隔 0.10s）
    s = []
    t = 0.0
    for _ in range(5):
        s += [(t, False, True), (t + 0.03, False, True), (t + 0.05, False, False)]
        t += 0.15
    s += [(2.0 + i * 0.03, False, False) for i in range(10)]
    s.sort()
    ev = I.classify_buttons(s)
    check("只写一条氮气", len(ev) == 1, str(ev))
    check(" 写成 N:0:2:<实测间隔>", bool(ev) and ev[0].op.startswith("N:0:2:")
          and ev[0].interval_ms > 0, str(ev[0].op if ev else None))
    check(" 脉冲数记成 2（每百分比最多两次）", bool(ev) and ev[0].pulses == 2,
          str(ev[0].pulses if ev else None))
    # 完美氮气确认
    ev2 = I.classify_buttons(s, perfect_nitro=[(0.0, 1.0)])
    check(" OCR 见到「完美氮气」时标注出来",
          bool(ev2) and "完美氮气" in ev2[0].note, str(ev2[0].note if ev2 else None))

    print("\n=== T5 氮气单击 / 长按 ===")
    one = seq([(0.0, False, True), (0.03, False, True), (0.06, False, False)]
              + [(1.0 + i * 0.03, False, False) for i in range(10)])
    ev = I.classify_buttons(one)
    check("单击 -> N:100:1:100", len(ev) == 1 and ev[0].op == "N:100:1:100", str(ev))
    hold = seq([(i * 0.03, False, True) for i in range(30)])
    ev = I.classify_buttons(hold)
    check("长按 -> N:0:k:100", len(ev) == 1 and ev[0].op.startswith("N:0:")
          and ev[0].op.endswith(":100"), str(ev))

    print("\n=== T6 选路：按「选路段」取结果（用户 2026-09-15 的新口径）===")
    # 每个值都要连续 min_hold(4) 帧才认（真路标会持续几秒）
    # **持续没有路标**要连续 idle_hold 帧才算"选路结束"（默认 8）
    GAP = [(x, None) for x in (1.15 + i * 0.03 for i in range(9))]      # 9 帧 > 8
    SHORT_GAP = [(x, None) for x in (1.15 + i * 0.03 for i in range(3))]  # 3 帧 < 8

    # 规则 3：一段里数量/选中都没变 -> 取第一次
    tl = [(1.0 + i * 0.03, (2, 1)) for i in range(5)] + GAP
    ev = I.classify_choices(tl)
    check("段内一直没变 -> 一条，取**第一次**的结果和时刻",
          len(ev) == 1 and ev[0].op == "21" and abs(ev[0].t0 - 1.0) < 1e-6
          and "一直没变" in ev[0].note, str([(round(e.t0, 2), e.op) for e in ev]))

    # 规则 4：数量没变、只有选中变了 -> 取**最后一次变化**
    tl2 = [(2.0 + i * 0.03, (2, 1)) for i in range(5)] \
        + [(2.2 + i * 0.03, (2, 2)) for i in range(5)] \
        + [(2.4 + i * 0.03, (2, 3)) for i in range(5)] + GAP
    ev2 = I.classify_choices(tl2)
    check("数量没变、选中变了 -> 取**最后一次变化**的结果和时刻",
          len(ev2) == 1 and ev2[0].op == "23" and abs(ev2[0].t0 - 2.4) < 1e-6
          and "最后一次变化" in ev2[0].note,
          str([(round(e.t0, 2), e.op) for e in ev2]))

    # 规则 5：数量变了 -> **立刻**分段（不等"持续没有检测到"）
    tl3 = [(3.0 + i * 0.03, (3, 3)) for i in range(5)] \
        + [(3.2 + i * 0.03, (2, 1)) for i in range(5)] \
        + [(3.4 + i * 0.03, (2, 2)) for i in range(5)] + GAP
    ev3 = I.classify_choices(tl3)
    check("数量变了 -> 立刻结束这一段、开新的一段（两条）",
          len(ev3) == 2 and [e.op for e in ev3] == ["33", "22"],
          str([(round(e.t0, 2), e.op) for e in ev3]))
    check("  新的一段自己重新算「变化」（数量没变、选中变了 -> 取最后一次）",
          abs(ev3[1].t0 - 3.4) < 1e-6, str([e.t0 for e in ev3]))

    # 规则 1+2：持续没有检测到 -> 结束状态；下一次检测到 -> 新的一段
    tl4 = [(4.0 + i * 0.03, (2, 1)) for i in range(5)] + GAP \
        + [(4.6 + i * 0.03, (2, 2)) for i in range(5)] + GAP
    ev4 = I.classify_choices(tl4)
    check("持续没检测到 -> 结束这一段的判断；下一次检测到**从头开始**",
          len(ev4) == 2 and [e.op for e in ev4] == ["21", "22"],
          str([(round(e.t0, 2), e.op) for e in ev4]))
    check("  两段各自取第一次（第二段不是「最后一次变化」）",
          abs(ev4[1].t0 - 4.6) < 1e-6 and "一直没变" in ev4[1].note, ev4[1].note)

    # 反例对照：**没到 idle_hold 的短暂空档**不该把一段拆开
    tl5 = [(5.0 + i * 0.03, (2, 1)) for i in range(5)] + SHORT_GAP \
        + [(5.4 + i * 0.03, (2, 2)) for i in range(5)] + GAP
    ev5 = I.classify_choices(tl5)
    check("空档没到 idle_hold -> **同一段**（数量没变、选中变了 -> 取最后一次）",
          len(ev5) == 1 and ev5[0].op == "22", str([(round(e.t0, 2), e.op) for e in ev5]))

    # **稳定性闸**：只闪一两帧的不算（用户报过"实际 2 个却判成 3 个"、10 秒报 16 次 ✗）
    noisy = ([(0.0, (3, 3))]                                  # 只闪一帧
             + [(0.1 + i * 0.03, None) for i in range(4)]
             + [(0.3 + i * 0.03, (2, 2)) for i in range(5)])
    ev6 = I.classify_choices(noisy)
    check("只闪一两帧的景色闪光不算选路", len(ev6) == 1 and ev6[0].op == "22", str(ev6))

    # 参数来自 config / 显式传参：idle_hold 改小 -> 同一份时间轴会分成两段
    ev7 = I.classify_choices(tl5, idle_hold=2)
    check("idle_hold 可调：改小之后同一份时间轴就分成两段",
          len(ev7) == 2, str([(round(e.t0, 2), e.op) for e in ev7]))

    print("\n=== T7 同一百分比上的氮气只留一条 ===")
    intents = [I.Intent(kind="nitro", t0=1.0, op="N:0:2:100", percent=13),
               I.Intent(kind="nitro", t0=1.2, op="N:0:2:120", percent=13.4),
               I.Intent(kind="choice", t0=1.4, op="21", percent=13),
               I.Intent(kind="choice", t0=1.6, op="22", percent=13.2)]
    got = I.merge_by_percent(intents)
    check("氮气每百分比只留一条", sum(1 for e in got if e.kind == "nitro") == 1, str(got))
    check("选路每百分比也只留一条", sum(1 for e in got if e.kind == "choice") == 1, str(got))

    print("\n=== T8 参数来自 config（改 config 立刻生效）===")
    from a9route import config as cfgmod
    old_gap, old_hold = I.TAP_360_GAP, I.NITRO_HOLD
    try:
        cfgmod.apply(cfgmod.load_config() | {"intent": {"nitro_hold": 1.0}})
        check("config 改 nitro_hold -> 模块常量跟着变（不用改代码）",
              I.NITRO_HOLD == 1.0, str(I.NITRO_HOLD))
        # 按住 0.5s 的氮气：默认阈值 0.6 算"单击"，阈值抬到 1.0 才算"长按"
        s2 = seq([(i * 0.03, False, True) for i in range(16)]
                 + [(0.6 + i * 0.03, False, False) for i in range(10)])
        ev3 = I.classify_buttons(s2)
        check(" 阈值 1.0 时，0.5s 的按住不算长按 -> 单击",
              bool(ev3) and ev3[0].op == "N:100:1:100", str(ev3))
        cfgmod.apply(cfgmod.load_config() | {"intent": {"nitro_hold": 0.3}})
        ev4 = I.classify_buttons(s2)
        check(" 阈值降到 0.3 后，同一段信号变成快速连点 N:0:k:100",
              bool(ev4) and ev4[0].op.startswith("N:0:") and ev4[0].op != "N:100:1:100",
              str(ev4))
    finally:
        cfgmod.apply()
    check(" 恢复默认", I.TAP_360_GAP == old_gap and I.NITRO_HOLD == old_hold,
          f"gap={I.TAP_360_GAP} hold={I.NITRO_HOLD}")

    return summary("test_intent")


if __name__ == "__main__":
    sys.exit(main())
