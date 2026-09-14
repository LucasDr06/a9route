# -*- coding: utf-8 -*-
"""`core/intent.py` 的离线测试（纯函数，不需要视频、不需要设备）。

这套规则是用户 2026-09-13 口述的：**精细扫描只产时间戳，
操作目的由信号的"形状"判定**。

  T1 刹车成对脉冲（间隔 <300ms）-> 360
  T2 刹车独立短脉冲 -> 打断氮气（很短的 D:）
  T3 刹车按住 -> 漂移（实测时长）
  T4 氮气多次快速点击 -> 每百分比最多两次 + 实测间隔
  T5 氮气单击 / 长按
  T6 选路：一直不变取第一次；变了取第一次改变时
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

    print("\n=== T6 选路：不变取第一次，变了取第一次改变 ===")
    # 每个值都要连续 min_hold(4) 帧才认（真路标会持续几秒）
    tl = [(1.0 + i * 0.03, (2, 1)) for i in range(5)] + [(1.2, None)] \
        + [(2.0 + i * 0.03, (2, 2)) for i in range(5)] \
        + [(2.2 + i * 0.03, (2, 3)) for i in range(5)] + [(2.4, None)]
    ev = I.classify_choices(tl)
    check("两次岔路口共 3 条（21 / 22 / 23）", len(ev) == 3,
          str([(round(e.t0, 2), e.op) for e in ev]))
    check(" 同一个值不重复选", [e.op for e in ev] == ["21", "22", "23"],
          str([e.op for e in ev]))
    check(" 取的是第一次识别到 / 第一次改变的时刻",
          abs(ev[0].t0 - 1.0) < 1e-6 and abs(ev[1].t0 - 2.0) < 1e-6
          and abs(ev[2].t0 - 2.2) < 1e-6, str([e.t0 for e in ev]))
    same = I.classify_choices([(1.0 + i * 0.03, (2, 1)) for i in range(6)] + [(1.2, None)])
    check(" 一直不变 -> 只一条，取第一次", len(same) == 1 and abs(same[0].t0 - 1.0) < 1e-6,
          str(same))
    # **稳定性闸**：只闪一两帧的不算（用户报过"实际 2 个却判成 3 个"、10 秒报 16 次 ✗）
    noisy = ([(0.0, (3, 3))]                                  # 只闪一帧
             + [(0.1 + i * 0.03, None) for i in range(4)]
             + [(0.3 + i * 0.03, (2, 2)) for i in range(5)])   # 真岔路口
    ev2 = I.classify_choices(noisy)
    check("只闪一两帧的景色闪光不算选路", len(ev2) == 1 and ev2[0].op == "22", str(ev2))

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
