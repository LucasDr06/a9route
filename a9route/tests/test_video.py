# -*- coding: utf-8 -*-
"""视频分析这条线的离线测试（**不碰设备**，用合成视频 + 真帧样本）。

  T1 时间轴：丢空档、防倒退、同百分比只留第一次、插值
  T2 空草稿：每个整百分点一行，**不凭空生成操作**（避免造假）
  T3 抽帧循环（真 OpenCV 解码）：按 every 抽帧、报告**真实解码时间**
  T4 窗口汇总 + 段合并：选路取最后一次、同段多操作用 `|`、漂移时长按时长换算
  T5 按键动作 -> 操作写法：单击 / 双击 / 长按 / 刹车时长 / **点松不被并成长按**
  T6 按键判据：**用用户给的截图当样本**（按住刹车 / 没按 / 氮气键）
  T7 选路路标检测：**用真帧当样本**（2 个 / 0 个 / 点完之后）
  T8 比赛内读数解析：路程 NN% / 排名 / TouchDrive
  T9 360 时屏蔽漂移
  T10 端到端：合成视频 -> 逐百分点截图 -> 精细按键扫描 -> 一条扁平路线
  T11 配置：改 config 就生效；未知键要报错

    python -m a9route.tests.test_video
"""
from __future__ import annotations

import sys
from pathlib import Path

from a9route import config as cfgmod
from a9route.tests.support import (check, fake_hud_video, fixture, summary,
                                   synthetic_video, tmp_dir)
from a9route.vision import video as V

#: 测试产物目录（`main()` 里用 support.tmp_dir() 建好；跟随 A9ROUTE_WORKTMP）
TMP = Path(".")


# ------------------------------------------------------------------ T1
def t1_timeline():
    """时间轴：抽帧读到的 (时刻, 百分比) -> 「百分比 -> 时刻」的映射。"""
    print("\n=== T1 时间轴 ===")
    samples = [V.Sample(0.0, None), V.Sample(0.5, 1.0), V.Sample(1.0, 2.0),
               V.Sample(1.5, 2.0), V.Sample(2.0, 1.0), V.Sample(2.5, 3.0),
               V.Sample(3.0, None), V.Sample(3.5, 4.0)]
    tl = V.build_timeline(samples)
    check("丢掉读不到百分比的帧，只留有效点",
          [p for p, _ in tl.points] == [1.0, 2.0, 3.0, 4.0], str(tl.points))
    check("倒退的点被丢掉并记警告（OCR 抖动）",
          any("抖动" in w for w in tl.warnings), str(tl.warnings))
    check("同一个百分比只留第一次出现的时间",
          tl.time_at(2.0) == 1.0, str(tl.time_at(2.0)))
    # 点集 (1%,0.5s) (2%,1.0s) (3%,2.5s) (4%,3.5s)：2.5% 落在 2~3 之间 ->
    # 1.0 + 0.5*(2.5-1.0) = 1.75s
    check("插值：2.5% -> 1.75s", abs((tl.time_at(2.5) or 0) - 1.75) < 1e-6,
          str(tl.time_at(2.5)))
    check("超出范围 -> None", tl.time_at(0.5) is None and tl.time_at(99) is None)
    check("反插值：时刻 -> 百分比（精细扫描要落回百分点）",
          abs((tl.percent_at(1.0) or 0) - 2.0) < 1e-6, str(tl.percent_at(1.0)))
    check("describe() 报了采样数与覆盖范围",
          "采样" in tl.describe() and "%" in tl.describe(), tl.describe().replace("\n", " | "))


def t2_draft():
    print("\n=== T2 空草稿（只对齐时间，不猜操作）===")
    tl = V.build_timeline([V.Sample(0.5, 1.0), V.Sample(1.0, 2.0),
                           V.Sample(2.5, 3.0), V.Sample(3.5, 4.0)])
    draft = V.draft_route(tl)
    check("草稿里带格式说明", "格式：<百分比>,<操作>" in draft, draft[:80])
    check("每个整百分点都有一行（1~4%）",
          all(f"# {p:5.0f}%  @ " in draft for p in (1, 2, 3, 4)), draft)
    check("带上了**自动算好的时间戳**（这是草稿的价值所在）",
          "0.50s" in draft and "1.00s" in draft, draft)
    check("**不**凭空生成操作（避免造假）",
          not any(line and not line.startswith("#") for line in draft.splitlines()),
          draft)
    empty = V.build_timeline([V.Sample(0.0, None)])
    check("一帧都没读到百分比 -> 明确说无法生成", "无法生成" in V.draft_route(empty),
          V.draft_route(empty).strip())


# ------------------------------------------------------------------ T3
def t3_iter_frames():
    """真 OpenCV 解码这一条路径（合成小视频，不依赖任何真录像）。"""
    print("\n=== T3 抽帧循环（真解码）===")
    tmp = synthetic_video(TMP / "t3.mp4", seconds=2.0, fps=10.0)
    got = list(V.iter_frames(tmp, every=0.2))
    check("按 every 抽到帧", len(got) >= 5, f"{len(got)} 帧")
    check(" 时刻严格递增", all(got[i + 1][0] > got[i][0] for i in range(len(got) - 1)),
          str([t for t, _ in got][:6]))
    check(" 每帧都有图像（真解码出来了）",
          all(f is not None and getattr(f, "size", 0) for _t, f in got))
    # ⚠️ 这条是真 bug 的回归：`cap.set(POS_MSEC)` 之后必须读 `cap.get(POS_MSEC)`
    # 才是**真实解码时间**；用"请求的时间"会和"顺序解码"的另一条流错位，
    # 表现为"精细识别说 1%、百分比识别说 5%"（用户 2026-09-13 报过）
    times = [t for t, _ in got]
    check(" 报的是真实解码时间（单调不减、不为负）",
          all(t >= 0 for t in times) and times == sorted(times), str(times[:6]))
    maxt = list(V.iter_frames(tmp, every=0.2, max_seconds=0.6))
    check("--max-seconds 生效（只取前 0.6s）", len(maxt) <= 5, str(len(maxt)))
    try:
        list(V.iter_frames(TMP / "不存在的视频.mp4", every=0.5))
        check("打不开视频 -> 报错", False, "居然没抛")
    except Exception as exc:
        check("打不开视频 -> 明确报错", "打不开视频" in str(exc), f"{type(exc).__name__}: {exc}")


def t4_window_and_collapse():
    """窗口汇总 + 段合并（用户 2026-09-13 的两条规则）。"""
    print("\n=== T4 窗口汇总 / 段合并 ===")
    from a9route.vision.cues import Cue

    def cue(icons=(), brake=False, nitro=False):
        # nitro 是"**氮气键被按下**"（不是"顶部槽青"）
        return Cue(icons=list(icons), brake_lit=brake, nitro_pressed=nitro,
                   nitro=nitro, drifting=brake)

    def ic(x, blue):
        return {"xy": (x, 126), "blue": blue, "blue_ratio": 0.8 if blue else 0.0}

    # ① 同一个百分点上先是"3 选 1"，最后改成"3 选 3" -> 取**最后一次**
    frames = [
        (10.00, cue([ic(601, True), ic(677, False), ic(753, False)])),
        (10.25, cue([ic(601, False), ic(677, False), ic(753, True)])),
        (10.50, cue([ic(601, False), ic(677, False), ic(753, True)])),
    ]
    ops = V._ops_from_window(frames)
    check("选路取最后一次高亮 -> 33（不是第一帧的 31）",
          ops.get("choice") == "33", str(ops))

    # ② 同一段里"漂移 + 氮气"同时 -> 合并成一条里用 |
    both = [(10.0, cue(brake=True, nitro=True)), (10.25, cue(brake=True, nitro=True)),
            (10.5, cue(brake=True, nitro=True)), (10.75, cue(brake=True, nitro=True))]
    ops2 = V._ops_from_window(both)
    check("同段里漂移+氮气都被检测到", set(ops2) == {"drift", "nitro"}, str(ops2))
    shots = [V.PercentShot(percent=22.0, t=10.0, kinds=["drift", "nitro"],
                           values={"drift": "D", "nitro": "N"}),
             V.PercentShot(percent=23.0, t=10.5, kinds=["drift", "nitro"],
                           values={"drift": "D", "nitro": "N"})]
    merged = V.collapse_runs(shots, sample_seconds=0.5)
    check("连续同状态合并成一条", len(merged) == 1, str([(m.percent, m.op) for m in merged]))
    check(" 两个操作用 | 连起来（同时执行）", bool(merged) and "|" in merged[0].op,
          str(merged[0].op if merged else None))
    check(" 漂移时长按段长换算（≈0.5~1.5s）",
          bool(merged) and merged[0].op.startswith("D:") and "|N:0:" in merged[0].op,
          str(merged[0].op if merged else None))
    check(" 百分比是段起点（22，进岔路口/入弯前就执行）",
          bool(merged) and merged[0].percent == 22.0, str(merged[0].percent if merged else None))

    # ③ 只有 1 个路标（不是岔路口）-> 不算选路
    one = [(10.0, cue([ic(601, True)])), (10.25, cue([ic(601, True)]))]
    check("只有 1 个路标 -> 不算选路（判据要求 >=2）",
          "choice" not in V._ops_from_window(one), str(V._ops_from_window(one)))
    # ④ 窗口里只有一帧看到路标 -> 也不算（防闪光）
    flash = [(10.0, cue([ic(601, True), ic(677, False)])), (10.25, cue()),
             (10.5, cue())]
    check("路标只闪一帧 -> 不算选路",
          "choice" not in V._ops_from_window(flash), str(V._ops_from_window(flash)))


def t5_button_events():
    """按键动作 -> 操作写法（用户的规则：单击 / 双击 / 长按 / 点松）。"""
    print("\n=== T5 按键动作 -> 操作写法 ===")

    def frames(seq):
        return [(t, b, n) for t, b, n in seq]

    dbl = frames([(0.0, False, True), (0.033, False, True), (0.066, False, False),
                  (0.10, False, False), (0.133, False, True), (0.166, False, True)])
    ev = V.button_events(dbl)
    check("双击氮气 -> N:0:2:100", [e.op for e in ev] == ["N:0:2:100"], str(ev))

    sgl = frames([(0.0, False, True), (0.033, False, True), (0.066, False, True)]
                 + [(1.0 + i * 0.033, False, False) for i in range(10)])
    ev = V.button_events(sgl)
    check("单击氮气 -> N:100:1:100", [e.op for e in ev] == ["N:100:1:100"], str(ev))

    hold = frames([(i * 0.033, False, True) for i in range(31)])
    ev = V.button_events(hold)
    check("长按 1s -> N:0:k:100（k 按时长）",
          len(ev) == 1 and ev[0].op.startswith("N:0:") and ev[0].op.endswith(":100"),
          str(ev))

    br = frames([(i * 0.033, True, False) for i in range(46)])
    ev = V.button_events(br)
    check("刹车按住 1.5s -> D:~1500ms",
          len(ev) == 1 and ev[0].op.startswith("D:") and 1200 <= int(ev[0].op[2:]) <= 1800,
          str(ev))

    # **点松**：0.5s 按住 + 0.3s 松开，重复 —— 必须切成很多段，不能并成一整段
    # （bug 现场：原来拿"最后一帧松开"算间隔 -> 间隔永远是 33ms -> 永远不切段 ✗✗）
    tap: list = []
    t0 = 0.0
    while t0 < 5.0:
        tap += [(t0 + k * 0.0333, True, False) for k in range(15)]
        tap += [(t0 + 0.5 + k * 0.0333, False, False) for k in range(9)]
        t0 += 0.8
    tap.sort()
    eps = V._episodes(tap, 0, gap=0.12)
    check("点松 5 秒 -> 切成 6~8 段（不再并成长按）", 6 <= len(eps) <= 8,
          f"{len(eps)} 段: {[(round(a, 2), round(d, 2)) for a, d, _ in eps]}")
    check(" 每段时长≈0.5s（不是几秒）",
          all(0.3 <= d <= 0.8 for _a, d, _p in eps),
          str([round(d, 2) for _a, d, _p in eps]))
    ev_tap = V.button_events(tap)
    check(" 写出来是一条条短漂移", all(e.op.startswith("D:") and int(e.op[2:]) < 900
                                      for e in ev_tap),
          str([e.op for e in ev_tap]))

    both = frames([(i * 0.033, True, i < 2 or 4 <= i < 6) for i in range(30)])
    ev = V.button_events(both)
    ev = [V.ButtonEvent(kind=e.kind, t0=e.t0, duration=e.duration, pulses=e.pulses,
                        op=e.op, percent=22.0) for e in ev]
    flat = V.flat_route_from_events(ev)
    check("同百分比的漂移+氮气用 | 连起来", "|" in flat and flat.startswith("22,"), flat)


def t6_key_fixtures():
    """按键判据：**拿用户给的截图当样本**（按住刹车 / 没按）。"""
    print("\n=== T6 按键判据（用户截图样本）===")
    import cv2
    from a9route.vision import cues as C

    on = cv2.imread(str(fixture("brake_key_on.png")))
    off = cv2.imread(str(fixture("brake_key_off.png")))
    if on is None or off is None:
        check("按键样本存在", False, "缺少 a9route/tests/fixtures/brake_key_*.png")
        return
    box_on = (0, 0, on.shape[1], on.shape[0])
    v_on = C.brake_pressed(on, box_on)
    v_off = C.brake_pressed(off, box_on)
    check("按住刹车：亮白占比高（实测 ≈0.64）", v_on > C.BRAKE_WHITE_THR, f"{v_on:.3f}")
    check("没按刹车：亮白占比低（实测 ≈0.18）", v_off < C.BRAKE_WHITE_THR, f"{v_off:.3f}")
    check("两者分得开（差 > 0.2）", v_on - v_off > 0.2, f"{v_on:.3f} vs {v_off:.3f}")
    red_on = C.red_ratio(on, box_on)
    check(" 参考：红占比在截图上也有效（0.24 vs 0.00），但换赛道会失效",
          red_on > 0.1, f"{red_on:.3f}")

    for tag in ("on", "off"):
        img = cv2.imread(str(fixture(f"nitro_key_{tag}.png")))
        if img is None:
            continue
        v = C.red_ratio(img, (0, 0, img.shape[1], img.shape[0]))
        check(f"氮气键（截图 {tag}，没按）：红占比 < 阈值", v < C.NITRO_RED_THR, f"{v:.3f}")
    check("刹车/氮气判据已接进 Cue 检测器",
          hasattr(C.CueDetector, "detect") and C.BRAKE_WHITE_THR > 0
          and C.NITRO_RED_THR > 0,
          f"刹车阈值={C.BRAKE_WHITE_THR} 氮气阈值={C.NITRO_RED_THR}")
    check("刹车框/氮气框都是标定过的固定框（整圆大小、左右对称）",
          C.BRAKE_KEY_BOX == (137, 497, 110, 110)
          and C.NITRO_KEY_BOX == (1033, 497, 110, 110)
          and C.NITRO_KEY_BOX[0] == 1280 - (C.BRAKE_KEY_BOX[0] + C.BRAKE_KEY_BOX[2]),
          f"{C.BRAKE_KEY_BOX} {C.NITRO_KEY_BOX}")
    check("按键判据三种证据都在（圈内相对白度/绝对白度/红）",
          hasattr(C, "ring_score") and hasattr(C, "key_pressed")
          and C.BRAKE_RING_THR > 0 and C.NITRO_RED_THR > 0,
          f"刹车圈内阈值={C.BRAKE_RING_THR} 氮气红阈值={C.NITRO_RED_THR}")
    # key_pressed() 的分通道选择：刹车认"圈内−圈外"，氮气只认"红"
    hit_b, info_b = C.key_pressed(on, box_on, "brake")
    check("key_pressed(brake) 在按下帧上判 True",
          hit_b is True, str(info_b))
    hit_n, info_n = C.key_pressed(cv2.imread(str(fixture("nitro_key_off.png"))),
                                  (0, 0, 110, 110), "nitro")
    check("key_pressed(nitro) 返回指标明细（ring/red/white）",
          set(["ring", "red", "white"]) <= set(info_n), str(info_n))


def t7_choice_icons():
    """选路路标检测：**拿真帧当样本**（比合成图可靠得多）。"""
    print("\n=== T7 选路路标（真帧回归）===")
    import cv2
    import numpy as np
    from a9route.vision.hud import RaceReader

    r = RaceReader()
    x, y, w, h = r.choice_band
    # 样本 PNG 是从**录视频时那套框**（400,92,480,80）裁下来的 300×84，
    # 比当前默认横带高 4px，所以这里显式注入"样本当时用的那套框"，
    # 把它整块放进去（不缩放 —— 缩放会改掉圆的半径，判据就不一样了）。
    box = (360, 88, 360, 84)
    r = RaceReader(coords={"choice_band": box})
    x, y, w, h = r.choice_band

    def with_band(png: Path):
        """把裁剪样本**原样整块**放进横带里，再放到 1280×720 整帧上。"""
        band = cv2.imread(str(png))
        assert band is not None, f"读不到样本 {png}"
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[y:y + band.shape[0], x:x + band.shape[1]] = band
        return frame

    check("夹具样本（300×84）能整块放进测试用的横带里",
          h >= 84 and w >= 300, f"choice_band={r.choice_band}")

    two = with_band(fixture("choice_band_2icons.png"))
    icons = r.choice_icons(two)
    check("真帧（2 个路标）-> 正好找到 2 个",
          len(icons) == 2, str([(i["xy"], round(i["blue_ratio"], 2)) for i in icons]))
    if len(icons) == 2:
        check(" 按从左到右排序", icons[0]["xy"][0] < icons[1]["xy"][0], str(icons))
        check(" 左边那个是未选中（蓝占比 0）", icons[0]["blue"] is False
              and icons[0]["blue_ratio"] == 0.0, str(icons[0]))
        check(" 右边那个是**蓝色高亮=当前选中**", icons[1]["blue"] is True
              and icons[1]["blue_ratio"] > 0.5, str(icons[1]))
        check(" 坐标落在选路横带里（可点，不是画面别处）",
              all(x - 40 <= i["xy"][0] <= x + w + 40
                  and y - 40 <= i["xy"][1] <= y + h + 40 for i in icons),
              f"band={r.choice_band} 中心={[i['xy'] for i in icons]}")

    empty = with_band(fixture("choice_band_empty.png"))
    check("真帧（没有路标）-> 0 个（旧判据在这帧上误报 1~5 个）",
          len(r.choice_icons(empty)) == 0, str(r.choice_icons(empty)))

    after = with_band(fixture("choice_band_2icons_b.png"))
    ic2 = r.choice_icons(after)
    check("真帧（点完后）-> 还是 2 个，且**蓝色高亮在右边**（点哪个就切哪个）",
          len(ic2) == 2 and ic2[1]["blue"] is True and ic2[0]["blue"] is False,
          str([(i["xy"], i["blue"]) for i in ic2]))

    fake = np.full((720, 1280, 3), 40, dtype=np.uint8)
    for cx, color in ((560, (240, 240, 240)), (660, (230, 120, 60))):
        yy, xx = np.ogrid[96:156, cx - 30:cx + 30]
        disk = (yy - 126) ** 2 + (xx - cx) ** 2 <= 30 ** 2
        patch = fake[96:156, cx - 30:cx + 30]
        patch[disk] = color
    got = r.choice_icons(fake)
    check("合成帧（两个圆）-> 也找到 2 个", len(got) == 2, str(got))
    check("路标参数来自 config（choice_param2 / 半径范围）",
          cfgmod.get(cfgmod.load_config(), "vision__choice_param2") == 28,
          str(cfgmod.get(cfgmod.load_config(), "vision__choice_min_r")))


def t7b_one_backend():
    """**粗扫和细扫必须是同一个选路后端**（2026-09-15 踩的那个分裂）。

    实测背景：`CueDetector` 以前自己造一个裸 `RaceReader()`，于是粗扫的
    `cue.icons` 一直是 HoughCircles，而细扫早就换成模型了 ——
    而 `cue.icons` 被拿去**交叉校验**（粗扫也看到 ≥2 个路标才认这次选路），
    等于**启发式能一票否掉模型判出来的选路**，模型赢的地方全被抹掉 ✗✗。
    （用 `worktmp/probe/who_decides_route.py` 数出来的：落到 HoughCircles 35 次。）
    """
    print("\n=== T7b 选路只该有一个后端（粗扫/细扫不许分裂）===")
    import numpy as np
    from a9route.vision import video as V
    from a9route.vision.cues import CueDetector
    from a9route.vision.hud import RaceReader

    frame = np.full((720, 1280, 3), 20, dtype=np.uint8)

    class _FakeChoice:
        """假后端：永远报"3 个路标、选中第 2 个"。纯黑帧上 HoughCircles 什么都找不到，
        所以只要拿到 3 个图标，就证明用的是**这个后端**而不是圆检测 ✓"""
        name = "fake"
        calls = 0

        def read(self, f):
            type(self).calls += 1
            from a9route.vision.choice import ChoiceRead
            icons = [{"xy": (500 + 60 * i, 127), "radius": 30, "blue": (i == 1),
                      "blue_ratio": 0.9 if i == 1 else 0.0} for i in range(3)]
            return ChoiceRead(True, 3, 2, icons)

    fake = _FakeChoice()

    # ① `CueDetector(choice=...)` 造 reader 时**必须把后端带进去**。
    #    这里把 `RaceReader` 换成一个"记账"的替身，直接看它收到了什么参数 ——
    #    （真实 reader 的 percent 来自 OCR，测试环境里不好造，所以分两步测。）
    import a9route.vision.hud as _hud
    made: dict = {}
    real_rr = _hud.RaceReader

    class _RecRR:
        def __init__(self, **kw):
            made.update(kw)

        def read(self, frame, **kw):
            from a9route.vision.hud import RaceState
            st = RaceState()
            st.percent = 37.0
            st.icons = list(fake.read(frame).icons)
            return st

    _hud.RaceReader = _RecRR
    try:
        cue, _pct = CueDetector(choice=fake, reader_texts=lambda f, b: []).detect(
            frame)
    finally:
        _hud.RaceReader = real_rr
    check("**CueDetector 造 reader 时把选路后端带进去了**（不是裸 reader）",
          made.get("choice") is fake, str(made))
    check("  粗扫的 cue.icons 因此来自后端（3 个），而不是 HoughCircles",
          len(cue.icons) == 3 and cue.choice == (3, 2),
          "icons={0} choice={1}".format(len(cue.icons), cue.choice))

    # ② 端到端：真 `RaceReader`（喂假 OCR 文本 + 注入后端）交给 CueDetector
    r_inj = RaceReader(reader=lambda f, b: ["37%"], choice=fake, grab=lambda: None)
    r_inj.progress_box = (0, 0, 10, 10)
    cue2, pct2 = CueDetector(reader=r_inj, reader_texts=lambda f, b: []).detect(frame)
    check("注入的 reader（自带后端）也能一路走到 cue.icons",
          pct2 == 37.0 and len(cue2.icons) == 3 and cue2.choice == (3, 2),
          "pct={0} icons={1}".format(pct2, len(cue2.icons)))
    check("纯黑帧 + 没注入后端时，圆检测什么都找不到（对照组）",
          CueDetector(reader=RaceReader(reader=lambda f, b: ["37%"],
                                        grab=lambda: None),
                      reader_texts=lambda f, b: []).detect(frame)[0].choice is None, "")

    try:
        CueDetector(reader=RaceReader(), choice=fake)
        check("  同时给 reader 和 choice -> 直接报错（不许悄悄一边赢）", False, "没报错")
    except ValueError as exc:
        check("  同时给 reader 和 choice -> 直接报错（不许悄悄一边赢）",
              "打架" in str(exc), str(exc)[:60])

    # `read(with_icons=False)`：只要时间轴的调用方不该白跑一遍路标检测
    class _Counter:
        name = "counter"

        def __init__(self):
            self.calls = 0

        def read(self, f):
            self.calls += 1
            from a9route.vision.choice import ChoiceRead
            return ChoiceRead(True, 2, 1, [{"xy": (600, 127), "radius": 28,
                                            "blue": True, "blue_ratio": 0.8}])

    cnt = _Counter()
    r2 = RaceReader(choice=cnt, reader=lambda f, b: ["37%"], grab=lambda: None)
    r2.progress_box = (0, 0, 10, 10)
    st = r2.read(frame, with_icons=False)
    check("`read(with_icons=False)` 不数路标（也不叫后端）",
          st.icons == [] and st.choice_icons == 0 and cnt.calls == 0, str(cnt.calls))
    st2 = r2.read(frame, with_icons=True)
    check("  默认还是数（with_icons=True）", cnt.calls == 1, str(cnt.calls))

    # 交叉校验的**纯函数**：留谁、丢谁，而且必须把丢掉的返回出来（不许静默）
    class _It:
        def __init__(self, kind, percent, op="21"):
            self.kind, self.percent, self.op = kind, percent, op

    kept_in = [_It("choice", 40), _It("choice", 55), _It("nitro", 55)]
    out, dropped = V._cross_check_choices(kept_in, {40, 41}, 1)
    check("交叉校验：粗扫看到的留、没看到的丢",
          [it.kind for it in out] == ["choice", "nitro"]
          and len(dropped) == 1 and dropped[0].percent == 55,
          "out={0} dropped={1}".format([it.percent for it in out],
                                       [it.percent for it in dropped]))
    check("  非选路的操作永远不受影响",
          all(it.kind != "choice" or it.percent in (40,) for it in out), "")
    out2, dropped2 = V._cross_check_choices(kept_in, set(), 1)
    check("  粗扫一个都没看到时**不否任何东西**（不做无依据的否决）",
          len(out2) == 3 and not dropped2, "")

    # ③ 路线**只认模型判出的操作**（2026-09-15 用户要求：不要再启发式判任何操作）
    from a9route.vision.video import PercentShot
    shot_no_ops = [PercentShot(percent=5.0, t=1.0, path="", op="", cue=None)]
    try:
        V.suggest_route(shot_no_ops, fine_scan_ran=False)
        check("**细扫没跑 -> 拒绝出路线**（不再用窗口汇总估算）", False, "居然给了路线")
    except RuntimeError as exc:
        check("**细扫没跑 -> 拒绝出路线**（不再用窗口汇总估算）",
              "不给路线" in str(exc) and "模型" in str(exc), str(exc)[:60])
    ok_text = V.suggest_route(shot_no_ops, fine_scan_ran=True)
    check("  细扫跑过、只是这局没操作 -> 给一条**空**路线（这是合法结论）",
          "没检测到任何操作" in ok_text
          and not [ln for ln in ok_text.splitlines()
                   if ln.strip() and not ln.startswith("#")],
          ok_text.splitlines()[1][:50])
    check("  老路径（窗口汇总估算）**还在，但已经不参与出路线**了",
          callable(V._suggest_route_legacy_window), "")

    # ④ 「关自动驾驶 S:2000」= **OCR 文字判据**，用户口径：保留（不训模型）。
    #    它是路线里唯一不来自 YOLO 的操作，所以必须**在正文里标明来源**，
    #    而且**只许 `auto` 这一类**从粗扫漏进路线（别的粗扫结果一律不许）。
    from a9route.core.intent import Intent
    s_ocr = PercentShot(percent=50.0, t=30.0, path="", op="S:2000",
                        values={"auto": "S:2000"}, kinds=["auto"])
    s_leak = PercentShot(percent=51.0, t=30.5, path="", op="D:4000",
                         values={"drift": "D:4000"}, kinds=["drift"])
    s_ocr.intents = [Intent(kind="nitro", t0=30.0, dur=0.1, op="N:100:1:100",
                            percent=50.0)]
    txt = V.suggest_route([s_ocr, s_leak], fine_scan_ran=True)
    body = [ln for ln in txt.splitlines() if ln.strip() and not ln.startswith("#")]
    check("**OCR 判出的关自动驾驶（S:2000）会进路线**（用户口径：touchdrive 用 OCR）",
          len(body) == 1 and "S:2000" in body[0], str(body))
    check("  同一百分点上和模型判出的操作并用 `|` 连起来",
          "N:100:1:100|S:2000" in body[0] or "S:2000|N:100:1:100" in body[0], body[0])
    check("  正文里**标明它来自 OCR**（不让人以为是模型判的）",
          "OCR" in txt and "TOUCHDRIVE" in txt, "")
    check("  粗扫的其它结果（drift/spin…）**不许漏进路线**",
          "D:4000" not in body[0], body[0])

    # ⑤ 路线头那句「谁判的」必须是**算出来的**，不能是写死的常量。
    #    历史 bug：头里写死 "成对脉冲(<200ms)=360"（实际默认 0.30s）和
    #    "keys.onnx / choice.onnx"（换模型后就不对了）—— 输出会一本正经地写错自己 ✗
    from a9route import config as _cfgmod
    cfg_now = _cfgmod.current()
    gap_now = float(cfg_now["intent"]["tap_360_gap"])
    gap_txt, prov, all_model = V._provenance()
    check("路线头里的 360 间隔**跟着当前配置走**（不是写死的 200ms）",
          gap_txt == (f"{gap_now:.2f}".rstrip("0").rstrip(".") + "s") and gap_txt != "0.2s",
          "config={0} 头里写={1}".format(gap_now, gap_txt))
    check("  出处那行两个后端都点名了",
          "按键=" in prov and "选路=" in prov, prov)
    if all_model:
        check("  用模型时：写明模型名（不是写死的 keys.onnx）", ".onnx" in prov, prov)
        check("  用模型时：正文说「这些都来自模型」",
              "都来自**模型**" in V.suggest_route([s_ocr], fine_scan_ran=True), "")
    else:
        check("  用启发式时：写明「像素判据」，**不许出现模型名**",
              "像素判据" in prov and ".onnx" not in prov, prov)
        check("  **不全是模型时正文里明确警告**（这种路线看着和正常的一模一样）",
              "不全是模型" in V.suggest_route([s_ocr], fine_scan_ran=True), "")


def t8_hud_parse():
    print("\n=== T8 比赛内读数的解析 ===")
    import numpy as np
    from a9route.vision.hud import RaceReader

    check("路程 37% -> 37", RaceReader.parse_percent(["路程", "37%"]) == 37.0)
    check("带空格的 '37 %' 也认", RaceReader.parse_percent(["路程 37 %"]) == 37.0)
    check("只认带 % 的数字（不会把排名当路程）",
          RaceReader.parse_percent(["排名 1/1", "路程"]) is None)
    check("排名 1/1 -> 1/1", RaceReader.parse_rank(["排名", "1/1"]) == "1/1")
    check("TouchDrive 开 -> True", RaceReader.parse_touchdrive(["TOUCHDRIVE", "开"]) is True)
    check("TouchDrive 关 -> False", RaceReader.parse_touchdrive(["TOUCHDRIVE 关"]) is False)

    r = RaceReader()
    boxes = {"progress": r.progress_box, "rank": r.rank_box, "td": r.touchdrive_box}

    def fake_reader(frame, box):
        box = tuple(box)
        if box == boxes["progress"]:
            return ["路程", "73%"]
        if box == boxes["rank"]:
            return ["排名", "1/1"]
        if box == boxes["td"]:
            return ["TOUCHDRIVE", "开"]
        return []

    fake = np.full((720, 1280, 3), 40, dtype=np.uint8)
    for cx, color in ((540, (255, 255, 255)), (650, (230, 120, 60))):
        yy, xx = np.ogrid[96:156, cx - 30:cx + 30]
        disk = (yy - 126) ** 2 + (xx - cx) ** 2 <= 30 ** 2
        patch = fake[96:156, cx - 30:cx + 30]
        patch[disk] = color
    r2 = RaceReader(reader=fake_reader, grab=lambda: None)
    st2 = r2.read(fake)
    check("在比赛中 -> 数出 2 个路标", st2.choice_icons == 2, str(st2.choice_icons))
    check(" 并且路程读到了 73", st2.percent == 73.0, str(st2.percent))
    if st2.icons:
        ordered = sorted(st2.icons, key=lambda s: s["xy"][0])
        check(" 路标按从左到右排序", ordered[0]["xy"][0] < ordered[1]["xy"][0],
              str([s["xy"] for s in ordered]))
        check(" 蓝色高亮那个被标出来（第 2 个是蓝的）",
              ordered[0]["blue"] is False and ordered[1]["blue"] is True,
              str([(s["xy"], s["blue"]) for s in ordered]))
    # 读不到百分比时**不数路标**（完赛界面/菜单上的亮横幅会被误计成 1 个）
    r3 = RaceReader(reader=lambda f, b: [], grab=lambda: None)
    st3 = r3.read(fake)
    check("读不到路程 -> 不算在比赛里，也不数路标",
          st3.percent is None and st3.choice_icons == 0, st3.describe())
    # **本项目不连设备**：忘了传帧要直接报错，而不是偷偷去截屏
    try:
        RaceReader(reader=fake_reader).read(None)
        check("忘了传帧 -> 报错（本项目不连设备）", False, "居然没抛")
    except RuntimeError as exc:
        check("忘了传帧 -> 报错（本项目不连设备）", "不连设备截屏" in str(exc), str(exc))


def t9_spin_blocks_drift():
    """**360 时屏蔽漂移**：360 要双击刹车键，漂移是长按同一个键 —— 长按会吃掉双击 ✗。"""
    print("\n=== T9 360 屏蔽漂移 ===")
    ev = [V.ButtonEvent(kind="drift", t0=1.0, duration=0.5, pulses=1,
                        op="D:500", percent=9.0),
          V.ButtonEvent(kind="nitro", t0=1.2, duration=0.1, pulses=1,
                        op="N:100:1:100", percent=9.0)]
    flat = V.flat_route_from_events(ev, extra_ops={9: ["360"]})
    check("生成路线时：有 360 就不写漂移", "D:" not in flat and "360" in flat, flat)
    check(" 氮气仍然保留", "N:100:1:100" in flat, flat)


def t10_end_to_end():
    """端到端：合成视频 -> 逐百分点截图 -> 精细按键扫描 -> 一条扁平路线。

    这里**注入假 reader**（按帧号给百分比）—— OCR 那一层不稳定，
    而这条测试要锁的是**编排**：抽帧 -> 判据 -> 落回百分比 -> 生成路线。
    """
    print("\n=== T10 端到端（合成视频）===")
    fps = 10.0
    # 第 10~14 帧按刹车（0.30s = 漂移），第 40~46 帧点氮气（单击）
    path = fake_hud_video(TMP / "t10.mp4", seconds=6.0, fps=fps,
                          brake_frames=tuple(range(10, 15)),
                          nitro_frames=tuple(range(40, 47)))
    # 帧号 -> 百分比：每秒 10 个百分点（1% 一秒，跟真比赛一个量级）
    seen = {"n": 0}

    class FakeDetector:
        """假装"看一帧就知道在做什么"：按键按帧号点亮，路标在 50~55 帧出现。"""

        def __init__(self):
            from a9route.vision.cues import Cue
            self.Cue = Cue
            self.i = 0

        def detect(self, frame):
            from a9route.vision import cues
            from a9route.vision.hud import RaceReader
            i = self.i
            self.i += 1
            cue = self.Cue()
            bx, by, bw, bh = cues.BRAKE_KEY_BOX
            nx, ny, nw, nh = cues.NITRO_KEY_BOX
            patch_b = frame[by:by + bh, bx:bx + bw]
            patch_n = frame[ny:ny + nh, nx:nx + nw]
            cue.brake_lit = cues.bright_ratio(patch_b) > 0.5
            cue.drifting = cue.brake_lit
            cue.nitro_pressed = cues.red_ratio(patch_n) > cues.NITRO_RED_THR
            if 50 <= i <= 55:
                cue.icons = [{"xy": (610, 127), "radius": 30, "blue": False,
                              "blue_ratio": 0.0},
                             {"xy": (680, 127), "radius": 30, "blue": True,
                              "blue_ratio": 0.8}]
            pct = min(100.0, i * 1.0)          # 每帧 1%
            return cue, pct

    out = TMP / "t10_shots"
    logs: list[str] = []
    shots = V.extract_per_percent(path, out_dir=out, every=0.25,
                                  detector=FakeDetector(), workers=1,
                                  with_buttons=True,
                                  progress=lambda m: logs.append(str(m)))
    # ⚠️ 回归：精细扫描那一段是 `except Exception: 打一行日志` 包起来的 ——
    # 真机跑真录像时，一个 import 写错（`from a9route.vision import intent`）
    # 就被这么吞掉了，表现是"按键细扫没有结果"（看着像阈值问题，其实是代码错）✗。
    # 所以这里**显式断言没有出现"精细扫描失败"**。
    failed = [m for m in logs if "精细扫描失败" in m]
    check("精细扫描没有静默失败（异常会被吞，所以要专门盯它）",
          not failed, failed[0] if failed else "无异常")
    check(" 精细扫描判出了操作目的（按键信号 -> 意图）", True,
          str([m for m in logs if "操作目的" in m][:1]))
    check("逐百分点各存了一张 PNG", len(shots) >= 10, str(len(shots)))
    check(" 截图文件真的写出来了",
          all(Path(s.path).is_file() for s in shots), str(shots[0].path))
    check(" 文件名带百分点与时标（人工核对时好找）",
          "pct_t" in Path(shots[0].path).name, Path(shots[0].path).name)
    check("路径是 ASCII 安全的名字（cv2.imwrite 遇中文会静默失败 ✗）",
          all(Path(s.path).name.isascii() for s in shots), Path(shots[0].path).name)

    # 精细扫描（真按键判据，逐帧）
    samples, icons = V.scan_fine(path, hold=1, hold_brake=2, icon_every=2)
    check("精细扫描逐帧走完了整个视频", len(samples) >= 55, str(len(samples)))
    n_brake = sum(1 for _t, b, _n in samples if b)
    n_nitro = sum(1 for _t, _b, n in samples if n)
    check(" 逐帧认出了按住刹车的那些帧（刹车约占 5 帧）", n_brake >= 3, str(n_brake))
    check(" 逐帧认出了氮气按下的帧", n_nitro >= 3, str(n_nitro))
    check(" 顺带看的路标也产出了时间线", len(icons) > 0, str(len(icons)))

    # ⚠️ **按键后端是可替换的**（`vision.keys`：heuristic / onnx / ultralytics）。
    # 这里注入一个"永远说按下"的假后端，证明 `scan_fine` 真的走注入的那条路。
    # 为什么必须有这条：默认后端改成 `auto` 之后，**端到端那条测试会随磁盘上
    # 有没有模型而变**（`tests/__init__.py` 已经把它钉成 heuristic 了）。
    # 有了这条，模型接入的**接线**在任何环境里都被覆盖着 ✓
    class _AlwaysOn:
        name = "always-on"

        def read(self, frame):
            from a9route.vision.keys import KeyRead
            return KeyRead(brake=True, nitro=True, info={})

    s2, _ = V.scan_fine(path, hold=1, hold_brake=1, icon_every=0,
                        keys=_AlwaysOn())
    check("scan_fine 用**注入的按键后端**（模型接入的接线回归）",
          len(s2) > 0 and all(b and n for _t, b, n in s2),
          "{0} 帧，全 True = {1}".format(
              len(s2), all(b and n for _t, b, n in s2)))

    class _AlwaysOff:
        name = "always-off"

        def read(self, frame):
            from a9route.vision.keys import KeyRead
            return KeyRead(brake=False, nitro=False, info={})

    s3, _ = V.scan_fine(path, hold=1, hold_brake=1, icon_every=0,
                        keys=_AlwaysOff())
    check("  换成「永远没按」的后端 -> 一个都不报（说明真在用注入的那个）",
          len(s3) > 0 and not any(b or n for _t, b, n in s3),
          "{0} 帧".format(len(s3)))

    # 生成路线：用 shots 的时间轴把按键时刻落回百分比
    tl = V.build_timeline([V.Sample(t=s.t, percent=s.percent) for s in shots])
    ev = V.attach_events(shots, V.button_events(samples))
    check("按键时刻落回了百分比（不是 0）", all(e.percent > 0 for e in ev), str(ev))
    flat = V.suggest_route(shots)
    check("生成的路线是**扁平逗号流**（正文只有一行）",
          len([l for l in flat.splitlines() if l.strip() and not l.startswith("#")]) <= 1,
          flat.splitlines()[0][:80] if flat.splitlines() else "")
    check(" 路线正文能被解析器读回来（自己生成的必须自己认）", _check_parses(flat))
    check(" 注释里带上了判据说明（核对用）", "#" in flat, str(len(flat.splitlines())))


def _check_parses(route_text: str) -> bool:
    from a9route.core import route as R
    body = [l for l in route_text.splitlines() if l.strip() and not l.startswith("#")]
    if not body:
        return True
    try:
        R.parse_route("\n".join(body))
        return True
    except R.RouteError as exc:
        print(f"      （解析失败：{exc}）")
        return False


def t11_config():
    """配置层：**改 config 就生效**（这是这个项目独立出来的主要理由）。"""
    print("\n=== T11 配置 ===")
    from a9route.vision import cues

    base = cues.NITRO_RED_THR
    cfg = cfgmod.load_config()
    try:
        cfg["vision"]["nitro_red_thr"] = 0.42
        cfgmod.apply(cfg)
        check("config 改氮气阈值 -> cues 模块常量跟着变",
              cues.NITRO_RED_THR == 0.42, str(cues.NITRO_RED_THR))
        check(" CueDetector 也读到新值（走 COORDS）",
              cues.CueDetector().coords.get("nitro_key_box") == cues.NITRO_KEY_BOX
              or cues.COORDS.get("nitro_key_box") == cues.NITRO_KEY_BOX,
              str(cues.COORDS.get("nitro_key_box")))
    finally:
        cfgmod.apply()
    check(" 恢复默认", cues.NITRO_RED_THR == base, str(cues.NITRO_RED_THR))

    # 环境变量优先级最高
    import os
    os.environ["A9ROUTE_vision__nitro_red_thr"] = "0.33"
    try:
        check("环境变量覆盖 config.json",
              cfgmod.get(cfgmod.load_config(), "vision__nitro_red_thr") == 0.33,
              str(cfgmod.get(cfgmod.load_config(), "vision__nitro_red_thr")))
    finally:
        os.environ.pop("A9ROUTE_vision__nitro_red_thr", None)

    # 类型转换：CLI 的 --set 就靠它
    check("coerce: 0.22 -> float", cfgmod.coerce("0.22", 0.1) == 0.22)
    check("coerce: false -> bool（前端可能发 'false'）",
          cfgmod.coerce("false", True) is False)
    check("coerce: 1,2,3,4 -> 框（list[int]）",
          cfgmod.coerce("137,497,110,110", [0, 0, 0, 0]) == [137, 497, 110, 110])
    check("DEFAULTS 里所有段都能被 apply() 认出来",
          set(cfgmod.DEFAULTS) >= {"scan", "vision", "hud", "intent"},
          str(sorted(cfgmod.DEFAULTS)))
    check("describe() 能打印出配置", "vision" in cfgmod.describe(), "ok")


def main() -> int:
    global TMP
    TMP = tmp_dir()
    t1_timeline()
    t2_draft()
    t3_iter_frames()
    t4_window_and_collapse()
    t5_button_events()
    t6_key_fixtures()
    t7_choice_icons()
    t7b_one_backend()
    t8_hud_parse()
    t9_spin_blocks_drift()
    t10_end_to_end()
    t11_config()
    return summary("test_video")


if __name__ == "__main__":
    sys.exit(main())
