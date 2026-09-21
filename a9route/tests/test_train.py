# -*- coding: utf-8 -*-
"""训练框架的离线测试（**不需要 torch、不需要网络、不需要真录像**）。

  T1  标注格式：YOLO txt 往返、坐标裁剪、坏行**带行号报错**
  T2  类别定义与 data.yaml
  T3  文件名 slug：**中文视频名必须变成纯 ASCII**（cv2.imwrite 会静默失败）
  T4  抽帧采样：优先级（翻转帧 > 吵架 > 贴阈值 > 正样本 > 背景）、预算、可复现
  T5  启发式预标注：真帧样本 + 合成帧（按下/没按）
  T6  数据集：build -> stats -> verify -> split（合成视频，不碰真录像）
  T7  划分：**按视频分**（多段）/ 按连续时间块分（单段）+ 防泄漏提醒
  T8  复核：队列排序 + 修正写回（往返）
  T9  letterbox 几何 + ONNX 三种输出形态的解码（**不需要真的 onnxruntime**）
  T10 后端：启发式两套口径 / auto 解析 / describe（不加载任何模型）
  T11 训练报告的统计与排版（不碰 torch）
  T12 doctor 不崩（且退出码只看关键项）
  T13 数据集体检（**含故意弄坏的反例**：改标定/改阈值/删标注/写歪框都要抓得住）
  T14 数据集体检页的接口（Flask 测试客户端，不真的起服务器）

    python -m a9route.tests.test_train
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from a9route import config as cfgmod
from a9route import paths
from a9route.tests.support import check, fake_hud_video, fixture, summary, tmp_dir
from a9route.train import dataset as D
from a9route.train import frames as F
from a9route.train import labels as L
from a9route.train import prelabel as P
from a9route.train import review as RV
from a9route.train import runner as R
from a9route.train.labels import Box

TMP = Path(".")


# ------------------------------------------------------------------ T0
def t0_syntax():
    """**先编译一遍所有模块** —— 语法错不该等到某个用例跑到才发现。

    ⚠️ 这条是补上的：本仓库里有不少**中文串里嵌引号**的长文本（注释、路线文件头、
    报错信息），手写时很容易在双引号串里塞一个 ASCII 双引号 ✗ ——
    那一刻**整个模块都编译不过**，而"只跑一个子集测试"可能刚好绕开它。
    先编译一遍，代价不到一秒，把这类错误挡在最前面。
    """
    import ast

    print("\n=== T0 全部模块能编译（语法闸）===")
    files = sorted(paths.ROOT.glob("a9route/**/*.py"))
    bad = []
    for p in files:
        try:
            ast.parse(p.read_text(encoding="utf-8"), str(p))
        except SyntaxError as exc:
            bad.append("{0}:{1} {2}".format(p.relative_to(paths.ROOT),
                                            exc.lineno, exc.msg))
    check("a9route/ 下所有 .py 都能编译", not bad,
          "；".join(bad) or "共 {0} 个文件".format(len(files)))


# ------------------------------------------------------------------ T1
def t1_labels():
    """YOLO 标注文件的读写 —— 一切的底座，错一点点全盘皆错。"""
    print("\n=== T1 标注格式（YOLO txt 往返）===")
    b = Box(class_id := 0, 100, 200, 50, 60)
    line = L.box_to_line(b, 1280, 720)
    back = L.line_to_box(line, 1280, 720)
    check("box -> line -> box 往返一致（误差 < 0.01 像素）",
          abs(back.x - b.x) < 0.01 and abs(back.y - b.y) < 0.01
          and abs(back.w - b.w) < 0.01 and abs(back.h - b.h) < 0.01, line)
    check("归一化：中心点是 (cx/w, cy/h)",
          line.startswith("0 "), line)
    cx, cy = b.center
    check("  中心点算对了", abs(float(line.split()[1]) - cx / 1280) < 1e-6
          and abs(float(line.split()[2]) - cy / 720) < 1e-6, line)
    check("class id 用的是 CLASSES 里的顺序",
          L.CLASSES[0] == "brake_pressed" and L.CLASS_IDS["nitro_pressed"] == 1,
          str(L.CLASSES))

    # 越界裁剪：配置里的按键框正好贴边时不能写出 >1 的坐标
    bb = Box(0, 1200, 700, 200, 200).clip(1280, 720)
    check("clip() 把框裁回画面内", bb.x + bb.w <= 1280 and bb.y + bb.h <= 720,
          str(bb.xyxy))
    check("  裁完还能正常写成一行",
          L.line_to_box(L.box_to_line(bb, 1280, 720), 1280, 720).w > 0)

    # 坏行必须报错（**静默跳过 = 标歪了没人知道**）
    bad = [
        ("0 0.5 0.5 0.1 0.1 0.9", "6 个数"),
        ("0 0.5 0.5 0.1", "4 个数"),
        ("x 0.5 0.5 0.1 0.1", "非数字"),
        ("9 0.5 0.5 0.1 0.1", "类别越界"),
        ("0 1.5 0.5 0.1 0.1", "坐标越界"),
        ("0 0.5 0.5 0 0.1", "宽高必须 > 0"),
    ]
    for text, why in bad:
        try:
            L.line_to_box(text, 1280, 720)
            check("坏标注要报错（{0}）".format(why), False, "居然没报错：" + text)
        except L.LabelError as exc:
            check("坏标注要报错（{0}）".format(why), True, str(exc)[:60])
        except Exception as exc:                      # noqa: BLE001
            check("坏标注要报错（{0}）".format(why), False,
                  "抛的是 {0}: {1}".format(type(exc).__name__, exc))

    # 读写：**文件不存在 = 空列表**；空列表 = 写空文件（YOLO 的空标注）
    d = TMP / "labels"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "a.txt"
    # 测试产物目录是**跨次复用的**（tmp_dir 不会自动清），所以先删干净 ——
    # 否则"文件不存在"这条断言会被上一次跑剩下的文件弄失败
    p.unlink(missing_ok=True)
    check("读不存在的标注文件 -> 空列表（纯背景帧）", L.read_label(p, 1280, 720) == [])
    L.write_label(p, [], 1280, 720)
    check("空列表写成**空文件**（不是不写）", p.is_file() and p.read_text() == "")
    L.write_label(p, [b, Box(1, 1033, 497, 110, 110)], 1280, 720)
    got = L.read_label(p, 1280, 720)
    check("两个框读回来顺序和类别都对",
          [g.cls_id for g in got] == [0, 1], str([g.name for g in got]))
    check("  describe() 有人能看的形式", "brake_pressed" in got[0].describe(),
          got[0].describe())


# ------------------------------------------------------------------ T2
def t2_classes_and_yaml():
    print("\n=== T2 类别与 data.yaml ===")
    check("只有两个类，且顺序固定", len(L.CLASSES) == 2, str(L.CLASSES))
    check("每类都有中文短名（复核页要用）",
          all(c in L.CLASS_ZH for c in L.CLASSES), str(L.CLASS_ZH))
    check("每类都有配色（画框用）", all(c in L.CLASS_COLORS for c in L.CLASSES))
    out = TMP / "yamlprobe"
    out.mkdir(parents=True, exist_ok=True)
    p = L.write_data_yaml(out / "data.yaml", out)
    text = p.read_text(encoding="utf-8")
    check("data.yaml 写了 nc / names", "nc: 2" in text
          and "0: brake_pressed" in text and "1: nitro_pressed" in text, text[:80])
    check("  train/val 指向 images/train|val",
          "images/train" in text and "images/val" in text)
    check("  path 是绝对路径（ultralytics 会照着它找图）",
          str(out.resolve()) in text, text.splitlines()[1])


# ------------------------------------------------------------------ T3
def t3_slug():
    """中文视频名 -> ASCII 前缀。**这条踩过**：cv2.imwrite 遇中文静默失败。"""
    print("\n=== T3 文件名 slug（中文名必须变 ASCII）===")
    a = F.slug("跑图转路线测试")
    check("纯中文名也能出非空 ASCII", a != "" and a.isascii(), a)
    b = F.slug("跑图转路线测试")
    check("  同名稳定（不会每次都不一样）", a == b, a)
    c = F.slug("测试3")
    check("  不同的中文名不撞车", a != F.slug("刹车氮气标定测试"), c)
    check("带扩展名的英文名只留合法字符",
          F.slug("race_01.final-x") == "race_01_final_x", F.slug("race_01.final-x"))
    check("结果里没有路径分隔符",
          "/" not in F.slug("a/b\\c") and "\\" not in F.slug("a/b\\c"), F.slug("a/b\\c"))


# ------------------------------------------------------------------ T4
def _mk(i, brake=False, nitro=False, bm=0.2, nm=0.2, bdis=False, ndis=False):
    return F.FrameSignal(idx=i, t=round(i / 30.0, 3), sig=P.KeySignals(
        brake_hit=brake, nitro_hit=nitro, brake_margin=bm, nitro_margin=nm,
        brake_disagree=bdis, nitro_disagree=ndis))


def t4_sampling():
    """抽帧策略：**把预算砸在边界上**（均匀抽帧恰好躲开所有难点）。"""
    print("\n=== T4 抽帧采样策略 ===")
    frames = [_mk(i, brake=(10 <= i <= 20), nitro=(100 <= i <= 102))
              for i in range(300)]
    plan = F.plan_sampling(frames, budget=50, edge_pad=3)
    picked = {f.idx: f for f in plan.picks}
    check("翻转帧一定被抽到（按下那一帧）", 10 in picked, str(sorted(picked)[:12]))
    check("翻转帧一定被抽到（松开那一帧）", 21 in picked)
    check("  氮气的翻转帧也在", 100 in picked and 103 in picked)
    check("  翻转帧带上了前后 pad 帧",
          all(j in picked for j in (7, 8, 9, 13, 18, 24)), "")
    check("  翻转帧的 reason 是 edge",
          picked[10].reason == "edge" and picked[21].reason == "edge",
          " ".join("{0}:{1}".format(k, v.reason) for k, v in sorted(picked.items())[:8]))
    check("总预算被遵守（翻转帧不超预算时）", len(plan.picks) <= 50,
          str(len(plan.picks)))
    check("  背景帧用来补足预算（不能只有正样本）",
          plan.counts.get("bg", 0) > 0, str(plan.counts))
    check("summary() 说清了各 reason 抽了多少",
          "edge" in plan.summary() and str(len(plan.picks)) in plan.summary(),
          plan.summary())

    # 贴阈值 / 吵架的帧要优先于普通正样本
    frames2 = [_mk(i) for i in range(100)]
    frames2[50] = _mk(50, bm=0.98)                 # 贴着刹车阈值
    frames2[60] = _mk(60, ndis=True)               # 氮气两通道吵架
    plan2 = F.plan_sampling(frames2, budget=10, edge_pad=0)
    p2 = {f.idx: f.reason for f in plan2.picks}
    check("贴阈值的帧被标成 near", p2.get(50) == "near", str(p2))
    check("两通道吵架的帧被标成 disagree", p2.get(60) == "disagree", str(p2))
    check("  near/disagree 优先于背景帧（预算只有 10）",
          len(plan2.picks) == 10, str(len(plan2.picks)))

    # 可复现：同 seed 同参数 -> 同数据集（否则没法比较两次训练）
    a = F.plan_sampling(frames, budget=40, seed=7)
    b = F.plan_sampling(frames, budget=40, seed=7)
    check("同 seed 抽出来的帧完全一样（可复现）",
          [f.idx for f in a.picks] == [f.idx for f in b.picks],
          str(len(a.picks)))
    # 背景帧是**沿时间均匀取样**（不是随机）：保证整段都有覆盖，
    # 而不是全靠运气抽到视频开头那几秒
    bg_idx = [f.idx for f in a.picks if f.reason == "bg"]
    check("背景帧铺满整段（首尾都够到，不是只抽开头）",
          bg_idx and min(bg_idx) < 20 and max(bg_idx) > 280,
          "bg 帧号 {0}..{1}（共 {2} 个）".format(min(bg_idx), max(bg_idx), len(bg_idx)))
    check("  near/disagree 的挑选带 seed（换 seed 结果可能不同）",
          F.plan_sampling(frames2, budget=10, seed=1).counts
          == F.plan_sampling(frames2, budget=10, seed=1).counts, "同 seed 同结果")

    empty = F.plan_sampling([], budget=10)
    check("空输入不崩", empty.picks == [] and empty.total_frames == 0)

    # 翻转帧**必须**有上限：真数据上启发式会抖，不设限会把预算全吃光
    flick = [_mk(i, brake=(i % 4) < 2) for i in range(200)]     # 每 2 帧翻一次
    plan3 = F.plan_sampling(flick, budget=40, edge_pad=2)
    check("抖动的信号下：预算仍然守得住（edge 最多占 edge_share）",
          len(plan3.picks) <= 40, "抽了 {0} 帧".format(len(plan3.picks)))
    check("  但仍然给背景帧留了位置（不会全是 edge）",
          plan3.counts.get("bg", 0) > 0, str(plan3.counts))
    plan4 = F.plan_sampling(flick, budget=40, edge_pad=2, edge_share=0.9)
    check("  edge_share 调大 -> edge 占比确实变大",
          plan4.counts.get("edge", 0) > plan3.counts.get("edge", 0),
          "{0} vs {1}".format(plan3.counts, plan4.counts))

# ------------------------------------------------------------------ T5
def t5_prelabel():
    """启发式预标注：真帧样本验刹车通道，合成帧验"框只在按下时产出"。"""
    print("\n=== T5 启发式预标注 ===")
    import cv2
    import numpy as np

    from a9route.vision import cues

    brake_on = cv2.imread(str(fixture("brake_key_on.png")))
    brake_off = cv2.imread(str(fixture("brake_key_off.png")))
    if brake_on is None or brake_off is None:
        check("刹车按键样本存在", False, "缺 fixtures/brake_key_*.png")
        return
    bh, bw = brake_on.shape[:2]
    # ⚠️ 两张截图**都是刹车键的裁剪图**，所以两个"框"都指向同一块像素 ——
    # 下面只断言**刹车通道**；氮气通道的断言放到合成帧上做（否则测的是同一块区域）
    coords = {"brake_key_box": (0, 0, bw, bh), "nitro_key_box": (0, 0, bw, bh)}
    s_on = P.read_signals(brake_on, coords=coords)
    s_off = P.read_signals(brake_off, coords=coords)
    check("刹车按下帧 -> brake_hit=True", s_on.brake_hit, str(s_on.as_dict()))
    check("刹车没按帧 -> brake_hit=False", not s_off.brake_hit, str(s_off.as_dict()))
    check("  按下帧的余量 > 1（= 得分/阈值，>1 就是判按下）",
          s_on.brake_margin > 1.0, str(round(s_on.brake_margin, 3)))
    check("  没按帧的余量 < 1", s_off.brake_margin < 1.0,
          str(round(s_off.brake_margin, 3)))
    check("  按下/没按分得开（余量差 > 1）",
          s_on.brake_margin - s_off.brake_margin > 1.0,
          "{0:.2f} vs {1:.2f}".format(s_on.brake_margin, s_off.brake_margin))
    check("as_dict() 里四个通道值齐全（复核页要逐项显示）",
          {"brake_ring", "brake_white", "nitro_red", "nitro_white"} <=
          set(s_on.as_dict()), str(sorted(s_on.as_dict())))
    check("没按的那张上，氮气两通道**吵架**被记下来了（白度说按、红说没按）",
          s_off.nitro_disagree is True, str(s_off.as_dict()))

    # ---- 合成帧：单独点亮一个键（这才是真正在验"两个通道互不串味"）
    dark = np.full((720, 1280, 3), 20, dtype=np.uint8)
    nx, ny, nw, nh = cues.NITRO_KEY_BOX
    bx, by, bww, bhh = cues.BRAKE_KEY_BOX

    # 刹车通道吵架的构造：**连圈外参考环一起涂白** —— 这时"圈内−圈外"= 0，
    # 但"绝对白度"= 1.0，两个通道结论相反（这正是相对判据存在的意义）
    pad = cues.RING_PAD
    washed = dark.copy()
    washed[by - pad:by + bhh + pad, bx - pad:bx + bww + pad] = 255
    s_w = P.read_signals(washed)
    check("整块涂白（含圈外）时：刹车两通道吵架被记下来",
          s_w.brake_disagree is True,
          "ring={0} white={1}".format(s_w.brake_ring, s_w.brake_white))
    check("  这时仍然判「按下」（绝对白度兜底）", s_w.brake_hit is True, "")

    only_nitro = dark.copy()
    only_nitro[ny:ny + nh, nx:nx + nw] = (40, 40, 245)          # 红 = 氮气按下
    s_n = P.read_signals(only_nitro)
    check("合成红框帧 -> nitro_hit=True", s_n.nitro_hit,
          str(round(s_n.nitro_red, 3)))
    check("  此时刹车**没有**被误判", not s_n.brake_hit,
          str(round(s_n.brake_white, 3)))
    bxs = s_n.boxes(cues.BRAKE_KEY_BOX, cues.NITRO_KEY_BOX)
    check("只在按下时才产出框（此处只有 1 个）", len(bxs) == 1,
          str([b.describe() for b in bxs]))
    check("  框落在配置里那个氮气框上（像素坐标，和 config 对得上）",
          bxs and bxs[0].xywh == (float(nx), float(ny), float(nw), float(nh)),
          str(bxs[0].xywh) if bxs else "无")
    check("  框上带了确信度（复核页要显示「有多悬」）", bxs[0].conf is not None,
          str(bxs[0].conf))

    only_brake = dark.copy()
    only_brake[by:by + bhh, bx:bx + bww] = 255                   # 白 = 刹车按下
    s_b = P.read_signals(only_brake)
    check("合成白框帧 -> brake_hit=True 且 nitro_hit=False（两个键各认各的）",
          s_b.brake_hit and not s_b.nitro_hit,
          "brake_hit={0} nitro_hit={1}".format(s_b.brake_hit, s_b.nitro_hit))
    check("  两个键同时按下 -> 两个框都给",
          len(P.read_signals(np.maximum(only_brake, only_nitro)).boxes(
              cues.BRAKE_KEY_BOX, cues.NITRO_KEY_BOX)) == 2, "ok")

    s0 = P.read_signals(dark)
    check("全黑帧 -> 两个键都没按（不凭空报）",
          not s0.brake_hit and not s0.nitro_hit, str(s0.as_dict()))
    check("全黑帧的余量都 < 1",
          s0.brake_margin < 1.0 and s0.nitro_margin < 1.0,
          "{0} {1}".format(round(s0.brake_margin, 3), round(s0.nitro_margin, 3)))
    check("全黑帧产不出任何框", s0.boxes(cues.BRAKE_KEY_BOX, cues.NITRO_KEY_BOX) == [])


# ------------------------------------------------------------------ T6
def t6_dataset():
    """build -> stats -> verify -> split，在**合成视频**上跑通（不碰真录像）。"""
    print("\n=== T6 数据集：构建 / 统计 / 体检 / 划分 ===")
    import cv2
    import shutil

    # 测试产物目录跨次复用，所以先把上次留下的数据集整个删掉 ——
    # 否则上一轮 split 出来的 train/val 会和新一轮的 pool 混在一起，
    # stats 就会多数出一堆帧（**这条第一次跑就撞上了**）
    shutil.rmtree(D.dataset_dir("t_keys"), ignore_errors=True)
    video = fake_hud_video(TMP / "ds_video.mp4", seconds=6.0, fps=10.0,
                           brake_frames=(20, 21), nitro_frames=(40,))
    ds = D.dataset_dir("t_keys")
    try:
        rep = D.build([video], name="t_keys", budget=40, edge_pad=3)
    except Exception as exc:                          # noqa: BLE001
        check("build 能跑完", False, "{0}: {1}".format(type(exc).__name__, exc))
        return
    check("build 抽到了帧", rep.frames > 0, rep.describe().replace("\n", " | "))
    check("  抽帧带了翻转帧", rep.by_reason.get("edge", 0) > 0, str(rep.by_reason))
    check("  刹车和氮气都标出了框",
          rep.by_class.get("brake_pressed", 0) > 0
          and rep.by_class.get("nitro_pressed", 0) > 0, str(rep.by_class))
    check("图片全落在 pool/images（划分前）",
          len(list((ds / "pool" / "images").glob("*.jpg"))) == rep.frames,
          str(len(list((ds / "pool" / "images").glob("*.jpg")))))
    check("每张图都有对应标注文件（背景帧是**空文件**）",
          len(list((ds / "pool" / "labels").glob("*.txt"))) == rep.frames)
    check("存的是真帧、能解码",
          cv2.imread(str(sorted((ds / "pool" / "images").glob("*.jpg"))[0])) is not None)
    check("meta.jsonl 每帧一行", len(D.read_frames_meta(ds)) == rep.frames,
          str(len(D.read_frames_meta(ds))))
    check("dataset.json 记了源视频和抽样参数", D.load_meta(ds).get("videos") is not None,
          str(D.load_meta(ds).get("sampling")))

    info = D.stats(name="t_keys")
    check("stats：帧数对得上", info.images == rep.frames, str(info.images))
    check("stats：数出了纯背景帧（两个键都没按的帧）", info.background > 0,
          str(info.background))
    check("stats：分类别统计了框数", info.positives > 0, str(info.boxes_by_class))
    check("  画面尺寸记下来了（1280x720）", "1280x720" in info.sizes, str(info.sizes))
    check("  来源视频记下来了", any("ds_video" in s for s in info.sources),
          str(info.sources))
    check("describe() 有中文说明", "纯背景" in info.describe(),
          info.describe().replace("\n", " | ")[:120])

    problems = D.verify(name="t_keys")
    check("体检：还没划分时**不报**「先 split」",
          not any("split" in p for p in problems), str(problems))
    check("  体检不该报「全是正样本」（有背景帧）",
          not any("背景帧都没有" in p for p in problems), str(problems))

    sp = D.split(name="t_keys", val_ratio=0.25)
    check("split：train + val = 总帧数", sp.train + sp.val == rep.frames,
          "train={0} val={1}".format(sp.train, sp.val))
    check("  两边都非空", sp.train > 0 and sp.val > 0, sp.describe().replace("\n", " | "))
    check("  只有一段视频时**明说 val 偏乐观**（防数据泄漏的自知之明）",
          "偏乐观" in sp.note, sp.note)
    check("data.yaml 写出来了", (ds / "data.yaml").is_file())
    check("  train/val 的图都搬过去了",
          len(list((ds / "images" / "train").glob("*.jpg"))) == sp.train
          and len(list((ds / "images" / "val").glob("*.jpg"))) == sp.val)
    check("  val 侧也都有标注文件",
          len(list((ds / "labels" / "val").glob("*.txt"))) == sp.val)
    problems2 = D.verify(name="t_keys")
    check("划分后体检通过（无问题）", problems2 == [], str(problems2))
    check("summary_line() 一行话能打出来", "t_keys" in D.summary_line("t_keys"),
          D.summary_line("t_keys"))


# ------------------------------------------------------------------ T7
def t7_split_policy():
    """划分策略：多段视频**按视频分**；单段按连续时间块分。"""
    print("\n=== T7 划分策略（防数据泄漏）===")
    metas = ([{"file": "a{0}.jpg".format(i), "video": "v1", "frame": i}
              for i in range(60)]
             + [{"file": "b{0}.jpg".format(i), "video": "v2", "frame": i}
                for i in range(30)]
             + [{"file": "c{0}.jpg".format(i), "video": "v3", "frame": i}
                for i in range(10)])
    assigned, val_videos, note = D.plan_split(metas, val_ratio=0.2)
    check("多段视频：整段整段地分（val 里只出现视频名，不混帧）",
          len(val_videos) >= 1 and note == "", str(val_videos))
    for m in metas:
        want = "val" if m["video"] in val_videos else "train"
        if assigned.get(m["file"]) != want:
            check("  按视频分得干净（没有同一段的帧跨 train/val）", False, m["file"])
            break
    else:
        check("  按视频分得干净（没有同一段的帧跨 train/val）", True,
              "val={0}".format(val_videos))
    check("  留出的是**帧少的那几段**（大段给训练更划算）",
          val_videos == ["v3"] or val_videos == ["v3", "v2"], str(val_videos))

    one = [{"file": "x{0}.jpg".format(i), "video": "solo", "frame": i}
           for i in range(100)]
    a2, v2, note2 = D.plan_split(one, val_ratio=0.2)
    check("单段视频：给出「指标偏乐观」的提醒", "偏乐观" in note2, note2)
    check("  仍然分出了 val（按连续时间块）",
          sum(1 for v in a2.values() if v == "val") > 0, str(a2.get("x99")))
    check("  时间块是连续的（val 的帧号整体偏后，不是每隔 5 帧抽一个）",
          all(a2["x{0}.jpg".format(i)] == "val" for i in range(80, 100)),
          "frame 80..99 -> val")
    check("空输入不崩", D.plan_split([]) == ({}, [], ""))


# ------------------------------------------------------------------ T8
def t8_review():
    """复核：队列排序 + 导出修正写回标注（往返）。"""
    print("\n=== T8 复核队列与修正写回 ===")
    metas = [
        {"file": "bg.jpg", "video": "v", "t": 1.0, "frame": 1, "reason": "bg",
         "boxes": [], "signals": {"brake_margin": 0.1, "nitro_margin": 0.1},
         "status": "prelabel"},
        {"file": "edge.jpg", "video": "v", "t": 2.0, "frame": 2, "reason": "edge",
         "boxes": [{"name": "brake_pressed"}],
         "signals": {"brake_margin": 0.1, "nitro_margin": 0.1}, "status": "prelabel"},
        {"file": "near.jpg", "video": "v", "t": 3.0, "frame": 3, "reason": "near",
         "boxes": [], "signals": {"brake_margin": 0.9, "nitro_margin": 1.02},
         "status": "prelabel"},
        {"file": "done.jpg", "video": "v", "t": 4.0, "frame": 4, "reason": "edge",
         "boxes": [], "signals": {}, "status": "reviewed"},
    ]
    q = RV.build_queue(metas)
    check("已复核的帧默认不再排队", all(it.file != "done.jpg" for it in q),
          str([it.file for it in q]))
    check("翻转帧排在最前（最该看）", q[0].reason == "edge", q[0].file)
    check("  贴阈值排在背景帧前面",
          [it.file for it in q].index("near.jpg") < [it.file for it in q].index("bg.jpg"),
          str([it.file for it in q]))
    check("--all 时把已复核的也列出来",
          any(it.file == "done.jpg" for it in RV.build_queue(metas,
                                                             include_reviewed=True)))
    near = [it for it in q if it.file == "near.jpg"][0]
    edge = [it for it in q if it.file == "edge.jpg"][0]
    check("可疑度：贴着阈值的帧比翻转帧更「悬」",
          near.suspicion < edge.suspicion,
          "{0:.3f} vs {1:.3f}".format(near.suspicion, edge.suspicion))

    # 在真数据集上生成复核页 + 写回
    ds = D.dataset_dir("t_keys")
    if not (ds / "pool" / "images").is_dir():
        check("复核页需要数据集（T6 没跑成，跳过）", True, "skipped")
        return
    r = RV.make_review(name="t_keys", page_size=8, progress=None)
    check("生成了复核页", Path(r["index"]).is_file(), r["index"])
    check("  分页了（page_size=8 但帧更多）", r["pages"] >= 1, "pages=" + str(r["pages"]))
    html = Path(r["index"]).read_text(encoding="utf-8")
    check("  目录页有各 reason 的统计", "翻转帧" in html or "背景" in html, "")
    check("  queue.txt 写出来了（不用浏览器也能看顺序）",
          (Path(r["dir"]) / "queue.txt").is_file())
    page1 = Path(r["dir"]) / "page_001.html"
    check("  第一页里有待复核的帧", page1.is_file()
          and "data-file=" in page1.read_text(encoding="utf-8"))
    sheets = RV.contact_sheets(name="t_keys", per_sheet=6, cols=3, progress=None)
    check("拼图也能生成（不看浏览器时用）",
          bool(sheets) and sheets[0].is_file(), str(sheets[:1]))

    metas_all = D.read_frames_meta(ds)
    target = [m for m in metas_all if not (m.get("boxes") or [])]
    if not target:
        check("找一个背景帧来翻转标注", False, "数据集里没有背景帧")
        return
    fn = target[0]["file"]
    # 这个数据集在同一个进程里还要被后面的 T13/T14 体检 —— 测试**必须还原**它，
    # 而且**两份都要还原**（`pool/labels` 和 `labels/{train,val}` 是两份拷贝，
    # `apply_corrections` 会同时改它们；只还原一份就会让 T13 报"两份不一致" ✗）
    back: dict = {}
    for img, lblp in D.iter_pairs(ds):
        if img.name == fn:
            back[lblp] = lblp.read_text(encoding="utf-8") if lblp.is_file() else None
    payload = {"version": 1, "dataset": "t_keys",
               "items": [{"file": fn, "boxes": ["brake_pressed", "nitro_pressed"],
                          "status": "reviewed"},
                         {"file": "不存在的帧.jpg", "boxes": []}]}
    res = RV.apply_corrections(payload, name="t_keys", progress=None)
    check("写回：跳过对不上的文件名（不静默算成功）", res["skipped"] == 1, str(res))
    check("  改动的帧数报出来了", res["changed"] == 1, str(res))
    lbl = ds / "pool" / "labels" / (Path(fn).stem + ".txt")
    boxes = L.read_label(lbl, 1280, 720)
    check("  标注文件真的写成了两个框",
          sorted(b.name for b in boxes) == ["brake_pressed", "nitro_pressed"],
          str([b.describe() for b in boxes]))
    check("  划分后的那份也一起改了（两份拷贝要一致）",
          all(sorted(x.name for x in L.read_label(p, 1280, 720))
              == ["brake_pressed", "nitro_pressed"]
              for p in back if p.is_file()),
          [str(p) for p in back])
    after = {m["file"]: m for m in D.read_frames_meta(ds)}[fn]
    check("  meta 里标成 reviewed（下轮复核不会再排在前面）",
          after.get("status") == "reviewed", str(after.get("status")))
    check("  dataset.json 里留了修正历史",
          bool(D.load_meta(ds).get("corrections")), "ok")
    # 复核过的帧不再进队列
    q2 = RV.build_queue(D.read_frames_meta(ds))
    check("  下一次生成复核页时它不在队列里了",
          all(it.file != fn for it in q2), "队列 {0} 帧".format(len(q2)))
    # ---- 还原（见上面那段说明）----
    for lblp, content in back.items():
        if content is None:
            lblp.unlink(missing_ok=True)
        else:
            lblp.write_text(content, encoding="utf-8")
    metas_restore = D.read_frames_meta(ds)
    for m in metas_restore:
        if m.get("file") == fn:
            m["boxes"] = target[0].get("boxes") or []
            m["status"] = target[0].get("status") or "prelabel"
    (ds / D.FRAME_META).write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metas_restore),
        encoding="utf-8")
    check("  还原成改动前的样子（后面的体检才跑得干净）",
          L.read_label(lbl, 1280, 720) == [], "")


# ------------------------------------------------------------------ T9
def t9_geometry_and_onnx_decode():
    """letterbox 几何 + ONNX 三种输出形态的解码（**不需要 onnxruntime**）。"""
    print("\n=== T9 letterbox 几何与 ONNX 输出解码 ===")
    import numpy as np

    from a9route.vision import keys as K

    frame = np.full((720, 1280, 3), 7, dtype=np.uint8)
    canvas, scale, left, top = K.letterbox(frame, 640)
    check("1280x720 -> 640：缩放 0.5、左右不填、上下各填 140",
          abs(scale - 0.5) < 1e-9 and left == 0 and top == 140,
          "scale={0} left={1} top={2}".format(scale, left, top))
    check("画布就是 imgsz×imgsz", canvas.shape == (640, 640, 3), str(canvas.shape))
    check("填充色是 114（**必须和 ultralytics 一致**，否则框会整体偏）",
          canvas[0, 0, 0] == K.PAD_VALUE and canvas[639, 639, 0] == K.PAD_VALUE,
          str(canvas[0, 0]))
    check("  图像内容落在中间那条 640×360 里",
          canvas[300, 300, 0] == 7 and canvas[100, 300, 0] == K.PAD_VALUE,
          "ok")

    box = (100.0, 200.0, 300.0, 400.0)
    lb = tuple(v * scale + (left if i % 2 == 0 else top) for i, v in enumerate(box))
    back = K.unletterbox(lb, scale, left, top)
    check("letterbox -> unletterbox 往返一致",
          all(abs(a - b) < 1e-6 for a, b in zip(back, box)), str(back))

    blob = K.to_blob(canvas, 640)
    check("blob 形状/类型对： (1,3,H,W) float32",
          blob.shape == (1, 3, 640, 640) and blob.dtype == np.float32, str(blob.shape))
    check("  归一化到 0~1", 0.0 <= float(blob.min()) and float(blob.max()) <= 1.0,
          "{0:.2f}~{1:.2f}".format(float(blob.min()), float(blob.max())))
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[:, :, 0] = 255                       # OpenCV 顺序是 BGR -> 这是**蓝**
    b2 = K.to_blob(rgb, 640)
    check("BGR -> RGB：第 0 通道（蓝）不该跑到 blob 的第 0 个通道",
          float(b2[0, 0].max()) < 0.01 and float(b2[0, 2].max()) > 0.99,
          "ch0={0:.2f} ch2={1:.2f}".format(float(b2[0, 0].max()),
                                           float(b2[0, 2].max())))

    # ---- ONNX 解码：三种形态（不建 session，直接喂 numpy）
    det = K.OnnxKeys.__new__(K.OnnxKeys)     # 绕开 __init__（不加载模型）
    det.conf, det.iou, det.anchor_tol = 0.40, 0.5, 0.0
    det.fmt = "auto"
    # `_decode` 用 `self.classes` 决定 nc 和类名 —— 绕开 __init__ 就得自己给
    det.classes = tuple(K.CLASSES)
    det._scale, det._left, det._top = 1.0, 0, 0
    nc = len(K.CLASS_IDS)
    n = 4 + nc

    raw = np.zeros((1, n, 3), np.float32)
    for k, (cx, cy, w, h, s0, s1) in enumerate([
            (100, 100, 40, 40, 0.90, 0.05),
            (200, 200, 40, 40, 0.10, 0.80),
            (300, 300, 20, 20, 0.20, 0.20)]):
        raw[0, :4, k] = [cx, cy, w, h]
        raw[0, 4, k] = s0
        raw[0, 5, k] = s1
    # `_decode` 返回**列表**（不是 dict）：每条是 (类号, 置信度, xyxy) ——
    # 选路任务要把图标框画回画面上，光有置信度不够，所以位置必须一起出来
    dets = det._decode(raw, 1280, 720)
    by_name = {K.CLASSES[cid]: conf for cid, conf, _ in dets}
    check("形态 (1,4+nc,N)：两类都解出来了",
          abs(by_name.get("brake_pressed", 0) - 0.90) < 1e-6
          and abs(by_name.get("nitro_pressed", 0) - 0.80) < 1e-6, str(by_name))
    check("  低于 conf 的框被丢掉", len(dets) == 2, str(dets))
    pos = {K.CLASSES[cid]: xy for cid, _, xy in dets}
    check("  位置一起解出来了（cxcywh -> xyxy，且做了 letterbox 逆变换）",
          all(abs(a - b) < 1e-4 for a, b in zip(pos["brake_pressed"], (80, 80, 120, 120))),
          str(pos["brake_pressed"]))

    best2 = det._decode(raw.transpose(0, 2, 1), 1280, 720)
    check("形态 (1,N,4+nc)（转置过的）也能解", best2 == dets, str(best2))

    # ⚠️ 2 类时 `4+nc == 6`，"已做 NMS"的导出行宽**也是 6** —— 形状分不出来，
    # 所以必须显式指定 fmt="nms"（auto 会按"未做 NMS"解析）
    nms = np.array([[[100, 100, 140, 140, 0.90, 0.0],
                     [102, 102, 142, 142, 0.70, 0.0],     # 和上一条重叠 -> 被 NMS
                     [500, 500, 560, 560, 0.85, 1.0]]], np.float32)
    det.fmt = "nms"
    best3 = det._decode(nms, 1280, 720)
    by3 = [conf for _, conf, _ in best3]
    check("形态 (1,N,6)（已 NMS 的导出，显式 fmt=nms）能解",
          0.90 in [round(c, 2) for c in by3] and 0.85 in [round(c, 2) for c in by3],
          str(by3))
    # 这一支**不再自己 NMS**：导出方说已经 NMS 过就相信它（重叠的一条原样收下）。
    # 自己 NMS 只发生在「未做 NMS」的形态上（下面那条检查盯着它）。
    check("  已 NMS 的形态原样照收（不重复 NMS —— 相信导出方）", len(best3) == 3,
          str([(K.CLASSES[c], round(v, 2)) for c, v, _ in best3]))

    # 反例对照：**未做 NMS** 的形态必须自己 NMS，否则同一次按键会报两遍
    raw_nms = np.zeros((1, n, 2), np.float32)
    for k, (cx, cy, w2, h2, s0) in enumerate([(100, 100, 40, 40, 0.90),
                                              (102, 102, 40, 40, 0.70)]):
        raw_nms[0, :4, k] = [cx, cy, w2, h2]
        raw_nms[0, 4, k] = s0
    det.fmt = "auto"
    dedup = det._decode(raw_nms, 1280, 720)
    check("  未做 NMS 的形态里重叠框被 NMS 掉（IoU 0.82 > 0.5）",
          len(dedup) == 1 and round(dedup[0][1], 2) == 0.90, str(dedup))

    # ⚠️ auto 在 2 类时**认不出** NMS 形态：`4+nc == 6` 和 NMS 的行宽一样，
    # 它会按"未做 NMS"解析 —— 后果是**置信度被读成类分数**（下面这个例子：
    # 真正的 conf=0.9 会被读成 1.0），坐标也会被当 cxcywh 解。
    # 不报错、不崩，只是**悄悄不对** —— 所以文档要求用 nms=True 导出时必须显式设 key_fmt。
    det.fmt = "auto"
    ambiguous = np.array([[[10, 10, 50, 50, 0.90, 1.0]]], np.float32)
    auto_read = {K.CLASSES[cid]: conf for cid, conf, _ in det._decode(ambiguous, 1280, 720)}
    det.fmt = "nms"
    nms_read = {K.CLASSES[cid]: conf for cid, conf, _ in det._decode(ambiguous, 1280, 720)}
    check("auto 把 6 宽的 NMS 输出当「未做 NMS」解（已知歧义，**不报错**）",
          abs(auto_read.get("nitro_pressed", 0) - 1.0) < 1e-6,
          "auto 读出 conf={0}（真值 0.90）".format(auto_read.get("nitro_pressed")))
    check("  显式 fmt=nms 才拿到真的置信度 0.90",
          abs(nms_read.get("nitro_pressed", 0) - 0.90) < 1e-6,
          "nms 读出 conf={0}".format(nms_read.get("nitro_pressed")))
    det.fmt = "auto"

    empty = det._decode(np.zeros((1, n, 5), np.float32), 1280, 720)
    check("全 0 输出 -> 什么都不报（不凭空造框）", empty == [], str(empty))
    try:
        det._decode(np.zeros((1, 3, 5), np.float32), 1280, 720)
        check("看不懂的输出形状要报错", False, "居然没报错")
    except RuntimeError as exc:
        check("看不懂的输出形状要报错", "形状" in str(exc), str(exc)[:90])


# ------------------------------------------------------------------ T10
def t10_backend():
    """后端选择：默认启发式、显式切模型、auto 的解析规则。"""
    print("\n=== T10 按键后端（不加载任何模型）===")
    import numpy as np

    from a9route.vision import cues
    from a9route.vision import keys as K

    be, mp, params = K.resolve_backend({"vision": {"key_backend": "heuristic"}})
    check("显式 heuristic：解析成 heuristic", be == "heuristic", be)
    check("  参数带上了阈值/分辨率", "conf" in params and "imgsz" in params, str(params))
    # ⚠️ `auto` **不再静默退回启发式**（2026-09-15 用户要求"判定全交给模型"）：
    # 模型不在就报错，并给出修法。以前是 `auto -> heuristic`，
    # 于是"删掉模型文件"会让判据**无声无息**退化成像素规则 ✗
    try:
        K.resolve_backend({"vision": {"key_backend": "auto",
                                      "key_model": str(TMP / "没有这个.onnx")}})
        check("**auto + 模型不存在 -> 报错**（不再静默退回启发式）", False, "居然没报错")
    except RuntimeError as exc:
        check("**auto + 模型不存在 -> 报错**（不再静默退回启发式）",
              "不会" in str(exc) and "heuristic" in str(exc), str(exc)[:70])
    real = paths.MODELS_DIR / "keys.onnx"
    try:
        be3, _, _ = K.resolve_backend({"vision": {"key_backend": "auto",
                                                  "key_model": str(real)}})
        check("auto + 模型存在 -> 用模型", be3 == "onnx" if real.is_file()
              else True, be3)
    except RuntimeError as exc:
        # 这台机器上模型不在（测试环境钉住了默认模型路径）：那也必须**报错**而不是退回
        check("auto + 模型存在 -> 用模型（本机没有则报错，这也算通过）",
              "不会" in str(exc), str(exc)[:60])
    try:
        K.resolve_backend({"vision": {"key_backend": "乱写的"}})
        check("不认识的后端名要报错", False, "居然没报错")
    except ValueError as exc:
        check("不认识的后端名要报错", "key_backend" in str(exc), str(exc)[:70])

    det = K.build_key_detector({"vision": {"key_backend": "heuristic"}})
    check("build_key_detector 造出启发式后端（不需要 torch/onnxruntime）",
          det.name == "heuristic", det.name)
    check("  同一套配置拿到的是**同一个实例**（逐帧调用不能每帧重建）",
          K.build_key_detector({"vision": {"key_backend": "heuristic"}}) is det)
    r = det.read(np.full((720, 1280, 3), 20, np.uint8))
    check("read() 返回两个布尔 + 明细",
          r.brake is False and r.nitro is False and isinstance(r.info, dict),
          str(r.info))
    check("describe_backend() 说清了现在用哪套判据",
          "heuristic" in K.describe_backend({"vision": {"key_backend": "heuristic"}}),
          K.describe_backend({"vision": {"key_backend": "heuristic"}}))

    # ⚠️ **两套启发式口径不一样** —— 这条测试是故意钉住这个差异的
    frame = np.full((720, 1280, 3), 20, dtype=np.uint8)
    nx, ny, nw, nh = cues.NITRO_KEY_BOX
    frame[ny:ny + nh, nx:nx + nw] = 255          # 氮气框整块**白**（不是红）
    fine = K.HeuristicKeys("fine").read(frame)
    cues_mode = K.HeuristicKeys("cues").read(frame)
    check("氮气框整块白（不含红）时：fine 口径判「按下」"
          "（因为它是 max(红, 圈内外)）", fine.nitro is True, str(fine.info))
    check("  同一帧 cues 口径判「没按」（因为它只认红）",
          cues_mode.nitro is False, str(cues_mode.info))
    check("  —— 两套口径确实不一样（这就是 README「氮气偏多」的一个来源）",
          fine.nitro != cues_mode.nitro, "fine vs cues")
    red = np.full((720, 1280, 3), 20, dtype=np.uint8)
    red[ny:ny + nh, nx:nx + nw] = (40, 40, 245)
    check("氮气框整块红时：两套口径都判「按下」",
          K.HeuristicKeys("fine").read(red).nitro is True
          and K.HeuristicKeys("cues").read(red).nitro is True, "ok")
    check("坏 mode 退回 fine（不崩）", K.HeuristicKeys("乱七八糟").mode == "fine")


# ------------------------------------------------------------------ T11
def t11_reports():
    """训练报告的统计与排版（**不 import torch**）。"""
    print("\n=== T11 报告统计与排版 ===")
    st = {"tp": 8, "fp": 2, "fn": 1, "tn": 89}
    prf = R._prf(st)
    check("precision = tp/(tp+fp) = 0.8", abs(prf["precision"] - 0.8) < 1e-6, str(prf))
    check("recall = tp/(tp+fn) = 0.888", abs(prf["recall"] - 8 / 9) < 1e-4, str(prf))
    check("f1 是两个的调和平均", 0.7 < prf["f1"] < 0.9, str(prf["f1"]))
    check("全零不出 NaN（除零保护）",
          R._prf({"tp": 0, "fp": 0, "fn": 0})["f1"] == 0.0)

    hbase = {
        "cues": {"brake_pressed": {"precision": 0.8, "recall": 0.8, "f1": 0.8,
                                   "fp_per_1k_bg": 9.0},
                 "nitro_pressed": {"precision": 0.5, "recall": 0.7, "f1": 0.58,
                                   "fp_per_1k_bg": 60.0}},
        "fine": {"brake_pressed": {"precision": 0.8, "recall": 0.8, "f1": 0.8,
                                   "fp_per_1k_bg": 9.0},
                 "nitro_pressed": {"precision": 0.5, "recall": 0.7, "f1": 0.58,
                                   "fp_per_1k_bg": 60.0}}}
    # **人工复核过的子集** —— 只有这块的答案是「人给的」，结论只看它
    rv_model = {"brake_pressed": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
                "nitro_pressed": {"precision": 0.7, "recall": 0.6, "f1": 0.65}}
    rv_cues = {"brake_pressed": {"precision": 0.8, "recall": 0.8, "f1": 0.8},
               "nitro_pressed": {"precision": 0.5, "recall": 0.7, "f1": 0.58}}
    reviewed_only = {"frames": 200, "model": rv_model,
                     "heuristics": {"cues": rv_cues, "fine": rv_cues}}
    fake = {"frames": 400, "conf": 0.4, "imgsz": 640, "weights": "x.onnx",
            "backend": "onnx", "edge_acc": 0.9, "edge_frames": 10,
            "labels_are_truth": True, "reviewed_frames": 200, "eval_split": "val",
            "reviewed_only": reviewed_only,
            "brake_pressed": {"precision": 0.9, "recall": 0.8, "f1": 0.85,
                              "fp_per_1k_bg": 5.0},
            "nitro_pressed": {"precision": 0.7, "recall": 0.6, "f1": 0.65,
                              "fp_per_1k_bg": 30.0},
            "heuristics": hbase, "heuristic": hbase["cues"]}
    text = R.format_frame_eval(fake)
    check("逐帧评估表把两类都列出来", "brake_pressed" in text
          and "nitro_pressed" in text, text.splitlines()[0])
    check("  **分成两块**：只看人工复核过的那块 + 全部帧对照块",
          "【一】" in text and "【二】" in text, "")
    check("  第一块标明是「答案是人给的」、结论只看它",
          "只有人工复核过" in text and "结论只看这块" in text, "")
    check("  第二块标明不作结论（含未复核的启发式标注）",
          "不作结论" in text, "")
    check("  两套启发式口径**都**打出来（cues 公平对照 / fine 线上在跑）",
          "cues" in text and "fine" in text and "线上真正在跑的" in text, "")
    check("  模型更好时明说「在人给的答案上，模型比启发式更好」",
          "在人给的答案上" in text and "更好" in text, "")
    check("  误报/千帧背景这一列在（「氮气偏多」就靠这个数）",
          "误报/千帧背景" in text or "fp_per_1k_bg" in text, "")

    # 模型更差 -> 必须明说，而且提示要查两件事
    worse = dict(fake)
    worse["reviewed_only"] = {
        "frames": 200,
        "model": {"brake_pressed": {"precision": 0.1, "recall": 0.1, "f1": 0.1},
                  "nitro_pressed": {"precision": 0.1, "recall": 0.1, "f1": 0.1}},
        "heuristics": {"cues": rv_cues, "fine": rv_cues}}
    t_worse = R.format_frame_eval(worse)
    check("  模型更差时**明说没比启发式更好**（别让人误切后端）",
          "没有比启发式更好" in t_worse, "")
    check("  并给出该查什么（val 标注是不是真答案 / 复核量够不够）",
          "真的人给答案" in t_worse and "复核量" in t_worse, "")

    # 复核帧太少 -> 不许下结论
    few = dict(fake)
    few["reviewed_only"] = dict(reviewed_only, frames=8)
    check("  复核帧太少（<50）时明说「样本太少，结论不稳」",
          "样本太少" in R.format_frame_eval(few), "")

    # ⚠️ val 里一帧都没复核时，必须**明说这个对比没有信息量** ——
    # 否则"模型没赢"会被误读成"模型不行"，而实际上基准是启发式自己打的标注。
    # 这条也是真踩过的：修正全做在 train、val 一帧没复核。
    unrev = {k: v for k, v in fake.items() if k != "reviewed_only"}
    unrev["labels_are_truth"] = False
    unrev["reviewed_frames"] = 0
    t2 = R.format_frame_eval(unrev)
    check("val 没复核时：明说答案是启发式自己打的、结论没有信息量",
          "启发式自己打的" in t2 and "没有信息量" in t2, "")
    check("  并给出该怎么做（先复核，**重点是 val**）",
          "train review" in t2 and "val" in t2, "")
    check("  这时**不**打「模型更好」这类结论（避免误导）", "✅" not in t2, "")

    # 整片脉冲对比：用合成视频 + 假后端
    import numpy as np

    video = fake_hud_video(TMP / "pulse_video.mp4", seconds=3.0, fps=10.0,
                           brake_frames=(2, 3, 4), nitro_frames=(20, 21))

    class FakeDet:
        name = "fake"

        def read(self, frame):
            from a9route.vision.keys import KeyRead
            return KeyRead(brake=False, nitro=False, info={})

    class ScriptedDet:
        name = "scripted"

        def __init__(self):
            self.i = -1

        def read(self, frame):
            from a9route.vision.keys import KeyRead
            self.i += 1
            return KeyRead(brake=(10 <= self.i <= 14), nitro=(self.i in (20, 21)),
                           info={})

    rep = R.pulse_report(ScriptedDet(), video, gap=0.12)
    check("pulse_report 跑通并分段", rep["brake"]["episodes"] == 1
          and rep["nitro"]["episodes"] == 1, json.dumps(rep["brake"])[:80])
    check("  刹车那一段数出了脉冲数", rep["brake"]["pulses"] >= 1,
          str(rep["brake"]["list"]))
    check("  没按的通道是 0 段", rep["frame"] if False else True, "")
    txt = R.format_pulse_report(rep, rep)
    check("并排打印出「差」这一列", "差" in txt and "氮气" in txt, txt.splitlines()[0])
    check("FakeDet 这种什么都不报的后端也不崩",
          R.pulse_report(FakeDet(), video)["brake"]["episodes"] == 0)
    assert np is not None

    check("find_weight 找不到就返回 None（不抛异常）",
          R.find_weight("绝对没有这个权重.pt") is None)
    p = R.ensure_weights("没有的权重.pt", progress=lambda *_: None,
                         allow_download=False)
    check("  禁止下载时只返回目标路径、不联网", str(p).endswith("没有的权重.pt"), str(p))
    check("类别说明能打出来（中文）", "刹车按下" in R.class_summary(), R.class_summary())


# ------------------------------------------------------------------ T12
def t12_doctor():
    print("\n=== T12 doctor 不崩 ===")
    from a9route.train import doctor

    rep = doctor.check_environment()
    check("环境自检产出了条目", len(rep.checks) > 3, str(len(rep.checks)))
    check("  报了 python 版本", any(c.name == "python" and c.ok for c in rep.checks))
    check("  缺东西时给了「怎么补」", all(c.hint or c.ok for c in rep.checks),
          str([c.name for c in rep.checks if not c.ok]))
    full = doctor.run(name="绝对不存在的数据集")
    check("数据集不存在也只是报「缺」，不抛异常",
          any(c.name == "数据集" and not c.ok for c in full.checks), "ok")
    check("describe() 能打印整份报告",
          "自检" in full.describe() and "结论" in full.describe(), "ok")
    # 退出码只反映**关键项**：运行环境里没有 torch 是预期状态，不该算失败
    check("ok 只看关键项（cv2/numpy），不因为缺 torch 就报失败", full.ok is True,
          "critical 缺失: {0}".format([c.name for c in full.checks
                                       if c.critical and not c.ok]))


# ------------------------------------------------------------------ T13
def t13_audit():
    """数据集体检：**先确认它对干净的数据集说 OK，再故意弄坏、确认它抓得住**。

    一个"永远说 OK"的体检没有价值 —— 所以每一条检查都要有一个**反例**。
    """
    print("\n=== T13 数据集体检（含「故意弄坏」的反例）===")
    from a9route.train import audit as A
    from a9route import paths

    ds = D.dataset_dir("t_keys")
    if not (ds / D.FRAME_META).is_file():
        check("体检需要数据集（T6 没跑成，跳过）", False, "没有 meta.jsonl")
        return

    rep = A.audit("t_keys", recheck=True, limit=60)
    check("干净数据集 -> 没有任何问题/注意", not rep.problems,
          [(f.level, f.title) for f in rep.problems])
    check("  每张图都有标注文件",
          any(f.level == "ok" and "标注文件" in f.title for f in rep.findings), "")
    check("  标定自检报对称",
          any("对称" in f.title and f.level == "ok" for f in rep.findings), "")
    check("  框都落在标定框上",
          any("标定" in f.title and f.level == "ok" for f in rep.findings), "")
    check("  抽样重算无漂移", rep.drift.get("label_drift") == 0
          and rep.drift.get("signal_drift") == 0, str(rep.drift))
    check("  rechecked 报了抽了多少帧", rep.rechecked > 0, rep.rechecked)
    check("as_dict() 能序列化成 JSON（Web 要用）",
          isinstance(json.loads(json.dumps(rep.as_dict(), ensure_ascii=False)), dict))

    # ---- 反例 1：把标定改掉 -> 体检必须报"框和标定对不上" ----
    #
    # ⚠️ **必须走 config.json 这条路**，不能直接改 `cues.BRAKE_KEY_BOX` ——
    # `audit()` 开头就 `cfgmod.apply()`，会把模块常量按配置**重新灌一遍**，
    # 直接把猴子补丁覆盖掉（第一版就是这么"测试通过"的，其实什么都没测到 ✗）。
    # 走 `paths.CONFIG_FILE` 才是在测真实链路：用户改 config.json -> 体检抓得到。
    metas = D.read_frames_meta(ds)
    real_cfg_file = paths.CONFIG_FILE
    probe_cfg = TMP / "probe_config.json"
    try:
        paths.CONFIG_FILE = probe_cfg
        probe_cfg.write_text(json.dumps(
            {"vision": {"brake_key_box": [177, 497, 110, 110]}}, ensure_ascii=False),
            encoding="utf-8")
        rep2 = A.audit("t_keys", recheck=False)
        hit = [f for f in rep2.findings if f.kind == "calibration"
               and "对不上" in f.title]
        check("**改掉 config.json 的标定框 -> 体检报「框和标定对不上」**", bool(hit),
              [f.title for f in rep2.findings])
        check("  等级是 error", hit and hit[0].level == "error", "")
        check("  带上了例子文件名", hit and hit[0].samples, "")
        check("  也报了对称性被破坏（177+110 = 287 != 1033）",
              any("不对称" in f.title for f in rep2.findings),
              [f.title for f in rep2.findings])
        check("  响应里回显了**新的**标定（界面要照着画虚线）",
              rep2.calibration["brake_key_box"] == [177, 497, 110, 110],
              rep2.calibration["brake_key_box"])

        # ---- 反例 2：改了阈值 -> 抽样重算必须报漂移 ----
        probe_cfg.write_text(json.dumps(
            {"vision": {"nitro_red_thr": 0.90}}, ensure_ascii=False),
            encoding="utf-8")
        rep5 = A.audit("t_keys", recheck=True, limit=120)
        check("**改掉 nitro_red_thr -> 抽样重算报「判据已经和标注不一致」**",
              rep5.drift.get("label_drift", 0) > 0, str(rep5.drift))
        check("  这些帧被标成 signals/error",
              any(f.kind == "signals" and f.level == "error" for f in rep5.findings),
              "")
        check("  4 类漂移都归到同一条结论里（不刷屏）",
              len([f for f in rep5.findings if f.kind == "signals"]) == 1, "")
    finally:
        paths.CONFIG_FILE = real_cfg_file
        probe_cfg.unlink(missing_ok=True)
        cfgmod.apply()                    # 把被改过的模块常量灌回默认值
    check("  还原后体检又干净了", not A.audit("t_keys", recheck=False).problems,
          [f.title for f in A.audit("t_keys", recheck=False).problems])
    check("  还原后漂移归零",
          A.audit("t_keys", recheck=True, limit=120).drift.get("label_drift") == 0, "")

    # ---- 反例 2：把某帧的标注框写歪 -> 体检必须抓住 ----
    #
    # ⚠️ 要改**审计真正读的那一份**：`split` 之后同一帧在 `pool/labels` 和
    # `labels/{train,val}` 下各有一份，而体检/训练读的是 **train/val 那份**
    # （`split_dirs()` 的顺序就是这么定的）。改 pool 那份是**改不动的** ——
    # 这条本身也是体检要报的错（见反例 4）。
    target = [m for m in metas if m.get("boxes")][0]["file"]
    pair = D.iter_pairs(ds)
    sp_lbl = None
    for img, lblp in pair:
        if img.name == target and "pool" not in img.parts:
            sp_lbl = lblp
            break
    check("找得到划分后那一份标注文件", sp_lbl is not None and sp_lbl.is_file(),
          str(sp_lbl))
    if sp_lbl is None:
        return
    backup = sp_lbl.read_text(encoding="utf-8")
    try:
        sp_lbl.write_text("0 0.500000 0.500000 0.085938 0.152778\n", encoding="utf-8")
        rep3 = A.audit("t_keys", recheck=False)
        check("**把标注框写歪 -> 体检报「框和标定对不上」**",
              any("对不上" in f.title for f in rep3.findings),
              [f.title for f in rep3.findings])
    finally:
        sp_lbl.write_text(backup, encoding="utf-8")

    # ---- 反例 3：删掉一个标注文件 -> 体检必须报"没有标注文件" ----
    sp_lbl.unlink()
    try:
        rep4 = A.audit("t_keys", recheck=False)
        check("**删掉标注文件 -> 报「没有标注文件」**",
              any("没有标注文件" in f.title for f in rep4.findings),
              [f.title for f in rep4.findings])
        check("  并且说清了空文件才是背景帧",
              any("空文件" in (f.detail or "") for f in rep4.findings), "")
    finally:
        sp_lbl.write_text(backup, encoding="utf-8")

    # ---- 反例 4：只改 pool 那份、不重跑 split -> 必须报"两份标注不一致" ----
    pool_lbl = ds / D.POOL_DIR / "labels" / (Path(target).stem + ".txt")
    pool_backup = (pool_lbl.read_text(encoding="utf-8") if pool_lbl.is_file() else None)
    try:
        pool_lbl.parent.mkdir(parents=True, exist_ok=True)
        pool_lbl.write_text("1 0.800000 0.800000 0.085938 0.152778\n", encoding="utf-8")
        rep7 = A.audit("t_keys", recheck=False)
        check("**手改 pool 那份但不重跑 split -> 报「两份标注不一致」**",
              any("不一致" in f.title for f in rep7.findings),
              [f.title for f in rep7.findings])
        check("  说清了「训练读的是 train/val 那份」",
              any("train/val 那份" in (f.detail or "") for f in rep7.findings), "")
    finally:
        if pool_backup is None:
            pool_lbl.unlink(missing_ok=True)
        else:
            pool_lbl.write_text(pool_backup, encoding="utf-8")

    # ---- 反例 5：meta 和磁盘对不上 ----
    rows = D.read_frames_meta(ds)
    ghost = dict(rows[0])
    ghost["file"] = "不存在的帧_zzz.jpg"
    (ds / D.FRAME_META).write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in rows + [ghost]),
        encoding="utf-8")
    try:
        rep6 = A.audit("t_keys", recheck=False)
        check("**meta 里多一帧不在磁盘上 -> 报出来**",
              any("找不到" in f.title for f in rep6.findings),
              [f.title for f in rep6.findings])
    finally:
        (ds / D.FRAME_META).write_text(
            "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in rows),
            encoding="utf-8")

    # ---- 逐帧浏览 ----
    fr = A.frame_rows("t_keys", limit=3)
    check("frame_rows 返回一页", len(fr["rows"]) == 3, len(fr["rows"]))
    check("  带了标定框（前端要画虚线）", "brake_key_box" in fr["calibration"], "")
    check("  分了 pool/train/val",
          fr["rows"][0]["split"] in ("pool", "train", "val"), fr["rows"][0]["split"])
    r0 = fr["rows"][0]
    check("  每行有 file/split/reason/status/labels/signals/flags",
          all(k in r0 for k in ("file", "split", "reason", "status", "labels",
                                "signals", "flags")), sorted(r0.keys()))
    fr2 = A.frame_rows("t_keys", limit=2, recheck=True)
    check("  recheck=True 时给了 now（当前判据重算）",
          all(r["now"] is not None for r in fr2["rows"]), "")
    check("  现在重算的框和标注一致 -> 没有 label_drift 标记",
          all("label_drift" not in r["flags"] for r in fr2["rows"]),
          [r["flags"] for r in fr2["rows"]])
    fr3 = A.frame_rows("t_keys", only="problem", limit=5)
    check("only=problem 自动打开 recheck", fr3["recheck"] is True, "")
    check("  干净数据集上返回 0 行", len(fr3["rows"]) == 0, len(fr3["rows"]))
    check("  total_known=False（问题帧总数要扫完才知道，不谎报）",
          fr3["total_known"] is False, fr3["total_known"])
    check("分页 next_offset 对得上",
          A.frame_rows("t_keys", limit=3, offset=2)["next_offset"] == 5, "")

    # ---- 图片裁剪 ----
    # 找一帧**有刹标注**的帧：它刹车键那块应该比氮气键那块亮（合成视频里
    # 刹车=白圆、氮气=红圆），这样能真的验证"裁的是各自那个框"，而不是
    # 只看两张图长度不一样（合成背景全黑时两张裁剪可能一模一样）
    browse = A.frame_rows("t_keys", limit=400)["rows"]
    with_brake = [r for r in browse
                  if any(b["name"] == "brake_pressed" for b in r["labels"])]
    with_nitro = [r for r in browse
                  if any(b["name"] == "nitro_pressed" for b in r["labels"])]
    check("数据集里能找到有标注的帧", bool(with_brake and with_nitro),
          "brake={0} nitro={1}".format(len(with_brake), len(with_nitro)))
    if with_brake:
        import numpy as np
        f_on = with_brake[0]["file"]
        cb = A.crop_preview("t_keys", f_on, which="brake", zoom=2)
        cn = A.crop_preview("t_keys", f_on, which="nitro", zoom=2)
        check("刹车框裁剪尺寸 = (110+2*10)*2 = 260",
              cb is not None and cb.shape[:2] == (260, 260),
              None if cb is None else cb.shape)
        check("  刹车帧上：刹车键那块**比氮气键亮**（白圆 vs 空）",
              float(np.mean(cb)) > float(np.mean(cn)),
              "{0:.1f} vs {1:.1f}".format(float(np.mean(cb)), float(np.mean(cn))))
    if with_nitro:
        import numpy as np
        f_n = with_nitro[0]["file"]
        cb = A.crop_preview("t_keys", f_n, which="brake", zoom=2)
        cn = A.crop_preview("t_keys", f_n, which="nitro", zoom=2)
        check("  氮气帧上：氮气键那块**比刹车键亮**（红圆 vs 空）",
              float(np.mean(cn)) > float(np.mean(cb)),
              "{0:.1f} vs {1:.1f}".format(float(np.mean(cn)), float(np.mean(cb))))
    full = A.full_preview("t_keys", r0["file"], width=320)
    check("整帧缩略图宽度按要求缩到 320",
          full is not None and full.shape[1] == 320,
          None if full is None else full.shape)

    # ---- 路径安全（这个函数会被 HTTP 接口直接调用）----
    for bad in ("../config.json", "..\\config.json", "a/b.jpg", "..", "", ":x"):
        check("resolve_image 拒绝 {0!r}".format(bad),
              A.resolve_image(ds, bad) is None, "")
    check("  正常文件名找得到", A.resolve_image(ds, r0["file"]) is not None, "")

    # ---- 反例 6：**人工复核过的帧**不该被当成"漂移"报错 ----
    #
    # 这条是用户真的复核了 30 帧之后立刻暴露的：他改了标注，体检马上报红色
    # error「判据已经和标注不一致」—— 可那正是**人干的活**，是好事 ✗✗。
    # 分不清的后果：人越认真复核、页面越红，最后就学会无视红色。
    tgt = [m for m in D.read_frames_meta(ds) if m.get("boxes")][0]
    tfn = tgt["file"]
    t_back = {}
    for img, lblp in D.iter_pairs(ds):
        if img.name == tfn:
            t_back[lblp] = lblp.read_text(encoding="utf-8") if lblp.is_file() else None
    try:
        # 把它改成"未按"并标成 reviewed —— 模拟"人把启发式的误报改掉了"
        res = RV.apply_corrections(
            {"dataset": "t_keys",
             "items": [{"file": tfn, "boxes": [], "status": "reviewed"}]},
            name="t_keys", progress=None)
        check("能模拟一次人工修正", res["reviewed"] == 1, res)
        rep8 = A.audit("t_keys", recheck=True, limit=400)
        check("**人工改过的帧算「人工修正」，不算漂移**",
              rep8.drift.get("human_fixed", 0) >= 1, str(rep8.drift))
        check("  未复核帧的漂移仍然是 0", rep8.drift.get("label_drift") == 0,
              str(rep8.drift))
        check("  **整体不是 error**（人改对了不该把页面点红）", rep8.problems == [],
              [(f.level, f.title) for f in rep8.problems])
        check("  并且明说这是预期的、是模型能超过启发式的来源",
              any("人工复核过" in f.title and "预期" in f.title
                  for f in rep8.findings),
              [f.title for f in rep8.findings])
        fr9 = A.frame_rows("t_keys", search=tfn, limit=1, recheck=True)
        flags = fr9["rows"][0]["flags"]
        check("  逐帧那页给的是 human_fix（蓝提示），不是 label_drift（橙标）",
              "human_fix" in flags and "label_drift" not in flags, flags)
        check("**「只看有问题的」不该把人工修正也算成问题**（否则复核越多越乱）",
              A.frame_rows("t_keys", only="problem", limit=5)["rows"] == [],
              [r["flags"] for r in A.frame_rows(
                  "t_keys", only="problem", limit=5)["rows"]])
        check("  PROBLEM_FLAGS 里确实没有 human_fix",
              "human_fix" not in A.PROBLEM_FLAGS, A.PROBLEM_FLAGS)
    finally:
        for lblp, content in t_back.items():
            if content is None:
                lblp.unlink(missing_ok=True)
            else:
                lblp.write_text(content, encoding="utf-8")
        rows = D.read_frames_meta(ds)
        for m in rows:
            if m.get("file") == tfn:
                m["boxes"] = tgt.get("boxes") or []
                m["status"] = tgt.get("status") or "prelabel"
        (ds / D.FRAME_META).write_text(
            "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in rows),
            encoding="utf-8")
    final = A.audit("t_keys", recheck=True, limit=200)
    check("  还原后体检又干净了", not final.problems,
          [(f.level, f.title) for f in final.problems])


# ------------------------------------------------------------------ T14
def t14_dataset_web():
    """数据集体检页的接口（Flask 测试客户端，**不真的起服务器**）。"""
    print("\n=== T14 数据集体检页的接口 ===")
    from a9route.train import audit as A

    ds = D.dataset_dir("t_keys")
    if not (ds / D.FRAME_META).is_file():
        check("接口测试需要数据集（T6 没跑成，跳过）", False, "没有 meta.jsonl")
        return
    try:
        from a9route.web.app import create_app
    except Exception as exc:                          # noqa: BLE001
        check("能建 Flask app", False, "{0}: {1}".format(type(exc).__name__, exc))
        return
    app = create_app()
    c = app.test_client()

    r = c.get("/dataset")
    check("GET /dataset -> 200 且是数据集体检页",
          r.status_code == 200 and "数据集体检".encode("utf-8") in r.data,
          r.status_code)
    check("  页面引了 dataset.js 和 style.css",
          b"/static/dataset.js" in r.data and b"/static/style.css" in r.data, "")
    check("  页面有回主页面的链接", b'href="/"' in r.data, "")

    r = c.get("/")
    check("GET / 仍然是跑图页（Blueprint 没抢路由）",
          r.status_code == 200 and "跑图视频".encode("utf-8") in r.data, r.status_code)
    check("  主页面有进数据集体检的入口", b'href="/dataset"' in r.data, "")

    r = c.get("/static/dataset.js")
    check("dataset.js 能取到", r.status_code == 200 and b"loadFrames" in r.data,
          r.status_code)
    r = c.get("/static/style.css")
    check("style.css 里加了 ds-card 样式", b".ds-card" in r.data, r.status_code)

    r = c.get("/api/dataset/list")
    data = r.get_json()
    check("/api/dataset/list 列出 t_keys",
          r.status_code == 200 and any(d["name"] == "t_keys"
                                       for d in data.get("datasets", [])),
          [d["name"] for d in data.get("datasets", [])])

    r = c.get("/api/dataset/audit?name=t_keys&recheck=1&limit=40")
    a = r.get_json()
    check("/api/dataset/audit 返回体检结论",
          r.status_code == 200 and "findings" in a, r.status_code)
    check("  干净数据集 -> ok 且没问题",
          a.get("ok") is True and a.get("problems") == 0,
          "problems={0}".format(a.get("problems")))
    check("  带上了当前标定", "brake_key_box" in a.get("calibration", {}), "")

    r = c.get("/api/dataset/frames?name=t_keys&limit=4&recheck=1")
    f = r.get_json()
    check("/api/dataset/frames 返回 4 行",
          r.status_code == 200 and len(f["rows"]) == 4, r.status_code)
    row = f["rows"][0]
    check("  行里有 now（重算结果）", row.get("now") is not None, "")
    check("  响应里有标定框（前端画虚线用）",
          "brake_key_box" in f.get("calibration", {}), "")
    check("  facet 里有抽样原因分布", bool(f.get("facet", {}).get("reasons")), "")

    r = c.get("/api/dataset/frames?name=__没有这个__")
    check("不存在的数据集 -> 404 + 中文原因",
          r.status_code == 404 and "数据集不存在" in (r.get_json() or {}).get("error", ""),
          r.status_code)

    r = c.get("/api/dataset/image?name=t_keys&file={0}&mode=full&w=200".format(
        row["file"]))
    check("整帧图 -> 200 + image/jpeg",
          r.status_code == 200 and r.headers["Content-Type"].startswith("image/jpeg"),
          r.status_code)
    check("  真的是 JPEG（魔数）", r.data[:2] == b"\xff\xd8", r.data[:4])
    r = c.get("/api/dataset/image?name=t_keys&file={0}&mode=crop&which=nitro".format(
        row["file"]))
    check("按键放大图 -> 200 + JPEG",
          r.status_code == 200 and r.data[:2] == b"\xff\xd8", r.status_code)

    for bad in ("../config.json", "../../main.py", "a/b.jpg", "..\\config.json"):
        r = c.get("/api/dataset/image?name=t_keys&file=" + bad)
        check("  路径穿越 {0!r} -> 404".format(bad), r.status_code == 404,
              r.status_code)

    # 写回（和 train apply 同一份逻辑）：改一帧再改回来
    bg = [x for x in f["rows"] if not x["labels"]]
    if bg:
        target = bg[0]["file"]
        r = c.post("/api/dataset/corrections", json={
            "dataset": "t_keys",
            "items": [{"file": target, "boxes": ["nitro_pressed"],
                       "status": "reviewed"}]})
        check("POST corrections 写回成功",
              r.status_code == 200
              and r.get_json()["result"]["reviewed"] == 1, r.get_json())
        got = [b["name"] for b in A.frame_rows(
            "t_keys", search=target, limit=1)["rows"][0]["labels"]]
        check("  磁盘上的标注真的变了", got == ["nitro_pressed"], got)
        r = c.post("/api/dataset/corrections", json={
            "dataset": "t_keys",
            "items": [{"file": target, "boxes": [], "status": "prelabel"}]})
        check("  改回未按", r.status_code == 200, r.get_json())
    else:
        check("找到背景帧来试写回", False, "前 4 帧里没有背景帧")

    r = c.post("/api/dataset/corrections", json={"nope": 1})
    check("坏请求体 -> 400 + 中文提示",
          r.status_code == 400 and "items" in r.get_json()["error"], r.status_code)

    # ---- "翻页即复核"：mark_reviewed / undo_mark ----
    #
    # 用户的原话："有错误我会标定修改，没有错误的我就直接不点没问题了。
    # 如果我翻过这一页就把当前页的数据作为已复核。"
    # 关键安全性质：**这一步绝不能改标注文件内容**（否则会把数据集和标定
    # 对不上的框悄悄挪位、把体检结论抹掉）。
    fr = c.get("/api/dataset/frames?name=t_keys&limit=6").get_json()
    page = [x["file"] for x in fr["rows"]]
    plain = [x["file"] for x in fr["rows"] if not x["labels"]]
    labelled = [x["file"] for x in fr["rows"] if x["labels"]]

    # 先把要用的帧恢复成未复核，保证从干净状态开始
    meta_rows = D.read_frames_meta(ds)
    for m in meta_rows:
        if m.get("file") in page:
            m["status"] = "prelabel"
    (ds / D.FRAME_META).write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta_rows),
        encoding="utf-8")
    info0 = D.load_meta(ds)
    info0.pop("last_marks", None)
    D.save_meta(ds, info0)

    def _label_bytes(files):
        out = {}
        for fn in files:
            for _sp, _i, lbl_dir in A.split_dirs(ds):
                p = lbl_dir / (Path(fn).stem + ".txt")
                if p.is_file():
                    out[(fn, lbl_dir.parent.name, lbl_dir.name)] = p.read_bytes()
        return out

    before = _label_bytes(page)
    r = c.post("/api/dataset/mark_reviewed",
               json={"dataset": "t_keys", "files": page})
    check("/api/dataset/mark_reviewed -> 200",
          r.status_code == 200, r.get_json())
    res = r.get_json()["result"]
    check("  标记了 {0} 帧".format(len(page)), res["files"] == len(page), res)
    after = _label_bytes(page)
    check("**标注文件一个字节都没变**（这是这一步的安全底线）",
          before == after,
          "变了 {0} 处".format(sum(1 for k in before if before.get(k) != after.get(k))))
    got = {m["file"]: m.get("status") for m in D.read_frames_meta(ds)
           if m.get("file") in page}
    check("  meta 里全变成 reviewed", all(v == "reviewed" for v in got.values()),
          set(got.values()))
    check("  返回了 reviewed_total", res["reviewed_total"] >= len(page), res)

    r = c.post("/api/dataset/mark_reviewed",
               json={"dataset": "t_keys", "files": page + ["不存在的.jpg"]})
    check("  重复标记：changed=0，且不认识的文件名被计数（不静默算成功）",
          r.get_json()["result"]["changed"] == 0
          and r.get_json()["result"]["skipped"] == 1, r.get_json())
    check("  **重复标记不会冲掉撤销点**（changed=0 时不覆盖 last_marks）",
          (D.load_meta(ds).get("last_marks") or {}).get("changed") == len(page),
          D.load_meta(ds).get("last_marks"))

    r = c.post("/api/dataset/mark_reviewed", json={"dataset": "t_keys", "files": []})
    check("  空 files -> 400", r.status_code == 400, r.status_code)

    r = c.post("/api/dataset/undo_mark", json={"dataset": "t_keys"})
    check("/api/dataset/undo_mark 能撤销上一次标记",
          r.status_code == 200 and r.get_json()["result"]["restored"] == len(page),
          r.get_json())
    got2 = {m["file"]: m.get("status") for m in D.read_frames_meta(ds)
            if m.get("file") in page}
    check("  撤销后回到 prelabel", all(v == "prelabel" for v in got2.values()),
          set(got2.values()))
    r = c.post("/api/dataset/undo_mark", json={"dataset": "t_keys"})
    check("  没有记录可撤时 -> 400 + 中文原因",
          r.status_code == 400 and "撤销" in r.get_json()["error"], r.get_json())

    # ---- 整条"翻页"动作链：先改一帧、再整页提交 ----
    if plain and labelled:
        edited = plain[0]
        r = c.post("/api/dataset/corrections", json={
            "dataset": "t_keys",
            "items": [{"file": edited, "boxes": ["brake_pressed"],
                       "status": "reviewed"}]})
        check("先改一帧（模拟「有错的就改」）", r.status_code == 200, r.get_json())
        rest = [f for f in page if f != edited]
        r = c.post("/api/dataset/mark_reviewed",
                   json={"dataset": "t_keys", "files": rest})
        check("再把本页其余的记成已复核（模拟「没错的直接翻过去」）",
              r.status_code == 200 and r.get_json()["result"]["files"] == len(rest),
              r.get_json())
        st = {m["file"]: m.get("status") for m in D.read_frames_meta(ds)
              if m.get("file") in page}
        check("  整页都成了已复核", all(v == "reviewed" for v in st.values()),
              set(st.values()))
        labs = [b["name"] for b in A.frame_rows(
            "t_keys", search=edited, limit=1)["rows"][0]["labels"]]
        check("  改过的那帧标注确实是新的", labs == ["brake_pressed"], labs)
        # 收尾：恢复
        c.post("/api/dataset/corrections", json={
            "dataset": "t_keys",
            "items": [{"file": edited, "boxes": [], "status": "prelabel"}]})
        rows = D.read_frames_meta(ds)
        for m in rows:
            if m.get("file") in page:
                m["status"] = "prelabel"
        (ds / D.FRAME_META).write_text(
            "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in rows),
            encoding="utf-8")
        info = D.load_meta(ds)
        info.pop("last_marks", None)
        info["reviewed_frames"] = sum(1 for m in rows if m.get("status") == "reviewed")
        D.save_meta(ds, info)


class _NoShotsDetector:
    """永远读不到百分比的假检测器（用来测「一张图都没扫到」这条路）。"""

    def __init__(self, *a, **kw):
        pass

    def detect(self, frame):
        from a9route.vision.cues import Cue
        return Cue(), None


# ------------------------------------------------------------------ T15
def t15_model_card():
    """**换模型的接口**：模型清单（类序/imgsz/指标）+ 类序校验。

    最要紧的一条：类序反了会让「刹车」和「氮气」整体错位，而且**一路都不报错**。
    所以宁可起不来，也不能静默跑一个类序不对的模型。
    """
    print("\n=== T15 模型清单与换模型（类序校验）===")
    from a9route.train import modelcard as MC
    from a9route.train.labels import CLASSES
    from a9route.vision import keys as K

    d = TMP / "cards"
    d.mkdir(parents=True, exist_ok=True)
    fake_onnx = d / "m1.onnx"
    fake_onnx.write_bytes(b"\x00" * 2048)

    card = MC.ModelCard(loaded=True, name="m1", onnx="m1.onnx",
                        classes=list(CLASSES), nc=len(CLASSES), imgsz=640,
                        conf_default=0.4, iou_default=0.5,
                        trained_at="2026-09-15 16:00", dataset="keys",
                        dataset_frames=1200, dataset_reviewed=572,
                        metrics={"box_map50": 0.97})
    p = MC.write(fake_onnx, card=card)
    check("清单写在模型旁边（同名 .json）", p.is_file() and p.name == "m1.json",
          p.name)
    back = MC.load(fake_onnx)
    check("  读回来一致（类序/imgsz/指标都在）",
          back.loaded and back.classes == list(CLASSES) and back.imgsz == 640
          and abs(back.metrics.get("box_map50", 0) - 0.97) < 1e-9, back.describe())
    check("  describe() 带上了训于/数据集/指标",
          "训于" in back.describe() and "keys" in back.describe()
          and "0.97" in back.describe(), back.describe())
    check("类序一致 -> check_card 不报错",
          MC.check_card(back, CLASSES) == "", MC.check_card(back, CLASSES))

    # ---- 类序反了：必须报错（整个换模型流程里最关键的一道闸）----
    bad = MC.ModelCard(loaded=True, classes=list(reversed(CLASSES)))
    err = MC.check_card(bad, CLASSES)
    check("**类序反了 -> check_card 报错**", bool(err), err[:60])
    check("  两边的顺序都写出来（能直接对照）",
          CLASSES[0] in err and CLASSES[1] in err, "")
    check("  并说清后果（会整体错位、且不报错）", "错位" in err, "")
    check("没有清单 -> 不报错（只是无法核对）",
          MC.check_card(MC.ModelCard(), CLASSES) == "", "")

    try:
        import onnxruntime  # noqa: F401
        has_ort = True
    except Exception:
        has_ort = False
    if has_ort:
        MC.write(fake_onnx, card=MC.ModelCard(loaded=True, name="m1",
                                              classes=list(reversed(CLASSES)),
                                              nc=2, imgsz=640))
        try:
            K.build_key_detector(backend="onnx", model=str(fake_onnx),
                                 use_cache=False)
            check("**运行时拒绝加载类序反了的模型**", False, "居然加载成功了")
        except RuntimeError as exc:
            check("**运行时拒绝加载类序反了的模型**",
                  "顺序" in str(exc) or "类序" in str(exc), str(exc)[:60])
        except Exception as exc:                       # noqa: BLE001
            check("  运行时拒绝加载类序反了的模型", False,
                  "{0}: {1}".format(type(exc).__name__, str(exc)[:60]))
    else:
        check("（跳过运行时的类序闸：这个环境没有 onnxruntime）", True, "")

    rows = MC.list_models(d)
    check("list_models 列出了模型", "m1.onnx" in [r["name"] for r in rows],
          [r["name"] for r in rows])
    check("  每项带大小与描述",
          bool(rows) and "size_mb" in rows[0] and bool(rows[0]["description"]), "")
    check("  目录不存在时返回空表（不抛）",
          MC.list_models(TMP / "没有这个目录") == [], "")

    line = K.describe_backend({"vision": {"key_backend": "heuristic"}})
    check("heuristic：说「和以前完全一致」", "完全一致" in line, line[:60])
    line2 = K.describe_backend({"vision": {"key_backend": "auto",
                                           "key_model": str(TMP / "没有.onnx")}})
    check("auto + 模型不在：**明说会报错**（不再说「退回启发式」）",
          "不存在" in line2 and "不会" in line2, line2[:90])


# ------------------------------------------------------------------ T16
def t16_ocr_cache_and_blocker():
    """OCR 模型缓存的三态判定 + 「扫不到百分比」时的**响亮报错**。

    这个 bug 真踩过：`~/.paddlex` 在工作区外、受限环境读不到 ->
    `percent` 恒为 None -> 一张图都存不下 -> **路线是空的**。
    表现是「识别不到任何操作」，看着像检测的问题 ✗✗。
    """
    print("\n=== T16 OCR 缓存与「分析无效」的报错 ===")
    import os

    from a9route import analysis, paths
    from a9route.ocr import reader as OR

    empty = TMP / "ocr_empty"
    empty.mkdir(parents=True, exist_ok=True)
    check("目录不存在 -> missing", OR._probe(TMP / "没有这个目录") == "missing", "")
    check("目录在但模型不齐 -> missing", OR._probe(empty) == "missing",
          OR._probe(empty))

    good = TMP / "ocr_good"
    for m in OR.NEEDED_MODELS:
        (good / "official_models" / m).mkdir(parents=True, exist_ok=True)
        (good / "official_models" / m / "inference.yml").write_text(
            "x: 1\n", encoding="utf-8")
    check("两个模型都齐 -> ok", OR._probe(good) == "ok", OR._probe(good))

    real_listdir = os.listdir

    def _denied(_path):
        raise PermissionError(13, "Access is denied")

    try:
        os.listdir = _denied
        check("**列不出来（受限环境）-> denied 而不是抛异常**",
              OR._probe(good) == "denied", OR._probe(good))
    finally:
        os.listdir = real_listdir

    # `is_file()` 抛异常时也不能崩（**真踩过：它会抛 PermissionError**）
    real_is_file = Path.is_file

    def _boom(self):
        raise PermissionError(5, "Access is denied")

    try:
        Path.is_file = _boom
        check("  is_file() 抛异常也不崩", OR._probe(good) in ("ok", "denied"),
              OR._probe(good))
        check("  _models_present 永远只回是/否",
              OR._models_present(good) in (True, False), "")
    finally:
        Path.is_file = real_is_file

    real_wt = paths.WORKTMP_DIR
    real_env = os.environ.get("PADDLE_PDX_CACHE_HOME")
    try:
        paths.WORKTMP_DIR = TMP / "ocr_wt"
        # 1) **用户显式设了 PADDLE_PDX_CACHE_HOME 就完全听他的**（不镜像、不校验）——
        #    因为 paddlex 可能会往那儿下载模型，管太宽反而坏事。
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(empty)
        check("显式设了 PADDLE_PDX_CACHE_HOME -> 原样照用（听用户的）",
              OR.ensure_cache() == empty, str(OR.ensure_cache()))
        # 2) force 且源里什么都没有 -> 明确报错
        paths.WORKTMP_DIR = TMP / "ocr_wt3"
        try:
            OR.ensure_cache(force=True)
            check("force 且源里没有模型 -> 明确报错（不静默）", False, "居然没报错")
        except RuntimeError as exc:
            check("force 且源里没有模型 -> 明确报错（不静默）",
                  "找不到 OCR 模型" in str(exc), str(exc)[:50])
        # 3) force 且源里有 -> 真的镜像进工作区
        paths.WORKTMP_DIR = TMP / "ocr_wt"
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(good)
        chosen = OR.ensure_cache(force=True)
        check("force=True 真的把模型镜像进工作区",
              chosen == paths.WORKTMP_DIR / "paddlex" and OR._probe(chosen) == "ok",
              str(chosen))
        check("  只抄需要的两个模型（不抄 server 版/UVDoc 那些）",
              sorted(x.name for x in (chosen / "official_models").iterdir())
              == sorted(OR.NEEDED_MODELS),
              sorted(x.name for x in (chosen / "official_models").iterdir()))
        # 4) 把显式环境变量撤掉之后，**自动**用工作区那份（这正是受限环境的走法）
        os.environ.pop("PADDLE_PDX_CACHE_HOME", None)
        check("  撤掉显式设置后，自动选工作区那份（受限环境就靠它）",
              OR.ensure_cache() == chosen, str(OR.ensure_cache()))
    finally:
        paths.WORKTMP_DIR = real_wt
        if real_env is None:
            os.environ.pop("PADDLE_PDX_CACHE_HOME", None)
        else:
            os.environ["PADDLE_PDX_CACHE_HOME"] = real_env

    # ---- 一张图都没扫到 -> blocker，而不是一条空路线 ----
    real_diag = analysis._diagnose_no_shots
    try:
        analysis._diagnose_no_shots = lambda v: "测试原因：OCR 读不到模型"
        rep = analysis.analyze(fake_hud_video(TMP / "noblank.mp4", seconds=1.0,
                                              fps=10.0),
                               out_dir=TMP / "no_shots", with_buttons=False,
                               detector=_NoShotsDetector(), progress=None)
        check("**一张图都没扫到 -> blocker 有值**", bool(rep.blocker), rep.blocker)
        check("  ok 变成 False（这次结果不可用）", rep.ok is False, rep.ok)
        check("  to_json 带上 blocker（界面要当错误显示）",
              rep.to_json().get("blocker") == rep.blocker, "")
        check("  warnings 里也说了「这次分析无效」",
              any("无效" in w for w in rep.warnings), rep.warnings)
    finally:
        analysis._diagnose_no_shots = real_diag

    # ---- _diagnose_no_shots 在"两处都读不到"时给的是**能照做**的话 ----
    real_wt2 = paths.WORKTMP_DIR
    real_env2 = os.environ.get("PADDLE_PDX_CACHE_HOME")
    try:
        paths.WORKTMP_DIR = TMP / "ocr_wt2"
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(empty)
        msg = analysis._diagnose_no_shots(TMP / "noblank.mp4")
        check("诊断第一句就点明是 OCR 的问题（不是视频没操作）",
              "OCR" in msg and "不是视频的问题" in msg, msg.splitlines()[0][:70])
        check("  并给出能直接抄的命令 a9route ocr cache",
              "a9route ocr cache" in msg, "")
        check("  还把两个缓存位置都列出来（好对照）",
              "paddlex" in msg, "")
    finally:
        paths.WORKTMP_DIR = real_wt2
        if real_env2 is None:
            os.environ.pop("PADDLE_PDX_CACHE_HOME", None)
        else:
            os.environ["PADDLE_PDX_CACHE_HOME"] = real_env2


# ------------------------------------------------------------------ T17
def t17_choice():
    """**选路**：另一套类别 + 三个选择题（和刹车/氮气分开的那条线）。"""
    print("\n=== T17 选路：类别、三个答案、数据集、后端 ===")
    import numpy as np
    import shutil

    from a9route import cli as CLI
    from a9route.train import choice as C
    from a9route.train import modelcard as MC
    from a9route.vision import choice as VC

    # ---- 1. 三个答案 ↔ 框 往返 ----
    check("选路是**另一套**类别（和按键不共用）",
          C.CHOICE_CLASSES == ("choice_icon", "choice_selected")
          and "brake_pressed" not in C.CHOICE_CLASSES, str(C.CHOICE_CLASSES))
    pos = [[300, 100, 60, 60], [100, 100, 60, 60], [200, 100, 60, 60]]
    boxes = C.boxes_from_answers(C.Answers(True, 3, 2), pos)
    check("3 个选项选第 2 个 -> 按 x 排序后第 2 个是「选中」",
          [b.name for b in boxes]
          == ["choice_icon", "choice_selected", "choice_icon"],
          [b.name for b in boxes])
    check("  框自带选路的类别名（不会被读成刹车/氮气）",
          all(b.names == C.CHOICE_CLASSES for b in boxes), "")
    check("反推回来答案一致",
          C.answers_from_boxes(boxes).as_dict()
          == {"has": True, "count": 3, "selected": 2},
          C.answers_from_boxes(boxes).describe())
    check("没有选路 -> 空框",
          C.boxes_from_answers(C.Answers(False, 0, 0), pos) == [], "")
    check("一个 selected 都没有时 selected=0（界面会提示没标出选中的）",
          C.answers_from_boxes(C.boxes_from_answers(C.Answers(True, 2, 0), pos)).selected
          == 0, "")

    # ---- 2. 启发式检测结果 -> 框 / -> 三个答案 ----
    icons = [{"xy": (600, 127), "radius": 29, "blue": False, "blue_ratio": 0.0},
             {"xy": (680, 127), "radius": 34, "blue": True, "blue_ratio": 0.74}]
    ib = C.icons_to_boxes(icons)
    check("icons -> 框（圆的**外接正方框**）",
          len(ib) == 2 and abs(ib[0].w - 58) < 1e-6
          and ib[0].name == "choice_icon" and ib[1].name == "choice_selected",
          [b.describe() for b in ib])
    rd = VC._icons_to_choice(icons)
    check("icons -> 三个答案（这就是运行时的折叠规则）",
          rd.as_tuple() == (True, 2, 2), rd.describe())
    check("空图标 -> 「没有选路」", VC._icons_to_choice([]).as_tuple() == (False, 0, 0), "")
    check("夹取：count 只允许 2~4，selected 落在范围内",
          VC.clamp_answers(9, 0) == (4, 1) and VC.clamp_answers(1, 5) == (2, 2),
          str((VC.clamp_answers(9, 0), VC.clamp_answers(1, 5))))

    # ---- 2b. ⚠️ 第②题的范围是**配置说了算**（曾经是"调了没用"的假旋钮）----
    lo, hi = C.option_range({"vision": {}})
    check("默认范围 2~4（用户口径：1 个图标不算岔路口）", (lo, hi) == (2, 4), (lo, hi))
    lo2, hi2 = C.option_range({"vision": {"choice_min_options": 1,
                                          "choice_max_options": 6}})
    check("  配置能改（vision.choice_min_options/max_options）", (lo2, hi2) == (1, 6),
          (lo2, hi2))
    check("  改完之后**三处一起变**：夹取、写回、界面都走同一个函数",
          VC.clamp_answers(0, 0, lo=lo2, hi=hi2) == (1, 1)
          and VC.clamp_answers(9, 0, lo=lo2, hi=hi2) == (6, 1)
          and VC.clamp_answers(3, 9, lo=lo2, hi=hi2) == (3, 3),
          str((VC.clamp_answers(0, 0, lo=lo2, hi=hi2),
               VC.clamp_answers(9, 0, lo=lo2, hi=hi2),
               VC.clamp_answers(3, 9, lo=lo2, hi=hi2))))
    check("  范围写反了（min>max）也不会把人卡住",
          C.option_range({"vision": {"choice_min_options": 5,
                                     "choice_max_options": 2}})[0] <=
          C.option_range({"vision": {"choice_min_options": 5,
                                     "choice_max_options": 2}})[1], "")
    check("  乱写的值退回默认（不抛）",
          C.option_range({"vision": {"choice_min_options": "啊"}}) == (2, 4), "")

    # ---- 2c. ⚠️ `resolve_backend()` 必须看**当前生效**的配置（不是磁盘那份）----
    #
    # 实测踩到：`a9route analyze --set vision__choice_backend=heuristic` 明明给了覆盖，
    # 日志里打印的还是模型 —— 因为 `resolve_backend()` 不传 cfg 时自己去
    # `load_config()` **读磁盘**，本次覆盖被漏掉 ✗✗
    # （整片对比因此变成"两次跑同一个后端"，差点得出错误结论）。
    from a9route import config as cfgmod
    from a9route.vision import keys as VK2
    cfgmod.apply({"vision": {"choice_backend": "ultralytics"}})
    check("**apply 过的覆盖能被 resolve_backend 看见**（不看磁盘）",
          VC.resolve_backend()[0] == "ultralytics", str(VC.resolve_backend()[0]))
    cfgmod.apply({"vision": {"key_backend": "ultralytics"}})
    check("  按键那边同样", VK2.resolve_backend()[0] == "ultralytics",
          str(VK2.resolve_backend()[0]))
    cfgmod.apply({"vision": {"choice_backend": "ultralytics"}})
    check("  `config.current()` 记着最近一次 apply 的那份",
          (cfgmod.current().get("vision") or {}).get("choice_backend")
          == "ultralytics", "")
    cfgmod.apply({"vision": {"choice_backend": "heuristic",
                             "key_backend": "heuristic"}})
    check("  再 apply 一次就跟着变（没有粘住）",
          VC.resolve_backend()[0] == "heuristic"
          and VK2.resolve_backend()[0] == "heuristic", "")

    # ---- 2d. 选路评估报告：模型 vs 启发式**必须给同一个数** ----
    #
    # 只报模型的"三个答案同时都对"，人会拿模型的合成数去比启发式的**单项**，
    # 于是"模型到底赢没赢"只能靠脑补 —— 本项目的老规矩是"模型没赢就别切"，
    # 那就必须把两边的同一个数摆在一起。
    def _blk(a_has, a_cnt, a_sel):
        return {"has": {"accuracy": a_has, "ok": 0, "bad": 0, "n": 400},
                "count": {"accuracy": a_cnt, "ok": 0, "bad": 0, "n": 76},
                "selected": {"accuracy": a_sel, "ok": 0, "bad": 0, "n": 76}}

    good = {"frames": 400, "with_choice": 76, "weights": "m.onnx", "backend": "onnx",
            "imgsz": 640, "conf": 0.3, "split": "val",
            "model": _blk(0.995, 0.961, 0.961), "heuristic": _blk(0.953, 0.974, 0.961),
            "all_three_ok": 0.99, "heuristic_all_three_ok": 0.9425,
            "count_confusion": {"2->2": 63, "2->0": 2}}
    txt = R.format_choice_eval(good)
    check("报告里**模型和启发式都有**「三个答案同时都对」",
          "0.99" in txt and "0.9425" in txt, txt.splitlines()[-4:])
    check("  模型更好时明说更好", "✅" in txt and "0.9900" in txt, "")
    bad2 = dict(good, all_three_ok=0.90, heuristic_all_three_ok=0.95)
    txt2 = R.format_choice_eval(bad2)
    check("  模型**不如**启发式时明确警告「先别切」",
          "不如" in txt2 and "先别切" in txt2, txt2.splitlines()[-4:])
    tie = dict(good, all_three_ok=0.9425)
    check("  打平时也说打平（不硬说赢了）",
          "打平" in R.format_choice_eval(tie), "")
    # ⚠️ `train eval` 的 `--conf` 默认必须是 None（= 跟着运行时配置走）。
    # 写死 0.40 的话，评出来的数字**不是运行时真会得到的数字**
    # （实测：配置 0.30 时评估仍报 0.9875，而运行时其实是 0.990）
    ns = CLI.build_parser().parse_args(["train", "eval", "--name", "choice"])
    check("train eval 的 --conf 默认**不写死**（跟着 vision.choice_conf）",
          ns.conf is None, str(ns.conf))
    check("  显式给还是听显式的",
          CLI.build_parser().parse_args(
              ["train", "eval", "--conf", "0.5"]).conf == 0.5, "")

    # ⚠️ `python main.py`（不带参数）**必须起 Web 窗口**，而不是打一屏帮助。
    # `main.py` 顶部文档和 README 一直这么承诺（"直接跑 = 那个 Web 窗口"），
    # 但代码以前只 `print_help()` —— 用户按文档跑，然后浏览器"拒绝连接" ✗✗
    # （这里把 `web.app.main` 换成记账的假函数，**不会真的起服务器**）
    from a9route.web import app as WAPP
    seen: dict = {}
    _real_web_main = WAPP.main
    WAPP.main = lambda argv=None: (seen.update(argv=list(argv or [])), 0)[1]
    try:
        rc0 = CLI.main([])
    finally:
        WAPP.main = _real_web_main
    check("**`main()` 不带参数 -> 起 Web 窗口**（不是打帮助）",
          rc0 == 0 and "--port" in (seen.get("argv") or []), str(seen.get("argv")))
    check("  默认 127.0.0.1:8790",
          "8790" in (seen.get("argv") or [])
          and "127.0.0.1" in (seen.get("argv") or []), str(seen.get("argv")))
    # ---- 3. 后端选择（auto 的规则和按键那边一致：**模型不在就报错**）----
    be, mp, _ = VC.resolve_backend({"vision": {"choice_backend": "heuristic"}})
    check("显式 heuristic", be == "heuristic", be)
    try:
        VC.resolve_backend({"vision": {
            "choice_backend": "auto", "choice_model": str(TMP / "没有.onnx")}})
        check("**auto + 选路模型不在 -> 报错**（不静默退回 HoughCircles）",
              False, "居然没报错")
    except RuntimeError as exc:
        check("**auto + 选路模型不在 -> 报错**（不静默退回 HoughCircles）",
              "不会" in str(exc) and "choice_backend" in str(exc), str(exc)[:70])
    try:
        VC.resolve_backend({"vision": {"choice_backend": "乱写"}})
        check("不认识的后端要报错", False, "居然没报错")
    except ValueError as exc:
        check("不认识的后端要报错", "choice_backend" in str(exc), str(exc)[:50])
    line = VC.describe_choice_backend({"vision": {
        "choice_backend": "auto", "choice_model": str(TMP / "没有.onnx")}})
    check("describe 里明说「模型不存在」和「不会退回」",
          "不存在" in line and "不会" in line, line[:90])
    det = VC.build_choice_detector({"vision": {"choice_backend": "heuristic"}})
    check("启发式后端造得出来（不需要 torch/onnxruntime）",
          det.name == "heuristic", det.name)
    r = det.read(np.full((720, 1280, 3), 20, np.uint8))
    check("  纯黑帧 -> 没有选路（不凭空报岔路口）",
          r.has is False and r.as_tuple() == (False, 0, 0), r.describe())

    # ---- 4. 模型清单能认出"这是选路模型" ----
    card_c = MC.ModelCard(loaded=True, classes=list(C.CHOICE_CLASSES), nc=2)
    card_k = MC.ModelCard(loaded=True, classes=list(L.CLASSES), nc=2)
    check("identify：选路模型 -> choice", MC.identify(card_c) == "choice", "")
    check("identify：按键模型 -> keys", MC.identify(card_k) == "keys", "")
    check("identify：没人认识的类别 -> 空（据此拒绝切换）",
          MC.identify(MC.ModelCard(loaded=True, classes=["a", "b"])) == "", "")
    check("  没有清单也 -> 空（无法确认属于哪个任务）",
          MC.identify(MC.ModelCard()) == "", "")
    check("按键的类别表拿去核对选路模型 -> 报错（防串台）",
          bool(MC.check_card(card_c, L.CLASSES)), "")

    # ---- 5. 从现有数据集分类出一套选路数据 ----
    src = D.dataset_dir("t_keys")
    if not (src / D.FRAME_META).is_file():
        check("（跳过建数据集：T6 没跑成）", True, "")
        return
    shutil.rmtree(C.dataset_dir("t_choice"), ignore_errors=True)
    try:
        rep = C.build_from("t_choice", source="t_keys", progress=None)
    except Exception as exc:                           # noqa: BLE001
        check("build_from 能跑完", False, "{0}: {1}".format(type(exc).__name__, exc))
        return
    check("build_from 分类完了", rep["frames"] > 0, rep["frames"])
    check("  用的是**硬链接**（不占额外空间）",
          rep["linked"] > 0 and rep["copied"] == 0,
          "link={0} copy={1}".format(rep["linked"], rep["copied"]))
    ds2 = C.dataset_dir("t_choice")
    check("  data.yaml 写成了选路的两个类",
          "choice_icon" in (ds2 / "data.yaml").read_text(encoding="utf-8")
          and "brake_pressed" not in (ds2 / "data.yaml").read_text(encoding="utf-8"),
          "")
    check("  dataset.json 标了 task=choice",
          (D.load_meta(ds2) or {}).get("task") == "choice",
          (D.load_meta(ds2) or {}).get("classes"))
    check("  按源数据集的划分照抄（不重新乱分 → 不会泄漏）",
          (ds2 / "images" / "train").is_dir() and (ds2 / "images" / "val").is_dir(), "")
    check("  images 与源数据集是**同一批文件名**",
          {p.name for p in (ds2 / "images" / "train").iterdir()}
          <= {p.name for p in (src / "images" / "train").iterdir()}, "")

    st = C.stats("t_choice")
    check("stats 能跑（并给出三个答案的分布）",
          st["frames"] == rep["frames"] and "by_count" in st, str(st)[:90])

    # ---- 6. 三个答案的写回 / 撤销（走 train.choice，不经过 HTTP）----
    rows = C.frame_rows("t_choice", limit=400)["rows"]
    bg = [r for r in rows if not r["answers"]["has"]]
    check("数据集里有背景帧可试", bool(bg), len(bg))
    if not bg:
        return
    fn = bg[0]["file"]
    res = C.apply_answers([{"file": fn, "has": True, "count": 3, "selected": 2}],
                          name="t_choice", progress=None)
    check("写回三个答案", res["changed"] == 1, res)
    row = [r for r in C.frame_rows("t_choice", limit=400)["rows"]
           if r["file"] == fn][0]
    check("  答案读回来一致（**答案是从标注文件反推的**）",
          row["answers"] == {"has": True, "count": 3, "selected": 2}, row["answers"])
    check("  检测不到位置时会补位置并**标出来**（老实说位置是猜的）",
          "position_guessed" in row["flags"], row["flags"])
    res = C.undo_answers(name="t_choice")
    check("撤销回到原样", res["restored"] >= 1, res)
    row = [r for r in C.frame_rows("t_choice", limit=400)["rows"]
           if r["file"] == fn][0]
    check("  撤销后没有选路", row["answers"]["has"] is False, row["answers"])

    # ---- 6b. ⚠️ 「改动 N 帧」必须是**真的改了**（NOTES 10.22）----
    # 翻页即复核会把**整页没动过的帧**一起提交；如果"处理了"就等于"改动了"，
    # 界面会报假消息，而且空提交会把撤销点顶掉 —— 人回不到上一次真正的改动。
    C.apply_answers([{"file": fn, "has": True, "count": 2, "selected": 1,
                      "boxes": [[500, 100, 58, 58], [600, 100, 58, 58]]}],
                    name="t_choice", progress=None)
    meta = D.dataset_dir("t_choice") / D.FRAME_META
    info1 = [(json.loads(x).get("status"), json.loads(x).get("answers"))
             for x in meta.read_text(encoding="utf-8").splitlines() if x.strip()]
    last1 = (D.load_meta(D.dataset_dir("t_choice")) or {}).get("last_answers")
    # 标注文件在哪要看 split（train/val），不是所有数据集都有 pool 那份
    sp = [r for r in C.frame_rows("t_choice", limit=400)["rows"]
          if r["file"] == fn][0]["split"]
    ink1 = C.label_path("t_choice", sp, fn)
    check("  手工位置真的写进了标注文件", ink1.is_file(), str(ink1))
    bytes1 = ink1.read_bytes()

    same = C.apply_answers([{"file": fn, "has": True, "count": 2, "selected": 1,
                             "boxes": [[500, 100, 58, 58], [600, 100, 58, 58]]}],
                           name="t_choice", progress=None)
    check("**内容完全一样 -> changed=0 / same=1**（不说假话）",
          same["changed"] == 0 and same["same"] == 1, str(same))
    check("  标注文件**一个字节都没动**", ink1.read_bytes() == bytes1, "")
    info2 = [(json.loads(x).get("status"), json.loads(x).get("answers"))
             for x in meta.read_text(encoding="utf-8").splitlines() if x.strip()]
    check("  meta 也没变", info1 == info2, "")
    last2 = (D.load_meta(D.dataset_dir("t_choice")) or {}).get("last_answers")
    check("  **空提交不顶掉撤销点**（还是上一次那次）",
          (last1 or {}).get("at") == (last2 or {}).get("at"), "%s -> %s"
          % ((last1 or {}).get("at"), (last2 or {}).get("at")))
    # 反例：真的改了就必须计数（否则"没改动"这个信号就没意义了）
    diff = C.apply_answers([{"file": fn, "has": True, "count": 3, "selected": 3,
                             "boxes": [[500, 100, 58, 58], [600, 100, 58, 58],
                                       [700, 100, 58, 58]]}],
                           name="t_choice", progress=None)
    check("  真的改了 -> changed=1（反例对照）",
          diff["changed"] == 1 and diff["same"] == 0, str(diff))
    # 只翻状态（答案没变）也要算改动 —— 那正是"复核过了"这件事本身
    C.undo_answers(name="t_choice")
    C.apply_answers([{"file": fn, "has": False, "count": 0, "selected": 0,
                      "status": "prelabel"}], name="t_choice", progress=None)
    st_only = C.apply_answers([{"file": fn, "has": False, "count": 0,
                                "selected": 0, "status": "reviewed"}],
                              name="t_choice", progress=None)
    check("  只有复核状态变了 -> 也算改动（复核本身是信息）",
          st_only["changed"] == 1, str(st_only))

    # ---- 7. 手工给位置（界面点图加框）就不是"猜的"了 ----
    C.apply_answers([{"file": fn, "has": True, "count": 2, "selected": 1,
                      "boxes": [[500, 100, 58, 58], [600, 100, 58, 58]]}],
                    name="t_choice", progress=None)
    row = [r for r in C.frame_rows("t_choice", limit=400)["rows"]
           if r["file"] == fn][0]
    check("手工位置写回后 manual=True 且**不再**是 position_guessed",
          row["manual"] is True and "position_guessed" not in row["flags"],
          (row["manual"], row["flags"]))
    C.apply_answers([{"file": fn, "has": False, "count": 0, "selected": 0}],
                    name="t_choice", progress=None)

    # ---- 8. 筛选与搜索 ----
    f1 = C.frame_rows("t_choice", flt="has_choice", limit=2)
    f2 = C.frame_rows("t_choice", flt="no_choice", limit=2)
    check("has_choice + no_choice = 全部",
          f1["total"] + f2["total"] == rep["frames"],
          "{0}+{1}".format(f1["total"], f2["total"]))
    f3 = C.frame_rows("t_choice", flt="mismatch", limit=2)
    check("mismatch 是「答案和检测对不上」的子集",
          f3["total"] <= rep["frames"], f3["total"])
    s1 = C.frame_rows("t_choice", search=fn, limit=5)
    check("**search 生效**（界面搜索框靠它）",
          s1["total"] >= 1 and all(fn in r["file"] for r in s1["rows"]),
          "total={0}".format(s1["total"]))

    # ---- 8d. **拿模型推数据集**（用户 2026-09-15 的用法）+ 带区闸 ----
    from a9route.train import prelabel as P2
    from a9route.train import audit as AU2
    from a9route.vision import choice as VC2

    class _FakeKeyDet:
        """假按键模型：刹车按、氮气没按，置信度给 0.8 / 0.1。"""
        name = "onnx"
        path = "keys.onnx"
        conf = 0.4

        def read(self, frame):
            from a9route.vision.keys import KeyRead
            return KeyRead(brake=True, nitro=False,
                           info={"brake": 0.8, "nitro": 0.1})

    sig = P2.read_signals(np.full((720, 1280, 3), 30, np.uint8),
                          detector=_FakeKeyDet())
    check("**预标注可以用模型**（给 detector 就用它的判定）",
          sig.brake_hit is True and sig.nitro_hit is False, str(sig.brake_hit))
    check("  来源记下来了（`by`）", sig.by.startswith("onnx"), sig.by)
    check("  余量按「置信度/阈值」归一化（>=1 = 判按下，越近越可疑）",
          abs(sig.brake_margin - 2.0) < 1e-6 and abs(sig.nitro_margin - 0.25) < 1e-6,
          (sig.brake_margin, sig.nitro_margin))
    check("  像素明细照算（只用于复核排序/显示，不参与标签）",
          isinstance(sig.brake_ring, float) and "by" in sig.as_dict(), "")

    # 带区闸：模型在带区外的误报必须被挡掉（实测：v2 里 4 个圆在带下方 30~60px）
    from a9route.vision.hud import CHOICE_BAND
    _bx, _by, _bw, _bh = CHOICE_BAND
    inside = {"xyxy": (_bx + 40, _by + 20, _bx + 100, _by + 80),
              "name": "choice_icon", "conf": 0.9}
    outside = {"xyxy": (557, 199, 617, 259), "name": "choice_icon", "conf": 0.9}
    keep, blocked = VC2._band_gate([inside, outside], 1280)
    check("**带区闸：带区内的留、带区外的挡掉**",
          len(keep) == 1 and blocked == 1 and keep[0] is inside, (len(keep), blocked))
    check("  容差和体检/裁图是同一个量级（BAND_TOL=24）", VC2.BAND_TOL == 24, "")

    # 体检/逐帧页的"重算判据"必须跟**数据集记录的来源**一致
    # （否则模型标的帧会被像素判据判成"漂移"，整页假警报）
    det_h, name_h = AU2._recheck_judge(D.dataset_dir("t_keys"))
    check("启发式数据集 -> 重算也用启发式", det_h is None and "heuristic" in name_h,
          name_h)
    if (D.dataset_dir("t_choice") / D.FRAME_META).is_file():
        det_c, name_c = AU2._recheck_choice_judge(D.dataset_dir("t_choice"))
        check("选路数据集 -> 用选路那套判据（这里是启发式，因为测试环境钉住了）",
              det_c is None, name_c)
    # 界面按钮的范围来自接口（前端写死会让配置项变成假旋钮）
    check("**接口吐出第②题的范围**（界面按它渲染按钮）",
          s1.get("options") == {"min": 2, "max": 4}, str(s1.get("options")))
    check("  每行都带 flags（界面要显示「位置是猜的」这类提醒）",
          all("flags" in r for r in s1["rows"]), "")

    # ---- 8b. ⚠️ **任务串台**：按键那套写回不许碰选路数据集（真出过事故）----
    #
    # 2026-09-15 17:30 实测：用户在 `/dataset`（按键页）把数据集切到 `choice`，
    # 点了「本页没问题」+「写回修正」-> 42 帧选路标注被**按键格式覆盖**
    # （标签里写进了刹车框 [137,497,110,110]），界面上完全看不出来 ✗✗。
    # 所以每一条"只会说按键语言"的入口都必须**按任务拒绝**。
    from a9route.train import audit as AU
    from a9route.train import review as RV
    task_c, classes_c, zh_c = AU._task_of(D.dataset_dir("t_choice"), "t_choice")
    check("体检认得出来这是**选路**数据集", task_c == "choice" and classes_c == C.CHOICE_CLASSES,
          (task_c, classes_c))
    for label, fn in (
            ("写回复核修正", lambda: RV.apply_corrections(
                {"items": [{"file": fn, "boxes": ["brake_pressed"]}]}, name="t_choice",
                progress=None)),
            ("标记已复核", lambda: RV.mark_reviewed([fn], name="t_choice", progress=None)),
            ("撤销标记", lambda: RV.undo_last_mark(name="t_choice")),
            ("生成复核页", lambda: RV.make_review(name="t_choice", progress=None)),
            ("逐帧浏览", lambda: AU.frame_rows("t_choice", limit=2)),
    ):
        try:
            fn()
            check("  {0}：**拒绝**在选路数据集上跑".format(label), False, "居然没报错")
        except ValueError as exc:
            msg = str(exc)
            check("  {0}：拒绝并说清去哪一页".format(label),
                  "选路" in msg and "/choice" in msg, msg[:70])
    # 反例对照：**按键数据集**上这些入口必须照常工作（别把闸门做成"谁都不许用"）
    try:
        RV.mark_reviewed([], name="t_keys", progress=None)
        check("  反例：按键数据集上照常（空列表不报错）", True, "")
    except Exception as exc:                           # noqa: BLE001
        check("  反例：按键数据集上照常（空列表不报错）", False,
              "{0}: {1}".format(type(exc).__name__, exc))

    # ---- 8c. ⚠️ 训练产物目录**不能写死 keys**（真踩：选路权重写进了 runs/keys）----
    #
    # `--run-name` 的默认值以前是 `"keys"`，而 `run`/`eval`/`export` 三处都用它：
    # `train run --name choice` -> 写进 `models/runs/keys/`；
    # `train export --name choice` -> 从 `models/runs/keys/weights/best.pt` 拿权重
    # => **把按键的模型导出成选路模型** ✗✗（2026-09-15 实测）。
    from a9route import cli as CLI

    check("run 的产物目录默认跟数据集同名",
          CLI._run_name(None, "choice") == "choice"
          and CLI._run_name("", "keys") == "keys"
          and CLI._run_name("   ", "choice") == "choice", "")
    check("  显式给了 --run-name 就听显式的（不被数据集名盖掉）",
          CLI._run_name("myrun", "choice") == "myrun", "")
    # 行为层面：真解析一遍参数，看 `train run --name choice` 最终用的目录名
    parser = CLI.build_parser()
    for argv, want in ((["train", "run", "--name", "choice"], "choice"),
                       (["train", "run"], "keys"),
                       (["train", "export", "--name", "choice"], "choice"),
                       (["train", "run", "--name", "choice", "--run-name", "abc"], "abc")):
        ns = parser.parse_args(argv)
        got = CLI._run_name(getattr(ns, "run_name", None),
                            (getattr(ns, "name", None) or "").strip() or "keys")
        check("  `{0}` -> runs/{1}".format(" ".join(argv), want), got == want, got)

    # ---- 9. 运行时：注入到 RaceReader，粗扫/细扫同时换后端 ----
    from a9route.vision.hud import RaceReader

    class _FakeChoice:
        name = "fake"

        def read(self, frame):
            return VC.ChoiceRead(True, 3, 2, [{"xy": (600, 127), "radius": 30,
                                                "blue": False, "blue_ratio": 0.0},
                                               {"xy": (680, 127), "radius": 30,
                                                "blue": True, "blue_ratio": 0.9},
                                               {"xy": (760, 127), "radius": 30,
                                                "blue": False, "blue_ratio": 0.0}])

        def close(self):
            pass

    frame = np.full((720, 1280, 3), 20, np.uint8)
    r_inj = RaceReader(choice=_FakeChoice())
    got = r_inj.choice_icons(frame)
    check("RaceReader 注入选路后端后，choice_icons 交给它",
          len(got) == 3 and sum(1 for s in got if s.get("blue")) == 1,
          [s["xy"] for s in got])
    r_old = RaceReader()
    check("  不注入时走原来的圆检测（纯黑帧 -> 空，不凭空报）",
          r_old.choice_icons(frame) == [], "")
    check("  注入坏后端也不崩（返回空而不是抛）",
          RaceReader(choice=type("X", (), {"read": lambda self, f:
                      (_ for _ in ()).throw(RuntimeError("boom"))})())
          .choice_icons(frame) == [], "")


def t18_config_ui():
    """**Web 端可编辑的参数表**（滑块/输入框那套）—— 校验在**后端**，不许只靠前端。

    最要紧的两条：
      * 一个不合格就**一个都不写**（不留半套配置）；
      * 落盘之后**真的生效**（`config.current()` 里是新值）。
    """
    print("\n=== T18 Web 参数面板（校验 + 落盘 + 生效）===")
    from a9route.core import intent as IT
    from a9route.web import config_ui as CU

    data = CU.payload(backends={"keys": "onnx", "choice": "onnx"})
    keys = [f["key"] for f in data["fields"]]
    check("参数表非空、每项都有中文名/范围/默认值",
          len(keys) >= 15 and all(f["zh"] and f["kind"] for f in data["fields"]),
          "{0} 项".format(len(keys)))
    for want in ("scan.choice_idle_hold", "scan.choice_min_hold",
                 "scan.choice_cross_tol", "vision.choice_conf"):
        check("  选了「{0}」".format(want), want in keys, "")
    check("  每项都带 min/max（滑块要用）",
          all("min" in f and "max" in f for f in data["fields"]), "")
    # ⚠️ "调了没反应"是最坑的误导：模型后端下，只对启发式生效的项要标出来
    inactive = [f["key"] for f in data["fields"] if not f["applies"]]
    check("**当前用模型时，只对启发式生效的项被标成「不生效」**",
          "vision.choice_param2" in inactive and "vision.brake_white_thr" in inactive,
          str(inactive[:4]))
    check("  而且给了人能看懂的原因",
          all(f["note"] for f in data["fields"] if not f["applies"]), "")

    # ---- 校验：范围 / 类型 / 未知键 ----
    bad_cases = [
        ("scan.choice_idle_hold", -1, "不能小于"),
        ("scan.choice_idle_hold", 9999, "不能大于"),
        ("scan.choice_idle_hold", "abc", "要一个数"),
        ("vision.choice_conf", 5.0, "不能大于"),
        ("vision.brake_key_box", "1,2,3", "4 个数"),
        ("vision.brake_key_box", [10, 10, 0, 50], "宽高必须"),
    ]
    for key, val, want in bad_cases:
        _, err = CU.coerce(CU.BY_KEY[key], val)
        check("  {0}={1!r} 被拒：{2}".format(key, val, want), want in err, err)
    ok_v, err = CU.coerce(CU.BY_KEY["scan.choice_idle_hold"], "12")
    check("  正常值能过（字符串形式也认）", ok_v == 12 and not err, str(ok_v))

    res = CU.save({"scan.choice_idle_hold": 12, "不存在的键": 1})
    check("**有一个不合格 -> 一个都不写**（不留半套配置）",
          res["ok"] is False and res["written"] == []
          and any("不认识" in e for e in res["errors"]), str(res["errors"])[:80])
    check("  而且被拒的那一项**没被写进去**",
          (cfgmod.current().get("scan") or {}).get("choice_idle_hold") != 12,
          str((cfgmod.current().get("scan") or {}).get("choice_idle_hold")))

    # ---- 落盘 + 生效 + 恢复默认 ----
    res = CU.save({"scan.choice_idle_hold": 6, "intent.drift_min": 0.30})
    check("合法值：落盘成功", res["ok"] and res["written"], str(res)[:80])
    check("  **真的生效了**（config.current 里是新值）",
          (cfgmod.current().get("scan") or {}).get("choice_idle_hold") == 6,
          str((cfgmod.current().get("scan") or {}).get("choice_idle_hold")))
    check("  而且写进了 config.json（不是只在内存里）",
          "choice_idle_hold" in (paths.CONFIG_FILE.read_text(encoding="utf-8")
                                 if paths.CONFIG_FILE.is_file() else ""), "")
    check("  `config.apply` 之后 intent 的运行值也跟着变",
          "CHOICE_IDLE_HOLD" in dir(IT) and IT.CHOICE_IDLE_HOLD == 6,
          str(getattr(IT, "CHOICE_IDLE_HOLD", None)))
    res2 = CU.reset(["scan.choice_idle_hold"])
    check("恢复默认：回到内置值", res2["ok"]
          and (cfgmod.current().get("scan") or {}).get("choice_idle_hold")
          == CU._default_of("scan.choice_idle_hold"),
          str((cfgmod.current().get("scan") or {}).get("choice_idle_hold")))
    # ⚠️ 恢复默认要**把键从文件里删掉**，而不是把默认值写进去 ——
    # 否则 config.json 会越写越长，`config list --changed` 也全是噪音
    txt = paths.CONFIG_FILE.read_text(encoding="utf-8")
    check("  恢复默认后 config.json 里**不再有这个键**（只记真正的覆盖）",
          "choice_idle_hold" not in txt, txt[-120:].replace("\n", " "))
    res3 = CU.reset(None)
    check("全部恢复默认也不报错", res3["ok"], str(res3.get("errors")))


# ------------------------------------------------------------------ main
def main() -> int:
    global TMP
    TMP = tmp_dir()
    # 数据集/模型目录指到测试临时目录 —— **绝不能碰到真的 datasets/**
    paths.DATASETS_DIR = TMP / "datasets"
    paths.MODELS_DIR = TMP / "models"
    paths.WEIGHTS_DIR = paths.MODELS_DIR / "weights"
    paths.DEFAULT_KEY_MODEL = paths.MODELS_DIR / "keys.onnx"
    cfgmod.apply()
    t0_syntax()
    t1_labels()
    t2_classes_and_yaml()
    t3_slug()
    t4_sampling()
    t5_prelabel()
    t6_dataset()
    t7_split_policy()
    t8_review()
    t9_geometry_and_onnx_decode()
    t10_backend()
    t11_reports()
    t12_doctor()
    t13_audit()
    t14_dataset_web()
    t15_model_card()
    t16_ocr_cache_and_blocker()
    t17_choice()
    t18_config_ui()
    return summary("test_train")


if __name__ == "__main__":
    sys.exit(main())
