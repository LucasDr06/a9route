# -*- coding: utf-8 -*-
"""测试用的小工具：`check()` 计数、夹具路径、合成视频。"""
from __future__ import annotations

from pathlib import Path

from a9route.paths import FIXTURES_DIR

PASS, FAIL = "OK", "FAIL"

#: 所有测试模块共用的结果表（各模块自己的 `main()` 用 `summary()` 收尾）
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    """记一项断言：打印 `[OK]/[FAIL]` 并计数。"""
    results.append((name, bool(ok), detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  — {detail}" if detail else ""))
    return bool(ok)


def summary(title: str = "") -> int:
    """打印统计并返回退出码（0 = 全过）。"""
    n_fail = sum(1 for _, ok, _ in results if not ok)
    if title:
        print(f"\n=== {title} ===")
    print(f"\n{'=' * 46}\n共 {len(results)} 项，失败 {n_fail} 项")
    print("ALL TESTS PASS" if n_fail == 0 else "SOME TESTS FAILED")
    return 1 if n_fail else 0


def fixture(name: str) -> Path:
    """取 `a9route/tests/fixtures/<name>` 的路径。"""
    return FIXTURES_DIR / name


def tmp_dir() -> Path:
    """测试产物目录（建好再返回）。

    走 `paths.WORKTMP_DIR`，所以 `$env:A9ROUTE_WORKTMP` 一样能把它指到别处
    —— 受限沙箱/只读盘里跑测试时用得上。
    """
    from a9route import paths
    d = paths.WORKTMP_DIR / "test_tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- 合成素材
def synthetic_video(path: Path, *, seconds: float = 3.0, fps: float = 10.0,
                    size: tuple[int, int] = (320, 180),
                    percent_from: float = 0.0, percent_per_sec: float = 20.0,
                    hud=None) -> Path:
    """写一个小视频（不依赖真录像），用来测"抽帧循环"这条真路径。

    `hud(frame_idx, t, frame)`：可选的"画 HUD"回调 —— 不画就是纯渐变帧
    （那种帧里读不到「路程 NN%」，用来验证"读不到时如实报告"）。
    """
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * fps)
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for i in range(n):
            t = i / fps
            frame = np.full((h, w, 3), 20 + (i % 12) * 15, dtype=np.uint8)
            if hud is not None:
                hud(i, t, frame)
            writer.write(frame)
    finally:
        writer.release()
    return path


def fake_hud_video(path: Path, *, seconds: float = 6.0, fps: float = 10.0,
                   size: tuple[int, int] = (1280, 720), start_percent: float = 0.0,
                   percent_per_sec: float = 10.0,
                   brake_frames: tuple[int, ...] = (),
                   nitro_frames: tuple[int, ...] = (),
                   choice_frames=()) -> Path:
    """写一个**带假 HUD** 的小视频：左上角有「路程 NN%」，可按帧号画按键/路标。

    为什么要它：真录像几十 MB 不适合入库，而"端到端一条龙"（抽帧 → 读数 →
    判据 → 路线）必须有一条能用真 OpenCV 解码的路径来测。这里用最小的画面把
    每一路判据都"点亮"出来，于是整条流水线都能在 CI 里跑。
    """
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    # 按键框用**配置里的真实坐标**（1280×720），这样判据框是"真的"贴合
    from a9route.vision import cues
    n = int(seconds * fps)
    if start_percent <= 0:
        start_percent = 1.0
    try:
        for i in range(n):
            t = i / fps
            frame = np.full((h, w, 3), 30, dtype=np.uint8)
            # ---- 左上角 HUD：白字黑底，OCR 能读到的概率取决于字体，
            #      所以这里**不指望 OCR**：测试里注入 reader/reader_texts。
            pct = int(start_percent + t * percent_per_sec)
            cv2.putText(frame, f"{pct}%", (150, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)
            # ---- 刹车键：整圆涂白 = 按下（判据是"亮白占比/圈内-圈外"）
            if i in brake_frames:
                bx, by, bw, bh = cues.BRAKE_KEY_BOX
                cv2.circle(frame, (bx + bw // 2, by + bh // 2),
                           min(bw, bh) // 2 - 2, (255, 255, 255), -1)
            # ---- 氮气键：整圆涂红 = 按下（判据是"红占比"）
            if i in nitro_frames:
                nx, ny, nw, nh = cues.NITRO_KEY_BOX
                cv2.circle(frame, (nx + nw // 2, ny + nh // 2),
                           min(nw, nh) // 2 - 2, (40, 40, 245), -1)
            # ---- 选路路标：上方两个圆（一个蓝高亮）
            if i in choice_frames:
                for cx, color in ((610, (240, 240, 240)), (680, (230, 120, 60))):
                    cv2.circle(frame, (cx, 127), 30, color, -1)
            writer.write(frame)
    finally:
        writer.release()
    return path
