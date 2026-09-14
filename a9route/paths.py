# -*- coding: utf-8 -*-
"""集中路径常量 —— 全部相对**项目根**，不相对当前工作目录。

（这条是照搬 `asphalt9auto` 项目的教训：`Path("config")` 写在 4 个文件里，
换目录启动就读错标定。新项目从第一行就收口。）
"""
from __future__ import annotations

import os
from pathlib import Path

# a9route/paths.py -> a9route/ -> 项目根
ROOT = Path(__file__).resolve().parent.parent

# ---- 入库的配置（明文、需要版本管理）----
#: 判据阈值 / 框位置 / 采样间隔等**所有可调参数**都在这里（见 a9route/config.py）
CONFIG_FILE = ROOT / "config.json"
#: 路线脚本样例（格式说明 + 用户真实路线）
ROUTES_DIR = ROOT / "routes"
#: 离线测试用的小帧样本（PNG）
FIXTURES_DIR = ROOT / "a9route" / "tests" / "fixtures"

# ---- 运行产物（已 gitignore，随时可删）----
OUTPUT_DIR = ROOT / "output"
#: 运行产物目录。默认项目内 `worktmp/`，可用 **`A9ROUTE_WORKTMP`** 指到别处 ——
#: 受限沙箱 / 只读盘 / CI 里项目目录写不进去时，指到临时目录即可。
WORKTMP_DIR = Path(os.environ.get("A9ROUTE_WORKTMP") or (ROOT / "worktmp"))
#: 「跑图视频 -> 路线」的分析产物（每个任务一个子目录：上传的视频 + 逐百分点截图 + 路线）
ANALYSIS_DIR = WORKTMP_DIR / "analysis"
#: 上传的视频存这里
UPLOAD_DIR = WORKTMP_DIR / "uploads"

# ---- 本机私有设置（已 gitignore，不入库）----
LOCAL_DIR = ROOT / ".local"

# ---- Web 资源 ----
WEB_DIR = ROOT / "a9route" / "web"
WEB_TEMPLATES_DIR = WEB_DIR / "templates"
WEB_STATIC_DIR = WEB_DIR / "static"


def video_dirs() -> list[Path]:
    """「服务器上已有视频」下拉框要扫哪些目录。

    默认 `output/`（约定：把录像丢这里）；
    可用环境变量 `A9ROUTE_VIDEO_DIR` 追加，多个目录用 `os.pathsep` 分隔 ——
    这样几十 MB 的录像**不必**放进仓库（`.gitignore` 里也排除了视频后缀）。
    """
    out = [OUTPUT_DIR]
    extra = os.environ.get("A9ROUTE_VIDEO_DIR") or ""
    for part in extra.split(os.pathsep):
        part = part.strip()
        if part:
            out.append(Path(part))
    return out


def ensure_dirs(verbose: bool = False) -> None:
    """建好运行产物目录。**建不出来就直接说清楚怎么绕过**（别把裸的权限错误甩给用户）。"""
    for d in (OUTPUT_DIR, WORKTMP_DIR, ANALYSIS_DIR, UPLOAD_DIR, LOCAL_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"建不出运行目录 {d}（{type(exc).__name__}: {exc}）。\n"
                f"  多半是项目目录不可写（受限沙箱 / 只读盘 / CI）。绕过办法：\n"
                f'    把运行目录指到可写的地方，例如  $env:A9ROUTE_WORKTMP = "$env:TEMP\\a9route"'
            ) from exc
    if verbose:
        print(f"运行目录: {WORKTMP_DIR}")
