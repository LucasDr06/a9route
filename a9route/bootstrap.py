# -*- coding: utf-8 -*-
"""控制台编码修复（UTF-8）。

Windows 中文版控制台默认代码页是 **GBK(cp936)**，而本项目会打印 `✓` / `✗` 和中文，
直接跑会崩：

    UnicodeEncodeError: 'gbk' codec can't encode character '\u2713'

做法与 `asphalt9auto` 项目一致（那段是实测踩出来的，直接照搬）：

1. 把 `stdout` / `stderr` 重新配置成 UTF-8 + `errors="replace"`（绝不因编码中断流程）；
2. Windows 下把控制台代码页切到 65001，让子进程/外部程序也一致；
3. 设 `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` 给派生的子进程继承。

用 `A9ROUTE_NO_CONSOLE_SETUP=1` 可关掉（把本项目当库用时）。
"""
from __future__ import annotations

import os
import sys


def configure_console() -> None:
    """把当前进程的控制台统一到 UTF-8。幂等，可重复调用。"""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # 被重定向到非 TextIOWrapper（如 pythonw）时跳过
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # 已经写入过内容 / 流已关闭时会抛，忽略即可
            pass

    if os.name == "nt":
        _set_windows_codepage(65001)


def _set_windows_codepage(cp: int) -> None:
    """切换 Windows 控制台代码页；无控制台（服务/重定向）时静默跳过。"""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(cp)
        kernel32.SetConsoleCP(cp)
    except Exception:
        pass
