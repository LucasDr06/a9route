# -*- coding: utf-8 -*-
"""根目录入口 —— 与 `python -m a9route` 等价（方便"双击/直接跑"）。

    python main.py                  # 起本地 Web 窗口（默认 http://127.0.0.1:8790/）
    python main.py analyze 录像.mp4  # 逐百分点截图 + 推断操作 + 生成路线
    python main.py check --text "1,31,2,N:0:2:750"
    python main.py --help
"""
from __future__ import annotations

import sys

from a9route.cli import main

if __name__ == "__main__":
    sys.exit(main())
