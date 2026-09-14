# -*- coding: utf-8 -*-
"""`python -m a9route ...` —— 与 `a9route ...`（`a9route/cli.py`）等价。"""
from __future__ import annotations

import sys

from a9route.cli import main

if __name__ == "__main__":
    sys.exit(main())
