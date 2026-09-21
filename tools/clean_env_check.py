# -*- coding: utf-8 -*-
"""tools/clean_env_check.py —— **验证 README 里的 clone 流程是真的能跑**。

README 写着：

    python -m venv .venv ; pip install -e ".[model]"
    a9route test all        # 应该全绿

这句话很容易在不知不觉中变成假的：只要有人在**运行路径**上顺手
`import yaml` / `import torch` / `import paddleocr`，本机（那些包都装着）
毫无感觉，而**干净 clone 一跑就炸**。
已经踩过一次：`train/labels.py` 在函数开头无条件 `import yaml`，
而 pyyaml 在 `.[train]` 这个 extra 里 ✗（2026-09-15）。

这个脚本把「运行环境不该有的依赖」**真的屏蔽掉**再跑一遍全套测试，
于是"README 那句话还成不成立"变成一个 10 秒内能回答的问题。

    python tools/clean_env_check.py

退出码 0 = 干净环境跑得通；非 0 = 有人在运行路径上引了训练/OCR 依赖。
"""
from __future__ import annotations

import builtins
import importlib
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

#: 「运行环境」= 基础依赖（numpy/opencv/flask）+ `.[model]`（onnxruntime）。
#: 下面这些**不属于**运行环境 —— 屏蔽掉它们再跑测试。
BLOCKED = {
    "yaml",              # .[train]
    "torch",             # .[train]，训练环境
    "torchvision",
    "ultralytics",       # .[train]
    "matplotlib",        # ultralytics 的依赖
    "paddleocr",         # .[ocr]，几百 MB
    "paddle",
    "paddlex",
    "pytest",            # .[dev]
}

MODULES = ["a9route.tests.test_route", "a9route.tests.test_intent",
           "a9route.tests.test_video", "a9route.tests.test_webvideo",
           "a9route.tests.test_train"]

_real_import = builtins.__import__


def _blocked(name, *args, **kwargs):
    if str(name).split(".")[0] in BLOCKED:
        raise ImportError("[clean_env_check] 故意屏蔽 {0}（它不在运行依赖里）".format(name))
    return _real_import(name, *args, **kwargs)


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    builtins.__import__ = _blocked
    # 有些模块可能已经被别的东西顺带导入过了 —— 先踢掉，才能真的验出依赖
    for mod in list(sys.modules):
        if mod.split(".")[0] in BLOCKED:
            del sys.modules[mod]

    results = []
    from a9route.tests.support import results as shared

    for name in MODULES:
        buf = io.StringIO()
        before = len(shared)          # 各模块共用一张结果表 —— 取差值才是这个模块自己的项数
        try:
            mod = importlib.import_module(name)
            with redirect_stdout(buf):
                code = mod.main()
        except ImportError as exc:
            results.append((name, None, "ImportError: {0}".format(exc)))
            continue
        except Exception as exc:                       # noqa: BLE001
            results.append((name, None, "{0}: {1}".format(type(exc).__name__, exc)))
            continue
        n_fail = sum(1 for _, ok, _ in shared[before:] if not ok)
        results.append((name, code, "共 {0} 项，失败 {1} 项".format(
            len(shared) - before, n_fail)))

    print("=" * 64)
    print("模拟「干净 clone」：只装了 numpy/opencv/flask/onnxruntime，屏蔽掉：")
    print("   " + ", ".join(sorted(BLOCKED)))
    print("=" * 64)
    bad = [r for r in results if r[1] != 0]
    for name, code, info in results:
        flag = "OK  " if code == 0 else "FAIL"
        print("  [{0}] {1}  {2}".format(flag, name, info))
    print("=" * 64)
    if bad:
        print("结论：**README 里那句「pip install -e \".[model]\" 之后 test all 全绿」现在不成立** ✗")
        print("      上面 FAIL 的那个模块在运行路径上引了不该有的依赖。")
        return 1
    print("结论：干净环境跑得通 ✓（README 的 clone 流程是真的）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
