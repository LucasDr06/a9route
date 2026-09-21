# -*- coding: utf-8 -*-
"""离线测试包（**不需要设备、不需要网络，跑几秒**）。

    python -m a9route test all          # 或者 a9route test all
    python -m a9route.tests.test_video  # 单独跑一个

五个模块：

| 模块 | 锁什么 |
|---|---|
| `test_route`  | 路线脚本格式的逐条语义（解析 / 校验 / 扁平逗号流兼容） |
| `test_intent` | 信号形状 -> 操作目的（360 / 漂移 / 打断氮气 / 单击双击长按） |
| `test_video`  | 时间轴、逐百分点截图、按键段、路线生成、**真帧回归样本** |
| `test_webvideo` | Web 窗口几个接口（含上传、目录穿越防护、端到端一条龙） |
| `test_train`  | **训练框架**（YOLO 标注读写、抽帧优先级、预标注、数据集划分、 复核写回、letterbox 几何、ONNX 输出解码、后端选择） |

`support.py` 放共用的小工具（`check()` 计数、夹具路径、合成视频）。

⚠️ `test_train` **不需要 torch / ultralytics / onnxruntime** ——
训练那一路的依赖都在函数内 import，所以运行环境（只有 cv2/numpy）也能跑全量回归。
ONNX 的输出解码是用**手工构造的 numpy 数组**验的（不需要真的模型）。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from a9route import config as _cfgmod
from a9route import paths as _paths

#: 把 config.json 的配置灌进各模块 —— 测试要跑在**和真实运行同一套**常量上，
#: 否则测出来的判据和线上不是一回事（这条吃过亏）。
_cfg = _cfgmod.load_config()

#: ⚠️ **回归测试必须与"磁盘上有没有模型"以及"用户改过 config.json 没有"无关。**
#:
#: 默认后端是 `auto`（`models/keys.onnx` 在就用模型）。于是同一条测试会在
#: "没训过模型"的机器上走启发式、在"训过模型"的机器上走 ONNX ——
#: **同一份代码在不同机器上测试结果不同** ✗✗。
#: 实测就是这么踩的：`test_video` 端到端那条在合成帧上认不出按键，因为它
#: 悄悄换成了真模型（真模型没见过"画出来的圆圈"，当然不亮）。
#:
#: 两道闸一起上：
#:
#: 1. `CONFIG_FILE` 指到不存在的文件 —— 测试读的是**代码内置默认值**，
#:    不是"这台机器上恰好被改成的样子"（`config.json` 里正好写着 `onnx` 时，
#:    只钉 `key_backend` 是没用的：`scan_fine` 会重新 `load_config()`）；
#: 2. `DEFAULT_KEY_MODEL` 指到不存在的文件 —— 即使后端是 `auto`，
#:    也只会退回启发式。
#:
#: 要测模型那一路的用例，自己显式传 `keys=` / `backend=`
#: （见 `test_video` 的 T10、`test_train` 的 T15）。


def _safe_tmp() -> Path:
    """测试要用的临时目录：**绝对路径、在仓库之外、而且真的可写**。

    ⚠️ **别直接信 `tempfile.gettempdir()`**（实测踩过）：在受限沙箱里它曾经指向一个
    **相对路径**，于是 `_CFG_FILE = gettempdir()/"a9route-tests-config.json"`
    落到了**仓库根目录**，成了一个待提交的垃圾文件 ✗；
    同一个原因，探针写的小视频和截图目录也掉在了仓库根里。
    所以：绝对 + 不在仓库内才用它，否则退到 `worktmp/test_tmp/`（已 gitignore）。
    """
    try:
        cand = Path(tempfile.gettempdir())
        if not cand.is_absolute():
            raise ValueError("tempdir 不是绝对路径")
        if cand.resolve() == _paths.ROOT or _paths.ROOT in cand.resolve().parents:
            raise ValueError("tempdir 落在仓库里")
        cand.mkdir(parents=True, exist_ok=True)
        probe = cand / ".a9route-write-test"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        return cand
    except Exception:                                  # noqa: BLE001
        fallback = _paths.ROOT / "worktmp" / "test_tmp"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


_TMPDIR = _safe_tmp()
#: ⚠️ 光在内存里 `apply({"key_backend": "heuristic"})` **不够**：
#: `analysis.analyze()` 一进来就 `load_config()` + `apply()`，会把内存里的改动**盖掉**
#: （`load_config()` 读的是 `CONFIG_FILE`，而它被指到了一个不存在的文件 ->
#:  回到内置默认值 `auto`）。以前这么写没暴露，是因为 `auto` 那时会**静默退回启发式**；
#: 现在 `auto` 改成"模型不在就报错"之后，`test_webvideo` 立刻红了 ——
#: 那正好说明：**以前这些测试是在依赖那个静默兜底**。
#: 所以配置要落成**文件**，让任何一次 `load_config()` 都读到同一份。
_CFG_FILE = _TMPDIR / "a9route-tests-config.json"
_CFG_FILE.write_text(json.dumps({
    "vision": {"key_backend": "heuristic", "key_model": "",
               "choice_backend": "heuristic", "choice_model": ""},
}, ensure_ascii=False, indent=2), encoding="utf-8")
_paths.CONFIG_FILE = _CFG_FILE
_paths.DEFAULT_KEY_MODEL = _TMPDIR / "a9route-tests-no-such-model.onnx"
#: 选路那一套同理（2026-09-15 补）：**钉住之前真的红过** ——
#: 机器上训出 `models/choice.onnx` 之后，端到端分析开始自动加载真模型，
#: "这台机器上恰好有模型"又变成了测试结果的一部分 ✗
_paths.DEFAULT_CHOICE_MODEL = _TMPDIR / "a9route-tests-no-such-choice.onnx"
_cfg = _cfgmod.load_config()
_cfgmod.apply(_cfg)
assert not os.environ.get("A9ROUTE_vision__key_backend"), \
    "测试环境里不该用环境变量指定按键后端（会让测试结果依赖环境）"
