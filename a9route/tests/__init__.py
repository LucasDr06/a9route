# -*- coding: utf-8 -*-
"""离线测试包（**不需要设备、不需要网络，跑几秒**）。

    python -m a9route test all          # 或者 a9route test all
    python -m a9route.tests.test_video  # 单独跑一个

四个模块：

| 模块 | 锁什么 |
|---|---|
| `test_route`  | 路线脚本格式的逐条语义（解析 / 校验 / 扁平逗号流兼容） |
| `test_intent` | 信号形状 -> 操作目的（360 / 漂移 / 打断氮气 / 单击双击长按） |
| `test_video`  | 时间轴、逐百分点截图、按键段、路线生成、**真帧回归样本** |
| `test_webvideo` | Web 窗口几个接口（含上传、目录穿越防护、端到端一条龙） |

`support.py` 放共用的小工具（`check()` 计数、夹具路径、合成视频）。
"""
from __future__ import annotations

from a9route import config as _cfg

#: 把 config.json 的配置灌进各模块 —— 测试要跑在**和真实运行同一套**常量上，
#: 否则测出来的判据和线上不是一回事（这条吃过亏）。
_cfg.apply()
