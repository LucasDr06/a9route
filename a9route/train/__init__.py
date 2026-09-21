# -*- coding: utf-8 -*-
"""a9route.train —— **按键识别模型（YOLOv8）的训练框架**。

```powershell
# 0) 先自检：现在这个环境能干什么、缺什么
a9route train doctor

# 1) 抽帧 + 启发式预标注（复用 cues.key_pressed 的实测判据）
a9route train build output\录像.mp4 --name keys

# 2) 生成复核页，人工把标错的改掉（**这一步才是模型的增量**）
a9route train review --name keys
a9route train apply 下载的_corrections.json --name keys

# 3) 按视频划分 train/val（防数据泄漏）并体检
a9route train split --name keys
a9route train verify --name keys

# 4) 训练（要在有 torch 的环境里跑）→ 评估 → 导出 ONNX
a9route train run --name keys --epochs 100
a9route train eval --name keys
a9route train export

# 5) 切运行时后端（**显式**，不会因为磁盘上多了个文件就悄悄变）
a9route analyze 录像.mp4 --set vision__key_backend=onnx
```

## 这个框架的核心判断（读代码前先看这三条）

1. **标注的来源是启发式 + 人工纠错，不是从零手标。**
   `vision/cues.py` 的判据是实测出来的，召回不差；差的是边界和假阳。
   所以 `frames.py` 按**优先级**抽帧（翻转帧 / 两通道吵架 / 贴着阈值），
   `review.py` 按"最可能标错"排序 —— 把人的时间花在真正有分歧的帧上。

2. **模型要赢的是"逐帧 P/R 和误报数"，不是 mAP。**
   两个按键位置固定、框永远一样大，mAP 天生接近满分，什么也证明不了。
   `runner.evaluate_frames()` 直接和**人工复核过的标注**比，
   并把**启发式当对照组**并排打出来 —— 模型没赢就别切后端。

3. **训练环境 ≠ 运行环境。**
   训练要 torch（本机 `ai_joy` 有 cu128），
   运行时只要 `onnxruntime`（`alphash9auto` 里装 15 MB 就够了，
   不必为此塞 2.5 GB 的 torch）。

## 模块地图

| 文件 | 管什么 |
|---|---|
| `labels.py` | 类别定义、YOLO 标注读写、`data.yaml` |
| `frames.py` | 抽哪些帧（优先级采样：翻转帧/吵架/贴阈值/正样本/背景） |
| `prelabel.py` | 启发式预标注（**和运行时同一套判据**） |
| `dataset.py` | 数据集布局、构建、**按视频划分**、校验、统计 |
| `review.py` | 复核页 + 导出修正 + 写回标注 + 拼图 |
| `runner.py` | 训练 / 逐帧评估 / 导出 ONNX / 整片脉冲对比 |
| `doctor.py` | 环境与素材自检（"现在这环境能干什么"） |

⚠️ **本包只在需要时 import torch/ultralytics**（都在函数内 import），
所以运行环境（只有 cv2/numpy）也能 `import a9route.train.dataset`。
"""
from __future__ import annotations

__all__ = [
    "labels", "frames", "prelabel", "dataset", "review", "runner", "doctor",
]
