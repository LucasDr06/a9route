# models/ —— 模型与权重（**分两类管：运行时模型入库，训练产物不入库**）

```
models/
├── weights/                ✗ 不入库
│   ├── yolov8n.pt          YOLOv8n 预训练权重（6.5 MB）
│   └── yolo26n.pt          ultralytics 做 AMP 自检要用的（可选）
├── keys.onnx               ✓ **入库** 按键（刹车/氮气），运行时加载的就是它
├── keys.json               ✓ **入库** 模型清单：类别顺序 / imgsz / 训于 / 数据集 / 指标
├── choice.onnx             ✓ **入库** 选路（另一套类别，2026-09-15 训）
├── choice.json             ✓ **入库** 同上（classes=choice_icon,choice_selected）
├── runs/keys/              ✗ 不入库 训练产物：weights/best.pt + 曲线图 + results.csv
├── runs/choice/            ✗ 不入库 （同上，名称跟数据集走）
└── .ultralytics/           ✗ 不入库 ultralytics 的 settings 缓存（自动生成，可删）
```

## 为什么运行时模型要入库（2026-09-15 改的口径）

* **没有它们 `analyze` 会直接报错** —— `key_backend`/`choice_backend` 默认是 `auto`，
  而 `auto` **不再静默退回启发式**（模型不在就报"这次分析无效"并给出修法）。
  所以模型不入库的话，别人（以及你换机器之后）clone 下来**跑不了**；
* 体积**只有 ~23 MB**（两个 ONNX 各 11.67 MB + 两份 1 KB 清单），
  GitHub 单文件 100 MB 的线远得很，也不值得上 LFS；
* **训练产物反过来不入库**：`models/runs/` 有 66 MB（best.pt / last.pt / 曲线 / 混淆矩阵），
  它们是**过程产物**，重训一遍就有；`weights/`（预训练权重）能用
  `a9route train fetch` 重新下；`keys_v1_uncorrected.onnx` 那种旧备份也留在本地。

> 清单里的 `source_weights` 记的是**相对 `models/` 的路径**（`runs/choice/weights/best.pt`），
> 不是本机绝对路径 —— 清单会跟着仓库走，一条 `D:\...` 对别人毫无意义 ✗
> （`runner.export_onnx` 现在就是这么写的）。

**两个任务是两套模型、两套类别、两套配置**（`key_*` / `choice_*`），
互相之间切错会被清单里的类序检查拦住（`train use` 和运行时都会拒绝）。

## 换模型：清单 + 一条命令

`keys.json` / `choice.json` 是模型的**身份证**，存在的意义只有一个：
**别让"换一个模型"变成一次猜**。里面记着类别顺序（第 0 个是刹车还是路标）、
推理分辨率、训练时间、数据集和指标。

```powershell
& $py -m a9route train models       # 列出现有模型（含类序/imgsz/指标，标出正在用哪个）
& $py -m a9route train use keys     # 切换：一次写好 后端 + 模型 + imgsz
& $py -m a9route train use choice   # 选路那个（写 choice_backend/choice_model/choice_imgsz）
& $py -m a9route config set vision__key_backend=heuristic   # 退回启发式
```

* **类序对不上就拒绝**：`train use` 会拒绝切换，运行时 `OnnxKeys` 也会拒绝加载。
  因为类别顺序反了会让刹车/氮气**整体错位、而且一路都不报错** —— 宁可起不来。
* **`imgsz` 跟着模型走**：`train use` 会把 `key_imgsz` 设成清单里的值；
  运行时若 `key_imgsz` 还是默认值，也会跟随清单（避免"导出 640、推理别的值"的静默错位）。
* 没有清单的老模型照常能用，只是少了自动核对（`train models` 会标出来）。

## 运行时用哪个

`vision.key_backend` / `vision.choice_backend` 默认都是 **`auto`**：对应的
`models/<名字>.onnx` 在就用模型，不在就退回启发式 —— **每次 analyze 的日志里都会打一行**
说明用的是哪个、为什么（不会让人以为在用模型）。

```powershell
& $py -m a9route config set vision__key_backend=onnx      # 强制用模型（文件不在就报错，不降级）
& $py -m a9route config set vision__choice_conf=0.30      # 选路的置信度（实测这一档最好）
& $py -m a9route config list --changed
```

`.onnx` 走 onnxruntime（运行时**不需要 torch**）；`.pt` 走 ultralytics（要 torch）。

> ⚠️ `analyze --set vision__choice_backend=heuristic` 这类**临时覆盖**以前是无效的
> （`resolve_backend()` 自己去读了磁盘配置），现在修好了，见 NOTES §10.28。

放到别的盘：`$env:A9ROUTE_MODELS = "E:\a9route-models"`

详见仓库根的 `TRAINING.md`（§0.2 是换模型，§0.3 是选路那条线）。
