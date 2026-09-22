# models/ —— 运行时模型（**入库**）

```
models/
├── keys.onnx               ✓ **入库** 按键（刹车/氮气），运行时加载的就是它
├── keys.json               ✓ **入库** 模型清单：类别顺序 / imgsz / 训于 / 数据集 / 指标
├── choice.onnx             ✓ **入库** 选路（另一套类别）
├── choice.json             ✓ **入库** 同上（classes=choice_icon,choice_selected）
├── keys_v1_uncorrected.onnx  本机留的旧备份（**不入库**，可删）
└── （训练产物 runs/、预训练权重 weights/：**都在 `a9lab` 那边**）
```

## 为什么运行时模型要入库（2026-09-15 改的口径）

* **没有它们 `analyze` 会直接报错** —— `key_backend`/`choice_backend` 默认是 `auto`，
  而 `auto` **不再静默退回启发式**（模型不在就报"这次分析无效"并给出修法）。
  所以模型不入库的话，别人（以及你换机器之后）clone 下来**跑不了**；
* 体积**只有 ~23 MB**（两个 ONNX 各 11.67 MB + 两份 1 KB 清单），
  GitHub 单文件 100 MB 的线远得很，也不值得上 LFS；
* **训练产物反过来不入库**：`runs/`（best.pt / last.pt / 曲线 / 混淆矩阵，几十 MB）
  是**过程产物**，重训一遍就有 —— 它们是 `a9lab` 的资产，在本项目里一个都不留。

> 清单里的 `source_weights` 记的是**相对路径**（`runs/choice/weights/best.pt`），
> 不是本机绝对路径 —— 清单会跟着仓库走，一条 `D:\...` 对别人毫无意义 ✗

**两个任务是两套模型、两套类别、两套配置**（`key_*` / `choice_*`），
互相之间切错会被清单里的类序检查拦住（`a9lab install` 和运行时都会拒绝）。

## 换模型：清单 + 一条命令

`keys.json` / `choice.json` 是模型的**身份证**，存在的意义只有一个：
**别让"换一个模型"变成一次猜**。里面记着类别顺序（第 0 个是刹车还是路标）、
推理分辨率、训练时间、数据集和指标。

```powershell
# 训练与导出在另一个项目里做；真正"装回本项目"只需要一条命令：
a9lab models                    # 两边各有什么模型（含类序/imgsz/指标，标出正在用哪个）
a9lab install v3.onnx           # 装进 models/ + 写好本项目的配置（后端/模型/imgsz）
                                # 会**核对类别顺序**：选路模型装成 keys.onnx 直接拒绝

& $py -m a9route config set vision__key_backend=heuristic   # 退回启发式（调试用）
```

* **类序对不上就拒绝**：`a9lab install` 会拒绝装，运行时 `OnnxKeys` 也会拒绝加载。
  因为类别顺序反了会让刹车/氮气**整体错位、而且一路都不报错** —— 宁可起不来；
* **`imgsz` 跟着模型走**：装模型时会把 `key_imgsz` 设成清单里的值；
  运行时若 `key_imgsz` 还是默认值，也会跟随清单（避免"导出 640、推理别的值"的静默错位）；
* 没有清单的老模型照常能用，只是少了自动核对（`a9lab models` 会标出来）。

## 运行时用哪个

`vision.key_backend` / `vision.choice_backend` 默认都是 **`auto`**：对应的
`models/<名字>.onnx` 在就用模型；**不在就报错**（不退回启发式）——
每次 analyze 的日志里都会打一行说明用的是哪个、为什么。

```powershell
& $py -m a9route config set vision__key_backend=onnx      # 强制用模型（文件不在就报错）
& $py -m a9route config set vision__choice_conf=0.30      # 选路的置信度（实测这一档最好）
& $py -m a9route config list --changed
```

`.onnx` 走 onnxruntime（运行时**不需要 torch**）；`.pt` 走 ultralytics（要 torch）。

> ⚠️ `analyze --set vision__choice_backend=heuristic` 这类**临时覆盖**以前是无效的
> （`resolve_backend()` 自己去读了磁盘配置），现在修好了，见 NOTES §10.28。

放到别的盘：`$env:A9ROUTE_MODELS = "E:\a9route-models"`

清单的格式与校验代码在 `a9route/formats.py` —— 它是**两个项目之间的契约**，
`a9lab export` 写、这里加载时验。用法细节见 `a9lab` 的 README 和 `TRAINING.md`。
