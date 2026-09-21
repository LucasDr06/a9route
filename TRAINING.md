# TRAINING —— 用 YOLOv8 训练「刹车 / 氮气」按键识别

把 `README.md`「已知限制」里那两条（**氮气偏多**、边界帧不稳）从"调阈值"换成"训模型"。

> 一句话结论：**框架已经建好，并且在这台机器上端到端跑通过一遍**
> （数据集 → 复核页 → 训练 → 逐帧评估 → 导出 ONNX → 运行时推理）。
> 你真正要准备的东西**只有录像和人工复核的时间**；
> 环境、依赖、预训练权重我都在这台机器上装好/下好了。

---

## 0. 实测结果（2026-09-15，手上这 3 段录像 + 572 帧人工复核）

| 环节 | 实测结果 |
|---|---|
| 抽帧 + 预标注 | 3 段录像 → **1200 帧 / 175 MB**（edge 720 / disagree 240 / bg 240） |
| 划分 | train 800 / val 400（val = **整段** `跑图转路线测试.mp4`，按视频分，无泄漏） |
| **人工复核** | 已复核 **572 帧**（train 466 / val 106），其中约 **115 帧真的改了标注** |
| 训练（**第 2 版**，用修正后的标注） | yolov8n / imgsz 640 / batch 16 / RTX 5070 Ti，**8.4 分钟**，epoch 21 提前停 |
| 训练指标 | **P 0.989 / R 0.978 / mAP50 0.970 / mAP50-95 0.970**（第 1 版：0.913 / 0.883 / 0.955 / 0.935） |
| **逐帧评估：只看人工复核过的 106 帧**（唯一可信的基准） | 刹车 **F1 0.933**、氮气 **F1 1.000** |
| 对照：启发式（预标注就是它） | 刹车 F1 **0.200**、氮气 F1 **0.195** |
| 对照：第 1 版模型（未修正标注） | 刹车 F1 0.219、氮气 F1 0.258 |
| 翻转帧正确率（val） | 0.926 → **0.982** |
| 导出 | `models/keys.onnx` **12.2 MB**（第 1 版留档 `models/keys_v1_uncorrected.onnx`） |
| 整片刹车动作数（标定视频） | 启发式 18 / 第 1 版 21 / **第 2 版 11** |

> **这两个 F1 是这个项目第一次拿到"可信的"模型评估** —— 基准是**人给的**，
> 不是启发式自己打的（那样必然满分、必然"模型没赢"，见 §6）。

### ⚠️ 但先别急着高兴：**刹车那半，调一行配置就够了**

对同一批人工答案做归因，发现两个通道的**病根完全不同**：

| 通道 | 病根 | 能不能靠调参解决 |
|---|---|---|
| **刹车** | `key_pressed()` 是 `圈内−圈外 **or** 绝对白度`。人标"未按"却判"按下"的 118 帧里，**117 帧（99%）是"只有绝对白度越线"**（圈内−圈外 平均 0.049，白度 平均 0.534）；而人标"按下"的 102 帧，圈内−圈外**全部**越线 | **能**：`--set vision__brake_white_thr=1.0` 关掉那条分支，启发式刹车 F1 **0.200 → 0.933**（和模型打平） |
| **氮气** | "瓶子变红 = 已充满"和"正在点"在**红占比**上**反相关**：误报中位 **0.466** > 真按下中位 **0.246** | **不能**：阈值 0.15→0.50，真按下先死光（32→0），误报还剩一半 |

**能用配置修的别用模型**：刹车那半一行配置即可；**氮气才是模型真正不可替代的地方**
（同一批帧上模型 F1 = 1.000）。详见 `NOTES.md` §10.16。

---

## 0.2 换模型：`models/<名字>.json` 清单 + 一条命令切换

模型不只有权重。要正确用起来还得知道**类别顺序**（第 0 个是刹车还是氮气）
和**推理分辨率** —— 顺序错了会让刹车/氮气整体错位，**而且一路都不报错** ✗✗。
所以 `train export` 会在 ONNX 旁边写一份**同名清单**：

```
models/keys.onnx
models/keys.json      <- classes / imgsz / conf / 训于 / 数据集 / 指标
```

```powershell
& $py -m a9route train models            # 看有什么模型（含类序、imgsz、指标）
& $py -m a9route train use keys          # 切换（**会核对类序，对不上直接拒绝**）
& $py -m a9route config set vision__key_backend=heuristic   # 退回启发式
```

* `train use` 会同时写好 `key_backend`、`key_model`、`key_imgsz`（跟着模型走）；
* **运行时也会核对**：类序对不上就**拒绝加载**，而不是静默跑一个错位的模型；
* 清单可缺（老模型/别人给的模型照常能用），只是少了自动核对；
* `key_backend` 默认是 **`auto`**：`models/keys.onnx` 在就用模型，不在就退回启发式
  —— 而且 `analyze` 的日志里**每次都会打一行**说明用的到底是哪个、为什么
  （不会让人以为在用模型）。

> 手工换：`a9route config set vision__key_model="D:\别的模型.onnx"`。
> `.onnx` 走 onnxruntime（不需要 torch），`.pt` 走 ultralytics（要 torch）。

---

## 0.3 第二套模型：**选路**（三个选择题）

按键那条线（刹车/氮气）之外，**选路**也是一套独立的模型 —— 它有自己的类别、
自己的数据集、自己的界面。两边**不共用**任何东西，这样换其中一个不会碰坏另一个。

```powershell
& $py -m a9route train choice-build --source keys      # 从现有数据集分类出 datasets/choice
& $py -m a9route train choice-stats                    # 看分布（几个选项/选第几个）
# 浏览器打开 http://127.0.0.1:8790/choice  ← **单独的界面，不和刹车/氮气放一起**
& $py -m a9route train run --name choice               # 训练（训练环境）
& $py -m a9route train export --name choice            # -> models/choice.onnx + models/choice.json
& $py -m a9route train eval --name choice --weights models\choice.onnx   # 三个答案的准确率
& $py -m a9route train use choice                      # 切换运行时（核对类序）
```

### 实测结果（2026-09-15，1200 帧**全部人工复核**）

```
训练：yolov8n / 60 轮 / 10.3 分钟 / imgsz 640
box_map50 = 0.9843（choice_icon 0.984、choice_selected 0.984）
导出：models/choice.onnx（11.7 MB）+ models/choice.json 清单
```

**但 mAP 不算数** —— 选路看的是三个答案对不对（`train eval`）：

| conf | 是否有选路 | 有几个选项 | 选第几个 | **三个同时都对** |
|---|---|---|---|---|
| 0.30（**当前用的**） | **0.998** | 0.961 | 0.961 | **0.990** |
| 0.35 | 0.995 | 0.974 | 0.947 | 0.990 |
| **启发式（对照组）** | 0.953 | 0.974 | 0.961 | 0.9425 |

* **赢在「是否有选路」**：错误 19 帧 -> **1 帧**。这正是启发式最烦人的毛病
  （屏幕上任何圆形装饰都可能被当成岔路口，多报）；
* 「有几个选项 / 选第几个」在 76 个真岔路口里差 1~3 帧，属于噪声级；
* 所以切模型是有依据的（合成口径 0.990 vs 0.9425），`vision.choice_conf` 取 **0.30**
  —— 报告里**两边都给这个合成数**，模型不如启发式时会直接写「先别切」。

> **整片对比**（同一条录像跑模型 / 启发式各一遍）：**内容完全一样**
> （13 条选路、值相同），只有两处选路的百分比差 1%（56↔57、82↔83，稳定闸的边界）。
> 也就是说这条录像上两条路线**等价**，模型的收益（少报假岔路口）要换录像/更长录像才兑现。

### 路线识别里**谁在做决定**（可以自己数一遍）

| 环节 | 靠什么判 |
|---|---|
| 「路程 NN%」 | PaddleOCR（一直是模型） |
| 刹车 / 氮气 | `models/keys.onnx` |
| 选路路标（几个 / 选第几个） | `models/choice.onnx` —— 粗扫和细扫**同一个后端** |
| 360 / 漂移 / 打断氮气 / 双击长按 | **信号形状规则**（`core/intent.py`）：输入是**模型判出的按键信号**，规则是用户定的口径 —— 刻意不用模型（要稳定、可解释） |
| **关自动驾驶 `S:2000`** | **OCR 文字**（读到「TOUCHDRIVE」）—— 用户口径："touchdrive 还是用 ocr 识别"。它是路线里唯一不来自 YOLO 的操作，所以路线正文里**单独列一段**标明来源 |

**不再参与判定的东西**（2026-09-15 按"判定全交给模型"清理）：

* `auto` 后端**不静默退回启发式** —— 模型不在就报错（这次分析无效），
  避免"看着正常、其实是像素判据"的路线；
* 「窗口汇总」那套估算（顶部氮气槽青 / 「漂移NN米」文字 / 「完成360度旋转」文字）
  **已经不出路线了**：没有模型判出的操作时，宁可**拒绝出路线**并说清原因；
* 漂移不再叠加 OCR 文字那一路（有按键模型时**只认模型**：漂移 = 按住刹车）；
* 粗扫的窗口汇总结果里，**只有** `auto`（OCR 判据）能进路线，其余一律不进。

日志里那几行只是**声明**。想知道"实际谁在判"，跑这个（会数每个后端被调用了多少次、
以及有没有偷偷落回启发式）：

```powershell
& $py worktmp\probe\who_decides_route.py "output\跑图转录像.mp4" 20
```

> 2026-09-15 就是靠它发现"粗扫还在跑 HoughCircles、而且能一票否掉模型判出的选路"的，
> 见 NOTES §10.30；"判定全交给模型"这一轮的清理见 §10.31。

### 标定时只回答**三个选择题**

| 问题 | 取值 | 落到 YOLO 标签上 |
|---|---|---|
| ① 是否有选路 | 没有 / 有 | 没有 -> **空标注文件**（背景帧）；有 -> 至少 2 个框 |
| ② 有几个选项 | 2 / 3 / 4 | 框的**个数**（按 x 从左到右排） |
| ③ 选第几个 | 1 / 2 / 3 / 4 | 第 k 个框是 `choice_selected`，其余都是 `choice_icon` |

* **只有 2/3/4 才算岔路口**（1 个图标是普通路标，不算选路）—— 所以 ② 没有「1」；
  这个范围可以在 `vision.choice_min_options` / `choice_max_options` 里改，
  **改了是真生效的**：界面按钮、后端夹取、运行时折叠三处都跟着变；
* 位置**不用手工画**：启发式预标注先给好位置，人只需要点三个答案；
  位置确实错了才在带状图上点一下加/删框 —— **点在检测到的路标附近会直接吸附到
  那个圆的精确位置**（并弹提示说明是"吸附"还是"按你点的位置放"）。
  改过位置的帧会记 `manual_position`；
* 反过来，**标注文件才是唯一真相**：页面上显示的第 ③ 题答案是从标注文件反推的
  （`answers_from_boxes()`），不是另存一份答案 —— 不会出现"答案和标注不一致"；
* 每帧下面还有后端算的**提醒标签**，照它看最省事：
  `位置是猜的`（检测到的图标不够，后端往右补了框 —— 补的位置不一定真）、
  `不算岔路口`（图标少于最小选项数）、`没标出选中`（没测到蓝高亮）。

### 为什么选路也要模型

现在的启发式是「在选路带里找圆 + 判蓝色高亮」。实测的问题是**多报**：
屏幕上很多圆形装饰（路标、图标、光圈）都会被当成选路图标，
而**它没有任何"是不是岔路口"的概念** —— 那三题的答案现在是从图标数硬折出来的。
模型能同时学到"这里到底有没有岔路口"和"高亮的是哪一个"，这两件事纯几何判据都很脆。

### 和按键那条线的区别（对照着看不会串）

| | 按键（刹车/氮气） | 选路 |
|---|---|---|
| 类别 | `brake_pressed` / `nitro_pressed` | `choice_icon` / `choice_selected` |
| 数据集 | `datasets/keys` | `datasets/choice` |
| 复核页面 | `/dataset` | **`/choice`**（另一页，另一套 JS） |
| 后端配置 | `vision.key_backend` / `key_model` / … | `vision.choice_backend` / `choice_model` / `choice_imgsz` / `choice_conf` / `choice_iou` / `choice_provider` / `choice_fmt` |
| 界面口径 | 每帧两个按钮按/没按 | 每帧三个选择题 + 选路带预览 |
| 运行时入口 | `RaceReader(brake/nitro)` | `RaceReader(choice=...)` 的 `choice_icons()` |

> 防串台：模型清单里记着 `classes`，`train use` 和运行时加载都会**核对类别**。
> 把选路模型切给按键（或反过来）会**直接报错**，不会静默跑出一个错位的结果。

### 数据集是**复用**按键那批帧的，不占额外磁盘

`choice-build` 用**硬链接**把 `datasets/keys` 的帧挂进 `datasets/choice`
（实测 1200 帧：link=1200 / copy=0，**额外占用 0 字节**），
并且**照抄源数据集的 train/val 划分** —— 自己重新随机分会导致同一段视频的帧
同时出现在两边，评出来的分数是假的（数据泄漏）。参见 §4。

---

## 0.1 TL;DR：要准备什么

| 要准备的 | 现在有吗 | 说明 |
|---|---|---|
| **跑图录像**（1280×720、完整 HUD） | ✅ 已有 3 段（共 70 MB / 4666 帧） | 想训得更好要**再多几段不同赛道/车**，见 §1 |
| **人工复核时间** | ⏳ **待你安排 —— 这是唯一还没做的一步** | 约 **20~40 分钟 / 300 帧**，见 §2 |
| **训练环境**（torch+ultralytics） | ✅ `ai_joy` 已装好 | torch 2.8.0+cu128 + ultralytics 8.4.152 + onnx 1.19.1 |
| **运行时依赖**（onnxruntime） | ✅ 已装 | `alphash9auto` 里 onnxruntime 1.30.0（**不需要 torch**） |
| **预训练权重** | ✅ 已下好 | `models/weights/yolov8n.pt`（6.5 MB） |
| **磁盘** | — | 实测 **约 145 KB/帧**；1200 帧 = 175 MB，3000 帧 ≈ 440 MB |
| 手机 / adb / 标定 | ❌ 不需要 | 这个项目从头到尾不碰设备 |

**不缺的东西**（别去准备）：不需要手工画框（两个按键的位置**固定已知**，预标注直接给）、
不需要从零标（启发式先标好，人只改错的）、训练不需要联网。

---

## 1. 素材：录像要准备成什么样

### 硬性要求（和 `analyze` 一样，因为用的是同一套坐标）

* **分辨率 1280×720**。按键框是 `BRAKE_KEY_BOX=(137,497,110,110)`、
  `NITRO_KEY_BOX=(1033,497,110,110)`，都是在这个分辨率上量的。
  → 换分辨率**不是不能训**，但预标注的框会错位，得先按 README「换分辨率时怎么重量框」
  把框调到新分辨率上，再 `train build`。
* **画面里有完整比赛 HUD**（左上角「路程 NN%」）。加载/倒计时那几屏也**有用** ——
  它们是"没有按键可点"的负样本，模型得学会那时**什么都不输出**。
* 格式随便（mp4/mkv/avi/mov/webm），能解码就行。

### 数量与覆盖度

| | 起步 | 理想 |
|---|---|---|
| 录像段数 | 1 段 | 5~10 段 |
| 总时长 | ≥ 5 分钟 | 30 分钟以上 |
| 抽样帧数 | 1,000~1,500 | 3,000~6,000 |

**为什么强调"多段"**：一段录像里的赛道背景、车辆、天气、UI 明暗基本固定，
模型很容易把"**这条赛道的背景**"当成特征背下来。换一段录像就崩。
`train split` 的**按视频划分**也正是为此 —— 同一段录像的相邻帧只差 33ms，
按帧随机划分会让 val 和 train 几乎一模一样，指标漂亮但全是假的（见 §5）。

### ⚠️ 素材里**必须**有的场景（负样本/困难样本的来源）

这些是启发式现在的误报老家，也是模型最需要学的：

1. **氮气瓶"充满变红"但没有按** —— NOTES §1 实测过：瓶子红 = **已充满/可用**，
   **不是"正在喷"**，两者只有 65% 一致。这是"氮气偏多"的头号嫌疑。
2. **赛道明暗剧变**（隧道进出口、雪地、夜间）—— 刹车键是**半透明叠加**，
   绝对白度会被背景带走，所以判据才要用"圈内 − 圈外"。
3. **漂移 / 360 期间**：刹车键长按、双击，本身就是"按下"，不要当负样本标掉
   （判"这次按下是为了什么"是下游 `core/intent.py` 的活）。
4. **非比赛画面**：加载、结算、菜单 —— 全是负样本。
5. **单击只有 1 帧**的情况（33ms）：NOTES §4 记着用户 17% 那两次点击
   在 59 帧里只亮 2 帧。这也是抽帧策略要专门捞"翻转帧"的原因。

### 现在手里这几段够不够

```
output\刹车氮气标定测试.mp4   16.3 MB   1146 帧   ← **最值钱的一段**
output\测试3.mp4              23.3 MB   1529 帧
output\跑图转路线测试.mp4      30.7 MB   1991 帧
```

`刹车氮气标定测试.mp4` 是 NOTES 里那份**用户亲手标定过**的录像
（刹车段 10~20s、氮气段 20~29s）。它现在被当成普通素材用了，
**建议单独留一份当"验收基准"**：训完模型在这段上跑 `train pulses`，
看刹车/氮气的动作数和你当初标的对不对得上 —— 这是**唯一不靠启发式也能判断好坏**的检查。

---

## 2. 标注：从哪来、要多少人工

### 流程：启发式预标注 → 人工只改错的

```
启发式（cues.key_pressed，实测过的阈值）  →  自动写标注
                     ↓
        人工复核**最可能标错的那批帧**
                     ↓
              复核过的标注 = 训练集
```

**为什么不能跳过复核**：拿启发式当标签训模型，是**自己教自己** ——
模型只会学会复现同一套阈值，一个误报都修不掉。复核这一步才是全部的增量。

### 复核页把最该看的帧排在最前面

`train build` 抽帧时就按优先级抽（`train/frames.py`），`train review` 再按
"最可能标错"排序：

| 顺序 | 类型 | 为什么先看它 |
|---|---|---|
| 1 | `edge` **翻转帧** | 单击/双击/长按的分界全在按下/松开那几帧 |
| 2 | `disagree` **两通道吵架** | 例如"圈内白度说按了、红说没按" —— 已知误报的老家 |
| 3 | `near` **贴着阈值** | `红 0.151 > 0.15` 这种，最可能标反 |
| 4 | `pos` / `bg` 普通正样本 / 背景帧 | 抽检用 |

### 人工量估算

* 复核页上一行 = 一张帧，**判"按没按"只看右边两张放大图**（用户原话：
  "图标没点是透明的，按下变成部分白色不透明"）；
* 熟练后 **2~5 秒/帧**；
* **300 帧 ≈ 20~30 分钟**。300 帧足够把主要错误类型纠完；
  想要更稳就做 600~1000 帧（1~1.5 小时）。

> 也可以完全跳过复核，直接 `train run` —— 但那样训出来的模型**不会比启发式好**，
> `train eval` 会直接把这话打出来（它并排显示启发式对照组）。

### 用什么工具改

* **推荐**：`train review` 生成的静态复核页（键盘 `B`/`N`/`空格`/`↑↓`，改完点
  「导出修正」下载 JSON → `train apply`）。不用起服务、不占端口。
* **查"对不对得上"**：`a9route serve` 之后打开
  **http://127.0.0.1:8790/dataset**（见 §4.1）—— 那个页面是专门查
  "标注 ↔ 标定 ↔ 判据"对应关系的。
* **任意 YOLO 标注工具**：标注文件就是标准 YOLO 格式
  （一行一个框 `cls cx cy w h`，空文件 = 没按）。
  X-AnyLabeling / labelImg 都能直接打开这个数据集。
* **手改也行**，但 ⚠️ **要改对那一份**：`train split` 之后同一帧在
  `pool/labels/` 和 `labels/{train,val}/` 下**各有一份**，而体检/训练读的是
  **`labels/{train,val}/` 那份**。改了 `pool/` 却不重跑 `train split` =
  **改动完全不生效**，而且以前没有任何提示 ✗（现在体检会报"两份标注不一致"）。

### 4.1 图形界面：数据集体检页（`/dataset`）

```powershell
& $py -m a9route serve --no-browser      # 然后打开 http://127.0.0.1:8790/dataset
& $py -m a9route train audit             # 同一份逻辑，命令行也能跑
```

页面干两件事：

1. **对应关系体检** —— 一张结论表，逐条说有没有问题；

   | 查什么 | 抓的是哪类坑 |
   |---|---|
   | 图片 ↔ 标注 ↔ `meta.jsonl` 三者是否一一对应 | 少一个标注文件会让 YOLO 把那张图当"没标"而不是"背景帧" |
   | **标注框 vs 当前标定框** | 数据集是用**另一套** `brake_key_box` 建的 —— 模型会学错位置 |
   | 标定自身的左右对称（`1033 = 1280-(137+110)`） | 框量歪了 |
   | 画面尺寸 vs 标定（标定是按 1280×720 量的） | 换分辨率后框必然不对 |
   | **pool 与 train/val 的两份标注是否一致** | 手改了 `pool/labels` 但忘了重跑 `train split`（改动静默失效） |
   | **抽样用当前判据重算** vs 存下来的信号/标注 | `config.json` 的阈值被改过 —— 标注不再代表当前判据 |

2. **逐帧对照** —— 缩略图上同时画**两套框**：

   * **实线** = 数据集里的标注框（黄 = 刹车、红 = 氮气）
   * **虚线** = **当前 `config.json`** 的标定框

   两者应当**完全重合**；不重合就是"数据集和标定不对应"。
   右边两张是按键框的放大图（判据真正读的像素，服务端裁的，
   和判据读的是同一块）。还能按 `划分 / 抽样原因 / 标注类别 / 复核状态` 过滤、
   搜文件名、以及**只看有问题的帧**。

#### 复核习惯：**翻页即复核**（不用逐个点「没问题」）

按页整理时的正常流程就是"**有错的才改，没错的直接翻过去**"，所以**离开本页时
自动结算这一页**：

| 这一页上的帧 | 动作 | 走哪个接口 |
|---|---|---|
| **你改过的** | 写回标注文件（按当前标定对齐框） | `/api/dataset/corrections` |
| **没改过的** | 只记成「已复核」 | `/api/dataset/mark_reviewed` |
| **框和标定对不上的** | **不记**（免得把问题"翻过去"） | —— |

⚠️ 两个接口**必须分开**：`corrections` 会用**当前标定**重写标注框 ——
数据集要是用另一套标定建的，翻一页就会把整页标注**悄悄挪位**，
连"数据集和标定对不上"这个体检结论都一起抹掉 ✗。
而 `mark_reviewed` **只动 `meta.jsonl` 的 `status`，一个字节的标注文件都不改**
（`test_train.py` 里有一条断言就是**逐字节比对**标注文件没变）。

* 手快翻多了：工具栏有 **「撤销上次标记」**（只认最近一次 = "上一页"的粒度）。
  重复标记同一页**不会**冲掉撤销点（`changed == 0` 时不覆盖记录）。
* 「没问题」按钮还在（给"我现在就想记下这一帧"用），
  「保存本页」可以不等翻页、手动结算当前页。
* 补一句："只看有问题的"**不把「人工修正过」算成问题** ——
  那是你复核的成果，不是错；算进去的话复核越多、这一栏越乱。

> 已经复核过的那几十帧**不受影响** —— 新机制只在你继续翻页时生效。

> **故意弄坏了也要能报出来** —— `test_train.py` 的 T13 有 5 个反例
> （改标定、改阈值、删标注、写歪框、只改 pool 不重划分），
> 每个都要求体检**必须抓住**。一个永远说 OK 的体检没有价值。

---

## 3. 环境：这台机器上是**两个** Python 环境

| | 训练环境 | 运行环境 |
|---|---|---|
| 环境 | `C:\Users\Admin\miniconda3\envs\ai_joy` | `C:\Users\Admin\miniconda3\envs\alphash9auto` |
| Python | 3.9.25 | 3.12.13 |
| 有 | torch **2.8.0+cu128**、torchvision 0.23、ultralytics 8.4.152、cv2 5.0 | cv2 4.10、numpy 2.4、flask、paddleocr |
| 干什么 | `train run` / `train export`（**只有这两个需要它**） | `analyze` / `serve` / `train build|review|eval` |
| 缺 | — | **onnxruntime**（推理要用，见下） |

**为什么分开**：运行时只要推一个 2 类小模型，装 2.5 GB 的 torch 不划算。
`train export` 导出 **ONNX**，运行时用 **onnxruntime**（约 15 MB）就够了。

### 一次性安装（都已经做过了，除非你要重装）

```powershell
# ① 运行时推理（在 alphash9auto 里，约 15 MB）
& "C:\Users\Admin\miniconda3\envs\alphash9auto\python.exe" -m pip install `
    -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime

# ② 训练环境（ai_joy 里已经有 torch cu128 了，只缺 ultralytics）
& "C:\Users\Admin\miniconda3\envs\ai_joy\python.exe" -m pip install `
    -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics
```

> ⚠️ **RTX 50 系（sm_120）必须 cu128 及以上**的 torch，否则
> `torch.cuda.is_available()` 是 False。`ai_joy` 里的
> `torch 2.8.0+cu128` 正好合适（已实测：`cap (12, 0)`、CUDA 可用）。

### 自检

```powershell
& $py -m a9route train doctor
```

它会告诉你**当前这个环境能干什么、缺什么、怎么补**，还会顺便看一眼另一个环境。

---

## 4. 磁盘

实测（三段录像 / 1280×720 / JPEG q92）：

| | 实测值 |
|---|---|
| 单帧 JPEG | **约 145 KB** |
| 1200 帧数据集 | **175 MB** |
| 3000 帧预估 | 约 440 MB |
| 训练产物 `models/runs/keys/` | 每轮几十 MB（权重 + 曲线图） |

目录约定（都不入库，但**都不是垃圾**，别当 `worktmp` 删）：

```
datasets/keys/            ← 数据集（含**人工复核过**的标注）
  pool/{images,labels}/       抽帧 + 预标注的原始池（复核对着它看）
  images/{train,val}/         划分后的图
  labels/{train,val}/         划分后的标注
  data.yaml                   ultralytics 直接吃
  meta.jsonl                  每帧一行：帧号/时刻/抽样原因/判据明细/复核状态
  review/                     复核页（page_NNN.html + index.html + queue.txt）
models/                   ← 模型与权重
  weights/yolov8n.pt           预训练权重（6.5 MB）
  keys.onnx                    **运行时加载的就是它**
  runs/keys/weights/best.pt    训练产出
```

想放到别的盘：

```powershell
$env:A9ROUTE_DATASETS = "E:\a9route-datasets"
$env:A9ROUTE_MODELS   = "E:\a9route-models"
```

---

## 5. 完整流程（可直接抄）

```powershell
cd D:\Projects\a9route
$tr = "C:\Users\Admin\miniconda3\envs\ai_joy\python.exe"        # 训练环境
$py = "C:\Users\Admin\miniconda3\envs\alphash9auto\python.exe"  # 运行环境

# 0) 自检
& $py -m a9route train doctor

# 1) 抽帧 + 启发式预标注（运行环境；不需要 torch）
#    预算按**段数均分**：--budget 3000 三段视频 -> 每段 1000 帧
& $py -m a9route train build output\*.mp4 --budget 3000

# 2) 体检 + 看统计
& $py -m a9route train stats
& $py -m a9route train verify          # 有问题会直说（类别不平衡/没有背景帧/…）

# 3) 人工复核（**这一步才是模型的增量**）
& $py -m a9route train review          # 生成 datasets/keys/review/index.html
#    浏览器打开 -> B/N/空格 改 -> 点「导出修正 JSON」
& $py -m a9route train apply "$env:USERPROFILE\Downloads\keys_corrections.json"

# 4) 按视频划分 train/val（防数据泄漏）+ 再体检
& $py -m a9route train split --val-ratio 0.25
& $py -m a9route train verify

# 5) 训练（训练环境）
& $tr -m a9route train run --epochs 100 --imgsz 640 --batch 16 --device 0

# 6) 逐帧评估（并排打启发式对照组 —— **模型没赢就别切**）
#    给 .onnx 就在**运行环境**里评估（只要 onnxruntime，不需要 torch）；
#    给 .pt 则要训练环境。评估的就是运行时真正加载的那个文件，不会评错对象。
& $py -m a9route train eval  --weights models\keys.onnx
#    （只想看 ultralytics 的 mAP：& $tr -m a9route train run 已经顺带打过了）

# 7) 导出 ONNX（训练环境；产出 models/keys.onnx）
& $tr -m a9route train export

# 8) 切运行时后端（**显式**，不会因为磁盘上多了个文件就悄悄变）
& $py -m a9route analyze output\测试3.mp4 --set vision__key_backend=onnx
#    想固化成默认：
& $py -m a9route config set vision__key_backend=onnx
```

想对比"换模型前后路线会怎么变"：

```powershell
& $py -m a9route train pulses output\刹车氮气标定测试.mp4 `
        --weights models\keys.onnx
# 会并排打印：模型 vs 启发式 的 动作数 / 按下总时长 / 脉冲数 / 差
```

### 参数在哪

* 数据集/训练参数在 `a9route/train/` 各模块里，CLI 上都有对应开关
  （`--budget`、`--edge-pad`、`--edge-share`、`--val-ratio`、`--epochs`…）；
* **运行时**的按键后端在 `config.json` 的 `vision` 段：

```powershell
& $py -m a9route config set vision__key_backend=onnx      # heuristic(默认)/auto/onnx/ultralytics
& $py -m a9route config set vision__key_conf=0.45         # 置信度：调高 = 更保守，误报更少
& $py -m a9route config set vision__key_model="D:\别的模型.onnx"
& $py -m a9route config list --changed
```

---

## 6. 怎么判断"训好了"（**别看 mAP**）

两个按键位置固定、框永远一样大 —— **mAP 天生接近满分，什么也证明不了**。
真正相关的只有一件事：**每帧"按没按"判得对不对**。

```powershell
& $py -m a9route train eval
```

打出来的是这个（`train/runner.evaluate_frames`，下面是**实测**的真输出）：

```
逐帧评估（400 帧，conf=0.4, imgsz=640）
  权重 models\keys.onnx  [onnx]
  类别                    精确率       召回       F1        误报/千帧背景
  brake_pressed       0.882    0.938    0.909          57.85
  nitro_pressed       0.939    0.821    0.876          12.40
  --- 对照：启发式 cues（和标注同口径） ---
  brake_pressed       1.000    1.000    1.000           0.00
  nitro_pressed       1.000    1.000    1.000           0.00
  --- 对照：启发式 fine（**线上真正在跑的**） ---
  brake_pressed       1.000    1.000    1.000           0.00
  nitro_pressed       0.509    1.000    0.675         223.14

  ⚠️ **这批标注还没经过人工复核**（是启发式预标注）——
     所以上面 `cues` 那行必然接近满分（标注就是它打的），
     模型**不可能**在它上面「赢」，这个对比没有信息量。
     现在能看的只有两件事：
       · 相对 `fine`（线上真正在跑的）：氮气误报 223.1 -> 12.4 / 千帧背景
       · 翻转帧正确率 0.906（96 帧）
     **要真正判断模型好坏，先把 val 那部分人工复核一遍**：
       a9route train review   →   a9route train apply <导出的json>
     然后重跑本命令 —— 那时 `cues` 那行才是公平对照。
```

**这段输出本身就是这个框架最重要的一条设计**：它不会让你把
"跟自己的标注比"误读成"模型赢了"。盯这几个数：

| 指标 | 含义 | 目标 |
|---|---|---|
| `nitro_pressed` 的 **误报/千帧背景** | **"氮气偏多"就是这个数**（拿 `fine` 那行比） | 明显低于 `fine` 行 |
| 每类 **F1**（复核过的标注下和 `cues` 比） | 综合 | ≥ `cues` |
| **翻转帧正确率** | 最难那批（单击/双击边界） | 越高越好 |
| 整片 **动作数对比** | `train pulses` 里"差"那一列 | 氮气动作数应**减少**（现在是偏多） |

### ⚠️ 为什么必须人工复核 val，否则评不出东西

预标注是**启发式打的** → 同口径的启发式在它上面必然满分（实测确实 1.000）
→ 模型"没比它好"是必然的，**不是模型不行**。

所以 `train eval` 现在会**打印一句警告**（见上），并在报告里带上复核进度
（`reviewed_frames` / `labels_are_truth`）。**最少只要复核 val 那 400 帧**
（约 20~40 分钟），`cues` 那行才会变成有意义的基准。

**没赢就别切后端** —— `key_backend` 默认是 `heuristic`，不切就不会有任何行为变化。

---

## 7. 本机的坑（都实测踩过，别再踩）

| 现象 | 真实原因 | 解法 |
|---|---|---|
| `YOLO("yolov8n.pt")` 报 `Download failure ... Curl return value 35` | ultralytics 下载权重是**shell 调 curl**，而这台机器 curl 的 schannel 坏了（`SEC_E_NO_CREDENTIALS`）。**Python 自己的 TLS 是好的** | `a9route train fetch`（用 urllib 下）；或手工放到 `models/weights/` |
| `pip download` 卡死 / `Invoke-WebRequest` 报 TLS 错 | 同上（schannel）。TCP 443 是通的 | pip 加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 训练刚开就 `PermissionError: [WinError 5]` | matplotlib 要往 `~/.matplotlib` 写字体缓存，受限环境写不进去 | **已在代码里修掉**：`runner.prepare_env()` 把 `MPLCONFIGDIR` 和 `YOLO_CONFIG_DIR` 都指到项目内（和 NOTES §9 里 `.paddlex` 那个坑同一个解法） |
| 训练走到 `Fast image access` 之后报 `PermissionError: [WinError 5]`，栈底是 `multiprocessing → connection.Pipe → _winapi.CreateFile` | ultralytics 给标注建缓存时会开 `multiprocessing.Pool`，而 Windows 上 `Pool` 靠**命名管道**通信 —— **受限沙箱禁止命名管道** | 在**普通终端**里跑训练就没这个问题（你自己的机器上不会遇到）。若必须在沙箱里跑，需要放宽该次命令的文件/进程权限 |
| 训了几轮之后 `RuntimeError: DataLoader worker (pid(s) …) exited unexpectedly` | 同一个根因：DataLoader 的 worker 进程之间**也走命名管道** | 加 **`--workers 0`**（单进程加载，小数据集几乎不影响速度）：`train run --workers 0` |
| `AMP checks skipped ... unable to download YOLO26n` | ultralytics 的 AMP 自检要下 `yolo26n.pt`，同样被 curl 挡住 | 无害（会自动继续）。`models/weights/yolo26n.pt` 放好了就没这条 |
| `torch.cuda.is_available()` 是 False | torch 是 CPU 版 / CUDA 版本低于 12.8（RTX 50 系是 sm_120） | 装 cu128 及以上的 torch |
| 在 `alphash9auto` 里跑 `train run` 报"没有 ultralytics" | 环境跑错了（这是**运行环境**） | 用 `ai_joy` 跑训练；`train doctor` 会直说 |
| 在 `ai_joy` 里跑 `analyze` 读不到路程百分比 | 那边没装 OCR —— 但**按键细扫仍然有效** | 正常现象，见 README「已知限制」 |
| **`analyze` 一张截图都没有、路线是空的**（看着像"识别不到任何操作"） | **OCR 读不到模型**：`~/.paddlex` 在**工作区外**，受限环境（沙箱/只读 HOME/服务进程）读不到 → 读不到「路程 NN%」→ 每个百分点都进不了 | `a9route ocr status` 看状态；`a9route ocr cache` **抄一份到工作区**（约 21 MB，做一次就够）。现在 `analyze` 会在这种时候**直接报错并说明原因**，CLI 和界面都会显示 |

---

## 8. 已知限制（老实说）

1. **这个模型补的是"判据"，不是"目的"。** 它只说"这一帧刹车键/氮气键按下外观"，
   "这次按下是为了漂移 / 360 / 打断氮气"仍然由 `core/intent.py` 按**信号形状**判
   （用户 2026-09-13 定的规则）。所以模型再好也不会自动修好"360 被当漂移"这类问题。
2. **框位置固定 = 位置先验会被学进去。** 预标注的框永远在同一个地方，
   模型很容易学到"这个位置 + 这种亮度 = 按下"。这**恰好是我们要的**（HUD 不动），
   但代价是：**换分辨率 / 换 UI 布局后必须重量框并重训**，
   别指望它自己迁移。
3. **只有一段视频时 val 偏乐观。** `train split` 只有在**多段**视频时才按视频分；
   单段只能按连续时间块分，`split` 会明确打印"指标偏乐观"的提醒。
4. **2 类 + ONNX 有个输出歧义。** 我们只有 2 类 → `4+nc == 6`，
   而"已做 NMS"的导出行宽**也是 6**，光看形状分不出来。`vision.key_fmt=auto`
   按"未做 NMS"解析（`train export` 导出的就是这种，我们**不传 `nms=True`**）；
   你若自己用 `nms=True` 导过，必须设 `vision.key_fmt=nms`，否则会**静默**把
   坐标当分数用（不报错，只是全错）。
5. **`heuristic` 后端其实有"两套口径"，而且默认用的是可能有问题的那套。**
   接线时发现的（**没有改它，只是把它摊开**）：

   | | 刹车 | 氮气 |
   |---|---|---|
   | `key_heuristic_mode=fine`（**默认**，`scan_fine` 一直在用） | 绝对白度 > 阈值 | **`max(红, 圈内外)` > 阈值** |
   | `key_heuristic_mode=cues`（`cues.key_pressed()`，粗扫在用） | 圈内外 ∪ 白度 | **只认红** |

   `NOTES.md` §1 实测的结论是"**圈内判据对氮气没信号**（按下 0.004），
   并进来只会把误报抬高 ✗"，`cues.py` 的注释也写着"氮气只认红 >0.15" ——
   可真正产出路线里 `N:` 条目的是 `scan_fine`，它用的是**被 NOTES 否掉的那个并集**。
   README 里「氮气偏多」那条已知问题，**至少有一部分是这儿来的**。

   → `train eval` 和 `train pulses` 都会把启发式当对照组打出来，
   而且现在你可以用一行命令验证这件事：

   ```powershell
   & $py -m a9route train pulses output\测试3.mp4   # 默认 fine 口径
   # 对比：
   & $py -m a9route analyze output\测试3.mp4 --set vision__key_heuristic_mode=cues
   ```

   **默认行为一个字节都没改**（守住了"迁移时判据逻辑逐字未改"这条）。
   要不要统一成 `cues` 口径，等你用上面的对比数据决定 —— 也可能模型直接把这个
   问题绕过去了，那就更不用改。

6. **iOS/手机端没测过。** 这条线只在 Windows + 这台机器上跑过。
