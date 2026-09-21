# a9route —— 跑图视频 → 路线

把一段《狂野飙车9》的比赛录像喂进来，**每个整百分点截一张图 + 判断那一刻在做什么**，
生成一条可以直接粘贴的百分比路线（`1,31,2,N:0:2:750,4,360,…`）。

**这个项目不连手机。** 一次都不需要 adb、不需要标定、不需要设备 —— 只要录像文件。

---

## 为什么要单独做一个项目

这套东西原来长在 `asphalt9auto`（手机自动化：adb / 标定 / 手柄 / Web 控制台）里，
但视频分析这条线**跟设备没关系**。分开的直接好处是**调参不用再跑整个项目**：

| | 原来 | 现在 |
|---|---|---|
| 改一个阈值 | 改 `a9auto/race/cues.py` 里的常量 | 改 `config.json`（或 `--set`，或环境变量） |
| 验证改动 | 跑整个项目的全量回归 | `a9route test all`（约 1~2 分钟，纯离线） |

判据和算法**一行没改**（逐字搬过来的），改的只是"参数从哪来"和"谁来跑"。
搬迁过程中发现并修掉了一个真 bug，见文末「搬迁时的改动」。

---

## 快速开始

```powershell
cd D:\Projects\a9route
$py = "C:\Users\Admin\miniconda3\envs\alphash9auto\python.exe"

# ① 主命令：分析一段录像（产物在 worktmp/analysis/<视频名>/）
& $py -m a9route analyze output\跑图转路线测试.mp4

# ② 起本地 Web 窗口（拖视频进去 -> 左边路线、右边逐百分点截图）
& $py -m a9route serve           # http://127.0.0.1:8790/
                                #   http://127.0.0.1:8790/dataset  ← 按键数据集：体检 / 逐帧对照
                                #   http://127.0.0.1:8790/choice   ← **选路**数据集：三个选择题
                                # 也可以直接双击根目录的 serve.cmd（= python main.py）
                                # ⚠️ 服务就跑在**这个终端窗口的进程里**，窗口别关；Ctrl+C 停
                                # 页面右上「参数」= **可编辑的判据参数**（滑块/数字框，
                                # 保存即写进 config.json，下一次分析生效）

# ③ 只想对齐时间、自己填操作
& $py -m a9route video output\跑图转路线测试.mp4

# ④ 校验一条路线（不碰视频）
& $py -m a9route check --text "1,31,2,N:0:2:750,4,360"

# ⑤ 离线回归（不需要设备、不需要网络）
& $py -m a9route test all

# ⑥ 按键识别用训练出来的 YOLOv8 模型（**默认已开启**：模型在就用）
& $py -m a9route train doctor                    # 先看环境能干什么
& $py -m a9route train build output\*.mp4        # 抽帧 + 启发式预标注
& $py -m a9route train review                    # 人工复核（**模型的增量全在这**）
& $py -m a9route train run                       # 训练（要在有 torch 的环境里）
& $py -m a9route train eval                      # 逐帧评估（只看人工复核过的帧）
& $py -m a9route train export                    # 导出 ONNX + 写模型清单
& $py -m a9route train models                    # 看有哪些模型
& $py -m a9route train use keys                  # 换模型（会**核对类别顺序**）

# ⑦ **选路**也用 YOLO（另一套类别 / 另一个数据集 / 另一个界面）
& $py -m a9route train choice-build --source keys   # 从按键数据集分类出来（硬链接，不占空间）
& $py -m a9route train choice-stats                 # 看分布
                                #   http://127.0.0.1:8790/choice ← 标定：有没有选路 / 几个 / 第几个
& $py -m a9route train run --name choice            # 训练
& $py -m a9route train export --name choice         # -> models/choice.onnx
& $py -m a9route train use choice                   # 切运行时后端

# ⑧ OCR 模型缓存（读不到时会读不出「路程 NN%」，路线就空了）
& $py -m a9route ocr status                      # 看能不能读到
& $py -m a9route ocr cache                       # 抄一份到工作区（做一次即可）
```

> **环境**：只用 conda 环境 `alphash9auto` 里已有的那套即可
> （Python 3.12 / numpy / opencv 4.10 / flask 3.1 / paddleocr 3.4 + CUDA）。
> 依赖清单见 `pyproject.toml`；OCR 是**可选的**（起不来时会退化，见「已知限制」）。
> `python main.py` 与 `python -m a9route` 等价。
> **判定全部来自模型**：按键 `key_backend=auto` → `models/keys.onnx`，
> 选路 `choice_backend=auto` → `models/choice.onnx`。
> ⚠️ `auto` **不静默退回启发式**：模型文件不在就**报错**（这次分析无效，并告诉你
> 怎么修），因为"看起来正常、其实是像素判据"的路线比报错更坏。
> 想用启发式请**显式**设 `vision.key_backend=heuristic`（只用于调试/预标注）。
> 「路程 NN%」用 PaddleOCR；`360 / 漂移 / 双击氮气` 由**信号形状规则**推出
> （`core/intent.py`，模型给的按键信号 + 用户定的规则）。
>
> **两个训好的模型（~23 MB）就在仓库里** —— clone 下来直接能跑 `analyze`；
> 数据集（几百 MB、含人工标注）和训练产物（`models/runs/`，66 MB）不入库，
> 见 `models/README.md`。本机调好的参数可以抄 `config.example.json` 成 `config.json`。

### 录像要求

* **分辨率必须是 1280×720**。所有框都是在这个分辨率上量的；
  换了分辨率要先按下面的办法把 `config.json` 里的框重量一遍。
* 画面要**完整的比赛 HUD**（左上角有「路程 NN%」）——加载/赛道介绍那几屏本来就读不到，
  工具会如实报告"没读到百分比"，那不是错误。

---

## 参数在哪调（**这是本项目存在的意义**）

所有可调项集中在 `config.json`（没有这个文件就全用代码内置的默认值）。
优先级：**环境变量 > `config.json` > 内置默认值**。

```powershell
& $py -m a9route config list            # 看当前生效的全部参数
& $py -m a9route config list --changed  # 只看被我改过的
& $py -m a9route config set vision__nitro_red_thr=0.22      # 写入 config.json
& $py -m a9route config set scan__every=0.2 scan__workers=4
& $py -m a9route config reset vision__nitro_red_thr         # 单个键恢复默认
& $py -m a9route config reset                                # 全部恢复默认

# 只影响这一次运行（不写盘）——调参时最常用
& $py -m a9route analyze 录像.mp4 --set vision__nitro_red_thr=0.22 --verbose
# 环境变量（临时/CI 用）
$env:A9ROUTE_vision__nitro_red_thr = "0.22"
```

Web 窗口右上角「参数」按钮会把本次生效的 `config.json` + `scan/vision` 摊开给你看。

### 四段参数

| 段 | 管什么 | 里面最可能要动的 |
|---|---|---|
| `scan` | 抽样节奏与并行 | `every`（抽帧间隔）、`max_seconds`、`workers`、`with_buttons`、`hold`/`hold_brake`（分通道去抖）、`choice_min_hold`（选路稳定性闸）、`choice_cross_tol`（选路交叉校验） |
| `vision` | **判据的框和阈值** | `nitro_key_box` / `brake_key_box`（两个按键的圆）、`nitro_red_thr`、`brake_ring_thr`、`brake_white_thr`、`gauge_box`、`status_box`、`choice_*`（路标圆检测） |
| `hud` | 比赛内读数框 | `progress_box`（**路程 NN%，路线百分比只认它**）、`rank_box`、`touchdrive_box`、`choice_band` |
| `intent` | 信号形状 → 操作目的 | `tap_360_gap`（成对脉冲判 360）、`drift_min`（按住 vs 短脉冲）、`nitro_gap`、`nitro_hold` |

**调参不要凭感觉。** 每个默认值后面都写着实测数据（`config.py` 里有完整注释），例如：

* 氮气键：`红占比 > 0.15`（实测"按下 0.36 / 没按 0.09"）；
* 刹车键：`圈内白度 − 圈外参照白度 > 0.35`（实测 0.37 / 0.11）——
  半透明图标**必须用相对量**，绝对值会被赛道明暗带走；
* 这两个框是**整个圆**：`BRAKE_KEY_BOX=(137,497,110,110)`、
  `NITRO_KEY_BOX=(1033,497,110,110)`（圆心 (200,552)/(1080,552) 半径 ≈50，
  关于屏幕中线 x=640 完全对称）。

### 换分辨率时怎么重量框

1. 从录像里抽一张比赛帧（`--verbose` 跑一次，或直接看 `worktmp/analysis/*/shots/` 里的图）；
2. 把候选框画回帧上存成 PNG 看一眼（**必须画出来核对** —— 原来那两个按键框
   就是这么发现"只压到图标左上弧"的 ✗）；
3. `config set vision__nitro_key_box=x,y,w,h`（逗号分隔，会按 `list[int]` 解析）；
4. `a9route test video` 复跑 —— 里面有真帧样本回归，能立刻告诉你框对不对。

---

## 输出长什么样

`analyze` 的产物：

```
worktmp/analysis/<视频名>/
├── route.txt                  # 完整路线：第一段是扁平逗号流，后面是逐条判据注释
└── shots/
    ├── 001pct_t012.50s.png    # 每个整百分点一张（文件名带百分比与时刻）
    └── ...
```

`route.txt` 的正文就是那一行：

```
# 由跑图视频**自动推断**的路线（操作是判据猜的，请核对后使用）
# 按键细扫：42 个动作（刹车 11 / 氮气 31）
#   —— 时长与单击/双击都来自逐帧按键，不是估算 ✓
...
1,D:464,2,D:598,4,32|D:364,11,N:0:2:167,16,N:0:2:167,27,N:100:1:100,...
```

**路线格式**（与手上那份现成脚本 1:1 兼容，`a9route/core/route.py` 是唯一事实来源）：

| 写法 | 含义 |
|---|---|
| `NN`（两位数字） | 选路：前一位=图标数量、后一位=选第几个（从左数，1 起） |
| `N:延时:次数:间隔` | 点氮气。`N:0:2:750` 完美氮气；`N:500:10:100` 快速连点 |
| `D:毫秒` | 漂移 |
| `S:毫秒` | 关自动驾驶（之后自动打开，不建议用） |
| `360` | 360 操作 |
| `A\|B` | 同时执行：`40,21\|D:2050` |

解析器会主动挑错：中文标点（`，：｜`）、缺逗号、选路越界、氮气 0 次、
连点却把间隔写成 0、漂移 0ms、百分比越界，都带行号报错；
警告只留两条：同百分比出现多次、**氮气落在漂移窗口内**（点氮气会取消漂移）。

---

## 三条判据线（算法骨架）

1. **粗扫**（`vision/video.py` 的 `extract_per_percent`）：按 `scan.every`（默认 0.25s）
   抽帧，逐帧算出「在做什么」+ 读「路程 NN%」→ 每个整百分点**存一张 PNG**，
   并把该百分点附近的判据汇总成一个操作（单帧会闪，窗口汇总才稳）。
2. **精细扫描**（`scan_fine`，和粗扫**并行**，各自一路解码）：
   逐帧只看两个按键（刹车/氮气）+ 每 2 帧看一次选路路标，**只产时间戳**。
3. **判目的**（`core/intent.py`）：由信号的**形状**判定 ——
   刹车**成对脉冲**（间隔 <`tap_360_gap`，默认 300ms）= **360**；
   刹车**独立短脉冲** = **打断氮气**（很短的 `D:`）；刹车**按住** ≥`drift_min` = **漂移**；
   氮气**多次快速点击** = 每百分比最多取两次 + **实测间隔**（`N:0:2:167` 里的 167 是逐帧量出来的）；
   选路：**一直不变取第一次识别到**、**变了取第一次改变时**。

   最后用「时间 → 百分比」的插值关系把时间戳落回百分比（`attach_events`）。

> 为什么第 2 步不直接产百分比：路线脚本的精细度只有 **100 次**，
> 同一段里可能连做好几个动作，光有"这个百分点按键亮着"分不出**这次按键是为了什么**。
> 这是用户 2026-09-13 明确给的规则。

三条判据各自的**实测数据、走过的死路、为什么不用模板匹配**：见 `NOTES.md`。

---

## 目录结构

```
a9route/
├── main.py                   # 根入口（= python -m a9route）
├── config.json               # （可选）所有可调参数；不存在就用内置默认
├── TRAINING.md               # **按键 YOLOv8 模型：要准备什么、怎么训、怎么切**
├── routes/                   # 路线样例：demo.txt（格式说明）+ 用户真实路线（47 条）
├── datasets/                 # （不入库）训练数据集：帧 + 人工复核过的标注
├── models/                   # （不入库）预训练权重 / keys.onnx / 训练产物
├── a9route/
│   ├── cli.py                # 所有子命令
│   ├── config.py             # **可调参数的唯一事实来源**（含实测注释）
│   ├── analysis.py           # 「视频 → 路线」的编排（CLI 与 Web 共用这一份）
│   ├── paths.py              # 路径常量（不相对 CWD）
│   ├── bootstrap.py          # Windows 中文控制台的 UTF-8 修复
│   ├── core/
│   │   ├── route.py          # 路线脚本格式：解析 / 校验 / 扁平逗号流
│   │   └── intent.py         # 信号形状 → 操作目的
│   ├── vision/
│   │   ├── video.py          # 抽帧、时间轴、逐百分点截图、精细扫描、生成路线
│   │   ├── cues.py           # 一帧 → "现在在做什么"（按键/路标/状态文字）
│   │   ├── keys.py           #   **通用 ONNX 检测器** + 按键的三个后端：heuristic / onnx / ultralytics
│   │   ├── choice.py         #   **选路识别**：图标 → (有没有岔路 / 几个 / 选第几个)
│   │   └── hud.py            # 一帧 → 比赛读数（路程 NN% / 排名 / 路标）
│   ├── train/                # **YOLOv8 训练框架**（只有它 import torch）
│   │   ├── labels.py         #   类别定义 + YOLO 标注读写 + data.yaml
│   │   ├── choice.py         #   **选路这条线**：三个答案 ↔ 标注框、数据集分类、统计
│   │   ├── frames.py         #   抽哪些帧（优先级采样：翻转帧/吵架/贴阈值/…）
│   │   ├── prelabel.py       #   启发式预标注（和运行时同一套判据）
│   │   ├── dataset.py        #   数据集构建 / **按视频划分** / 校验 / 统计
│   │   ├── audit.py          #   **数据集体检**：标注 ↔ 标定 ↔ 判据 对不对得上
│   │   ├── modelcard.py      #   **模型清单**（类序/imgsz/指标）+ 换模型的类序校验
│   │   ├── review.py         #   复核页 + 导出修正 + 写回标注
│   │   ├── runner.py         #   训练 / 逐帧评估 / 导出 ONNX / 整片脉冲对比
│   │   └── doctor.py         #   环境与素材自检
│   ├── ocr/reader.py         # PaddleOCR 懒加载 + **模型缓存的自动镜像**（见 NOTES §10.17）
│   ├── web/                  # 本地窗口（Flask + 原生 JS，无构建步骤）
│   │   ├── app.py            #   「跑图视频 → 路线」页 + `/api/*`
│   │   ├── dataset.py        #   「数据集体检」页（`/dataset` + `/api/dataset/*`）
│   │   └── choice.py         #   **「选路标定」页**（`/choice` + `/api/choice/*`，和上面那页分开）
│   └── tests/                # 离线回归 + fixtures（真帧样本）
└── NOTES.md                  # 踩坑记录：判据怎么量出来的、哪些路走不通
```

---

## 测试

```powershell
& $py -m a9route test all      # route / intent / video / web / train 五个模块
& $py -m a9route test video    # 单独跑一个
```

**不需要设备、不需要网络**，跑一两分钟。五块各锁什么：

| 模块 | 锁什么 |
|---|---|
| `test_route` | 路线格式的逐条语义（含用户真实 47 条路线原样可解析） |
| `test_intent` | 信号形状 → 目的（360 / 漂移 / 打断氮气 / 单击双击长按 / 选路取第一次改变） |
| `test_video` | 时间轴、抽帧真解码、窗口汇总、按键段、**真帧样本回归**、端到端一条龙、配置生效 |
| `test_webvideo` | Web 接口：上传、轮询、截图防目录穿越、路线下载、OCR 不可用时不崩 |
| `test_train` | **训练框架（不需要 torch）**：YOLO 标注读写、抽帧优先级与预算、预标注、 数据集构建/划分/校验、**数据集体检（含 5 个"故意弄坏"的反例）**、 复核写回、letterbox 几何、ONNX 输出解码、后端选择、体检页的接口、 **选路那条线（三个答案 ↔ 标注框往返、数据集分类、模型清单认任务）** |

真帧样本在 `a9route/tests/fixtures/`（3 张选路路标帧 + 刹车/氮气按键截图），
都是当年在真实比赛画面里裁下来的，所以能抓住"框改歪了""阈值调坏了"这类回归。

---

## 已知限制（老实说）

**判据是启发式的** —— 视频里看不到按键真值，只能从画面推断，误报难免。
所以定位是「**自动生成草稿 + 人工核对**」，Web 窗口会把每个百分点的判据一并摊开。

两条已知不准的（原项目交接文档 §6.1 记的，**本轮按要求没有动判据**）：

1. **氮气偏多**：现在 `红 > 0.15` 且允许单帧，整段视频里会产出偏多的单发氮气。
   下一步建议：改用**相对基线**（该框自己最近 2~3 秒的中位数），或要求"成对/成段"。

   > **2026-09-14 补**：接训练框架时发现了一个**具体的**来源 ——
   > 项目里其实有**两套启发式口径**，而且真正产出路线里 `N:` 条目的
   > `scan_fine()` 用的是**被 NOTES §1 否掉的那套**：
   >
   > | | 刹车 | 氮气 |
   > |---|---|---|
   > | `scan_fine()`（**默认，产出路线**） | 绝对白度 > 阈值 | **`max(红, 圈内外)` > 阈值** |
   > | `cues.key_pressed()`（粗扫在用） | 圈内外 ∪ 白度 | **只认红** |
   >
   > NOTES §1 实测的结论是"**圈内判据对氮气没信号**（测试3 按下 0.004），
   > 并进来只会把误报抬高 ✗"，`cues.py` 的注释也写着"氮气只认红 >0.15" ——
   > 所以"氮气偏多"至少有一部分是**把没信号的通道并了进来**。
   > **没有改默认行为**（迁移时"判据逐字未改"这条守住了），
   > 但现在是**可测的**：`a9route train pulses <录像>` 会把启发式当对照组打出来，
   > 也可以 `--set vision__key_heuristic_mode=cues` 直接对比两套口径。
   > 见 `TRAINING.md` §8.5。
2. **选路条目比实际多 1 条**：用户说真实只有 5 个岔路口，输出 6 条，
   疑似开头 `4,21` + `5,22` 是同一个岔路口被读成两个值。
   建议：把"1 个百分点内的改变"合并成一条（取最后一次），或把稳定性闸从 4 帧提到 6 帧。

其它：

* **OCR 起不来会退化**：读不到「路程 NN%」时，`shots` 会是空的、路线正文为空 ——
  这时工具会明确报告"没扫到百分比"，**不会编一个数**。
  精细扫描那一半（按键）仍然有效（实测在真录像上 1529 帧 → 42 个按键动作）。

  > ⚠️ **2026-09-15 补**：这条"退化"曾经**看起来像 bug** ——
  > 用户报"识别路线识别不到任何操作"，查下去发现真凶是
  > **`~/.paddlex` 在工作区外、受限环境读不到 OCR 模型** →
  > 每个百分点都进不了 → 路线必然是空的，**跟检测毫无关系** ✗✗。
  >
  > 现在做了两件事：① `a9route ocr cache` 把模型**抄一份到工作区**（约 21 MB，
  > 自动镜像，做一次即可）；② 一张图都没扫到时 `analyze` 会**当错误报出来**，
  > 附上具体原因和能照做的命令（CLI 退出码非 0、界面弹红框），
  > 而不是给一条空路线让人以为"这视频没操作"。见 NOTES §10.17。
* **模板匹配已经被判定走不通**（图标是半透明叠加，模板主要在匹配背景），别再去调模板。
* 只在 **1280×720** 的录像上验证过。

---

## 搬迁时的改动（相对原项目）

**逻辑逐字未改**，改的只有三件事：

1. **包名与目录**：`a9auto/race/*` → `a9route/{core,vision}/`；
   `progress.py` → `vision/hud.py`（内容一样，只删掉"去 adb 截屏"那条路径
   —— 本项目只处理视频帧，忘了传帧会直接报错，而不是偷偷去连手机）。
2. **参数从代码常量改成配置**：`cues.py` / `hud.py` / `intent.py` 里的阈值改为
   从 `config.py` 灌入（`config.apply()`），**默认值与原来完全一致**。
3. **Web 端重写**：原控制台（任务编排 + 日志 + 设备 + 跑图窗口）里只留跑图窗口，
   重写成单页；四个接口从 `/api/race/video/*` 改名为 `/api/videos`、`/api/analyze`、
   `/api/status`、`/api/route` + `/media/<job>/<name>`。

**顺手修掉的真 bug**：精细扫描那段是 `except Exception: 打一行日志` 包起来的，
搬迁后一个 import 路径写错（`from a9route.vision import intent`，应是 `a9route.core`）
**被静默吞掉**，表现成"按键细扫没有结果"（看着像阈值问题）。
现在 `test_video` 里有一条专门盯"精细扫描失败"这行日志的回归。

想跟原项目对照的话：判据函数、`intent` 规则、路线格式都是逐字等价的，
`git diff` 一下对应文件就能确认。
