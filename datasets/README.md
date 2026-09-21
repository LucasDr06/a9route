# datasets/ —— 训练数据集（**不是缓存，别当垃圾删**）

这里放两套训练数据，**各自独立、互不共用**：

| 目录 | 任务 | 类别 | 界面 |
|---|---|---|---|
| `keys/`（默认） | 刹车 / 氮气按键 | `brake_pressed` / `nitro_pressed` | http://127.0.0.1:8790/dataset |
| `choice/` | **选路**（有没有岔路口 / 几个选项 / 选第几个） | `choice_icon` / `choice_selected` | http://127.0.0.1:8790/choice |

`choice/` 是用 `a9route train choice-build --source keys` 从 `keys/` **分类**出来的：
帧是**硬链接**（不额外占磁盘），train/val 划分**照抄源数据集**（不重新乱分）。

**体积大所以不入库**，但和 `worktmp/` 不一样：
**这里面有人工复核过的标注，是整条线上最贵的东西**。

## 长什么样

```
datasets/<名字>/                 # 默认名字 keys
├── dataset.json                 源视频、抽样参数、修正历史
├── meta.jsonl                   每帧一行：帧号/时刻/抽样原因/判据明细/复核状态
├── pool/images|labels/          抽帧 + 启发式预标注的原始池（复核对着它看）
├── images/{train,val}/          划分后的图
├── labels/{train,val}/          划分后的标注（标准 YOLO 格式）
├── data.yaml                    ultralytics 直接吃这个
└── review/                      复核页 + queue.txt
```

## 别踩的坑

* **不要手动挑帧 / 手动删"看着没用"的帧** —— `pool/` 是唯一事实来源，
  `train split` 会照着 `meta.jsonl` 重新划分；手动改会让两边对不上。
  要改标注请用 `a9route train review` + `train apply`，或直接编辑 `labels/*.txt`。
* **`labels/*.txt` 里空文件 = "这一帧两个键都没按"**，不是"这张图没标"。
  删文件会让训练报"图片没有对应标注"。
* 上一轮 `split` 出来的 `images/train|val` 会留着，`stats` 会把 pool 和 train/val
  去重统计 —— 所以**改了抽样参数后重新 `build` + `split`**，
  别新旧混着用（`build` 会重建 pool，`split` 会清掉旧的 train/val）。
* ⚠️ **同一帧有两份标注**：`pool/labels/x.txt` 和 `labels/{train,val}/x.txt`。
  **训练和体检读的是 `labels/{train,val}/` 那份** —— 只改 `pool/` 而不重跑
  `a9route train split`，改动**完全不生效**（以前没有任何提示 ✗，
  现在体检会报「两份标注不一致」）。要么改两份，要么改完 `pool` 就重跑 `split`。
* **改过 `config.json` 的按键框/阈值之后**，这份数据集的标注就不一定还成立了
  → 跑 `a9route train audit`，或开 **http://127.0.0.1:8790/dataset**
  看一眼「标注框 vs 标定框」和「现在重算 vs 存下来的信号」。

## 放到别的盘

```powershell
$env:A9ROUTE_DATASETS = "E:\a9route-datasets"
```

## 选路那套（`choice/`）多出来的两条口径

* **三个答案是从标注文件反推的**（`answers_from_boxes()`）—— 标注才是唯一真相，
  不存在"答案存了一份、标注又一份"的撕裂；
* **只有 2/3/4 个图标才算岔路口**（1 个图标是普通路标）。这个范围由
  `vision.choice_min_options` / `choice_max_options` 决定，改了界面按钮、
  后端夹取、运行时折叠**三处一起变**。

⚠️ **`/dataset`（按键那页）碰不了选路数据集**：那页只有「刹车按下 / 氮气按下」，
写回时会把**按键框**写进标注（2026-09-15 真出过，42 帧被覆盖，见 NOTES §10.25）。
后端现在会按任务拒绝并指到 `/choice`；`train review` / `train apply` 同理。

详见仓库根的 `TRAINING.md`（§0.3 是选路那条线，§4.1 是图形界面的用法）。
