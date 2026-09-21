# -*- coding: utf-8 -*-
"""train.review —— **人工复核**：把"最可能标错的帧"摆到人眼前，一键改掉。

这是整个训练框架里**唯一能让模型超过启发式**的环节。
预标注只是把启发式的判断抄成标注；如果直接拿去训练，模型只会学会复现
同一套阈值（见 `train/prelabel.py` 的说明）。真正的增量全在这儿。

## 复核页排在最前面的是哪些帧

按 `(还没复核, reason 优先级, 离阈值有多近)` 排 —— 也就是：

1. **翻转帧**（`edge`）：按键按下/松开的那几帧，单击/双击/长按的分界；
2. **两通道吵架**（`disagree`）：例如"圈内白度说按了、红说没按" ——
   "瓶子充满变红但其实没按"这类已知误报的老家；
3. **贴着阈值**（`near`）：`红 0.151 > 0.15` 这种，最可能标反；
4. 剩下的正样本/背景帧（抽检用）。

## 怎么用（不重建一个标注 GUI）

* `review` 生成 `review/page_NNN.html` + `index.html`。每行一张帧：
  左边整帧（画好框）、右边两个按键的放大图 —— **判"按没按"就看右边那两张**。
* 键盘 `B` / `N` 切换"刹车按下 / 氮气按下"，`空格` 标记"看过了（没问题）"，
  `↑/↓` 上下帧。改完点「导出修正」下载一个 JSON。
* 把 JSON 交给 `a9route train apply <json>` 写回标注 —— 改的是
  `labels/*.txt`（YOLO 原始格式），所以**别的工具也能接着改**
  （X-AnyLabeling、labelImg 都直接认这个数据集）。
* 想直接手改也行：标注文件就在 `pool/labels/*.txt`，一行一个框，
  `0 cx cy w h`，**空文件 = 没按**。

> 为什么不做成 Web 应用里那套：项目里已经有一个 Flask 窗口，但那是**看路线**的；
> 复核标注这件事用**静态页面 + 导出 JSON** 更省事 —— 不用起服务、不占端口、
> 也不和 `a9route serve` 抢 8790。

## ⚠️ 页面里的图片是 base64 内嵌的

一页太大（几百帧 × 3 张图）浏览器会卡，所以默认每页 120 帧。
`--page-size` 可调；帧特别多时多生成几页，比一页塞一万行好。
"""
from __future__ import annotations

import base64
import html
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from a9route.train import frames as F
from a9route.train.dataset import (FRAME_META, POOL_DIR, dataset_dir,
                                   load_meta, read_frames_meta, save_meta)
from a9route.train.labels import CLASSES, CLASS_IDS, Box, write_label

REVIEW_DIR = "review"
#: 一页最多几行（每行 3 张图 base64 嵌进 HTML，太多浏览器会卡）
DEFAULT_PAGE_SIZE = 120

#: 页面上 reason -> CSS class（配色见 _HTML_HEAD）
_REASON_CLS = {"edge": "e", "disagree": "d", "near": "n", "pos": "p", "bg": "g"}
_REASON_ZH = {"edge": "翻转帧", "disagree": "两通道吵架", "near": "贴着阈值",
              "pos": "普通正样本", "bg": "背景"}


@dataclass
class ReviewItem:
    """复核页上的一行。"""

    file: str
    reason: str
    t: float
    frame: int
    video: str
    boxes: list = field(default_factory=list)      # 当前标注里的类别名
    signals: dict = field(default_factory=dict)
    status: str = "prelabel"

    @property
    def suspicion(self) -> float:
        """有多可疑（**越小越该先看**）：离判定阈值最近的那个通道差多远。

        `margin` 是"得分 / 阈值"，1.0 正好是判定边界 —— 所以 `|margin-1|`
        越小越可疑。两个通道取更可疑的那个。
        """
        b = abs(float(self.signals.get("brake_margin", 0.0)) - 1.0)
        n = abs(float(self.signals.get("nitro_margin", 0.0)) - 1.0)
        return min(b, n)


def build_queue(metas: list[dict], *, include_reviewed: bool = False) -> list[ReviewItem]:
    """把 `meta.jsonl` 变成**按"该先看谁"排好序**的队列。"""
    out: list[ReviewItem] = []
    for m in metas:
        if not m.get("file"):
            continue
        status = m.get("status") or "prelabel"
        if status == "reviewed" and not include_reviewed:
            continue
        out.append(ReviewItem(
            file=m["file"], reason=m.get("reason") or "bg", t=float(m.get("t") or 0.0),
            frame=int(m.get("frame") or 0), video=m.get("video") or "?",
            boxes=[b.get("name") for b in (m.get("boxes") or [])],
            signals=m.get("signals") or {}, status=status))
    out.sort(key=lambda it: (F.REASON_ORDER.get(it.reason, 9), it.suspicion, it.file))
    return out


# ---------------------------------------------------------------- 小工具
def _b64_jpeg(img, quality: int = 82) -> str:
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _draw(frame, boxes, brake_box, nitro_box):
    """在整帧上画**当前标注**（核对框对不对用；改框不在这里做）。"""
    import cv2

    out = frame.copy()
    for name, col, box in (("brake_pressed", (0, 220, 255), brake_box),
                           ("nitro_pressed", (60, 60, 245), nitro_box)):
        if name not in boxes:
            continue
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 3)
    return out


def _crop(frame, box, zoom: int = 2):
    """按键框 + 一点留白，放大 `zoom` 倍（**用最近邻**，别模糊掉"透明/不透明"的差别）。"""
    import cv2
    x, y, w, h = [int(v) for v in box]
    pad = 8
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return patch
    return cv2.resize(patch, (patch.shape[1] * zoom, patch.shape[0] * zoom),
                      interpolation=cv2.INTER_NEAREST)


_HTML_HEAD = """<!doctype html><meta charset="utf-8">
<title>a9route 按键标注复核</title>
<style>
 body{background:#14161a;color:#e6e6e6;font:14px/1.5 "Segoe UI",system-ui,sans-serif;margin:0;padding:16px}
 h1{font-size:18px;margin:0 0 4px} .sub{color:#9aa4b2;margin-bottom:14px}
 .bar{position:sticky;top:0;background:#14161af2;padding:8px 0;border-bottom:1px solid #2a2f38;z-index:9}
 button{background:#2a2f38;color:#e6e6e6;border:1px solid #3a414d;border-radius:6px;
        padding:6px 12px;cursor:pointer;margin-right:8px;font-size:13px}
 button:hover{background:#3a414d}
 .row{display:flex;gap:14px;align-items:center;border-bottom:1px solid #232830;padding:10px 0;outline:none}
 .row:focus{background:#1b1f26;border-radius:6px}
 .row.done{opacity:.45}
 .meta{width:225px;font-size:12px;color:#9aa4b2;flex:none}
 .meta b{color:#e6e6e6;font-weight:600}
 .full img{border-radius:4px;display:block}
 .key{text-align:center;flex:none}
 .key img{border:2px solid #333;border-radius:6px;display:block}
 .key.on img{border-color:#4ade80}
 .key .lab{font-size:12px;margin-top:4px}
 .tag{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;margin-right:4px}
 .e{background:#7c2d12;color:#fdba74} .d{background:#7f1d1d;color:#fca5a5}
 .n{background:#78350f;color:#fde68a} .p{background:#14532d;color:#86efac}
 .g{background:#1e293b;color:#94a3b8}
 code{background:#1e2228;padding:1px 5px;border-radius:4px}
 ul{color:#9aa4b2}
 a{text-decoration:none}
</style>
"""

# `INIT` 是这一页每帧的**当前标注**（Python 注入），`ST` 只存人动过的那些 ——
# 于是「导出修正」导出的**是人的判断**，不是把机器的标注再抄一遍。
_HTML_JS = """
var INIT = __INIT__;
var DSNAME = __DNAME__;
var ST = {};
var CLS = ['brake_pressed', 'nitro_pressed'];

function rowEl(file) {
  var rows = document.querySelectorAll('.row');
  for (var i = 0; i < rows.length; i++) {
    if (rows[i].dataset.file === file) return rows[i];
  }
  return null;
}
function stateOf(file) {
  if (!ST[file]) ST[file] = {boxes: (INIT[file] || []).slice(), status: 'reviewed'};
  return ST[file];
}
function paint(file) {
  var cur = stateOf(file), row = rowEl(file);
  if (!row) return;
  CLS.forEach(function (cls) {
    var k = row.querySelector('.key[data-cls="' + cls + '"]');
    if (k) k.classList.toggle('on', cur.boxes.indexOf(cls) >= 0);
  });
  var s = row.querySelector('.state');
  if (s) s.textContent = cur.boxes.length ? cur.boxes.join(' + ') : '（未按）';
  row.classList.add('done');
}
function toggle(file, cls) {
  var cur = stateOf(file), i = cur.boxes.indexOf(cls);
  if (i >= 0) cur.boxes.splice(i, 1); else cur.boxes.push(cls);
  paint(file);
  var r = rowEl(file); if (r) r.focus();
}
function ok(file) {
  var cur = stateOf(file);
  cur.boxes = (INIT[file] || []).slice();
  paint(file);
  var r = rowEl(file); if (r) r.focus();
}
function exportJson() {
  var items = Object.keys(ST).map(function (f) {
    return {file: f, boxes: ST[f].boxes.slice(), status: 'reviewed'};
  });
  var payload = {version: 1, dataset: DSNAME, exported: new Date().toISOString(),
                 items: items};
  var blob = new Blob([JSON.stringify(payload, null, 1)], {type: 'application/json'});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = DSNAME + '_corrections.json';
  a.click();
}
document.addEventListener('keydown', function (e) {
  var f = document.body.dataset.cursor;
  if (!f) return;
  if (e.key === 'b' || e.key === 'B') { e.preventDefault(); toggle(f, 'brake_pressed'); }
  else if (e.key === 'n' || e.key === 'N') { e.preventDefault(); toggle(f, 'nitro_pressed'); }
  else if (e.key === ' ') { e.preventDefault(); ok(f); }
  else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    var rows = Array.prototype.slice.call(document.querySelectorAll('.row'));
    var i = -1;
    for (var k = 0; k < rows.length; k++) if (rows[k].dataset.file === f) i = k;
    var j = Math.min(rows.length - 1, Math.max(0, i + (e.key === 'ArrowDown' ? 1 : -1)));
    if (rows[j]) {
      e.preventDefault();
      rows[j].focus();
      rows[j].scrollIntoView({block: 'center'});
    }
  }
});
document.addEventListener('DOMContentLoaded', function () {
  var rows = Array.prototype.slice.call(document.querySelectorAll('.row'));
  rows.forEach(function (r) {
    r.tabIndex = 0;
    r.addEventListener('focus', function () { document.body.dataset.cursor = r.dataset.file; });
    r.addEventListener('click', function () { r.focus(); });
  });
  if (rows.length) rows[0].focus();
});
"""

_ROW_TMPL = """<div class="row" data-file="{file}">
  <div class="meta">
    <div><b>{file}</b></div>
    <div>{video} @ {t:.2f}s（帧 {frame}）</div>
    <div><span class="tag {rcls}">{rzh}</span>{done}</div>
    <div>氮气红 {red:.3f} &nbsp;判定线 1.000 &rarr; 余量 {nm:.2f}</div>
    <div>刹车圈内外 {ring:.3f} &rarr; 余量 {bm:.2f}</div>
    <div>圈内白 {bw:.3f} / 氮气白 {nw:.3f}</div>
    <div>现在标成：<span class="state">{cur}</span></div>
    <div style="margin-top:6px">
      <button onclick="toggle('{file}','brake_pressed')">刹车 B</button>
      <button onclick="toggle('{file}','nitro_pressed')">氮气 N</button>
      <button onclick="ok('{file}')">没问题</button>
    </div>
  </div>
  <div class="full"><img src="{full}" width="360"></div>
  <div class="key{onb}" data-cls="brake_pressed">
    <img src="{cb}" width="200"><div class="lab">刹车键（放大 2×）</div></div>
  <div class="key{onn}" data-cls="nitro_pressed">
    <img src="{cn}" width="200"><div class="lab">氮气键（放大 2×）</div></div>
</div>
"""


def _page_html(items: list[ReviewItem], *, name: str, page: int, pages: int,
               brake_box, nitro_box, pool: Path, quality: int = 80) -> str:
    import cv2

    rows = []
    init: dict = {}
    for it in items:
        frame = cv2.imread(str(pool / "images" / it.file))
        if frame is None:
            continue
        init[it.file] = list(it.boxes)
        # 整帧：**先在原图上画框、再缩小**（拿缩小图去画会偏）
        marked = _draw(frame, it.boxes, brake_box, nitro_box)
        scale = 360.0 / marked.shape[1]
        small = cv2.resize(marked, (360, max(1, int(marked.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
        sig = it.signals
        cur = " + ".join(it.boxes) if it.boxes else "（未按）"
        rows.append(_ROW_TMPL.format(
            file=html.escape(it.file), video=html.escape(it.video), t=it.t,
            frame=it.frame, rcls=_REASON_CLS.get(it.reason, "g"),
            rzh=_REASON_ZH.get(it.reason, it.reason),
            done="<span class='tag p'>已复核</span>" if it.status == "reviewed" else "",
            cur=html.escape(cur),
            red=float(sig.get("nitro_red", 0.0)), nm=float(sig.get("nitro_margin", 0.0)),
            ring=float(sig.get("brake_ring", 0.0)), bm=float(sig.get("brake_margin", 0.0)),
            bw=float(sig.get("brake_white", 0.0)), nw=float(sig.get("nitro_white", 0.0)),
            full=_b64_jpeg(small, quality),
            cb=_b64_jpeg(_crop(frame, brake_box), quality),
            cn=_b64_jpeg(_crop(frame, nitro_box), quality),
            onb=" on" if "brake_pressed" in it.boxes else "",
            onn=" on" if "nitro_pressed" in it.boxes else ""))
    js = (_HTML_JS.replace("__INIT__", json.dumps(init, ensure_ascii=False))
          .replace("__DNAME__", json.dumps(name, ensure_ascii=False)))
    nav = []
    if page > 1:
        nav.append('<a href="page_{0:03d}.html"><button>&larr; 上一页</button></a>'
                   .format(page - 1))
    if page < pages:
        nav.append('<a href="page_{0:03d}.html"><button>下一页 &rarr;</button></a>'
                   .format(page + 1))
    head = ('<h1>按键标注复核 &middot; {0}</h1>'
            '<div class="sub">第 {1}/{2} 页，本页 {3} 帧 &nbsp;|&nbsp; '
            '判"按没按"<b>只看右边两张放大图</b>（用户原话：'
            '"图标没点是透明的，按下变成部分白色不透明"）&nbsp;|&nbsp; '
            '<code>B</code> 刹车 <code>N</code> 氮气 <code>空格</code> 没问题 '
            '<code>&uarr;&darr;</code> 上下帧 &nbsp;|&nbsp; 改完点「导出修正」</div>'
            ).format(html.escape(name), page, pages, len(rows))
    bar = ('<div class="bar"><button onclick="exportJson()">导出修正 JSON</button>'
           + " ".join(nav)
           + '<a href="index.html"><button>目录</button></a></div>')
    return _HTML_HEAD + head + bar + "".join(rows) + "<script>{0}</script>".format(js)


def _index_html(name: str, items: list[ReviewItem], by_reason: dict, pages: int,
                reviewed: int) -> str:
    lis = []
    for r in F.REASONS:
        if not by_reason.get(r):
            continue
        lis.append('<li><span class="tag {0}">{1}</span> {2}：<b>{3}</b> 帧</li>'.format(
            _REASON_CLS.get(r, "g"), r, html.escape(_REASON_ZH.get(r, r)),
            by_reason[r]))
    links = "".join('<a href="page_{0:03d}.html"><button>第 {0} 页</button></a>'.format(p)
                    for p in range(1, pages + 1))
    return (_HTML_HEAD
            + "<h1>按键标注复核 &middot; {0}</h1>".format(html.escape(name))
            + '<div class="sub">待复核 {0} 帧 / 共 {1} 页；已复核 {2} 帧</div>'.format(
                len(items), pages, reviewed)
            + "<ul>{0}</ul>".format("".join(lis))
            + "<div>{0}</div>".format(links)
            + '<div class="sub" style="margin-top:18px">'
              "改完在页面里点「导出修正 JSON」，然后写回标注：<br>"
              "<code>a9route train apply 下载的_corrections.json --name {0}</code><br>"
              "也可以直接用别的工具改 <code>{0}/pool/labels/*.txt</code>"
              "（标准 YOLO 格式，X-AnyLabeling / labelImg 都认）。</div>".format(
                  html.escape(name)))


# ---------------------------------------------------------------- 入口
def make_review(name: str = "keys", *, page_size: int = DEFAULT_PAGE_SIZE,
                limit: int | None = None, quality: int = 80,
                include_reviewed: bool = False, progress=print) -> dict:
    """生成复核页（`datasets/<name>/review/`）。"""
    from a9route import config as cfgmod
    from a9route.vision import cues

    cfgmod.apply()
    ds = dataset_dir(name)
    # 这个页面只有「刹车按下 / 氮气按下」两个按钮，**只对按键数据集有意义**
    _require_keys_dataset(ds, name, "生成复核页")
    pool = ds / POOL_DIR
    if not (pool / "images").is_dir():
        raise RuntimeError("{0}/images 不存在 —— 先 `a9route train build`".format(pool))
    metas = read_frames_meta(ds)
    if not metas:
        raise RuntimeError("{0} 里没有 {1}".format(ds, FRAME_META))
    items = build_queue(metas, include_reviewed=include_reviewed)
    total = len(items)
    if limit:
        items = items[:int(limit)]
    out = ds / REVIEW_DIR
    out.mkdir(parents=True, exist_ok=True)
    pages = max(1, (len(items) + max(1, page_size) - 1) // max(1, page_size))
    for p in range(pages):
        chunk = items[p * page_size:(p + 1) * page_size]
        if not chunk:
            continue
        (out / "page_{0:03d}.html".format(p + 1)).write_text(
            _page_html(chunk, name=name, page=p + 1, pages=pages,
                       brake_box=cues.BRAKE_KEY_BOX, nitro_box=cues.NITRO_KEY_BOX,
                       pool=pool, quality=quality), encoding="utf-8")
        if progress:
            progress("    复核页 {0}/{1}（{2} 帧）".format(p + 1, pages, len(chunk)))
    by_reason: dict = {}
    for it in items:
        by_reason[it.reason] = by_reason.get(it.reason, 0) + 1
    reviewed = sum(1 for m in metas if m.get("status") == "reviewed")
    (out / "index.html").write_text(
        _index_html(name, items, by_reason, pages, reviewed), encoding="utf-8")
    (out / "queue.txt").write_text(
        "# 复核顺序（先看上面的）\n"
        "# file\treason\tvideo\tt\t现在标成\t可疑度\n"
        + "".join("{0}\t{1}\t{2}\t{3:.2f}\t{4}\t{5:.3f}\n".format(
            it.file, it.reason, it.video, it.t, ",".join(it.boxes) or "-", it.suspicion)
            for it in items), encoding="utf-8")
    info = load_meta(ds)
    info["review"] = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "queue": total,
                      "pages": pages, "page_size": page_size,
                      "reviewed": reviewed}
    save_meta(ds, info)
    return {"dir": str(out), "index": str(out / "index.html"), "pages": pages,
            "queued": total, "by_reason": by_reason, "reviewed": reviewed}


def _require_keys_dataset(ds, name: str, what: str) -> None:
    """这个函数只会说**刹车/氮气**的类别语言，所以不许用在别的任务的数据集上。

    ⚠️ 这条是**真踩出来的事故**（2026-09-15 17:30）：用户在 `/dataset` 页面上
    把数据集下拉框切到 `choice`（选路），然后点了「本页没问题」和「写回修正」——
    于是 **42 帧选路标注被按键格式覆盖**：
    `labels/train/xxxx.txt` 里写进了 `[137,497,110,110]` 这种**刹车框**，
    meta 里却还留着选路那三个答案。后果是训练集里混进了一批"路标在左下角"的假框，
    而界面上看起来一切正常 ✗✗（体检是靠"框不在选路带里"才抓出来的）。

    所以：**数据集的任务和这里说的类别不一致 -> 直接拒绝，并告诉他去哪一页**。
    """
    from a9route.train.audit import _task_of
    task = _task_of(Path(ds), name)[0]
    if task != "keys":
        raise ValueError(
            "数据集 `{0}` 是**{1}**数据集，不是刹车/氮气 —— 「{2}」只认刹车/氮气的"
            "类别（`brake_pressed` / `nitro_pressed`），对它操作会把数据写坏。"
            "选路请用 http://127.0.0.1:8790/choice （三个选择题那个页面）"
            .format(name, "选路" if task == "choice" else task, what))


def apply_corrections(payload: dict | str | Path, *, name: str = "keys",
                      progress=print) -> dict:
    """把复核页导出的 JSON 写回标注。

    改 `pool/labels/*.txt`（以及已有的 `labels/{train,val}/*.txt`），
    并把 `meta.jsonl` 里这些帧标成 `reviewed`（复核过没改的也算 ——
    否则下次生成复核页又会把它们排在前面）。
    **没点到的帧保持原样**，所以可以分几次、分几页慢慢改。

    ⚠️ **只认刹车/氮气数据集**（见 `_require_keys_dataset`）。
    """
    from a9route import config as cfgmod
    from a9route.vision import cues

    cfgmod.apply()
    if isinstance(payload, (str, Path)):
        p = Path(payload)
        if not p.is_file():
            raise FileNotFoundError("找不到修正文件：{0}".format(p))
        payload = json.loads(p.read_text(encoding="utf-8"))
    items = (payload or {}).get("items") or []
    if not items:
        raise ValueError("修正文件里没有 items"
                         "（要用复核页的「导出修正 JSON」按钮生成的那种）")
    ds = dataset_dir(name)
    _require_keys_dataset(ds, name, "写回复核修正")
    pool = ds / POOL_DIR
    metas = read_frames_meta(ds)
    by_file = {m.get("file"): m for m in metas if m.get("file")}
    box_of = {"brake_pressed": cues.BRAKE_KEY_BOX, "nitro_pressed": cues.NITRO_KEY_BOX}
    changed = reviewed = skipped = 0
    for it in items:
        fn = it.get("file")
        m = by_file.get(fn)
        if not m:
            skipped += 1
            continue
        want = [c for c in (it.get("boxes") or []) if c in CLASS_IDS]
        before = [b.get("name") for b in (m.get("boxes") or [])]
        w = int(m.get("width") or 1280)
        h = int(m.get("height") or 720)
        boxes = [Box(CLASS_IDS[c], *[float(v) for v in box_of[c]], conf=None)
                 for c in want]
        # pool 里那份一定要改（那是"唯一事实来源"）；
        # train/val 里已经有副本的话也一起改，免得两边不一致
        targets = [pool / "labels" / "{0}.txt".format(Path(fn).stem)]
        for sp in ("train", "val"):
            t = ds / "labels" / sp / "{0}.txt".format(Path(fn).stem)
            if t.is_file():
                targets.append(t)
        for t in targets:
            t.parent.mkdir(parents=True, exist_ok=True)
            write_label(t, boxes, w, h)
        m["boxes"] = [{"cls": CLASS_IDS[c], "name": c,
                       "xywh": [float(v) for v in box_of[c]], "conf": None}
                      for c in want]
        m["status"] = "reviewed"
        reviewed += 1
        if sorted(before) != sorted(want):
            changed += 1
            if progress:
                progress("    {0}: {1} -> {2}".format(
                    fn, ",".join(before) or "（未按）", ",".join(want) or "（未按）"))
    (ds / FRAME_META).write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metas),
        encoding="utf-8")
    info = load_meta(ds)
    hist = info.get("corrections") or []
    hist.append({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "items": len(items),
                 "changed": changed, "reviewed": reviewed, "skipped": skipped})
    info["corrections"] = hist[-20:]
    info["reviewed_frames"] = sum(1 for m in metas if m.get("status") == "reviewed")
    save_meta(ds, info)
    return {"items": len(items), "changed": changed, "reviewed": reviewed,
            "skipped": skipped}


# ---------------------------------------------------------------- 只标"看过"，不动标注
def mark_reviewed(files, *, name: str = "keys", progress=print) -> dict:
    """把若干帧标成 `reviewed`，**一个字节的标注文件都不改**。

    ## 为什么单独做这一个动作（而不是复用 `apply_corrections`）

    用户的复核习惯是**按页翻**：有错的才改，没错的直接翻过去。
    让他为每张正确的帧点一次「没问题」是纯浪费，所以界面在**翻页时**
    自动把当前页记成已复核。

    但这一步**绝不能**走 `apply_corrections`：

    * `apply_corrections` 会用**当前标定**（`cues.BRAKE_KEY_BOX`）重写标注框 ——
      若数据集是用另一套标定建的，翻一页就把整页标注**悄悄挪到新位置** ✗✗，
      顺带把"数据集和标定对不上"这个体检结论也抹掉了（而那正是它要报的错）；
    * 这里只动 `meta.jsonl` 的 `status` 字段，**标注几何完全不动**，
      于是"翻页"在语义上就只是「人看过这一页、没发现问题」。

    `files` 里不存在的文件名会被跳过并计数（**不静默算成功**）。
    返回值带 `prev`（每个文件原来的 status），供 `undo_last_mark()` 撤销。
    """
    ds = dataset_dir(name)
    # 状态这一层虽然不碰几何，但"选路数据集被按键页面标成已复核"同样是坏数据：
    # 它会把选路那套的「已复核」计数掺水，而 `train eval` 正是拿
    # 「已复核帧」当**真答案**用的（见 runner 的 `labels_are_truth`）。
    _require_keys_dataset(ds, name, "标记已复核")
    metas = read_frames_meta(ds)
    if not metas:
        raise RuntimeError("{0} 里没有 {1}".format(ds, FRAME_META))
    by_file = {m.get("file"): m for m in metas if m.get("file")}
    prev: dict = {}
    changed = 0
    skipped = 0
    for fn in files:
        m = by_file.get(fn)
        if m is None:
            skipped += 1
            continue
        old = m.get("status") or "prelabel"
        prev[fn] = old
        if old != "reviewed":
            m["status"] = "reviewed"
            changed += 1
    (ds / FRAME_META).write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metas),
        encoding="utf-8")
    info = load_meta(ds)
    info["reviewed_frames"] = sum(1 for m in metas if m.get("status") == "reviewed")
    # 只留**最近一次、而且真的改动过**的批次 —— 它就是"撤销上一页"的粒度。
    # ⚠️ `changed == 0`（这一页本来就已经全标记过了）时**不要覆盖**旧记录：
    # 否则"重翻一次同一页"这种无意义的动作会把撤销点冲掉，
    # 让人再也撤不回真正想撤的那一页（这条是测试当场抓出来的）。
    if changed:
        info["last_marks"] = {"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "files": list(prev), "prev": prev, "changed": changed}
    save_meta(ds, info)
    if progress and changed:
        progress("    标记 {0} 帧已复核（标注文件未改动）".format(changed))
    return {"files": len(prev), "changed": changed, "skipped": skipped,
            "reviewed_total": info["reviewed_frames"], "prev": prev}


def undo_last_mark(*, name: str = "keys") -> dict:
    """撤销**上一次** `mark_reviewed()`（= "手快连着翻了两页"）。

    只认最近一次，因为那正好是"上一页"的粒度。恢复的是每帧**原来**的 status
    （原来是 `prelabel` 就回 `prelabel`，原来已经是 `reviewed` 就保持 `reviewed`）。
    """
    ds = dataset_dir(name)
    _require_keys_dataset(ds, name, "撤销标记")
    info = load_meta(ds)
    last = info.get("last_marks")
    if not last or not last.get("prev"):
        raise RuntimeError("没有可撤销的记录（没标记过，或者已经撤过了）")
    prev = last["prev"]
    metas = read_frames_meta(ds)
    restored = 0
    for m in metas:
        fn = m.get("file")
        if fn in prev and (m.get("status") or "prelabel") != prev[fn]:
            m["status"] = prev[fn]
            restored += 1
    (ds / FRAME_META).write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metas),
        encoding="utf-8")
    info["reviewed_frames"] = sum(1 for m in metas if m.get("status") == "reviewed")
    info.pop("last_marks", None)
    save_meta(ds, info)
    return {"restored": restored, "files": len(prev),
            "reviewed_total": info["reviewed_frames"], "at": last.get("at")}


# ---------------------------------------------------------------- 拼图
def contact_sheets(name: str = "keys", *, per_sheet: int = 24, cols: int = 4,
                   limit: int | None = None, progress=print) -> list[Path]:
    """"一整张图看一片帧"：每格 = 两个按键的放大图 + 文字。

    ⚠️ 这里**只能用 ASCII 文字**：`cv2.putText` 画不了中文（画出来是问号），
    所以格子上是 `B`/`N`/reason 的英文；中文说明看 `queue.txt`。
    """
    import cv2
    import numpy as np

    from a9route import config as cfgmod
    from a9route.vision import cues

    cfgmod.apply()
    ds = dataset_dir(name)
    pool = ds / POOL_DIR
    items = build_queue(read_frames_meta(ds))
    if limit:
        items = items[:int(limit)]
    out = ds / REVIEW_DIR
    out.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h = 300, 150
    made: list[Path] = []
    for si in range(0, len(items), per_sheet):
        chunk = items[si:si + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = np.full((rows * cell_h, cols * cell_w, 3), 24, np.uint8)
        for k, it in enumerate(chunk):
            frame = cv2.imread(str(pool / "images" / it.file))
            if frame is None:
                continue
            r, c = divmod(k, cols)
            cb = _crop(frame, cues.BRAKE_KEY_BOX, zoom=1)
            cn = _crop(frame, cues.NITRO_KEY_BOX, zoom=1)
            h = min(cb.shape[0], cn.shape[0], cell_h - 46)

            def _fit(im, hh=h):
                return cv2.resize(im, (max(1, int(im.shape[1] * hh / im.shape[0])), hh),
                                  interpolation=cv2.INTER_NEAREST)

            cb2, cn2 = _fit(cb), _fit(cn)
            y0, x0 = r * cell_h + 30, c * cell_w + 4
            sheet[y0:y0 + h, x0:x0 + cb2.shape[1]] = cb2
            x1 = x0 + cb2.shape[1] + 6
            sheet[y0:y0 + h, x1:x1 + cn2.shape[1]] = cn2
            cv2.putText(sheet, "{0} {1}".format(it.file[:24], it.reason),
                        (c * cell_w + 4, r * cell_h + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (200, 200, 200), 1, cv2.LINE_AA)
            lab = "{0}{1}  r={2:.2f} ring={3:.2f}".format(
                "B" if "brake_pressed" in it.boxes else "-",
                "N" if "nitro_pressed" in it.boxes else "-",
                float(it.signals.get("nitro_red", 0.0)),
                float(it.signals.get("brake_ring", 0.0)))
            cv2.putText(sheet, lab, (c * cell_w + 4, r * cell_h + cell_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 220, 255), 1, cv2.LINE_AA)
        path = out / "sheet_{0:03d}.jpg".format(si // per_sheet + 1)
        cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        made.append(path)
        if progress:
            progress("    拼图 {0}（{1} 帧）".format(len(made), len(chunk)))
    return made
