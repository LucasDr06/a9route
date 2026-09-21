/* a9route 选路标注页 —— 前端逻辑
 *
 * 原生 ES2020，无构建步骤、无外部依赖（完全离线可用）。
 * 只依赖这些接口（后端在 web/choice.py 里，前端不关心实现）：
 *   GET  /api/choice/list     数据集列表（帧数 / 已标注 / 有选路）
 *   GET  /api/choice/frames   一页帧（答案 + 检测结果 + 框 + band 带区）
 *   GET  /api/choice/image    整帧缩略图 / band 放大图
 *   POST /api/choice/answer   提交答案（可选带 boxes）
 *   POST /api/choice/undo     撤销上一次标记
 *
 * ## 这个页面和 `/dataset` 是**两件事**，别混
 *
 * `/dataset` 管的是「刹车 / 氮气」那套按键标注和标定体检；
 * 这里管的是「选路（岔路口）」。两页各有各的数据、各有各的接口，互不读写。
 *
 * ## 界面上最要紧的那件事：三个选择题
 *
 *   ① 有没有选路 —— 没有 / 有
 *   ② 几个选项   —— 2 / 3 / 4        （只有答了「有」才可点）
 *   ③ 选第几个   —— 1..N            （按钮跟着②当场长出来）
 *
 * 用户的游戏规则：一屏只有 2/3/4 条路，**至少 2 个路标**才算岔路口。
 * 所以 ①「有」的帧，② 的候选最少是 2；③ 的下标按**从左到右**数，1 起。
 *
 * 每帧要 show 两张图：
 *   * **band 图** —— 屏幕上方那条路标带的放大图（判据真正看的那条带）；
 *   * **整帧缩略图** —— 点开放大。
 *
 * band 图上叠三套框（全用整帧像素坐标算，最后转成**百分比**定位，
 * 所以窗口怎么缩放都不会错位）：
 *   * 红实线 = 你选中的那一个（第③题的答案）；
 *   * 黄实线 = 其余选项；
 *   * 蓝虚线 = 服务器 heuristic 检出的圆（`detected.icons`）。
 *
 * 换算：band 图是 `crop_rect = [x, y, w, h]` 这块区域按 `zoom` 放大后的**裁剪图**
 * （`crop_rect` 由服务器逐行给出，和实际裁出来的图**必然是同一块**）。
 * 所以：
 *   * 图内像素 -> 整帧：`整帧x = crop_rect[0] + 图内像素x * (crop_rect[2] / 图的自然宽度)`；
 *   * 整帧 -> overlay 的百分比：`left% = (x - crop_rect[0]) / crop_rect[2] * 100`。
 *
 * ⚠️ **不要用 `band_rect` 换算**：那只是"带区本身"，裁图时各边还多留了一点边，
 * 拿它算会整体偏 4%（越靠右偏得越多）。`band_rect` 现在只是给界面显示用的信息。
 *
 * 反过来要点图加框，就用 `img.getBoundingClientRect()` 把鼠标位置按**实际显示尺寸**
 * 换算回去 —— **不假设显示尺寸**（CSS 缩放 / 窗口变宽都会变）。
 *
 * **翻页即复核**（和 `/dataset` 页同一个习惯）：离开当前页时，把本页
 * **没答过的帧按「检测结果」当默认答案**、**答过/改过的按你的答案**，
 * 一起提交成「已标注」。所以没错的帧不用逐个点。
 */
'use strict';

(function () {
  var $ = function (s) { return document.querySelector(s); };
  var $$ = function (s) { return Array.prototype.slice.call(document.querySelectorAll(s)); };

  var el = {
    banner: $('#connBanner'), toasts: $('#toasts'),
    dsSelect: $('#dsSelect'), dsRefreshBtn: $('#dsRefreshBtn'),
    dsHint: $('#dsHint'), refreshBtn: $('#refreshBtn'),
    exportBtn: $('#exportBtn'),
    fFilter: $('#fFilter'), fLimit: $('#fLimit'), fZoom: $('#fZoom'),
    fSearch: $('#fSearch'), fApply: $('#fApply'),
    fMarkPage: $('#fMarkPage'), fUndoMark: $('#fUndoMark'),
    framePill: $('#framePill'), frameNote: $('#frameNote'),
    optRange: $('#optRange'),
    errorBox: $('#errorBox'), pendingHint: $('#pendingHint'),
    chGrid: $('#chGrid'), chPager: $('#chPager'),
    lightbox: $('#lightbox'), lightboxImg: $('#lightboxImg'),
    lightboxCap: $('#lightboxCap')
  };

  var state = {
    dataset: '', offset: 0, limit: 24, page: null,
    /* 第②题「有几个选项」允许的范围 —— **由后端给**（`vision.choice_min_options`
     * / `choice_max_options`）。前端以前写死 2/3/4，于是那两个配置项
     * "看起来能调、调了没用"；现在按钮、夹取、写回三处都跟着这个走。 */
    opts: { min: 2, max: 4 },
    answers: {},          // file -> {has, count, selected, countEdited, boxes?}
    manual: {},           // file -> true（位置被人工改过 —— 提交要带 boxes）
    /* 自动提交失败（后端 400/500）后**不再**自动重试：否则每翻一次页就重发一次
     * 注定失败的请求，用户只看到刷屏的报错。手动点「本页没问题」可以再试。 */
    autoCommitBroken: false,
    committing: null,     // 正在提交的 Promise（防止并发翻页重复提交）
    cursor: ''            // 键盘操作的那一行
  };

  // ------------------------------------------------------------ 小工具
  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function toast(text, kind) {
    var d = document.createElement('div');
    d.className = 'toast' + (kind ? ' toast-' + kind : '');
    d.textContent = text;
    el.toasts.appendChild(d);
    setTimeout(function () { d.remove(); }, 4600);
  }

  /* 顶部红色提示条 —— 接口 400/500 的**原因**必须显示出来，不许静默失败 */
  function banner(text) {
    if (!text) { el.banner.classList.add('hidden'); return; }
    el.banner.textContent = text;
    el.banner.classList.remove('hidden');
  }

  /* 页面里的错误（读帧 / 提交失败）走这条 —— 红色框，不静默 */
  function showError(text) {
    if (!text) { el.errorBox.classList.add('hidden'); el.errorBox.textContent = ''; return; }
    el.errorBox.textContent = text;
    el.errorBox.classList.remove('hidden');
  }

  function api(path, opts) {
    opts = opts || {};
    var init = { method: opts.method || 'GET', headers: {} };
    if (opts.body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(opts.body);
    }
    return fetch(path, init).then(function (r) {
      return r.text().then(function (t) {
        var data = null;
        try { data = t ? JSON.parse(t) : null; } catch (e) { data = null; }
        if (!r.ok) {
          // 后端错误体约定是 {"error": "中文原因"}；没有就退回 HTTP 状态码
          throw new Error((data && (data.error || data.message)) || ('HTTP ' + r.status));
        }
        return data;
      });
    });
  }

  function q(params) {
    return Object.keys(params).map(function (k) {
      return encodeURIComponent(k) + '=' + encodeURIComponent(params[k]);
    }).join('&');
  }

  function imgUrl(file, mode, extra) {
    var p = { name: state.dataset, file: file, mode: mode || 'full' };
    Object.keys(extra || {}).forEach(function (k) { p[k] = extra[k]; });
    return '/api/choice/image?' + q(p);
  }

  function cssEsc(s) {
    return String(s).replace(/["\\]/g, '\\$&');
  }

  function rowOf(file) {
    var rows = (state.page && state.page.rows) || [];
    for (var i = 0; i < rows.length; i++) {
      if (rows[i].file === file) { return rows[i]; }
    }
    return null;
  }

  // ------------------------------------------------------------ 三连问的答案
  /* 把各种来源（服务器答案 / 前端改动 / detected 兜底）统一成同一个形状：
   * {has, count, selected, countEdited, boxes?}
   * 归一化规则（用户的游戏规则）：没有 -> count=0, selected=0；
   * 有 + 2/3/4 -> selected 落在 0..count。 */
  function normAns(a) {
    a = a || {};
    var has = !!a.has;
    var count = parseInt(a.count, 10) || 0;
    var selected = parseInt(a.selected, 10) || 0;
    if (!has) { return { has: false, count: 0, selected: 0,
                         countEdited: !!a.countEdited, boxes: a.boxes }; }
    if (count < state.opts.min) { count = state.opts.min; }
    if (count > state.opts.max) { count = state.opts.max; }
    if (selected < 0 || selected > count) { selected = count; }
    return { has: true, count: count, selected: selected,
             countEdited: !!a.countEdited, boxes: a.boxes };
  }

  /* 框**按中心 x 从左到右**排序 —— 界面上说的「第 k 个」和标注里写的
   * 「第 k 个」必须是同一个（后端 `boxes_from_answers` 也是按 x 排序的）。
   *
   * ⚠️ 不排的话：点图加的框会**接到数组末尾**，于是"你选的"红框指着一个圆，
   * 而落进标注的 `choice_selected` 是另一个圆 —— 加/删框看起来"不准"，
   * 其实一半是"顺序不一致"（NOTES 10.21 那条的同族问题）。 */
  function sortBoxes(boxes) {
    return (boxes || []).slice().sort(function (a, b) {
      return (a[0] + a[2] / 2) - (b[0] + b[2] / 2);
    });
  }

  /* 当前答案：本页改过就用改过的，否则用服务器给的。
   * 注意把服务器的 `boxes` **一起带上** —— 点 band 图加框时是"框 + 新框"，
   * 不是从零开始（不然会把手改的基准丢了）。 */
  function ansOf(file) {
    var a = state.answers[file];
    if (a) { return a; }
    var row = rowOf(file);
    var n = normAns(row ? row.answers : null);
    n.boxes = sortBoxes(((row && row.boxes) || []).map(function (b) {
      return b.slice();
    }));
    return n;
  }

  /* 「检测结果」当默认答案 —— 翻页自动复核、以及「本页没问题」都用它。
   * ⚠️ 检测到的图标**不到 min 个**（默认 2）就按「没有选路」算：这是运行时的
   * 同一口径（`video.scan_fine` 里 `len(found) >= 2` 才算岔路口）。
   * 以前这里会把它凑成 2 个 -> 后端补一个**猜出来的框**，等于凭空造了个路标 ✗ */
  function detectAns(row) {
    var d = (row && row.detected) || {};
    var c = parseInt(d.count, 10) || 0;
    if (!c || c < state.opts.min) { return { has: false, count: 0, selected: 0 }; }
    return normAns({ has: true, count: c,
                     selected: parseInt(d.selected, 10) || 0 });
  }

  function setAns(file, ans) {
    var n = normAns(ans);
    if (n.boxes) { n.boxes = sortBoxes(n.boxes); }   // 顺序统一（见 sortBoxes）
    state.answers[file] = n;
    paint(file);
    updatePending();
  }

  /* 加框时猜一个数：优先「服务器答的 / 你答的」，其次检测到的框数 */
  function guessCount(row) {
    var a = (row && row.answers) || {};
    var inA = parseInt(a.count, 10) || 0;
    var d = (row && row.detected) || {};
    var inD = parseInt(d.count, 10) || 0;
    var g = Math.max(inA, inD, state.opts.min);
    return Math.min(state.opts.max, Math.max(state.opts.min, g));
  }

  /* ① 有没有选路 */
  function answerHas(file, has) {
    var a = ansOf(file);
    var next = normAns({ has: has, count: a.count, selected: a.selected,
                         countEdited: a.countEdited, boxes: a.boxes });
    if (has && !a.has && !a.count) {
      // 从「没有」切到「有」：给个合理的初始值（用检测结果），再让人改
      var g = guessCount(rowOf(file));
      next.count = g;
      next.selected = Math.min(Math.max(1, next.selected || g), g);
    }
    if (!has) { state.manual[file] = false; }
    setAns(file, next);
  }

  /* ② 有几个选项 */
  function answerCount(file, count) {
    var a = ansOf(file);
    var next = normAns({ has: true, count: count, selected: a.selected,
                         countEdited: true, boxes: a.boxes });
    setAns(file, next);
  }

  /* ③ 选第几个（1 起） */
  function answerSelected(file, i) {
    var a = ansOf(file);
    setAns(file, { has: true, count: a.count || 2, selected: i,
                   countEdited: a.countEdited, boxes: a.boxes });
  }

  /* 本帧「没问题」= 就直接把「检测结果」当答案收下（没检测到就是「没有选路」） */
  function answerOk(file) {
    var row = rowOf(file);
    if (!row) { return; }
    setAns(file, detectAns(row));
    state.manual[file] = false;
    toast('本帧按检测结果记为已标注（' + ansText(state.answers[file]) + '）', 'ok');
  }

  function ansText(a) {
    a = normAns(a);
    if (!a.has) { return '没有选路'; }
    return a.count + ' 个选项 · 选第 ' + (a.selected ? a.selected : '？（还没选）');
  }

  // ------------------------------------------------------------ 数据集列表
  function loadDatasets(keep) {
    return api('/api/choice/list').then(function (data) {
      var list = (data && data.datasets) || [];
      el.dsSelect.innerHTML = '';
      if (!list.length) {
        var o = document.createElement('option');
        o.value = ''; o.textContent = '（还没有数据集）';
        el.dsSelect.appendChild(o);
        el.dsHint.textContent = '数据集目录：' + ((data && data.root) || '?');
        el.chGrid.innerHTML = '<p class="muted">还没有可选路标注的数据集。</p>';
        return;
      }
      list.forEach(function (d) {
        var opt = document.createElement('option');
        opt.value = d.name;
        opt.textContent = d.name + '  (' + d.frames + ' 帧'
          + (d.reviewed ? '，已标注 ' + d.reviewed : '')
          + (d.with_choice ? '，有选路 ' + d.with_choice : '')
          + ')';
        el.dsSelect.appendChild(opt);
      });
      if (keep && list.some(function (d) { return d.name === state.dataset; })) {
        el.dsSelect.value = state.dataset;
      }
      state.dataset = el.dsSelect.value;
      var d0 = list.filter(function (d) { return d.name === state.dataset; })[0] || {};
      el.dsHint.textContent = '数据集目录：' + ((data && data.root) || '?')
        + (d0.has_split ? '' : '  ⚠️ 这个数据集还没有 train/val 划分（全是 pool）');
    }).catch(function (err) {
      banner('后端连不上：' + err.message + '（接口 GET /api/choice/list）');
      el.dsHint.textContent = '读取失败：' + err.message;
    });
  }

  // ------------------------------------------------------------ 提交
  /* 本页还剩多少「没提交过」的帧 —— 没答过的也算（会用检测结果当默认答案），
   * 这正是「翻页即复核」要提交的东西 */
  function pendingFiles() {
    var d = state.page;
    if (!d || !d.rows) { return []; }
    return d.rows.filter(function (r) {
      return state.answers[r.file] || state.manual[r.file] || r.status !== 'reviewed';
    });
  }

  function updatePending() {
    var pend = pendingFiles();
    if (!pend.length) {
      el.pendingHint.classList.add('hidden');
      el.pendingHint.textContent = '';
      return;
    }
    var hand = 0;
    pend.forEach(function (r) { if (state.answers[r.file] || state.manual[r.file]) { hand++; } });
    el.pendingHint.textContent = '待提交 ' + pend.length + ' 帧'
      + (hand ? '（其中 ' + hand + ' 帧是你答的 / 改过的）' : '')
      + ' —— 离开本页（翻页 / 换筛选 / 刷新）会自动提交为「已标注」；'
      + (state.autoCommitBroken ? '上次自动提交失败，点「本页没问题」可重试。' : '');
    el.pendingHint.classList.remove('hidden');
  }

  /* 把一帧变成 POST /api/choice/answer 要的 item */
  function itemFor(row, ans, manual) {
    ans = normAns(ans);
    var it = { file: row.file, has: ans.has, count: ans.count,
               selected: ans.selected, status: 'reviewed' };
    // 只有在**位置被人工改过**时才带 boxes —— 不然就交给服务器用检测到的位置，
    // 免得把启发式的结果又原样写回去一遍
    if (manual && ans.boxes) {
      it.boxes = ans.boxes.map(function (b) {
        return [Math.round(b[0]), Math.round(b[1]), Math.round(b[2]), Math.round(b[3])];
      });
    }
    return it;
  }

  function commitPage(reason, opts) {
    opts = opts || {};
    if (state.committing) { return state.committing; }
    var d = state.page;
    if (!d || !d.rows || !d.rows.length) { return Promise.resolve(0); }

    var items = [];
    d.rows.forEach(function (r) {
      var isManual = !!state.manual[r.file];
      var ans = state.answers[r.file];
      if (!ans) {
        if (r.status === 'reviewed' && !isManual) { return; }   // 本来就已标注，不用重发
        ans = detectAns(r);                                     // 没答过 -> 用检测结果兜底
      }
      items.push(itemFor(r, ans, isManual));
    });
    if (!items.length) {
      if (!opts.quiet) { toast('本页没有要提交的（都已经是「已标注」了）', 'warn'); }
      return Promise.resolve(0);
    }

    state.committing = api('/api/choice/answer', {
      method: 'POST',
      body: { dataset: d.dataset, items: items }
    }).then(function (res) {
      var r = (res && res.result) || {};
      d.rows.forEach(function (row) { row.status = 'reviewed'; });
      // 本页已提交 -> 清掉本地暂存，免得同一页被重复提交
      d.rows.forEach(function (row) {
        delete state.answers[row.file];
        delete state.manual[row.file];
      });
      state.autoCommitBroken = false;
      showError('');
      updatePending();
      var n = r.changed != null ? r.changed : items.length;
      // 后端现在只统计**真的改了**的帧（内容/状态和原来不同）——
      // 翻页顺手提交一整页时，没动过的帧不该被报成"改动"
      // （NOTES 10.22：假消息 + 会把撤销点顶掉）
      var sameN = r.same != null ? r.same : Math.max(0, items.length - n);
      var detail = '改动 ' + n + ' 帧';
      if (sameN) { detail += '，' + sameN + ' 帧本来就对'; }
      toast((reason ? reason + '：' : '') + '提交 ' + items.length + ' 帧（'
        + detail + (r.skipped ? '，跳过 ' + r.skipped : '') + '）✓', 'ok');
      return items.length;
    }).catch(function (err) {
      state.autoCommitBroken = true;
      showError('提交失败（POST /api/choice/answer）：' + err.message
        + ' —— 答案还留在页面上没丢，修好后点「本页没问题」重试。');
      toast('提交失败：' + err.message, 'error');
      return 0;
    }).then(function (n) {
      state.committing = null;
      return n;
    });
    return state.committing;
  }

  // ------------------------------------------------------------ 读一页帧
  function loadFrames(offset, opts) {
    opts = opts || {};
    if (!state.dataset) { return; }
    // 离开当前页之前先把这一页结掉 —— **用户要的「翻过就算复核」**。
    // 自动提交失败过就不自动重试了（见 state.autoCommitBroken），
    // 否则每翻一页都重发一次注定失败的请求。
    if (!opts.skipCommit && !state.autoCommitBroken) {
      var n = commitPage(opts.reason || '翻页');
      if (n && n.then) {
        return n.then(function () { return loadFrames(offset, { skipCommit: true }); });
      }
    }
    if (offset !== undefined) { state.offset = Math.max(0, offset); }
    state.limit = parseInt(el.fLimit.value, 10) || 24;
    var params = {
      name: state.dataset, offset: state.offset, limit: state.limit,
      filter: el.fFilter.value, search: el.fSearch.value.trim(), recheck: 1
    };
    el.framePill.textContent = '加载中…';
    el.framePill.className = 'pill pill-running';
    api('/api/choice/frames?' + q(params)).then(function (d) {
      state.page = d;
      // 第②题的范围**由后端定**（跟随 `vision.choice_min_options/…max_options`）
      if (d && d.options) {
        var lo = parseInt(d.options.min, 10) || 2;
        var hi = parseInt(d.options.max, 10) || 4;
        state.opts = { min: lo, max: Math.max(lo, hi) };
        // 说明文字里的范围也跟着变（不然配置改了、页面还写着 2/3/4）
        if (el.optRange) {
          var list = [];
          for (var k = state.opts.min; k <= state.opts.max; k++) { list.push(k); }
          el.optRange.textContent = list.join('/');
        }
      }
      renderFrames(d);
    }).catch(function (err) {
      el.chGrid.innerHTML = '<p class="alert alert-error">读帧失败（GET /api/choice/frames）：'
        + esc(err.message) + '</p>';
      el.framePill.textContent = '失败';
      el.framePill.className = 'pill pill-error';
      showError('读帧失败：' + err.message);
    });
  }

  // ------------------------------------------------------------ 画框
  /* 全帧像素框 -> 百分比字符串（left/top/width/height），用百分比定位，
   * 所以窗口怎么缩放、图片实际显示多大，都不会错位 */
  function boxPct(b) {
    return 'left:' + (b[0]).toFixed(3) + '%;top:' + (b[1]).toFixed(3)
      + '%;width:' + (b[2]).toFixed(3) + '%;height:' + (b[3]).toFixed(3) + '%';
  }

  /* 整帧坐标的圆 -> band 图内的百分比框。
   *
   * ⚠️ **百分比只跟 crop_rect 有关，`zoom` / 显示尺寸都不该出现**：
   * 叠加层是贴在 band 图**自己那个盒子**上的（`.ch-ovs { inset: 0 }`），
   * 而那张图显示的就是 crop 那一块 —— 所以"占宽度的几成" = 距 crop 左边几成。
   * 第一版在这里多除了一个 `* scale`（= 整帧像素/图内像素），
   * 结果 `scale = 1/zoom` 把它整体放大了 `zoom` 倍：框跑到右下、还大一倍 ✗✗
   * （用户截图里那一堆套在一起的大黄圈就是这个）—— 这个 bug 还很坏：
   * **画出来的圈和真正能点的区域不是一个地方**，所以"加/删一个框"也不准。
   * 教训：`scale` 是给"点击 -> 整帧坐标"用的，**画百分比用不到它**。 */
  function bandPct(i, crop) {
    var cw = crop[2] || 1, ch = crop[3] || 1;
    var d = (i.r || 0) * 2;
    return 'left:' + ((i.x - i.r - crop[0]) / cw * 100).toFixed(3) + '%;top:'
      + ((i.y - i.r - crop[1]) / ch * 100).toFixed(3) + '%;width:'
      + (d / cw * 100).toFixed(3) + '%;height:'
      + (d / ch * 100).toFixed(3) + '%';
  }

  /* band 图**实际覆盖的整帧区域** `[x, y, w, h]`。
   *
   * 服务器每行都给 `crop_rect`（就是 `train.choice.band_crop_rect()` 那个值），
   * 它和 `/api/choice/image?mode=band` 裁出来那张图**是同一块** —— 后端两边共用
   * 同一个函数，所以不可能漂移。
   *
   * ⚠️ 换算**必须**用它，不能用 `band_rect`：`band_rect` 只是"带区本身"，
   * 而裁图时上下左右各留了一点边（`BAND_PAD`），所以 `band_rect × zoom` 比真图小一圈，
   * 拿它换算会整体偏约 4%（越靠右偏得越多，最多三百多像素，一眼可见）。
   *
   * 万一两个字段都没有（后端太老），返回 null —— 调用方会退回"不画框"这个安全选项：
   * **绝不拿 `band_rect` 顶替，那会画出偏 4% 的框**，比不画框更坏（看着对、其实错）。 */
  function cropOf(row) {
    if (row && row.crop_rect && row.crop_rect.length === 4) { return row.crop_rect; }
    if (state.page && state.page.crop_rect && state.page.crop_rect.length === 4) {
      return state.page.crop_rect;          // 顶层也给了，行里没给就用它
    }
    return null;
  }

  /* band 图**每 1 个图内像素**代表多少整帧像素 —— **只给点击换算用**。
   *
   * ⚠️ 别拿它去算百分比（那是第一版的 bug，见 `bandPct` 的说明）：
   * 百分比是"占 crop 的几成"，和 zoom / 显示尺寸无关，scale 会在约分里消掉。
   *
   * `img.naturalWidth` 是那张图实际返回的像素宽（JPEG 有损，偶尔差 1px），
   * 拿它当分母最保险；图还没 load 出来时退回 `1/zoom`（裁图就是 `crop × zoom`）。 */
  function cropScale(crop, zoom, naturalW) {
    if (!crop) { return 1 / (zoom || 2); }
    var cw = crop[2] || 1;
    if (naturalW) { return cw / naturalW; }
    return 1 / (zoom || 2);               // 图还没 load 出来时的兜底
  }

  /* band 图上的叠加层。三种框，颜色按**你的答案**来：
   *   蓝虚线 = 检测到的圆；红实线 = 你选中的那一个；黄实线 = 其余选项。
   *
   * ⚠️ 画的是 `ans.boxes`（**你当前答案里的框**），不是服务器那一份：
   * 点图加/删之后必须**立刻**跟着变，否则"点了没反应/画的和答案不是一回事" ✗
   * （第一版画的是 `row.boxes`，于是加删框在画面上完全看不出来）。 */
  function bandOverlay(row, ans) {
    var crop = cropOf(row);
    if (!crop) { return ''; }        // 没有 crop_rect 就不画框（见 cropOf 的说明）
    var cw = crop[2] || 1, ch = crop[3] || 1;
    var boxes = (ans && ans.boxes) || row.boxes || [];
    var out = '';

    (row.detected && row.detected.icons || []).forEach(function (i) {
      out += '<div class="ch-ov ch-ov-det" style="' + bandPct(i, crop)
        + '" title="检测到的圆 x=' + i.x + ' y=' + i.y + ' r=' + i.r
        + (i.blue ? '（蓝色路标）' : '') + '"></div>';
    });

    boxes.forEach(function (b, k) {
      // 整帧像素 -> band 图内的百分比：减掉 crop 左上角，再除以 crop 的宽/高
      var l = (b[0] - crop[0]) / cw * 100;
      var t = (b[1] - crop[1]) / ch * 100;
      var w = b[2] / cw * 100;
      var h = b[3] / ch * 100;
      var pick = !!(ans.has && ans.selected === k + 1);
      out += '<div class="ch-ov ' + (pick ? 'ch-ov-pick' : 'ch-ov-opt')
        + '" style="left:' + l.toFixed(3) + '%;top:' + t.toFixed(3)
        + '%;width:' + w.toFixed(3) + '%;height:' + h.toFixed(3) + '%"'
        + ' title="第 ' + (k + 1) + ' 个' + (pick ? '（你选的）' : '') + '"></div>';
    });
    return out;
  }

  /* 整帧缩略图上的叠加：同一批框，但画在**整帧**坐标系里，所以直接按整帧尺寸算百分比。
   * 不假设整帧尺寸 —— `fw/fh` 是该图在整帧像素坐标系里的总尺寸
   * （服务器给 `full_w/full_h` 就用它，没有就退回 1280×720，缩略图本来只是"点开看清"用）。 */
  /* 整帧缩略图上的叠加：同一批框，但画在**整帧**坐标系里，所以直接按整帧尺寸算百分比。
   * 不假设整帧尺寸 —— `fw/fh` 是该图在整帧像素坐标系里的总尺寸
   * （服务器给 `full_w/full_h` 就用它，没有就退回 1280×720，缩略图本来只是"点开看清"用）。
   *
   * ⚠️ 这里**也**踩过同一个坑：第一版把宽高清了一堆 `cw/scale/naturalWidth`，
   * 于是缩略图上的框比真框大十倍，糊在画面上（截图里那些大圈有一半是它）。
   * 整帧坐标 -> 整帧百分比就是 `x/W`，**一个多余的系数都不该有**。
   * 前提是 `.ch-thumb` 必须铺满它自己的盒子（CSS 里的 `object-fit: contain` +
   * `max-height` 会让图在盒子里留黑边，百分比就对不上了 —— 已经去掉）。 */
  function fullOverlay(row, ans, band, zoom, fw, fh) {
    var crop = cropOf(row);
    if (!crop) { return ''; }        // 同 bandOverlay：没有 crop_rect 就不画框
    var W = fw || 1280, H = fh || 720;
    var boxes = (ans && ans.boxes) || row.boxes || [];
    var out = '';
    (row.detected && row.detected.icons || []).forEach(function (i) {
      var d = (i.r || 0) * 2;
      out += '<div class="ch-ov ch-ov-det" style="' + boxPct(
        [(i.x - i.r) / W * 100, (i.y - i.r) / H * 100,
         d / W * 100, d / H * 100]) + '"></div>';
    });
    boxes.forEach(function (b, k) {
      var bf = [b[0] / W * 100, b[1] / H * 100, b[2] / W * 100, b[3] / H * 100];
      var pick = !!(ans.has && ans.selected === k + 1);
      out += '<div class="ch-ov ' + (pick ? 'ch-ov-pick' : 'ch-ov-opt')
        + '" style="' + boxPct(bf) + '"></div>';
    });
    return out;
  }

  // ------------------------------------------------------------ 渲染
  function optBtn(qn, val, file, ans, label, cls) {
    var on = (qn === 1 && !ans.has) || (qn === 2 && ans.has && ans.count === val)
      || (qn === 3 && ans.has && ans.selected === val);
    return '<button type="button" class="ch-opt' + (cls ? ' ' + cls : '')
      + (on ? ' on' : '') + '" data-q="' + qn + '" data-val="' + val + '"'
      + ' data-file="' + esc(file) + '">' + esc(label) + '</button>';
  }

  /* 第 ③ 题的候选按钮**跟着第 ② 题当场长出来**（选 3 就出现 1/2/3） */
  function selectedBtns(file, ans) {
    var out = '';
    var n = ans.has ? ans.count : 0;
    for (var i = 1; i <= n; i++) { out += optBtn(3, i, file, ans, '第 ' + i + ' 个'); }
    if (!n) { out = '<span class="muted">先答「有几个选项」</span>'; }
    return out;
  }

  /* 第 ② 题的按钮**按后端给的范围长**（`state.opts`）——
   * 以前写死 2/3/4，于是 `vision.choice_min_options` 这个配置项"调了没用" ✗ */
  function countBtns(file, ans) {
    var out = '';
    for (var v = state.opts.min; v <= state.opts.max; v++) {
      if (ans.has) { out += optBtn(2, v, file, ans, String(v)); }
      else {
        out += '<button type="button" class="ch-opt" data-q="2" data-val="' + v
          + '" data-file="' + esc(file) + '" disabled>' + v + '</button>';
      }
    }
    return out;
  }

  function mismatchBadge(row, ans) {
    var det = (row.detected && row.detected.count) || 0;
    var mine = ans.has ? ans.count : 0;
    if (mine === det) { return ''; }
    return '<span class="ch-bad">⚠ 检测到 ' + det + ' 个，你答 ' + mine + ' 个</span>';
  }

  /* 后端算出来的每一帧的"要留意什么" —— **必须显示**：
   * 不显示的话，"位置是猜的"（那一帧里有后端补出来的框）就完全看不出来，
   * 人会把一个猜出来的路标当成真的 ✗（NOTES 10.23）。 */
  var FLAG_ZH = {
    position_guessed: '位置是猜的（检测到的图标不够，后端往右补了框）',
    detected_mismatch: '答的个数和检测到的不一样',
    no_selected: '没标出"选中的是哪一个"',
    too_few: '少于最小选项数（按规则不算岔路口）',
    below_min: '检测到图标了，但不够最小选项数 → 按规则算「没有选路」',
    human_fix: '人工改过'
  };

  function flagsHtml(r) {
    var fl = r.flags || [];
    if (!fl.length) { return ''; }
    var out = '';
    fl.forEach(function (f) {
      var txt = FLAG_ZH[f] || f;
      if (f === 'too_few') { txt = '只有 ' + state.opts.min + ' 个以下 → 按规则不算岔路口'; }
      out += '<span class="ch-flag ch-flag-' + esc(f) + '" title="' + esc(f) + '">'
        + esc(txt) + '</span>';
    });
    return '<div class="ch-flags">' + out + '</div>';
  }

  function renderFrames(d) {
    var zoom = parseInt(el.fZoom.value, 10) || 2;
    var html = '';
    (d.rows || []).forEach(function (r) {
      var ans = ansOf(r.file);
      var manual = !!state.manual[r.file];
      var nBox = (r.boxes || []).length;

      html += '<div class="ch-card" data-file="' + esc(r.file) + '" tabindex="0">'
        + '<div class="ch-shot">'
        + '<div class="ch-band-wrap">'
        + '<img class="ch-band" loading="lazy" alt="band"'
        + ' src="' + esc(imgUrl(r.file, 'band', { zoom: zoom })) + '"'
        + ' data-zoom="' + zoom + '" />'
        + '<div class="ch-ovs">' + bandOverlay(r, ans, zoom, 1 / zoom) + '</div>'
        + '</div>'
        + '<div class="ch-full-wrap">'
        + '<img class="ch-thumb" loading="lazy" alt="整帧"'
        + ' src="' + esc(imgUrl(r.file, 'full', { w: 480 })) + '"'
        + ' data-full="' + esc(imgUrl(r.file, 'full', { w: 1280 })) + '"'
        + ' data-cap="' + esc(r.file + '  ' + (r.video || '') + ' @ ' + r.t + 's') + '" />'
        + '<div class="ch-ovs ch-ovs-full"></div>'
        + '</div>'
        + '</div>'
        + '<div class="ch-meta">'
        + '<div class="ch-file">' + esc(r.file) + '</div>'
        + '<div class="ch-sub">' + esc(r.video || '?') + ' @ ' + r.t + 's · 划分 '
        + esc(r.split || '?') + '</div>'
        + '<div class="ch-sub">检测到：<b>' + ((r.detected && r.detected.count) || 0)
        + ' 个</b>（第 ' + (((r.detected && r.detected.selected) || 0) || '?') + ' 个）'
        + (nBox ? ' · 框 ' + nBox + ' 个' : '')
        + (r.status === 'reviewed' ? ' · 已标注' : ' · 待标注')
        + (manual ? ' · <b class="ch-manual">位置已手工修正</b>' : '')
        + (nBox && nBox !== ans.count ? ' · <b class="ch-manual">框数(' + nBox
            + ')和「几个选项」(' + ans.count + ')不一样</b>' : '')
        + '</div>'
        + '<div class="ch-sub ch-you">你答：<b class="ch-ans">' + esc(ansText(ans))
        + '</b>' + mismatchBadge(r, ans) + '</div>'
        + flagsHtml(r)
        + '</div>'

        + '<div class="ch-q">'
        + '<span class="ch-q-n">①</span><span class="ch-q-t">有没有选路</span>'
        + '<span class="ch-opts">'
        + optBtn(1, 1, r.file, ans, '有')
        + optBtn(1, 0, r.file, ans, '没有')
        + '</span></div>'

        + '<div class="ch-q"><span class="ch-q-n">②</span>'
        + '<span class="ch-q-t">有几个选项</span><span class="ch-opts">'
        + countBtns(r.file, ans)
        + '</span></div>'

        + '<div class="ch-q"><span class="ch-q-n">③</span>'
        + '<span class="ch-q-t">选第几个（左→右）</span>'
        + '<span class="ch-opts" data-slot="sel">'
        + (ans.has ? selectedBtns(r.file, ans)
                   : '<span class="muted">先答「有选路」</span>')
        + '</span></div>'

        + '<div class="ch-act">'
        + '<button class="btn btn-sm btn-ghost ch-ok" type="button" data-file="'
        + esc(r.file) + '">本帧没问题（用检测结果）</button>'
        + '</div>'
        + '</div>';
    });

    el.chGrid.innerHTML = html || '<p class="muted">这一页没有帧（换个筛选条件试试）。</p>';

    // 图加载完才知道显示尺寸 —— 这时才用 naturalWidth / 实际显示宽度算缩放，画 overlay
    $$('.ch-card').forEach(function (card) {
      var band = card.querySelector('.ch-band');
      var full = card.querySelector('.ch-thumb');
      var ovs = card.querySelector('.ch-ovs:not(.ch-ovs-full)');
      var fullOvs = card.querySelector('.ch-ovs-full');
      var row = rowOf(card.dataset.file);

      function drawBand() {
        if (!row || !ovs) { return; }
        // ⚠️ 这里**不需要** scale：画百分比只跟 crop_rect 有关（见 bandPct 的说明）
        ovs.innerHTML = bandOverlay(row, ansOf(row));
      }
      function drawFull() {
        if (!row || !fullOvs) { return; }
        // 整帧图是缩过的缩略图，它在**整帧像素坐标系**里的总尺寸由服务器给的
        // full_w/full_h 决定（拿不到就退回 1280×720）—— 不假设图片显示多大
        var fw = row.full_w || 1280, fh = row.full_h || 720;
        fullOvs.innerHTML = fullOverlay(row, ansOf(row), band, zoom, fw, fh);
      }
      if (band && ovs) {
        band.addEventListener('load', drawBand);
        band.addEventListener('error', function () { band.alt = 'band 图读不出来'; });
      }
      if (full) {
        full.addEventListener('load', drawFull);
        full.addEventListener('error', function () { full.alt = '整帧读不出来'; });
      }
      drawBand();
      drawFull();
    });

    var shown = (d.rows || []).length;
    el.framePill.textContent = shown + ' / ' + (d.total_known === false ? '?' : d.total) + ' 帧';
    el.framePill.className = 'pill';
    var st = d.stats || {};
    el.frameNote.textContent = '（数据集共 ' + d.all_frames + ' 帧；有选路 '
      + (st.with_choice != null ? st.with_choice : '?') + '，已标注 '
      + (st.reviewed != null ? st.reviewed : '?') + '，待标注 '
      + (st.pending != null ? st.pending : '?')
      + (d.total_known === false ? '；"只看不一致"模式下总数要扫完才知道' : '') + '）';
    renderPager(d);
    updatePending();
  }

  function renderPager(d) {
    var prev = Math.max(0, state.offset - state.limit);
    var next = d.next_offset;
    var hasNext = d.total_known === false ? (d.rows || []).length > 0 : next < d.total;
    el.chPager.innerHTML =
      '<button class="btn btn-sm" id="pgFirst" ' + (state.offset <= 0 ? 'disabled' : '')
      + '>« 首页</button>'
      + '<button class="btn btn-sm" id="pgPrev" ' + (state.offset <= 0 ? 'disabled' : '')
      + '>‹ 上一页</button>'
      + '<span class="muted">第 ' + (Math.floor(state.offset / state.limit) + 1)
      + ' 页（第 ' + (state.offset + 1) + ' 帧起）</span>'
      + '<button class="btn btn-sm" id="pgNext" ' + (hasNext ? '' : 'disabled')
      + '>下一页 ›</button>';
    var b;
    if ((b = $('#pgFirst'))) { b.onclick = function () { loadFrames(0, { reason: '首页' }); }; }
    if ((b = $('#pgPrev'))) { b.onclick = function () { loadFrames(prev, { reason: '上一页' }); }; }
    if ((b = $('#pgNext'))) { b.onclick = function () { loadFrames(next, { reason: '下一页' }); }; }
  }

  /* 答完之后**只重画这一张卡**里和答案有关的部分（不重新拉图，图不闪） */
  function paint(file) {
    var card = document.querySelector('.ch-card[data-file="' + cssEsc(file) + '"]');
    if (!card) { return; }
    var row = rowOf(file);
    if (!row) { return; }
    var ans = ansOf(file);
    var manual = !!state.manual[file];

    $$('button.ch-opt[data-file="' + cssEsc(file) + '"]').forEach(function (b) {
      var qn = parseInt(b.dataset.q, 10);
      var v = parseInt(b.dataset.val, 10);
      var on = (qn === 1 && (v === 1) === ans.has) || (qn === 2 && ans.has && v === ans.count)
        || (qn === 3 && ans.has && v === ans.selected);
      b.classList.toggle('on', !!on);
    });

    // ② 答完要能点、③ 的候选按钮要跟着数量现长出来
    $$('button.ch-opt[data-q="2"]').forEach(function (b) {
      if (b.dataset.file === file) { b.disabled = !ans.has; }
    });
    var slot = card.querySelector('.ch-opts[data-slot="sel"]');
    if (slot) { slot.innerHTML = selectedBtns(file, ans); }

    var lab = card.querySelector('.ch-ans');
    if (lab) { lab.textContent = ansText(ans); }
    var meta = card.querySelector('.ch-meta');
    var old = card.querySelector('.ch-bad');
    if (old) { old.remove(); }
    var badge = mismatchBadge(row, ans);
    if (badge && meta) {
      var holder = card.querySelector('.ch-you');
      if (holder) { holder.insertAdjacentHTML('beforeend', badge); }
    }
    card.classList.toggle('dirty', manual);
    var zoom = parseInt(el.fZoom.value, 10) || 2;
    var bandImg = card.querySelector('.ch-band');
    var ovs = card.querySelector('.ch-ovs:not(.ch-ovs-full)');
    if (ovs) {
      // 画的是**当前答案**里的框（`ans`）—— 点图加/删之后立刻就能看见
      ovs.innerHTML = bandOverlay(row, ans);
    }
    var fullOvs = card.querySelector('.ch-ovs-full');
    if (fullOvs) {
      fullOvs.innerHTML = fullOverlay(row, ans, bandImg, zoom,
        row.full_w || 1280, row.full_h || 720);
    }
    card.classList.toggle('manual', manual);
  }

  // ------------------------------------------------------------ 点 band 图加/删框
  /* 换算基准是**图上实际显示的那个矩形**（不假设显示尺寸）：
   *   整帧 x = crop_rect.x + (鼠标 x - 图上左边界) / 图上宽度 * crop_rect.w
   * `crop_rect` 就是那张 band 图实际覆盖的整帧区域，所以横向覆盖的整帧像素宽
   * 正好是 `crop_rect[2]` —— 不需要再去猜 pad 或倒推 zoom。 */
  function hitBox(row, ans, fx, fy) {
    var boxes = ans.boxes || [];
    for (var k = 0; k < boxes.length; k++) {
      var b = boxes[k];
      if (fx >= b[0] && fx <= b[0] + b[2] && fy >= b[1] && fy <= b[1] + b[3]) { return k; }
    }
    return -1;
  }

  function medianRadius(row) {
    var r = ((row.detected && row.detected.icons) || []).map(function (i) {
      return i.r || 0;
    }).filter(function (v) { return v > 0; }).sort(function (a, b) { return a - b; });
    return r.length ? r[Math.floor(r.length / 2)] : 30;
  }

  /* 帧里已有框的**宽度中位数** —— 一个检测结果都没有时，用它当新框的大小 */
  function medianSize(row) {
    var ws = (row.boxes || []).map(function (b) { return b[2] || 0; })
      .filter(function (v) { return v > 0; }).sort(function (a, b) { return a - b; });
    return ws.length ? ws[Math.floor(ws.length / 2)] : 58;
  }

  /* 点击位置附近**检测到的那个圆**（没有就返回 null）。
   *
   * 为什么要"吸附"：band 图在卡片里只有 ~400 CSS px 宽（整条带 500 整帧像素），
   * 一个路标 ~58px 在屏幕上是 ~48px —— 手点上去最多只能准到 1~2 整帧像素，
   * 但"加/删一个框"的真实意图几乎总是"**这个路标**漏了/多报了"。
   * 所以点在某个检测到的圆附近（默认 1 个半径内）就直接用**那个圆的精确位置**，
   * 而不是把框落在手指点的那个像素上（那才是"加一个框不准"的来源）。 */
  function nearestIcon(row, fx, fy, tol) {
    var icons = (row.detected && row.detected.icons) || [];
    var best = null, bd = 1e9;
    icons.forEach(function (i) {
      var d = Math.sqrt(Math.pow(i.x - fx, 2) + Math.pow(i.y - fy, 2));
      if (d < bd) { bd = d; best = i; }
    });
    if (!best) { return null; }
    var lim = (tol || 0) || Math.max(medianRadius(row), 22);
    return bd <= lim ? best : null;
  }

  function boxClick(card, img, e) {
    var file = card.dataset.file;
    var row = rowOf(file);
    if (!row) { return; }
    var crop = cropOf(row) || [0, 0, 1280, 720];   // 没有 crop_rect 时按整帧算（不画框也会走这）
    var r = img.getBoundingClientRect();                 // 图上**实际显示**的矩形
    // 图上 (clientX - 左边界) / 显示宽度 = 落在整张 band 图横向的第几成，
    // 再乘上"这张图横向覆盖多少整帧像素"（= crop_rect[2]，就是它本身）就是整帧 x
    var fx = crop[0] + ((e.clientX - r.left) / (r.width || 1)) * (crop[2] || 1);
    var fy = crop[1] + ((e.clientY - r.top) / (r.height || 1)) * (crop[3] || 1);

    var ans = ansOf(file);
    var boxes = (ans.boxes || []).map(function (b) { return b.slice(); });
    var hit = hitBox({ boxes: boxes }, ans, fx, fy);

    if (hit >= 0) {
      var gone = boxes.splice(hit, 1)[0];
      toast('删掉第 ' + (hit + 1) + ' 个框（' + Math.round(gone[0]) + ','
        + Math.round(gone[1]) + ' 起 ' + Math.round(gone[2]) + '×'
        + Math.round(gone[3]) + '；共 ' + boxes.length + ' 个）', 'warn');
    } else {
      // 点在空白 = 加一个：**优先吸附到检测到的圆**，位置就用它的外接正方框
      var ic = nearestIcon(row, fx, fy);
      if (ic) {
        var rr = ic.r || 0;
        boxes.push([Math.round(ic.x - rr), Math.round(ic.y - rr),
                    Math.round(rr * 2), Math.round(rr * 2)]);
        toast('加了检测到的那一个圆（x=' + Math.round(ic.x) + ' y='
          + Math.round(ic.y) + '；共 ' + boxes.length + ' 个）', 'ok');
      } else {
        var s = Math.max(medianRadius(row) * 2, 20);     // 附近没有检测结果：按中位数大小放
        if (!boxes.length) { s = Math.max(medianSize(row), 20); }
        boxes.push([Math.round(fx - s / 2), Math.round(fy - s / 2),
                    Math.round(s), Math.round(s)]);
        toast('加了一个框（' + Math.round(fx) + ',' + Math.round(fy) + ' 起 '
          + Math.round(s) + '×' + Math.round(s) + '；附近没有检测到的圆，'
          + '所以放在你点的位置）—— 当前答案按「' + boxes.length + ' 个选项」走', 'ok');
      }
    }

    var next = normAns({ has: boxes.length > 0, count: boxes.length || 0,
                         selected: ans.selected, boxes: boxes, countEdited: true });
    // 加/删之后「有几个选项」跟着变成当前框数；原来没选的给个合理的默认
    if (boxes.length && !next.selected) { next.selected = 1; }
    state.manual[file] = boxes.length > 0;
    setAns(file, next);                  // setAns 会按 x 重新排序（见 sortBoxes）
  }

  // ------------------------------------------------------------ 放大 / 键盘
  function openLightbox(src, cap) {
    el.lightboxImg.src = src;
    el.lightboxCap.textContent = cap || '';
    el.lightbox.classList.remove('hidden');
  }

  function closeLightbox() {
    el.lightbox.classList.add('hidden');
    el.lightboxImg.removeAttribute('src');
  }

  function focusOffset(delta) {
    var cards = $$('.ch-card');
    var i = -1;
    cards.forEach(function (c, k) { if (c.dataset.file === state.cursor) { i = k; } });
    var j = Math.min(cards.length - 1, Math.max(0, (i < 0 ? 0 : i) + delta));
    if (cards[j]) {
      state.cursor = cards[j].dataset.file;
      cards[j].focus();
      cards[j].scrollIntoView({ block: 'center' });
    }
  }

  // ------------------------------------------------------------ 导出
  function exportJson() {
    if (!state.dataset) { toast('先选一个数据集', 'warn'); return; }
    var items = [];
    ((state.page && state.page.rows) || []).forEach(function (r) {
      var isManual = !!state.manual[r.file];
      var ans = state.answers[r.file];
      if (!ans) {
        if (r.status === 'reviewed' && !isManual) { return; }
        ans = detectAns(r);
      }
      items.push(itemFor(r, ans, isManual));
    });
    if (!items.length) { toast('没有可导出的帧', 'warn'); return; }
    var payload = {
      version: 1, dataset: state.dataset,
      exported: new Date().toISOString(),
      items: items
    };
    var blob = new Blob([JSON.stringify(payload, null, 1)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = state.dataset + '_choice.json';
    a.click();
    toast('已下载（' + items.length + ' 帧；形状和 POST /api/choice/answer 的 items 一致）', 'ok');
  }

  // ------------------------------------------------------------ 事件
  function bind() {
    el.dsSelect.addEventListener('change', function () {
      // 先把**旧数据集当前这一页**结掉再切（pendingFiles 用的是 state.page，旧的）
      var n = commitPage('切数据集', { quiet: true });
      var go = function () {
        state.dataset = el.dsSelect.value;
        state.answers = {};
        state.manual = {};
        state.autoCommitBroken = false;
        showError('');
        updatePending();
        loadFrames(0, { skipCommit: true });
      };
      if (n && n.then) { n.then(go); } else { go(); }
    });

    el.dsRefreshBtn.addEventListener('click', function () {
      loadDatasets(true).then(function () {
        loadFrames(0, { reason: '刷新' });
      });
    });
    el.refreshBtn.addEventListener('click', function () { loadFrames(0, { reason: '刷新' }); });

    el.fApply.addEventListener('click', function () { loadFrames(0, { reason: '换筛选' }); });
    el.fFilter.addEventListener('change', function () { loadFrames(0, { reason: '换筛选' }); });
    el.fLimit.addEventListener('change', function () { loadFrames(0, { reason: '改每页' }); });
    el.fSearch.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { loadFrames(0, { reason: '换筛选' }); }
    });
    el.fZoom.addEventListener('change', function () {
      if (state.page) { renderFrames(state.page); }   // 只换图，不重新拉数据
    });

    el.fMarkPage.addEventListener('click', function () {
      var p = commitPage('本页没问题');
      if (p && p.then) {
        p.then(function () { loadFrames(state.offset, { skipCommit: true }); });
      }
    });

    el.fUndoMark.addEventListener('click', function () {
      if (!state.dataset) { toast('先选一个数据集', 'warn'); return; }
      api('/api/choice/undo', { method: 'POST', body: { dataset: state.dataset } })
        .then(function (res) {
          var r = (res && res.result) || {};
          toast('已撤销 ' + (r.restored != null ? r.restored : '?')
            + ' 帧；现在共 ' + (r.reviewed_total != null ? r.reviewed_total : '?')
            + ' 帧已标注', 'ok');
          state.answers = {};
          state.manual = {};
          state.autoCommitBroken = false;
          showError('');
          loadFrames(state.offset, { skipCommit: true });
        }).catch(function (err) {
          showError('撤销失败（POST /api/choice/undo）：' + err.message);
          toast('撤销失败：' + err.message, 'warn');
        });
    });

    el.exportBtn.addEventListener('click', exportJson);

    // 三连问的按钮 + band 图点击（事件委托，卡片是整批重画的）
    el.chGrid.addEventListener('click', function (e) {
      var card = e.target.closest ? e.target.closest('.ch-card') : null;
      if (!card) { return; }
      var file = card.dataset.file;

      var ok = e.target.closest ? e.target.closest('.ch-ok') : null;
      if (ok) { answerOk(file); return; }

      var opt = e.target.closest ? e.target.closest('button.ch-opt') : null;
      if (opt) {
        var v = parseInt(opt.dataset.val, 10);
        var qn = parseInt(opt.dataset.q, 10);
        if (qn === 1) { answerHas(file, v === 1); }
        else if (qn === 2) { answerCount(file, v); }
        else { answerSelected(file, v); }
        return;
      }

      var band = e.target.closest ? e.target.closest('.ch-band') : null;
      if (band) { boxClick(card, band, e); return; }

      var full = e.target.closest ? e.target.closest('.ch-thumb') : null;
      if (full) {
        openLightbox(full.dataset.full || full.src,
          full.dataset.cap || file);
      }
    });

    el.chGrid.addEventListener('focusin', function (e) {
      var card = e.target.closest ? e.target.closest('.ch-card') : null;
      if (card) { state.cursor = card.dataset.file; }
    });

    el.lightbox.addEventListener('click', closeLightbox);

    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { closeLightbox(); return; }
      // 输入框 / 下拉框聚焦时**不要劫持按键**（不然在搜索框里打字会被吃掉）
      var tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'select' || tag === 'textarea') { return; }
      if (e.ctrlKey || e.metaKey || e.altKey) { return; }
      var asNum = parseInt(e.key, 10);
      // 数字键 = 答第②题「有几个选项」，范围跟着 `state.opts`（默认 2~4）
      if (/^[0-9]$/.test(e.key) && asNum >= state.opts.min && asNum <= state.opts.max) {
        e.preventDefault();
        if (state.cursor) { answerCount(state.cursor, asNum); }
      } else if (e.key === '0') {
        e.preventDefault();
        if (state.cursor) { answerHas(state.cursor, false); }
      } else if (e.key === ' ') {
        e.preventDefault();
        if (state.cursor) { answerOk(state.cursor); }
      } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault(); focusOffset(1);
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault(); focusOffset(-1);
      }
    });
  }

  function boot() {
    bind();
    updatePending();
    loadDatasets(false).then(function () {
      if (!state.dataset) { return; }
      loadFrames(0, { skipCommit: true });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
