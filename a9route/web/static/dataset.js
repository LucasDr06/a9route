/* a9route 数据集体检页 —— 前端逻辑
 *
 * 原生 ES2020，无构建步骤、无外部依赖（完全离线可用）。
 * 只依赖这些接口（都在 web/dataset.py 里）：
 *   GET  /api/dataset/list        数据集列表
 *   GET  /api/dataset/audit       体检结论
 *   GET  /api/dataset/frames      一页帧（可带 recheck 重算对比）
 *   GET  /api/dataset/image       整帧缩略图 / 按键框放大图
 *   POST /api/dataset/corrections 写回标注（和命令行的 train apply 同一份逻辑）
 *
 * ## 界面上最要紧的那件事
 *
 * 缩略图上画**两套框**：
 *   * 实线 = 数据集里的**标注框**（labels/*.txt）
 *   * 虚线 = **当前** config.json 的标定框
 * 两者必须完全重合。不重合 = "数据集和标定不对应"，而那正是这个页面存在的理由。
 * 框用**百分比**定位（x/imgW*100%），所以页面怎么缩放都对得上。
 */
'use strict';

(function () {
  var $ = function (s) { return document.querySelector(s); };
  var $$ = function (s) { return Array.prototype.slice.call(document.querySelectorAll(s)); };

  var el = {
    banner: $('#connBanner'), toasts: $('#toasts'),
    dsSelect: $('#dsSelect'), dsRefreshBtn: $('#dsRefreshBtn'),
    dsAuditBtn: $('#dsAuditBtn'), dsHint: $('#dsHint'),
    dsCounts: $('#dsCounts'), dsFindings: $('#dsFindings'),
    dsAuditPill: $('#dsAuditPill'), dsAuditNote: $('#dsAuditNote'),
    dsCalibBox: $('#dsCalibBox'), dsCalib: $('#dsCalib'),
    dsExportBtn: $('#dsExportBtn'), dsApplyBtn: $('#dsApplyBtn'),

    fSplit: $('#fSplit'), fReason: $('#fReason'), fCls: $('#fCls'),
    fStatus: $('#fStatus'), fLimit: $('#fLimit'), fSearch: $('#fSearch'),
    fRecheck: $('#fRecheck'), fProblemOnly: $('#fProblemOnly'),
    fShowCalib: $('#fShowCalib'), fApply: $('#fApply'),
    fMarkPage: $('#fMarkPage'), fUndoMark: $('#fUndoMark'),
    dsGrid: $('#dsGrid'), dsPager: $('#dsPager'),
    dsFramePill: $('#dsFramePill'), dsFrameNote: $('#dsFrameNote'),

    lightbox: $('#lightbox'), lightboxImg: $('#lightboxImg'),
    lightboxCap: $('#lightboxCap')
  };

  var state = {
    dataset: '', offset: 0, limit: 24, page: null,
    corrections: {},      // file -> {boxes: [...], status: 'reviewed'}（人改过的）
    marked: {},           // file -> true（本页翻过去时已记成"已复核"）
    cursor: '',           // 键盘操作的那一行
    busy: false,
    committing: null      // 正在提交的 Promise（防止并发翻页重复提交）
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

  function banner(text) {
    if (!text) { el.banner.classList.add('hidden'); return; }
    el.banner.textContent = text;
    el.banner.classList.remove('hidden');
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
    return '/api/dataset/image?' + q(p);
  }

  // ------------------------------------------------------------ 数据集列表
  function loadDatasets(keep) {
    return api('/api/dataset/list').then(function (data) {
      var list = (data && data.datasets) || [];
      el.dsSelect.innerHTML = '';
      if (!list.length) {
        var o = document.createElement('option');
        o.value = ''; o.textContent = '（还没有数据集 —— 先跑 a9route train build）';
        el.dsSelect.appendChild(o);
        el.dsHint.textContent = '数据集目录：' + ((data && data.root) || '?');
        return;
      }
      list.forEach(function (d) {
        var opt = document.createElement('option');
        opt.value = d.name;
        var isChoice = (d.task === 'choice');
        opt.textContent = d.name + '  (' + d.frames + ' 帧'
          + (d.reviewed ? '，已复核 ' + d.reviewed : '')
          + (d.videos && d.videos.length ? '，' + d.videos.length + ' 段录像' : '')
          // ⚠️ **选路数据集要显眼标出来**：这一页是「刹车/氮气」专用的，
          // 在这页上对选路数据集点「写回修正」，会把按键的框写进选路标注里
          // （2026-09-15 真出过：42 帧被覆盖）。后端现在会拒绝，但**别让人先选错**。
          + (isChoice ? '  ← 选路，用 /choice 页' : '')
          + ')';
        el.dsSelect.appendChild(opt);
      });
      if (keep && list.some(function (d) { return d.name === state.dataset; })) {
        el.dsSelect.value = state.dataset;
      }
      state.dataset = el.dsSelect.value;
      var d0 = list.filter(function (d) { return d.name === state.dataset; })[0] || {};
      el.dsHint.textContent = '数据集目录：' + ((data && data.root) || '?')
        + '；模型目录：' + ((data && data.models_dir) || '?')
        + (d0.has_split ? '' : '  ⚠️ 这个数据集还没 `train split`（只有 pool）')
        + (d0.task === 'choice'
           ? '  ⚠️ 这是**选路**数据集：本页的写回只认刹车/氮气，会被拒绝 —— '
             + '选路请去 /choice'
           : '');
    }).catch(function (err) {
      banner('后端连不上：' + err.message);
      el.dsHint.textContent = '读取失败：' + err.message;
    });
  }

  // ------------------------------------------------------------ 体检
  var LEVEL_TXT = { ok: '通过', warn: '注意', error: '问题' };

  function runAudit() {
    if (!state.dataset) { toast('先选一个数据集', 'warn'); return; }
    el.dsAuditPill.textContent = '体检中…';
    el.dsAuditPill.className = 'pill pill-running';
    el.dsFindings.innerHTML = '<p class="muted">正在体检（会抽样解图重算判据）…</p>';
    api('/api/dataset/audit?' + q({ name: state.dataset, recheck: 1, limit: 400 }))
      .then(function (a) {
        state.audit = a;
        renderAudit(a);
        el.dsAuditPill.textContent = a.ok
          ? (a.problems ? a.problems + ' 个注意' : '全部通过')
          : a.problems + ' 个问题';
        el.dsAuditPill.className = 'pill ' + (a.ok ? (a.problems ? 'pill-running' : 'pill-done') : 'pill-error');
      })
      .catch(function (err) {
        el.dsFindings.innerHTML = '<p class="alert alert-error">体检失败：'
          + esc(err.message) + '</p>';
        el.dsAuditPill.textContent = '失败';
        el.dsAuditPill.className = 'pill pill-error';
      });
  }

  function renderAudit(a) {
    var c = a.counts || {};
    var order = ['图片', 'meta 行', '分布', '已复核', '纯背景帧', '双键同时按下',
                 '框·刹车按下', '框·氮气按下', '画面尺寸'];
    var seen = {};
    var html = '';
    order.forEach(function (k) {
      if (c[k] === undefined) { return; }
      seen[k] = 1;
      html += card(k, c[k]);
    });
    Object.keys(c).forEach(function (k) {
      if (!seen[k]) { html += card(k, c[k]); }
    });
    el.dsCounts.innerHTML = html;

    var f = (a.findings || []).map(function (x) {
      var samples = (x.samples || []).length
        ? '<div class="ds-samples">' + x.samples.map(function (s) {
            return '<code>' + esc(s) + '</code>';
          }).join(' ') + '</div>' : '';
      return '<div class="finding finding-' + esc(x.level) + '">'
        + '<span class="finding-tag">' + esc(LEVEL_TXT[x.level] || x.level) + '</span>'
        + '<span class="finding-kind">' + esc(x.kind) + '</span>'
        + '<div class="finding-body"><b>' + esc(x.title) + '</b>'
        + (x.detail ? '<div class="finding-detail">' + esc(x.detail) + '</div>' : '')
        + samples + '</div></div>';
    }).join('');
    el.dsFindings.innerHTML = f || '<p class="muted">没有结论。</p>';

    var cal = a.calibration || {};
    el.dsCalib.innerHTML = Object.keys(cal).sort().map(function (k) {
      return '<div class="ds-calib-row"><span>' + esc(k) + '</span><code>'
        + esc(JSON.stringify(cal[k])) + '</code></div>';
    }).join('');
    el.dsAuditNote.textContent = a.rechecked
      ? '（抽检重算 ' + a.rechecked + ' 帧；信号漂移 '
        + ((a.drift && a.drift.signal_drift) || 0) + ' / 标注漂移 '
        + ((a.drift && a.drift.label_drift) || 0) + '）'
      : '（没做重算对比）';
  }

  function card(k, v) {
    return '<div class="ds-count"><span class="ds-count-k">' + esc(k)
      + '</span><span class="ds-count-v">' + esc(String(v)) + '</span></div>';
  }

  // ------------------------------------------------------------ 翻页 = 本页已复核
  //
  // 用户的原话："有错误我会标定修改，没有错误的我就直接不点没问题了。
  // 如果我翻过这一页就把当前页的数据作为已复核。"
  //
  // 所以**离开当前页时**自动做两件事：
  //   * 人改过的帧 -> `/corrections`（写标注文件，按当前标定对齐框）；
  //   * 没改过的帧 -> `/mark_reviewed`（**只改 status，绝不碰标注文件**）。
  //
  // 为什么两个接口必须分开：`corrections` 会用**当前标定**重写标注框 ——
  // 若数据集是用另一套标定建的，翻一页就把整页标注悄悄挪位，
  // 连"数据集和标定对不上"这个体检结论都会被抹掉。翻页只是"我看过了"，
  // 不该有这个副作用。
  function pendingWork() {
    var d = state.page;
    if (!d || !d.rows || !d.rows.length) { return null; }
    var edited = [], untouched = [], skipped = [];
    d.rows.forEach(function (r) {
      var c = state.corrections[r.file];
      if (c) {
        edited.push({ file: r.file, boxes: c.boxes.slice(), status: 'reviewed' });
      } else if (r.status !== 'reviewed' && !state.marked[r.file]) {
        // 框和标定对不上的帧**不自动记成已复核** —— 那正是体检要报的错，
        // 不该被"翻过去"这个动作抹掉。它得靠改标定/重建数据集来解决。
        if ((r.flags || []).indexOf('calib_mismatch') >= 0) {
          skipped.push(r.file);
        } else {
          untouched.push(r.file);
        }
      }
    });
    if (!edited.length && !untouched.length && !skipped.length) { return null; }
    return { dataset: d.dataset, edited: edited, untouched: untouched,
             skipped: skipped, rows: d.rows };
  }

  function commitCurrentPage(reason, opts) {
    opts = opts || {};
    if (state.committing) { return state.committing; }
    var work = pendingWork();
    if (!work) { return Promise.resolve(0); }
    var jobs = [];
    if (work.edited.length) {
      jobs.push(api('/api/dataset/corrections', {
        method: 'POST',
        body: { version: 1, dataset: work.dataset, items: work.edited }
      }));
    } else {
      jobs.push(Promise.resolve(null));
    }
    if (work.untouched.length) {
      jobs.push(api('/api/dataset/mark_reviewed', {
        method: 'POST',
        body: { dataset: work.dataset, files: work.untouched }
      }));
    } else {
      jobs.push(Promise.resolve(null));
    }
    state.committing = Promise.all(jobs).then(function (res) {
      var fixed = work.edited.length;
      var marked = (res[1] && res[1].result && res[1].result.changed) || 0;
      // 本地也更新，避免同一页被重复提交
      work.rows.forEach(function (r) { r.status = 'reviewed'; });
      work.untouched.forEach(function (f) { state.marked[f] = true; });
      work.edited.forEach(function (it) { delete state.corrections[it.file]; });
      updateDirty();
      var bits = [];
      if (fixed) { bits.push('改 ' + fixed + ' 帧'); }
      if (marked) { bits.push('记 ' + marked + ' 帧已复核'); }
      if (bits.length) {
        toast((reason ? reason + '：' : '') + bits.join('，') + ' ✓', 'ok');
      }
      if (work.skipped.length) {
        toast(work.skipped.length + ' 帧因「框和标定对不上」没记为已复核 —— '
          + '先跑 train audit / 看体检结论', 'warn');
      }
      return bits.length;
    }).catch(function (err) {
      toast('保存失败：' + err.message + '（这一页没记上，重翻一次即可）', 'error');
      return 0;
    }).then(function (n) {
      state.committing = null;
      return n;
    });
    return state.committing;
  }

  // ------------------------------------------------------------ 逐帧
  function loadFrames(offset, opts) {
    opts = opts || {};
    if (!state.dataset) { return; }
    // 离开当前页之前，先把这一页结掉（**用户要的"翻过就算复核"**）
    if (!opts.skipCommit) {
      var n = commitCurrentPage(opts.reason || '翻页');
      if (n && n.then) {
        return n.then(function () { return loadFrames(offset, { skipCommit: true }); });
      }
    }
    if (offset !== undefined) { state.offset = Math.max(0, offset); }
    state.limit = parseInt(el.fLimit.value, 10) || 24;
    var params = {
      name: state.dataset, offset: state.offset, limit: state.limit,
      split: el.fSplit.value, reason: el.fReason.value, cls: el.fCls.value,
      status: el.fStatus.value, search: el.fSearch.value.trim(),
      recheck: el.fRecheck.checked ? 1 : 0,
      only: el.fProblemOnly.checked ? 'problem' : ''
    };
    el.dsFramePill.textContent = '加载中…';
    el.dsFramePill.className = 'pill pill-running';
    api('/api/dataset/frames?' + q(params)).then(function (d) {
      state.page = d;
      renderFrames(d);
    }).catch(function (err) {
      el.dsGrid.innerHTML = '<p class="alert alert-error">读帧失败：'
        + esc(err.message) + '</p>';
      el.dsFramePill.textContent = '失败';
      el.dsFramePill.className = 'pill pill-error';
    });
  }

  var FLAG_TXT = {
    // 人复核过的帧和启发式不同 = **正常**（人就是在改启发式的错）-> 蓝色提示，别打红
    human_fix: ['人工修正过（和启发式不同，正常）', 'i'],
    label_drift: ['标注和当前判据不一致', 'd'],
    signal_drift: ['信号和 meta 里存的不一样', 'd'],
    calib_mismatch: ['框和标定对不上', 'e'],
    missing_file: ['图不见了', 'e'],
    undecodable: ['图解不开', 'e']
  };

  function boxOverlay(box, imgW, imgH, cls, title) {
    if (!imgW || !imgH) { return ''; }
    var l = (box[0] / imgW) * 100, t = (box[1] / imgH) * 100;
    var w = (box[2] / imgW) * 100, h = (box[3] / imgH) * 100;
    return '<div class="ds-box ' + cls + '" style="left:' + l.toFixed(3) + '%;top:'
      + t.toFixed(3) + '%;width:' + w.toFixed(3) + '%;height:' + h.toFixed(3)
      + '%" title="' + esc(title) + '"></div>';
  }

  function renderFrames(d) {
    var cal = d.calibration || {};
    var showCalib = el.fShowCalib.checked;
    var html = '';
    (d.rows || []).forEach(function (r) {
      var imgW = r.width || 1280, imgH = r.height || 720;
      var boxes = '';
      // 虚线：当前标定框（在最底层）
      if (showCalib) {
        boxes += boxOverlay(cal.brake_key_box, imgW, imgH, 'calib', '标定：刹车键');
        boxes += boxOverlay(cal.nitro_key_box, imgW, imgH, 'calib', '标定：氮气键');
      }
      // 实线：数据集里的标注框
      (r.labels || []).forEach(function (b) {
        boxes += boxOverlay(b.box, imgW, imgH,
          'label-' + (b.name === 'nitro_pressed' ? 'nitro' : 'brake'),
          '标注：' + (b.zh || b.name));
      });

      var flags = (r.flags || []).map(function (fl) {
        var t = FLAG_TXT[fl] || [fl, 'd'];
        return '<span class="ds-flag ds-flag-' + t[1] + '">' + esc(t[0]) + '</span>';
      }).join('');

      var now = r.now;
      var cmp = '';
      if (now) {
        var same = (now.brake === r.labels.some(function (b) { return b.name === 'brake_pressed'; }))
          && (now.nitro === r.labels.some(function (b) { return b.name === 'nitro_pressed'; }));
        cmp = '<div class="ds-cmp' + (same ? '' : ' diff') + '">'
          + '现在重算：刹车 ' + (now.brake ? '按下' : '没按')
          + '（余量 ' + now.brake_margin + '）／氮气 '
          + (now.nitro ? '按下' : '没按') + '（余量 ' + now.nitro_margin + '）'
          + (same ? ' ✓ 和标注一致' : ' ✗ **和标注不一致**') + '</div>';
      }

      var sig = r.signals || {};
      var lbl = (r.labels || []).length
        ? r.labels.map(function (b) { return b.zh || b.name; }).join(' + ')
        : '（未按）';
      var rzh = (d.facet.reasons_zh || {})[r.reason] || r.reason;

      html += '<div class="ds-card" data-file="' + esc(r.file) + '" tabindex="0">'
        + '<div class="ds-shot">'
        + '<img loading="lazy" src="' + esc(imgUrl(r.file, 'full', { w: 480 })) + '"'
        + ' data-full="' + esc(imgUrl(r.file, 'full', { w: 1280 })) + '"'
        + ' data-cap="' + esc(r.file + '  ' + r.video + ' @ ' + r.t + 's') + '" alt="" />'
        + boxes + '</div>'
        + '<div class="ds-crops">'
        + '<img loading="lazy" src="' + esc(imgUrl(r.file, 'crop', { which: 'brake', zoom: 2 }))
        + '" alt="刹车键" title="刹车键（放大 2×）" />'
        + '<img loading="lazy" src="' + esc(imgUrl(r.file, 'crop', { which: 'nitro', zoom: 2 }))
        + '" alt="氮气键" title="氮气键（放大 2×）" />'
        + '</div>'
        + '<div class="ds-meta">'
        + '<div class="ds-file">' + esc(r.file) + '</div>'
        + '<div class="ds-sub">' + esc(r.split) + ' · ' + esc(r.video)
        + ' @ ' + r.t + 's · 帧 ' + r.frame + ' · ' + esc(rzh) + '</div>'
        + '<div class="ds-sub">信号：红 ' + (sig.nitro_red != null ? sig.nitro_red : '—')
        + ' · 圈内外 ' + (sig.brake_ring != null ? sig.brake_ring : '—')
        + ' · 白 ' + (sig.brake_white != null ? sig.brake_white : '—') + '</div>'
        + '<div class="ds-labels">标注：<b>' + esc(lbl) + '</b>'
        + '<span class="pill ds-status '
        + (r.status === 'reviewed' ? 'pill-done' : '') + '">'
        + (r.status === 'reviewed' ? '已复核' : '未复核') + '</span></div>'
        + cmp + flags
        + '<div class="ds-buttons">'
        + '<button class="btn btn-sm" data-act="brake">刹车 B</button>'
        + '<button class="btn btn-sm" data-act="nitro">氮气 N</button>'
        + '<button class="btn btn-sm" data-act="ok">没问题</button>'
        + '</div></div></div>';
    });
    el.dsGrid.innerHTML = html || '<p class="muted">这一页没有帧（换个筛选条件试试）。</p>';

    // 把已改动的高亮出来
    Object.keys(state.corrections).forEach(function (f) { paint(f); });

    var shown = (d.rows || []).length;
    el.dsFramePill.textContent = shown + ' / ' + (d.total_known === false ? '?' : d.total) + ' 帧';
    el.dsFramePill.className = 'pill';
    el.dsFrameNote.textContent = '（数据集共 ' + d.all_frames + ' 帧'
      + (d.recheck ? '，已重算对比' : '，未重算')
      + (d.total_known === false ? '；"只看有问题的"模式下总数要扫完才知道' : '')
      + '）';
    renderPager(d);
  }

  function renderPager(d) {
    var prev = Math.max(0, state.offset - state.limit);
    var next = d.next_offset;
    var hasNext = d.total_known === false ? (d.rows || []).length > 0
      : next < d.total;
    el.dsPager.innerHTML =
      '<button class="btn btn-sm" id="pgFirst" ' + (state.offset <= 0 ? 'disabled' : '')
      + '>« 首页</button>'
      + '<button class="btn btn-sm" id="pgPrev" ' + (state.offset <= 0 ? 'disabled' : '')
      + '>‹ 上一页</button>'
      + '<span class="muted">第 ' + (Math.floor(state.offset / state.limit) + 1)
      + ' 页（第 ' + (state.offset + 1) + ' 帧起）</span>'
      + '<button class="btn btn-sm" id="pgNext" ' + (hasNext ? '' : 'disabled')
      + '>下一页 ›</button>';
    var b;
    if ((b = $('#pgFirst'))) { b.onclick = function () { loadFrames(0); }; }
    if ((b = $('#pgPrev'))) { b.onclick = function () { loadFrames(prev); }; }
    if ((b = $('#pgNext'))) { b.onclick = function () { loadFrames(next); }; }
  }

  // ------------------------------------------------------------ 改标注
  function currentBoxes(file) {
    var c = state.corrections[file];
    if (c) { return c.boxes; }
    var row = (state.page && state.page.rows || []).filter(function (r) {
      return r.file === file;
    })[0];
    return row ? (row.labels || []).map(function (b) { return b.name; }) : [];
  }

  function toggle(file, cls) {
    var boxes = currentBoxes(file).slice();
    var i = boxes.indexOf(cls);
    if (i >= 0) { boxes.splice(i, 1); } else { boxes.push(cls); }
    state.corrections[file] = { boxes: boxes, status: 'reviewed' };
    paint(file);
    updateDirty();
  }

  // "没问题" = 只记"我看过"，**不写标注文件**（走 mark_reviewed 那条安全路径）。
  // 其实翻页时也会自动记，这个按钮是给"我想现在就记下这一帧"用的。
  function markOk(file) {
    delete state.corrections[file];
    state.marked[file] = true;
    paint(file);
    updateDirty();
  }

  function paint(file) {
    var card = document.querySelector('.ds-card[data-file="' + cssEsc(file) + '"]');
    if (!card) { return; }
    var edited = !!state.corrections[file];
    var seen = !!state.marked[file];
    card.classList.toggle('dirty', edited);
    card.classList.toggle('seen', seen && !edited);
    var pill = card.querySelector('.ds-status');
    if (pill && (edited || seen)) {
      pill.textContent = '已复核';
      pill.className = 'pill ds-status pill-done';
    }
    var lab = card.querySelector('.ds-labels b');
    if (lab && edited) {
      var boxes = state.corrections[file].boxes;
      var zh = { brake_pressed: '刹车按下', nitro_pressed: '氮气按下' };
      lab.textContent = boxes.length
        ? boxes.map(function (b) { return zh[b] || b; }).join(' + ')
        : '（未按）';
    }
  }

  function cssEsc(s) {
    return String(s).replace(/["\\]/g, '\\$&');
  }

  function updateDirty() {
    var n = Object.keys(state.corrections).length + Object.keys(state.marked).length;
    el.dsApplyBtn.textContent = n ? ('保存本页 (' + n + ')') : '保存本页';
    el.dsExportBtn.disabled = !Object.keys(state.corrections).length;
  }

  function payload() {
    return {
      version: 1, dataset: state.dataset,
      exported: new Date().toISOString(),
      items: Object.keys(state.corrections).map(function (f) {
        return { file: f, boxes: state.corrections[f].boxes.slice(),
                 status: 'reviewed' };
      })
    };
  }

  function applyCorrections() {
    // 和翻页走同一条路：改过的写标注文件，没改过的只记"已复核"
    el.dsApplyBtn.disabled = true;
    var p = payload();
    var doIt = (Object.keys(state.marked).length || !p.items.length)
      ? commitCurrentPage('已保存')
      : api('/api/dataset/corrections', { method: 'POST', body: p })
          .then(function (res) {
            var r = res.result || {};
            toast('写回 ' + r.reviewed + ' 帧（改动 ' + r.changed + ' 帧'
              + (r.skipped ? '，跳过 ' + r.skipped : '') + '）✓', 'ok');
            state.corrections = {};
            updateDirty();
          });
    doIt.catch(function (err) {
      toast('保存失败：' + err.message, 'error');
    }).then(function () {
      el.dsApplyBtn.disabled = false;
      loadFrames(state.offset, { skipCommit: true });
    });
  }

  function exportCorrections() {
    var p = payload();
    var blob = new Blob([JSON.stringify(p, null, 1)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = state.dataset + '_corrections.json';
    a.click();
    toast('已下载（命令行也能用：a9route train apply <这个文件>）', 'ok');
  }

  // ------------------------------------------------------------ 放大 / 键盘
  function openLightbox(src, cap) {
    el.lightboxImg.src = src;
    el.lightboxCap.textContent = cap || '';
    el.lightbox.classList.remove('hidden');
  }

  function focusOffset(delta) {
    var cards = $$('.ds-card');
    var i = cards.findIndex(function (c) { return c.dataset.file === state.cursor; });
    var j = Math.min(cards.length - 1, Math.max(0, (i < 0 ? 0 : i) + delta));
    if (cards[j]) {
      state.cursor = cards[j].dataset.file;
      cards[j].focus();
      cards[j].scrollIntoView({ block: 'center' });
    }
  }

  function bind() {
    el.dsSelect.addEventListener('change', function () {
      // 先把**旧数据集当前这一页**结掉，再切 —— 所以这里不能提前改 state.dataset
      // （`pendingWork()` 用的是 `state.page.dataset`，那是旧的那份）
      var n = commitCurrentPage('切数据集');
      var go = function () {
        state.dataset = el.dsSelect.value;
        state.corrections = {};
        state.marked = {};
        updateDirty();
        runAudit();
        loadFrames(0, { skipCommit: true });
      };
      if (n && n.then) { n.then(go); } else { go(); }
    });
    el.dsRefreshBtn.addEventListener('click', function () {
      loadDatasets(true).then(function () { runAudit(); loadFrames(0, { reason: '刷新' }); });
    });
    el.dsAuditBtn.addEventListener('click', runAudit);
    el.fApply.addEventListener('click', function () {
      loadFrames(0, { reason: '刷新' });
    });
    el.fMarkPage.addEventListener('click', function () {
      var p = commitCurrentPage('本页没问题');
      if (p && p.then) {
        p.then(function () { loadFrames(state.offset, { skipCommit: true }); });
      }
    });
    el.fUndoMark.addEventListener('click', function () {
      api('/api/dataset/undo_mark', {
        method: 'POST', body: { dataset: state.dataset }
      }).then(function (res) {
        var r = res.result || {};
        toast('已撤销 ' + r.restored + ' 帧（该批共 ' + r.files + ' 帧）；'
          + '现在共 ' + r.reviewed_total + ' 帧已复核', 'ok');
        state.marked = {};
        loadFrames(state.offset, { skipCommit: true });
      }).catch(function (err) {
        toast('撤销失败：' + err.message, 'warn');
      });
    });
    el.fLimit.addEventListener('change', function () {
      loadFrames(0, { reason: '改每页' });
    });
    el.fProblemOnly.addEventListener('change', function () {
      loadFrames(0, { reason: '换筛选' });
    });
    el.fShowCalib.addEventListener('change', function () {
      if (state.page) { renderFrames(state.page); }
    });
    el.fSearch.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { loadFrames(0); }
    });
    el.dsApplyBtn.addEventListener('click', applyCorrections);
    el.dsExportBtn.addEventListener('click', exportCorrections);

    el.dsGrid.addEventListener('click', function (e) {
      var card = e.target.closest ? e.target.closest('.ds-card') : null;
      if (!card) { return; }
      var file = card.dataset.file;
      var btn = e.target.closest ? e.target.closest('button[data-act]') : null;
      if (btn) {
        var act = btn.dataset.act;
        if (act === 'brake') { toggle(file, 'brake_pressed'); }
        else if (act === 'nitro') { toggle(file, 'nitro_pressed'); }
        else { markOk(file); }
        return;
      }
      var img = e.target.closest ? e.target.closest('img') : null;
      if (img && img.dataset.full) {
        openLightbox(img.dataset.full, img.dataset.cap || file);
      } else if (img) {
        openLightbox(img.src, img.alt + ' —— ' + file);
      }
    });
    el.dsGrid.addEventListener('focusin', function (e) {
      var card = e.target.closest ? e.target.closest('.ds-card') : null;
      if (card) { state.cursor = card.dataset.file; }
    });

    el.lightbox.addEventListener('click', function () {
      el.lightbox.classList.add('hidden');
      el.lightboxImg.removeAttribute('src');
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        el.lightbox.classList.add('hidden');
        return;
      }
      var tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'select' || tag === 'textarea') { return; }
      if (!state.cursor) { return; }
      if (e.key === 'b' || e.key === 'B') { e.preventDefault(); toggle(state.cursor, 'brake_pressed'); }
      else if (e.key === 'n' || e.key === 'N') { e.preventDefault(); toggle(state.cursor, 'nitro_pressed'); }
      else if (e.key === ' ') { e.preventDefault(); markOk(state.cursor); }
      else if (e.key === 'ArrowDown') { e.preventDefault(); focusOffset(1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); focusOffset(-1); }
    });
  }

  function boot() {
    bind();
    updateDirty();
    loadDatasets(false).then(function () {
      if (!state.dataset) { return; }
      runAudit();
      loadFrames(0);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
