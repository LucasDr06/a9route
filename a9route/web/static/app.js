/* a9route Web 窗口 —— 前端逻辑
 *
 * 原生 ES2020，无构建步骤、无外部依赖（完全离线可用）。
 * 只依赖后端这几个接口：
 *   GET  /api/version  /api/videos  /api/jobs  /api/status  /api/route
 *   POST /api/analyze                      （上传文件 或 给服务器上的路径）
 *   GET  /media/<job>/<name>               （某百分点截图）
 */
'use strict';

(function () {
  var $ = function (sel) { return document.querySelector(sel); };
  var $$ = function (sel) { return Array.prototype.slice.call(document.querySelectorAll(sel)); };

  var el = {
    banner: $('#connBanner'),
    toasts: $('#toasts'),
    videoSelect: $('#videoSelect'),
    videoScanBtn: $('#videoScanBtn'),
    videoAnalyzeBtn: $('#videoAnalyzeBtn'),
    videoHint: $('#videoHint'),
    paramsBtn: $('#paramsBtn'),
    stopWatchBtn: $('#stopWatchBtn'),
    jobPill: $('#jobPill'),

    dropZone: $('#dropZone'),
    pickFileBtn: $('#pickFileBtn'),
    videoFile: $('#videoFile'),
    everySel: $('#everySel'),
    maxSecInput: $('#maxSecInput'),

    videoFill: $('#videoFill'),
    videoStatus: $('#videoStatus'),
    fatalBox: $('#fatalBox'),

    videoResult: $('#videoResult'),
    videoRouteText: $('#videoRouteText'),
    videoRouteNotes: $('#videoRouteNotes'),
    videoRouteStat: $('#videoRouteStat'),
    videoCopyBtn: $('#videoCopyBtn'),
    videoDownloadBtn: $('#videoDownloadBtn'),
    videoShots: $('#videoShots'),

    paramsModal: $('#paramsModal'),
    paramsModalClose: $('#paramsModalClose'),
    paramsDump: $('#paramsDump'),

    lightbox: $('#lightbox'),
    lightboxImg: $('#lightboxImg'),
    lightboxCap: $('#lightboxCap')
  };

  var state = { job: '', timer: null, route: '', flat: '', notes: '', name: '',
                version: null, lastMsg: '' };

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
    setTimeout(function () { d.remove(); }, 4200);
  }

  function fatal(text) {
    if (!text) { el.fatalBox.classList.add('hidden'); el.fatalBox.textContent = ''; return; }
    el.fatalBox.textContent = text;
    el.fatalBox.classList.remove('hidden');
  }

  function api(path, opts) {
    opts = opts || {};
    var init = { method: opts.method || 'GET', headers: {} };
    if (opts.body instanceof FormData) {
      init.body = opts.body;
    } else if (opts.body !== undefined) {
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

  function setProgress(text, ratio) {
    state.lastMsg = text || '';
    el.videoStatus.textContent = state.lastMsg;
    if (ratio == null) { return; }
    el.videoFill.style.width = Math.max(2, Math.min(100, ratio * 100)) + '%';
  }

  function setPill(text, kind) {
    el.jobPill.textContent = text;
    el.jobPill.className = 'pill' + (kind ? ' pill-' + kind : '');
  }

  // ------------------------------------------------------------ 视频列表
  function scanVideos() {
    el.videoHint.textContent = '正在扫描视频目录…';
    api('/api/videos').then(function (data) {
      var list = (data && data.videos) || [];
      var prev = el.videoSelect.value;
      el.videoSelect.innerHTML = '';
      if (!list.length) {
        var o = document.createElement('option');
        o.value = '';
        o.textContent = '（没找到视频 —— 拖文件进来，或设 A9ROUTE_VIDEO_DIR）';
        el.videoSelect.appendChild(o);
      }
      list.forEach(function (v) {
        var opt = document.createElement('option');
        opt.value = v.path;
        opt.textContent = v.name + (v.size_mb ? '  (' + v.size_mb + ' MB)' : '');
        el.videoSelect.appendChild(opt);
      });
      if (prev) {
        var hit = list.some(function (v) { return v.path === prev; });
        if (hit) { el.videoSelect.value = prev; }
      }
      var dirs = (data && data.dirs) || [];
      el.videoHint.textContent = list.length
        ? (list.length + ' 个视频；目录：' + dirs.join('  '))
        : ('没找到视频。目录：' + (dirs.join('  ') || '(无)'));
    }).catch(function (err) {
      el.videoHint.textContent = '扫描失败：' + err.message;
      showBanner('后端连不上：' + err.message);
    });
  }

  function showBanner(text) {
    el.banner.textContent = text;
    el.banner.classList.remove('hidden');
  }

  // ------------------------------------------------------------ 分析
  function startWatch(job, name) {
    state.job = job;
    state.name = name || '';
    state.route = '';
    state.flat = '';
    if (state.timer) { clearTimeout(state.timer); }
    el.videoResult.classList.add('hidden');
    fatal('');
    setPill('分析中', 'running');
    setProgress('已提交，开始分析…', 0);
    poll();
  }

  function stopWatch(silent) {
    if (state.timer) { clearTimeout(state.timer); state.timer = null; }
    if (!silent) { setProgress('已停止跟随（后台分析还在跑，可重新点"开始分析"）', null); }
  }

  function poll() {
    if (!state.job) { return; }
    api('/api/status?job=' + encodeURIComponent(state.job)).then(function (job) {
      if (job.state === 'error') {
        setPill('失败', 'error');
        setProgress('失败：' + (job.error || '未知错误'), 1);
        fatal('分析失败：' + (job.error || '未知错误'));
        return;
      }
      var msg = job.message || '分析中…';
      // 进度条：从"覆盖 N 个百分点"里读，读不到就用帧数粗略推进
      var m = /覆盖\s*(\d+)\s*个百分点/.exec(msg);
      var ratio = m ? Math.min(1, parseInt(m[1], 10) / 100) : null;
      setProgress(msg, ratio);
      if (job.state === 'done') {
        setPill('完成', 'done');
        setProgress(msg, 1);
        render(job);
        return;
      }
      state.timer = setTimeout(poll, 900);
    }).catch(function (err) {
      setProgress('轮询失败：' + err.message, null);
      showBanner('后端连不上：' + err.message);
    });
  }

  function render(job) {
    var rowHtml = '';
    (job.shots || []).forEach(function (s) {
      var intents = (s.intents || []).map(function (it) {
        return it.kind + ' ' + it.op + (it.note ? '（' + it.note + '）' : '');
      }).join(' / ');
      var cap = s.percent + '% @ ' + s.t + 's  操作 ' + (s.op || '—')
        + '  判据 ' + (s.cue || '—') + (intents ? '  细扫 ' + intents : '');
      var url = '/media/' + encodeURIComponent(job.id) + '/'
        + encodeURIComponent(s.image || '');
      rowHtml += '<figure class="shot-card' + (s.op ? ' has-op' : '') + '"'
        + ' title="' + esc(cap) + '">'
        + (s.image
            ? '<img class="shot-img" loading="lazy" src="' + esc(url)
              + '" data-full="' + esc(url) + '" data-cap="' + esc(cap) + '" alt="" />'
            : '')
        + '<figcaption><span class="shot-pct">' + s.percent + '%</span>'
        + '<span class="shot-t">' + s.t + 's</span>'
        + '<span class="shot-op">' + (s.op ? esc(s.op) : '—') + '</span>'
        + '<span class="shot-cue">' + esc(s.cue || '') + '</span></figcaption></figure>';
    });
    el.videoShots.innerHTML = rowHtml;

    state.route = job.route || '';
    state.flat = job.flat || '';
    el.videoRouteText.textContent = state.flat || '（没检测到任何操作）';
    var n = state.flat ? Math.floor(state.flat.split(',').length / 2) : 0;
    var nShot = (job.shots || []).length;
    el.videoRouteStat.textContent = n + ' 条操作 / ' + nShot + ' 张截图'
      + ' / 细扫 ' + (job.button_events || 0) + ' 个按键动作, '
      + (job.intents || 0) + ' 个操作目的'
      + (job.seconds ? ' / ' + job.seconds + 's' : '');

    // 注释部分（判据明细）留在下面，核对用
    var notes = state.route.split('\n').filter(function (l) {
      return l.trim().startsWith('#');
    }).join('\n');
    el.videoRouteNotes.textContent = notes || '（没有注释）';
    el.videoResult.classList.remove('hidden');
  }

  function upload(file) {
    if (!file) { return; }
    var fd = new FormData();
    fd.append('video', file);
    fd.append('every', el.everySel.value);
    if (el.maxSecInput.value) { fd.append('max_seconds', el.maxSecInput.value); }
    setProgress('上传 ' + file.name + ' …', null);
    fetch('/api/analyze', { method: 'POST', body: fd }).then(function (r) {
      return r.json().then(function (data) {
        if (!r.ok) { throw new Error((data && data.error) || ('HTTP ' + r.status)); }
        return data;
      });
    }).then(function (data) {
      startWatch(data.job, data.name);
      scanVideos();
    }).catch(function (err) {
      setProgress('上传失败：' + err.message, null);
      fatal('上传失败：' + err.message);
    });
  }

  function analyzePath() {
    var p = el.videoSelect.value;
    if (!p) { toast('先选一个视频（或拖文件进来）', 'warn'); return; }
    var body = { path: p, every: parseFloat(el.everySel.value) };
    if (el.maxSecInput.value) { body.max_seconds = parseFloat(el.maxSecInput.value); }
    setProgress('开始分析 ' + p.split(/[\\/]/).pop() + ' …', 0);
    api('/api/analyze', { method: 'POST', body: body }).then(function (data) {
      startWatch(data.job, data.name);
    }).catch(function (err) {
      setProgress('启动失败：' + err.message, null);
      fatal('启动失败：' + err.message);
    });
  }

  // ------------------------------------------------------------ 路线文本
  function flatOnly() { return state.flat || ''; }

  function copyRoute() {
    var text = flatOnly();
    if (!text) { toast('还没有路线', 'warn'); return; }
    var done = function () { toast('已复制那一行路线 ✓', 'ok'); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text, done); });
    } else {
      fallbackCopy(text, done);
    }
  }

  function fallbackCopy(text, done) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); done(); }
    catch (e) { toast('复制失败，请手动选中复制', 'warn'); }
    ta.remove();
  }

  function downloadRoute() {
    if (!state.job) { return; }
    window.location.href = '/api/route?job=' + encodeURIComponent(state.job);
  }

  // ------------------------------------------------------------ 放大
  function openLightbox(src, caption) {
    el.lightboxImg.src = src;
    el.lightboxCap.textContent = caption || '';
    el.lightbox.classList.remove('hidden');
  }

  function closeLightbox() {
    el.lightbox.classList.add('hidden');
    el.lightboxImg.removeAttribute('src');
  }

  // ------------------------------------------------------------ 参数
  function showParams() {
    var v = state.version;
    if (!v) { toast('还没取到后端信息', 'warn'); return; }
    var lines = ['项目根: ' + v.root, '配置文件: ' + v.config_file
      + (v.config_exists ? '' : '（不存在 —— 用内置默认值）'), ''];
    ['scan', 'vision'].forEach(function (sec) {
      lines.push('[' + sec + ']');
      Object.keys(v[sec] || {}).sort().forEach(function (k) {
        lines.push('  ' + k + ' = ' + JSON.stringify(v[sec][k]));
      });
      lines.push('');
    });
    el.paramsDump.textContent = lines.join('\n');
    el.paramsModal.classList.remove('hidden');
  }

  // ------------------------------------------------------------ 绑定
  function bind() {
    el.pickFileBtn.addEventListener('click', function () { el.videoFile.click(); });
    el.dropZone.addEventListener('click', function (e) {
      if (e.target === el.dropZone) { el.videoFile.click(); }
    });
    el.videoFile.addEventListener('change', function () {
      upload(el.videoFile.files && el.videoFile.files[0]);
    });
    el.videoScanBtn.addEventListener('click', scanVideos);
    el.videoAnalyzeBtn.addEventListener('click', analyzePath);
    el.stopWatchBtn.addEventListener('click', function () { stopWatch(false); });
    el.videoCopyBtn.addEventListener('click', copyRoute);
    el.videoDownloadBtn.addEventListener('click', downloadRoute);
    el.paramsBtn.addEventListener('click', showParams);
    el.paramsModalClose.addEventListener('click', function () {
      el.paramsModal.classList.add('hidden');
    });
    el.paramsModal.addEventListener('click', function (e) {
      if (e.target === el.paramsModal) { el.paramsModal.classList.add('hidden'); }
    });

    el.videoShots.addEventListener('click', function (e) {
      var img = e.target.closest ? e.target.closest('.shot-img') : null;
      if (!img) { return; }
      openLightbox(img.dataset.full || img.src, img.dataset.cap || '');
    });
    el.lightbox.addEventListener('click', closeLightbox);
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        closeLightbox();
        el.paramsModal.classList.add('hidden');
      }
    });

    ['dragenter', 'dragover'].forEach(function (ev) {
      el.dropZone.addEventListener(ev, function (e) {
        e.preventDefault(); e.stopPropagation();
        el.dropZone.classList.add('drag-over');
      });
    });
    ['dragleave', 'drop'].forEach(function (ev) {
      el.dropZone.addEventListener(ev, function (e) {
        e.preventDefault(); e.stopPropagation();
        el.dropZone.classList.remove('drag-over');
      });
    });
    el.dropZone.addEventListener('drop', function (e) {
      var dt = e.dataTransfer;
      var f = dt && dt.files && dt.files[0];
      if (f) { upload(f); }
    });
  }

  function boot() {
    bind();
    setPill('空闲');
    api('/api/version').then(function (data) {
      state.version = data;
      document.title = '跑图视频 → 路线 · a9route ' + data.version;
    }).catch(function (err) { showBanner('后端连不上：' + err.message); });
    scanVideos();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
