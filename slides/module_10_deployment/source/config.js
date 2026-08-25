(function () {
  // Module 10 config. No Manim clips: the only spatial figures are the
  // bandwidth-wall bar chart and the batching curve, both built as positioned
  // HTML with reveal.js fragments, plus the exercise's matplotlib output.
  // ===================================================================
  // WIDGET: kvBudget — fit weights and KV cache into one GPU's memory
  // and read off how many concurrent requests actually fit.
  // ===================================================================
  function kvBudget(host) {
    var U = WIDGET_UTIL, COL = WIDGET_UTIL.COL;
    var GPUS = {
      a100: { name: 'A100 80GB', hbm: 80 },
      h100: { name: 'H100 80GB', hbm: 80 },
      l40s: { name: 'L40S 48GB', hbm: 48 },
      rtx: { name: 'RTX 4090 24GB', hbm: 24 }
    };
    var state = { gpu: 'a100', kvheads: 32, wbits: 16 };

    host.innerHTML =
      '<div class="iw">' +
        '<div class="iw-canvas-wrap"><canvas class="iw-canvas"></canvas></div>' +
        '<div class="iw-stats">' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="perTok">0</span><span class="iw-stat-lab">cache per token</span></div>' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="perReq">0</span><span class="iw-stat-lab">cache per request</span></div>' +
          '<div class="iw-stat" data-el="fitbox"><span class="iw-stat-num" data-el="fit">0</span><span class="iw-stat-lab">requests that fit</span></div>' +
        '</div>' +
        '<div class="iw-sliders">' +
          U.sliderHTML('params', 'model size (B)', 1, 180, 1, 7) +
          U.sliderHTML('layers', 'layers', 8, 96, 1, 32) +
          U.sliderHTML('ctx', 'context (K tokens)', 1, 128, 1, 4) +
          U.sliderHTML('conc', 'concurrent requests', 1, 128, 1, 8) +
        '</div>' +
        '<div class="iw-controls">' +
          '<span class="iw-label">GPU:</span>' +
          '<button class="iw-btn active" data-gpu data-val="a100">A100 80GB</button>' +
          '<button class="iw-btn" data-gpu data-val="l40s">L40S 48GB</button>' +
          '<button class="iw-btn" data-gpu data-val="rtx">RTX 4090 24GB</button>' +
          '<span class="iw-label">attention:</span>' +
          '<button class="iw-btn active" data-attn data-val="32">MHA (32 KV heads)</button>' +
          '<button class="iw-btn" data-attn data-val="8">GQA (8)</button>' +
          '<button class="iw-btn" data-attn data-val="1">MQA (1)</button>' +
        '</div>' +
        '<p class="iw-readout"></p>' +
      '</div>';
    U.stop(host);
    var canvas = host.querySelector('.iw-canvas');
    var readout = host.querySelector('.iw-readout');
    var el = {};
    host.querySelectorAll('[data-el]').forEach(function (n) { el[n.getAttribute('data-el')] = n; });

    function compute() {
      var v = read();
      var dHead = 128;                      // fixed, as in most production models
      // 2 (K and V) * layers * kv-heads * head-dim * 2 bytes (fp16 cache)
      var perTok = 2 * v.layers * state.kvheads * dHead * 2;
      var perReq = perTok * v.ctx * 1024 / 1e9;                 // GB
      var weights = v.params * 1e9 * (state.wbits / 8) / 1e9;   // GB
      var hbm = GPUS[state.gpu].hbm;
      var overhead = hbm * 0.08;            // activations, fragmentation, runtime
      var free = Math.max(0, hbm - weights - overhead);
      return {
        v: v, perTok: perTok, perReq: perReq, weights: weights,
        hbm: hbm, overhead: overhead, free: free,
        used: perReq * v.conc,
        fit: perReq > 0 ? Math.floor(free / perReq) : 0
      };
    }

    function draw() {
      var c = compute();
      var f = U.fit(canvas); if (!f) return;
      var ctx = f.ctx, W = f.w, H = f.h;
      ctx.clearRect(0, 0, W, H);

      var padL = 26, padR = 26, barY = 62, barH = 62;
      var bw = W - padL - padR;
      var scale = bw / c.hbm;

      ctx.textAlign = 'center'; ctx.font = '12px Inter, sans-serif'; ctx.fillStyle = COL.muted;
      ctx.fillText(GPUS[state.gpu].name + ' memory, allocated left to right', W / 2, 22);

      // `gb` is the drawn width; `shown` is the number to print when the
      // segment had to be clipped to the edge of the card.
      function seg(x, gb, fill, stroke, label, shown) {
        var w = Math.max(0, gb * scale);
        if (w <= 0) return x;
        ctx.fillStyle = fill;
        ctx.fillRect(x, barY, Math.min(w, padL + bw - x), barH);
        ctx.strokeStyle = stroke; ctx.lineWidth = 1;
        ctx.strokeRect(x, barY, Math.min(w, padL + bw - x), barH);
        if (w > 60) {
          ctx.fillStyle = stroke;
          ctx.font = '600 12px Inter, sans-serif';
          ctx.fillText(label, x + Math.min(w, padL + bw - x) / 2, barY + barH / 2 + 4);
          ctx.font = '11px Inter, sans-serif';
          ctx.fillStyle = COL.muted;
          ctx.fillText(U.fmtBytes(shown === undefined ? gb : shown) +
            (shown !== undefined && shown > gb + 1e-9 ? ' needed' : ''),
            x + Math.min(w, padL + bw - x) / 2, barY + barH / 2 + 20);
        }
        return x + w;
      }

      ctx.strokeStyle = COL.line; ctx.lineWidth = 1.5;
      ctx.strokeRect(padL, barY, bw, barH);

      var x = padL;
      x = seg(x, c.weights, 'rgba(74,158,255,0.55)', COL.primary, 'weights');
      x = seg(x, c.overhead, 'rgba(136,146,164,0.30)', COL.muted, 'runtime');
      var overflow = c.used > c.free;
      x = seg(x, Math.min(c.used, c.free), 'rgba(245,166,35,0.6)', COL.secondary, 'KV cache', c.used);
      if (overflow) {
        ctx.fillStyle = 'rgba(231,76,60,0.35)';
        ctx.fillRect(padL, barY, bw, barH);
        ctx.fillStyle = COL.red;
        ctx.font = '600 15px Inter, sans-serif';
        ctx.fillText('out of memory: needs ' + U.fmtBytes(c.weights + c.overhead + c.used) +
          ' on a ' + c.hbm + ' GB card', W / 2, barY + barH + 34);
      } else {
        ctx.fillStyle = COL.muted; ctx.font = '12px Inter, sans-serif';
        ctx.fillText(U.fmtBytes(c.free - c.used) + ' still free for more requests', W / 2, barY + barH + 30);
      }

      // Second row: how the cache grows with concurrency at this context length.
      var gy = barY + barH + 58;
      var gh = H - gy - 24;
      if (gh > 40) {
        ctx.textAlign = 'left'; ctx.fillStyle = COL.muted; ctx.font = '11px Inter, sans-serif';
        ctx.fillText('cache demand vs concurrent requests', padL, gy - 6);
        ctx.strokeStyle = COL.line; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(padL, gy + gh); ctx.lineTo(W - padR, gy + gh); ctx.stroke();
        var maxR = 128, maxGB = Math.max(c.free, c.perReq * maxR);
        ctx.strokeStyle = COL.secondary; ctx.lineWidth = 2;
        ctx.beginPath();
        for (var r = 1; r <= maxR; r++) {
          var px = padL + ((r - 1) / (maxR - 1)) * (W - padL - padR);
          var py = gy + gh - Math.min(1, (c.perReq * r) / maxGB) * gh;
          if (r === 1) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
        var fy = gy + gh - Math.min(1, c.free / maxGB) * gh;
        ctx.strokeStyle = COL.green; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(padL, fy); ctx.lineTo(W - padR, fy); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COL.green;
        ctx.fillText('memory left after weights', padL + 6, fy - 6);
        ctx.fillStyle = COL.muted;
        ctx.textAlign = 'left';
        ctx.fillText('1 request', padL, gy + gh + 14);
        ctx.textAlign = 'right';
        ctx.fillText(maxR + ' requests', W - padR, gy + gh + 14);
        ctx.textAlign = 'center';
      }

      el.perTok.textContent = U.fmtBytes(c.perTok / 1e9);
      el.perReq.textContent = U.fmtBytes(c.perReq);
      el.fit.textContent = String(c.fit);
      el.fitbox.className = 'iw-stat ' + (c.fit < 1 ? 'warn' : c.fit >= c.v.conc ? 'good' : 'warn');

      U.setVal(host, 'params', c.v.params + 'B');
      U.setVal(host, 'layers', String(c.v.layers));
      U.setVal(host, 'ctx', c.v.ctx + 'K');
      U.setVal(host, 'conc', String(c.v.conc));

      readout.innerHTML = c.fit < 1
        ? 'The weights alone leave no room for a single request at this context length.'
        : 'At <strong>' + c.v.ctx + 'K context</strong> this card serves <strong>' + c.fit +
          '</strong> concurrent requests. The cache, not the arithmetic, sets that ceiling &mdash; which is why GQA and paging exist.';
    }

    var read = U.bindSliders(host, draw);
    U.bindToggle(host, '[data-gpu]', function (val) { state.gpu = val; draw(); });
    U.bindToggle(host, '[data-attn]', function (val) { state.kvheads = parseInt(val, 10); draw(); });
    draw();
    return { resize: draw };
  }

  // ===================================================================
  // WIDGET: quantExplorer — round a weight distribution onto a grid of
  // 2^bits levels and watch both the file size and the error move.
  // ===================================================================
  function quantExplorer(host) {
    var U = WIDGET_UTIL, COL = WIDGET_UTIL.COL;
    var state = { bits: 8, outlier: false, params: 7 };

    // A fixed bell-shaped sample of weights, plus a couple of far outliers.
    var WEIGHTS = (function () {
      var rand = U.rng(31337), out = [];
      for (var i = 0; i < 600; i++) {
        var u = rand(), v = rand();
        out.push(Math.sqrt(-2 * Math.log(u + 1e-9)) * Math.cos(2 * Math.PI * v) * 0.25);
      }
      return out;
    }());
    var OUTLIERS = [2.9, -2.6, 3.4];

    host.innerHTML =
      '<div class="iw">' +
        '<div class="iw-canvas-wrap"><canvas class="iw-canvas"></canvas></div>' +
        '<div class="iw-stats">' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="lv">0</span><span class="iw-stat-lab">distinct levels</span></div>' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="sz">0</span><span class="iw-stat-lab">7B checkpoint</span></div>' +
          '<div class="iw-stat" data-el="errbox"><span class="iw-stat-num" data-el="err">0</span><span class="iw-stat-lab">RMS error</span></div>' +
        '</div>' +
        '<div class="iw-sliders">' +
          U.sliderHTML('bits', 'bits per weight', 2, 16, 1, 8) +
        '</div>' +
        '<div class="iw-controls">' +
          '<button class="iw-btn" data-act="outlier">Add three outlier weights</button>' +
        '</div>' +
        '<p class="iw-readout"></p>' +
      '</div>';
    U.stop(host);
    var canvas = host.querySelector('.iw-canvas');
    var readout = host.querySelector('.iw-readout');
    var el = {};
    host.querySelectorAll('[data-el]').forEach(function (n) { el[n.getAttribute('data-el')] = n; });

    function values() { return state.outlier ? WEIGHTS.concat(OUTLIERS) : WEIGHTS; }

    // Absmax scaling: one scale for the whole tensor, so the range is set by
    // the largest magnitude present, outliers included.
    function quantize(vals, bits) {
      var maxAbs = 0;
      vals.forEach(function (w) { maxAbs = Math.max(maxAbs, Math.abs(w)); });
      var levels = Math.pow(2, bits);
      var step = (2 * maxAbs) / (levels - 1);
      var err = 0;
      var q = vals.map(function (w) {
        var r = Math.round((w + maxAbs) / step) * step - maxAbs;
        err += (r - w) * (r - w);
        return r;
      });
      return { q: q, step: step, maxAbs: maxAbs, rms: Math.sqrt(err / vals.length), levels: levels };
    }

    function draw() {
      var v = read();
      state.bits = v.bits;
      var vals = values();
      var res = quantize(vals, state.bits);

      var f = U.fit(canvas); if (!f) return;
      var ctx = f.ctx, W = f.w, H = f.h;
      ctx.clearRect(0, 0, W, H);

      var padL = 30, padR = 30, padT = 28, padB = 40;
      var pw = W - padL - padR, ph = H - padT - padB;
      var lim = res.maxAbs * 1.05;
      function X(w) { return padL + ((w + lim) / (2 * lim)) * pw; }

      ctx.textAlign = 'center'; ctx.font = '12px Inter, sans-serif'; ctx.fillStyle = COL.muted;
      ctx.fillText('weight values (bars) snapped onto the representable grid (ticks below)', W / 2, 16);

      // Histogram of the original weights.
      var BINS = 90, hist = new Array(BINS).fill(0);
      vals.forEach(function (w) {
        var b = Math.floor(((w + lim) / (2 * lim)) * BINS);
        if (b >= 0 && b < BINS) hist[b]++;
      });
      var hmax = Math.max.apply(null, hist);
      var bw = pw / BINS;
      for (var i = 0; i < BINS; i++) {
        if (!hist[i]) continue;
        var h = (hist[i] / hmax) * (ph - 46);
        ctx.fillStyle = 'rgba(74,158,255,0.55)';
        ctx.fillRect(padL + i * bw, padT + (ph - 46) - h, Math.max(1, bw - 1), h);
      }

      // Grid ticks; drawn only when sparse enough to read.
      var gy = padT + ph - 34;
      ctx.strokeStyle = COL.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, gy); ctx.lineTo(W - padR, gy); ctx.stroke();
      if (res.levels <= 256) {
        ctx.strokeStyle = 'rgba(245,166,35,0.85)'; ctx.lineWidth = res.levels <= 32 ? 1.6 : 0.7;
        for (var L = 0; L < res.levels; L++) {
          var val = -res.maxAbs + L * res.step;
          var x = X(val);
          ctx.beginPath(); ctx.moveTo(x, gy - 8); ctx.lineTo(x, gy + 8); ctx.stroke();
        }
      } else {
        ctx.fillStyle = COL.secondary; ctx.font = '12px Inter, sans-serif';
        ctx.fillText(res.levels.toLocaleString() + ' levels: the grid is finer than the pixels', W / 2, gy + 5);
      }

      ctx.fillStyle = COL.muted; ctx.font = '11px Inter, sans-serif';
      ctx.fillText('-' + res.maxAbs.toFixed(2), padL + 12, gy + 26);
      ctx.fillText('+' + res.maxAbs.toFixed(2), W - padR - 12, gy + 26);
      ctx.fillText('0', X(0), gy + 26);

      el.lv.textContent = res.levels >= 1000 ? U.fmtCount(res.levels) : String(res.levels);
      el.sz.textContent = U.fmtBytes(state.params * 1e9 * state.bits / 8 / 1e9);
      el.err.textContent = res.rms.toFixed(4);
      el.errbox.className = 'iw-stat ' + (res.rms > 0.05 ? 'warn' : res.rms < 0.005 ? 'good' : '');
      U.setVal(host, 'bits', state.bits + '-bit');

      readout.innerHTML = state.outlier
        ? 'Three outliers stretched the range to <strong>&plusmn;' + res.maxAbs.toFixed(2) +
          '</strong>, so every ordinary weight now shares a coarser grid and RMS error jumps to <strong>' +
          res.rms.toFixed(4) + '</strong>. Handling outliers separately is the whole trick behind LLM.int8 and NF4.'
        : 'A 7B model in fp16 is <strong>14 GB</strong>; at <strong>' + state.bits + '-bit</strong> it is <strong>' +
          U.fmtBytes(state.params * 1e9 * state.bits / 8 / 1e9) +
          '</strong>. Below about 4 bits the grid gets coarse enough that quality starts to move.';
    }

    var read = U.bindSliders(host, draw);
    host.querySelector('[data-act="outlier"]').addEventListener('click', function (e) {
      state.outlier = !state.outlier;
      e.currentTarget.classList.toggle('active', state.outlier);
      e.currentTarget.textContent = state.outlier ? 'Remove the outlier weights' : 'Add three outlier weights';
      draw();
    });
    draw();
    return { resize: draw };
  }

  window.MODULE_CONFIG = {
    title: 'LLMs 0 to 100 - Module 10',
    manimSections: {},
    widgets: {
      kvBudget: kvBudget,
      quantExplorer: quantExplorer
    }
  };
}());
