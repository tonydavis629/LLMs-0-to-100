(function () {
  // Module 5 config. Manim steppers are keyed by the `scene` slug used in the
  // :::manim fences; each maps to its ordered list of section clips.
  // ===================================================================
  // WIDGET: scalingPlanner — split a fixed compute budget between
  // parameters and tokens, and read the predicted loss off the curve.
  // Loss uses the Chinchilla parametric fit L(N, D) = E + A/N^alpha + B/D^beta,
  // with the corrected coefficients from Besiroglu et al. (2024). Those are
  // the ones whose optimum reproduces the paper's own ~20 tokens per parameter;
  // the coefficients printed in Hoffmann et al. (2022) put it near 60.
  // ===================================================================
  function scalingPlanner(host) {
    var U = WIDGET_UTIL, COL = WIDGET_UTIL.COL;
    var E = 1.8172, A = 482.01, ALPHA = 0.3478, B = 2085.43, BETA = 0.3658;
    // Reference points: tokens per parameter actually used by real runs.
    var MARKS = [
      { label: 'GPT-3', ratio: 1.7 },
      { label: 'Chinchilla', ratio: 20 },
      { label: 'Llama 3 8B', ratio: 1875 }
    ];

    host.innerHTML =
      '<div class="iw">' +
        '<div class="iw-canvas-wrap"><canvas class="iw-canvas"></canvas></div>' +
        '<div class="iw-stats">' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="N">0</span><span class="iw-stat-lab">parameters</span></div>' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="D">0</span><span class="iw-stat-lab">training tokens</span></div>' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="L">0</span><span class="iw-stat-lab">predicted loss</span></div>' +
          '<div class="iw-stat" data-el="gapbox"><span class="iw-stat-num" data-el="gap">0</span><span class="iw-stat-lab">loss above optimal</span></div>' +
        '</div>' +
        '<div class="iw-sliders">' +
          U.sliderHTML('logC', 'compute FLOPs', 19, 26, 0.1, 22) +
          U.sliderHTML('logR', 'tokens / param', 0, 3.4, 0.02, 1.3) +
        '</div>' +
        '<div class="iw-controls">' +
          '<button class="iw-btn iw-primary" data-act="opt">Jump to compute-optimal</button>' +
          '<button class="iw-btn" data-act="gpt3">GPT-3 split</button>' +
          '<button class="iw-btn" data-act="llama">Llama 3 split</button>' +
        '</div>' +
        '<p class="iw-readout"></p>' +
      '</div>';
    U.stop(host);

    var canvas = host.querySelector('.iw-canvas');
    var readout = host.querySelector('.iw-readout');
    var el = {};
    host.querySelectorAll('[data-el]').forEach(function (n) { el[n.getAttribute('data-el')] = n; });

    // C = 6ND with D = ratio*N gives N = sqrt(C / (6 * ratio)).
    function split(C, ratio) {
      var N = Math.sqrt(C / (6 * ratio));
      return { N: N, D: ratio * N };
    }
    function loss(N, D) { return E + A / Math.pow(N, ALPHA) + B / Math.pow(D, BETA); }

    function bestRatio(C) {
      var best = 1, bestL = Infinity;
      for (var lr = 0; lr <= 3.4; lr += 0.01) {
        var r = Math.pow(10, lr), s = split(C, r), L = loss(s.N, s.D);
        if (L < bestL) { bestL = L; best = r; }
      }
      return { ratio: best, loss: bestL };
    }

    function draw() {
      var v = read();
      var C = Math.pow(10, v.logC), ratio = Math.pow(10, v.logR);
      var cur = split(C, ratio), curL = loss(cur.N, cur.D);
      var opt = bestRatio(C);

      var f = U.fit(canvas); if (!f) return;
      var ctx = f.ctx, W = f.w, H = f.h;
      ctx.clearRect(0, 0, W, H);

      var padL = 60, padR = 24, padT = 30, padB = 46;
      var pw = W - padL - padR, ph = H - padT - padB;

      // Sample the loss curve across the whole ratio range at this budget.
      var pts = [], lo = Infinity, hi = -Infinity;
      for (var lr = 0; lr <= 3.4; lr += 0.02) {
        var s = split(C, Math.pow(10, lr)), L = loss(s.N, s.D);
        pts.push([lr, L]);
        if (L < lo) lo = L; if (L > hi) hi = L;
      }
      var span = Math.max(hi - lo, 0.05);
      hi = lo + span * 1.12; lo = lo - span * 0.08;

      function X(lr) { return padL + (lr / 3.4) * pw; }
      function Y(L) { return padT + ph - ((L - lo) / (hi - lo)) * ph; }

      ctx.strokeStyle = COL.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + ph); ctx.lineTo(W - padR, padT + ph); ctx.stroke();

      ctx.textAlign = 'center';
      ctx.font = '12px Inter, sans-serif';
      ctx.fillStyle = COL.muted;
      ctx.fillText('tokens per parameter (log scale) at a fixed compute budget', padL + pw / 2, H - 10);
      ctx.save();
      ctx.translate(16, padT + ph / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillText('predicted loss', 0, 0);
      ctx.restore();

      [0, 1, 2, 3].forEach(function (t) {
        ctx.fillStyle = COL.muted;
        ctx.fillText(U.fmtCount(Math.pow(10, t)).replace('K', 'k'), X(t), padT + ph + 18);
      });

      ctx.strokeStyle = COL.primary; ctx.lineWidth = 2.5;
      ctx.beginPath();
      pts.forEach(function (p, i) { if (i === 0) ctx.moveTo(X(p[0]), Y(p[1])); else ctx.lineTo(X(p[0]), Y(p[1])); });
      ctx.stroke();

      // Reference runs, drawn as faint verticals with a label.
      ctx.font = '11px Inter, sans-serif';
      MARKS.forEach(function (m) {
        var lx = X(Math.log10(m.ratio));
        ctx.strokeStyle = 'rgba(136,146,164,0.35)';
        ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(lx, padT); ctx.lineTo(lx, padT + ph); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COL.muted;
        ctx.fillText(m.label, lx, padT - 8);
      });

      // The optimum for this budget, and where the sliders currently sit.
      var ox = X(Math.log10(opt.ratio)), oy = Y(opt.loss);
      ctx.fillStyle = COL.green;
      ctx.beginPath(); ctx.arc(ox, oy, 5, 0, Math.PI * 2); ctx.fill();
      ctx.fillText('optimal', ox, oy + 20);

      var cx = X(v.logR), cy = Y(curL);
      ctx.fillStyle = COL.secondary;
      ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = COL.secondary; ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, padT + ph); ctx.stroke();
      ctx.setLineDash([]);

      el.N.textContent = U.fmtCount(cur.N);
      el.D.textContent = U.fmtCount(cur.D);
      el.L.textContent = curL.toFixed(3);
      var gap = curL - opt.loss;
      el.gap.textContent = '+' + gap.toFixed(3);
      el.gapbox.className = 'iw-stat ' + (gap < 0.01 ? 'good' : gap > 0.08 ? 'warn' : '');

      U.setVal(host, 'logC', '1e' + v.logC.toFixed(1));
      U.setVal(host, 'logR', ratio < 10 ? ratio.toFixed(1) : String(Math.round(ratio)));

      var atOpt = Math.abs(Math.log10(ratio) - Math.log10(opt.ratio)) < 0.05;
      readout.innerHTML = atOpt
        ? 'This is the compute-optimal split for <strong>1e' + v.logC.toFixed(1) +
          '</strong> FLOPs: about <strong>' + Math.round(opt.ratio) + ' tokens per parameter</strong>.'
        : 'Same budget, worse loss by <strong>' + gap.toFixed(3) + '</strong>. The optimum here is <strong>' +
          Math.round(opt.ratio) + ' tokens per parameter</strong> &mdash; ' +
          (ratio < opt.ratio ? 'this model is too big for its data.' : 'this model is smaller than training alone would want, which is what buys cheap serving.');
    }

    var read = U.bindSliders(host, draw);

    function setTo(logC, ratio) {
      host.querySelector('[data-key="logC"]').value = logC;
      host.querySelector('[data-key="logR"]').value = Math.log10(ratio);
      draw();
    }
    host.querySelector('[data-act="opt"]').addEventListener('click', function () {
      var v = read();
      setTo(v.logC, bestRatio(Math.pow(10, v.logC)).ratio);
    });
    host.querySelector('[data-act="gpt3"]').addEventListener('click', function () { setTo(read().logC, 1.7); });
    host.querySelector('[data-act="llama"]').addEventListener('click', function () { setTo(read().logC, 1875); });

    draw();
    return { resize: draw };
  }

  // ===================================================================
  // WIDGET: servingPlanner — two models that reach the SAME loss.
  // Model A is the Chinchilla choice (N_A params on 20 N_A tokens).
  // Model B is smaller; the tokens it needs to match A's loss come from
  // inverting the corrected Chinchilla fit L = E + A/N^alpha + B/D^beta.
  // Lifetime cost = training 6ND + serving 2N FLOPs per token served
  // (Sardana et al., 2023). B costs more to train but less to serve, so
  // the two cost lines cross at a break-even serving volume.
  // ===================================================================
  function servingPlanner(host) {
    var U = WIDGET_UTIL, COL = WIDGET_UTIL.COL;
    var E = 1.8172, A = 482.01, ALPHA = 0.3478, B = 2085.43, BETA = 0.3658;
    var X_MIN = 9, X_MAX = 15;  // tokens served, log10

    host.innerHTML =
      '<div class="iw">' +
        '<div class="iw-canvas-wrap"><canvas class="iw-canvas"></canvas></div>' +
        '<div class="iw-stats">' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="A">0</span><span class="iw-stat-lab">model A (Chinchilla)</span></div>' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="B">0</span><span class="iw-stat-lab">model B (smaller)</span></div>' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="be">0</span><span class="iw-stat-lab">break-even tokens served</span></div>' +
          '<div class="iw-stat" data-el="winbox"><span class="iw-stat-num" data-el="win">0</span><span class="iw-stat-lab">B vs A, lifetime compute</span></div>' +
        '</div>' +
        '<div class="iw-sliders">' +
          U.sliderHTML('logA', 'model A size', 9, 11.3, 0.05, 10.85) +
          U.sliderHTML('frac', 'model B size (share of A)', 0.2, 0.9, 0.01, 0.4) +
          U.sliderHTML('logInf', 'tokens served over the lifetime', X_MIN, X_MAX, 0.1, 13) +
        '</div>' +
        '<p class="iw-readout"></p>' +
      '</div>';
    U.stop(host);

    var canvas = host.querySelector('.iw-canvas');
    var readout = host.querySelector('.iw-readout');
    var el = {};
    host.querySelectorAll('[data-el]').forEach(function (n) { el[n.getAttribute('data-el')] = n; });

    function loss(N, D) { return E + A / Math.pow(N, ALPHA) + B / Math.pow(D, BETA); }
    function tokensFor(N, Lt) {
      var room = Lt - E - A / Math.pow(N, ALPHA);
      return room > 0 ? Math.pow(B / room, 1 / BETA) : Infinity;
    }
    function fmtFlops(f) {
      var e = Math.floor(Math.log10(f));
      return (f / Math.pow(10, e)).toFixed(1) + 'e' + e;
    }

    function draw() {
      var v = read();
      var NA = Math.pow(10, v.logA), DA = 20 * NA;
      var Lt = loss(NA, DA);
      var NB = v.frac * NA, DB = tokensFor(NB, Lt);
      var trA = 6 * NA * DA, trB = 6 * NB * DB;
      var served = Math.pow(10, v.logInf);
      function cost(tr, N, d) { return tr + 2 * N * d; }
      // Lines cross where trA + 2 NA d = trB + 2 NB d.
      var be = (trB - trA) / (2 * (NA - NB));

      var f = U.fit(canvas); if (!f) return;
      var ctx = f.ctx, W = f.w, H = f.h;
      ctx.clearRect(0, 0, W, H);
      var padL = 70, padR = 24, padT = 40, padB = 46;
      var pw = W - padL - padR, ph = H - padT - padB;

      var yLo = Math.log10(Math.min(trA, trB)) - 0.3;
      var yHi = Math.log10(cost(trA, NA, Math.pow(10, X_MAX))) + 0.1;
      function X(ld) { return padL + (ld - X_MIN) / (X_MAX - X_MIN) * pw; }
      function Y(val) { return padT + ph - (Math.log10(val) - yLo) / (yHi - yLo) * ph; }

      ctx.strokeStyle = COL.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + ph); ctx.lineTo(W - padR, padT + ph); ctx.stroke();

      ctx.font = '12px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.fillStyle = COL.muted;
      ctx.fillText('tokens served over the model’s lifetime (log scale)', padL + pw / 2, H - 10);
      for (var t = X_MIN; t <= X_MAX; t++) {
        ctx.fillText(t >= 12 ? Math.pow(10, t - 12).toLocaleString() + 'T' : U.fmtCount(Math.pow(10, t)), X(t), padT + ph + 18);
      }
      ctx.save();
      ctx.translate(16, padT + ph / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillText('lifetime compute: training + serving (FLOPs, log)', 0, 0);
      ctx.restore();

      function line(tr, N, color) {
        ctx.strokeStyle = color; ctx.lineWidth = 3;
        ctx.beginPath();
        for (var ld = X_MIN; ld <= X_MAX + 1e-9; ld += 0.02) {
          var px = X(ld), py = Y(cost(tr, N, Math.pow(10, ld)));
          if (ld === X_MIN) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
      line(trA, NA, COL.primary);
      line(trB, NB, COL.secondary);

      // Legend above the plot.
      ctx.textAlign = 'left';
      var lx = padL + 10;
      [['A: ' + U.fmtCount(NA) + ' on ' + U.fmtCount(DA) + ' tokens', COL.primary],
       ['B: ' + U.fmtCount(NB) + ' on ' + U.fmtCount(DB) + ' tokens', COL.secondary]].forEach(function (it) {
        ctx.fillStyle = it[1]; ctx.fillRect(lx, padT - 22, 16, 3);
        ctx.fillStyle = COL.muted; ctx.fillText(it[0], lx + 24, padT - 18);
        lx += 24 + ctx.measureText(it[0]).width + 30;
      });

      // Break-even marker.
      ctx.textAlign = 'center';
      var lbe = Math.log10(be);
      if (lbe > X_MIN && lbe < X_MAX) {
        var bx = X(lbe), by = Y(cost(trA, NA, be));
        ctx.fillStyle = COL.green;
        ctx.beginPath(); ctx.arc(bx, by, 6, 0, Math.PI * 2); ctx.fill();
        ctx.fillText('break-even', bx - 44, by - 10);
      }

      // The chosen serving volume.
      var sx = X(v.logInf);
      ctx.strokeStyle = 'rgba(232,234,240,0.5)'; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(sx, padT); ctx.lineTo(sx, padT + ph); ctx.stroke(); ctx.setLineDash([]);
      [[trA, NA, COL.primary], [trB, NB, COL.secondary]].forEach(function (m) {
        ctx.fillStyle = m[2];
        ctx.beginPath(); ctx.arc(sx, Y(cost(m[0], m[1], served)), 5, 0, Math.PI * 2); ctx.fill();
      });

      var cA = cost(trA, NA, served), cB = cost(trB, NB, served);
      var diff = cB / cA - 1;
      el.A.textContent = U.fmtCount(NA);
      el.B.textContent = U.fmtCount(NB);
      el.be.textContent = U.fmtCount(be);
      el.win.textContent = (diff > 0 ? '+' : '−') + Math.abs(Math.round(diff * 100)) + '%';
      el.winbox.className = 'iw-stat ' + (diff < 0 ? 'good' : 'warn');

      U.setVal(host, 'logA', U.fmtCount(NA));
      U.setVal(host, 'frac', Math.round(v.frac * 100) + '%');
      U.setVal(host, 'logInf', U.fmtCount(served));

      readout.innerHTML =
        'Both models reach the same loss. With fewer parameters, B must see more data to get there: <strong>' +
        (DB / DA).toFixed(1) + '&times;</strong> as many tokens as A (' + Math.round(DB / NB).toLocaleString() +
        ' per parameter instead of 20), so it costs <strong>' + (trB / trA).toFixed(1) +
        '&times;</strong> as much to train. Each token it serves costs ' + Math.round(v.frac * 100) +
        '% as much. Past <strong>' + U.fmtCount(be) + '</strong> tokens served, B is cheaper; at ' +
        U.fmtCount(served) + ' it uses <strong>' + Math.abs(Math.round(diff * 100)) + '% ' +
        (diff < 0 ? 'less' : 'more') + '</strong> lifetime compute (' + fmtFlops(cB) + ' vs ' + fmtFlops(cA) + ' FLOPs).';
    }

    var read = U.bindSliders(host, function () { draw(); });
    draw();
    return { resize: draw };
  }

  window.MODULE_CONFIG = {
    title: 'LLMs 0 to 100 - Module 5',
    manimSections: {
      'next-token': [
        'NextTokenScene_0000_sequence.mp4',
        'NextTokenScene_0001_shift.mp4',
        'NextTokenScene_0002_predict.mp4',
        'NextTokenScene_0003_target.mp4',
        'NextTokenScene_0004_loss.mp4'
      ],
      'sequence-packing': [
        'SequencePackingScene_0000_docs.mp4',
        'SequencePackingScene_0001_concat.mp4',
        'SequencePackingScene_0002_chop.mp4',
        'SequencePackingScene_0003_batch.mp4'
      ],
      'lr-schedule': [
        'LRScheduleScene_0000_axes.mp4',
        'LRScheduleScene_0001_warmup.mp4',
        'LRScheduleScene_0002_cosine.mp4',
        'LRScheduleScene_0003_annotate.mp4'
      ],
      'scaling-laws': [
        'ScalingLawScene_0000_powerlaw.mp4',
        'ScalingLawScene_0001_extrapolate.mp4',
        'ScalingLawScene_0002_chinchilla.mp4',
        'ScalingLawScene_0003_rule.mp4'
      ],
      'data-parallel': [
        'DataParallelScene_0000_replicas.mp4',
        'DataParallelScene_0001_split.mp4',
        'DataParallelScene_0002_localgrad.mp4',
        'DataParallelScene_0003_allreduce.mp4'
      ],
      'tensor-parallel': [
        'TensorParallelScene_0000_matmul.mp4',
        'TensorParallelScene_0001_split.mp4',
        'TensorParallelScene_0002_partial.mp4',
        'TensorParallelScene_0003_gather.mp4'
      ],
      'fsdp': [
        'FSDPScene_0000_copies.mp4',
        'FSDPScene_0001_shard.mp4',
        'FSDPScene_0002_batch.mp4',
        'FSDPScene_0003_gather.mp4',
        'FSDPScene_0004_free.mp4'
      ],
      'perplexity': [
        'PerplexityScene_0000_spread.mp4',
        'PerplexityScene_0001_perplexity.mp4',
        'PerplexityScene_0002_sharpen.mp4',
        'PerplexityScene_0003_bits.mp4'
      ]
    },
    widgets: {
      scalingPlanner: scalingPlanner,
      servingPlanner: servingPlanner
    }
  };
}());
