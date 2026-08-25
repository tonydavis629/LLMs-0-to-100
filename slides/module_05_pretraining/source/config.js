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
      'training-loop': [
        'TrainingLoopScene_0000_setup.mp4',
        'TrainingLoopScene_0001_forward.mp4',
        'TrainingLoopScene_0002_backward.mp4',
        'TrainingLoopScene_0003_update.mp4',
        'TrainingLoopScene_0004_descend.mp4'
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
      scalingPlanner: scalingPlanner
    }
  };
}());
