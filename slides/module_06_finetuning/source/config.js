(function () {
  // Module 6 config. Manim steppers are keyed by the `scene` slug used in the
  // :::manim fences; each maps to its ordered list of section clips.
  // ===================================================================
  // WIDGET: lossMask — toggle which chat-template spans contribute loss
  // and watch the trained-token count (and the taught behavior) change.
  // ===================================================================
  function lossMask(host) {
    var U = WIDGET_UTIL;
    // One two-turn conversation, already rendered in a chat template.
    var SPANS = [
      { kind: 'special', text: '<|system|>' },
      { kind: 'prompt', text: 'You are a helpful assistant.' },
      { kind: 'special', text: '<|user|>' },
      { kind: 'prompt', text: 'What is the capital of France?' },
      { kind: 'special', text: '<|assistant|>' },
      { kind: 'response', turn: 1, text: 'The capital of France is Paris.' },
      { kind: 'special', text: '<|user|>' },
      { kind: 'prompt', text: 'And of Japan?' },
      { kind: 'special', text: '<|assistant|>' },
      { kind: 'response', turn: 2, text: 'The capital of Japan is Tokyo.' }
    ];
    var state = { mode: 'all' };  // all | last | none

    host.innerHTML =
      '<div class="iw">' +
        '<div class="iw-body iw-hug">' +
          '<div class="iw-panel">' +
            '<h4>The training example, token by token</h4>' +
            '<div class="iw-tokens" data-el="seq"></div>' +
          '</div>' +
        '</div>' +
        '<div class="iw-stats">' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="tot">0</span><span class="iw-stat-lab">tokens in sequence</span></div>' +
          '<div class="iw-stat good"><span class="iw-stat-num" data-el="trained">0</span><span class="iw-stat-lab">tokens with loss</span></div>' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="pct">0</span><span class="iw-stat-lab">of the sequence</span></div>' +
        '</div>' +
        '<div class="iw-controls">' +
          '<span class="iw-label">loss on:</span>' +
          '<button class="iw-btn active" data-pick data-val="all">every assistant turn</button>' +
          '<button class="iw-btn" data-pick data-val="last">the last turn only</button>' +
          '<button class="iw-btn" data-pick data-val="none">no mask (whole sequence)</button>' +
        '</div>' +
        '<p class="iw-readout"></p>' +
      '</div>';
    U.stop(host);
    var seq = host.querySelector('[data-el="seq"]');
    var readout = host.querySelector('.iw-readout');
    var el = {};
    host.querySelectorAll('[data-el]').forEach(function (n) { el[n.getAttribute('data-el')] = n; });

    // Rough token split: words plus punctuation, enough to make the point.
    function tokens(text) { return text.match(/\S+/g) || []; }

    function trained(span) {
      if (state.mode === 'none') return true;
      if (span.kind !== 'response') return false;
      if (state.mode === 'all') return true;
      return span.turn === 2;
    }

    function draw() {
      var total = 0, hit = 0, html = '';
      SPANS.forEach(function (span) {
        var on = trained(span);
        tokens(span.text).forEach(function (t) {
          total++;
          if (on) hit++;
          var cls = on ? 'good' : (span.kind === 'special' ? 'hot' : 'dim');
          html += '<span class="iw-chip ' + cls + '">' + t.replace(/</g, '&lt;') + '</span>';
        });
      });
      seq.innerHTML = html;
      el.tot.textContent = total;
      el.trained.textContent = hit;
      el.pct.textContent = Math.round((hit / total) * 100) + '%';

      readout.innerHTML =
        state.mode === 'all'
          ? 'Green tokens carry loss; grey ones are set to <strong>-100</strong> and skipped. The prompt is still <strong>read</strong> as context, it is just never a target.'
          : state.mode === 'last'
            ? 'Training only the final turn wastes the earlier response: the same forward pass could have supplied <strong>' +
              (function () { var a = 0; SPANS.forEach(function (s) { if (s.kind === 'response' && s.turn === 1) a += tokens(s.text).length; }); return a; }()) +
              '</strong> more supervised tokens for free.'
            : 'With no mask the model is trained to produce <strong>user turns and system prompts too</strong>. At generation time it invents its own questions and answers them.';
    }

    U.bindToggle(host, '[data-pick]', function (val) { state.mode = val; draw(); });
    draw();
    return { resize: draw };
  }

  // ===================================================================
  // WIDGET: loraCalculator — rank r and the alpha/r scale against a full
  // weight matrix, in parameters, optimizer memory, and checkpoint size.
  // ===================================================================
  function loraCalculator(host) {
    var U = WIDGET_UTIL, COL = WIDGET_UTIL.COL;

    host.innerHTML =
      '<div class="iw">' +
        '<div class="iw-canvas-wrap"><canvas class="iw-canvas"></canvas></div>' +
        '<div class="iw-stats">' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="base">0</span><span class="iw-stat-lab">frozen parameters</span></div>' +
          '<div class="iw-stat good"><span class="iw-stat-num" data-el="tr">0</span><span class="iw-stat-lab">trainable parameters</span></div>' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="pct">0</span><span class="iw-stat-lab">trainable share</span></div>' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="ckpt">0</span><span class="iw-stat-lab">checkpoint per task</span></div>' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="scale">0</span><span class="iw-stat-lab">scale &alpha; / r</span></div>' +
        '</div>' +
        '<div class="iw-sliders">' +
          U.sliderHTML('d', 'hidden size d', 512, 8192, 256, 4096) +
          U.sliderHTML('r', 'rank r', 1, 256, 1, 8) +
          U.sliderHTML('alpha', 'alpha', 1, 256, 1, 16) +
          U.sliderHTML('L', 'layers', 4, 96, 4, 32) +
        '</div>' +
        '<p class="iw-readout"></p>' +
      '</div>';
    U.stop(host);
    var canvas = host.querySelector('.iw-canvas');
    var readout = host.querySelector('.iw-readout');
    var el = {};
    host.querySelectorAll('[data-el]').forEach(function (n) { el[n.getAttribute('data-el')] = n; });

    function draw() {
      var v = read();
      var d = v.d, r = v.r, L = v.L, alpha = v.alpha;
      var scale = alpha / r;
      // Adapters on the four attention projections, the common default.
      var perMat = d * d, mats = 4 * L;
      var base = perMat * mats;
      var lora = (2 * d * r) * mats;

      var f = U.fit(canvas); if (!f) return;
      var ctx = f.ctx, W = f.w, H = f.h;
      ctx.clearRect(0, 0, W, H);
      ctx.textAlign = 'center';
      ctx.font = '12px Inter, sans-serif';
      ctx.fillStyle = COL.muted;
      ctx.fillText('one attention projection, drawn to scale: W is d x d, A is r x d, B is d x r', W / 2, 18);

      var side = Math.min(H - 124, (W - 200) / 2.6);
      var wx = W / 2 - side - 90, wy = 40;

      ctx.fillStyle = 'rgba(136,146,164,0.14)';
      ctx.strokeStyle = COL.muted; ctx.lineWidth = 1.5;
      U.rounded(ctx, wx, wy, side, side, 4); ctx.fill(); ctx.stroke();
      ctx.fillStyle = COL.muted;
      ctx.fillText('W (frozen)', wx + side / 2, wy + side / 2);
      ctx.fillText(d + ' x ' + d, wx + side / 2, wy + side / 2 + 18);

      ctx.fillStyle = COL.text;
      ctx.font = '20px Inter, sans-serif';
      ctx.fillText('+', wx + side + 30, wy + side / 2 + 6);
      // The alpha/r scale multiplies the low-rank product, not W.
      ctx.fillStyle = COL.secondary;
      ctx.font = '600 13px Inter, sans-serif';
      ctx.fillText('x ' + scale.toFixed(2), wx + side + 30, wy + side / 2 + 32);

      // B (d x r) then A (r x d), both drawn at the same pixels-per-unit as W.
      var unit = side / d;
      var bw = Math.max(3, r * unit), ah = Math.max(3, r * unit);
      var bx = wx + side + 68, by = wy;
      ctx.font = '12px Inter, sans-serif';
      ctx.fillStyle = 'rgba(63,185,80,0.35)';
      ctx.strokeStyle = COL.green; ctx.lineWidth = 1.5;
      U.rounded(ctx, bx, by, bw, side, 3); ctx.fill(); ctx.stroke();
      ctx.fillStyle = COL.green;
      ctx.fillText('B', bx + bw / 2, by + side + 16);

      var ax = bx + bw + 30, ay = wy + side / 2 - ah / 2;
      ctx.fillStyle = 'rgba(74,158,255,0.35)';
      ctx.strokeStyle = COL.primary;
      U.rounded(ctx, ax, ay, side, ah, 3); ctx.fill(); ctx.stroke();
      ctx.fillStyle = COL.primary;
      ctx.fillText('A', ax + side / 2, ay + ah + 16);

      ctx.fillStyle = COL.muted;
      ctx.fillText('trainable: 2 x d x r = ' + U.fmtCount(2 * d * r) + ' per matrix', ax + side / 2, wy + side + 40);
      ctx.fillStyle = COL.text;
      ctx.font = '600 15px Inter, sans-serif';
      ctx.fillText("W' = W + (" + alpha + ' / ' + r + ') BA = W + ' + scale.toFixed(2) + ' BA',
        W / 2, wy + side + 70);

      el.base.textContent = U.fmtCount(base);
      el.tr.textContent = U.fmtCount(lora);
      el.pct.textContent = (100 * lora / base).toFixed(lora / base < 0.001 ? 3 : 2) + '%';
      el.ckpt.textContent = U.fmtBytes(lora * 2 / 1e9);   // bf16 adapter weights
      el.scale.textContent = scale.toFixed(2);

      U.setVal(host, 'd', String(d));
      U.setVal(host, 'r', String(r));
      U.setVal(host, 'alpha', String(alpha));
      U.setVal(host, 'L', String(L));

      // AdamW keeps two fp32 moments per trainable parameter.
      var optFull = base * 8 / 1e9, optLora = lora * 8 / 1e9;
      readout.innerHTML = 'Optimizer state drops from <strong>' + U.fmtBytes(optFull) + '</strong> to <strong>' +
        U.fmtBytes(optLora) + '</strong>, and each task ships as a <strong>' + U.fmtBytes(lora * 2 / 1e9) +
        '</strong> adapter over one shared base.';
    }

    var read = U.bindSliders(host, draw);
    draw();
    return { resize: draw };
  }

  window.MODULE_CONFIG = {
    title: 'LLMs 0 to 100 - Module 6',
    manimSections: {
      'chat-template': [
        'ChatTemplateScene_0000_turns.mp4',
        'ChatTemplateScene_0001_markers.mp4',
        'ChatTemplateScene_0002_flatten.mp4',
        'ChatTemplateScene_0003_highlight.mp4'
      ],
      'loss-mask': [
        'LossMaskScene_0000_row.mp4',
        'LossMaskScene_0001_predict.mp4',
        'LossMaskScene_0002_mask.mp4',
        'LossMaskScene_0003_loss.mp4'
      ],
      'lora': [
        'LoRAScene_0000_weight.mp4',
        'LoRAScene_0001_freeze.mp4',
        'LoRAScene_0002_lowrank.mp4',
        'LoRAScene_0003_rank.mp4',
        'LoRAScene_0004_merge.mp4'
      ]
    },
    widgets: {
      lossMask: lossMask,
      loraCalculator: loraCalculator
    }
  };
}());
