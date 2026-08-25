(function () {
  // ===================================================================
  // Module 3 interactive widgets. Each factory owns its host element and
  // redraws on resize / when its slide becomes visible. Registered via
  // MODULE_CONFIG.widgets, which deck.js merges into INTERACTIVE_WIDGETS.
  // ===================================================================

  var COL = {
    text: '#e8eaf0', muted: '#8892a4', line: '#2a3450',
    primary: '#4a9eff', secondary: '#f5a623', green: '#3fb950',
    red: '#e74c3c', purple: '#c792ea', bg: '#0d1225'
  };
  var SLOTC = ['#4a9eff', '#f5a623', '#50c878', '#c792ea'];

  function fit(canvas) {
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return null;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h };
  }

  // Keep slider/button interaction from leaking to reveal navigation.
  function stop(host) {
    ['pointerdown', 'keydown', 'click'].forEach(function (ev) {
      host.addEventListener(ev, function (e) { e.stopPropagation(); });
    });
  }

  function rounded(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function normalizeRows(M) {
    return M.map(function (row) {
      var s = row.reduce(function (a, b) { return a + b; }, 0);
      return row.map(function (v) { return v / s; });
    });
  }

  // ===================================================================
  // WIDGET: mlpRigid  — weights are bound to fixed slots
  // ===================================================================
  function mlpRigid(host) {
    var words = ['the', 'cat', 'sat', 'on', 'the', 'mat'];
    var trackIdx = 2; // "sat"
    var state = { start: 0 };

    host.innerHTML =
      '<div class="m3-widget">' +
        '<div class="m3-canvas-wrap"><canvas class="m3-canvas"></canvas></div>' +
        '<div class="m3-controls">' +
          '<button class="m3-btn m3-primary" data-act="slide">Slide window &rarr;</button>' +
          '<button class="m3-btn m3-reset" data-act="reset">Reset</button>' +
        '</div>' +
        '<p class="m3-readout"></p>' +
      '</div>';
    var canvas = host.querySelector('.m3-canvas');
    var readout = host.querySelector('.m3-readout');
    stop(host);

    host.querySelector('[data-act="slide"]').addEventListener('click', function () {
      state.start = (state.start + 1) % 3; draw();
    });
    host.querySelector('[data-act="reset"]').addEventListener('click', function () {
      state.start = 0; draw();
    });

    function draw() {
      var f = fit(canvas); if (!f) return;
      var ctx = f.ctx, W = f.w, H = f.h;
      ctx.clearRect(0, 0, W, H);
      ctx.textAlign = 'center';

      var n = words.length;
      var step = Math.min(120, (W - 80) / n);
      var x0 = (W - step * n) / 2 + step / 2;
      var bw = step - 14;

      // strip of full sentence
      var stripY = 64;
      ctx.font = '600 14px Inter, sans-serif';
      ctx.fillStyle = COL.muted;
      ctx.fillText('the model reads a fixed window of 4 tokens', W / 2, 20);
      // window rectangle
      var winX = x0 + state.start * step - step / 2 + 6;
      ctx.strokeStyle = COL.secondary; ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      rounded(ctx, winX, stripY - 22, step * 4 - 12, 44, 6); ctx.stroke();
      ctx.setLineDash([]);

      var slotCx = [];
      for (var i = 0; i < n; i++) {
        var cx = x0 + i * step;
        var inWin = (i >= state.start && i < state.start + 4);
        var slot = i - state.start;
        if (inWin) slotCx[slot] = cx;
        ctx.fillStyle = COL.bg;
        var tracked = (i === trackIdx);
        ctx.strokeStyle = tracked ? COL.secondary : (inWin ? COL.text : COL.line);
        ctx.lineWidth = tracked ? 2.5 : 1.4;
        rounded(ctx, cx - bw / 2, stripY - 16, bw, 32, 5); ctx.fill(); ctx.stroke();
        ctx.fillStyle = inWin ? COL.text : COL.muted;
        ctx.font = '600 14px Inter, sans-serif';
        ctx.fillText(words[i], cx, stripY + 5);
      }

      // weight blocks row
      var wbY = H - 86;
      ctx.font = '600 12px Inter, sans-serif';
      for (var s = 0; s < 4; s++) {
        if (slotCx[s] === undefined) continue;
        var c = slotCx[s];
        // arrow from windowed word down to weight block
        ctx.strokeStyle = SLOTC[s]; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(c, stripY + 18); ctx.lineTo(c, wbY - 2); ctx.stroke();
        ctx.fillStyle = 'rgba(255,255,255,0.04)';
        ctx.strokeStyle = SLOTC[s];
        rounded(ctx, c - 36, wbY, 72, 44, 6); ctx.fill(); ctx.stroke();
        ctx.fillStyle = SLOTC[s];
        ctx.fillText('slot ' + (s + 1), c, wbY + 18);
        ctx.fillText('W' + (s + 1), c, wbY + 36);
      }

      // tracked word annotation
      var trackSlot = trackIdx - state.start;
      ctx.font = '600 13px Inter, sans-serif';
      if (trackSlot >= 0 && trackSlot < 4) {
        var tc = slotCx[trackSlot];
        ctx.fillStyle = COL.secondary;
        ctx.fillText('"sat" here', tc, wbY - 16);
        readout.innerHTML = '"sat" now sits in <strong>slot ' + (trackSlot + 1) +
          '</strong>, processed by weight block <strong>W' + (trackSlot + 1) +
          '</strong>. Slide it and it meets entirely different parameters &mdash; nothing it learned transfers.';
      }
    }
    draw();
    return { resize: draw };
  }

  // ===================================================================
  // WIDGET: attentionHeatmap — NxN attention weights for a sentence
  // ===================================================================
  function attentionHeatmap(host) {
    var examples = {
      a: {
        tokens: ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        sel: 2,
        M: [
          [0.00, 0.34, 0.18, 0.12, 0.28, 0.08],
          [0.14, 0.00, 0.42, 0.16, 0.08, 0.20],
          [0.06, 0.46, 0.00, 0.30, 0.08, 0.10],
          [0.08, 0.12, 0.50, 0.00, 0.20, 0.10],
          [0.30, 0.08, 0.12, 0.22, 0.00, 0.28],
          [0.26, 0.24, 0.30, 0.12, 0.08, 0.00]
        ]
      },
      b: {
        tokens: ['alice', 'gave', 'bob', 'her', 'book'],
        sel: 3,
        M: [
          [0.00, 0.20, 0.25, 0.30, 0.25],
          [0.25, 0.00, 0.35, 0.20, 0.20],
          [0.20, 0.40, 0.00, 0.15, 0.25],
          [0.60, 0.15, 0.10, 0.00, 0.15],
          [0.35, 0.20, 0.10, 0.35, 0.00]
        ]
      }
    };
    var state = { key: 'a', sel: examples.a.sel };

    host.innerHTML =
      '<div class="m3-widget">' +
        '<div class="m3-canvas-wrap"><canvas class="m3-canvas"></canvas></div>' +
        '<div class="m3-controls">' +
          '<button class="m3-btn active" data-ex="a">"the cat sat on the mat"</button>' +
          '<button class="m3-btn" data-ex="b">"alice gave bob her book"</button>' +
        '</div>' +
        '<p class="m3-readout"></p>' +
      '</div>';
    var canvas = host.querySelector('.m3-canvas');
    var readout = host.querySelector('.m3-readout');
    stop(host);

    host.querySelectorAll('[data-ex]').forEach(function (b) {
      b.addEventListener('click', function () {
        state.key = b.getAttribute('data-ex');
        state.sel = examples[state.key].sel;
        host.querySelectorAll('[data-ex]').forEach(function (x) { x.classList.toggle('active', x === b); });
        draw();
      });
    });

    var layout = null;
    canvas.addEventListener('click', function (e) {
      if (!layout) return;
      var rect = canvas.getBoundingClientRect();
      // rescale the click into canvas layout pixels (reveal scales the deck).
      var sy = canvas.clientHeight / rect.height || 1;
      var my = (e.clientY - rect.top) * sy;
      var row = Math.floor((my - layout.oy) / layout.cell);
      if (row >= 0 && row < layout.n) { state.sel = row; draw(); }
    });

    function draw() {
      var f = fit(canvas); if (!f) return;
      var ctx = f.ctx, W = f.w, H = f.h;
      ctx.clearRect(0, 0, W, H);
      var ex = examples[state.key];
      var toks = ex.tokens, M = ex.M, n = toks.length;
      var pad = 64;
      var avail = Math.min(W - pad - 40, H - pad - 30);
      var cell = avail / n;
      var ox = pad + (W - pad - 40 - n * cell) / 2;
      var oy = pad;
      layout = { ox: ox, oy: oy, cell: cell, n: n };

      ctx.font = '12px Inter, sans-serif';
      // column (key) labels
      ctx.fillStyle = COL.secondary; ctx.textAlign = 'center';
      ctx.fillText('keys (attended to)', ox + n * cell / 2, 22);
      for (var j = 0; j < n; j++) {
        ctx.fillStyle = COL.secondary;
        ctx.fillText(toks[j], ox + j * cell + cell / 2, oy - 8);
      }
      // row (query) labels
      ctx.textAlign = 'right';
      for (var i = 0; i < n; i++) {
        ctx.fillStyle = (i === state.sel) ? COL.secondary : COL.primary;
        ctx.fillText(toks[i], ox - 8, oy + i * cell + cell / 2 + 4);
      }
      ctx.save();
      ctx.translate(ox - 58, oy + n * cell / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center'; ctx.fillStyle = COL.primary;
      ctx.fillText('queries', 0, 0);
      ctx.restore();

      // cells
      for (var r = 0; r < n; r++) {
        for (var c = 0; c < n; c++) {
          var w = M[r][c];
          ctx.fillStyle = (r === c)
            ? 'rgba(13,18,37,0.95)'
            : 'rgba(74,158,255,' + (0.08 + 0.9 * w).toFixed(3) + ')';
          ctx.fillRect(ox + c * cell + 1, oy + r * cell + 1, cell - 2, cell - 2);
          ctx.fillStyle = (r === c) ? COL.muted : COL.text; ctx.textAlign = 'center';
          ctx.font = '11px Inter, sans-serif';
          ctx.fillText(w.toFixed(2), ox + c * cell + cell / 2, oy + r * cell + cell / 2 + 4);
        }
      }
      // highlight selected query row
      ctx.strokeStyle = COL.secondary; ctx.lineWidth = 2.5;
      ctx.strokeRect(ox, oy + state.sel * cell, n * cell, cell);

      // readout: top attended key for selected query
      var row = M[state.sel];
      var best = 0; for (var k = 1; k < n; k++) if (row[k] > row[best]) best = k;
      readout.innerHTML = 'Click a row to pick a query. <strong>"' + toks[state.sel] +
        '"</strong> attends most to <strong>"' + toks[best] + '"</strong> (' +
        (row[best] * 100).toFixed(0) + '%). The diagonal is set to 0 here, and each row sums to 1.';
    }
    draw();
    return { resize: draw };
  }

  // ===================================================================
  // WIDGET: attentionArrows — click a word; arcs fan out to the words it
  // attends to. Thicker, brighter arc = more attention.
  // ===================================================================
  function attentionArrows(host) {
    var tokens = ['the', 'cat', 'sat', 'on', 'the', 'mat'];
    // rows = query (the word doing the attending), cols = key (attended to).
    // The diagonal is 0: a word does not attend to itself here, so every arc
    // points at a different word. normalizeRows then spreads each row's weight
    // across the other tokens.
    var M = normalizeRows([
      [0.00, 0.18, 0.10, 0.10, 0.22, 0.10],  // the
      [0.12, 0.00, 0.26, 0.06, 0.10, 0.16],  // cat
      [0.05, 0.46, 0.00, 0.09, 0.05, 0.15],  // sat
      [0.05, 0.10, 0.34, 0.00, 0.06, 0.25],  // on
      [0.10, 0.06, 0.10, 0.14, 0.00, 0.30],  // the
      [0.05, 0.16, 0.26, 0.18, 0.10, 0.00]   // mat
    ]);
    var state = { sel: 2 };

    host.innerHTML =
      '<div class="m3-widget">' +
        '<div class="m3-canvas-wrap"><canvas class="m3-canvas"></canvas></div>' +
        '<p class="m3-readout"></p>' +
      '</div>';
    var canvas = host.querySelector('.m3-canvas');
    var readout = host.querySelector('.m3-readout');
    stop(host);

    var boxes = [];
    canvas.addEventListener('click', function (e) {
      var rect = canvas.getBoundingClientRect();
      // reveal scales the whole deck, so the on-screen rect differs from the
      // canvas layout size; rescale the click into layout (drawing) pixels.
      var sx = canvas.clientWidth / rect.width || 1;
      var sy = canvas.clientHeight / rect.height || 1;
      var mx = (e.clientX - rect.left) * sx, my = (e.clientY - rect.top) * sy;
      for (var i = 0; i < boxes.length; i++) {
        var b = boxes[i];
        if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) {
          state.sel = i; draw(); return;
        }
      }
    });

    function head(ctx, fromx, fromy, tox, toy, color) {
      var ang = Math.atan2(toy - fromy, tox - fromx);
      var s = 8;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(tox, toy);
      ctx.lineTo(tox - s * Math.cos(ang - 0.45), toy - s * Math.sin(ang - 0.45));
      ctx.lineTo(tox - s * Math.cos(ang + 0.45), toy - s * Math.sin(ang + 0.45));
      ctx.closePath();
      ctx.fill();
    }

    function draw() {
      var f = fit(canvas); if (!f) return;
      var ctx = f.ctx, W = f.w, H = f.h;
      ctx.clearRect(0, 0, W, H);
      var n = tokens.length;
      ctx.textAlign = 'center';

      // Word row sits low; arcs bow up into the open space above it.
      ctx.font = '600 17px Inter, sans-serif';
      var bh = 40, rowY = H - 74, gap = 16;
      var widths = tokens.map(function (t) { return Math.max(66, ctx.measureText(t).width + 36); });
      var totalW = widths.reduce(function (a, b) { return a + b; }, 0) + gap * (n - 1);
      var x = (W - totalW) / 2;
      boxes = [];
      var cx = [];
      for (var i = 0; i < n; i++) {
        boxes.push({ x: x, y: rowY, w: widths[i], h: bh });
        cx.push(x + widths[i] / 2);
        x += widths[i] + gap;
      }

      var row = M[state.sel];
      var maxw = Math.max.apply(null, row);
      var topY = rowY - 4;
      var capBow = H - 116;

      // Arcs from the query to every other token.
      for (var k = 0; k < n; k++) {
        if (k === state.sel) continue;
        var w = row[k];
        var sx = cx[state.sel], ex = cx[k];
        var dist = Math.abs(ex - sx);
        var bow = Math.min(capBow, 46 + dist * 0.42);
        var midx = (sx + ex) / 2, cy = topY - bow;
        var col = 'rgba(74,158,255,' + (0.16 + 0.74 * (w / maxw)).toFixed(3) + ')';
        ctx.strokeStyle = col;
        ctx.lineWidth = 1 + 13 * w;
        ctx.beginPath();
        ctx.moveTo(sx, topY);
        ctx.quadraticCurveTo(midx, cy, ex, topY);
        ctx.stroke();
        head(ctx, midx, cy, ex, topY, col);
        if (w > 0.12) {
          ctx.fillStyle = COL.text; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
          ctx.fillText((w * 100).toFixed(0) + '%', ex, topY - 9);
        }
      }
      // Word boxes (query highlighted).
      for (var b2 = 0; b2 < n; b2++) {
        var on = (b2 === state.sel);
        ctx.fillStyle = on ? 'rgba(245,166,35,0.16)' : '#0d1225';
        ctx.strokeStyle = on ? COL.secondary : COL.line;
        ctx.lineWidth = on ? 2.5 : 1.4;
        rounded(ctx, boxes[b2].x, boxes[b2].y, boxes[b2].w, bh, 7);
        ctx.fill(); ctx.stroke();
        ctx.fillStyle = on ? COL.secondary : COL.text;
        ctx.font = '600 17px Inter, sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(tokens[b2], cx[b2], rowY + bh / 2 + 6);
      }
      ctx.fillStyle = COL.muted; ctx.font = '13px Inter, sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('query: "' + tokens[state.sel] + '"   (click any word)', W / 2, H - 14);

      var best = 0; for (var t = 0; t < n; t++) if (row[t] > row[best]) best = t;
      readout.innerHTML = '<strong>"' + tokens[state.sel] + '"</strong> attends most to <strong>"' +
        tokens[best] + '"</strong> (' + (row[best] * 100).toFixed(0) +
        '%). Thicker arrows carry more attention; the output for this word is a weighted blend of all the value vectors.';
    }
    draw();
    return { resize: draw };
  }

  // ===================================================================
  // WIDGET: permutationShuffle — order does not change attention
  // ===================================================================
  function permutationShuffle(host) {
    var ids = ['dog', 'bites', 'man'];
    // affinity by token identity (row-normalized), independent of position
    var AFF = normalizeRows([
      [0.20, 0.30, 0.50],  // dog -> dog, bites, man
      [0.45, 0.10, 0.45],  // bites
      [0.50, 0.30, 0.20]   // man
    ]);
    var perms = [[0, 1, 2], [2, 1, 0], [1, 2, 0], [2, 0, 1]];
    var state = { p: 0 };

    host.innerHTML =
      '<div class="m3-widget">' +
        '<div class="m3-canvas-wrap"><canvas class="m3-canvas"></canvas></div>' +
        '<div class="m3-controls">' +
          '<button class="m3-btn m3-primary" data-act="shuffle">Reorder tokens</button>' +
          '<button class="m3-btn m3-reset" data-act="reset">Reset</button>' +
        '</div>' +
        '<p class="m3-readout"></p>' +
      '</div>';
    var canvas = host.querySelector('.m3-canvas');
    var readout = host.querySelector('.m3-readout');
    stop(host);
    host.querySelector('[data-act="shuffle"]').addEventListener('click', function () {
      state.p = (state.p + 1) % perms.length; draw();
    });
    host.querySelector('[data-act="reset"]').addEventListener('click', function () {
      state.p = 0; draw();
    });

    function draw() {
      var f = fit(canvas); if (!f) return;
      var ctx = f.ctx, W = f.w, H = f.h;
      ctx.clearRect(0, 0, W, H);
      var perm = perms[state.p];
      var n = 3;
      var order = perm.map(function (k) { return ids[k]; });

      // token row
      ctx.textAlign = 'center';
      ctx.fillStyle = COL.muted; ctx.font = '13px Inter, sans-serif';
      ctx.fillText('current order', W / 2, 24);
      var tw = 96, gap = 18, totalW = n * tw + (n - 1) * gap;
      var tx = (W - totalW) / 2;
      for (var i = 0; i < n; i++) {
        ctx.fillStyle = COL.bg; ctx.strokeStyle = COL.primary; ctx.lineWidth = 1.5;
        rounded(ctx, tx + i * (tw + gap), 36, tw, 34, 6); ctx.fill(); ctx.stroke();
        ctx.fillStyle = COL.text; ctx.font = '600 15px Inter, sans-serif';
        ctx.fillText(order[i], tx + i * (tw + gap) + tw / 2, 58);
      }

      // attention matrix in current order
      var cell = Math.min(96, (H - 130) / n);
      var mW = n * cell;
      var ox = (W - mW) / 2, oy = 96 + ((H - 96) - (n * cell + 24)) / 2;
      ctx.font = '12px Inter, sans-serif';
      for (var j = 0; j < n; j++) {
        ctx.fillStyle = COL.secondary; ctx.textAlign = 'center';
        ctx.fillText(order[j], ox + j * cell + cell / 2, oy - 8);
        ctx.fillStyle = COL.primary; ctx.textAlign = 'right';
        ctx.fillText(order[j], ox - 8, oy + j * cell + cell / 2 + 4);
      }
      for (var r = 0; r < n; r++) {
        for (var c = 0; c < n; c++) {
          var w = AFF[perm[r]][perm[c]];
          ctx.fillStyle = 'rgba(74,158,255,' + (0.10 + 0.85 * w).toFixed(3) + ')';
          ctx.fillRect(ox + c * cell + 1, oy + r * cell + 1, cell - 2, cell - 2);
          ctx.fillStyle = COL.text; ctx.textAlign = 'center'; ctx.font = '11px Inter, sans-serif';
          ctx.fillText(w.toFixed(2), ox + c * cell + cell / 2, oy + r * cell + cell / 2 + 4);
        }
      }
      // highlight the dog->man pair wherever it now sits
      var dr = perm.indexOf(0), dc = perm.indexOf(2);
      ctx.strokeStyle = COL.secondary; ctx.lineWidth = 2.5;
      ctx.strokeRect(ox + dc * cell, oy + dr * cell, cell, cell);

      readout.innerHTML = 'attention("dog" &rarr; "man") = <strong>' +
        AFF[0][2].toFixed(2) + '</strong> in every ordering. Reorder the tokens and the value is unchanged &mdash; <strong>order carries no information</strong>.';
    }
    draw();
    return { resize: draw };
  }

  // ===================================================================
  // WIDGET: positionalEncoding — sinusoidal heatmap + distance signal
  // ===================================================================
  function positionalEncoding(host) {
    var state = { d: 32, exp: 4, L: 40 };

    host.innerHTML =
      '<div class="m3-widget">' +
        '<div class="m3-dual">' +
          '<div class="m3-canvas-wrap"><canvas class="m3-canvas pe-heat"></canvas></div>' +
          '<div class="m3-canvas-wrap"><canvas class="m3-canvas pe-sim"></canvas></div>' +
        '</div>' +
        '<div class="m3-sliders">' +
          '<div class="m3-slider"><label>d_model</label><input type="range" min="8" max="64" step="8" value="32" data-k="d"><p class="m3-val">32</p></div>' +
          '<div class="m3-slider"><label>base 10^</label><input type="range" min="2" max="5" step="1" value="4" data-k="exp"><p class="m3-val">4</p></div>' +
        '</div>' +
        '<p class="m3-readout"></p>' +
      '</div>';
    var heat = host.querySelector('.pe-heat');
    var sim = host.querySelector('.pe-sim');
    var readout = host.querySelector('.m3-readout');
    stop(host);
    host.querySelectorAll('input[type="range"]').forEach(function (inp) {
      inp.addEventListener('input', function () {
        state[inp.getAttribute('data-k')] = parseInt(inp.value, 10);
        inp.parentElement.querySelector('.m3-val').textContent = inp.value;
        draw();
      });
    });

    function pe(pos, i, d, base) {
      var denom = Math.pow(base, (2 * Math.floor(i / 2)) / d);
      var angle = pos / denom;
      return (i % 2 === 0) ? Math.sin(angle) : Math.cos(angle);
    }

    function draw() {
      var base = Math.pow(10, state.exp);
      var d = state.d, L = state.L;
      // ---- heatmap ----
      var f = fit(heat); if (!f) return;
      var ctx = f.ctx, W = f.w, H = f.h;
      ctx.clearRect(0, 0, W, H);
      ctx.textAlign = 'center'; ctx.fillStyle = COL.muted; ctx.font = '12px Inter, sans-serif';
      ctx.fillText('encoding value (x = sequence position, y = dimension)', W / 2, 16);
      var ox = 46, oy = 50, gw = W - ox - 14, gh = H - oy - 30;
      // legend for the diverging colour scale (sits between the title and the grid)
      var lgy = 30;
      ctx.textAlign = 'left'; ctx.font = '11px Inter, sans-serif';
      ctx.fillStyle = 'rgba(74,158,255,0.85)'; ctx.fillRect(ox, lgy, 16, 11);
      ctx.fillStyle = COL.muted; ctx.fillText('positive', ox + 22, lgy + 9);
      ctx.fillStyle = 'rgba(245,166,35,0.85)'; ctx.fillRect(ox + 92, lgy, 16, 11);
      ctx.fillStyle = COL.muted; ctx.fillText('negative', ox + 114, lgy + 9);
      var cw = gw / L, ch = gh / d;
      for (var p = 0; p < L; p++) {
        for (var i = 0; i < d; i++) {
          var v = pe(p, i, d, base);
          ctx.fillStyle = v >= 0
            ? 'rgba(74,158,255,' + (0.12 + 0.85 * v).toFixed(3) + ')'
            : 'rgba(245,166,35,' + (0.12 + 0.85 * (-v)).toFixed(3) + ')';
          ctx.fillRect(ox + p * cw, oy + i * ch, cw + 0.5, ch + 0.5);
        }
      }
      ctx.fillStyle = COL.muted; ctx.textAlign = 'right';
      ctx.fillText('dim 0', ox - 4, oy + 8);
      ctx.fillText('dim ' + (d - 1), ox - 4, oy + gh);
      ctx.textAlign = 'center';
      ctx.fillText('pos 0', ox + cw / 2 + 4, H - 8);
      ctx.fillText('pos ' + (L - 1), ox + gw - cw, H - 8);

      // ---- similarity vs distance ----
      var g = fit(sim); if (!g) return;
      var c2 = g.ctx, W2 = g.w, H2 = g.h;
      c2.clearRect(0, 0, W2, H2);
      c2.textAlign = 'center'; c2.fillStyle = COL.muted; c2.font = '12px Inter, sans-serif';
      c2.fillText('similarity to position 0 as distance grows', W2 / 2, 16);
      var px = 40, py = 28, pw = W2 - px - 16, ph = H2 - py - 28;
      // axes
      c2.strokeStyle = COL.line; c2.lineWidth = 1;
      c2.beginPath(); c2.moveTo(px, py); c2.lineTo(px, py + ph); c2.lineTo(px + pw, py + ph); c2.stroke();
      // compute dot products PE(0).PE(k)
      var vals = [];
      for (var k = 0; k < L; k++) {
        var dot = 0;
        for (var ii = 0; ii < d; ii++) dot += pe(0, ii, d, base) * pe(k, ii, d, base);
        vals.push(dot);
      }
      var maxv = vals[0] || 1;
      c2.strokeStyle = COL.primary; c2.lineWidth = 2;
      c2.beginPath();
      for (var kk = 0; kk < L; kk++) {
        var xx = px + (kk / (L - 1)) * pw;
        var yy = py + ph - (vals[kk] / maxv) * ph;
        if (kk === 0) c2.moveTo(xx, yy); else c2.lineTo(xx, yy);
      }
      c2.stroke();
      c2.fillStyle = COL.muted; c2.textAlign = 'left';
      c2.fillText('0', px, py + ph + 16);
      c2.textAlign = 'right'; c2.fillText('distance ' + (L - 1), px + pw, py + ph + 16);
      c2.save();
      c2.translate(12, py + ph / 2);
      c2.rotate(-Math.PI / 2);
      c2.textAlign = 'center';
      c2.fillText('similarity', 0, 0);
      c2.restore();

      readout.innerHTML = 'd_model = <strong>' + d + '</strong>, base = <strong>' + base.toLocaleString() +
        '</strong>. The heatmap shows positions across the x-axis and dimensions down the y-axis. The right plot compares each position to position 0; nearby positions tend to be more similar, giving attention a distance signal.';
    }
    draw();
    return { resize: draw };
  }

  // ===================================================================
  // WIDGET: softmaxExplorer — logits to a distribution, with temperature
  // ===================================================================
  function softmaxExplorer(host) {
    var toks = ['mat', 'rug', 'sofa', 'roof', 'sky'];
    var state = { logits: [3.2, 2.6, 1.4, 0.9, -0.3], T: 1.0 };

    var logitHTML = '';
    toks.forEach(function (t, i) {
      logitHTML += '<div class="m3-slider"><label>"' + t + '"</label>' +
        '<input type="range" min="-2" max="6" step="0.1" value="' + state.logits[i] + '" data-i="' + i + '">' +
        '<p class="m3-val">' + state.logits[i].toFixed(1) + '</p></div>';
    });
    var tempHTML = '<div class="m3-slider"><label>temperature</label>' +
      '<input type="range" min="0.2" max="10" step="0.1" value="1" data-t="1"><p class="m3-val">1.0</p></div>';

    host.innerHTML =
      '<div class="m3-widget">' +
        '<div class="m3-canvas-wrap"><canvas class="m3-canvas"></canvas></div>' +
        '<div class="m3-sliders m3-logits">' + logitHTML + '</div>' +
        '<div class="m3-sliders m3-temp">' + tempHTML + '</div>' +
        '<p class="m3-readout"></p>' +
      '</div>';
    var canvas = host.querySelector('.m3-canvas');
    var readout = host.querySelector('.m3-readout');
    stop(host);
    host.querySelectorAll('input[data-i]').forEach(function (inp) {
      inp.addEventListener('input', function () {
        state.logits[parseInt(inp.getAttribute('data-i'), 10)] = parseFloat(inp.value);
        inp.parentElement.querySelector('.m3-val').textContent = parseFloat(inp.value).toFixed(1);
        draw();
      });
    });
    host.querySelector('input[data-t]').addEventListener('input', function () {
      state.T = parseFloat(this.value);
      this.parentElement.querySelector('.m3-val').textContent = state.T.toFixed(1);
      draw();
    });

    function softmax(z, T) {
      var m = Math.max.apply(null, z);
      var ex = z.map(function (v) { return Math.exp((v - m) / T); });
      var s = ex.reduce(function (a, b) { return a + b; }, 0);
      return ex.map(function (v) { return v / s; });
    }

    function draw() {
      var f = fit(canvas); if (!f) return;
      var ctx = f.ctx, W = f.w, H = f.h;
      ctx.clearRect(0, 0, W, H);
      var n = toks.length;
      var p = softmax(state.logits, state.T);
      var colW = (W - 80) / n;
      var midY = H / 2;
      ctx.textAlign = 'center';

      // top: logits ; bottom: probabilities
      ctx.font = '12px Inter, sans-serif'; ctx.fillStyle = COL.muted;
      ctx.fillText('logits (raw scores)', 70, 18);
      ctx.fillText('probabilities (after softmax)', 110, midY + 16);

      var maxL = 6, minL = -2;
      var topH = midY - 40;
      for (var i = 0; i < n; i++) {
        var cx = 60 + i * colW + colW / 2;
        // logit bar (above midline)
        var lh = ((state.logits[i] - minL) / (maxL - minL)) * topH;
        ctx.fillStyle = 'rgba(245,166,35,0.55)';
        ctx.fillRect(cx - colW * 0.3, midY - 24 - lh, colW * 0.6, lh);
        ctx.fillStyle = COL.text; ctx.font = '11px Inter, sans-serif';
        ctx.fillText(state.logits[i].toFixed(1), cx, midY - 28 - lh);
        // token label at midline
        ctx.fillStyle = COL.primary; ctx.font = '600 13px Inter, sans-serif';
        ctx.fillText(toks[i], cx, midY - 6);
        // prob bar (below midline)
        var botH = H - midY - 50;
        var ph = p[i] * botH;
        ctx.fillStyle = 'rgba(74,158,255,' + (0.35 + 0.6 * p[i]).toFixed(3) + ')';
        ctx.fillRect(cx - colW * 0.3, midY + 24, colW * 0.6, ph);
        ctx.fillStyle = COL.text; ctx.font = '11px Inter, sans-serif';
        ctx.fillText((p[i] * 100).toFixed(0) + '%', cx, midY + 24 + ph + 14);
      }
      ctx.strokeStyle = COL.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(30, midY); ctx.lineTo(W - 20, midY); ctx.stroke();

      var top = 0; for (var k = 1; k < n; k++) if (p[k] > p[top]) top = k;
      var spread = state.T >= 4
        ? ' At high temperature the probabilities flatten toward uniform (' + (100 / n).toFixed(0) + '% each).'
        : ' Lower temperature sharpens the peak; higher temperature flattens it.';
      readout.innerHTML = 'Temperature <strong>' + state.T.toFixed(1) + '</strong>: top token is <strong>"' +
        toks[top] + '"</strong> at <strong>' + (p[top] * 100).toFixed(0) + '%</strong>.' + spread;
    }
    draw();
    return { resize: draw };
  }

  // ===================================================================
  // WIDGET: kvCache — recompute every step vs cache and reuse
  // ===================================================================
  function kvCache(host) {
    var toks = ['the', 'cat', 'sat', 'on', 'the', 'mat'];
    var N = toks.length;
    var state = { t: 1 };

    host.innerHTML =
      '<div class="m3-widget">' +
        '<div class="m3-canvas-wrap"><canvas class="m3-canvas"></canvas></div>' +
        '<div class="m3-controls">' +
          '<button class="m3-btn m3-primary" data-act="next">Generate next token</button>' +
          '<button class="m3-btn m3-reset" data-act="reset">Reset</button>' +
        '</div>' +
        '<p class="m3-readout"></p>' +
      '</div>';
    var canvas = host.querySelector('.m3-canvas');
    var readout = host.querySelector('.m3-readout');
    stop(host);
    host.querySelector('[data-act="next"]').addEventListener('click', function () {
      if (state.t < N) state.t++; draw();
    });
    host.querySelector('[data-act="reset"]').addEventListener('click', function () {
      state.t = 1; draw();
    });

    function panel(ctx, x0, y0, pw, ph, title, cached) {
      ctx.textAlign = 'center';
      ctx.fillStyle = cached ? COL.green : COL.secondary;
      ctx.font = '600 14px Inter, sans-serif';
      ctx.fillText(title, x0 + pw / 2, y0);
      var cell = Math.min((pw - 30) / N, (ph - 50) / N);
      var gx = x0 + (pw - cell * N) / 2, gy = y0 + 28;
      // column labels (positions)
      ctx.font = '10px Inter, sans-serif'; ctx.fillStyle = COL.muted;
      for (var c = 0; c < N; c++) ctx.fillText(toks[c], gx + c * cell + cell / 2, gy - 5);
      var computed = 0;
      for (var s = 0; s < state.t; s++) {
        for (var c2 = 0; c2 <= s; c2++) {
          var bright, fill;
          if (!cached) { bright = (s === state.t - 1); fill = 'rgba(245,166,35,' + (bright ? 0.7 : 0.28) + ')'; computed++; }
          else {
            if (c2 === s) { fill = 'rgba(63,185,80,0.7)'; computed++; }
            else fill = 'rgba(136,146,164,0.18)';
          }
          ctx.fillStyle = fill;
          ctx.fillRect(gx + c2 * cell + 1, gy + s * cell + 1, cell - 2, cell - 2);
        }
      }
      // grid frame
      ctx.strokeStyle = COL.line; ctx.lineWidth = 1;
      ctx.strokeRect(gx, gy, cell * N, cell * state.t);
      ctx.fillStyle = COL.muted; ctx.font = '9px Inter, sans-serif'; ctx.textAlign = 'left';
      ctx.fillText('step', gx - 2, gy + cell * state.t + 12);
      return computed;
    }

    function draw() {
      var f = fit(canvas); if (!f) return;
      var ctx = f.ctx, W = f.w, H = f.h;
      ctx.clearRect(0, 0, W, H);
      ctx.textAlign = 'center'; ctx.fillStyle = COL.muted; ctx.font = '12px Inter, sans-serif';
      ctx.fillText('rows = generation step, columns = token position (orange/green = K,V computed this step)', W / 2, 18);
      var pw = (W - 60) / 2, ph = H - 70, y0 = 44;
      var noCache = panel(ctx, 20, y0, pw, ph, 'Without cache: recompute all', false);
      var withCache = panel(ctx, 40 + pw, y0, pw, ph, 'With cache: compute only the new one', true);
      readout.innerHTML = 'After <strong>' + state.t + '</strong> tokens, K,V projections computed: ' +
        '<strong>' + noCache + '</strong> without cache vs <strong>' + withCache +
        '</strong> with cache. Without caching, every past token is re-projected at every step.';
    }
    draw();
    return { resize: draw };
  }

  window.MODULE_CONFIG = {
    title: 'LLMs 0 to 100 - Module 3',
    manimSections: {},
    widgets: {
      mlpRigid: mlpRigid,
      attentionArrows: attentionArrows,
      attentionHeatmap: attentionHeatmap,
      permutationShuffle: permutationShuffle,
      positionalEncoding: positionalEncoding,
      softmaxExplorer: softmaxExplorer,
      kvCache: kvCache
    }
  };
}());
