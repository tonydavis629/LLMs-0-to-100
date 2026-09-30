(function () {
  // Module 4 config. Manim steppers are keyed by the `scene` slug used in the
  // :::manim fences; each maps to its ordered list of section clips.
  // ===================================================================
  // WIDGET: samplingExplorer — temperature, top-k and top-p over one
  // real-shaped next-token distribution, with live sampling.
  // ===================================================================
  function samplingExplorer(host) {
    var U = WIDGET_UTIL, COL = WIDGET_UTIL.COL;
    var TOKENS = ['Paris', 'the', 'a', 'located', 'now', 'home', 'known', 'one', 'situated', 'France', 'Lyon', 'called'];
    var LOGITS = [6.6, 5.4, 5.1, 4.6, 4.0, 3.7, 3.4, 3.0, 2.6, 2.2, 1.9, 1.5];
    var state = { T: 1.0, k: 12, p: 1.0, draws: null };

    host.innerHTML =
      '<div class="iw">' +
        '<div class="iw-canvas-wrap"><canvas class="iw-canvas"></canvas></div>' +
        '<div class="iw-sliders">' +
          U.sliderHTML('T', 'temperature', 0.1, 2.0, 0.05, 1.0) +
          U.sliderHTML('k', 'top-k', 1, 12, 1, 12) +
          U.sliderHTML('p', 'top-p', 0.05, 1.0, 0.01, 1.0) +
        '</div>' +
        '<div class="iw-controls">' +
          '<button class="iw-btn iw-primary" data-act="sample">Draw 200 samples</button>' +
          '<button class="iw-btn iw-reset" data-act="reset">Reset</button>' +
        '</div>' +
        '<p class="iw-readout"></p>' +
      '</div>';
    U.stop(host);
    var canvas = host.querySelector('.iw-canvas');
    var readout = host.querySelector('.iw-readout');

    // The serving-time order: temperature first, then truncate, then renormalize.
    function distribution() {
      var probs = U.softmax(LOGITS, state.T);
      var order = probs.map(function (v, i) { return i; })
        .sort(function (a, b) { return probs[b] - probs[a]; });
      var kept = {}, cum = 0;
      for (var r = 0; r < order.length; r++) {
        if (r >= state.k) break;
        kept[order[r]] = true;
        cum += probs[order[r]];
        if (cum >= state.p) break;   // top-p stops once the nucleus is covered
      }
      var mass = 0;
      probs.forEach(function (v, i) { if (kept[i]) mass += v; });
      var final = probs.map(function (v, i) { return kept[i] ? v / mass : 0; });
      return { raw: probs, kept: kept, final: final, size: Object.keys(kept).length };
    }

    function draw() {
      var f = U.fit(canvas); if (!f) return;
      var ctx = f.ctx, W = f.w, H = f.h;
      ctx.clearRect(0, 0, W, H);
      var d = distribution();

      // Tokens are already sorted by score, so cumulative mass reads left to right.
      var padL = 96, padR = 56, padT = 34, padB = 78;
      var plotW = W - padL - padR, plotH = H - padT - padB;
      var n = TOKENS.length;
      var step = plotW / n, bw = Math.min(50, step - 14);
      var yBase = padT + plotH;
      var maxP = Math.max.apply(null, d.final.concat(d.raw));
      var scale = plotH / (maxP * 1.18);
      var xOf = function (i) { return padL + step * i + step / 2; };

      ctx.font = '12px Inter, sans-serif';
      ctx.fillStyle = COL.muted; ctx.textAlign = 'left';
      ctx.fillText('bars: probability after temperature (outline) and after the cut, renormalized (filled)', padL, 16);
      ctx.textAlign = 'right'; ctx.fillStyle = COL.secondary;
      ctx.fillText('orange line: cumulative probability', W - padR, 16);

      // Right axis for the cumulative line (0 to 100%).
      var cy = function (c) { return yBase - c * plotH; };
      ctx.strokeStyle = COL.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(W - padR + 6, cy(0)); ctx.lineTo(W - padR + 6, cy(1)); ctx.stroke();
      ctx.fillStyle = COL.secondary; ctx.textAlign = 'left'; ctx.font = '11px Inter, sans-serif';
      [0, 0.5, 1].forEach(function (c) { ctx.fillText(Math.round(c * 100) + '%', W - padR + 10, cy(c) + 4); });

      var cum = 0, cums = [];
      for (var i = 0; i < n; i++) {
        var cx = xOf(i);
        var hRaw = d.raw[i] * scale;
        ctx.strokeStyle = 'rgba(136,146,164,0.55)'; ctx.lineWidth = 1.2;
        ctx.strokeRect(cx - bw / 2, yBase - hRaw, bw, hRaw);
        var hFin = d.final[i] * scale;
        if (hFin > 0.5) {
          ctx.fillStyle = 'rgba(74,158,255,0.75)';
          U.rounded(ctx, cx - bw / 2, yBase - hFin, bw, hFin, 3);
          ctx.fill();
        }
        if (state.draws) {
          var hs = (state.draws[i] / state.draws.total) * scale;
          if (hs > 0.5) {
            ctx.fillStyle = 'rgba(63,185,80,0.9)';
            ctx.fillRect(cx - bw / 2 + bw * 0.3, yBase - hs, bw * 0.4, hs);
          }
        }
        // Probability label on top of each bar.
        var shown = d.kept[i] ? d.final[i] : d.raw[i];
        ctx.textAlign = 'center';
        ctx.fillStyle = d.kept[i] ? COL.text : COL.muted;
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText((shown * 100).toFixed(shown < 0.1 ? 1 : 0) + '%', cx, yBase - Math.max(hRaw, hFin) - 6);
        cum += d.raw[i]; cums.push(cum);
      }

      // Cumulative probability (before truncation): the quantity top-p thresholds.
      ctx.strokeStyle = COL.secondary; ctx.lineWidth = 2;
      ctx.beginPath();
      for (i = 0; i < n; i++) {
        if (i === 0) ctx.moveTo(xOf(i), cy(cums[i])); else ctx.lineTo(xOf(i), cy(cums[i]));
      }
      ctx.stroke();
      for (i = 0; i < n; i++) {
        ctx.fillStyle = COL.secondary;
        ctx.beginPath(); ctx.arc(xOf(i), cy(cums[i]), 3, 0, 2 * Math.PI); ctx.fill();
      }
      if (state.p < 1) {
        ctx.setLineDash([5, 4]); ctx.strokeStyle = 'rgba(245,166,35,0.7)'; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(padL, cy(state.p)); ctx.lineTo(W - padR, cy(state.p)); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COL.secondary; ctx.textAlign = 'right'; ctx.font = '12px Inter, sans-serif';
        ctx.fillText('p = ' + state.p.toFixed(2), W - padR - 4, cy(state.p) - 6);
      }

      // Cut line: everything to the right of it can never be sampled.
      if (d.size < n) {
        var xc = padL + step * d.size;
        ctx.setLineDash([6, 4]); ctx.strokeStyle = COL.text; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(xc, padT - 4); ctx.lineTo(xc, yBase + 58); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(136,146,164,0.08)';
        ctx.fillRect(xc, padT - 4, W - padR - xc, plotH + 4);
        ctx.fillStyle = COL.text; ctx.textAlign = 'left'; ctx.font = '600 12px Inter, sans-serif';
        var why = (state.k <= d.size && state.k < n) ? 'top-k cut' : 'top-p cut';
        ctx.fillText(why + ': never sampled', xc + 6, padT + 10);
      }

      ctx.strokeStyle = COL.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, yBase); ctx.lineTo(W - padR, yBase); ctx.stroke();

      // Table under the axis: token, raw score (logit), cumulative probability.
      var rows = [['token', yBase + 18], ['score', yBase + 38], ['cumulative', yBase + 58]];
      ctx.textAlign = 'right'; ctx.fillStyle = COL.muted; ctx.font = '12px Inter, sans-serif';
      rows.forEach(function (r) { ctx.fillText(r[0], padL - 12, r[1]); });
      for (i = 0; i < n; i++) {
        ctx.textAlign = 'center';
        ctx.fillStyle = d.kept[i] ? COL.text : COL.muted;
        ctx.font = (d.kept[i] ? '600 ' : '') + '12px Inter, sans-serif';
        ctx.fillText(TOKENS[i], xOf(i), rows[0][1]);
        ctx.font = '12px Inter, sans-serif';
        ctx.fillStyle = COL.muted;
        ctx.fillText((LOGITS[i] / state.T).toFixed(1), xOf(i), rows[1][1]);
        ctx.fillStyle = COL.secondary;
        ctx.fillText(Math.round(cums[i] * 100) + '%', xOf(i), rows[2][1]);
      }

      var ent = 0;
      d.final.forEach(function (v) { if (v > 0) ent -= v * Math.log2(v); });
      readout.innerHTML = 'Scores are logits divided by T. Kept <strong>' + d.size + ' of ' + n +
        '</strong> tokens; top token now at <strong>' + (Math.max.apply(null, d.final) * 100).toFixed(1) +
        '%</strong>, entropy <strong>' + ent.toFixed(2) + ' bits</strong>.' +
        (state.draws ? ' Green = where 200 draws landed.' : '');
    }

    var read = U.bindSliders(host, function (key, value) {
      state[key] = value;
      state.draws = null;
      sync();
      draw();
    });

    function sync() {
      var v = read();
      U.setVal(host, 'T', v.T.toFixed(2));
      U.setVal(host, 'k', String(v.k));
      U.setVal(host, 'p', v.p.toFixed(2));
    }

    host.querySelector('[data-act="sample"]').addEventListener('click', function () {
      var d = distribution();
      var rand = U.rng(20240917);
      var counts = new Array(TOKENS.length).fill(0);
      for (var s = 0; s < 200; s++) {
        var r = rand(), acc = 0, pick = 0;
        for (var i = 0; i < d.final.length; i++) { acc += d.final[i]; if (r <= acc) { pick = i; break; } }
        counts[pick]++;
      }
      counts.total = 200;
      state.draws = counts;
      draw();
    });
    host.querySelector('[data-act="reset"]').addEventListener('click', function () {
      state = { T: 1.0, k: 12, p: 1.0, draws: null };
      host.querySelector('[data-key="T"]').value = 1.0;
      host.querySelector('[data-key="k"]').value = 12;
      host.querySelector('[data-key="p"]').value = 1.0;
      sync(); draw();
    });

    sync();
    draw();
    return { resize: draw };
  }

  // ===================================================================
  // WIDGET: cosineDial — drag two vectors lying in a tilted 2-D plane;
  // cos(theta) is drawn as a bar along the third (vertical) axis.
  // ===================================================================
  function cosineDial(host) {
    var U = WIDGET_UTIL, COL = WIDGET_UTIL.COL;
    host.innerHTML = '<canvas class="cos-canvas"></canvas>';
    U.stop(host);
    var canvas = host.querySelector('canvas');
    var vec = { a: [1.0, 0.15], b: [0.45, 0.9] };
    var geo = null, drag = null;
    // Camera: rotate the floor by AZ about the vertical axis, tilt it back by EL.
    var AZ = -0.62, EL = 0.42;
    var cA = Math.cos(AZ), sA = Math.sin(AZ), sE = Math.sin(EL), cE = Math.cos(EL);

    function proj(X, Y, Z) {
      var x1 = X * cA - Y * sA, y1 = X * sA + Y * cA;
      return [geo.cx + x1 * geo.R, geo.cy - (y1 * sE + (Z || 0) * cE * geo.zs) * geo.R];
    }
    function toFloor(px, py) {
      var x1 = (px - geo.cx) / geo.R, y1 = (geo.cy - py) / (geo.R * sE);
      return [x1 * cA + y1 * sA, -x1 * sA + y1 * cA];
    }
    function line(ctx, p, q) { ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); }
    function poly(ctx, pts, fill) {
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
      for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
    }
    function arrow(ctx, from, to, color) {
      var ang = Math.atan2(to[1] - from[1], to[0] - from[0]);
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3;
      line(ctx, from, to);
      ctx.beginPath(); ctx.moveTo(to[0], to[1]);
      ctx.lineTo(to[0] - 12 * Math.cos(ang - 0.4), to[1] - 12 * Math.sin(ang - 0.4));
      ctx.lineTo(to[0] - 12 * Math.cos(ang + 0.4), to[1] - 12 * Math.sin(ang + 0.4));
      ctx.closePath(); ctx.fill();
      ctx.beginPath(); ctx.arc(to[0], to[1], 6, 0, 2 * Math.PI); ctx.globalAlpha = 0.25; ctx.fill(); ctx.globalAlpha = 1;
    }

    function draw() {
      var f = U.fit(canvas); if (!f) return;
      var ctx = f.ctx, W = f.w, H = f.h;
      geo = { cx: W * 0.36, cy: H * 0.6, R: Math.min(W * 0.26, H * 0.42), zs: 0.95 };
      ctx.clearRect(0, 0, W, H);

      var a = vec.a, b = vec.b;
      var cos = (a[0] * b[0] + a[1] * b[1]) / (Math.hypot(a[0], a[1]) * Math.hypot(b[0], b[1]));
      var thA = Math.atan2(a[1], a[0]), thB = Math.atan2(b[1], b[0]);
      var d = thB - thA;
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;

      // Floor: a square grid seen in perspective, so the plane reads as 3-D.
      var E = 1.3;
      poly(ctx, [proj(-E, -E), proj(E, -E), proj(E, E), proj(-E, E)], 'rgba(74,158,255,0.05)');
      ctx.strokeStyle = 'rgba(136,146,164,0.22)'; ctx.lineWidth = 1;
      for (var g = -E; g <= E + 1e-9; g += 0.325) {
        line(ctx, proj(g, -E), proj(g, E));
        line(ctx, proj(-E, g), proj(E, g));
      }
      ctx.strokeStyle = 'rgba(136,146,164,0.7)'; ctx.lineWidth = 1.4;
      line(ctx, proj(-E, 0), proj(E, 0)); line(ctx, proj(0, -E), proj(0, E));
      ctx.fillStyle = COL.muted; ctx.font = '12px Inter, sans-serif'; ctx.textAlign = 'center';
      var lx = proj(E + 0.18, 0), ly = proj(0, E + 0.18);
      ctx.fillText('dim 1', lx[0], lx[1] + 4); ctx.fillText('dim 2', ly[0], ly[1] + 4);

      // Angle arc on the floor.
      ctx.strokeStyle = COL.secondary; ctx.lineWidth = 2;
      ctx.beginPath();
      for (var t = 0; t <= 30; t++) {
        var ang = thA + d * t / 30, q = proj(0.4 * Math.cos(ang), 0.4 * Math.sin(ang));
        if (t === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
      }
      ctx.stroke();

      // Vertical cos axis with ticks.
      var o = proj(0, 0);
      ctx.setLineDash([4, 4]); ctx.strokeStyle = 'rgba(232,234,240,0.5)'; ctx.lineWidth = 1.2;
      line(ctx, proj(0, 0, -1), proj(0, 0, 1)); ctx.setLineDash([]);
      [-1, -0.5, 0.5, 1].forEach(function (z) {
        var p = proj(0, 0, z);
        ctx.strokeStyle = 'rgba(232,234,240,0.5)'; line(ctx, [p[0] - 5, p[1]], [p[0] + 5, p[1]]);
        ctx.fillStyle = COL.muted; ctx.textAlign = 'right'; ctx.font = '11px Inter, sans-serif';
        ctx.fillText((z > 0 ? '+' : '') + z, p[0] - 9, p[1] + 4);
      });
      var zt = proj(0, 0, 1.18);
      ctx.fillStyle = COL.secondary; ctx.textAlign = 'center'; ctx.font = '600 13px Inter, sans-serif';
      ctx.fillText('cos θ', zt[0], zt[1]);

      // The bar: a small box from the floor up (or down) to z = cos, with shaded faces.
      var w = 0.07, z = cos;
      var c = [[-w, -w], [w, -w], [w, w], [-w, w]];
      var bot = c.map(function (p) { return proj(p[0], p[1], 0); });
      var top = c.map(function (p) { return proj(p[0], p[1], z); });
      var faces = [[0, 1], [1, 2], [2, 3], [3, 0]];
      faces.forEach(function (fc, k) {
        var shade = ['rgba(245,166,35,0.55)', 'rgba(245,166,35,0.75)', 'rgba(245,166,35,0.55)', 'rgba(245,166,35,0.75)'][k];
        poly(ctx, [bot[fc[0]], bot[fc[1]], top[fc[1]], top[fc[0]]], shade);
      });
      poly(ctx, top, 'rgba(255,200,110,0.95)');

      arrow(ctx, o, proj(a[0], a[1]), COL.primary);
      arrow(ctx, o, proj(b[0], b[1]), COL.green);
      ctx.font = '600 15px Inter, sans-serif'; ctx.textAlign = 'left';
      var la = proj(a[0] * 1.12, a[1] * 1.12), lb = proj(b[0] * 1.12, b[1] * 1.12);
      ctx.fillStyle = COL.primary; ctx.fillText('a', la[0] + 4, la[1] + 4);
      ctx.fillStyle = COL.green; ctx.fillText('b', lb[0] + 4, lb[1] + 4);

      var rx = W * 0.72, ry = H * 0.28;
      ctx.textAlign = 'left';
      ctx.fillStyle = COL.secondary; ctx.font = '600 22px Inter, sans-serif';
      ctx.fillText('cos θ = ' + cos.toFixed(2), rx, ry);
      ctx.fillStyle = COL.text; ctx.font = '14px Inter, sans-serif';
      ctx.fillText('θ = ' + Math.round(Math.abs(d) * 180 / Math.PI) + '°', rx, ry + 26);
      ctx.fillStyle = COL.muted; ctx.font = '12px Inter, sans-serif';
      ctx.fillText('bar height = cos θ', rx, ry + 54);
      ctx.fillText('+1 same direction, 0 unrelated,', rx, ry + 72);
      ctx.fillText('-1 opposite', rx, ry + 88);
      ctx.fillText('drag the arrow tips on the floor', rx, ry + 114);
    }

    function pointer(e) {
      var r = canvas.getBoundingClientRect();
      return toFloor((e.clientX - r.left) * canvas.clientWidth / r.width, (e.clientY - r.top) * canvas.clientHeight / r.height);
    }
    canvas.addEventListener('pointerdown', function (e) {
      if (!geo) return;
      var P = pointer(e);
      drag = Math.hypot(P[0] - vec.a[0], P[1] - vec.a[1]) < Math.hypot(P[0] - vec.b[0], P[1] - vec.b[1]) ? 'a' : 'b';
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', function (e) {
      if (!drag) return;
      var P = pointer(e), n = Math.hypot(P[0], P[1]);
      if (n < 0.25) return;
      var s = Math.min(n, 1.25) / n;
      vec[drag] = [P[0] * s, P[1] * s];
      draw();
    });
    canvas.addEventListener('pointerup', function () { drag = null; });

    draw();
    return { resize: draw };
  }

  window.MODULE_CONFIG = {
    title: 'LLMs 0 to 100 - Module 4',
    manimSections: {
      'bpe-training': [
        'BPETrainingScene_0000_start.mp4',
        'BPETrainingScene_0001_count_pairs.mp4',
        'BPETrainingScene_0002_merge_lo.mp4',
        'BPETrainingScene_0003_merge_low.mp4',
        'BPETrainingScene_0004_result.mp4'
      ],
      'recurrence-vs-attention': [
        'RecurrenceVsAttentionScene_0000_recurrence.mp4',
        'RecurrenceVsAttentionScene_0001_attention.mp4'
      ],
      'parallel-forward': [
        'ParallelForwardScene_0000_input.mp4',
        'ParallelForwardScene_0001_block1.mp4',
        'ParallelForwardScene_0002_mask.mp4',
        'ParallelForwardScene_0003_stack.mp4',
        'ParallelForwardScene_0004_predict.mp4',
        'ParallelForwardScene_0005_shift.mp4',
        'ParallelForwardScene_0006_last_row.mp4'
      ],
      'embedding-space': [
        'EmbeddingSpaceScene_0000_init.mp4',
        'EmbeddingSpaceScene_0001_row_is_point.mp4',
        'EmbeddingSpaceScene_0002_train.mp4',
        'EmbeddingSpaceScene_0003_clusters.mp4',
        'EmbeddingSpaceScene_0004_directions.mp4',
        'EmbeddingSpaceScene_0005_high_dim.mp4'
      ],
      'embedding-lookup': [
        'EmbeddingLookupScene_0000_word.mp4',
        'EmbeddingLookupScene_0001_lookup.mp4',
        'EmbeddingLookupScene_0002_vector.mp4'
      ],
      'ffn-expand': [
        'FFNExpandScene_0000_vector.mp4',
        'FFNExpandScene_0001_expand.mp4',
        'FFNExpandScene_0002_activate.mp4',
        'FFNExpandScene_0003_contract.mp4'
      ],
      'norm-demo': [
        'NormDemoScene_0000_vector.mp4',
        'NormDemoScene_0001_layernorm.mp4',
        'NormDemoScene_0002_rmsnorm.mp4'
      ],
      'residual-stream': [
        'ResidualStreamScene_0000_stream.mp4',
        'ResidualStreamScene_0001_block1.mp4',
        'ResidualStreamScene_0002_block2.mp4',
        'ResidualStreamScene_0003_readout.mp4'
      ]
    },
    widgets: {
      samplingExplorer: samplingExplorer,
      cosineDial: cosineDial
    }
  };
}());
