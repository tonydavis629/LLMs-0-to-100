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

      var padL = 46, padR = 20, padB = 62, padT = 30;
      var plotW = W - padL - padR, plotH = H - padT - padB;
      var n = TOKENS.length;
      var step = plotW / n, bw = Math.min(52, step - 12);
      var maxP = Math.max.apply(null, d.final.concat(d.raw));
      var scale = plotH / (maxP * 1.12);

      ctx.textAlign = 'center';
      ctx.font = '12px Inter, sans-serif';
      ctx.fillStyle = COL.muted;
      ctx.fillText('probability after temperature (outline) and after truncation + renormalize (filled)' +
        (state.draws ? '; orange = how often 200 draws actually landed there' : ''), W / 2, 16);

      for (var i = 0; i < n; i++) {
        var cx = padL + step * i + step / 2;
        var yBase = padT + plotH;

        // Outline: what temperature alone produced, before any cutoff.
        var hRaw = d.raw[i] * scale;
        ctx.strokeStyle = 'rgba(136,146,164,0.55)';
        ctx.lineWidth = 1.2;
        ctx.strokeRect(cx - bw / 2, yBase - hRaw, bw, hRaw);

        var hFin = d.final[i] * scale;
        if (hFin > 0.5) {
          ctx.fillStyle = d.kept[i] ? 'rgba(74,158,255,0.75)' : 'rgba(136,146,164,0.15)';
          U.rounded(ctx, cx - bw / 2, yBase - hFin, bw, hFin, 3);
          ctx.fill();
        }

        // Sample counts land as an orange overlay so drift from the bars is visible.
        if (state.draws) {
          var hs = (state.draws[i] / state.draws.total) * scale;
          if (hs > 0.5) {
            ctx.fillStyle = 'rgba(245,166,35,0.85)';
            ctx.fillRect(cx - bw / 2 + bw * 0.28, yBase - hs, bw * 0.44, hs);
          }
        }

        ctx.save();
        ctx.translate(cx, yBase + 10);
        ctx.rotate(-Math.PI / 5);
        ctx.textAlign = 'right';
        ctx.fillStyle = d.kept[i] ? COL.text : COL.muted;
        ctx.font = (d.kept[i] ? '600 ' : '') + '12px Inter, sans-serif';
        ctx.fillText(TOKENS[i], 0, 0);
        ctx.restore();
      }

      ctx.strokeStyle = COL.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, padT + plotH); ctx.lineTo(W - padR, padT + plotH); ctx.stroke();

      var ent = 0;
      d.final.forEach(function (v) { if (v > 0) ent -= v * Math.log2(v); });
      readout.innerHTML = 'Kept <strong>' + d.size + ' of ' + n + '</strong> tokens. Top token now at <strong>' +
        (Math.max.apply(null, d.final) * 100).toFixed(1) + '%</strong>, entropy <strong>' + ent.toFixed(2) +
        ' bits</strong>. Truncation is what stops the long tail from ever being drawn.';
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
      'sampling-demo': [
        'SamplingScene_0000_dist.mp4',
        'SamplingScene_0001_temp_sharp.mp4',
        'SamplingScene_0002_temp_flat.mp4',
        'SamplingScene_0003_topk.mp4',
        'SamplingScene_0004_topp.mp4'
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
      samplingExplorer: samplingExplorer
    }
  };
}());
