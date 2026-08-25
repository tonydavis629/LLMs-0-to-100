(function () {
  // Module 9 config. This deck has no Manim clips: the only genuinely spatial
  // figure is the benchmark saturation chart, which is a to-scale HTML span chart
  // built from positioned divs and reveal.js fragments.
  // ===================================================================
  // WIDGET: passAtK — two models with different per-sample success rates,
  // compared as k grows. pass@k = 1 - (1 - p)^k for independent samples.
  // ===================================================================
  function passAtK(host) {
    var U = WIDGET_UTIL, COL = WIDGET_UTIL.COL;
    var KMAX = 100;

    host.innerHTML =
      '<div class="iw">' +
        '<div class="iw-canvas-wrap"><canvas class="iw-canvas"></canvas></div>' +
        '<div class="iw-stats">' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="a1">0</span><span class="iw-stat-lab">model A, pass@1</span></div>' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="b1">0</span><span class="iw-stat-lab">model B, pass@1</span></div>' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="ak">0</span><span class="iw-stat-lab">model A at this k</span></div>' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="bk">0</span><span class="iw-stat-lab">model B at this k</span></div>' +
        '</div>' +
        '<div class="iw-sliders">' +
          U.sliderHTML('pa', 'model A rate', 0.01, 0.9, 0.01, 0.30) +
          U.sliderHTML('pb', 'model B rate', 0.01, 0.9, 0.01, 0.10) +
          U.sliderHTML('k', 'k (samples)', 1, 100, 1, 10) +
        '</div>' +
        '<p class="iw-readout"></p>' +
      '</div>';
    U.stop(host);
    var canvas = host.querySelector('.iw-canvas');
    var readout = host.querySelector('.iw-readout');
    var el = {};
    host.querySelectorAll('[data-el]').forEach(function (n) { el[n.getAttribute('data-el')] = n; });

    function passk(p, k) { return 1 - Math.pow(1 - p, k); }

    function draw() {
      var v = read();
      var f = U.fit(canvas); if (!f) return;
      var ctx = f.ctx, W = f.w, H = f.h;
      ctx.clearRect(0, 0, W, H);

      var padL = 54, padR = 24, padT = 26, padB = 44;
      var pw = W - padL - padR, ph = H - padT - padB;
      function X(k) { return padL + ((k - 1) / (KMAX - 1)) * pw; }
      function Y(y) { return padT + ph - y * ph; }

      ctx.strokeStyle = COL.line; ctx.lineWidth = 1;
      [0, 0.25, 0.5, 0.75, 1].forEach(function (g) {
        ctx.beginPath(); ctx.moveTo(padL, Y(g)); ctx.lineTo(W - padR, Y(g)); ctx.stroke();
        ctx.fillStyle = COL.muted; ctx.textAlign = 'right'; ctx.font = '11px Inter, sans-serif';
        ctx.fillText((g * 100).toFixed(0) + '%', padL - 6, Y(g) + 4);
      });

      ctx.textAlign = 'center'; ctx.fillStyle = COL.muted; ctx.font = '12px Inter, sans-serif';
      ctx.fillText('k, the number of samples drawn per problem', padL + pw / 2, H - 10);
      ctx.font = '11px Inter, sans-serif';
      [1, 25, 50, 75, 100].forEach(function (t) {
        ctx.fillText(String(t), X(t), padT + ph + 16);
      });

      [[v.pa, COL.primary, 'model A'], [v.pb, COL.secondary, 'model B']].forEach(function (m) {
        ctx.strokeStyle = m[1]; ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (var k = 1; k <= KMAX; k++) {
          var x = X(k), y = Y(passk(m[0], k));
          if (k === 1) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      });

      // Legend, so the two saturating curves stay distinguishable.
      ctx.textAlign = 'left'; ctx.font = '600 12px Inter, sans-serif';
      [[COL.primary, 'model A'], [COL.secondary, 'model B']].forEach(function (m, i) {
        var ly = padT + 14 + i * 20;
        ctx.strokeStyle = m[0]; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(W - padR - 90, ly - 4); ctx.lineTo(W - padR - 68, ly - 4); ctx.stroke();
        ctx.fillStyle = m[0];
        ctx.fillText(m[1], W - padR - 62, ly);
      });

      var kx = X(v.k);
      ctx.strokeStyle = 'rgba(232,234,240,0.5)'; ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(kx, padT); ctx.lineTo(kx, padT + ph); ctx.stroke();
      ctx.setLineDash([]);
      [[v.pa, COL.primary], [v.pb, COL.secondary]].forEach(function (m) {
        ctx.fillStyle = m[1];
        ctx.beginPath(); ctx.arc(kx, Y(passk(m[0], v.k)), 5, 0, Math.PI * 2); ctx.fill();
      });

      el.a1.textContent = (v.pa * 100).toFixed(0) + '%';
      el.b1.textContent = (v.pb * 100).toFixed(0) + '%';
      el.ak.textContent = (passk(v.pa, v.k) * 100).toFixed(1) + '%';
      el.bk.textContent = (passk(v.pb, v.k) * 100).toFixed(1) + '%';
      U.setVal(host, 'pa', v.pa.toFixed(2));
      U.setVal(host, 'pb', v.pb.toFixed(2));
      U.setVal(host, 'k', String(v.k));

      var gap1 = Math.abs(v.pa - v.pb) * 100;
      var gapk = Math.abs(passk(v.pa, v.k) - passk(v.pb, v.k)) * 100;
      readout.innerHTML = 'A gap of <strong>' + gap1.toFixed(0) + ' points at k = 1</strong> shrinks to <strong>' +
        gapk.toFixed(1) + ' points at k = ' + v.k +
        '</strong>. Both curves saturate, so a pass@k headline can hide the number a user actually experiences, which is pass@1.';
    }

    var read = U.bindSliders(host, draw);
    draw();
    return { resize: draw };
  }

  window.MODULE_CONFIG = {
    title: 'LLMs 0 to 100 - Module 9',
    manimSections: {},
    widgets: {
      passAtK: passAtK
    }
  };
}());
