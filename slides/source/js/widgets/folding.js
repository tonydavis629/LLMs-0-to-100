// =====================================================================
// Folding widget — INTERACTIVE_WIDGETS.folding
// Input space ↔ hidden-space line values, live.
// The left pane shows each hidden neuron's line w·x+b=0. The right pane
// shows continuous signed line values: crossing a neuron line in input
// space crosses the matching axis in hidden space.
// =====================================================================
INTERACTIVE_WIDGETS.folding = function(host) {
  var GREEN = '#3fb950', RED = '#e74c3c', MUTED = '#8892a4',
      LINEC = '#2a3450', TEXT = '#e8eaf0', PRIMARY = '#4a9eff';
  var NEURON_COLORS = ['#4a9eff', '#f5a623'];

  // --- XOR-pattern data: opposite corners share a class ---
  function mulberry32(a) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  var rnd = mulberry32(42);
  var clusters = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
  var data = [];
  clusters.forEach(function(c) {
    for (var i = 0; i < 14; i++) {
      data.push({
        x: c[0] + (rnd() - 0.5) * 0.22,
        y: c[1] + (rnd() - 0.5) * 0.22,
        cls: c[2]
      });
    }
  });

  // Preset line positions for XOR: the two units detect points far above
  // or below the diagonal strip.
  function preset(k) {
    var p = k === 1 ? [
      { w1: 1, w2: -1, b: -0.3 }
    ] : [
      { w1: 1, w2: -1, b: -0.3 },
      { w1: -1, w2: 1, b: -0.3 }
    ];
    return p.map(function(o) { return { w1: o.w1, w2: o.w2, b: o.b }; });
  }
  var state = { k: 2, neurons: preset(2) };

  // --- DOM ---
  host.innerHTML =
    '<div class="fold-widget">' +
      '<div class="fold-views">' +
        '<figure class="fold-view"><figcaption>Input space</figcaption>' +
          '<canvas class="fold-canvas" data-pane="input"></canvas></figure>' +
        '<figure class="fold-view"><figcaption>Hidden space</figcaption>' +
          '<canvas class="fold-canvas" data-pane="hidden"></canvas></figure>' +
        '<figure class="fold-view fold-graph-view"><figcaption>Weights as edges</figcaption>' +
          '<canvas class="fold-canvas" data-pane="graph"></canvas></figure>' +
      '</div>' +
      '<div class="fold-controls">' +
        '<div class="fold-modes">' +
          '<button data-neurons="1">1 neuron</button>' +
          '<button data-neurons="2">2 neurons</button>' +
          '<button class="fold-reset">Reset to XOR solution</button>' +
        '</div>' +
        '<div class="fold-sliders"></div>' +
        '<p class="fold-readout"></p>' +
      '</div>' +
    '</div>';

  var inCanvas = host.querySelector('[data-pane="input"]');
  var graphCanvas = host.querySelector('[data-pane="graph"]');
  var hidCanvas = host.querySelector('[data-pane="hidden"]');
  var slidersEl = host.querySelector('.fold-sliders');
  var readoutEl = host.querySelector('.fold-readout');
  var modeBtns = host.querySelectorAll('.fold-modes button[data-neurons]');

  // Keep slider interaction from bubbling into reveal navigation.
  ['pointerdown', 'keydown'].forEach(function(ev) {
    host.addEventListener(ev, function(e) { e.stopPropagation(); });
  });

  function buildSliders() {
    slidersEl.innerHTML = '';
    state.neurons.forEach(function(n, idx) {
      var color = NEURON_COLORS[idx];
      [['w1', 'w₁'], ['w2', 'w₂'], ['b', 'b']].forEach(function(f) {
        var key = f[0];
        var row = document.createElement('div');
        row.className = 'fold-slider';
        var lab = document.createElement('label');
        lab.textContent = 'N' + (idx + 1) + ' ' + f[1];
        lab.style.color = color;
        var inp = document.createElement('input');
        inp.type = 'range';
        inp.min = '-3'; inp.max = '3'; inp.step = '0.1';
        inp.value = String(n[key]);
        var val = document.createElement('p');
        val.className = 'fold-val';
        val.textContent = (+n[key]).toFixed(1);
        inp.addEventListener('input', function() {
          n[key] = parseFloat(inp.value);
          val.textContent = n[key].toFixed(1);
          draw();
        });
        row.appendChild(lab); row.appendChild(inp); row.appendChild(val);
        slidersEl.appendChild(row);
      });
    });
  }

  function setK(k) {
    state.k = k;
    state.neurons = preset(k);
    modeBtns.forEach(function(b) {
      b.classList.toggle('active', +b.getAttribute('data-neurons') === k);
    });
    buildSliders();
    draw();
  }
  modeBtns.forEach(function(b) {
    b.addEventListener('click', function() { setK(+b.getAttribute('data-neurons')); });
  });
  host.querySelector('.fold-reset').addEventListener('click', function() {
    state.neurons = preset(state.k);
    buildSliders();
    draw();
  });

  // --- canvas helpers ---
  function fitCanvas(c) {
    var dpr = window.devicePixelRatio || 1;
    var w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return null;
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    var ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h };
  }

  function makeMapper(w, h, xr, yr, pad) {
    return function(x, y) {
      return [
        pad + (x - xr[0]) / (xr[1] - xr[0]) * (w - 2 * pad),
        h - pad - (y - yr[0]) / (yr[1] - yr[0]) * (h - 2 * pad)
      ];
    };
  }

  function drawFrame(ctx, w, h, pad, xr, yr, xlab, ylab) {
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = LINEC; ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, w - 2 * pad, h - 2 * pad);
    var m = makeMapper(w, h, xr, yr, pad);
    // zero axes if visible
    ctx.strokeStyle = 'rgba(136,146,164,0.35)';
    if (xr[0] < 0 && xr[1] > 0) {
      var zx = m(0, yr[0])[0];
      ctx.beginPath(); ctx.moveTo(zx, pad); ctx.lineTo(zx, h - pad); ctx.stroke();
    }
    if (yr[0] < 0 && yr[1] > 0) {
      var zy = m(xr[0], 0)[1];
      ctx.beginPath(); ctx.moveTo(pad, zy); ctx.lineTo(w - pad, zy); ctx.stroke();
    }
    ctx.fillStyle = MUTED; ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'right'; ctx.fillText(xlab, w - pad - 4, h - pad - 6);
    ctx.textAlign = 'left'; ctx.save();
    ctx.translate(pad + 4, pad + 4); ctx.fillText(ylab, 0, 8); ctx.restore();
    return m;
  }

  function lineSegment(n, xr, yr) {
    var a = n.w1, b = n.w2, c = n.b;
    var pts = [];
    function add(x, y) {
      if (x < xr[0] - 1e-6 || x > xr[1] + 1e-6 || y < yr[0] - 1e-6 || y > yr[1] + 1e-6) return;
      for (var i = 0; i < pts.length; i++) {
        if (Math.abs(pts[i][0] - x) < 1e-6 && Math.abs(pts[i][1] - y) < 1e-6) return;
      }
      pts.push([x, y]);
    }
    if (Math.abs(b) > 1e-9) {
      add(xr[0], -(a * xr[0] + c) / b);
      add(xr[1], -(a * xr[1] + c) / b);
    }
    if (Math.abs(a) > 1e-9) {
      add(-(b * yr[0] + c) / a, yr[0]);
      add(-(b * yr[1] + c) / a, yr[1]);
    }
    return pts.length >= 2 ? [pts[0], pts[1]] : null;
  }

  function edge(ctx, x1, y1, x2, y2, color, label) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    var mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    ctx.fillStyle = '#0d1225';
    ctx.fillRect(mx - 22, my - 10, 44, 18);
    ctx.fillStyle = color;
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, mx, my + 4);
  }

  function node(ctx, x, y, label, stroke) {
    ctx.beginPath();
    ctx.arc(x, y, 18, 0, 6.2832);
    ctx.fillStyle = '#0d1225';
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2.4;
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = TEXT;
    ctx.font = '14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, x, y + 5);
  }

  function drawGraph() {
    var fg = fitCanvas(graphCanvas);
    if (!fg) return;
    var ctx = fg.ctx, w = fg.w, h = fg.h;
    ctx.clearRect(0, 0, w, h);
    var leftX = w * 0.24, rightX = w * 0.74;
    var inY = [h * 0.34, h * 0.66];
    var outY = state.k === 1 ? [h * 0.5] : [h * 0.34, h * 0.66];
    state.neurons.forEach(function(n, idx) {
      edge(ctx, leftX + 18, inY[0], rightX - 18, outY[idx], NEURON_COLORS[idx], 'w₁=' + n.w1.toFixed(1));
      edge(ctx, leftX + 18, inY[1], rightX - 18, outY[idx], NEURON_COLORS[idx], 'w₂=' + n.w2.toFixed(1));
      ctx.fillStyle = NEURON_COLORS[idx];
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('b=' + n.b.toFixed(1), rightX + 24, outY[idx] + 4);
    });
    node(ctx, leftX, inY[0], 'x₁', PRIMARY);
    node(ctx, leftX, inY[1], 'x₂', PRIMARY);
    state.neurons.forEach(function(n, idx) { node(ctx, rightX, outY[idx], 'h' + (idx + 1), NEURON_COLORS[idx]); });
  }

  function dot(ctx, px, py, color, r) {
    ctx.beginPath(); ctx.arc(px, py, r || 4.5, 0, 6.2832);
    ctx.fillStyle = color; ctx.globalAlpha = 0.9; ctx.fill(); ctx.globalAlpha = 1;
  }

  function signedDistance(n, p) {
    var len = Math.sqrt(n.w1 * n.w1 + n.w2 * n.w2) || 1;
    return (n.w1 * p.x + n.w2 * p.y + n.b) / len;
  }

  function tangentCoordinate(n, p) {
    var len = Math.sqrt(n.w1 * n.w1 + n.w2 * n.w2) || 1;
    return (-n.w2 * p.x + n.w1 * p.y) / len;
  }

  function foldCoordinatePoint(p) {
    var n1 = state.neurons[0];
    return {
      x: signedDistance(n1, p),
      y: state.k > 1 ? signedDistance(state.neurons[1], p) : tangentCoordinate(n1, p),
      cls: p.cls
    };
  }

  function spanWithZero(vals) {
    var lo = Math.min.apply(null, vals.concat([0]));
    var hi = Math.max.apply(null, vals.concat([0]));
    if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
    var pad = Math.max(0.08, (hi - lo) * 0.16);
    return [lo - pad, hi + pad];
  }

  function draw() {
    // ---- Input space (left): data + each neuron's decision line ----
    var fi = fitCanvas(inCanvas);
    if (fi) {
      var pad = 26, xr = [-0.5, 1.5], yr = [-0.5, 1.5];
      var m = drawFrame(fi.ctx, fi.w, fi.h, pad, xr, yr, 'x₁', 'x₂');
      var ctx = fi.ctx;
      ctx.save();
      ctx.beginPath();
      ctx.rect(pad, pad, fi.w - 2 * pad, fi.h - 2 * pad);
      ctx.clip();
      // Shade each neuron's active half-plane, where its line value is positive.
      state.neurons.forEach(function(n, idx) {
        var a = n.w1, b = n.w2, c = n.b;
        ctx.fillStyle = idx === 0 ? 'rgba(74,158,255,0.06)' : 'rgba(245,166,35,0.06)';
        // Simple approximation: fill the canvas then clip to the active side
        ctx.save();
        ctx.beginPath();
        if (Math.abs(b) > 1e-9) {
          var y0 = -(a * xr[0] + c) / b;
          var y1 = -(a * xr[1] + c) / b;
          if (b > 0) {
            ctx.moveTo(m(xr[0], yr[0])[0], m(xr[0], yr[0])[1]);
            ctx.lineTo(m(xr[1], yr[0])[0], m(xr[1], yr[0])[1]);
            ctx.lineTo(m(xr[1], y1)[0], m(xr[1], y1)[1]);
            ctx.lineTo(m(xr[0], y0)[0], m(xr[0], y0)[1]);
          } else {
            ctx.moveTo(m(xr[0], yr[1])[0], m(xr[0], yr[1])[1]);
            ctx.lineTo(m(xr[1], yr[1])[0], m(xr[1], yr[1])[1]);
            ctx.lineTo(m(xr[1], y1)[0], m(xr[1], y1)[1]);
            ctx.lineTo(m(xr[0], y0)[0], m(xr[0], y0)[1]);
          }
        } else if (Math.abs(a) > 1e-9) {
          var xv = -c / a;
          if (a > 0) {
            ctx.moveTo(m(xv, yr[0])[0], m(xv, yr[0])[1]);
            ctx.lineTo(m(xr[1], yr[0])[0], m(xr[1], yr[0])[1]);
            ctx.lineTo(m(xr[1], yr[1])[0], m(xr[1], yr[1])[1]);
            ctx.lineTo(m(xv, yr[1])[0], m(xv, yr[1])[1]);
          } else {
            ctx.moveTo(m(xv, yr[0])[0], m(xv, yr[0])[1]);
            ctx.lineTo(m(xr[0], yr[0])[0], m(xr[0], yr[0])[1]);
            ctx.lineTo(m(xr[0], yr[1])[0], m(xr[0], yr[1])[1]);
            ctx.lineTo(m(xv, yr[1])[0], m(xv, yr[1])[1]);
          }
        }
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      });
      state.neurons.forEach(function(n, idx) {
        var seg = lineSegment(n, xr, yr);
        if (!seg) return;
        var p0 = m(seg[0][0], seg[0][1]);
        var p1 = m(seg[1][0], seg[1][1]);
        ctx.strokeStyle = NEURON_COLORS[idx];
        ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.stroke();
        var dx = p1[0] - p0[0], dy = p1[1] - p0[1];
        var len = Math.sqrt(dx * dx + dy * dy) || 1;
        var sign = idx === 0 ? -1 : 1;
        var lx = (p0[0] + p1[0]) / 2 - dy / len * 18 * sign;
        var ly = (p0[1] + p1[1]) / 2 + dx / len * 18 * sign;
        ctx.fillStyle = '#0d1225';
        ctx.fillRect(lx - 14, ly - 10, 28, 18);
        ctx.fillStyle = NEURON_COLORS[idx];
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('N' + (idx + 1), lx, ly + 4);
      });
      data.forEach(function(p) {
        var q = m(p.x, p.y);
        dot(ctx, q[0], q[1], p.cls ? GREEN : RED);
      });
      ctx.restore();
    }

    drawGraph();

    // ---- Hidden space (middle): smooth line-value coordinates ----
    var fh = fitCanvas(hidCanvas);
    if (!fh) { return; }
    var pts = data.map(foldCoordinatePoint);
    var hxr = spanWithZero(pts.map(function(p) { return p.x; }));
    var hyr = spanWithZero(pts.map(function(p) { return p.y; }));
    var pad2 = 26;
    var mh = drawFrame(fh.ctx, fh.w, fh.h, pad2, hxr, hyr,
                       'N1 value', state.k > 1 ? 'N2 value' : 'along N1');
    var ctx2 = fh.ctx;

    ctx2.save();
    ctx2.beginPath();
    ctx2.rect(pad2, pad2, fh.w - 2 * pad2, fh.h - 2 * pad2);
    ctx2.clip();

    var n1x = mh(0, hyr[0])[0];
    ctx2.strokeStyle = NEURON_COLORS[0];
    ctx2.lineWidth = 2.4;
    ctx2.beginPath(); ctx2.moveTo(n1x, pad2); ctx2.lineTo(n1x, fh.h - pad2); ctx2.stroke();
    ctx2.fillStyle = NEURON_COLORS[0];
    ctx2.font = '12px Inter, sans-serif';
    ctx2.textAlign = 'left';
    ctx2.fillText('N1 line', n1x + 5, pad2 + 14);

    if (state.k > 1) {
      var n2y = mh(hxr[0], 0)[1];
      ctx2.strokeStyle = NEURON_COLORS[1];
      ctx2.lineWidth = 2.4;
      ctx2.beginPath(); ctx2.moveTo(pad2, n2y); ctx2.lineTo(fh.w - pad2, n2y); ctx2.stroke();
      ctx2.fillStyle = NEURON_COLORS[1];
      ctx2.textAlign = 'right';
      ctx2.fillText('N2 line', fh.w - pad2 - 5, n2y - 6);
    }

    pts.forEach(function(p) {
      var q = mh(p.x, p.y);
      dot(ctx2, q[0], q[1], p.cls ? GREEN : RED);
    });
    ctx2.restore();

    readoutEl.innerHTML =
      state.k === 1
        ? 'Hidden space: crossing the N1 line in input space crosses the vertical N1 axis here.'
        : 'Hidden space: x is the N1 line value, y is the N2 line value; crossing a line on the left crosses its matching axis here.';
  }

  buildSliders();
  modeBtns.forEach(function(b) {
    b.classList.toggle('active', +b.getAttribute('data-neurons') === state.k);
  });

  return { resize: draw };
};
