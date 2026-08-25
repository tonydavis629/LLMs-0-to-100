// =====================================================================
// Boundary Explorer widget — INTERACTIVE_WIDGETS.boundaryExplorer
// Explicit two-neuron hidden layer for XOR.
// =====================================================================
INTERACTIVE_WIDGETS.boundaryExplorer = function(host) {
  var GREEN = '#3fb950', RED = '#e74c3c', MUTED = '#8892a4',
      LINEC = '#2a3450', TEXT = '#e8eaf0', PRIMARY = '#4a9eff', SECONDARY = '#f5a623';
  var NEURON_COLORS = [PRIMARY, SECONDARY];
  var GRID = 48;
  var generateData = MLP.generateData;
  var state = { dataset: 'xor', neurons: [], outW: [], outB: 0, sample: { x: 0.82, y: 0.82 } };

  host.innerHTML =
    '<div class="perc-widget expl-depth-widget">' +
      '<div class="perc-canvas-wrap"><canvas class="perc-canvas"></canvas></div>' +
      '<div class="perc-controls">' +
        '<div class="expl-datasets"></div>' +
        '<div class="expl-sliders"></div>' +
        '<p class="perc-readout"></p>' +
      '</div>' +
    '</div>';
  var canvas = host.querySelector('.mlp-canvas');
  canvas = host.querySelector('.perc-canvas');
  var readout = host.querySelector('.perc-readout');
  var dsEl = host.querySelector('.expl-datasets');
  var slidersEl = host.querySelector('.expl-sliders');

  ['pointerdown', 'keydown'].forEach(function(ev) {
    host.addEventListener(ev, function(e) { e.stopPropagation(); });
  });

  function preset(name) {
    if (name === 'linear') {
      state.neurons = [
        { w1: 1.0, w2: 1.0, b: -1.0 },
        { w1: 1.0, w2: 1.0, b: -1.6 }
      ];
      state.outW = [1.0, 0.0];
      state.outB = -0.5;
      state.sample = { x: 0.78, y: 0.78 };
    } else {
      state.neurons = [
        { w1: 1.0, w2: 1.0, b: -0.7 },
        { w1: 1.0, w2: 1.0, b: -1.3 }
      ];
      state.outW = [-1.0, 1.0];
      state.outB = 0.5;
      state.sample = { x: 0.82, y: 0.82 };
    }
  }

  function applyDatasetPreset(name) {
    state.dataset = name;
    preset(name);
    buildSliders();
  }

  var datasets = ['linear', 'xor'];
  datasets.forEach(function(name) {
    var btn = document.createElement('button');
    btn.className = 'expl-ds-btn';
    btn.textContent = name === 'xor' ? 'XOR' : name.charAt(0).toUpperCase() + name.slice(1);
    btn.dataset.ds = name;
    btn.addEventListener('click', function() {
      applyDatasetPreset(name);
      updateUI(); draw();
    });
    dsEl.appendChild(btn);
  });
  var resetBtn = document.createElement('button');
  resetBtn.className = 'expl-arch-btn expl-reset';
  resetBtn.textContent = 'Reset weights';
  resetBtn.addEventListener('click', function() {
    preset(state.dataset);
    buildSliders();
    updateUI();
    draw();
  });
  dsEl.appendChild(resetBtn);

  function buildSliders() {
    slidersEl.innerHTML = '';
    state.neurons.forEach(function(n, idx) {
      [['w1', 'w₁'], ['w2', 'w₂'], ['b', 'b']].forEach(function(spec) {
        var key = spec[0];
        var row = document.createElement('div');
        row.className = 'perc-slider';
        var lab = document.createElement('label');
        lab.textContent = 'N' + (idx + 1) + ' ' + spec[1];
        lab.style.color = NEURON_COLORS[idx];
        var inp = document.createElement('input');
        inp.type = 'range';
        inp.min = '-3'; inp.max = '3'; inp.step = '0.1';
        inp.value = n[key].toFixed(1);
        var val = document.createElement('p');
        val.className = 'perc-val';
        val.textContent = n[key].toFixed(1);
        inp.addEventListener('input', function() {
          n[key] = parseFloat(inp.value);
          val.textContent = n[key].toFixed(1);
          draw();
        });
        row.appendChild(lab);
        row.appendChild(inp);
        row.appendChild(val);
        slidersEl.appendChild(row);
      });
    });
  }

  function updateUI() {
    host.querySelectorAll('.expl-ds-btn').forEach(function(b) {
      b.classList.toggle('active', b.dataset.ds === state.dataset);
    });
  }

  function fitCanvas(c) {
    var dpr = window.devicePixelRatio || 1;
    var w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return null;
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    var ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h };
  }

  function hiddenVals(x, y) {
    var zs = state.neurons.map(function(n) { return n.w1 * x + n.w2 * y + n.b; });
    var hs = zs.map(function(z) { return z >= 0 ? 1 : 0; });
    var score = state.outW[0] * hs[0] + state.outW[1] * hs[1] + state.outB;
    var yhat = 1 / (1 + Math.exp(-score));
    return { zs: zs, hs: hs, score: score, y: yhat, pred: yhat >= 0.5 ? 1 : 0 };
  }

  function predict(x, y) {
    return hiddenVals(x, y).pred;
  }

  function signed(v) {
    return (v < 0 ? ' - ' : ' + ') + Math.abs(v).toFixed(1);
  }

  function lineSegment(n) {
    var pts = [];
    function add(x, y) {
      if (x < -1e-6 || x > 1 + 1e-6 || y < -1e-6 || y > 1 + 1e-6) return;
      for (var i = 0; i < pts.length; i++) {
        if (Math.abs(pts[i][0] - x) < 1e-6 && Math.abs(pts[i][1] - y) < 1e-6) return;
      }
      pts.push([x, y]);
    }
    if (Math.abs(n.w2) > 1e-9) {
      add(0, -(n.w1 * 0 + n.b) / n.w2);
      add(1, -(n.w1 * 1 + n.b) / n.w2);
    }
    if (Math.abs(n.w1) > 1e-9) {
      add(-(n.w2 * 0 + n.b) / n.w1, 0);
      add(-(n.w2 * 1 + n.b) / n.w1, 1);
    }
    return pts.length >= 2 ? [pts[0], pts[1]] : null;
  }

  function drawNetwork(ctx, x0, y0, w, h) {
    function node(x, y, label, color) {
      ctx.beginPath();
      ctx.arc(x, y, 13, 0, 6.2832);
      ctx.fillStyle = '#0d1225';
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = TEXT;
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, x, y + 4);
    }
    function edge(x1, y1, x2, y2, color, label) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      if (!label) return;
      var mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      ctx.fillStyle = '#0d1225';
      ctx.fillRect(mx - 18, my - 8, 36, 14);
      ctx.fillStyle = color;
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, mx, my + 3);
    }
    var xi = x0 + 28, xh = x0 + w * 0.50, xo = x0 + w - 28;
    var y1 = y0 + h * 0.30, y2 = y0 + h * 0.70, yo = y0 + h * 0.50;
    edge(xi + 13, y1, xh - 13, y1, NEURON_COLORS[0], 'w₁');
    edge(xi + 13, y2, xh - 13, y1, NEURON_COLORS[0], 'w₂');
    edge(xi + 13, y1, xh - 13, y2, NEURON_COLORS[1], 'w₁');
    edge(xi + 13, y2, xh - 13, y2, NEURON_COLORS[1], 'w₂');
    edge(xh + 13, y1, xo - 13, yo, PRIMARY, 'v₁=' + state.outW[0].toFixed(1));
    edge(xh + 13, y2, xo - 13, yo, SECONDARY, 'v₂=' + state.outW[1].toFixed(1));
    node(xi, y1, 'x₁', SECONDARY);
    node(xi, y2, 'x₂', SECONDARY);
    node(xh, y1, 'h₁', NEURON_COLORS[0]);
    node(xh, y2, 'h₂', NEURON_COLORS[1]);
    node(xo, yo, 'ŷ', SECONDARY);
    ctx.fillStyle = MUTED;
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('b=' + state.outB.toFixed(1), xo, yo + 36);
  }

  function draw() {
    var f = fitCanvas(canvas);
    if (!f) return;
    var ctx = f.ctx, W = f.w, H = f.h;
    ctx.clearRect(0, 0, W, H);

    var data = generateData(state.dataset);

    var pad = 18;
    var rightPanel = 320;
    var side = Math.min(W - 2 * pad - rightPanel, H - 2 * pad);
    if (side < 120) { rightPanel = 0; side = Math.min(W - 2 * pad, H - 2 * pad); }
    var ox = pad + (W - 2 * pad - rightPanel - side) / 2;
    var oy = pad + (H - 2 * pad - side) / 2;

    // Decision grid background
    for (var i = 0; i < GRID; i++) {
      for (var j = 0; j < GRID; j++) {
        var gx = i / (GRID - 1), gy = j / (GRID - 1);
        var p = predict(gx, gy);
        ctx.fillStyle = p ? 'rgba(63,185,80,0.16)' : 'rgba(231,76,60,0.10)';
        ctx.fillRect(ox + gx * side, oy + (1 - gy) * side, side / GRID + 1, side / GRID + 1);
      }
    }

    state.neurons.forEach(function(n, idx) {
      var seg = lineSegment(n);
      if (!seg) return;
      ctx.strokeStyle = NEURON_COLORS[idx];
      ctx.lineWidth = 2.8;
      ctx.beginPath();
      ctx.moveTo(ox + seg[0][0] * side, oy + (1 - seg[0][1]) * side);
      ctx.lineTo(ox + seg[1][0] * side, oy + (1 - seg[1][1]) * side);
      ctx.stroke();
    });

    // Data scatter
    var correct = 0;
    data.forEach(function(p) {
      var px = ox + p.x * side, py = oy + (1 - p.y) * side;
      ctx.beginPath(); ctx.arc(px, py, 4.5, 0, 6.2832);
      ctx.fillStyle = p.cls ? GREEN : RED; ctx.fill();
      if (predict(p.x, p.y) === p.cls) correct++;
    });

    var sample = state.sample;
    var sv = hiddenVals(sample.x, sample.y);
    var spx = ox + sample.x * side, spy = oy + (1 - sample.y) * side;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(spx, spy, 8, 0, 6.2832); ctx.stroke();
    ctx.fillStyle = sv.pred ? GREEN : RED;
    ctx.beginPath(); ctx.arc(spx, spy, 4, 0, 6.2832); ctx.fill();

    // Frame
    ctx.strokeStyle = LINEC; ctx.lineWidth = 1;
    ctx.strokeRect(ox, oy, side, side);

    if (rightPanel > 80) {
      drawNetwork(ctx, ox + side + 22, oy + 8, rightPanel - 34, side - 16);
    }

    var acc = correct / data.length;
    var colorWord = sv.pred ? 'green' : 'red';
    readout.innerHTML =
      'x=(' + sample.x.toFixed(2) + ', ' + sample.y.toFixed(2) + ') → ' +
      'h=(' + sv.hs[0] + ', ' + sv.hs[1] + ') because line values are (' +
      sv.zs[0].toFixed(2) + ', ' + sv.zs[1].toFixed(2) + '), ' +
      'y=sigmoid(' + state.outW[0].toFixed(1) + '·' + sv.hs[0] +
      signed(state.outW[1]) + '·' + sv.hs[1] + signed(state.outB) +
      ')=' + sv.y.toFixed(2) + (sv.y >= 0.5 ? ' ≥ 0.5' : ' < 0.5') +
      ' ⇒ <strong>' + colorWord + '</strong>' +
      ' | Accuracy: <strong>' + (acc * 100).toFixed(0) + '%</strong>';
  }

  preset(state.dataset);
  buildSliders();
  updateUI();
  return { resize: draw };
};
