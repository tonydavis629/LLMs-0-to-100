// =====================================================================
// Single Perceptron widget — INTERACTIVE_WIDGETS.singlePerceptron
// Adjustable line on data, no training.
// =====================================================================
INTERACTIVE_WIDGETS.singlePerceptron = function(host) {
  var GREEN = '#3fb950', RED = '#e74c3c', MUTED = '#8892a4',
      LINEC = '#2a3450', TEXT = '#e8eaf0', PRIMARY = '#4a9eff', SECONDARY = '#f5a623';
  var state = { dataset: 'linear', w1: 0.5, w2: 0.5, b: -0.5, sample: { x: 0.78, y: 0.78 } };

  function mulberry32(a) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  // 2D datasets under the unit square.
  function generateData(name) {
    var rnd = mulberry32(7);
    var pts = [];
    if (name === 'linear') {
      for (var i = 0; i < 40; i++) {
        var x = rnd() * 2 - 1, y = rnd() * 2 - 1;
        pts.push({ x: x + 1.5, y: y + 1.5, cls: 1 });
        pts.push({ x: x - 1.5, y: y - 1.5, cls: 0 });
      }
    } else if (name === 'xor') {
      var clusters = [[1.5, 1.5, 1], [1.5, -1.5, 0], [-1.5, 1.5, 0], [-1.5, -1.5, 1]];
      clusters.forEach(function(c) {
        for (var i = 0; i < 14; i++) {
          pts.push({ x: c[0] + (rnd() - 0.5) * 0.6, y: c[1] + (rnd() - 0.5) * 0.6, cls: c[2] });
        }
      });
    }
    // Normalize to [0.04, 0.96]
    var xs = pts.map(function(p) { return p.x; });
    var ys = pts.map(function(p) { return p.y; });
    var xmn = Math.min.apply(null, xs), xmx = Math.max.apply(null, xs);
    var ymn = Math.min.apply(null, ys), ymx = Math.max.apply(null, ys);
    var xr = xmx - xmn || 1, yr = ymx - ymn || 1;
    pts.forEach(function(p) {
      p.x = 0.04 + 0.92 * (p.x - xmn) / xr;
      p.y = 0.04 + 0.92 * (p.y - ymn) / yr;
    });
    return pts;
  }

  host.innerHTML =
    '<div class="perc-widget">' +
      '<div class="perc-canvas-wrap"><canvas class="perc-canvas"></canvas></div>' +
      '<div class="perc-controls">' +
        '<div class="perc-datasets"></div>' +
        '<div class="perc-sliders"></div>' +
        '<p class="perc-readout"></p>' +
      '</div>' +
    '</div>';
  var canvas = host.querySelector('.perc-canvas');
  var readout = host.querySelector('.perc-readout');
  var dsEl = host.querySelector('.perc-datasets');
  var slidersEl = host.querySelector('.perc-sliders');

  ['pointerdown', 'keydown'].forEach(function(ev) {
    host.addEventListener(ev, function(e) { e.stopPropagation(); });
  });

  var datasets = ['linear', 'xor'];
  datasets.forEach(function(name) {
    var btn = document.createElement('button');
    btn.className = 'perc-ds-btn';
    btn.textContent = name === 'xor' ? 'XOR' : name.charAt(0).toUpperCase() + name.slice(1);
    btn.dataset.ds = name;
    btn.addEventListener('click', function() {
      state.dataset = name;
      state.sample = name === 'xor' ? { x: 0.82, y: 0.82 } : { x: 0.78, y: 0.78 };
      updateUI(); draw();
    });
    dsEl.appendChild(btn);
  });

  function buildSliders() {
    slidersEl.innerHTML = '';
    [['w1', 'w₁', -3, 3, 0.1], ['w2', 'w₂', -3, 3, 0.1], ['b', 'b', -3, 3, 0.1]].forEach(function(spec) {
      var key = spec[0], label = spec[1], mn = spec[2], mx = spec[3], st = spec[4];
      var row = document.createElement('div');
      row.className = 'perc-slider';
      var lab = document.createElement('label');
      lab.textContent = label;
      var inp = document.createElement('input');
      inp.type = 'range'; inp.min = mn; inp.max = mx; inp.step = st;
      inp.value = state[key].toFixed(1);
      var val = document.createElement('p');
      val.className = 'perc-val';
      val.textContent = state[key].toFixed(1);
      inp.addEventListener('input', function() {
        state[key] = parseFloat(inp.value);
        val.textContent = state[key].toFixed(1);
        draw();
      });
      row.appendChild(lab); row.appendChild(inp); row.appendChild(val);
      slidersEl.appendChild(row);
    });
  }

  function updateUI() {
    host.querySelectorAll('.perc-ds-btn').forEach(function(b) {
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

  function predict(x, y) {
    return probability(x, y) >= 0.5 ? 1 : 0;
  }

  function score(x, y) {
    return state.w1 * x + state.w2 * y + state.b;
  }

  function probability(x, y) {
    return 1 / (1 + Math.exp(-score(x, y)));
  }

  function signed(v) {
    return (v < 0 ? ' - ' : ' + ') + Math.abs(v).toFixed(1);
  }

  function drawNetwork(ctx, x0, y0, w, h) {
    function node(x, y, label, color) {
      ctx.beginPath();
      ctx.arc(x, y, 16, 0, 6.2832);
      ctx.fillStyle = '#0d1225';
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = TEXT;
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, x, y + 5);
    }
    function edge(x1, y1, x2, y2, label) {
      ctx.strokeStyle = 'rgba(74,158,255,0.55)';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      var mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      ctx.fillStyle = '#0d1225';
      ctx.fillRect(mx - 25, my - 9, 50, 16);
      ctx.fillStyle = PRIMARY;
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, mx, my + 4);
    }
    var xi = x0 + 34, xo = x0 + w - 42;
    var y1 = y0 + h * 0.32, y2 = y0 + h * 0.68, yo = y0 + h * 0.50;
    edge(xi + 16, y1, xo - 16, yo, 'w₁=' + state.w1.toFixed(1));
    edge(xi + 16, y2, xo - 16, yo, 'w₂=' + state.w2.toFixed(1));
    node(xi, y1, 'x₁', SECONDARY);
    node(xi, y2, 'x₂', SECONDARY);
    node(xo, yo, 'ŷ', PRIMARY);
    ctx.fillStyle = MUTED;
    ctx.font = '13px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('h = w·x + b', xo, yo + 38);
    ctx.fillText('y = sigmoid(h)', xo, yo + 56);
  }

  function draw() {
    var f = fitCanvas(canvas);
    if (!f) return;
    var ctx = f.ctx, W = f.w, H = f.h;
    ctx.clearRect(0, 0, W, H);

    var pad = 18;
    var rightPanel = 320;
    var side = Math.min(W - 2 * pad - rightPanel, H - 2 * pad);
    if (side < 120) { rightPanel = 0; side = Math.min(W - 2 * pad, H - 2 * pad); }
    var ox = pad + (W - 2 * pad - rightPanel - side) / 2;
    var oy = pad + (H - 2 * pad - side) / 2;

    // Background shading: which side of the line each point falls on
    var GRID = 36;
    for (var i = 0; i < GRID; i++) {
      for (var j = 0; j < GRID; j++) {
        var gx = i / (GRID - 1), gy = j / (GRID - 1);
        var pred = predict(gx, gy);
        ctx.fillStyle = pred ? 'rgba(63,185,80,0.14)' : 'rgba(231,76,60,0.08)';
        ctx.fillRect(ox + gx * side, oy + (1 - gy) * side, side / GRID + 1, side / GRID + 1);
      }
    }

    // Decision line w1*x + w2*y + b = 0, clipped to frame
    var a = state.w1, b = state.w2, c = state.b;
    var xr = [0, 1], yr = [0, 1];
    var p0, p1;
    if (Math.abs(b) > 1e-9) {
      p0 = [ox + xr[0] * side, oy + (1 - (-(a * xr[0] + c) / b)) * side];
      p1 = [ox + xr[1] * side, oy + (1 - (-(a * xr[1] + c) / b)) * side];
    } else if (Math.abs(a) > 1e-9) {
      var xv = -c / a;
      p0 = [ox + xv * side, oy + yr[0] * side];
      p1 = [ox + xv * side, oy + yr[1] * side];
    } else {
      p0 = null; p1 = null;
    }

    if (p0 && p1) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(ox, oy, side, side);
      ctx.clip();
      ctx.strokeStyle = SECONDARY; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.stroke();
      ctx.restore();
    }

    // Data points
    var data = generateData(state.dataset);
    data.forEach(function(p) {
      var px = ox + p.x * side, py = oy + (1 - p.y) * side;
      ctx.beginPath(); ctx.arc(px, py, 4.5, 0, 6.2832);
      ctx.fillStyle = p.cls ? GREEN : RED; ctx.fill();
      // Misclassification ring
      if (predict(p.x, p.y) !== p.cls) {
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.arc(px, py, 6.6, 0, 6.2832); ctx.stroke();
      }
    });

    var sample = state.sample;
    var h = score(sample.x, sample.y);
    var yhat = probability(sample.x, sample.y);
    var sx = ox + sample.x * side, sy = oy + (1 - sample.y) * side;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(sx, sy, 8, 0, 6.2832); ctx.stroke();
    ctx.fillStyle = yhat >= 0.5 ? GREEN : RED;
    ctx.beginPath(); ctx.arc(sx, sy, 4, 0, 6.2832); ctx.fill();

    // Frame
    ctx.strokeStyle = LINEC; ctx.lineWidth = 1;
    ctx.strokeRect(ox, oy, side, side);

    if (rightPanel > 80) {
      drawNetwork(ctx, ox + side + 24, oy + 8, rightPanel - 38, side - 16);
    }

    var correct = 0;
    data.forEach(function(p) { if (predict(p.x, p.y) === p.cls) correct++; });
    var acc = data.length ? correct / data.length : 0;
    var colorWord = yhat >= 0.5 ? 'green' : 'red';
    readout.innerHTML =
      'x=(' + sample.x.toFixed(2) + ', ' + sample.y.toFixed(2) + ') → h = ' +
      state.w1.toFixed(1) + '·' + sample.x.toFixed(2) +
      signed(state.w2) + '·' + sample.y.toFixed(2) +
      signed(state.b) + ' = ' + h.toFixed(2) +
      ', y=sigmoid(h)=' + yhat.toFixed(2) + (yhat >= 0.5 ? ' ≥ 0.5' : ' < 0.5') +
      ' ⇒ <strong>' + colorWord + '</strong> | Accuracy: <strong>' +
      (acc * 100).toFixed(0) + '%</strong>';
  }

  buildSliders();
  updateUI();
  return { resize: draw };
};
