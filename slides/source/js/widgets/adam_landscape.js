// =====================================================================
// Adam Landscape widget — INTERACTIVE_WIDGETS.adamLandscape
// Loss surface + Adam terms (m, v, m̂, v̂). Shows the actual Adam update
// formula step-by-step on a noisy quadratic bowl.
// =====================================================================
INTERACTIVE_WIDGETS.adamLandscape = function(host) {
  var TEXT = '#e8eaf0', MUTED = '#8892a4', PRIMARY = '#4a9eff',
      SECONDARY = '#f5a623', RED = '#e74c3c', LINEC = '#2a3450', GREEN = '#3fb950';
  var state = { w1: -1.0, w2: 2.0, eta: 0.3, batch: 8, beta1: 0.9, beta2: 0.999,
                m1: 0, m2: 0, v1: 0, v2: 0, t: 0, lastStep: null,
                az: -0.8, el: 0.65, zoom: 1.0, panX: 0, panY: 0,
                dragging: false, dragType: null, lastMX: 0, lastMY: 0 };
  var WMIN = -3, WMAX = 5;

  function loss(w1, w2) {
    return 0.5 * ((w1 - 1.5) * (w1 - 1.5) + (w2 + 1.0) * (w2 + 1.0));
  }
  function grad(w1, w2) {
    return { g1: w1 - 1.5, g2: w2 + 1.0 };
  }
  function noise(seed) {
    var x = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
    return (x - Math.floor(x)) * 2 - 1;
  }
  function batchGrad(w1, w2, batch, step) {
    var g = grad(w1, w2);
    var n1 = 0, n2 = 0;
    for (var i = 0; i < batch; i++) {
      n1 += noise(step * 101 + i * 17 + 1);
      n2 += noise(step * 137 + i * 19 + 2);
    }
    var scale = 2.0;
    return {
      g1: g.g1 + scale * n1 / batch,
      g2: g.g2 + scale * n2 / batch,
      trueG1: g.g1,
      trueG2: g.g2
    };
  }

  host.innerHTML =
    '<div class="land-widget adam-landscape-widget">' +
      '<div class="land-content">' +
        '<div class="land-main">' +
          '<canvas class="land-canvas"></canvas>' +
        '</div>' +
        '<div class="land-param-diagram">' +
          '<p class="land-diagram-title">Two trainable weights</p>' +
          '<div class="land-weight-row"><span>w₁</span><strong data-r="w1">-1.0</strong></div>' +
          '<div class="land-weight-row"><span>w₂</span><strong data-r="w2">2.0</strong></div>' +
          '<div class="land-loss-box">L(w₁, w₂)</div>' +
        '</div>' +
      '</div>' +
      '<div class="land-controls">' +
        '<div class="mlp-slider"><label>η</label><input data-k="eta" type="range" min="0.05" max="1.5" step="0.05" value="0.3"><p data-r="eta">0.3</p></div>' +
        '<div class="mlp-slider"><label>Batch</label><input data-k="batch" type="range" min="1" max="64" step="1" value="8"><p data-r="batch">8</p></div>' +
        '<div class="mlp-slider"><label>β₁</label><input data-k="beta1" type="range" min="0.5" max="0.99" step="0.01" value="0.9"><p data-r="beta1">0.90</p></div>' +
        '<div class="mlp-slider"><label>β₂</label><input data-k="beta2" type="range" min="0.9" max="0.999" step="0.001" value="0.999"><p data-r="beta2">0.999</p></div>' +
        '<button class="land-step">Step</button>' +
        '<button class="land-reset-btn">Reset</button>' +
      '</div>' +
      '<p class="land-readout"></p>' +
    '</div>';
  var canvas = host.querySelector('.land-canvas');
  var readout = host.querySelector('.land-readout');
  ['pointerdown', 'keydown'].forEach(function(ev) {
    host.addEventListener(ev, function(e) { e.stopPropagation(); });
  });

  canvas.addEventListener('pointerdown', function(e) {
    state.dragging = true;
    state.lastMX = e.clientX; state.lastMY = e.clientY;
    state.dragType = e.shiftKey ? 'pan' : 'rotate';
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', function(e) {
    if (!state.dragging) return;
    var dx = e.clientX - state.lastMX, dy = e.clientY - state.lastMY;
    if (state.dragType === 'pan') {
      state.panX += dx; state.panY += dy;
    } else {
      state.az -= dx * 0.008;
      state.el = Math.max(0.1, Math.min(1.2, state.el - dy * 0.008));
    }
    state.lastMX = e.clientX; state.lastMY = e.clientY;
    draw();
  });
  canvas.addEventListener('pointerup', function() { state.dragging = false; state.dragType = null; });
  canvas.addEventListener('wheel', function(e) {
    e.preventDefault();
    state.zoom = Math.max(0.4, Math.min(2.5, state.zoom * (e.deltaY > 0 ? 0.92 : 1.08)));
    draw();
  }, { passive: false });

  function syncWeightReadouts() {
    host.querySelector('[data-r="w1"]').textContent = state.w1.toFixed(2);
    host.querySelector('[data-r="w2"]').textContent = state.w2.toFixed(2);
  }

  function formatControl(key) {
    if (key === 'batch') return String(state.batch);
    if (key === 'beta2') return state.beta2.toFixed(3);
    return state[key].toFixed(2);
  }

  host.querySelectorAll('input').forEach(function(inp) {
    inp.addEventListener('input', function() {
      state[inp.dataset.k] = inp.dataset.k === 'batch' ? parseInt(inp.value, 10) : parseFloat(inp.value);
      host.querySelector('[data-r="' + inp.dataset.k + '"]').textContent = formatControl(inp.dataset.k);
      draw();
    });
  });

  host.querySelector('.land-step').addEventListener('click', function() {
    var g = batchGrad(state.w1, state.w2, state.batch, state.t + 1);
    var oldW1 = state.w1, oldW2 = state.w2;
    state.t++;
    state.m1 = state.beta1 * state.m1 + (1 - state.beta1) * g.g1;
    state.m2 = state.beta1 * state.m2 + (1 - state.beta1) * g.g2;
    state.v1 = state.beta2 * state.v1 + (1 - state.beta2) * g.g1 * g.g1;
    state.v2 = state.beta2 * state.v2 + (1 - state.beta2) * g.g2 * g.g2;
    var m1h = state.m1 / (1 - Math.pow(state.beta1, state.t));
    var m2h = state.m2 / (1 - Math.pow(state.beta1, state.t));
    var v1h = state.v1 / (1 - Math.pow(state.beta2, state.t));
    var v2h = state.v2 / (1 - Math.pow(state.beta2, state.t));
    var eps = 1e-8;
    state.w1 -= state.eta * m1h / (Math.sqrt(v1h) + eps);
    state.w2 -= state.eta * m2h / (Math.sqrt(v2h) + eps);
    state.lastStep = { oldW1: oldW1, oldW2: oldW2, newW1: state.w1, newW2: state.w2, batch: state.batch };
    syncWeightReadouts();
    draw();
  });

  host.querySelector('.land-reset-btn').addEventListener('click', function() {
    state.w1 = -1.0; state.w2 = 2.0; state.eta = 0.3; state.batch = 8;
    state.beta1 = 0.9; state.beta2 = 0.999;
    state.m1 = 0; state.m2 = 0; state.v1 = 0; state.v2 = 0; state.t = 0; state.lastStep = null;
    state.az = -0.8; state.el = 0.65; state.zoom = 1.0; state.panX = 0; state.panY = 0;
    host.querySelector('[data-k="eta"]').value = state.eta;
    host.querySelector('[data-k="batch"]').value = state.batch;
    host.querySelector('[data-k="beta1"]').value = state.beta1;
    host.querySelector('[data-k="beta2"]').value = state.beta2;
    host.querySelector('[data-r="eta"]').textContent = state.eta.toFixed(2);
    host.querySelector('[data-r="batch"]').textContent = String(state.batch);
    host.querySelector('[data-r="beta1"]').textContent = state.beta1.toFixed(2);
    host.querySelector('[data-r="beta2"]').textContent = state.beta2.toFixed(3);
    syncWeightReadouts();
    draw();
  });

  function fitCanvas(c) {
    var dpr = window.devicePixelRatio || 1;
    var w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return null;
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    var ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h };
  }

  function heightColor(z, zmin, zmax) {
    var t = Math.max(0, Math.min(1, (z - zmin) / (zmax - zmin || 1)));
    var r, g, b;
    if (t < 0.25) { r = 0; g = t * 4; b = 1; }
    else if (t < 0.5) { r = 0; g = 1; b = 1 - (t - 0.25) * 4; }
    else if (t < 0.75) { r = (t - 0.5) * 4; g = 1; b = 0; }
    else { r = 1; g = 1 - (t - 0.75) * 4; b = 0; }
    return 'rgba(' + Math.round(r * 255) + ',' + Math.round(g * 255) + ',' + Math.round(b * 255) + ',0.82)';
  }
  function lineColor(z, zmin, zmax) {
    return heightColor(z, zmin, zmax).replace('0.82)', '0.30)');
  }

  function project3D(x, y, z, W, H) {
    var cx = (WMIN + WMAX) / 2, cy = (WMIN + WMAX) / 2;
    var dx = x - cx, dy = y - cy;
    var ca = Math.cos(state.az), sa = Math.sin(state.az);
    var ce = Math.cos(state.el), se = Math.sin(state.el);
    var xr = dx * ca - dy * sa;
    var yr = dx * sa + dy * ca;
    var zr = -z * 0.22;
    var yr2 = yr * ce - zr * se;
    var zr2 = yr * se + zr * ce;
    var scale = Math.min(W, H) * 0.16 * state.zoom;
    return { x: W * 0.5 + state.panX + xr * scale, y: H * 0.55 + state.panY - yr2 * scale * 0.72, depth: zr2 };
  }

  function drawAxis(ctx, from, to, label, color, W, H) {
    var a = project3D(from[0], from[1], from[2], W, H);
    var b = project3D(to[0], to[1], to[2], W, H);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    var lx = Math.max(22, Math.min(W - 22, b.x));
    var ly = Math.max(18, Math.min(H - 14, b.y - 6));
    ctx.fillText(label, lx, ly);
  }

  function drawAxes(ctx, W, H, zmax) {
    var base = [WMIN, WMIN, 0];
    drawAxis(ctx, base, [WMAX, WMIN, 0], 'w₁', PRIMARY, W, H);
    drawAxis(ctx, base, [WMIN, WMAX, 0], 'w₂', SECONDARY, W, H);
    drawAxis(ctx, base, [WMIN, WMIN, zmax * 0.55], 'loss', TEXT, W, H);
  }

  function draw() {
    var f = fitCanvas(canvas);
    if (!f) return;
    var ctx = f.ctx, W = f.w, H = f.h;
    ctx.clearRect(0, 0, W, H);

    var grid = 32;
    var zmin = 0, zmax = loss(WMIN, WMAX);
    var pts = [];
    for (var i = 0; i <= grid; i++) {
      pts[i] = [];
      for (var j = 0; j <= grid; j++) {
        var w1 = WMIN + (WMAX - WMIN) * i / grid;
        var w2 = WMIN + (WMAX - WMIN) * j / grid;
        var z = loss(w1, w2);
        pts[i][j] = project3D(w1, w2, z, W, H);
        pts[i][j].z = z;
      }
    }

    var tris = [];
    for (var i = 0; i < grid; i++) {
      for (var j = 0; j < grid; j++) {
        var a = pts[i][j], b = pts[i + 1][j], c = pts[i][j + 1], d = pts[i + 1][j + 1];
        tris.push({ p: [a, b, c], z: (a.z + b.z + c.z) / 3, d: (a.depth + b.depth + c.depth) / 3 });
        tris.push({ p: [b, d, c], z: (b.z + c.z + d.z) / 3, d: (b.depth + c.depth + d.depth) / 3 });
      }
    }
    tris.sort(function(a, b) { return b.d - a.d; });

    tris.forEach(function(t) {
      ctx.beginPath(); ctx.moveTo(t.p[0].x, t.p[0].y);
      ctx.lineTo(t.p[1].x, t.p[1].y); ctx.lineTo(t.p[2].x, t.p[2].y);
      ctx.closePath(); ctx.fillStyle = heightColor(t.z, zmin, zmax); ctx.fill();
    });

    ctx.lineWidth = 0.6;
    for (var i = 0; i <= grid; i++) {
      for (var j = 0; j < grid; j++) {
        var a = pts[i][j], b = pts[i][j + 1];
        ctx.strokeStyle = lineColor((a.z + b.z) / 2, zmin, zmax);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    }
    for (var j2 = 0; j2 <= grid; j2++) {
      for (var i2 = 0; i2 < grid; i2++) {
        var a2 = pts[i2][j2], b2 = pts[i2 + 1][j2];
        ctx.strokeStyle = lineColor((a2.z + b2.z) / 2, zmin, zmax);
        ctx.beginPath(); ctx.moveTo(a2.x, a2.y); ctx.lineTo(b2.x, b2.y); ctx.stroke();
      }
    }

    drawAxes(ctx, W, H, zmax);

    var curL = loss(state.w1, state.w2);
    var curP = project3D(state.w1, state.w2, curL, W, H);
    ctx.shadowColor = RED; ctx.shadowBlur = 14;
    ctx.beginPath(); ctx.arc(curP.x, curP.y, 8, 0, 6.2832);
    ctx.fillStyle = RED; ctx.fill(); ctx.shadowBlur = 0;

    if (state.lastStep) {
      var oldP = project3D(state.lastStep.oldW1, state.lastStep.oldW2,
                           loss(state.lastStep.oldW1, state.lastStep.oldW2), W, H);
      ctx.strokeStyle = SECONDARY; ctx.lineWidth = 2.5;
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(oldP.x, oldP.y); ctx.lineTo(curP.x, curP.y); ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.fillStyle = MUTED; ctx.font = '12px Inter, sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('Drag: rotate | Shift+drag: pan | Scroll: zoom', 10, H - 10);

    var g = batchGrad(state.w1, state.w2, state.batch, state.t + 1);
    var eps = 1e-8;
    if (state.t > 0) {
      var m1h = state.m1 / (1 - Math.pow(state.beta1, state.t));
      var m2h = state.m2 / (1 - Math.pow(state.beta1, state.t));
      var v1h = state.v1 / (1 - Math.pow(state.beta2, state.t));
      var v2h = state.v2 / (1 - Math.pow(state.beta2, state.t));
      readout.innerHTML =
        'Step ' + state.t + ': w=[' + state.w1.toFixed(2) + ', ' + state.w2.toFixed(2) +
        '] loss=<strong>' + curL.toFixed(3) + '</strong> | B=' + state.batch + ' ' +
        '<span style="color:' + PRIMARY + '">m̂</span>=[' + m1h.toFixed(3) + ', ' + m2h.toFixed(3) + '] ' +
        '<span style="color:' + SECONDARY + '">v̂</span>=[' + v1h.toFixed(3) + ', ' + v2h.toFixed(3) + '] ' +
        '| <span style="color:' + GREEN + '">step</span>=[' +
        (state.eta * m1h / (Math.sqrt(v1h) + eps)).toFixed(4) + ', ' +
        (state.eta * m2h / (Math.sqrt(v2h) + eps)).toFixed(4) + ']';
    } else {
      readout.innerHTML =
        'w=[' + state.w1.toFixed(2) + ', ' + state.w2.toFixed(2) +
        '] loss=<strong>' + curL.toFixed(3) + '</strong> | Batch gradient B=' + state.batch +
        ': [' + g.g1.toFixed(3) + ', ' + g.g2.toFixed(3) + '] (true: [' +
        g.trueG1.toFixed(3) + ', ' + g.trueG2.toFixed(3) + ']).';
    }
  }
  syncWeightReadouts();
  return { resize: draw };
};
