// =====================================================================
// Loss landscape widget — INTERACTIVE_WIDGETS.lossLandscape
// 3D draggable surface with height coloring. Shows SGD step-by-step
// with noisy batch gradients.
// =====================================================================
INTERACTIVE_WIDGETS.lossLandscape = function(host) {
  var TEXT = '#e8eaf0', MUTED = '#8892a4', PRIMARY = '#4a9eff',
      SECONDARY = '#f5a623', RED = '#e74c3c', LINEC = '#2a3450';
  var state = { w1: -1.0, w2: 2.0, eta: 0.5, batch: 8, t: 0, lastStep: null,
                az: -0.8, el: 0.65, zoom: 1.0, panX: 0, panY: 0,
                dragging: false, dragType: null, lastMX: 0, lastMY: 0 };
  var WMIN = -3, WMAX = 5;
  var INIT_W1 = -1.0, INIT_W2 = 2.0, INIT_ETA = 0.5, INIT_BATCH = 8;
  var INIT_AZ = -0.8, INIT_EL = 0.65, INIT_ZOOM = 1.0;

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
    '<div class="land-widget loss-landscape-widget">' +
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
        '<div class="mlp-slider"><label>η</label><input data-k="eta" type="range" min="0.1" max="1.5" step="0.1" value="0.5"><p data-r="eta">0.5</p></div>' +
        '<div class="mlp-slider"><label>Batch</label><input data-k="batch" type="range" min="1" max="64" step="1" value="8"><p data-r="batch">8</p></div>' +
        '<button class="land-step">Step</button>' +
        '<button class="land-reset-btn">Reset</button>' +
      '</div>' +
      '<p class="land-readout"></p>' +
    '</div>';
  var canvas = host.querySelector('.land-canvas');
  var readout = host.querySelector('.land-readout');
  var resetBtn = host.querySelector('.land-reset-btn');
  ['pointerdown', 'keydown'].forEach(function(ev) {
    host.addEventListener(ev, function(e) { e.stopPropagation(); });
  });

  canvas.addEventListener('pointerdown', function(e) {
    state.dragging = true;
    state.lastMX = e.clientX; state.lastMY = e.clientY;
    if (e.shiftKey) {
      state.dragType = 'pan';
    } else {
      state.dragType = 'rotate';
    }
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', function(e) {
    if (!state.dragging) return;
    var dx = e.clientX - state.lastMX, dy = e.clientY - state.lastMY;
    if (state.dragType === 'pan') {
      state.panX += dx;
      state.panY += dy;
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

  host.querySelectorAll('input').forEach(function(inp) {
    inp.addEventListener('input', function() {
      state[inp.dataset.k] = inp.dataset.k === 'batch' ? parseInt(inp.value, 10) : parseFloat(inp.value);
      state.lastStep = null;
      host.querySelector('[data-r="' + inp.dataset.k + '"]').textContent =
        inp.dataset.k === 'batch' ? String(state.batch) : state[inp.dataset.k].toFixed(1);
      draw();
    });
  });
  host.querySelector('.land-step').addEventListener('click', function() {
    var g = batchGrad(state.w1, state.w2, state.batch, state.t + 1);
    var oldW1 = state.w1, oldW2 = state.w2;
    state.w1 += -state.eta * g.g1;
    state.w2 += -state.eta * g.g2;
    state.t++;
    state.lastStep = { oldW1: oldW1, oldW2: oldW2, g1: g.g1, g2: g.g2,
                       trueG1: g.trueG1, trueG2: g.trueG2, eta: state.eta,
                       batch: state.batch, newW1: state.w1, newW2: state.w2 };
    syncWeightReadouts();
    draw();
  });
  resetBtn.addEventListener('click', function() {
    state.w1 = INIT_W1; state.w2 = INIT_W2; state.eta = INIT_ETA; state.batch = INIT_BATCH; state.t = 0;
    state.lastStep = null;
    state.az = INIT_AZ; state.el = INIT_EL; state.zoom = INIT_ZOOM;
    state.panX = 0; state.panY = 0;
    host.querySelector('[data-k="eta"]').value = state.eta;
    host.querySelector('[data-k="batch"]').value = state.batch;
    syncWeightReadouts();
    host.querySelector('[data-r="eta"]').textContent = state.eta.toFixed(1);
    host.querySelector('[data-r="batch"]').textContent = String(state.batch);
    draw();
  });

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
    var c = heightColor(z, zmin, zmax);
    return c.replace('0.82)', '0.30)');
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
    var px = W * 0.5 + state.panX + xr * scale;
    var py = H * 0.55 + state.panY - yr2 * scale * 0.72;
    return { x: px, y: py, depth: zr2 };
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
        var zt1 = (a.z + b.z + c.z) / 3;
        var zt2 = (b.z + c.z + d.z) / 3;
        var dt1 = (a.depth + b.depth + c.depth) / 3;
        var dt2 = (b.depth + c.depth + d.depth) / 3;
        tris.push({ p: [a, b, c], z: zt1, d: dt1 });
        tris.push({ p: [b, d, c], z: zt2, d: dt2 });
      }
    }
    tris.sort(function(a, b) { return b.d - a.d; });

    tris.forEach(function(t) {
      ctx.beginPath();
      ctx.moveTo(t.p[0].x, t.p[0].y);
      ctx.lineTo(t.p[1].x, t.p[1].y);
      ctx.lineTo(t.p[2].x, t.p[2].y);
      ctx.closePath();
      ctx.fillStyle = heightColor(t.z, zmin, zmax);
      ctx.fill();
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
    ctx.fillStyle = RED; ctx.fill();
    ctx.shadowBlur = 0;

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
    if (state.lastStep) {
      readout.innerHTML =
        'Step: [' + state.lastStep.newW1.toFixed(2) + ', ' + state.lastStep.newW2.toFixed(2) +
        '] = [' + state.lastStep.oldW1.toFixed(2) + ', ' + state.lastStep.oldW2.toFixed(2) +
        '] + (-' + state.lastStep.eta.toFixed(1) + ' · batch-gradient B=' + state.lastStep.batch + ' [' +
        state.lastStep.g1.toFixed(3) + ', ' + state.lastStep.g2.toFixed(3) +
        ']). Current loss: <strong>' + curL.toFixed(3) + '</strong>.';
    } else {
      readout.innerHTML =
        'Batch gradient B=' + state.batch + ': [' + g.g1.toFixed(3) + ', ' + g.g2.toFixed(3) +
        '] (true: [' + g.trueG1.toFixed(3) + ', ' + g.trueG2.toFixed(3) + '])' +
        '. Next: w_new = [' + state.w1.toFixed(2) + ', ' + state.w2.toFixed(2) +
        '] + (-' + state.eta.toFixed(1) + ' · [' + g.g1.toFixed(3) + ', ' + g.g2.toFixed(3) + ']).';
    }
  }
  syncWeightReadouts();
  return { resize: draw };
};
