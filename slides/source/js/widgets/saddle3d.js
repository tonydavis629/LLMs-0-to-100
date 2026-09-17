// =====================================================================
// Saddle widget — INTERACTIVE_WIDGETS.saddle3d
// A loss surface with a hill in front of the bowl. Along one weight the
// ball is stuck behind the hill (a local minimum of the 1D slice). With a
// second weight the same descent walks around the hill. Drag to rotate.
// =====================================================================
INTERACTIVE_WIDGETS.saddle3d = function(host) {
  var TEXT = '#e8eaf0', MUTED = '#8892a4', PRIMARY = '#4a9eff', GREEN = '#3fb950', ORANGE = '#f5a623';
  var R = 1.9, GRID = 40;
  function loss(a, b) {
    return 0.3 * ((a - 1.4) * (a - 1.4) + b * b) + 2.4 * Math.exp(-((a - 0.1) * (a - 0.1) / 0.45 + b * b / 0.9));
  }
  function grad(a, b) {
    var h = 1e-4;
    return [(loss(a + h, b) - loss(a - h, b)) / (2 * h), (loss(a, b + h) - loss(a, b - h)) / (2 * h)];
  }
  function descend(a, b, free, steps, lr) {
    var path = [[a, b]];
    for (var i = 0; i < steps; i++) {
      var g = grad(a, b);
      a -= lr * g[0];
      if (free) b -= lr * g[1];
      path.push([a, b]);
    }
    return path;
  }
  var START = [-1.4, 0.2], STEPS = 600, LR = 0.08;
  var path1 = descend(START[0], START[1], false, STEPS, LR);
  var path2 = descend(START[0], START[1], true, STEPS, LR);
  var VIEW0 = { az: -2.4, el: -0.7 };
  var state = { az: VIEW0.az, el: VIEW0.el, t: 0, playing: false, timer: null, dragging: false, lastX: 0, lastY: 0 };

  host.innerHTML =
    '<div class="mlp-widget saddle-widget">' +
      '<div class="mlp-canvas-wrap"><canvas class="mlp-canvas"></canvas></div>' +
      '<div class="mlp-controls">' +
        '<div class="expl-datasets"><button class="expl-ds-btn saddle-play">Run descent</button><button class="expl-ds-btn saddle-reset">Reset</button><button class="expl-ds-btn saddle-view">Reset view</button></div>' +
      '</div>' +
    '</div>';
  var canvas = host.querySelector('.mlp-canvas');
  ['pointerdown', 'keydown'].forEach(function(ev) {
    host.addEventListener(ev, function(e) { e.stopPropagation(); });
  });
  canvas.addEventListener('pointerdown', function(e) {
    state.dragging = true; state.lastX = e.clientX; state.lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', function(e) {
    if (!state.dragging) return;
    state.az -= (e.clientX - state.lastX) * 0.008;
    state.el = Math.max(-1.4, Math.min(1.4, state.el + (e.clientY - state.lastY) * 0.008));
    state.lastX = e.clientX; state.lastY = e.clientY;
    draw();
  });
  ['pointerup', 'pointercancel', 'pointerleave'].forEach(function(ev) {
    canvas.addEventListener(ev, function() { state.dragging = false; });
  });
  function stop() { if (state.timer) { clearInterval(state.timer); state.timer = null; } state.playing = false; }
  host.querySelector('.saddle-play').addEventListener('click', function() {
    if (state.playing) { stop(); return; }
    if (state.t >= STEPS) state.t = 0;
    state.playing = true;
    state.timer = setInterval(function() {
      if (!host.isConnected) { stop(); return; }
      state.t = Math.min(STEPS, state.t + 6);
      draw();
      if (state.t >= STEPS) stop();
    }, 40);
  });
  host.querySelector('.saddle-reset').addEventListener('click', function() { stop(); state.t = 0; draw(); });
  host.querySelector('.saddle-view').addEventListener('click', function() { state.az = VIEW0.az; state.el = VIEW0.el; draw(); });

  function fitCanvas(c) {
    var dpr = window.devicePixelRatio || 1;
    var wd = c.clientWidth, h = c.clientHeight;
    if (!wd || !h) return null;
    c.width = Math.round(wd * dpr); c.height = Math.round(h * dpr);
    var ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: wd, h: h };
  }
  var ZSCALE = 0.3;
  function project(a, b, z, Wd, H) {
    var x = a / R, y = b / R, zz = z * ZSCALE - 0.45;
    var ca = Math.cos(state.az), sa = Math.sin(state.az);
    var ce = Math.cos(state.el), se = Math.sin(state.el);
    var xr = x * ca - y * sa;
    var yr = x * sa + y * ca;
    var y2 = yr * ce - zz * se;
    var depth = yr * se + zz * ce;
    var scale = Math.min(Wd, H) * 0.36;
    return { x: Wd * 0.5 + xr * scale, y: H * 0.54 - y2 * scale, depth: depth };
  }
  function heightColor(z, zmin, zmax, alpha) {
    var t = Math.max(0, Math.min(1, (z - zmin) / (zmax - zmin || 1)));
    var bands = [[38, 92, 255], [0, 214, 255], [38, 217, 87], [248, 227, 64], [255, 159, 28], [255, 59, 48]];
    var c = bands[Math.min(bands.length - 1, Math.floor(t * bands.length))];
    return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + alpha + ')';
  }

  function draw() {
    var f = fitCanvas(canvas);
    if (!f) return;
    var ctx = f.ctx, Wd = f.w, H = f.h;
    ctx.clearRect(0, 0, Wd, H);

    var pts = [], zmin = Infinity, zmax = -Infinity;
    for (var i = 0; i <= GRID; i++) {
      pts[i] = [];
      for (var j = 0; j <= GRID; j++) {
        var a = -R + 2 * R * i / GRID, b = -R + 2 * R * j / GRID, z = loss(a, b);
        zmin = Math.min(zmin, z); zmax = Math.max(zmax, z);
        var q = project(a, b, z, Wd, H); q.z = z; pts[i].push(q);
      }
    }
    var tris = [];
    for (var ti = 0; ti < GRID; ti++) {
      for (var tj = 0; tj < GRID; tj++) {
        var p00 = pts[ti][tj], p10 = pts[ti + 1][tj], p01 = pts[ti][tj + 1], p11 = pts[ti + 1][tj + 1];
        tris.push({ p: [p00, p10, p01], z: (p00.z + p10.z + p01.z) / 3, d: (p00.depth + p10.depth + p01.depth) / 3 });
        tris.push({ p: [p10, p11, p01], z: (p10.z + p11.z + p01.z) / 3, d: (p10.depth + p11.depth + p01.depth) / 3 });
      }
    }
    tris.sort(function(x, y) { return x.d - y.d; });
    tris.forEach(function(t) {
      ctx.beginPath(); ctx.moveTo(t.p[0].x, t.p[0].y); ctx.lineTo(t.p[1].x, t.p[1].y); ctx.lineTo(t.p[2].x, t.p[2].y); ctx.closePath();
      ctx.fillStyle = heightColor(t.z, zmin, zmax, 0.9); ctx.fill();
      ctx.strokeStyle = 'rgba(8,12,24,0.35)'; ctx.lineWidth = 0.5; ctx.stroke();
    });

    // Axis lines along the two base edges that meet at the nearest corner,
    // with w1 / w2 labels at the midpoints of those edges
    var zb = zmin - 0.15, near = null;
    [[-R, -R], [R, -R], [R, R], [-R, R]].forEach(function(c) {
      var q = project(c[0], c[1], zb, Wd, H);
      if (!near || q.depth > near.depth) near = { a: c[0], b: c[1], depth: q.depth };
    });
    var c0 = project(near.a, near.b, zb, Wd, H);
    var ea = project(-near.a, near.b, zb, Wd, H), eb = project(near.a, -near.b, zb, Wd, H);
    ctx.strokeStyle = MUTED; ctx.lineWidth = 1.5; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(c0.x, c0.y); ctx.lineTo(ea.x, ea.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(c0.x, c0.y); ctx.lineTo(eb.x, eb.y); ctx.stroke();
    ctx.fillStyle = TEXT; ctx.font = '15px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    var sb = near.b > 0 ? 1 : -1, sa = near.a > 0 ? 1 : -1;
    var lx = project(0, near.b + sb * 0.4, zb, Wd, H), ly = project(near.a + sa * 0.4, 0, zb, Wd, H);
    ctx.fillText('w₁', lx.x, lx.y); ctx.fillText('w₂', ly.x, ly.y);
    ctx.textBaseline = 'alphabetic';

    // The 1D slice (w2 = 0.15) that the one-weight descent lives on
    ctx.strokeStyle = 'rgba(245,166,35,0.55)'; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.beginPath();
    for (var k = 0; k <= 60; k++) {
      var sa = -R + 2 * R * k / 60, sq = project(sa, START[1], loss(sa, START[1]) + 0.02, Wd, H);
      if (k === 0) ctx.moveTo(sq.x, sq.y); else ctx.lineTo(sq.x, sq.y);
    }
    ctx.stroke(); ctx.setLineDash([]);

    function drawPath(path, color, upto) {
      ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.beginPath();
      for (var m = 0; m <= upto; m++) {
        var q = project(path[m][0], path[m][1], loss(path[m][0], path[m][1]) + 0.03, Wd, H);
        if (m === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y);
      }
      ctx.stroke();
      var e = path[upto], qe = project(e[0], e[1], loss(e[0], e[1]) + 0.03, Wd, H);
      ctx.beginPath(); ctx.arc(qe.x, qe.y, 7, 0, 6.2832);
      ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = TEXT; ctx.lineWidth = 1.5; ctx.stroke();
    }
    drawPath(path1, ORANGE, state.t);
    drawPath(path2, GREEN, state.t);

    ctx.fillStyle = TEXT; ctx.font = '13px Inter, sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('step ' + state.t + ' of ' + STEPS, 12, 20);
    ctx.fillStyle = ORANGE; ctx.fillText('one weight: L = ' + loss(path1[state.t][0], path1[state.t][1]).toFixed(2), 12, 40);
    ctx.fillStyle = GREEN; ctx.fillText('two weights: L = ' + loss(path2[state.t][0], path2[state.t][1]).toFixed(2), 12, 58);
  }
  return { resize: draw, setView: function(az, el) { state.az = az; state.el = el; draw(); }, setT: function(t) { state.t = t; draw(); } };
};
