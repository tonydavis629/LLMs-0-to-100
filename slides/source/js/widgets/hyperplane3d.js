// =====================================================================
// Hyperplane widget — INTERACTIVE_WIDGETS.hyperplane3d
// Real data: Fisher's Iris (1936), setosa vs. versicolor, three
// measurements per flower. A logistic-regression perceptron fitted on
// these three inputs gives w and b; its boundary w . x + b = 0 is a
// plane, drawn clipped to the data box. Drag to rotate.
// =====================================================================
INTERACTIVE_WIDGETS.hyperplane3d = function(host) {
  var TEXT = '#e8eaf0', MUTED = '#8892a4', PRIMARY = '#4a9eff', RED = '#e74c3c', GREEN = '#3fb950';
  // [sepal length, petal length, petal width, class] in cm; class 0 = setosa, 1 = versicolor
  var DATA = [[5.1,1.4,0.2,0],[4.9,1.4,0.2,0],[4.7,1.3,0.2,0],[4.6,1.5,0.2,0],[5.0,1.4,0.2,0],[5.4,1.7,0.4,0],[4.6,1.4,0.3,0],[5.0,1.5,0.2,0],[4.4,1.4,0.2,0],[4.9,1.5,0.1,0],[5.4,1.5,0.2,0],[4.8,1.6,0.2,0],[4.8,1.4,0.1,0],[4.3,1.1,0.1,0],[5.8,1.2,0.2,0],[5.7,1.5,0.4,0],[5.4,1.3,0.4,0],[5.1,1.4,0.3,0],[5.7,1.7,0.3,0],[5.1,1.5,0.3,0],[5.4,1.7,0.2,0],[5.1,1.5,0.4,0],[4.6,1.0,0.2,0],[5.1,1.7,0.5,0],[4.8,1.9,0.2,0],[5.0,1.6,0.2,0],[5.0,1.6,0.4,0],[5.2,1.5,0.2,0],[5.2,1.4,0.2,0],[4.7,1.6,0.2,0],[4.8,1.6,0.2,0],[5.4,1.5,0.4,0],[5.2,1.5,0.1,0],[5.5,1.4,0.2,0],[4.9,1.5,0.1,0],[5.0,1.2,0.2,0],[5.5,1.3,0.2,0],[4.9,1.5,0.1,0],[4.4,1.3,0.2,0],[5.1,1.5,0.2,0],[5.0,1.3,0.3,0],[4.5,1.3,0.3,0],[4.4,1.3,0.2,0],[5.0,1.6,0.6,0],[5.1,1.9,0.4,0],[4.8,1.4,0.3,0],[5.1,1.6,0.2,0],[4.6,1.4,0.2,0],[5.3,1.5,0.2,0],[5.0,1.4,0.2,0],[7.0,4.7,1.4,1],[6.4,4.5,1.5,1],[6.9,4.9,1.5,1],[5.5,4.0,1.3,1],[6.5,4.6,1.5,1],[5.7,4.5,1.3,1],[6.3,4.7,1.6,1],[4.9,3.3,1.0,1],[6.6,4.6,1.3,1],[5.2,3.9,1.4,1],[5.0,3.5,1.0,1],[5.9,4.2,1.5,1],[6.0,4.0,1.0,1],[6.1,4.7,1.4,1],[5.6,3.6,1.3,1],[6.7,4.4,1.4,1],[5.6,4.5,1.5,1],[5.8,4.1,1.0,1],[6.2,4.5,1.5,1],[5.6,3.9,1.1,1],[5.9,4.8,1.8,1],[6.1,4.0,1.3,1],[6.3,4.9,1.5,1],[6.1,4.7,1.2,1],[6.4,4.3,1.3,1],[6.6,4.4,1.4,1],[6.8,4.8,1.4,1],[6.7,5.0,1.7,1],[6.0,4.5,1.5,1],[5.7,3.5,1.0,1],[5.5,3.8,1.1,1],[5.5,3.7,1.0,1],[5.8,3.9,1.2,1],[6.0,5.1,1.6,1],[5.4,4.5,1.5,1],[6.0,4.5,1.6,1],[6.7,4.7,1.5,1],[6.3,4.4,1.3,1],[5.6,4.1,1.3,1],[5.5,4.0,1.3,1],[5.5,4.4,1.2,1],[6.1,4.6,1.4,1],[5.8,4.0,1.2,1],[5.0,3.3,1.0,1],[5.6,4.2,1.3,1],[5.7,4.2,1.2,1],[5.7,4.2,1.3,1],[6.2,4.3,1.3,1],[5.1,3.0,1.1,1],[5.7,4.1,1.3,1]];
  var W = [0.517, 2.347, 5.788], B = -13.145;
  var AX = [
    { name: 'sepal length (cm)', lo: 4.0, hi: 7.4, ticks: [4.5, 5.5, 6.5] },
    { name: 'petal length (cm)', lo: 0.6, hi: 5.6, ticks: [1, 2, 3, 4, 5] },
    { name: 'petal width (cm)', lo: 0.0, hi: 2.0, ticks: [0.5, 1.0, 1.5] }
  ];
  var state = { az: -0.7854, el: -0.6155, dragging: false, lastX: 0, lastY: 0, spin: false, timer: null };
  var VIEW0 = { az: -0.7854, el: -0.6155 };

  host.innerHTML =
    '<div class="mlp-widget hyper-widget">' +
      '<div class="mlp-canvas-wrap"><canvas class="mlp-canvas"></canvas></div>' +
      '<div class="mlp-controls">' +
        '<div class="expl-datasets"><button class="expl-ds-btn hyper-spin">Rotate</button><button class="expl-ds-btn hyper-reset">Reset view</button></div>' +
        '<p class="mlp-readout">Iris (Fisher 1936): <span style="color:#e74c3c">setosa</span> vs. <span style="color:#3fb950">versicolor</span>, 100 flowers. Drag to rotate.</p>' +
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
    state.el = Math.max(-1.3, Math.min(1.3, state.el + (e.clientY - state.lastY) * 0.008));
    state.lastX = e.clientX; state.lastY = e.clientY;
    draw();
  });
  ['pointerup', 'pointercancel', 'pointerleave'].forEach(function(ev) {
    canvas.addEventListener(ev, function() { state.dragging = false; });
  });
  var spinBtn = host.querySelector('.hyper-spin');
  spinBtn.addEventListener('click', function() {
    state.spin = !state.spin;
    spinBtn.classList.toggle('active', state.spin);
    if (state.spin && !state.timer) {
      state.timer = setInterval(function() {
        if (!state.spin || !host.isConnected) { clearInterval(state.timer); state.timer = null; state.spin = false; spinBtn.classList.remove('active'); return; }
        state.az += 0.012; draw();
      }, 40);
    }
  });
  host.querySelector('.hyper-reset').addEventListener('click', function() {
    state.az = VIEW0.az; state.el = VIEW0.el; draw();
  });

  // Raw cm -> normalized [-1, 1] per axis so the box is a cube on screen.
  function nrm(p) {
    return [
      (p[0] - AX[0].lo) / (AX[0].hi - AX[0].lo) * 2 - 1,
      (p[1] - AX[1].lo) / (AX[1].hi - AX[1].lo) * 2 - 1,
      (p[2] - AX[2].lo) / (AX[2].hi - AX[2].lo) * 2 - 1
    ];
  }
  function fitCanvas(c) {
    var dpr = window.devicePixelRatio || 1;
    var wd = c.clientWidth, h = c.clientHeight;
    if (!wd || !h) return null;
    c.width = Math.round(wd * dpr); c.height = Math.round(h * dpr);
    var ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: wd, h: h };
  }
  function project(n, Wd, H) {
    var ca = Math.cos(state.az), sa = Math.sin(state.az);
    var ce = Math.cos(state.el), se = Math.sin(state.el);
    var xr = n[0] * ca - n[1] * sa;
    var yr = n[0] * sa + n[1] * ca;
    var y2 = yr * ce - n[2] * se;
    var depth = yr * se + n[2] * ce;
    var scale = Math.min(Wd, H) * 0.30;
    return { x: Wd * 0.5 + xr * scale, y: H * 0.5 - y2 * scale, depth: depth };
  }
  function viewDir() {
    var ce = Math.cos(state.el), se = Math.sin(state.el);
    var ca = Math.cos(state.az), sa = Math.sin(state.az);
    return [sa * se, ca * se, ce];
  }
  function planeVal(p) { return W[0] * p[0] + W[1] * p[1] + W[2] * p[2] + B; }

  // Polygon where the plane cuts the data box (in raw cm), ordered around its centroid.
  function planePolygon() {
    var lo = [AX[0].lo, AX[1].lo, AX[2].lo], hi = [AX[0].hi, AX[1].hi, AX[2].hi];
    var corners = [];
    for (var i = 0; i < 8; i++) corners.push([i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]]);
    var edges = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
    var pts = [];
    edges.forEach(function(e) {
      var a = corners[e[0]], b = corners[e[1]];
      var va = planeVal(a), vb = planeVal(b);
      if ((va > 0) === (vb > 0)) return;
      var t = va / (va - vb);
      pts.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]);
    });
    if (pts.length < 3) return [];
    var c = [0, 0, 0];
    pts.forEach(function(p) { c[0] += p[0] / pts.length; c[1] += p[1] / pts.length; c[2] += p[2] / pts.length; });
    var n = nrm(pts[0]), cn = nrm(c);
    var u = [n[0] - cn[0], n[1] - cn[1], n[2] - cn[2]];
    var ul = Math.hypot(u[0], u[1], u[2]) || 1; u = [u[0] / ul, u[1] / ul, u[2] / ul];
    var q = nrm(pts[1]);
    var t2 = [q[0] - cn[0], q[1] - cn[1], q[2] - cn[2]];
    var nn = [u[1] * t2[2] - u[2] * t2[1], u[2] * t2[0] - u[0] * t2[2], u[0] * t2[1] - u[1] * t2[0]];
    var nl = Math.hypot(nn[0], nn[1], nn[2]) || 1; nn = [nn[0] / nl, nn[1] / nl, nn[2] / nl];
    var v = [nn[1] * u[2] - nn[2] * u[1], nn[2] * u[0] - nn[0] * u[2], nn[0] * u[1] - nn[1] * u[0]];
    pts.sort(function(p1, p2) {
      var a1 = nrm(p1), a2 = nrm(p2);
      var d1 = [a1[0] - cn[0], a1[1] - cn[1], a1[2] - cn[2]], d2 = [a2[0] - cn[0], a2[1] - cn[1], a2[2] - cn[2]];
      return Math.atan2(d1[0] * v[0] + d1[1] * v[1] + d1[2] * v[2], d1[0] * u[0] + d1[1] * u[1] + d1[2] * u[2]) -
             Math.atan2(d2[0] * v[0] + d2[1] * v[1] + d2[2] * v[2], d2[0] * u[0] + d2[1] * u[1] + d2[2] * u[2]);
    });
    return { pts: pts, center: c, normal: nn, centerN: cn };
  }

  function draw() {
    var f = fitCanvas(canvas);
    if (!f) return;
    var ctx = f.ctx, Wd = f.w, H = f.h;
    ctx.clearRect(0, 0, Wd, H);

    // Box
    var c = [];
    for (var i = 0; i < 8; i++) c.push(project([i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1], Wd, H));
    var edges = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
    ctx.strokeStyle = 'rgba(136,146,164,0.3)'; ctx.lineWidth = 1;
    edges.forEach(function(e) {
      ctx.beginPath(); ctx.moveTo(c[e[0]].x, c[e[0]].y); ctx.lineTo(c[e[1]].x, c[e[1]].y); ctx.stroke();
    });

    // Axes from the (lo, lo, lo) corner with ticks and names
    ctx.font = '12px Inter, sans-serif';
    AX.forEach(function(ax, k) {
      var a = [-1, -1, -1], b = [-1, -1, -1]; b[k] = 1;
      var pa = project(a, Wd, H), pb = project(b, Wd, H);
      ctx.strokeStyle = MUTED; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
      var dx = pb.x - pa.x, dy = pb.y - pa.y, len = Math.hypot(dx, dy) || 1;
      var px = -dy / len, py = dx / len;
      ctx.fillStyle = MUTED; ctx.textAlign = 'center';
      ax.ticks.forEach(function(t) {
        var q = [-1, -1, -1]; q[k] = (t - ax.lo) / (ax.hi - ax.lo) * 2 - 1;
        var pq = project(q, Wd, H);
        ctx.beginPath(); ctx.moveTo(pq.x - px * 4, pq.y - py * 4); ctx.lineTo(pq.x + px * 4, pq.y + py * 4); ctx.stroke();
        ctx.fillText(String(t), pq.x + px * 14, pq.y + py * 14 + 4);
      });
      var e = [-1, -1, -1]; e[k] = 1.28;
      var pe = project(e, Wd, H);
      ctx.fillStyle = TEXT; ctx.font = '13px Inter, sans-serif';
      ctx.fillText(ax.name, pe.x, pe.y + 4);
      ctx.font = '12px Inter, sans-serif';
    });

    // Points split by which side of the plane faces the camera
    var vd = viewDir();
    var poly = planePolygon();
    var nrmN = poly.normal || [0, 0, 1];
    var camSide = vd[0] * nrmN[0] + vd[1] * nrmN[1] + vd[2] * nrmN[2];
    var behind = [], front = [];
    DATA.forEach(function(p) {
      var n = nrm(p), d = [n[0] - poly.centerN[0], n[1] - poly.centerN[1], n[2] - poly.centerN[2]];
      var side = d[0] * nrmN[0] + d[1] * nrmN[1] + d[2] * nrmN[2];
      (side * camSide > 0 ? front : behind).push(p);
    });
    function drawPts(list) {
      list.map(function(p) { var q = project(nrm(p), Wd, H); q.cls = p[3]; return q; })
          .sort(function(a, b) { return a.depth - b.depth; })
          .forEach(function(q) {
            ctx.beginPath(); ctx.arc(q.x, q.y, 4.5, 0, 6.2832);
            ctx.fillStyle = q.cls ? GREEN : RED; ctx.globalAlpha = 0.92; ctx.fill(); ctx.globalAlpha = 1;
          });
    }
    drawPts(behind);

    if (poly.pts && poly.pts.length >= 3) {
      ctx.beginPath();
      poly.pts.forEach(function(p, i) { var q = project(nrm(p), Wd, H); if (i === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y); });
      ctx.closePath();
      ctx.fillStyle = 'rgba(74,158,255,0.22)'; ctx.fill();
      ctx.strokeStyle = PRIMARY; ctx.lineWidth = 1.6; ctx.stroke();

      // Normal vector w from the plane's center, pointing toward the versicolor side
      var sign = camSide;
      var vers = DATA.filter(function(p) { return p[3] === 1; })[0];
      var dv = nrm(vers), dd = [dv[0] - poly.centerN[0], dv[1] - poly.centerN[1], dv[2] - poly.centerN[2]];
      var toVers = (dd[0] * nrmN[0] + dd[1] * nrmN[1] + dd[2] * nrmN[2]) > 0 ? 1 : -1;
      var o = project(poly.centerN, Wd, H);
      var tipN = [poly.centerN[0] + nrmN[0] * 0.55 * toVers, poly.centerN[1] + nrmN[1] * 0.55 * toVers, poly.centerN[2] + nrmN[2] * 0.55 * toVers];
      var tip = project(tipN, Wd, H);
      ctx.strokeStyle = PRIMARY; ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(tip.x, tip.y); ctx.stroke();
      var dx = tip.x - o.x, dy = tip.y - o.y, len = Math.hypot(dx, dy) || 1;
      var ux = dx / len, uy = dy / len;
      ctx.fillStyle = PRIMARY;
      ctx.beginPath(); ctx.moveTo(tip.x, tip.y);
      ctx.lineTo(tip.x - ux * 10 - uy * 5, tip.y - uy * 10 + ux * 5);
      ctx.lineTo(tip.x - ux * 10 + uy * 5, tip.y - uy * 10 - ux * 5);
      ctx.closePath(); ctx.fill();
      ctx.font = 'bold 14px Inter, sans-serif'; ctx.textAlign = 'left';
      ctx.fillStyle = 'rgba(13,18,37,0.85)'; ctx.fillRect(tip.x + ux * 8 + 1, tip.y + uy * 8 - 8, 16, 17);
      ctx.fillStyle = PRIMARY; ctx.fillText('w', tip.x + ux * 8 + 4, tip.y + uy * 8 + 5);
    }
    drawPts(front);

    ctx.fillStyle = TEXT; ctx.font = '13px Inter, sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('z = w · x + b = 0 on the plane', 12, 20);
    ctx.fillStyle = GREEN; ctx.fillText('w side: z > 0, \u0177 > 0.5, versicolor', 12, 40);
    ctx.fillStyle = RED; ctx.fillText('other side: z < 0, \u0177 < 0.5, setosa', 12, 58);
  }
  return { resize: draw, setView: function(az, el) { state.az = az; state.el = el; draw(); } };
};
