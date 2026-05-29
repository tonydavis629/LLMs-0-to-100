(function() {
  function createLossSurfaceWidget(host, mode) {
    var isAdam = mode === "adam";
    var TEXT = "#e8eaf0";
    var PRIMARY = "#4a9eff";
    var SECONDARY = "#f5a623";
    var GRAD = "#e74c3c";
    var STEP = "#3fb950";
    var TANGENT = "#ff2d55";
    var WMIN = -3;
    var WMAX = 5;
    var EPS = 1e-8;
    var INIT_AZ = -0.92;
    var INIT_EL = 0.36;
    var INIT_ZOOM = 0.96;
    var defaults = isAdam
      ? { eta: 0.3, batch: 8, beta1: 0.9, beta2: 0.999 }
      : { eta: 0.5, batch: 8 };
    var state = {
      w1: -1,
      w2: 2,
      eta: defaults.eta,
      batch: defaults.batch,
      beta1: isAdam ? defaults.beta1 : 0.9,
      beta2: isAdam ? defaults.beta2 : 0.999,
      m1: 0,
      m2: 0,
      v1: 0,
      v2: 0,
      t: 0,
      lastStep: null,
      az: INIT_AZ,
      el: INIT_EL,
      zoom: INIT_ZOOM,
      panX: 0,
      panY: 0,
      dragging: false,
      dragType: null,
      lastMX: 0,
      lastMY: 0
    };

    function gaussian(w1, w2, cx, cy, sx, sy) {
      var dx = w1 - cx;
      var dy = w2 - cy;
      return Math.exp(-0.5 * (dx * dx / sx + dy * dy / sy));
    }

    function loss(w1, w2) {
      var x = w1 - 0.8;
      var y = w2 + 0.5;
      var basin = 0.11 * x * x + 0.085 * y * y;
      var ridge1 = 1.35 * gaussian(w1, w2, -1.45, 1.65, 0.38, 0.55);
      var ridge2 = 0.95 * gaussian(w1, w2, 2.85, 1.15, 0.75, 0.42);
      var valley1 = -1.05 * gaussian(w1, w2, 1.8, -1.55, 0.95, 0.32);
      var valley2 = -0.75 * gaussian(w1, w2, -0.35, -1.85, 0.45, 0.75);
      var saddle = 0.09 * (w1 + 0.35) * (w2 - 0.25);
      var ripple =
        0.38 * Math.sin(2.25 * w1 + 0.35) * Math.cos(1.9 * w2 - 0.4) +
        0.22 * Math.sin(3.2 * w1 - 1.15 * w2 + 0.7) +
        0.14 * Math.cos(4.1 * w2 + 0.55);
      return 2.2 + basin + ridge1 + ridge2 + valley1 + valley2 + saddle + ripple;
    }

    function grad(w1, w2) {
      var h = 0.01;
      return {
        g1: (loss(w1 + h, w2) - loss(w1 - h, w2)) / (2 * h),
        g2: (loss(w1, w2 + h) - loss(w1, w2 - h)) / (2 * h)
      };
    }

    function surfaceRoughness() {
      return Math.pow((64 - state.batch) / 63, 0.8);
    }

    function surfaceNoise(w1, w2) {
      return (
        0.46 * Math.sin(5.2 * w1 + 1.1) * Math.sin(4.9 * w2 - 0.6) +
        0.24 * Math.sin(8.5 * w1 - 3.6 * w2 + 0.4) +
        0.18 * Math.cos(6.3 * w1 + 5.1 * w2 - 1.7)
      );
    }

    function displayedLoss(w1, w2) {
      return loss(w1, w2) + surfaceRoughness() * surfaceNoise(w1, w2);
    }

    function displayedGrad(w1, w2) {
      var h = 0.01;
      return {
        g1: (displayedLoss(w1 + h, w2) - displayedLoss(w1 - h, w2)) / (2 * h),
        g2: (displayedLoss(w1, w2 + h) - displayedLoss(w1, w2 - h)) / (2 * h)
      };
    }

    function noise(seed) {
      var x = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
      return (x - Math.floor(x)) * 2 - 1;
    }

    function batchGrad(w1, w2, batch, step) {
      var g = grad(w1, w2);
      var n1 = 0;
      var n2 = 0;
      for (var i = 0; i < batch; i++) {
        n1 += noise(step * 101 + i * 17 + 1);
        n2 += noise(step * 137 + i * 19 + 2);
      }
      var scale = 2.4;
      return {
        g1: g.g1 + scale * n1 / batch,
        g2: g.g2 + scale * n2 / batch,
        trueG1: g.g1,
        trueG2: g.g2
      };
    }

    function clampWeight(value) {
      return Math.max(WMIN + 0.05, Math.min(WMAX - 0.05, value));
    }

    function vectorText(a, b, digits) {
      return "[" + a.toFixed(digits) + ", " + b.toFixed(digits) + "]";
    }

    function randomStart() {
      for (var tries = 0; tries < 60; tries++) {
        var w1 = WMIN + 0.45 + Math.random() * (WMAX - WMIN - 0.9);
        var w2 = WMIN + 0.45 + Math.random() * (WMAX - WMIN - 0.9);
        var g = grad(w1, w2);
        var slope = Math.hypot(g.g1, g.g2);
        if (slope > 0.35 && loss(w1, w2) < 6.2) {
          return { w1: w1, w2: w2 };
        }
      }
      return { w1: -1.0, w2: 2.0 };
    }

    function controlHtml() {
      var html =
        '<div class="mlp-slider"><label>&eta;</label><input data-k="eta" type="range" min="' +
        (isAdam ? "0.05" : "0.1") + '" max="1.5" step="' + (isAdam ? "0.05" : "0.1") +
        '" value="' + defaults.eta + '"><p data-r="eta">' + (isAdam ? defaults.eta.toFixed(2) : defaults.eta.toFixed(1)) + '</p></div>' +
        '<div class="mlp-slider"><label>Batch</label><input data-k="batch" type="range" min="1" max="64" step="1" value="' +
        defaults.batch + '"><p data-r="batch">' + defaults.batch + '</p></div>';
      if (isAdam) {
        html +=
          '<div class="mlp-slider"><label>&beta;&#8321;</label><input data-k="beta1" type="range" min="0.5" max="0.99" step="0.01" value="' +
          defaults.beta1 + '"><p data-r="beta1">' + defaults.beta1.toFixed(2) + '</p></div>' +
          '<div class="mlp-slider"><label>&beta;&#8322;</label><input data-k="beta2" type="range" min="0.9" max="0.999" step="0.001" value="' +
          defaults.beta2 + '"><p data-r="beta2">' + defaults.beta2.toFixed(3) + '</p></div>';
      }
      return html;
    }

    host.innerHTML =
      '<div class="land-widget module2-loss-widget ' + (isAdam ? "adam-landscape-widget" : "loss-landscape-widget") + '">' +
        '<div class="land-content">' +
          '<div class="land-main">' +
            '<canvas class="land-canvas"></canvas>' +
          '</div>' +
          '<div class="land-param-diagram">' +
            '<p class="land-diagram-title">Two trainable weights</p>' +
            '<div class="land-weight-row"><span>w&#8321;</span><strong data-r="w1">0.00</strong></div>' +
            '<div class="land-weight-row"><span>w&#8322;</span><strong data-r="w2">0.00</strong></div>' +
            '<div class="land-loss-box">L(w&#8321;, w&#8322;)</div>' +
            '<div class="land-slope-card">' +
              '<p class="land-slope-label">Local tangent line</p>' +
              '<p class="land-slope-value" data-r="slope">|&nabla;L| = 0.000</p>' +
            '</div>' +
          '</div>' +
        '</div>' +
        '<div class="land-controls">' +
          controlHtml() +
          '<button class="land-step">Step</button>' +
          '<button class="land-reset-btn">Reset</button>' +
        '</div>' +
        '<div class="land-readout"></div>' +
      '</div>';

    var canvas = host.querySelector(".land-canvas");
    var readout = host.querySelector(".land-readout");

    ["pointerdown", "keydown"].forEach(function(ev) {
      host.addEventListener(ev, function(e) { e.stopPropagation(); });
    });

    canvas.addEventListener("pointerdown", function(e) {
      state.dragging = true;
      state.lastMX = e.clientX;
      state.lastMY = e.clientY;
      state.dragType = e.shiftKey ? "pan" : "rotate";
      canvas.setPointerCapture(e.pointerId);
    });

    canvas.addEventListener("pointermove", function(e) {
      if (!state.dragging) return;
      var dx = e.clientX - state.lastMX;
      var dy = e.clientY - state.lastMY;
      if (state.dragType === "pan") {
        state.panX += dx;
        state.panY += dy;
      } else {
        state.az -= dx * 0.008;
        state.el = Math.max(-0.8, Math.min(1.45, state.el - dy * 0.01));
      }
      state.lastMX = e.clientX;
      state.lastMY = e.clientY;
      draw();
    });

    ["pointerup", "pointercancel", "pointerleave"].forEach(function(ev) {
      canvas.addEventListener(ev, function() {
        state.dragging = false;
        state.dragType = null;
      });
    });

    canvas.addEventListener("wheel", function(e) {
      e.preventDefault();
      state.zoom = Math.max(0.4, Math.min(2.5, state.zoom * (e.deltaY > 0 ? 0.92 : 1.08)));
      draw();
    }, { passive: false });

    function formatControl(key) {
      if (key === "batch") return String(state.batch);
      if (key === "eta" && !isAdam) return state.eta.toFixed(1);
      if (key === "beta2") return state.beta2.toFixed(3);
      return state[key].toFixed(2);
    }

    function syncReadouts() {
      host.querySelector('[data-r="w1"]').textContent = state.w1.toFixed(2);
      host.querySelector('[data-r="w2"]').textContent = state.w2.toFixed(2);
      ["eta", "batch", "beta1", "beta2"].forEach(function(key) {
        var read = host.querySelector('[data-r="' + key + '"]');
        var input = host.querySelector('[data-k="' + key + '"]');
        if (!read || !input) return;
        input.value = state[key];
        read.textContent = formatControl(key);
      });
    }

    host.querySelectorAll("input").forEach(function(inp) {
      inp.addEventListener("input", function() {
        state[inp.dataset.k] = inp.dataset.k === "batch" ? parseInt(inp.value, 10) : parseFloat(inp.value);
        state.lastStep = null;
        syncReadouts();
        draw();
      });
    });

    function resetState() {
      var start = randomStart();
      state.w1 = start.w1;
      state.w2 = start.w2;
      state.eta = defaults.eta;
      state.batch = defaults.batch;
      state.beta1 = isAdam ? defaults.beta1 : state.beta1;
      state.beta2 = isAdam ? defaults.beta2 : state.beta2;
      state.m1 = 0;
      state.m2 = 0;
      state.v1 = 0;
      state.v2 = 0;
      state.t = 0;
      state.lastStep = null;
      state.az = INIT_AZ;
      state.el = INIT_EL;
      state.zoom = INIT_ZOOM;
      state.panX = 0;
      state.panY = 0;
      syncReadouts();
      draw();
    }

    host.querySelector(".land-step").addEventListener("click", function() {
      if (isAdam) {
        var ag = batchGrad(state.w1, state.w2, state.batch, state.t + 1);
        var aOldW1 = state.w1;
        var aOldW2 = state.w2;
        var aOldL = loss(aOldW1, aOldW2);
        state.t++;
        state.m1 = state.beta1 * state.m1 + (1 - state.beta1) * ag.g1;
        state.m2 = state.beta1 * state.m2 + (1 - state.beta1) * ag.g2;
        state.v1 = state.beta2 * state.v1 + (1 - state.beta2) * ag.g1 * ag.g1;
        state.v2 = state.beta2 * state.v2 + (1 - state.beta2) * ag.g2 * ag.g2;
        var m1h = state.m1 / (1 - Math.pow(state.beta1, state.t));
        var m2h = state.m2 / (1 - Math.pow(state.beta1, state.t));
        var v1h = state.v1 / (1 - Math.pow(state.beta2, state.t));
        var v2h = state.v2 / (1 - Math.pow(state.beta2, state.t));
        var step1 = state.eta * m1h / (Math.sqrt(v1h) + EPS);
        var step2 = state.eta * m2h / (Math.sqrt(v2h) + EPS);
        state.w1 = clampWeight(state.w1 - step1);
        state.w2 = clampWeight(state.w2 - step2);
        state.lastStep = {
          kind: "adam",
          oldW1: aOldW1,
          oldW2: aOldW2,
          newW1: state.w1,
          newW2: state.w2,
          oldLoss: aOldL,
          newLoss: loss(state.w1, state.w2),
          eta: state.eta,
          batch: state.batch,
          m1h: m1h,
          m2h: m2h,
          v1h: v1h,
          v2h: v2h,
          step1: step1,
          step2: step2
        };
      } else {
        var g = batchGrad(state.w1, state.w2, state.batch, state.t + 1);
        var oldW1 = state.w1;
        var oldW2 = state.w2;
        var oldL = loss(oldW1, oldW2);
        state.w1 = clampWeight(state.w1 - state.eta * g.g1);
        state.w2 = clampWeight(state.w2 - state.eta * g.g2);
        state.t++;
        state.lastStep = {
          kind: "sgd",
          oldW1: oldW1,
          oldW2: oldW2,
          newW1: state.w1,
          newW2: state.w2,
          oldLoss: oldL,
          newLoss: loss(state.w1, state.w2),
          g1: g.g1,
          g2: g.g2,
          eta: state.eta,
          batch: state.batch
        };
      }
      syncReadouts();
      draw();
    });

    host.querySelector(".land-reset-btn").addEventListener("click", resetState);

    function fitCanvas(c) {
      var dpr = window.devicePixelRatio || 1;
      var w = c.clientWidth;
      var h = c.clientHeight;
      if (!w || !h) return null;
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
      var ctx = c.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx: ctx, w: w, h: h };
    }

    function heightColor(z, zmin, zmax, alpha) {
      var t = Math.max(0, Math.min(1, (z - zmin) / (zmax - zmin || 1)));
      var bands = [
        [38, 92, 255],
        [0, 214, 255],
        [38, 217, 87],
        [248, 227, 64],
        [255, 159, 28],
        [255, 59, 48]
      ];
      var idx = Math.min(bands.length - 1, Math.floor(t * bands.length));
      var c = bands[idx];
      return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + alpha + ")";
    }

    function project3D(x, y, z, W, H) {
      var cx = (WMIN + WMAX) / 2;
      var cy = (WMIN + WMAX) / 2;
      var dx = x - cx;
      var dy = y - cy;
      var ca = Math.cos(state.az);
      var sa = Math.sin(state.az);
      var ce = Math.cos(state.el);
      var se = Math.sin(state.el);
      var xr = dx * ca - dy * sa;
      var yr = dx * sa + dy * ca;
      var zr = -z * 0.34;
      var yr2 = yr * ce - zr * se;
      var zr2 = yr * se + zr * ce;
      var scale = Math.min(W, H) * 0.16 * state.zoom;
      return {
        x: W * 0.5 + state.panX + xr * scale,
        y: H * 0.58 + state.panY - yr2 * scale * 0.72,
        depth: zr2
      };
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
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      var lx = Math.max(22, Math.min(W - 22, b.x));
      var ly = Math.max(18, Math.min(H - 14, b.y - 6));
      ctx.fillText(label, lx, ly);
    }

    function drawAxes(ctx, W, H, zmin, zmax) {
      var base = [WMIN, WMIN, zmin];
      drawAxis(ctx, base, [WMAX, WMIN, zmin], "w\u2081", PRIMARY, W, H);
      drawAxis(ctx, base, [WMIN, WMAX, zmin], "w\u2082", SECONDARY, W, H);
      drawAxis(ctx, base, [WMIN, WMIN, zmax], "loss", TEXT, W, H);
    }

    function drawTangentLine(ctx, W, H, curL, localG) {
      var slope = Math.hypot(localG.g1, localG.g2);
      var ux = slope > 0.0001 ? localG.g1 / slope : 1;
      var uy = slope > 0.0001 ? localG.g2 / slope : 0;
      var len = 1.05;
      var low = project3D(state.w1 - ux * len, state.w2 - uy * len, curL - slope * len, W, H);
      var high = project3D(state.w1 + ux * len, state.w2 + uy * len, curL + slope * len, W, H);

      ctx.save();
      ctx.shadowColor = TANGENT;
      ctx.shadowBlur = 9;
      ctx.strokeStyle = TANGENT;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(low.x, low.y);
      ctx.lineTo(high.x, high.y);
      ctx.stroke();
      ctx.restore();
    }

    function drawSurface(ctx, W, H) {
      var grid = 36;
      var pts = [];
      var zmin = Infinity;
      var zmax = -Infinity;
      for (var i = 0; i <= grid; i++) {
        pts[i] = [];
        for (var j = 0; j <= grid; j++) {
          var w1 = WMIN + (WMAX - WMIN) * i / grid;
          var w2 = WMIN + (WMAX - WMIN) * j / grid;
          var z = displayedLoss(w1, w2);
          zmin = Math.min(zmin, z);
          zmax = Math.max(zmax, z);
          pts[i][j] = { w1: w1, w2: w2, z: z };
        }
      }

      for (var pi = 0; pi <= grid; pi++) {
        for (var pj = 0; pj <= grid; pj++) {
          var p = pts[pi][pj];
          var projected = project3D(p.w1, p.w2, p.z, W, H);
          p.x = projected.x;
          p.y = projected.y;
          p.depth = projected.depth;
        }
      }

      var tris = [];
      for (var ti = 0; ti < grid; ti++) {
        for (var tj = 0; tj < grid; tj++) {
          var a = pts[ti][tj];
          var b = pts[ti + 1][tj];
          var c = pts[ti][tj + 1];
          var d = pts[ti + 1][tj + 1];
          tris.push({ p: [a, b, c], z: (a.z + b.z + c.z) / 3, d: (a.depth + b.depth + c.depth) / 3 });
          tris.push({ p: [b, d, c], z: (b.z + c.z + d.z) / 3, d: (b.depth + c.depth + d.depth) / 3 });
        }
      }
      tris.sort(function(a, b) { return b.d - a.d; });

      tris.forEach(function(t) {
        ctx.beginPath();
        ctx.moveTo(t.p[0].x, t.p[0].y);
        ctx.lineTo(t.p[1].x, t.p[1].y);
        ctx.lineTo(t.p[2].x, t.p[2].y);
        ctx.closePath();
        ctx.fillStyle = heightColor(t.z, zmin, zmax, 0.94);
        ctx.fill();
      });

      ctx.lineWidth = 0.55;
      for (var li = 0; li <= grid; li++) {
        for (var lj = 0; lj < grid; lj++) {
          var p1 = pts[li][lj];
          var p2 = pts[li][lj + 1];
          ctx.strokeStyle = heightColor((p1.z + p2.z) / 2, zmin, zmax, 0.36);
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }
      for (var lj2 = 0; lj2 <= grid; lj2++) {
        for (var li2 = 0; li2 < grid; li2++) {
          var p3 = pts[li2][lj2];
          var p4 = pts[li2 + 1][lj2];
          ctx.strokeStyle = heightColor((p3.z + p4.z) / 2, zmin, zmax, 0.36);
          ctx.beginPath();
          ctx.moveTo(p3.x, p3.y);
          ctx.lineTo(p4.x, p4.y);
          ctx.stroke();
        }
      }
      return { zmin: zmin, zmax: zmax };
    }

    function updateEquation(curL, trueG) {
      var slope = Math.hypot(trueG.g1, trueG.g2);
      host.querySelector('[data-r="slope"]').innerHTML = "m = " + slope.toFixed(3);

      if (isAdam) {
        updateAdamEquation(curL, slope);
      } else {
        updateSgdEquation(curL, slope);
      }
    }

    function updateSgdEquation(curL, slope) {
      var g = batchGrad(state.w1, state.w2, state.batch, state.t + 1);
      var data = {
        oldW1: state.w1,
        oldW2: state.w2,
        newW1: clampWeight(state.w1 - state.eta * g.g1),
        newW2: clampWeight(state.w2 - state.eta * g.g2),
        g1: g.g1,
        g2: g.g2,
        eta: state.eta,
        batch: state.batch
      };

      readout.innerHTML =
        '<p class="land-equation-line land-equation-symbols">' +
        '<span class="land-equation-label">Update rule</span> ' +
        '<span class="eq-new">w<sub>new</sub></span> = ' +
        '<span class="eq-old">w<sub>old</sub></span> + ' +
        '(<span class="eq-eta">&minus;&eta;</span> &middot; ' +
        '<span class="eq-grad">&nabla;<sub>B</sub>L(w<sub>old</sub>)</span>)</p>' +
        '<p class="land-equation-line land-equation-numbers">' +
        '<span class="land-equation-label">Numbers</span> ' +
        '<span class="eq-new">' + vectorText(data.newW1, data.newW2, 2) + '</span> = ' +
        '<span class="eq-old">' + vectorText(data.oldW1, data.oldW2, 2) + '</span> + ' +
        '(<span class="eq-eta">&minus;' + data.eta.toFixed(1) + '</span> &middot; ' +
        '<span class="eq-grad">' + vectorText(data.g1, data.g2, 3) + '</span>)</p>' +
        '<p class="land-metric-line">Local tangent line: ' +
        '<span class="eq-tangent">slope m = ' + slope.toFixed(3) + '</span> | ' +
        'batch surface roughness <strong>' + surfaceRoughness().toFixed(2) + '</strong> | ' +
        'displayed loss <strong>' + curL.toFixed(3) + '</strong></p>';
    }

    function previewAdamStep() {
      var g = batchGrad(state.w1, state.w2, state.batch, state.t + 1);
      var nextT = state.t + 1;
      var nextM1 = state.beta1 * state.m1 + (1 - state.beta1) * g.g1;
      var nextM2 = state.beta1 * state.m2 + (1 - state.beta1) * g.g2;
      var nextV1 = state.beta2 * state.v1 + (1 - state.beta2) * g.g1 * g.g1;
      var nextV2 = state.beta2 * state.v2 + (1 - state.beta2) * g.g2 * g.g2;
      var m1h = nextM1 / (1 - Math.pow(state.beta1, nextT));
      var m2h = nextM2 / (1 - Math.pow(state.beta1, nextT));
      var v1h = nextV1 / (1 - Math.pow(state.beta2, nextT));
      var v2h = nextV2 / (1 - Math.pow(state.beta2, nextT));
      var step1 = state.eta * m1h / (Math.sqrt(v1h) + EPS);
      var step2 = state.eta * m2h / (Math.sqrt(v2h) + EPS);
      return {
        oldW1: state.w1,
        oldW2: state.w2,
        newW1: clampWeight(state.w1 - step1),
        newW2: clampWeight(state.w2 - step2),
        eta: state.eta,
        m1h: m1h,
        m2h: m2h,
        v1h: v1h,
        v2h: v2h,
        step1: step1,
        step2: step2,
        preview: true
      };
    }

    function updateAdamEquation(curL, slope) {
      var data = previewAdamStep();
      readout.innerHTML =
        '<p class="land-equation-line land-equation-symbols">' +
        '<span class="land-equation-label">Adam rule</span> ' +
        '<span class="eq-new">w<sub>new</sub></span> = ' +
        '<span class="eq-old">w<sub>old</sub></span> &minus; ' +
        '<span class="eq-eta">&eta;</span> &middot; ' +
        '<span class="eq-grad">m&#770;</span> / ' +
        '(<span class="eq-vterm">&radic;v&#770;</span> + &epsilon;)</p>' +
        '<p class="land-equation-line land-equation-numbers">' +
        '<span class="land-equation-label">Numbers</span> ' +
        '<span class="eq-new">' + vectorText(data.newW1, data.newW2, 2) + '</span> = ' +
        '<span class="eq-old">' + vectorText(data.oldW1, data.oldW2, 2) + '</span> &minus; ' +
        '<span class="eq-eta">' + data.eta.toFixed(2) + '</span> &middot; ' +
        '<span class="eq-grad">' + vectorText(data.m1h, data.m2h, 3) + '</span> / ' +
        '(<span class="eq-vterm">&radic;' + vectorText(data.v1h, data.v2h, 3) + '</span> + &epsilon;)</p>' +
        '<p class="land-metric-line">Local tangent line: ' +
        '<span class="eq-tangent">slope m = ' + slope.toFixed(3) + '</span> | ' +
        'batch surface roughness <strong>' + surfaceRoughness().toFixed(2) + '</strong> | ' +
        'displayed loss <strong>' + curL.toFixed(3) + '</strong></p>';
    }

    function draw() {
      var f = fitCanvas(canvas);
      if (!f) return;
      var ctx = f.ctx;
      var W = f.w;
      var H = f.h;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#080d1b";
      ctx.fillRect(0, 0, W, H);

      var range = drawSurface(ctx, W, H);
      drawAxes(ctx, W, H, range.zmin, range.zmax);

      var curL = displayedLoss(state.w1, state.w2);
      var localG = displayedGrad(state.w1, state.w2);
      drawTangentLine(ctx, W, H, curL, localG);

      var curP = project3D(state.w1, state.w2, curL, W, H);
      ctx.shadowColor = PRIMARY;
      ctx.shadowBlur = 14;
      ctx.beginPath();
      ctx.arc(curP.x, curP.y, 8, 0, 6.2832);
      ctx.fillStyle = PRIMARY;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = TEXT;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      updateEquation(curL, localG);
    }

    resetState();
    return { resize: draw };
  }

  function installModule2LossWidgets() {
    INTERACTIVE_WIDGETS.lossLandscape = function(host) {
      return createLossSurfaceWidget(host, "sgd");
    };
    INTERACTIVE_WIDGETS.adamLandscape = function(host) {
      return createLossSurfaceWidget(host, "adam");
    };

    document.querySelectorAll('.interactive-host[data-widget="lossLandscape"], .interactive-host[data-widget="adamLandscape"]').forEach(function(host) {
      if (!host.dataset.hydrated) return;
      host.dataset.hydrated = "";
      host._widget = null;
      var factory = INTERACTIVE_WIDGETS[host.getAttribute("data-widget")];
      host.dataset.hydrated = "1";
      host._widget = factory(host) || null;
      if (host._widget && typeof host._widget.resize === "function") host._widget.resize();
    });
  }

  window.MODULE_CONFIG = {
    title: "LLMs 0 to 100 - Module 2",
    // Manim step-through clips for this module, with files under media/sections.
    // Folding, the MLP decision boundary, and loss landscapes are interactive widgets.
    manimSections: {
      "overfit-viz": [
        "OverfittingCurveScene_0000_overfit_axes.mp4",
        "OverfittingCurveScene_0001_overfit_train.mp4",
        "OverfittingCurveScene_0002_overfit_val.mp4",
        "OverfittingCurveScene_0003_overfit_region.mp4",
        "OverfittingCurveScene_0004_fits.mp4"
      ]
    },
    widgets: {
      lossLandscape: function(host) {
        return createLossSurfaceWidget(host, "sgd");
      },
      adamLandscape: function(host) {
        return createLossSurfaceWidget(host, "adam");
      }
    },
    onReady: installModule2LossWidgets
  };
}());
