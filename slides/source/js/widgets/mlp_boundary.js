// =====================================================================
// MLP boundary widget — INTERACTIVE_WIDGETS.mlpBoundary
// Deterministic lookup of MLP boundary presets.
// "Many lines make a curve": each hidden neuron contributes one linear
// piece to the decision boundary. Width × depth = total pieces.
// =====================================================================
INTERACTIVE_WIDGETS.mlpBoundary = function(host) {
  var GREEN = '#3fb950', RED = '#e74c3c', MUTED = '#8892a4',
      LINEC = '#2a3450', TEXT = '#e8eaf0', PRIMARY = '#4a9eff', SECONDARY = '#f5a623';
  var state = { dataset: 'spiral', width: 8, depth: 2, net: null, acc: 0, view: -1, hover: null, pinned: null, nodePos: [] };
  var NEURON_COLORS = ['#4a9eff', '#f5a623', '#bd93f9', '#ff79c6', '#8be9fd', '#50fa7b', '#ffb86c', '#f1fa8c'];
  var clusterData = null, clusterCache = {};
  var GRID = 64;
  // Network + training live in the shared MLP module (defined above).
  var generateData = MLP.generateData, train = MLP.train,
      predictOne = MLP.predictOne, computeAccuracy = MLP.computeAccuracy;

  host.innerHTML =
    '<div class="mlp-widget">' +
      '<div class="mlp-canvas-wrap"><canvas class="mlp-canvas"></canvas></div>' +
      '<div class="mlp-controls">' +
        '<div class="expl-datasets"></div>' +
        '<div class="expl-datasets expl-views"></div>' +
        '<div class="mlp-slider"><label>Width</label><input type="range" min="1" max="8" step="1" value="8"><p data-readout="width">8</p></div>' +
        '<div class="mlp-slider"><label>Depth</label><input type="range" min="1" max="4" step="1" value="2"><p data-readout="depth">2</p></div>' +
        '<p class="mlp-readout"></p>' +
      '</div>' +
    '</div>';

  var canvas = host.querySelector('.mlp-canvas');
  var widthInput = host.querySelector('input[data-readout="width"]') || host.querySelector('.mlp-controls input');
  var depthInput = host.querySelectorAll('.mlp-controls input')[1];
  var widthText = host.querySelector('[data-readout="width"]');
  var depthText = host.querySelector('[data-readout="depth"]');
  var readout = host.querySelector('.mlp-readout');
  var dsEl = host.querySelector('.expl-datasets');
  var viewsEl = host.querySelector('.expl-views');

  ['pointerdown', 'keydown'].forEach(function(ev) {
    host.addEventListener(ev, function(e) { e.stopPropagation(); });
  });

  function nodeAt(e) {
    if (state.dataset !== 'clusters') return null;
    var r = canvas.getBoundingClientRect();
    var mx = (e.clientX - r.left) * (canvas.clientWidth / r.width);
    var my = (e.clientY - r.top) * (canvas.clientHeight / r.height);
    var best = null, bestD = 16;
    state.nodePos.forEach(function(n) {
      var d = Math.hypot(n.x - mx, n.y - my);
      if (d < bestD) { bestD = d; best = n; }
    });
    return best;
  }
  function sameNode(a, b) { return (!a && !b) || (a && b && a.layer === b.layer && a.idx === b.idx); }
  canvas.addEventListener('pointermove', function(e) {
    var n = nodeAt(e);
    canvas.style.cursor = n ? 'pointer' : 'default';
    if (sameNode(n, state.hover)) return;
    state.hover = n;
    var active = state.pinned || state.hover;
    if (active && active.layer !== state.view) { state.view = active.layer; }
    updateUI(); draw();
  });
  canvas.addEventListener('pointerleave', function() {
    if (!state.hover) return;
    state.hover = null; updateUI(); draw();
  });
  canvas.addEventListener('click', function(e) {
    var n = nodeAt(e);
    if (!n) { if (state.pinned) { state.pinned = null; updateUI(); draw(); } return; }
    state.pinned = sameNode(n, state.pinned) ? null : n;
    if (state.pinned) state.view = state.pinned.layer;
    updateUI(); draw();
  });

  var datasets = ['moons', 'spiral', 'clusters'];
  function applyDatasetPreset(name) {
    state.dataset = name;
    if (name === 'moons') { state.width = 6; state.depth = 1; }
    if (name === 'spiral') { state.width = 8; state.depth = 2; }
    if (name === 'clusters') { state.width = 6; state.depth = 2; }
    state.view = -1; state.hover = null; state.pinned = null;
    widthInput.value = state.width; depthInput.value = state.depth;
    widthText.textContent = String(state.width); depthText.textContent = String(state.depth);
    state.net = null;
  }
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

  // Find the actual slider inputs by their labels
  var allInputs = host.querySelectorAll('.mlp-slider input');
  widthInput = allInputs[0];
  depthInput = allInputs[1];

  widthInput.addEventListener('input', function() {
    state.width = +widthInput.value; widthText.textContent = widthInput.value; state.net = null;
    if (state.view >= state.depth) state.view = -1;
    updateUI(); draw();
  });
  depthInput.addEventListener('input', function() {
    state.depth = +depthInput.value; depthText.textContent = depthInput.value; state.net = null;
    if (state.view >= state.depth) state.view = -1;
    updateUI(); draw();
  });

  function updateUI() {
    host.querySelectorAll('.expl-datasets:not(.expl-views) .expl-ds-btn').forEach(function(b) {
      b.classList.toggle('active', b.dataset.ds === state.dataset);
    });
    // View buttons only make sense on the clusters tab, where a real net is trained.
    viewsEl.innerHTML = '';
    viewsEl.style.minHeight = '34px';
    viewsEl.style.visibility = state.dataset === 'clusters' ? 'visible' : 'hidden';
    if (state.dataset === 'clusters') {
      for (var v = 0; v <= state.depth; v++) {
        (function(v) {
          var isOut = v === state.depth;
          var btn = document.createElement('button');
          btn.className = 'expl-ds-btn' + ((isOut ? -1 : v) === state.view ? ' active' : '');
          btn.textContent = isOut ? 'Boundary' : 'Layer ' + (v + 1) + ' neurons';
          btn.addEventListener('click', function() { state.view = isOut ? -1 : v; state.pinned = null; updateUI(); draw(); });
          viewsEl.appendChild(btn);
        })(v);
      }
    }
    var totalUnits = state.width * state.depth;
    readout.innerHTML = state.depth + ' hidden layer' + (state.depth === 1 ? '' : 's') +
      ' &times; ' + state.width + ' neuron' + (state.width === 1 ? '' : 's') +
      ' = <strong>' + totalUnits + '</strong> hidden unit' + (totalUnits === 1 ? '' : 's') +
      ' | Accuracy: <strong>' + (state.acc * 100).toFixed(0) + '%</strong>';
    if (state.dataset === 'clusters' && state.view === 0) {
      readout.innerHTML += '<br>Layer 1: ' + state.width + ' straight lines. Fencing one cluster takes at least 3.';
    } else if (state.dataset === 'clusters' && state.view > 0) {
      readout.innerHTML += '<br>Layer ' + (state.view + 1) + ': ' + state.width + ' new piecewise-linear curves, bent only where layer ' + state.view + ' curves cross.';
    } else if (state.dataset === 'clusters') {
      readout.innerHTML += '<br>Boundary: one curve assembled from the last layer. It must enclose each cluster.';
    }
    var active = state.pinned || state.hover;
    if (state.dataset === 'clusters' && active) {
      readout.innerHTML += '<br><span style="color:' + NEURON_COLORS[active.idx % NEURON_COLORS.length] + '">Layer ' + (active.layer + 1) + ', neuron ' + (active.idx + 1) + '</span>: ' +
        (active.layer === 0 ? 'one straight line.' : 'straight pieces joined at the dashed layer ' + active.layer + ' curves, where a ReLU switches on or off.') +
        (state.pinned ? ' Click again to release.' : ' Click to pin.');
    } else if (state.dataset === 'clusters' && state.view >= 0) {
      readout.innerHTML += '<br>Hover a neuron in the diagram to highlight its line.';
    }
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

  function drawModelBoundaryLines(ctx, model, ox, oy, side) {
    if (!model || !model.lines || !model.lines.length) return;
    ctx.save();
    ctx.beginPath(); ctx.rect(ox, oy, side, side); ctx.clip();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    function strokeSegments(width, color) {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      model.lines.forEach(function(line) {
        ctx.moveTo(ox + line.x1 * side, oy + (1 - line.y1) * side);
        ctx.lineTo(ox + line.x2 * side, oy + (1 - line.y2) * side);
      });
      ctx.stroke();
    }
    strokeSegments(6.0, 'rgba(8,12,24,0.94)');
    strokeSegments(3.2, SECONDARY);
    ctx.restore();
  }

  function draw() {
    var f = fitCanvas(canvas);
    if (!f) return;
    var ctx = f.ctx, W = f.w, H = f.h;
    ctx.clearRect(0, 0, W, H);

    if (state.dataset === 'clusters') { drawClusters(ctx, W, H); return; }
    var data = generateData(state.dataset);
    if (!state.net) {
      state.net = train(data, state.width, state.depth, state.dataset);
      state.acc = computeAccuracy(data, state.net);
      updateUI();
    }

    var pad = 18;
    var rightPanel = 140;
    var side = Math.min(W - 2 * pad - rightPanel, H - 2 * pad);
    if (side < 120) { rightPanel = 0; side = Math.min(W - 2 * pad, H - 2 * pad); }
    var ox = pad + (W - 2 * pad - rightPanel - side) / 2;
    var oy = pad + (H - 2 * pad - side) / 2;

    // Decision grid background
    var cell = side / GRID;
    for (var i = 0; i < GRID; i++) {
      for (var j = 0; j < GRID; j++) {
        var gx = (i + 0.5) / GRID, gy = (j + 0.5) / GRID;
        var p = predictOne([gx, gy], state.net);
        ctx.fillStyle = p >= 0.5 ? 'rgba(63,185,80,0.16)' : 'rgba(231,76,60,0.10)';
        ctx.fillRect(ox + i * cell, oy + (GRID - 1 - j) * cell, cell + 0.5, cell + 0.5);
      }
    }

    drawModelBoundaryLines(ctx, state.net, ox, oy, side);

    // Data scatter
    data.forEach(function(p) {
      var px = ox + p.x * side, py = oy + (1 - p.y) * side;
      ctx.beginPath(); ctx.arc(px, py, 4.2, 0, 6.2832);
      ctx.fillStyle = p.cls ? GREEN : RED; ctx.fill();
    });

    // Frame
    ctx.strokeStyle = LINEC; ctx.lineWidth = 1;
    ctx.strokeRect(ox, oy, side, side);

    drawNetwork(ctx, W, H, ox, oy, side, pad, rightPanel, -2);
  }

  // highlight: hidden layer index to fill, -1 for the output, -2 for none
  function drawNetwork(ctx, W, H, ox, oy, side, pad, rightPanel, highlight) {
    if (rightPanel > 80) {
      var nx0 = ox + side + 20, nx1 = W - pad;
      var cols = state.depth + 2;
      var clusters = state.dataset === 'clusters';
      var perLayer = clusters ? state.width : Math.min(state.width, 6);
      var top = oy + 6, bot = oy + side - 6;
      state.nodePos = [];
      var active = state.pinned || state.hover;
      function colX(i) { return nx0 + (nx1 - nx0) * i / (cols - 1); }
      function ys(n) {
        if (n === 1) return [(top + bot) / 2];
        var a = []; for (var i = 0; i < n; i++) a.push(top + (bot - top) * i / (n - 1)); return a;
      }
      var layersDiag = [{ x: colX(0), ys: ys(2), label: ['x₁', 'x₂'], color: SECONDARY }];
      for (var d = 0; d < state.depth; d++) layersDiag.push({ x: colX(d + 1), ys: ys(perLayer), color: PRIMARY, hl: highlight === d });
      layersDiag.push({ x: colX(cols - 1), ys: ys(1), label: ['ŷ'], color: SECONDARY, hl: highlight === -1 });

      ctx.strokeStyle = 'rgba(74,158,255,0.28)'; ctx.lineWidth = 1;
      for (var L = 0; L < layersDiag.length - 1; L++) {
        var A = layersDiag[L], B = layersDiag[L + 1];
        A.ys.forEach(function(ay) {
          B.ys.forEach(function(by) {
            ctx.beginPath(); ctx.moveTo(A.x + 10, ay); ctx.lineTo(B.x - 10, by); ctx.stroke();
          });
        });
      }
      layersDiag.forEach(function(Lr, li) {
        var hiddenIdx = li - 1;
        Lr.ys.forEach(function(yy, idx) {
          var isHidden = clusters && hiddenIdx >= 0 && hiddenIdx < state.depth;
          var color = isHidden ? NEURON_COLORS[idx % NEURON_COLORS.length] : Lr.color;
          var isActive = isHidden && active && active.layer === hiddenIdx && active.idx === idx;
          if (isHidden) state.nodePos.push({ x: Lr.x, y: yy, layer: hiddenIdx, idx: idx });
          if (isActive) {
            ctx.beginPath(); ctx.arc(Lr.x, yy, 15, 0, 6.2832);
            ctx.strokeStyle = TEXT; ctx.lineWidth = 2; ctx.stroke();
          }
          ctx.beginPath(); ctx.arc(Lr.x, yy, 10, 0, 6.2832);
          ctx.fillStyle = (Lr.hl || isActive) ? color : '#0d1225'; ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.fill(); ctx.stroke();
          if (Lr.label) { ctx.fillStyle = TEXT; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.fillText(Lr.label[idx], Lr.x, yy + 3); }
        });
      });
    }
  }

  function sampleFields(net) {
    var n = GRID, pres = [], logit = [];
    for (var l = 0; l < net.layers.length; l++) {
      var per = [];
      for (var o = 0; o < net.layers[l].W.length; o++) per.push([]);
      pres.push(per);
    }
    for (var i = 0; i <= n; i++) {
      logit.push([]);
      pres.forEach(function(per) { per.forEach(function(fld) { fld.push([]); }); });
      for (var j = 0; j <= n; j++) {
        var f = CLUSTER_MLP.forward(net, i / n, j / n);
        logit[i].push(f.logit);
        pres.forEach(function(per, l) { per.forEach(function(fld, o) { fld[i].push(f.pres[l][o]); }); });
      }
    }
    return { pres: pres, logit: logit };
  }

  function strokeSegs(ctx, segs, ox, oy, side, color, width) {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = 'round';
    ctx.beginPath();
    segs.forEach(function(sg) {
      ctx.moveTo(ox + sg[0][0] * side, oy + (1 - sg[0][1]) * side);
      ctx.lineTo(ox + sg[1][0] * side, oy + (1 - sg[1][1]) * side);
    });
    ctx.stroke();
  }

  // Clusters tab: a real ReLU MLP, so every neuron's zero-line can be drawn.
  function drawClusters(ctx, W, H) {
    if (!clusterData) clusterData = CLUSTER_MLP.generateData();
    var key = state.width + 'x' + state.depth;
    if (!clusterCache[key]) {
      readout.textContent = 'Training…';
      clusterCache[key] = CLUSTER_MLP.train(clusterData, state.width, state.depth, { steps: 700, seeds: 5 });
    }
    var model = clusterCache[key];
    if (!state.net) { state.net = model.net; state.acc = model.acc; updateUI(); }
    var fields = sampleFields(model.net);

    var pad = 18, rightPanel = 140;
    var side = Math.min(W - 2 * pad - rightPanel, H - 2 * pad);
    if (side < 120) { rightPanel = 0; side = Math.min(W - 2 * pad, H - 2 * pad); }
    var ox = pad + (W - 2 * pad - rightPanel - side) / 2;
    var oy = pad + (H - 2 * pad - side) / 2;

    var cell = side / GRID;
    for (var i = 0; i < GRID; i++) {
      for (var j = 0; j < GRID; j++) {
        ctx.fillStyle = fields.logit[i][j] > 0 ? 'rgba(63,185,80,0.16)' : 'rgba(231,76,60,0.10)';
        ctx.fillRect(ox + i * cell, oy + (GRID - 1 - j) * cell, cell + 0.5, cell + 0.5);
      }
    }
    ctx.save();
    ctx.beginPath(); ctx.rect(ox, oy, side, side); ctx.clip();
    if (state.view < 0) {
      var segs = CLUSTER_MLP.contour(fields.logit, GRID);
      strokeSegs(ctx, segs, ox, oy, side, 'rgba(8,12,24,0.94)', 6);
      strokeSegs(ctx, segs, ox, oy, side, SECONDARY, 3.2);
    } else {
      var active = state.pinned || state.hover;
      if (state.view > 0) {
        ctx.setLineDash([4, 4]);
        fields.pres[state.view - 1].forEach(function(fld) {
          strokeSegs(ctx, CLUSTER_MLP.contour(fld, GRID), ox, oy, side, 'rgba(232,234,240,0.35)', 1.2);
        });
        ctx.setLineDash([]);
      }
      var focus = active && active.layer === state.view ? active.idx : -1;
      fields.pres[state.view].forEach(function(fld, k) {
        if (k === focus) return;
        ctx.globalAlpha = focus >= 0 ? 0.22 : 1;
        strokeSegs(ctx, CLUSTER_MLP.contour(fld, GRID), ox, oy, side, NEURON_COLORS[k % NEURON_COLORS.length], 2.2);
        ctx.globalAlpha = 1;
      });
      if (focus >= 0) {
        var segs2 = CLUSTER_MLP.contour(fields.pres[state.view][focus], GRID);
        strokeSegs(ctx, segs2, ox, oy, side, 'rgba(8,12,24,0.9)', 7);
        strokeSegs(ctx, segs2, ox, oy, side, NEURON_COLORS[focus % NEURON_COLORS.length], 3.6);
      }
    }
    ctx.restore();

    clusterData.forEach(function(p) {
      ctx.beginPath(); ctx.arc(ox + p.x * side, oy + (1 - p.y) * side, 4.2, 0, 6.2832);
      ctx.fillStyle = p.cls ? GREEN : RED; ctx.fill();
    });
    ctx.strokeStyle = LINEC; ctx.lineWidth = 1; ctx.strokeRect(ox, oy, side, side);
    drawNetwork(ctx, W, H, ox, oy, side, pad, rightPanel, state.view);
  }

  updateUI();
  return { resize: draw, nodes: function() { return state.nodePos; } };
};
