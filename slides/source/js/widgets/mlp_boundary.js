// =====================================================================
// MLP boundary widget — INTERACTIVE_WIDGETS.mlpBoundary
// Deterministic lookup of MLP boundary presets.
// "Many lines make a curve": each hidden neuron contributes one linear
// piece to the decision boundary. Width × depth = total pieces.
// =====================================================================
INTERACTIVE_WIDGETS.mlpBoundary = function(host) {
  var GREEN = '#3fb950', RED = '#e74c3c', MUTED = '#8892a4',
      LINEC = '#2a3450', TEXT = '#e8eaf0', PRIMARY = '#4a9eff', SECONDARY = '#f5a623';
  var state = { dataset: 'spiral', width: 8, depth: 2, net: null, acc: 0 };
  var GRID = 64;
  // Network + training live in the shared MLP module (defined above).
  var generateData = MLP.generateData, train = MLP.train,
      predictOne = MLP.predictOne, computeAccuracy = MLP.computeAccuracy;

  host.innerHTML =
    '<div class="mlp-widget">' +
      '<div class="mlp-canvas-wrap"><canvas class="mlp-canvas"></canvas></div>' +
      '<div class="mlp-controls">' +
        '<div class="expl-datasets"></div>' +
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

  ['pointerdown', 'keydown'].forEach(function(ev) {
    host.addEventListener(ev, function(e) { e.stopPropagation(); });
  });

  var datasets = ['moons', 'spiral'];
  function applyDatasetPreset(name) {
    state.dataset = name;
    if (name === 'moons') { state.width = 6; state.depth = 1; }
    if (name === 'spiral') { state.width = 8; state.depth = 2; }
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
    state.width = +widthInput.value; widthText.textContent = widthInput.value; state.net = null; draw();
  });
  depthInput.addEventListener('input', function() {
    state.depth = +depthInput.value; depthText.textContent = depthInput.value; state.net = null; draw();
  });

  function updateUI() {
    host.querySelectorAll('.expl-ds-btn').forEach(function(b) {
      b.classList.toggle('active', b.dataset.ds === state.dataset);
    });
    var totalUnits = state.width * state.depth;
    readout.innerHTML = state.depth + ' hidden layer' + (state.depth === 1 ? '' : 's') +
      ' &times; ' + state.width + ' neuron' + (state.width === 1 ? '' : 's') +
      ' = <strong>' + totalUnits + '</strong> hidden unit' + (totalUnits === 1 ? '' : 's') +
      ' | Accuracy: <strong>' + (state.acc * 100).toFixed(0) + '%</strong>';
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

    // Network diagram
    if (rightPanel > 80) {
      var nx0 = ox + side + 20, nx1 = W - pad;
      var cols = state.depth + 2;
      var perLayer = Math.min(state.width, 6);
      var top = oy + 6, bot = oy + side - 6;
      function colX(i) { return nx0 + (nx1 - nx0) * i / (cols - 1); }
      function ys(n) {
        if (n === 1) return [(top + bot) / 2];
        var a = []; for (var i = 0; i < n; i++) a.push(top + (bot - top) * i / (n - 1)); return a;
      }
      var layersDiag = [{ x: colX(0), ys: ys(2), label: ['x₁', 'x₂'], color: SECONDARY }];
      for (var d = 0; d < state.depth; d++) layersDiag.push({ x: colX(d + 1), ys: ys(perLayer), color: PRIMARY });
      layersDiag.push({ x: colX(cols - 1), ys: ys(1), label: ['ŷ'], color: SECONDARY });

      ctx.strokeStyle = 'rgba(74,158,255,0.28)'; ctx.lineWidth = 1;
      for (var L = 0; L < layersDiag.length - 1; L++) {
        var A = layersDiag[L], B = layersDiag[L + 1];
        A.ys.forEach(function(ay) {
          B.ys.forEach(function(by) {
            ctx.beginPath(); ctx.moveTo(A.x + 10, ay); ctx.lineTo(B.x - 10, by); ctx.stroke();
          });
        });
      }
      layersDiag.forEach(function(Lr) {
        Lr.ys.forEach(function(yy, idx) {
          ctx.beginPath(); ctx.arc(Lr.x, yy, 10, 0, 6.2832);
          ctx.fillStyle = '#0d1225'; ctx.strokeStyle = Lr.color; ctx.lineWidth = 1.8; ctx.fill(); ctx.stroke();
          if (Lr.label) { ctx.fillStyle = TEXT; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.fillText(Lr.label[idx], Lr.x, yy + 3); }
        });
      });
    }
  }

  updateUI();
  return { resize: draw };
};
