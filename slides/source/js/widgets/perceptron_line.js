// =====================================================================
// Perceptron Line widget — INTERACTIVE_WIDGETS.perceptronLine
// Plots the line w1*x1 + w2*x2 + b = 0 in the (x1, x2) plane and shades
// the side where sigma fires. Sliders for w1, w2, b. No data, no training.
// =====================================================================
INTERACTIVE_WIDGETS.perceptronLine = function(host) {
  var TEXT = '#e8eaf0', MUTED = '#8892a4', AXIS = '#2a3450',
      SECONDARY = '#f5a623', GREEN = '#3fb950';
  var R = 3; // plot range: x1, x2 in [-R, R]
  var state = { w1: 1.0, w2: 1.0, b: 0.0 };

  host.innerHTML =
    '<div class="pline-widget">' +
      '<canvas class="perc-canvas pline-canvas"></canvas>' +
      '<div class="perc-controls">' +
        '<div class="perc-sliders"></div>' +
        '<p class="perc-readout pline-readout"></p>' +
      '</div>' +
    '</div>';
  var canvas = host.querySelector('.pline-canvas');
  var readout = host.querySelector('.pline-readout');
  var slidersEl = host.querySelector('.perc-sliders');

  ['pointerdown', 'keydown'].forEach(function(ev) {
    host.addEventListener(ev, function(e) { e.stopPropagation(); });
  });

  [['w1', 'w₁'], ['w2', 'w₂'], ['b', 'b']].forEach(function(spec) {
    var key = spec[0];
    var row = document.createElement('div');
    row.className = 'perc-slider';
    var lab = document.createElement('label');
    lab.textContent = spec[1];
    var inp = document.createElement('input');
    inp.type = 'range'; inp.min = -3; inp.max = 3; inp.step = 0.1;
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

  function signed(v) {
    return (v < 0 ? ' − ' : ' + ') + Math.abs(v).toFixed(1);
  }

  function draw() {
    var dpr = window.devicePixelRatio || 1;
    var W = canvas.clientWidth, H = canvas.clientHeight;
    if (!W || !H) return;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    var pad = 10;
    var side = Math.min(W, H) - 2 * pad;
    var ox = (W - side) / 2, oy = (H - side) / 2;
    function px(x) { return ox + (x + R) / (2 * R) * side; }
    function py(y) { return oy + (R - y) / (2 * R) * side; }

    var a = state.w1, b = state.w2, c = state.b;

    // Shade: red everywhere, then green on the half-plane w·x + b >= 0.
    ctx.fillStyle = 'rgba(231,76,60,0.10)';
    ctx.fillRect(ox, oy, side, side);
    var n = Math.hypot(a, b);
    if (n > 1e-9) {
      var fx = -c * a / (n * n), fy = -c * b / (n * n); // closest point on line to origin
      ctx.save();
      ctx.beginPath(); ctx.rect(ox, oy, side, side); ctx.clip();
      ctx.translate(px(fx), py(fy));
      ctx.rotate(Math.atan2(-b, a)); // canvas y is flipped
      ctx.fillStyle = 'rgba(63,185,80,0.16)';
      ctx.fillRect(0, -4 * side, 4 * side, 8 * side);
      ctx.restore();
    } else if (c >= 0) {
      ctx.fillStyle = 'rgba(63,185,80,0.16)';
      ctx.fillRect(ox, oy, side, side);
    }

    // Axes
    ctx.strokeStyle = AXIS; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(px(-R), py(0)); ctx.lineTo(px(R), py(0)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(px(0), py(-R)); ctx.lineTo(px(0), py(R)); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '13px Inter, sans-serif';
    ctx.textAlign = 'right'; ctx.fillText('x₁', px(R) - 4, py(0) + 15);
    ctx.textAlign = 'right'; ctx.fillText('x₂', px(0) - 6, py(R) + 14); ctx.textAlign = 'left';

    // Decision line a*x + b*y + c = 0, clipped to the frame.
    var p0 = null, p1 = null;
    if (Math.abs(b) > 1e-9) {
      p0 = [px(-R), py(-(a * -R + c) / b)];
      p1 = [px(R), py(-(a * R + c) / b)];
    } else if (Math.abs(a) > 1e-9) {
      p0 = [px(-c / a), py(-R)];
      p1 = [px(-c / a), py(R)];
    }
    if (p0) {
      ctx.save();
      ctx.beginPath(); ctx.rect(ox, oy, side, side); ctx.clip();
      ctx.strokeStyle = SECONDARY; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.stroke();
      // Weight vector from the line's nearest point to the origin, pointing to the firing side.
      var L = 0.9;
      var tx = fx + a / n * L, ty = fy + b / n * L;
      ctx.strokeStyle = TEXT; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(px(fx), py(fy)); ctx.lineTo(px(tx), py(ty)); ctx.stroke();
      var ang = Math.atan2(py(ty) - py(fy), px(tx) - px(fx));
      ctx.fillStyle = TEXT;
      ctx.beginPath();
      ctx.moveTo(px(tx), py(ty));
      ctx.lineTo(px(tx) - 9 * Math.cos(ang - 0.4), py(ty) - 9 * Math.sin(ang - 0.4));
      ctx.lineTo(px(tx) - 9 * Math.cos(ang + 0.4), py(ty) - 9 * Math.sin(ang + 0.4));
      ctx.closePath(); ctx.fill();
      ctx.font = '13px Inter, sans-serif'; ctx.textAlign = 'left';
      ctx.fillText('w', px(tx) + 6, py(ty) + 4);
    }

    // Frame
    ctx.strokeStyle = AXIS; ctx.lineWidth = 1;
    ctx.strokeRect(ox, oy, side, side);

    readout.innerHTML =
      a.toFixed(1) + '·x₁' + signed(b) + '·x₂' + signed(c) + ' = 0' +
      ' &nbsp; <span style="color:' + GREEN + '">green</span>: w·x + b ≥ 0';
  }

  return { resize: draw };
};
