// =====================================================================
// Shared deck logic for every module. Per-module data and hooks arrive
// via window.MODULE_CONFIG, set by each module's config.js (which the
// build inlines BEFORE this file):
//
//   window.MODULE_CONFIG = {
//     title:          'LLMs 0 to 100 - Module N',  // sets document.title
//     manimSections:  { scene: ['Clip_0000.mp4', ...], ... },
//     widgets:        { name: function(host){...}, ... },  // optional
//     onReady:        function(reveal){...},               // optional
//     onSlideChanged: function(event){...}                 // optional
//   };
// =====================================================================
var MODULE_CONFIG = window.MODULE_CONFIG || {};

// ---- Manim section step-through player ----
// Section video lists are inline to avoid fetch() issues with file:// protocol.
var MANIM_SECTIONS = MODULE_CONFIG.manimSections || {};

var manimSteppers = {};

function initManimStepper(video) {
  var scene = video.getAttribute('data-manim-scene');
  var sections = MANIM_SECTIONS[scene] || [];
  var state = { sections: sections, current: -1, playing: false, video: video, scene: scene };
  manimSteppers[scene] = state;

  // Set poster frame from first section
  if (sections.length > 0) {
    video.src = 'media/sections/' + sections[0];
    video.load();
  }

  video.addEventListener('ended', function() {
    state.playing = false;
  });
}

function advanceManimStepper(scene) {
  var state = manimSteppers[scene];
  if (!state || state.playing) return false;
  if (state.current >= state.sections.length - 1) return false;

  state.current++;
  state.playing = true;
  var file = state.sections[state.current];
  state.video.src = 'media/sections/' + file;
  state.video.load();
  state.video.play().then(function() {}).catch(function() { state.playing = false; });
  return true;
}

function resetManimStepper(scene) {
  var state = manimSteppers[scene];
  if (!state) return;
  state.current = -1;
  state.playing = false;
  state.video.pause();
  if (state.sections.length > 0) {
    state.video.src = 'media/sections/' + state.sections[0];
    state.video.load();
  }
}

// Instantly finish the section that is currently playing by jumping to its
// final frame. This lets an impatient "next" complete the animation in
// place rather than leaking through to reveal and skipping the slide.
// Returns true if it consumed the action (a video was actually playing).
function skipManimStepper(scene) {
  var state = manimSteppers[scene];
  if (!state || !state.playing) return false;
  var v = state.video;
  v.pause();
  // Seek just shy of the end so the step's end state stays on screen.
  if (v.duration && isFinite(v.duration)) {
    try { v.currentTime = Math.max(0, v.duration - 0.04); } catch (e) {}
  }
  state.playing = false;
  return true;
}

// One forward action on a manim slide. If a section is mid-play, the press
// completes it (skip to the end frame); otherwise it advances to the next
// section. Returns false only when there is nothing left to animate, which
// is the signal that reveal.js may now move on to the next slide.
function manimForward(scene) {
  var state = manimSteppers[scene];
  if (!state) return false;
  if (state.playing) return skipManimStepper(scene);
  return advanceManimStepper(scene);
}

// ---- Initialize Reveal.js ----
Reveal.initialize({
  width: 1280,
  height: 720,
  margin: 0,
  controls: true,
  progress: true,
  slideNumber: false,
  hash: true,
  transition: 'slide',
  center: false,
  autoAnimateDuration: 0.8,
  plugins: [ RevealHighlight, RevealMarkdown ]
});

// ---- Syntax highlighting ----
// reveal-markdown converts the inline <textarea> into <pre><code> blocks
// in a deferred microtask, AFTER the highlight plugin's init() has already
// run, so code ships un-highlighted. Re-trigger the plugin once the blocks
// exist (on ready) and for any added later (on slidechanged).
function highlightAllCodeBlocks() {
  if (!Reveal.getPlugin) return;
  var hl = Reveal.getPlugin('highlight');
  if (!hl || typeof hl.highlightBlock !== 'function') return;
  document.querySelectorAll('pre code').forEach(function(block) {
    if (block.dataset.highlighted === 'yes') return;
    hl.highlightBlock(block);
  });
}
// reveal-markdown may materialize blocks in waves, so re-run the
// (idempotent) pass across the startup window rather than once.
Reveal.on('ready', function() {
  var tries = 0;
  var iv = setInterval(function() {
    tries++;
    highlightAllCodeBlocks();
    if (tries > 25) clearInterval(iv);
  }, 100);
});
Reveal.on('slidechanged', highlightAllCodeBlocks);
Reveal.on('fragmentshown', highlightAllCodeBlocks);

function initStepperClickHandlers() {
  document.querySelectorAll('.manim-stepper').forEach(function(video) {
    if (video.dataset.clickInit) return;
    video.dataset.clickInit = '1';
    video.style.cursor = 'pointer';
    video.addEventListener('click', function(e) {
      var scene = video.getAttribute('data-manim-scene');
      manimForward(scene);
      e.stopPropagation();
      e.preventDefault();
    });
  });
}

function initAllSteppers() {
  document.querySelectorAll('.manim-stepper').forEach(function(v) {
    if (!manimSteppers[v.getAttribute('data-manim-scene')]) {
      initManimStepper(v);
    }
  });
  initStepperClickHandlers();
}

function renderMath() {
  if (typeof renderMathInElement !== 'undefined') {
    renderMathInElement(document.body, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false},
        {left: '\\(', right: '\\)', display: false}
      ]
    });
  }
}

// reveal-markdown loads slides asynchronously, so the 'ready' event may
// fire before the markdown content is in the DOM. Poll briefly.
Reveal.on('ready', function() {
  var tries = 0;
  var iv = setInterval(function() {
    tries++;
    if (document.querySelectorAll('section').length > 5 || tries > 20) {
      clearInterval(iv);
      initAllSteppers();
      renderMath();
    }
  }, 100);
});

// Reset steppers when leaving a slide
Reveal.on('slidechanged', function(event) {
  initAllSteppers();
  Object.keys(manimSteppers).forEach(function(scene) {
    if (!event.currentSlide.querySelector('[data-manim-scene="' + scene + '"]')) {
      resetManimStepper(scene);
    }
  });
});

// Intercept right arrow / space / enter on manim slides to step through
// sections. manimForward consumes the press while there is still animation
// to play (completing the current section or starting the next), so reveal
// only advances the slide once every section has been shown.
document.addEventListener('keydown', function(e) {
  if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'Enter') {
    var currentSlide = Reveal.getCurrentSlide();
    var stepper = currentSlide ? currentSlide.querySelector('.manim-stepper') : null;
    if (stepper) {
      var scene = stepper.getAttribute('data-manim-scene');
      if (manimForward(scene)) {
        e.stopImmediatePropagation();
        e.preventDefault();
      }
    }
  }
}, true);

// The on-screen controls (controls: true) navigate directly and bypass the
// keydown handler above, so guard the forward arrows the same way: a forward
// control press on a manim slide steps the animation before leaving.
document.addEventListener('click', function(e) {
  var btn = e.target.closest ? e.target.closest('.navigate-right, .navigate-down') : null;
  if (!btn) return;
  var currentSlide = Reveal.getCurrentSlide();
  var stepper = currentSlide ? currentSlide.querySelector('.manim-stepper') : null;
  if (stepper && manimForward(stepper.getAttribute('data-manim-scene'))) {
    e.stopImmediatePropagation();
    e.preventDefault();
  }
}, true);

// =====================================================================
// Interactive widgets (:::interactive). Each widget is a factory keyed
// by its `data-widget` value; it owns its host element and (re)draws
// whenever its slide becomes visible or the window resizes.
// =====================================================================
var INTERACTIVE_WIDGETS = {};

function hydrateInteractiveWidgets() {
  document.querySelectorAll('.interactive-host[data-widget]').forEach(function(host) {
    if (host.dataset.hydrated) return;
    var factory = INTERACTIVE_WIDGETS[host.getAttribute('data-widget')];
    if (!factory) return;
    host.dataset.hydrated = '1';
    host._widget = factory(host) || null;
  });
}

function refreshInteractiveWidgets() {
  document.querySelectorAll('.interactive-host[data-widget]').forEach(function(host) {
    if (host._widget && typeof host._widget.resize === 'function') host._widget.resize();
  });
}

Reveal.on('ready', function() {
  var tries = 0;
  var iv = setInterval(function() {
    tries++;
    if (document.querySelector('.interactive-host[data-widget]') || tries > 25) {
      clearInterval(iv);
      hydrateInteractiveWidgets();
      requestAnimationFrame(refreshInteractiveWidgets);
    }
  }, 100);
});
Reveal.on('slidechanged', function() {
  hydrateInteractiveWidgets();
  requestAnimationFrame(refreshInteractiveWidgets);
});
window.addEventListener('resize', refreshInteractiveWidgets);

// ---- Folding widget: input space <-> hidden-space line values, live ----
// The left pane shows each hidden neuron's line w.x+b=0. The right pane
// shows continuous signed line values: crossing a neuron line in input
// space crosses the matching axis in hidden space.
INTERACTIVE_WIDGETS.folding = function(host) {
  var GREEN = '#3fb950', RED = '#e74c3c', MUTED = '#8892a4',
      LINEC = '#2a3450', TEXT = '#e8eaf0', PRIMARY = '#4a9eff';
  var NEURON_COLORS = ['#4a9eff', '#f5a623'];

  // --- XOR-pattern data: opposite corners share a class ---
  function mulberry32(a) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  var rnd = mulberry32(42);
  var clusters = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
  var data = [];
  clusters.forEach(function(c) {
    for (var i = 0; i < 14; i++) {
      data.push({
        x: c[0] + (rnd() - 0.5) * 0.22,
        y: c[1] + (rnd() - 0.5) * 0.22,
        cls: c[2]
      });
    }
  });

  // Preset line positions for XOR: the two units detect points far above
  // or below the diagonal strip.
  function preset(k) {
    var p = k === 1 ? [
      { w1: 1, w2: -1, b: -0.3 }
    ] : [
      { w1: 1, w2: -1, b: -0.3 },
      { w1: -1, w2: 1, b: -0.3 }
    ];
    return p.map(function(o) { return { w1: o.w1, w2: o.w2, b: o.b }; });
  }
  var state = { k: 2, neurons: preset(2) };

  // --- DOM ---
  host.innerHTML =
    '<div class="fold-widget">' +
      '<div class="fold-views">' +
        '<figure class="fold-view"><figcaption>Input space</figcaption>' +
          '<canvas class="fold-canvas" data-pane="input"></canvas></figure>' +
        '<figure class="fold-view"><figcaption>Hidden space</figcaption>' +
          '<canvas class="fold-canvas" data-pane="hidden"></canvas></figure>' +
        '<figure class="fold-view fold-graph-view"><figcaption>Weights as edges</figcaption>' +
          '<canvas class="fold-canvas" data-pane="graph"></canvas></figure>' +
      '</div>' +
      '<div class="fold-controls">' +
        '<div class="fold-modes">' +
          '<button data-neurons="1">1 neuron</button>' +
          '<button data-neurons="2">2 neurons</button>' +
          '<button class="fold-reset">Reset to XOR solution</button>' +
        '</div>' +
        '<div class="fold-sliders"></div>' +
        '<p class="fold-readout"></p>' +
      '</div>' +
    '</div>';

  var inCanvas = host.querySelector('[data-pane="input"]');
  var graphCanvas = host.querySelector('[data-pane="graph"]');
  var hidCanvas = host.querySelector('[data-pane="hidden"]');
  var slidersEl = host.querySelector('.fold-sliders');
  var readoutEl = host.querySelector('.fold-readout');
  var modeBtns = host.querySelectorAll('.fold-modes button[data-neurons]');

  // Keep slider interaction from bubbling into reveal navigation.
  ['pointerdown', 'keydown'].forEach(function(ev) {
    host.addEventListener(ev, function(e) { e.stopPropagation(); });
  });

  function buildSliders() {
    slidersEl.innerHTML = '';
    state.neurons.forEach(function(n, idx) {
      var color = NEURON_COLORS[idx];
      [['w1', 'w₁'], ['w2', 'w₂'], ['b', 'b']].forEach(function(f) {
        var key = f[0];
        var row = document.createElement('div');
        row.className = 'fold-slider';
        var lab = document.createElement('label');
        lab.textContent = 'N' + (idx + 1) + ' ' + f[1];
        lab.style.color = color;
        var inp = document.createElement('input');
        inp.type = 'range';
        inp.min = '-3'; inp.max = '3'; inp.step = '0.1';
        inp.value = String(n[key]);
        var val = document.createElement('p');
        val.className = 'fold-val';
        val.textContent = (+n[key]).toFixed(1);
        inp.addEventListener('input', function() {
          n[key] = parseFloat(inp.value);
          val.textContent = n[key].toFixed(1);
          draw();
        });
        row.appendChild(lab); row.appendChild(inp); row.appendChild(val);
        slidersEl.appendChild(row);
      });
    });
  }

  function setK(k) {
    state.k = k;
    state.neurons = preset(k);
    modeBtns.forEach(function(b) {
      b.classList.toggle('active', +b.getAttribute('data-neurons') === k);
    });
    buildSliders();
    draw();
  }
  modeBtns.forEach(function(b) {
    b.addEventListener('click', function() { setK(+b.getAttribute('data-neurons')); });
  });
  host.querySelector('.fold-reset').addEventListener('click', function() {
    state.neurons = preset(state.k);
    buildSliders();
    draw();
  });

  // --- canvas helpers ---
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

  function makeMapper(w, h, xr, yr, pad) {
    return function(x, y) {
      return [
        pad + (x - xr[0]) / (xr[1] - xr[0]) * (w - 2 * pad),
        h - pad - (y - yr[0]) / (yr[1] - yr[0]) * (h - 2 * pad)
      ];
    };
  }

  function drawFrame(ctx, w, h, pad, xr, yr, xlab, ylab) {
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = LINEC; ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, w - 2 * pad, h - 2 * pad);
    var m = makeMapper(w, h, xr, yr, pad);
    // zero axes if visible
    ctx.strokeStyle = 'rgba(136,146,164,0.35)';
    if (xr[0] < 0 && xr[1] > 0) {
      var zx = m(0, yr[0])[0];
      ctx.beginPath(); ctx.moveTo(zx, pad); ctx.lineTo(zx, h - pad); ctx.stroke();
    }
    if (yr[0] < 0 && yr[1] > 0) {
      var zy = m(xr[0], 0)[1];
      ctx.beginPath(); ctx.moveTo(pad, zy); ctx.lineTo(w - pad, zy); ctx.stroke();
    }
    ctx.fillStyle = MUTED; ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'right'; ctx.fillText(xlab, w - pad - 4, h - pad - 6);
    ctx.textAlign = 'left'; ctx.save();
    ctx.translate(pad + 4, pad + 4); ctx.fillText(ylab, 0, 8); ctx.restore();
    return m;
  }

  function lineSegment(n, xr, yr) {
    var a = n.w1, b = n.w2, c = n.b;
    var pts = [];
    function add(x, y) {
      if (x < xr[0] - 1e-6 || x > xr[1] + 1e-6 || y < yr[0] - 1e-6 || y > yr[1] + 1e-6) return;
      for (var i = 0; i < pts.length; i++) {
        if (Math.abs(pts[i][0] - x) < 1e-6 && Math.abs(pts[i][1] - y) < 1e-6) return;
      }
      pts.push([x, y]);
    }
    if (Math.abs(b) > 1e-9) {
      add(xr[0], -(a * xr[0] + c) / b);
      add(xr[1], -(a * xr[1] + c) / b);
    }
    if (Math.abs(a) > 1e-9) {
      add(-(b * yr[0] + c) / a, yr[0]);
      add(-(b * yr[1] + c) / a, yr[1]);
    }
    return pts.length >= 2 ? [pts[0], pts[1]] : null;
  }

  function edge(ctx, x1, y1, x2, y2, color, label) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    var mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    ctx.fillStyle = '#0d1225';
    ctx.fillRect(mx - 22, my - 10, 44, 18);
    ctx.fillStyle = color;
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, mx, my + 4);
  }

  function node(ctx, x, y, label, stroke) {
    ctx.beginPath();
    ctx.arc(x, y, 18, 0, 6.2832);
    ctx.fillStyle = '#0d1225';
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2.4;
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = TEXT;
    ctx.font = '14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, x, y + 5);
  }

  function drawGraph() {
    var fg = fitCanvas(graphCanvas);
    if (!fg) return;
    var ctx = fg.ctx, w = fg.w, h = fg.h;
    ctx.clearRect(0, 0, w, h);
    var leftX = w * 0.24, rightX = w * 0.74;
    var inY = [h * 0.34, h * 0.66];
    var outY = state.k === 1 ? [h * 0.5] : [h * 0.34, h * 0.66];
    state.neurons.forEach(function(n, idx) {
      edge(ctx, leftX + 18, inY[0], rightX - 18, outY[idx], NEURON_COLORS[idx], 'w₁=' + n.w1.toFixed(1));
      edge(ctx, leftX + 18, inY[1], rightX - 18, outY[idx], NEURON_COLORS[idx], 'w₂=' + n.w2.toFixed(1));
      ctx.fillStyle = NEURON_COLORS[idx];
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('b=' + n.b.toFixed(1), rightX + 24, outY[idx] + 4);
    });
    node(ctx, leftX, inY[0], 'x₁', PRIMARY);
    node(ctx, leftX, inY[1], 'x₂', PRIMARY);
    state.neurons.forEach(function(n, idx) { node(ctx, rightX, outY[idx], 'h' + (idx + 1), NEURON_COLORS[idx]); });
  }

  function dot(ctx, px, py, color, r) {
    ctx.beginPath(); ctx.arc(px, py, r || 4.5, 0, 6.2832);
    ctx.fillStyle = color; ctx.globalAlpha = 0.9; ctx.fill(); ctx.globalAlpha = 1;
  }

  function signedDistance(n, p) {
    var len = Math.sqrt(n.w1 * n.w1 + n.w2 * n.w2) || 1;
    return (n.w1 * p.x + n.w2 * p.y + n.b) / len;
  }

  function tangentCoordinate(n, p) {
    var len = Math.sqrt(n.w1 * n.w1 + n.w2 * n.w2) || 1;
    return (-n.w2 * p.x + n.w1 * p.y) / len;
  }

  function foldCoordinatePoint(p) {
    var n1 = state.neurons[0];
    return {
      x: signedDistance(n1, p),
      y: state.k > 1 ? signedDistance(state.neurons[1], p) : tangentCoordinate(n1, p),
      cls: p.cls
    };
  }

  function spanWithZero(vals) {
    var lo = Math.min.apply(null, vals.concat([0]));
    var hi = Math.max.apply(null, vals.concat([0]));
    if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
    var pad = Math.max(0.08, (hi - lo) * 0.16);
    return [lo - pad, hi + pad];
  }

  function draw() {
    // ---- Input space (left): data + each neuron's decision line ----
    var fi = fitCanvas(inCanvas);
    if (fi) {
      var pad = 26, xr = [-0.5, 1.5], yr = [-0.5, 1.5];
      var m = drawFrame(fi.ctx, fi.w, fi.h, pad, xr, yr, 'x₁', 'x₂');
      var ctx = fi.ctx;
      ctx.save();
      ctx.beginPath();
      ctx.rect(pad, pad, fi.w - 2 * pad, fi.h - 2 * pad);
      ctx.clip();
      // Shade each neuron's active half-plane, where its line value is positive.
      state.neurons.forEach(function(n, idx) {
        var a = n.w1, b = n.w2, c = n.b;
        ctx.fillStyle = idx === 0 ? 'rgba(74,158,255,0.06)' : 'rgba(245,166,35,0.06)';
        // Simple approximation: fill the canvas then clip to the active side
        ctx.save();
        ctx.beginPath();
        if (Math.abs(b) > 1e-9) {
          var y0 = -(a * xr[0] + c) / b;
          var y1 = -(a * xr[1] + c) / b;
          if (b > 0) {
            ctx.moveTo(m(xr[0], yr[0])[0], m(xr[0], yr[0])[1]);
            ctx.lineTo(m(xr[1], yr[0])[0], m(xr[1], yr[0])[1]);
            ctx.lineTo(m(xr[1], y1)[0], m(xr[1], y1)[1]);
            ctx.lineTo(m(xr[0], y0)[0], m(xr[0], y0)[1]);
          } else {
            ctx.moveTo(m(xr[0], yr[1])[0], m(xr[0], yr[1])[1]);
            ctx.lineTo(m(xr[1], yr[1])[0], m(xr[1], yr[1])[1]);
            ctx.lineTo(m(xr[1], y1)[0], m(xr[1], y1)[1]);
            ctx.lineTo(m(xr[0], y0)[0], m(xr[0], y0)[1]);
          }
        } else if (Math.abs(a) > 1e-9) {
          var xv = -c / a;
          if (a > 0) {
            ctx.moveTo(m(xv, yr[0])[0], m(xv, yr[0])[1]);
            ctx.lineTo(m(xr[1], yr[0])[0], m(xr[1], yr[0])[1]);
            ctx.lineTo(m(xr[1], yr[1])[0], m(xr[1], yr[1])[1]);
            ctx.lineTo(m(xv, yr[1])[0], m(xv, yr[1])[1]);
          } else {
            ctx.moveTo(m(xv, yr[0])[0], m(xv, yr[0])[1]);
            ctx.lineTo(m(xr[0], yr[0])[0], m(xr[0], yr[0])[1]);
            ctx.lineTo(m(xr[0], yr[1])[0], m(xr[0], yr[1])[1]);
            ctx.lineTo(m(xv, yr[1])[0], m(xv, yr[1])[1]);
          }
        }
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      });
      state.neurons.forEach(function(n, idx) {
        var seg = lineSegment(n, xr, yr);
        if (!seg) return;
        var p0 = m(seg[0][0], seg[0][1]);
        var p1 = m(seg[1][0], seg[1][1]);
        ctx.strokeStyle = NEURON_COLORS[idx];
        ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.stroke();
        var dx = p1[0] - p0[0], dy = p1[1] - p0[1];
        var len = Math.sqrt(dx * dx + dy * dy) || 1;
        var sign = idx === 0 ? -1 : 1;
        var lx = (p0[0] + p1[0]) / 2 - dy / len * 18 * sign;
        var ly = (p0[1] + p1[1]) / 2 + dx / len * 18 * sign;
        ctx.fillStyle = '#0d1225';
        ctx.fillRect(lx - 14, ly - 10, 28, 18);
        ctx.fillStyle = NEURON_COLORS[idx];
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('N' + (idx + 1), lx, ly + 4);
      });
      data.forEach(function(p) {
        var q = m(p.x, p.y);
        dot(ctx, q[0], q[1], p.cls ? GREEN : RED);
      });
      ctx.restore();
    }

    drawGraph();

    // ---- Hidden space (middle): smooth line-value coordinates ----
    var fh = fitCanvas(hidCanvas);
    if (!fh) { return; }
    var pts = data.map(foldCoordinatePoint);
    var hxr = spanWithZero(pts.map(function(p) { return p.x; }));
    var hyr = spanWithZero(pts.map(function(p) { return p.y; }));
    var pad2 = 26;
    var mh = drawFrame(fh.ctx, fh.w, fh.h, pad2, hxr, hyr,
                       'N1 value', state.k > 1 ? 'N2 value' : 'along N1');
    var ctx2 = fh.ctx;

    ctx2.save();
    ctx2.beginPath();
    ctx2.rect(pad2, pad2, fh.w - 2 * pad2, fh.h - 2 * pad2);
    ctx2.clip();

    var n1x = mh(0, hyr[0])[0];
    ctx2.strokeStyle = NEURON_COLORS[0];
    ctx2.lineWidth = 2.4;
    ctx2.beginPath(); ctx2.moveTo(n1x, pad2); ctx2.lineTo(n1x, fh.h - pad2); ctx2.stroke();
    ctx2.fillStyle = NEURON_COLORS[0];
    ctx2.font = '12px Inter, sans-serif';
    ctx2.textAlign = 'left';
    ctx2.fillText('N1 line', n1x + 5, pad2 + 14);

    if (state.k > 1) {
      var n2y = mh(hxr[0], 0)[1];
      ctx2.strokeStyle = NEURON_COLORS[1];
      ctx2.lineWidth = 2.4;
      ctx2.beginPath(); ctx2.moveTo(pad2, n2y); ctx2.lineTo(fh.w - pad2, n2y); ctx2.stroke();
      ctx2.fillStyle = NEURON_COLORS[1];
      ctx2.textAlign = 'right';
      ctx2.fillText('N2 line', fh.w - pad2 - 5, n2y - 6);
    }

    pts.forEach(function(p) {
      var q = mh(p.x, p.y);
      dot(ctx2, q[0], q[1], p.cls ? GREEN : RED);
    });
    ctx2.restore();

    readoutEl.innerHTML =
      state.k === 1
        ? 'Hidden space: crossing the N1 line in input space crosses the vertical N1 axis here.'
        : 'Hidden space: x is the N1 line value, y is the N2 line value; crossing a line on the left crosses its matching axis here.';
  }

  buildSliders();
  modeBtns.forEach(function(b) {
    b.classList.toggle('active', +b.getAttribute('data-neurons') === state.k);
  });

  return { resize: draw };
};

// =====================================================================
// Shared MLP core for the boundaryExplorer and mlpBoundary widgets.
// The slide widgets use deterministic lookup presets keyed by
// dataset/width/depth so a button click always loads the intended working
// or under-capacity example. The older Adam trainer is kept as a fallback
// helper for the debug harness, but the deck paths pass a dataset name and
// therefore use lookupModel() rather than browser-time training.
// =====================================================================
var MLP = (function () {
  var EPOCHS = 220, LR = 0.05, BATCH = 16;
  var MAX_RESTARTS = 24, TIME_BUDGET_MS = 450, EARLY_STOP = 0.999;
  var CENTER = 0.5; // input centering offset

  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  // Deterministic datasets (fixed seed) normalized into [0.04, 0.96].
  function generateData(name) {
    var rnd = mulberry32(7);
    var pts = [];
    if (name === 'linear') {
      for (var i = 0; i < 60; i++) {
        var x = rnd() * 2 - 1, y = rnd() * 2 - 1;
        pts.push({ x: x + 1.5, y: y + 1.5, cls: 1 });
        pts.push({ x: x - 1.5, y: y - 1.5, cls: 0 });
      }
    } else if (name === 'xor') {
      var clusters = [[1.5, 1.5, 1], [1.5, -1.5, 0], [-1.5, 1.5, 0], [-1.5, -1.5, 1]];
      clusters.forEach(function (c) {
        for (var i = 0; i < 20; i++) {
          pts.push({ x: c[0] + (rnd() - 0.5) * 0.6, y: c[1] + (rnd() - 0.5) * 0.6, cls: c[2] });
        }
      });
    } else if (name === 'moons') {
      for (var i = 0; i < 50; i++) {
        var t = Math.PI * i / 50;
        pts.push({ x: Math.cos(t) + (rnd() - 0.5) * 0.12, y: Math.sin(t) + (rnd() - 0.5) * 0.12, cls: 1, idx: i });
        pts.push({ x: 1.1 - Math.cos(t) + (rnd() - 0.5) * 0.12, y: 0.4 - Math.sin(t) + (rnd() - 0.5) * 0.12, cls: 0, idx: i });
      }
    } else if (name === 'spiral') {
      // Gentle ~1.15-turn two-arm spiral: clearly a spiral, yet learnable by
      // a depth-2 ReLU net in-budget (the old 1.7-turn tight spiral was not).
      var turns = 1.15, n = 60, tmax = turns * 2 * Math.PI;
      for (var i = 0; i < n; i++) {
        var t = tmax * i / n;
        var r = 0.15 + 0.30 * t / (2 * Math.PI);
        pts.push({ x: r * Math.cos(t) + (rnd() - 0.5) * 0.04, y: r * Math.sin(t) + (rnd() - 0.5) * 0.04, cls: 1, idx: i });
        pts.push({ x: -r * Math.cos(t) + (rnd() - 0.5) * 0.04, y: -r * Math.sin(t) + (rnd() - 0.5) * 0.04, cls: 0, idx: i });
      }
    }
    var xs = pts.map(function (p) { return p.x; }), ys = pts.map(function (p) { return p.y; });
    var xmn = Math.min.apply(null, xs), xmx = Math.max.apply(null, xs);
    var ymn = Math.min.apply(null, ys), ymx = Math.max.apply(null, ys);
    var xr = xmx - xmn || 1, yr = ymx - ymn || 1;
    pts.forEach(function (p) {
      p.x = 0.04 + 0.92 * (p.x - xmn) / xr;
      p.y = 0.04 + 0.92 * (p.y - ymn) / yr;
    });
    return pts;
  }

  function zeros(n) { var a = []; for (var i = 0; i < n; i++) a.push(0); return a; }
  function randn(rng) {
    var u = Math.max(1e-12, rng());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }
  // He initialization (std = sqrt(2/in)) -- the right scale for ReLU layers.
  function heLayer(outDim, inDim, rng) {
    var std = Math.sqrt(2 / inDim), W = [], b = zeros(outDim);
    for (var i = 0; i < outDim; i++) {
      var row = [];
      for (var j = 0; j < inDim; j++) row.push(randn(rng) * std);
      W.push(row);
    }
    return { W: W, b: b };
  }
  function matVec(W, x) { return W.map(function (row) { return row.reduce(function (s, wj, j) { return s + wj * x[j]; }, 0); }); }
  function vecAdd(a, b) { return a.map(function (v, i) { return v + b[i]; }); }

  function createNet(width, depth, rng) {
    var layers = [heLayer(width, 2, rng)];
    for (var d = 1; d < depth; d++) layers.push(heLayer(width, width, rng));
    layers.push(heLayer(1, width, rng));
    return layers;
  }

  // Forward pass on a RAW input (centering already folded into layer-0 bias).
  function predictOne(x, layers) {
    if (layers && layers.kind === 'lookup') return lookupPredict(x[0], x[1], layers);
    var a = x;
    for (var l = 0; l < layers.length - 1; l++) {
      a = vecAdd(matVec(layers[l].W, a), layers[l].b).map(function (v) { return Math.max(0, v); });
    }
    var z = vecAdd(matVec(layers[layers.length - 1].W, a), layers[layers.length - 1].b)[0];
    return 1 / (1 + Math.exp(-z));
  }

  function computeAccuracy(data, layers) {
    var c = 0;
    for (var i = 0; i < data.length; i++) {
      if ((predictOne([data[i].x, data[i].y], layers) >= 0.5 ? 1 : 0) === data[i].cls) c++;
    }
    return c / data.length;
  }

  function displayPieces(width, depth) {
    return Math.max(1, width * depth);
  }

  function effectivePieces(dataset, width, depth) {
    var pieces = displayPieces(width, depth);
    if (dataset === 'spiral' && depth < 2) return Math.min(pieces, 6);
    return pieces;
  }

  function requiredPieces(dataset) {
    if (dataset === 'linear') return 1;
    if (dataset === 'xor') return 2;
    if (dataset === 'moons') return 6;
    if (dataset === 'spiral') return 16;
    return 1;
  }

  function pickPrototypes(data, pieces) {
    var perClass = Math.max(1, Math.floor(pieces / 2));
    var protos = [];
    [0, 1].forEach(function(cls) {
      var arm = data.filter(function(p) { return p.cls === cls; });
      for (var i = 0; i < perClass; i++) {
        var idx = Math.round(i * (arm.length - 1) / Math.max(1, perClass - 1));
        protos.push(arm[idx]);
      }
    });
    return protos;
  }

  function nearestClass(x, y, points) {
    var best = points[0], bd = Infinity;
    for (var i = 0; i < points.length; i++) {
      var dx = x - points[i].x, dy = y - points[i].y;
      var d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = points[i]; }
    }
    return best.cls;
  }

  function distToSegment(x, y, a, b) {
    var vx = b.x - a.x, vy = b.y - a.y;
    var wx = x - a.x, wy = y - a.y;
    var den = vx * vx + vy * vy || 1;
    var t = Math.max(0, Math.min(1, (wx * vx + wy * vy) / den));
    var px = a.x + vx * t, py = a.y + vy * t;
    var dx = x - px, dy = y - py;
    return Math.sqrt(dx * dx + dy * dy);
  }

  function distToPolyline(x, y, pts) {
    var best = Infinity;
    for (var i = 0; i < pts.length - 1; i++) {
      best = Math.min(best, distToSegment(x, y, pts[i], pts[i + 1]));
    }
    return best;
  }

  function lineSide(x, y, line) {
    return (line.x2 - line.x1) * (y - line.y1) - (line.y2 - line.y1) * (x - line.x1);
  }

  function predictFromLineGeometry(x, y, model) {
    var lines = model.lines || [];
    if (!lines.length) return 0;
    if (model.lineMode === 'stripe' && lines.length > 1) {
      return lineSide(x, y, lines[0]) >= 0 && lineSide(x, y, lines[1]) <= 0 ? 1 : 0;
    }
    return lineSide(x, y, lines[0]) >= 0 ? 1 : 0;
  }

  function lookupPredict(x, y, model) {
    var pieces = model.pieces;
    if (model.dataset === 'linear') return x + y >= 1 ? 1 : 0;
    if (model.dataset === 'xor') {
      if (pieces < 2) return x + y >= 1 ? 1 : 0;
      return (x + y < 0.7 || x + y > 1.3) ? 1 : 0;
    }
    if ((model.dataset === 'moons' || model.dataset === 'spiral') && model.polygon) {
      return pointInPolygon(x, y, model.polygon) ? 1 : 0;
    }
    if ((model.dataset === 'moons' || model.dataset === 'spiral') &&
        (model.lineMode === 'halfPlane' || model.lineMode === 'stripe')) {
      return predictFromLineGeometry(x, y, model);
    }
    if ((model.dataset === 'moons' || model.dataset === 'spiral') && model.solved && model.redCenterline) {
      return distToPolyline(x, y, model.centerline) <= distToPolyline(x, y, model.redCenterline) ? 1 : 0;
    }
    if ((model.dataset === 'moons' || model.dataset === 'spiral') && model.centerline) {
      return distToPolyline(x, y, model.centerline) <= model.band ? 1 : 0;
    }
    if (model.dataset === 'moons' || model.dataset === 'spiral') {
      return nearestClass(x, y, model.lookupPoints);
    }
    return 0;
  }

  function sampleClassPath(data, cls, n) {
    var arm = data.filter(function(p) { return p.cls === cls; }).sort(function(a, b) {
      return (a.idx || 0) - (b.idx || 0);
    });
    arm = arm.map(function(p, idx) {
      var sx = 0, sy = 0, c = 0;
      for (var j = Math.max(0, idx - 2); j <= Math.min(arm.length - 1, idx + 2); j++) {
        sx += arm[j].x; sy += arm[j].y; c++;
      }
      return { x: sx / c, y: sy / c, idx: p.idx };
    });
    var pts = [];
    for (var i = 0; i < n; i++) {
      var pos = i * (arm.length - 1) / Math.max(1, n - 1);
      var k = Math.floor(pos), t = pos - k;
      var a = arm[k], b = arm[Math.min(arm.length - 1, k + 1)];
      pts.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
    }
    return pts;
  }

  function clampUnit(v) {
    return Math.max(0.015, Math.min(0.985, v));
  }

  function cleanPoint(p) {
    return {
      x: clampUnit(isFinite(p.x) ? p.x : 0.5),
      y: clampUnit(isFinite(p.y) ? p.y : 0.5)
    };
  }

  function offsetPolylinePoints(center, offset) {
    var pts = [];
    for (var i = 0; i < center.length; i++) {
      var prev = center[Math.max(0, i - 1)];
      var next = center[Math.min(center.length - 1, i + 1)];
      var tx = next.x - prev.x, ty = next.y - prev.y;
      var len = Math.sqrt(tx * tx + ty * ty) || 1;
      var nx = -ty / len, ny = tx / len;
      pts.push(cleanPoint({ x: center[i].x + nx * offset, y: center[i].y + ny * offset }));
    }
    return pts;
  }

  function edgeLength(a, b) {
    var dx = b.x - a.x, dy = b.y - a.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  function resampleClosedPolygon(poly, count) {
    count = Math.max(3, count);
    if (!poly || poly.length < 3) return null;
    var lens = [], perimeter = 0;
    for (var i = 0; i < poly.length; i++) {
      var d = edgeLength(poly[i], poly[(i + 1) % poly.length]);
      lens.push(d);
      perimeter += d;
    }
    if (!isFinite(perimeter) || perimeter < 1e-6) return null;
    var out = [];
    var edge = 0, before = 0;
    for (var s = 0; s < count; s++) {
      var target = s * perimeter / count;
      while (edge < lens.length - 1 && before + lens[edge] < target) {
        before += lens[edge];
        edge++;
      }
      var a = poly[edge], b = poly[(edge + 1) % poly.length];
      var t = lens[edge] > 1e-9 ? (target - before) / lens[edge] : 0;
      out.push(cleanPoint({
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t
      }));
    }
    return out;
  }

  function polygonEdges(poly) {
    if (!poly || poly.length < 2) return [];
    var lines = [];
    for (var i = 0; i < poly.length; i++) {
      var a = poly[i], b = poly[(i + 1) % poly.length];
      lines.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y });
    }
    return lines;
  }

  function resampleOpenPolyline(pts, count) {
    if (!pts || pts.length < 2) return null;
    count = Math.max(2, count);
    var lens = [], total = 0;
    for (var i = 0; i < pts.length - 1; i++) {
      var d = edgeLength(pts[i], pts[i + 1]);
      lens.push(d);
      total += d;
    }
    if (!isFinite(total) || total < 1e-6) return null;
    var out = [];
    var edge = 0, before = 0;
    for (var s = 0; s < count; s++) {
      var target = s * total / Math.max(1, count - 1);
      while (edge < lens.length - 1 && before + lens[edge] < target) {
        before += lens[edge];
        edge++;
      }
      var a = pts[edge], b = pts[edge + 1];
      var t = lens[edge] > 1e-9 ? (target - before) / lens[edge] : 0;
      out.push(cleanPoint({
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t
      }));
    }
    return out;
  }

  function openPolylineEdges(pts) {
    if (!pts || pts.length < 2) return [];
    var lines = [];
    for (var i = 0; i < pts.length - 1; i++) {
      lines.push({ x1: pts[i].x, y1: pts[i].y, x2: pts[i + 1].x, y2: pts[i + 1].y });
    }
    return lines;
  }

  function rotateAroundCenter(p, dir) {
    var cx = 0.5, cy = 0.5;
    var dx = p.x - cx, dy = p.y - cy;
    return dir > 0
      ? cleanPoint({ x: cx - dy, y: cy + dx })
      : cleanPoint({ x: cx + dy, y: cy - dx });
  }

  function spiralBoundaryLines(data, count) {
    var first = Math.max(1, Math.floor(count / 2));
    var second = Math.max(1, count - first);
    var centerA = sampleClassPath(data, 1, first + 1).map(function(p) {
      return rotateAroundCenter(p, 1);
    });
    var centerB = sampleClassPath(data, 1, second + 1).map(function(p) {
      return rotateAroundCenter(p, -1);
    });
    return openPolylineEdges(centerA).concat(openPolylineEdges(centerB)).slice(0, count);
  }

  function nearestPoint(p, pts) {
    var best = pts[0], bd = Infinity;
    for (var i = 0; i < pts.length; i++) {
      var dx = p.x - pts[i].x, dy = p.y - pts[i].y;
      var d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = pts[i]; }
    }
    return best;
  }

  function separatorLines(data, count) {
    if (count < 2) return simpleUnderfitLines(count);
    var samples = Math.max(24, count * 3);
    var green = sampleClassPath(data, 1, samples);
    var red = sampleClassPath(data, 0, samples);
    var mids = green.map(function(g) {
      var r = nearestPoint(g, red);
      return cleanPoint({ x: (g.x + r.x) / 2, y: (g.y + r.y) / 2 });
    });
    if (mids.length > 2) {
      var a = mids[0], b = mids[1];
      var dx = b.x - a.x, dy = b.y - a.y;
      var len = Math.sqrt(dx * dx + dy * dy) || 1;
      mids.unshift(cleanPoint({ x: a.x - dx / len * 0.35, y: a.y - dy / len * 0.35 }));
      a = mids[mids.length - 2]; b = mids[mids.length - 1];
      dx = b.x - a.x; dy = b.y - a.y;
      len = Math.sqrt(dx * dx + dy * dy) || 1;
      mids.push(cleanPoint({ x: b.x + dx / len * 0.35, y: b.y + dy / len * 0.35 }));
    }
    var path = resampleOpenPolyline(mids, count + 1);
    return openPolylineEdges(path);
  }

  function simplifyContourSegments(segments, count) {
    if (!segments.length) return simpleUnderfitLines(count);
    function key(p) { return p.x.toFixed(5) + ',' + p.y.toFixed(5); }
    var endpoints = {};
    var used = segments.map(function() { return false; });
    segments.forEach(function(s, idx) {
      var a = key(s.a), b = key(s.b);
      if (!endpoints[a]) endpoints[a] = [];
      if (!endpoints[b]) endpoints[b] = [];
      endpoints[a].push(idx);
      endpoints[b].push(idx);
    });
    function other(seg, pkey) {
      return key(seg.a) === pkey ? seg.b : seg.a;
    }
    function trace(startIdx) {
      var seg = segments[startIdx];
      var startKey = key(seg.a), endKey = key(seg.b);
      if ((endpoints[endKey] || []).length === 1 && (endpoints[startKey] || []).length !== 1) {
        startKey = key(seg.b);
      }
      var pts = [];
      var curKey = startKey;
      var guard = 0;
      while (guard++ < segments.length + 2) {
        var options = (endpoints[curKey] || []).filter(function(i) { return !used[i]; });
        if (!options.length) break;
        var idx = options[0];
        used[idx] = true;
        var s = segments[idx];
        if (!pts.length) pts.push(key(s.a) === curKey ? s.a : s.b);
        var nxt = other(s, curKey);
        pts.push(nxt);
        curKey = key(nxt);
      }
      return pts;
    }
    var paths = [];
    for (var i = 0; i < segments.length; i++) {
      if (!used[i]) {
        var p = trace(i);
        if (p.length > 1) paths.push(p);
      }
    }
    function pathLength(path) {
      var total = 0;
      for (var i = 0; i < path.length - 1; i++) total += edgeLength(path[i], path[i + 1]);
      return total;
    }
    paths = paths.map(function(path) { return { pts: path, len: pathLength(path) }; })
      .filter(function(p) { return p.len > 1e-4; })
      .sort(function(a, b) { return b.len - a.len; });
    if (!paths.length) return simpleUnderfitLines(count);
    paths = paths.slice(0, Math.min(paths.length, count));
    var totalLen = paths.reduce(function(s, p) { return s + p.len; }, 0) || 1;
    var alloc = paths.map(function(p) {
      return Math.max(1, Math.round(count * p.len / totalLen));
    });
    while (alloc.reduce(function(s, n) { return s + n; }, 0) > count) {
      var maxIdx = 0;
      for (var m = 1; m < alloc.length; m++) if (alloc[m] > alloc[maxIdx]) maxIdx = m;
      if (alloc[maxIdx] > 1) alloc[maxIdx]--;
      else break;
    }
    while (alloc.reduce(function(s, n) { return s + n; }, 0) < count) {
      var bestIdx = 0;
      for (var n = 1; n < paths.length; n++) if (paths[n].len > paths[bestIdx].len) bestIdx = n;
      alloc[bestIdx]++;
    }
    var out = [];
    paths.forEach(function(path, idx) {
      var resampled = resampleOpenPolyline(path.pts, alloc[idx] + 1);
      out = out.concat(openPolylineEdges(resampled));
    });
    return out.slice(0, count);
  }

  function classifierBoundaryLines(data, count, dataset) {
    var centerCount = dataset === 'spiral' ? 24 : Math.max(8, count * 2);
    var green = sampleClassPath(data, 1, centerCount);
    var red = sampleClassPath(data, 0, centerCount);
    function pred(x, y) {
      return distToPolyline(x, y, green) <= distToPolyline(x, y, red) ? 1 : 0;
    }
    var CRES = dataset === 'spiral' ? 92 : 72;
    var vals = [];
    for (var i = 0; i <= CRES; i++) {
      vals[i] = [];
      for (var j = 0; j <= CRES; j++) vals[i][j] = pred(i / CRES, j / CRES);
    }
    function pointOnEdge(a, b) {
      return {
        x: (a[0] + b[0]) / (2 * CRES),
        y: (a[1] + b[1]) / (2 * CRES)
      };
    }
    var raw = [];
    for (var ci = 0; ci < CRES; ci++) {
      for (var cj = 0; cj < CRES; cj++) {
        var corners = [
          [ci, cj, vals[ci][cj]],
          [ci + 1, cj, vals[ci + 1][cj]],
          [ci + 1, cj + 1, vals[ci + 1][cj + 1]],
          [ci, cj + 1, vals[ci][cj + 1]]
        ];
        var crossings = [];
        for (var e = 0; e < 4; e++) {
          var a = corners[e], b = corners[(e + 1) % 4];
          if (a[2] !== b[2]) crossings.push(pointOnEdge(a, b));
        }
        if (crossings.length >= 2) {
          for (var c = 0; c + 1 < crossings.length; c += 2) {
            raw.push({ a: crossings[c], b: crossings[c + 1] });
          }
        }
      }
    }
    return simplifyContourSegments(raw, count);
  }

  function tubePolygon(dataset, data, count, band) {
    if (count < 3) return null;
    var center = sampleClassPath(data, 1, Math.max(12, count + 6));
    var outer = offsetPolylinePoints(center, band);
    var inner = offsetPolylinePoints(center, -band).reverse();
    return resampleClosedPolygon(outer.concat(inner), count);
  }

  function cross(o, a, b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  }

  function convexHull(points) {
    var pts = points.map(cleanPoint).sort(function(a, b) {
      return a.x === b.x ? a.y - b.y : a.x - b.x;
    });
    if (pts.length <= 1) return pts;
    var lower = [];
    pts.forEach(function(p) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
      lower.push(p);
    });
    var upper = [];
    for (var i = pts.length - 1; i >= 0; i--) {
      var p = pts[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
      upper.push(p);
    }
    upper.pop();
    lower.pop();
    return lower.concat(upper);
  }

  function hullPolygon(data, count, margin) {
    if (count < 3) return null;
    var pts = data.filter(function(p) { return p.cls === 1; });
    var hull = convexHull(pts);
    if (hull.length < 3) return null;
    var cx = 0, cy = 0;
    hull.forEach(function(p) { cx += p.x; cy += p.y; });
    cx /= hull.length; cy /= hull.length;
    var grown = hull.map(function(p) {
      var dx = p.x - cx, dy = p.y - cy;
      var len = Math.sqrt(dx * dx + dy * dy) || 1;
      return cleanPoint({ x: p.x + margin * dx / len, y: p.y + margin * dy / len });
    });
    return resampleClosedPolygon(grown, count);
  }

  function pointInPolygon(x, y, poly) {
    if (!poly || poly.length < 3) return false;
    var inside = false;
    for (var i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      var xi = poly[i].x, yi = poly[i].y;
      var xj = poly[j].x, yj = poly[j].y;
      var intersect = ((yi > y) !== (yj > y)) &&
        (x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-9) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }

  function simpleUnderfitLines(count) {
    return [
      { x1: 0.04, y1: 0.86, x2: 0.86, y2: 0.04 },
      { x1: 0.14, y1: 0.96, x2: 0.96, y2: 0.14 }
    ].slice(0, count);
  }

  function geometryForModel(dataset, data, count, depth, band, solved) {
    count = Math.max(1, count);
    if (dataset === 'xor' && count >= 2) {
      return { polygon: null, lines: [
        { x1: 0.04, y1: 0.66, x2: 0.66, y2: 0.04 },
        { x1: 0.34, y1: 0.96, x2: 0.96, y2: 0.34 }
      ].slice(0, count) };
    }
    if (dataset === 'linear' || count === 1) {
      return { polygon: null, lineMode: 'halfPlane', lines: [{ x1: 0.04, y1: 0.96, x2: 0.96, y2: 0.04 }] };
    }
    if (dataset === 'moons' || dataset === 'spiral') {
      if (count < 3) {
        return { polygon: null, lineMode: count > 1 ? 'stripe' : 'halfPlane', lines: simpleUnderfitLines(count) };
      }
      if (!solved || (dataset === 'spiral' && depth < 2)) {
        var polygon = hullPolygon(data, count, dataset === 'spiral' ? 0.02 : 0.035);
        return { polygon: polygon, lines: polygonEdges(polygon) };
      }
      return { polygon: null, lines: classifierBoundaryLines(data, count, dataset) };
    }
    return { polygon: null, lineMode: 'halfPlane', lines: [{ x1: 0.04, y1: 0.96, x2: 0.96, y2: 0.04 }] };
  }

  function lookupModel(dataset, data, width, depth) {
    var pieces = effectivePieces(dataset, width, depth);
    var solved = pieces >= requiredPieces(dataset);
    var display = displayPieces(width, depth);
    var band = dataset === 'spiral' ? 0.078 : 0.12;
    var centerCount = solved ? (dataset === 'spiral' ? 20 : Math.max(4, Math.ceil(display / 2) + 1))
                             : Math.max(3, Math.ceil(pieces / 2) + 1);
    var centerline = (dataset === 'moons' || dataset === 'spiral') ? sampleClassPath(data, 1, centerCount) : null;
    var redCenterline = (dataset === 'moons' || dataset === 'spiral') ? sampleClassPath(data, 0, centerCount) : null;
    var lookupPoints = (dataset === 'moons' || dataset === 'spiral') && solved ? data : pickPrototypes(data, pieces);
    var geom = geometryForModel(dataset, data, display, depth, band, solved);
    return {
      kind: 'lookup',
      dataset: dataset,
      width: width,
      depth: depth,
      solved: solved,
      pieces: pieces,
      displayPieces: display,
      lookupPoints: lookupPoints,
      centerline: centerline,
      redCenterline: redCenterline,
      band: band,
      polygon: geom.polygon,
      lineMode: geom.lineMode,
      lines: geom.lines
    };
  }

  // One Adam run: EPOCHS epochs of shuffled mini-batches over CENTER-shifted
  // inputs. Standard backprop for a ReLU stack with a sigmoid + BCE output.
  function runAdam(data, layers, rng) {
    var pts = data.map(function (p) { return { x: [p.x - CENTER, p.y - CENTER], y: p.cls }; });
    var N = pts.length, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    var mW = layers.map(function (L) { return L.W.map(function (r) { return r.map(function () { return 0; }); }); });
    var vW = layers.map(function (L) { return L.W.map(function (r) { return r.map(function () { return 0; }); }); });
    var mb = layers.map(function (L) { return L.b.map(function () { return 0; }); });
    var vb = layers.map(function (L) { return L.b.map(function () { return 0; }); });
    var t = 0;
    for (var ep = 0; ep < EPOCHS; ep++) {
      for (var i = N - 1; i > 0; i--) { var jr = Math.floor(rng() * (i + 1)); var tmp = pts[i]; pts[i] = pts[jr]; pts[jr] = tmp; }
      for (var start = 0; start < N; start += BATCH) {
        var end = Math.min(start + BATCH, N), bn = end - start;
        var dW = layers.map(function (L) { return L.W.map(function (r) { return r.map(function () { return 0; }); }); });
        var db = layers.map(function (L) { return L.b.map(function () { return 0; }); });
        for (var s = start; s < end; s++) {
          var as = [pts[s].x], zs = [];
          for (var l = 0; l < layers.length; l++) {
            var z = vecAdd(matVec(layers[l].W, as[l]), layers[l].b);
            zs.push(z);
            if (l < layers.length - 1) as.push(z.map(function (v) { return Math.max(0, v); }));
            else as.push(z);
          }
          var pred = 1 / (1 + Math.exp(-as[as.length - 1][0]));
          var delta = [pred - pts[s].y];
          for (var l = layers.length - 1; l >= 0; l--) {
            var aPrev = as[l];
            for (var j = 0; j < layers[l].W.length; j++) {
              for (var k = 0; k < layers[l].W[j].length; k++) dW[l][j][k] += delta[j] * aPrev[k];
              db[l][j] += delta[j];
            }
            if (l > 0) {
              var W = layers[l].W, nd = [];
              for (var k = 0; k < layers[l - 1].b.length; k++) {
                var sum = 0;
                for (var j = 0; j < W.length; j++) sum += W[j][k] * delta[j];
                nd.push(sum * (zs[l - 1][k] > 0 ? 1 : 0));
              }
              delta = nd;
            }
          }
        }
        t++;
        var bc1 = 1 - Math.pow(b1, t), bc2 = 1 - Math.pow(b2, t);
        for (var l = 0; l < layers.length; l++) {
          for (var j = 0; j < layers[l].W.length; j++) {
            for (var k = 0; k < layers[l].W[j].length; k++) {
              var g = dW[l][j][k] / bn;
              mW[l][j][k] = b1 * mW[l][j][k] + (1 - b1) * g;
              vW[l][j][k] = b2 * vW[l][j][k] + (1 - b2) * g * g;
              layers[l].W[j][k] -= LR * (mW[l][j][k] / bc1) / (Math.sqrt(vW[l][j][k] / bc2) + eps);
            }
            var gb = db[l][j] / bn;
            mb[l][j] = b1 * mb[l][j] + (1 - b1) * gb;
            vb[l][j] = b2 * vb[l][j] + (1 - b2) * gb * gb;
            layers[l].b[j] -= LR * (mb[l][j] / bc1) / (Math.sqrt(vb[l][j] / bc2) + eps);
          }
        }
      }
    }
    return layers;
  }

  // Fold the input centering into layer-0 bias so the net evaluates on raw
  // coordinates: W.(x - 0.5) + b == W.x + (b - 0.5 * sum(W_row)).
  function absorbCenter(layers) {
    var L0 = layers[0];
    for (var j = 0; j < L0.W.length; j++) {
      var s = 0; for (var k = 0; k < L0.W[j].length; k++) s += L0.W[j][k];
      L0.b[j] -= CENTER * s;
    }
    return layers;
  }

  // Best-of-N restarts under a wall-clock budget: stop on a (near-)perfect
  // fit, else keep retrying until the budget is spent. Returns the most
  // accurate net, already converted for raw-coordinate evaluation.
  function train(data, width, depth, dataset) {
    if (dataset) return lookupModel(dataset, data, width, depth);
    var best = null, bestAcc = -1, t0 = Date.now();
    for (var r = 0; r < MAX_RESTARTS; r++) {
      var rng = mulberry32(1009 + data.length * 17 + width * 101 + depth * 10007 + r * 7919);
      var net = absorbCenter(runAdam(data, createNet(width, depth, rng), rng));
      var acc = computeAccuracy(data, net);
      if (acc > bestAcc) { bestAcc = acc; best = net; }
      if (bestAcc >= EARLY_STOP) break;
      if (Date.now() - t0 > TIME_BUDGET_MS) break;
    }
    return best;
  }

  return {
    generateData: generateData, train: train,
    predictOne: predictOne, computeAccuracy: computeAccuracy,
  };
})();

// ---- MLP boundary widget: deterministic lookup of MLP boundary presets ----
// "Many lines make a curve": each hidden neuron contributes one linear
// piece to the decision boundary. Width x depth = total pieces.
// The selected dataset/architecture loads a fixed preset so the visual is
// repeatable: each displayed straight piece corresponds to one hidden unit.
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
    for (var i = 0; i < GRID; i++) {
      for (var j = 0; j < GRID; j++) {
        var gx = i / (GRID - 1), gy = j / (GRID - 1);
        var p = predictOne([gx, gy], state.net);
        ctx.fillStyle = p >= 0.5 ? 'rgba(63,185,80,0.16)' : 'rgba(231,76,60,0.10)';
        ctx.fillRect(ox + gx * side, oy + (1 - gy) * side, side / GRID + 1, side / GRID + 1);
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

// ---- Boundary Explorer: explicit two-neuron hidden layer for XOR ----
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
// ---- Single Perceptron widget: adjustable line on data, no training ----
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

// ---- Loss landscape widget: 3D draggable surface with height coloring ----
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
        '] + (-' + state.lastStep.eta.toFixed(1) + ' &middot; batch-gradient B=' + state.lastStep.batch + ' [' +
        state.lastStep.g1.toFixed(3) + ', ' + state.lastStep.g2.toFixed(3) +
        ']). Current loss: <strong>' + curL.toFixed(3) + '</strong>.';
    } else {
      readout.innerHTML =
        'Batch gradient B=' + state.batch + ': [' + g.g1.toFixed(3) + ', ' + g.g2.toFixed(3) +
        '] (true: [' + g.trueG1.toFixed(3) + ', ' + g.trueG2.toFixed(3) + '])' +
        '. Next: w_new = [' + state.w1.toFixed(2) + ', ' + state.w2.toFixed(2) +
        '] + (-' + state.eta.toFixed(1) + ' &middot; [' + g.g1.toFixed(3) + ', ' + g.g2.toFixed(3) + ']).';
    }
  }
  syncWeightReadouts();
  return { resize: draw };
};

// ---- Adam Landscape widget: loss surface + Adam terms (m, v, m_hat, v_hat) ----
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

// ---- Per-module hooks (from MODULE_CONFIG) ----
if (MODULE_CONFIG.title) document.title = MODULE_CONFIG.title;
if (MODULE_CONFIG.widgets) Object.assign(INTERACTIVE_WIDGETS, MODULE_CONFIG.widgets);
if (typeof MODULE_CONFIG.onReady === 'function') {
  Reveal.on('ready', function() { MODULE_CONFIG.onReady(Reveal); });
}
if (typeof MODULE_CONFIG.onSlideChanged === 'function') {
  Reveal.on('slidechanged', function(event) { MODULE_CONFIG.onSlideChanged(event); });
}
