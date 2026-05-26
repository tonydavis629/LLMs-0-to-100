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

// ---- Folding widget: input space <-> hidden space, live ----
// One neuron j computes h_j = ReLU(w1_j*x1 + w2_j*x2 + b_j). The left
// pane shows each neuron's decision line w.x+b=0 in the input space;
// the right pane shows every point after the fold (h_1..h_K). Moving a
// slider moves a line on the left AND refolds the points on the right,
// making "a neuron's boundary == an axis of the fold" visible.
INTERACTIVE_WIDGETS.folding = function(host) {
  var GREEN = '#3fb950', RED = '#e74c3c', MUTED = '#8892a4',
      LINEC = '#2a3450', TEXT = '#e8eaf0', PRIMARY = '#4a9eff', SECON = '#f5a623';
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

  // Preset weights that fold XOR into a linearly separable layout.
  function preset(k) {
    var p = [
      { w1: 1, w2: -1, b: 0.5 },
      { w1: -1, w2: 1, b: 0.5 }
    ];
    return p.slice(0, k).map(function(o) { return { w1: o.w1, w2: o.w2, b: o.b }; });
  }
  var state = { k: 2, neurons: preset(2) };

  // --- DOM ---
  host.innerHTML =
    '<div class="fold-widget">' +
      '<div class="fold-views">' +
        '<figure class="fold-view"><figcaption>Input space</figcaption>' +
          '<canvas class="fold-canvas" data-pane="input"></canvas></figure>' +
        '<figure class="fold-view fold-graph-view"><figcaption>Weights as edges</figcaption>' +
          '<canvas class="fold-canvas" data-pane="graph"></canvas></figure>' +
        '<figure class="fold-view"><figcaption>Hidden space &mdash; h = ReLU(Wx + b)</figcaption>' +
          '<canvas class="fold-canvas" data-pane="hidden"></canvas></figure>' +
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

  function relu(z) { return z > 0 ? z : 0; }

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
    ctx.fillStyle = MUTED;
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Each edge label is the same weight used by the line on the left.', w / 2, h - 12);
  }

  function dot(ctx, px, py, color, r) {
    ctx.beginPath(); ctx.arc(px, py, r || 4.5, 0, 6.2832);
    ctx.fillStyle = color; ctx.globalAlpha = 0.9; ctx.fill(); ctx.globalAlpha = 1;
  }

  // Hidden coords for a point: (h1, h2[, h3]) via ReLU(Wx+b).
  function hidden(p) {
    return state.neurons.map(function(n) {
      return relu(n.w1 * p.x + n.w2 * p.y + n.b);
    });
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
      state.neurons.forEach(function(n, idx) {
        var a = n.w1, b = n.w2, c = n.b, p0, p1;
        if (Math.abs(b) > 1e-9) {
          p0 = m(xr[0], -(a * xr[0] + c) / b);
          p1 = m(xr[1], -(a * xr[1] + c) / b);
        } else if (Math.abs(a) > 1e-9) {
          p0 = m(-c / a, yr[0]); p1 = m(-c / a, yr[1]);
        } else { return; }
        ctx.strokeStyle = NEURON_COLORS[idx];
        ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.stroke();
        var lx = (p0[0] + p1[0]) / 2;
        var ly = (p0[1] + p1[1]) / 2;
        ctx.fillStyle = '#0d1225';
        ctx.fillRect(lx - 78, ly - 11, 156, 19);
        ctx.fillStyle = NEURON_COLORS[idx];
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(
          'h' + (idx + 1) + ': ' + n.w1.toFixed(1) + 'x₁ + ' +
          n.w2.toFixed(1) + 'x₂ + ' + n.b.toFixed(1) + ' = 0',
          lx, ly + 4
        );
      });
      data.forEach(function(p) {
        var q = m(p.x, p.y);
        dot(ctx, q[0], q[1], p.cls ? GREEN : RED);
      });
      ctx.restore();
    }

    drawGraph();

    // ---- Hidden space (right): folded points + centroid separator ----
    var fh = fitCanvas(hidCanvas);
    if (!fh) { return; }
    var H = data.map(function(p) { return { h: hidden(p), cls: p.cls }; });
    var k = state.k;
    // project to 2D: K=1 uses (h1, jitter); K>=2 uses (h1, h2)
    var pts = H.map(function(o, i) {
      if (k === 1) return { x: o.h[0], y: ((i % 7) - 3) * 0.12, cls: o.cls };
      return { x: o.h[0], y: o.h[1], cls: o.cls };
    });
    var xs = pts.map(function(p) { return p.x; });
    var ys = pts.map(function(p) { return p.y; });
    function span(arr) {
      var lo = Math.min.apply(null, arr), hi = Math.max.apply(null, arr);
      if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
      var pd = (hi - lo) * 0.12;
      return [lo - pd, hi + pd];
    }
    var hxr = span(xs), hyr = k === 1 ? [-1, 1] : span(ys);
    var pad2 = 26;
    var mh = drawFrame(fh.ctx, fh.w, fh.h, pad2, hxr, hyr,
                       'h₁', k === 1 ? '' : 'h₂');
    var ctx2 = fh.ctx;

    // Logistic-regression separator: a single, smooth, deterministic
    // linear boundary in hidden space. A perceptron stops at the FIRST of
    // the infinitely many lines that separate the points, so a tiny shift
    // in the fold makes it jump to a different valid line and the boundary
    // flips discontinuously. Gradient descent on the convex log-loss
    // instead has one optimum that moves continuously with the sliders, so
    // nudging a weight nudges the line. Features are standardized so a
    // fixed learning rate converges at any hidden-space scale; the fitted
    // line is then mapped back to raw coordinates for drawing.
    var oneD = (k === 1);
    function stats(sel) {
      var n = pts.length, m = 0;
      pts.forEach(function(p) { m += sel(p); });
      m /= n;
      var v = 0;
      pts.forEach(function(p) { var dz = sel(p) - m; v += dz * dz; });
      return { m: m, sd: Math.sqrt(v / n) || 1 };
    }
    var stx = stats(function(p) { return p.x; });
    var sty = oneD ? { m: 0, sd: 1 } : stats(function(p) { return p.y; });
    var a = 0, cY = 0, d = 0, lr = 0.6;
    for (var ep = 0; ep < 500; ep++) {
      var ga = 0, gc = 0, gd = 0;
      pts.forEach(function(p) {
        var zx = (p.x - stx.m) / stx.sd;
        var zy = oneD ? 0 : (p.y - sty.m) / sty.sd;
        var pr = 1 / (1 + Math.exp(-(a * zx + cY * zy + d)));
        var e = pr - p.cls;
        ga += e * zx; if (!oneD) gc += e * zy; gd += e;
      });
      var inv = lr / pts.length;
      a -= ga * inv; if (!oneD) cY -= gc * inv; d -= gd * inv;
    }
    // Map standardized weights back to raw hidden coords so the same
    // wx*x + wy*y + b = 0 line code below can draw it.
    var sw = {
      wx: a / stx.sd,
      wy: oneD ? 0 : cY / sty.sd,
      b: d - a * stx.m / stx.sd - (oneD ? 0 : cY * sty.m / sty.sd)
    };
    var correct = 0;
    pts.forEach(function(p) {
      var s = (sw.wx * p.x + (oneD ? 0 : sw.wy * p.y) + sw.b) >= 0 ? 1 : 0;
      if (s === p.cls) correct++;
    });
    var acc = correct / pts.length;

    ctx2.save();
    ctx2.beginPath();
    ctx2.rect(pad2, pad2, fh.w - 2 * pad2, fh.h - 2 * pad2);
    ctx2.clip();
    if (Math.abs(sw.wx) + Math.abs(sw.wy) > 1e-6) {
      // decision line sw.wx*x + sw.wy*y + sw.b = 0, clipped to frame
      var e0, e1;
      if (!oneD && Math.abs(sw.wy) > 1e-6) {
        e0 = mh(hxr[0], -(sw.wx * hxr[0] + sw.b) / sw.wy);
        e1 = mh(hxr[1], -(sw.wx * hxr[1] + sw.b) / sw.wy);
      } else {
        var xv = -sw.b / sw.wx;
        e0 = mh(xv, hyr[0]); e1 = mh(xv, hyr[1]);
      }
      ctx2.strokeStyle = SECON; ctx2.lineWidth = 2;
      ctx2.setLineDash([7, 6]);
      ctx2.beginPath(); ctx2.moveTo(e0[0], e0[1]); ctx2.lineTo(e1[0], e1[1]); ctx2.stroke();
      ctx2.setLineDash([]);
    }
    pts.forEach(function(p) {
      var q = mh(p.x, p.y);
      dot(ctx2, q[0], q[1], p.cls ? GREEN : RED);
    });
    ctx2.restore();

    var sep = acc > 0.999;
    readoutEl.innerHTML =
      'Straight-line separator in hidden space: <strong>' +
      (acc * 100).toFixed(0) + '%</strong> &mdash; ' +
      (sep ? 'the fold made the classes linearly separable'
           : 'still entangled: one flat cut cannot split them yet');
  }

  buildSliders();
  modeBtns.forEach(function(b) {
    b.classList.toggle('active', +b.getAttribute('data-neurons') === state.k);
  });

  return { resize: draw };
};

// ---- MLP boundary widget: width x depth half-plane cuts wrap a class ----
// A one-hidden-layer ReLU network carves a CONVEX region as the
// intersection of its neurons' half-planes: each neuron is one straight
// cut, the output neuron ANDs them. That region is a polygon, and adding
// neurons adds sides, so the boundary rounds toward the circle it is
// approximating -- "many lines make a curve". Depth lets a deep net compose
// cuts into more pieces, so width x depth is the piece budget. This is a
// faithful geometric construction of a representable boundary, not a fit.
INTERACTIVE_WIDGETS.mlpBoundary = function(host) {
  var GREEN = '#3fb950', RED = '#e74c3c', MUTED = '#8892a4',
      LINEC = '#2a3450', TEXT = '#e8eaf0', PRIMARY = '#4a9eff', SECONDARY = '#f5a623';
  var state = { width: 4, depth: 2 };

  // --- Disk-in-ring data: green class fills a central disk, red class fills
  // the surrounding ring. The circle between them is the "curve" the cuts
  // approximate. Seeded so the layout is stable across redraws. ---
  var CX = 0.5, CY = 0.5, GREEN_R = 0.17;
  var data = (function() {
    var a = 7;
    function rnd() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
    var pts = [], i;
    for (i = 0; i < 70; i++) {                 // green: uniform inside the disk
      var rg = Math.sqrt(rnd()) * GREEN_R, ag = rnd() * 6.2832;
      pts.push({ x: CX + rg * Math.cos(ag), y: CY + rg * Math.sin(ag), cls: 1 });
    }
    for (i = 0; i < 90; i++) {                 // red: uniform in the outer ring
      var rr = 0.27 + rnd() * 0.20, ar = rnd() * 6.2832;
      pts.push({
        x: Math.max(0.03, Math.min(0.97, CX + rr * Math.cos(ar))),
        y: Math.max(0.03, Math.min(0.97, CY + rr * Math.sin(ar))),
        cls: 0
      });
    }
    return pts;
  })();

  host.innerHTML =
    '<div class="mlp-widget">' +
      '<div class="mlp-canvas-wrap"><canvas class="mlp-canvas"></canvas></div>' +
      '<div class="mlp-controls">' +
        '<div class="mlp-slider"><label>Width</label><input type="range" min="1" max="8" step="1" value="4"><p data-readout="width">4</p></div>' +
        '<div class="mlp-slider"><label>Depth</label><input type="range" min="1" max="4" step="1" value="2"><p data-readout="depth">2</p></div>' +
        '<p class="mlp-readout"></p>' +
      '</div>' +
    '</div>';

  var canvas = host.querySelector('.mlp-canvas');
  var widthInput = host.querySelector('input');
  var depthInput = host.querySelectorAll('input')[1];
  var widthText = host.querySelector('[data-readout="width"]');
  var depthText = host.querySelector('[data-readout="depth"]');
  var readout = host.querySelector('.mlp-readout');
  ['pointerdown', 'keydown'].forEach(function(ev) {
    host.addEventListener(ev, function(e) { e.stopPropagation(); });
  });
  widthInput.addEventListener('input', function() {
    state.width = +widthInput.value; widthText.textContent = widthInput.value; draw();
  });
  depthInput.addEventListener('input', function() {
    state.depth = +depthInput.value; depthText.textContent = depthInput.value; draw();
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
  function dot(ctx, p, color, r) {
    ctx.beginPath(); ctx.arc(p[0], p[1], r || 4.5, 0, 6.2832);
    ctx.fillStyle = color; ctx.fill();
  }
  // Edge-normals of a regular S-gon centered on the disk. Each normal is one
  // hidden neuron's cut; a point is "inside" (predicted green) when it lies
  // on the interior side of ALL of them. RHO is the inradius (>= GREEN_R, so
  // the whole green disk is enclosed); the circumradius RHO/cos(pi/S) shrinks
  // toward RHO as S grows, which is exactly why more cuts hug the circle.
  var RHO = 0.205, THETA0 = -Math.PI / 2;
  function normals(S) {
    var ns = [];
    for (var k = 0; k < S; k++) {
      var phi = THETA0 + 2 * Math.PI * (k + 0.5) / S;
      ns.push([Math.cos(phi), Math.sin(phi)]);
    }
    return ns;
  }
  function insideRegion(p, ns) {
    for (var k = 0; k < ns.length; k++) {
      if ((p.x - CX) * ns[k][0] + (p.y - CY) * ns[k][1] > RHO) return false;
    }
    return true;
  }

  // Network diagram with one column per layer: input (x1, x2), `depth`
  // hidden columns of `width` neurons (capped at 6 drawn nodes), output.
  function drawNetwork(ctx, x0, x1, h) {
    var cols = state.depth + 2;
    var perLayer = Math.min(state.width, 6);
    var top = 56, bot = h - 64;
    function colX(i) { return x0 + (x1 - x0) * i / (cols - 1); }
    function ys(n) {
      if (n === 1) return [(top + bot) / 2];
      var a = []; for (var i = 0; i < n; i++) a.push(top + (bot - top) * i / (n - 1)); return a;
    }
    var layers = [{ x: colX(0), ys: ys(2), label: ['x₁', 'x₂'], color: SECONDARY }];
    for (var d = 0; d < state.depth; d++) layers.push({ x: colX(d + 1), ys: ys(perLayer), color: PRIMARY });
    layers.push({ x: colX(cols - 1), ys: ys(1), label: ['ŷ'], color: SECONDARY });

    ctx.strokeStyle = 'rgba(74,158,255,0.28)'; ctx.lineWidth = 1;
    for (var L = 0; L < layers.length - 1; L++) {
      var A = layers[L], B = layers[L + 1];
      A.ys.forEach(function(ay) {
        B.ys.forEach(function(by) {
          ctx.beginPath(); ctx.moveTo(A.x + 12, ay); ctx.lineTo(B.x - 12, by); ctx.stroke();
        });
      });
    }
    layers.forEach(function(Lr) {
      Lr.ys.forEach(function(yy, idx) {
        ctx.beginPath(); ctx.arc(Lr.x, yy, 12, 0, 6.2832);
        ctx.fillStyle = '#0d1225'; ctx.strokeStyle = Lr.color; ctx.lineWidth = 2; ctx.fill(); ctx.stroke();
        if (Lr.label) { ctx.fillStyle = TEXT; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.fillText(Lr.label[idx], Lr.x, yy + 4); }
      });
    });
    if (state.width > perLayer) {
      ctx.fillStyle = MUTED; ctx.font = '15px Inter, sans-serif'; ctx.textAlign = 'center';
      for (var d2 = 0; d2 < state.depth; d2++) ctx.fillText('⋮', colX(d2 + 1), bot + 18);
    }
    ctx.fillStyle = MUTED; ctx.font = '13px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(state.depth + ' hidden layer' + (state.depth === 1 ? '' : 's') +
                 ' × ' + state.width + ' neuron' + (state.width === 1 ? '' : 's'),
                 (x0 + x1) / 2, h - 8);
  }

  function draw() {
    var f = fitCanvas(canvas);
    if (!f) return;
    var ctx = f.ctx, w = f.w, h = f.h, pad = 30;
    ctx.clearRect(0, 0, w, h);
    var leftW = w * 0.60;                        // left region holds the plot
    // Square plot so the disk reads as a true circle, centered in the region.
    var side = Math.min(leftW - 2 * pad, h - 2 * pad);
    var ox = (leftW - side) / 2, oy = pad + (h - 2 * pad - side) / 2;
    function pmap(x, y) { return [ox + x * side, oy + (1 - y) * side]; }
    var S = state.width * state.depth;
    var ns = normals(Math.max(1, S));

    ctx.save();
    ctx.beginPath(); ctx.rect(ox, oy, side, side); ctx.clip();
    if (S >= 3) {
      var R = RHO / Math.cos(Math.PI / S);
      ctx.beginPath();
      for (var k = 0; k <= S; k++) {
        var th = THETA0 + 2 * Math.PI * k / S;
        var vp = pmap(CX + R * Math.cos(th), CY + R * Math.sin(th));
        if (k === 0) ctx.moveTo(vp[0], vp[1]); else ctx.lineTo(vp[0], vp[1]);
      }
      ctx.closePath();
      ctx.fillStyle = 'rgba(63,185,80,0.10)'; ctx.fill();
      ctx.strokeStyle = SECONDARY; ctx.lineWidth = 2.5; ctx.stroke();
    } else {
      // 1-2 cuts: an open half-plane or slab that cannot enclose a region.
      ctx.strokeStyle = SECONDARY; ctx.lineWidth = 2.5;
      ns.forEach(function(n) {
        var bx = CX + n[0] * RHO, by = CY + n[1] * RHO, tx = -n[1], ty = n[0];
        var a = pmap(bx + tx * 1.5, by + ty * 1.5);
        var b = pmap(bx - tx * 1.5, by - ty * 1.5);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      });
    }
    var correct = 0;
    data.forEach(function(p) {
      var pred = insideRegion(p, ns) ? 1 : 0;
      if (pred === p.cls) correct++;
      var q = pmap(p.x, p.y);
      dot(ctx, q, p.cls ? GREEN : RED, 4.2);
      if (pred !== p.cls) {                    // ring the points the cuts get wrong
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.arc(q[0], q[1], 6.6, 0, 6.2832); ctx.stroke();
      }
    });
    ctx.restore();

    ctx.strokeStyle = LINEC; ctx.lineWidth = 1;
    ctx.strokeRect(ox, oy, side, side);
    ctx.fillStyle = MUTED; ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'right'; ctx.fillText('x₁', ox + side - 4, oy + side - 6);
    ctx.textAlign = 'left'; ctx.fillText('x₂', ox + 4, oy + 12);

    drawNetwork(ctx, leftW + 20, w - 20, h);

    var acc = Math.round(100 * correct / data.length);
    if (S < 3) {
      readout.innerHTML = '<strong>' + S + '</strong> straight cut' + (S === 1 ? '' : 's') +
        ' (one per hidden unit) cannot enclose a region &mdash; accuracy <strong>' + acc +
        '%</strong>. Add neurons to wrap the class.';
    } else {
      var article = (S === 8 || S === 11 || S === 18) ? 'an' : 'a';
      readout.innerHTML = 'width × depth = <strong>' + S + '</strong> straight cuts form ' + article +
        ' <strong>' + S + '</strong>-sided region around the class &mdash; accuracy <strong>' + acc +
        '%</strong>. More cuts hug the circle tighter.';
    }
  }
  return { resize: draw };
};

// ---- Loss landscape widget: two weights, surface height, and update math ----
INTERACTIVE_WIDGETS.lossLandscape = function(host) {
  var TEXT = '#e8eaf0', MUTED = '#8892a4', PRIMARY = '#4a9eff',
      SECONDARY = '#f5a623', RED = '#e74c3c', GREEN = '#3fb950', LINEC = '#2a3450';
  var state = { w1: 0.2, w2: 0.2, eta: 0.8, lastStep: null };
  var data = [
    { x1: 0, x2: 0, y: 0 },
    { x1: 0, x2: 1, y: 1 },
    { x1: 1, x2: 0, y: 1 },
    { x1: 1, x2: 1, y: 1 }
  ];

  host.innerHTML =
    '<div class="land-widget">' +
      '<div class="land-main">' +
        '<canvas class="land-canvas"></canvas>' +
        '<canvas class="land-net"></canvas>' +
      '</div>' +
      '<div class="land-controls">' +
        '<div class="mlp-slider"><label>w₁</label><input data-k="w1" type="range" min="-4" max="4" step="0.1" value="0.2"><p data-r="w1">0.2</p></div>' +
        '<div class="mlp-slider"><label>w₂</label><input data-k="w2" type="range" min="-4" max="4" step="0.1" value="0.2"><p data-r="w2">0.2</p></div>' +
        '<div class="mlp-slider"><label>η</label><input data-k="eta" type="range" min="0.1" max="1.5" step="0.1" value="0.8"><p data-r="eta">0.8</p></div>' +
        '<button class="land-step">Step</button>' +
      '</div>' +
      '<p class="land-readout"></p>' +
    '</div>';
  var canvas = host.querySelector('.land-canvas');
  var net = host.querySelector('.land-net');
  var readout = host.querySelector('.land-readout');
  ['pointerdown', 'keydown'].forEach(function(ev) {
    host.addEventListener(ev, function(e) { e.stopPropagation(); });
  });
  host.querySelectorAll('input').forEach(function(inp) {
    inp.addEventListener('input', function() {
      state[inp.dataset.k] = parseFloat(inp.value);
      state.lastStep = null;
      host.querySelector('[data-r="' + inp.dataset.k + '"]').textContent = state[inp.dataset.k].toFixed(1);
      draw();
    });
  });
  host.querySelector('.land-step').addEventListener('click', function() {
    var oldW1 = state.w1, oldW2 = state.w2;
    var g = grad(state.w1, state.w2);
    state.w1 += -state.eta * g.g1;
    state.w2 += -state.eta * g.g2;
    state.lastStep = { oldW1: oldW1, oldW2: oldW2, g1: g.g1, g2: g.g2, eta: state.eta, newW1: state.w1, newW2: state.w2 };
    host.querySelector('[data-k="w1"]').value = state.w1;
    host.querySelector('[data-k="w2"]').value = state.w2;
    host.querySelector('[data-r="w1"]').textContent = state.w1.toFixed(2);
    host.querySelector('[data-r="w2"]').textContent = state.w2.toFixed(2);
    draw();
  });

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
  function loss(w1, w2) {
    var s = 0, eps = 1e-7, b = -0.5;
    data.forEach(function(p) {
      var pred = Math.max(eps, Math.min(1 - eps, sigmoid(w1 * p.x1 + w2 * p.x2 + b)));
      s += -(p.y * Math.log(pred) + (1 - p.y) * Math.log(1 - pred));
    });
    return s / data.length;
  }
  function grad(w1, w2) {
    var g1 = 0, g2 = 0, b = -0.5;
    data.forEach(function(p) {
      var e = sigmoid(w1 * p.x1 + w2 * p.x2 + b) - p.y;
      g1 += e * p.x1; g2 += e * p.x2;
    });
    return { g1: g1 / data.length, g2: g2 / data.length };
  }
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
  function project(w1, w2, z, W, H) {
    var sx = (w1 + 4) / 8, sy = (w2 + 4) / 8;
    var px = 64 + sx * (W - 150) + sy * 44;
    var py = H - 56 - sy * (H - 140) - z * 58;
    return [px, py];
  }
  function drawSurface(lastGrad) {
    var f = fitCanvas(canvas);
    if (!f) return;
    var ctx = f.ctx, W = f.w, H = f.h;
    ctx.clearRect(0, 0, W, H);
    for (var i = 0; i <= 28; i++) {
      var w1 = -4 + 8 * i / 28;
      ctx.beginPath();
      for (var j = 0; j <= 28; j++) {
        var w2 = -4 + 8 * j / 28, z = Math.min(3, loss(w1, w2));
        var p = project(w1, w2, z, W, H);
        if (j === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
      }
      ctx.strokeStyle = 'rgba(74,158,255,0.22)'; ctx.lineWidth = 1; ctx.stroke();
    }
    for (var j2 = 0; j2 <= 28; j2++) {
      var ww2 = -4 + 8 * j2 / 28;
      ctx.beginPath();
      for (var i2 = 0; i2 <= 28; i2++) {
        var ww1 = -4 + 8 * i2 / 28, zz = Math.min(3, loss(ww1, ww2));
        var pp = project(ww1, ww2, zz, W, H);
        if (i2 === 0) ctx.moveTo(pp[0], pp[1]); else ctx.lineTo(pp[0], pp[1]);
      }
      ctx.strokeStyle = 'rgba(245,166,35,0.18)'; ctx.lineWidth = 1; ctx.stroke();
    }
    var L = loss(state.w1, state.w2), pcur = project(state.w1, state.w2, Math.min(3, L), W, H);
    ctx.strokeStyle = SECONDARY; ctx.lineWidth = 2;
    var base = project(state.w1, state.w2, 0, W, H);
    ctx.beginPath(); ctx.moveTo(base[0], base[1]); ctx.lineTo(pcur[0], pcur[1]); ctx.stroke();
    ctx.beginPath(); ctx.arc(pcur[0], pcur[1], 7, 0, 6.2832); ctx.fillStyle = RED; ctx.fill();
    ctx.fillStyle = TEXT; ctx.font = '14px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('loss = ' + L.toFixed(3), pcur[0], pcur[1] - 12);
    ctx.fillStyle = MUTED; ctx.textAlign = 'left';
    ctx.fillText('w₁', W - 86, H - 42); ctx.fillText('w₂', 110, H - 20); ctx.fillText('L', 42, 54);
  }
  function drawNet() {
    var f = fitCanvas(net);
    if (!f) return;
    var ctx = f.ctx, W = f.w, H = f.h;
    ctx.clearRect(0, 0, W, H);
    function node(x, y, label, color) {
      ctx.beginPath(); ctx.arc(x, y, 19, 0, 6.2832);
      ctx.fillStyle = '#0d1225'; ctx.strokeStyle = color; ctx.lineWidth = 2.4; ctx.fill(); ctx.stroke();
      ctx.fillStyle = TEXT; ctx.font = '14px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.fillText(label, x, y + 5);
    }
    function edge(x1, y1, x2, y2, label, color) {
      ctx.strokeStyle = color; ctx.lineWidth = 2.3; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      ctx.fillStyle = color; ctx.font = '13px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.fillText(label, (x1 + x2) / 2, (y1 + y2) / 2 - 5);
    }
    node(42, H * 0.34, 'x₁', PRIMARY); node(42, H * 0.66, 'x₂', PRIMARY);
    node(W * 0.52, H * 0.50, 'σ', SECONDARY); node(W - 44, H * 0.50, 'ŷ', GREEN);
    edge(62, H * 0.34, W * 0.52 - 20, H * 0.50, 'w₁=' + state.w1.toFixed(2), PRIMARY);
    edge(62, H * 0.66, W * 0.52 - 20, H * 0.50, 'w₂=' + state.w2.toFixed(2), SECONDARY);
    edge(W * 0.52 + 20, H * 0.50, W - 64, H * 0.50, 'loss z', RED);
    ctx.fillStyle = MUTED; ctx.font = '13px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Coordinate: (' + state.w1.toFixed(2) + ', ' + state.w2.toFixed(2) + ')', W / 2, H - 28);
    ctx.fillText('Height: L = ' + loss(state.w1, state.w2).toFixed(3), W / 2, H - 10);
  }
  function draw(lastGrad) {
    drawSurface(lastGrad);
    drawNet();
    var g = grad(state.w1, state.w2);
    if (state.lastStep) {
      readout.innerHTML =
        'Last step: [' + state.lastStep.newW1.toFixed(2) + ', ' + state.lastStep.newW2.toFixed(2) +
        '] = [' + state.lastStep.oldW1.toFixed(2) + ', ' + state.lastStep.oldW2.toFixed(2) +
        '] + (-' + state.lastStep.eta.toFixed(1) + ' · [' +
        state.lastStep.g1.toFixed(3) + ', ' + state.lastStep.g2.toFixed(3) +
        ']). Current loss: <strong>' + loss(state.w1, state.w2).toFixed(3) + '</strong>.';
    } else {
      readout.innerHTML =
        'Gradient at current point: [' + g.g1.toFixed(3) + ', ' + g.g2.toFixed(3) +
        ']. Next update: w_new = [' + state.w1.toFixed(2) + ', ' + state.w2.toFixed(2) +
        '] + (-' + state.eta.toFixed(1) + ' · [' + g.g1.toFixed(3) + ', ' + g.g2.toFixed(3) + ']).';
    }
  }
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
