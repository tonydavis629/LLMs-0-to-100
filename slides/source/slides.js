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

  // Pre-activation (before ReLU) for each neuron: z_j = w1_j*x1 + w2_j*x2 + b_j
  function preact(p) {
    return state.neurons.map(function(n) {
      return n.w1 * p.x + n.w2 * p.y + n.b;
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
      // Shade each neuron's "active" half-plane (where z > 0, i.e. h > 0)
      state.neurons.forEach(function(n, idx) {
        var a = n.w1, b = n.w2, c = n.b;
        ctx.fillStyle = NEURON_COLORS[idx].replace(')', ',0.06)').replace('rgb', 'rgba').replace('#4a9eff', 'rgba(74,158,255,0.06)').replace('#f5a623', 'rgba(245,166,35,0.06)');
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

    // ---- Hidden space (right): folded points with fold lines ----
    var fh = fitCanvas(hidCanvas);
    if (!fh) { return; }
    var H = data.map(function(p) { return { h: hidden(p), z: preact(p), cls: p.cls }; });
    var k = state.k;
    var pts;
    if (k === 1) {
      // For 1 neuron: x = h1, y = z1 (pre-activation) so the fold at z=0 is visible
      pts = H.map(function(o) { return { x: o.h[0], y: o.z[0], cls: o.cls }; });
    } else {
      pts = H.map(function(o) { return { x: o.h[0], y: o.h[1], cls: o.cls }; });
    }
    var xs = pts.map(function(p) { return p.x; });
    var ys = pts.map(function(p) { return p.y; });
    function span(arr) {
      var lo = Math.min.apply(null, arr), hi = Math.max.apply(null, arr);
      if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
      var pd = (hi - lo) * 0.12;
      return [lo - pd, hi + pd];
    }
    var hxr = span(xs), hyr = span(ys);
    var pad2 = 26;
    var xlab = 'h₁', ylab = k === 1 ? 'z₁ (pre-ReLU)' : 'h₂';
    var mh = drawFrame(fh.ctx, fh.w, fh.h, pad2, hxr, hyr, xlab, ylab);
    var ctx2 = fh.ctx;

    // Logistic-regression separator in hidden space
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

    // Draw fold lines at h_j = 0 in hidden space, colored per neuron.
    // These show where the ReLU "folded" the space: points with h_j = 0
    // are on the inactive side of neuron j's line.
    state.neurons.forEach(function(n, idx) {
      ctx2.strokeStyle = NEURON_COLORS[idx];
      ctx2.lineWidth = 2;
      ctx2.setLineDash([6, 5]);
      // Fold for neuron idx: h_idx = 0 line
      if (idx === 0) {
        // h1 = 0 is the y-axis of the hidden space
        if (hxr[0] <= 0 && hxr[1] >= 0) {
          var f0 = mh(0, hyr[0]), f1 = mh(0, hyr[1]);
          ctx2.beginPath(); ctx2.moveTo(f0[0], f0[1]); ctx2.lineTo(f1[0], f1[1]); ctx2.stroke();
        }
      } else if (idx === 1 && !oneD) {
        // h2 = 0 is the x-axis of the hidden space
        if (hyr[0] <= 0 && hyr[1] >= 0) {
          var f0 = mh(hxr[0], 0), f1 = mh(hxr[1], 0);
          ctx2.beginPath(); ctx2.moveTo(f0[0], f0[1]); ctx2.lineTo(f1[0], f1[1]); ctx2.stroke();
        }
      }
      ctx2.setLineDash([]);
    });

    // Shade the "active" quadrant where all neurons are firing (h_j > 0)
    if (!oneD && hxr[0] < 0 && hxr[1] > 0 && hyr[0] < 0 && hyr[1] > 0) {
      var q0 = mh(0, 0), q1 = mh(hxr[1], 0), q2 = mh(hxr[1], hyr[1]), q3 = mh(0, hyr[1]);
      ctx2.fillStyle = 'rgba(74,158,255,0.06)';
      ctx2.beginPath(); ctx2.moveTo(q0[0], q0[1]); ctx2.lineTo(q1[0], q1[1]);
      ctx2.lineTo(q2[0], q2[1]); ctx2.lineTo(q3[0], q3[1]); ctx2.closePath(); ctx2.fill();
    }

    if (Math.abs(sw.wx) + Math.abs(sw.wy) > 1e-6) {
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

// ---- MLP boundary widget: trained MLP on moons/spiral datasets ----
// "Many lines make a curve": each hidden neuron contributes one linear
// piece to the decision boundary. Width x depth = total pieces.
// A real MLP is trained (via backprop) on the chosen dataset, and the
// learned boundary is rendered as a contour.
INTERACTIVE_WIDGETS.mlpBoundary = function(host) {
  var GREEN = '#3fb950', RED = '#e74c3c', MUTED = '#8892a4',
      LINEC = '#2a3450', TEXT = '#e8eaf0', PRIMARY = '#4a9eff', SECONDARY = '#f5a623';
  var state = { dataset: 'moons', width: 4, depth: 1, net: null, acc: 0 };
  var GRID = 40, TRAIN_STEPS = 800, LR = 0.4;

  function mulberry32(a) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function generateData(name) {
    var rnd = mulberry32(7);
    var pts = [];
    if (name === 'moons') {
      for (var i = 0; i < 50; i++) {
        var t = Math.PI * i / 50;
        pts.push({ x: Math.cos(t) + (rnd() - 0.5) * 0.18, y: Math.sin(t) + (rnd() - 0.5) * 0.18, cls: 1 });
        pts.push({ x: 1.1 - Math.cos(t) + (rnd() - 0.5) * 0.18, y: 0.4 - Math.sin(t) + (rnd() - 0.5) * 0.18, cls: 0 });
      }
    } else if (name === 'spiral') {
      for (var i = 0; i < 70; i++) {
        var t = 0.5 * Math.PI * i / 10;
        var r = 0.15 + 0.2 * t;
        pts.push({ x: r * Math.cos(t) + (rnd() - 0.5) * 0.12, y: r * Math.sin(t) + (rnd() - 0.5) * 0.12, cls: 1 });
        pts.push({ x: -r * Math.cos(t) + (rnd() - 0.5) * 0.12, y: -r * Math.sin(t) + (rnd() - 0.5) * 0.12, cls: 0 });
      }
    }
    var xs = pts.map(function(p) { return p.x; }), ys = pts.map(function(p) { return p.y; });
    var xmn = Math.min.apply(null, xs), xmx = Math.max.apply(null, xs);
    var ymn = Math.min.apply(null, ys), ymx = Math.max.apply(null, ys);
    var xr = xmx - xmn || 1, yr = ymx - ymn || 1;
    pts.forEach(function(p) {
      p.x = 0.04 + 0.92 * (p.x - xmn) / xr;
      p.y = 0.04 + 0.92 * (p.y - ymn) / yr;
    });
    return pts;
  }

  function zeros(n) { var a = []; for (var i = 0; i < n; i++) a.push(0); return a; }
  function randn() { return Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random()); }
  function xavier(outDim, inDim) {
    var std = Math.sqrt(2 / (inDim + outDim));
    var W = [], b = zeros(outDim);
    for (var i = 0; i < outDim; i++) {
      var row = [];
      for (var j = 0; j < inDim; j++) row.push(randn() * std);
      W.push(row);
    }
    return { W: W, b: b };
  }
  function matVec(W, x) { return W.map(function(row) { return row.reduce(function(s, wj, j) { return s + wj * x[j]; }, 0); }); }
  function vecAdd(a, b) { return a.map(function(v, i) { return v + b[i]; }); }

  function createNet(width, depth) {
    var layers = [xavier(width, 2)];
    for (var d = 1; d < depth; d++) layers.push(xavier(width, width));
    layers.push(xavier(1, width));
    return layers;
  }

  function predictOne(x, layers) {
    var a = x;
    for (var l = 0; l < layers.length - 1; l++) {
      a = vecAdd(matVec(layers[l].W, a), layers[l].b).map(function(v) { return Math.max(0, v); });
    }
    var z = vecAdd(matVec(layers[layers.length - 1].W, a), layers[layers.length - 1].b)[0];
    return 1 / (1 + Math.exp(-z));
  }

  function train(data, width, depth, steps, lr) {
    var layers = createNet(width, depth);
    var X = data.map(function(p) { return [p.x, p.y]; });
    var y = data.map(function(p) { return p.cls; });
    var N = data.length;
    for (var step = 0; step < steps; step++) {
      var dW = layers.map(function(L) { return L.W.map(function(row) { return row.map(function() { return 0; }); }); });
      var db = layers.map(function(L) { return L.b.map(function() { return 0; }); });
      for (var i = 0; i < N; i++) {
        var as = [X[i]], zs = [];
        for (var l = 0; l < layers.length; l++) {
          var z = vecAdd(matVec(layers[l].W, as[l]), layers[l].b);
          zs.push(z);
          if (l < layers.length - 1) as.push(z.map(function(v) { return Math.max(0, v); }));
          else as.push(z);
        }
        var pred = 1 / (1 + Math.exp(-as[as.length - 1][0]));
        var delta = [pred - y[i]];
        for (var l = layers.length - 1; l >= 0; l--) {
          var aPrev = as[l];
          for (var j = 0; j < layers[l].W.length; j++) {
            for (var k = 0; k < layers[l].W[j].length; k++) {
              dW[l][j][k] += delta[j] * aPrev[k];
            }
            db[l][j] += delta[j];
          }
          if (l > 0) {
            var W = layers[l].W;
            var newDelta = [];
            for (var k = 0; k < layers[l - 1].b.length; k++) {
              var s = 0;
              for (var j = 0; j < W.length; j++) s += W[j][k] * delta[j];
              newDelta.push(s * (zs[l - 1][k] > 0 ? 1 : 0));
            }
            delta = newDelta;
          }
        }
      }
      for (var l = 0; l < layers.length; l++) {
        for (var j = 0; j < layers[l].W.length; j++) {
          for (var k = 0; k < layers[l].W[j].length; k++) {
            layers[l].W[j][k] -= lr * dW[l][j][k] / N;
          }
          layers[l].b[j] -= lr * db[l][j] / N;
        }
      }
    }
    return layers;
  }

  function computeAccuracy(data, layers) {
    var correct = 0;
    for (var i = 0; i < data.length; i++) {
      var p = predictOne([data[i].x, data[i].y], layers);
      if ((p >= 0.5 ? 1 : 0) === data[i].cls) correct++;
    }
    return correct / data.length;
  }

  host.innerHTML =
    '<div class="mlp-widget">' +
      '<div class="mlp-canvas-wrap"><canvas class="mlp-canvas"></canvas></div>' +
      '<div class="mlp-controls">' +
        '<div class="expl-datasets"></div>' +
        '<div class="mlp-slider"><label>Width</label><input type="range" min="1" max="8" step="1" value="4"><p data-readout="width">4</p></div>' +
        '<div class="mlp-slider"><label>Depth</label><input type="range" min="1" max="4" step="1" value="1"><p data-readout="depth">1</p></div>' +
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
  datasets.forEach(function(name) {
    var btn = document.createElement('button');
    btn.className = 'expl-ds-btn';
    btn.textContent = name.charAt(0).toUpperCase() + name.slice(1);
    btn.dataset.ds = name;
    btn.addEventListener('click', function() {
      state.dataset = name; state.net = null;
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
      ' = <strong>' + totalUnits + '</strong> straight cut' + (totalUnits === 1 ? '' : 's') +
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

  function draw() {
    var f = fitCanvas(canvas);
    if (!f) return;
    var ctx = f.ctx, W = f.w, H = f.h;
    ctx.clearRect(0, 0, W, H);

    var data = generateData(state.dataset);
    if (!state.net) {
      state.net = train(data, state.width, state.depth, TRAIN_STEPS, LR);
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

    // 0.5 contour
    var CRES = 28;
    var cvals = [];
    for (var ci = 0; ci <= CRES; ci++) {
      cvals[ci] = [];
      for (var cj = 0; cj <= CRES; cj++) {
        cvals[ci][cj] = predictOne([ci / CRES, cj / CRES], state.net);
      }
    }
    ctx.strokeStyle = SECONDARY; ctx.lineWidth = 2.2; ctx.beginPath();
    for (var i = 0; i < CRES; i++) {
      for (var j = 0; j < CRES; j++) {
        var cell = [
          [i, j, cvals[i][j]],
          [i + 1, j, cvals[i + 1][j]],
          [i, j + 1, cvals[i][j + 1]],
          [i + 1, j + 1, cvals[i + 1][j + 1]]
        ];
        for (var e = 0; e < 4; e++) {
          var a = cell[e], b = cell[(e + 1) % 4];
          if ((a[2] - 0.5) * (b[2] - 0.5) < 0) {
            var t = (0.5 - a[2]) / (b[2] - a[2]);
            var gx = a[0] + t * (b[0] - a[0]);
            var gy = a[1] + t * (b[1] - a[1]);
            var sx = ox + (gx / CRES) * side;
            var sy = oy + (1 - gy / CRES) * side;
            ctx.moveTo(sx, sy);
            for (var e2 = e + 1; e2 < 4; e2++) {
              var a2 = cell[e2], b2 = cell[(e2 + 1) % 4];
              if ((a2[2] - 0.5) * (b2[2] - 0.5) < 0) {
                var t2 = (0.5 - a2[2]) / (b2[2] - a2[2]);
                var gx2 = a2[0] + t2 * (b2[0] - a2[0]);
                var gy2 = a2[1] + t2 * (b2[1] - a2[1]);
                var sx2 = ox + (gx2 / CRES) * side;
                var sy2 = oy + (1 - gy2 / CRES) * side;
                ctx.lineTo(sx2, sy2);
                break;
              }
            }
          }
        }
      }
    }
    ctx.stroke();

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

// ---- Boundary Explorer: real trained MLP, learned decision boundary ----
INTERACTIVE_WIDGETS.boundaryExplorer = function(host) {
  var GREEN = '#3fb950', RED = '#e74c3c', MUTED = '#8892a4',
      LINEC = '#2a3450', TEXT = '#e8eaf0', PRIMARY = '#4a9eff', SECONDARY = '#f5a623';
  var mode = host.dataset.mode || 'width';
  var state = { dataset: 'linear', width: 1, depth: 1, net: null, acc: 0 };
  var GRID = 45, TRAIN_STEPS = 1200, LR0 = 0.5, LR_DECAY = 0.98, RESTARTS = 3;

  function mulberry32(a) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

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
      clusters.forEach(function(c) {
        for (var i = 0; i < 20; i++) {
          pts.push({ x: c[0] + (rnd() - 0.5) * 0.6, y: c[1] + (rnd() - 0.5) * 0.6, cls: c[2] });
        }
      });
    } else if (name === 'moons') {
      for (var i = 0; i < 50; i++) {
        var t = Math.PI * i / 50;
        pts.push({ x: Math.cos(t) + (rnd() - 0.5) * 0.18, y: Math.sin(t) + (rnd() - 0.5) * 0.18, cls: 1 });
        pts.push({ x: 1.1 - Math.cos(t) + (rnd() - 0.5) * 0.18, y: 0.4 - Math.sin(t) + (rnd() - 0.5) * 0.18, cls: 0 });
      }
    } else if (name === 'spiral') {
      for (var i = 0; i < 70; i++) {
        var t = 0.5 * Math.PI * i / 10;
        var r = 0.15 + 0.2 * t;
        pts.push({ x: r * Math.cos(t) + (rnd() - 0.5) * 0.12, y: r * Math.sin(t) + (rnd() - 0.5) * 0.12, cls: 1 });
        pts.push({ x: -r * Math.cos(t) + (rnd() - 0.5) * 0.12, y: -r * Math.sin(t) + (rnd() - 0.5) * 0.12, cls: 0 });
      }
    }
    var xs = pts.map(function(p) { return p.x; }), ys = pts.map(function(p) { return p.y; });
    var xmn = Math.min.apply(null, xs), xmx = Math.max.apply(null, xs);
    var ymn = Math.min.apply(null, ys), ymx = Math.max.apply(null, ys);
    var xr = xmx - xmn || 1, yr = ymx - ymn || 1;
    pts.forEach(function(p) {
      p.x = 0.04 + 0.92 * (p.x - xmn) / xr;
      p.y = 0.04 + 0.92 * (p.y - ymn) / yr;
    });
    return pts;
  }

  function zeros(n) { var a = []; for (var i = 0; i < n; i++) a.push(0); return a; }
  function randn() { return Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random()); }
  function xavier(outDim, inDim) {
    var std = Math.sqrt(2 / (inDim + outDim));
    var W = [], b = zeros(outDim);
    for (var i = 0; i < outDim; i++) {
      var row = [];
      for (var j = 0; j < inDim; j++) row.push(randn() * std);
      W.push(row);
    }
    return { W: W, b: b };
  }
  function matVec(W, x) { return W.map(function(row) { return row.reduce(function(s, wj, j) { return s + wj * x[j]; }, 0); }); }
  function vecAdd(a, b) { return a.map(function(v, i) { return v + b[i]; }); }

  function createNet(width, depth) {
    var layers = [xavier(width, 2)];
    for (var d = 1; d < depth; d++) layers.push(xavier(width, width));
    layers.push(xavier(1, width));
    return layers;
  }

  function predictOne(x, layers) {
    var a = x;
    for (var l = 0; l < layers.length - 1; l++) {
      a = vecAdd(matVec(layers[l].W, a), layers[l].b).map(function(v) { return Math.max(0, v); });
    }
    var z = vecAdd(matVec(layers[layers.length - 1].W, a), layers[layers.length - 1].b)[0];
    return 1 / (1 + Math.exp(-z));
  }

  function train(data, width, depth, steps) {
    var X = data.map(function(p) { return [p.x, p.y]; });
    var y = data.map(function(p) { return p.cls; });
    var N = data.length;

    function run(initLayers) {
      var layers = initLayers;
      var lr = LR0;
      for (var step = 0; step < steps; step++) {
        var dW = layers.map(function(L) { return L.W.map(function(row) { return row.map(function() { return 0; }); }); });
        var db = layers.map(function(L) { return L.b.map(function() { return 0; }); });

        for (var i = 0; i < N; i++) {
          var as2 = [X[i]], zs2 = [];
          for (var l = 0; l < layers.length; l++) {
            var z = vecAdd(matVec(layers[l].W, as2[l]), layers[l].b);
            zs2.push(z);
            if (l < layers.length - 1) as2.push(z.map(function(v) { return Math.max(0, v); }));
            else as2.push(z);
          }
          var pred = 1 / (1 + Math.exp(-as2[as2.length - 1][0]));

          var delta = [pred - y[i]];
          for (var l = layers.length - 1; l >= 0; l--) {
            var aPrev = as2[l];
            for (var j = 0; j < layers[l].W.length; j++) {
              for (var k = 0; k < layers[l].W[j].length; k++) {
                dW[l][j][k] += delta[j] * aPrev[k];
              }
              db[l][j] += delta[j];
            }
            if (l > 0) {
              var W = layers[l].W;
              var newDelta = [];
              for (var k = 0; k < layers[l - 1].b.length; k++) {
                var s = 0;
                for (var j = 0; j < W.length; j++) s += W[j][k] * delta[j];
                newDelta.push(s * (zs2[l - 1][k] > 0 ? 1 : 0));
              }
              delta = newDelta;
            }
          }
        }

        for (var l = 0; l < layers.length; l++) {
          for (var j = 0; j < layers[l].W.length; j++) {
            for (var k = 0; k < layers[l].W[j].length; k++) {
              layers[l].W[j][k] -= lr * dW[l][j][k] / N;
            }
            layers[l].b[j] -= lr * db[l][j] / N;
          }
        }
        lr *= LR_DECAY;
      }
      return layers;
    }

    var best = null, bestAcc = -1;
    for (var r = 0; r < RESTARTS; r++) {
      var attempt = run(createNet(width, depth));
      var acc = computeAccuracy(data, attempt);
      if (acc > bestAcc) { bestAcc = acc; best = attempt; }
    }
    return best;
  }

  function computeAccuracy(data, layers) {
    var correct = 0;
    for (var i = 0; i < data.length; i++) {
      var p = predictOne([data[i].x, data[i].y], layers);
      if ((p >= 0.5 ? 1 : 0) === data[i].cls) correct++;
    }
    return correct / data.length;
  }

  host.innerHTML =
    '<div class="mlp-widget">' +
      '<div class="mlp-canvas-wrap"><canvas class="mlp-canvas"></canvas></div>' +
      '<div class="mlp-controls">' +
        '<div class="expl-datasets"></div>' +
        '<div class="expl-arch"></div>' +
        '<p class="mlp-readout"></p>' +
      '</div>' +
    '</div>';
  var canvas = host.querySelector('.mlp-canvas');
  var readout = host.querySelector('.mlp-readout');
  var dsEl = host.querySelector('.expl-datasets');
  var archEl = host.querySelector('.expl-arch');

  ['pointerdown', 'keydown'].forEach(function(ev) {
    host.addEventListener(ev, function(e) { e.stopPropagation(); });
  });

  var datasets = ['linear', 'xor', 'moons', 'spiral'];
  datasets.forEach(function(name) {
    var btn = document.createElement('button');
    btn.className = 'expl-ds-btn';
    btn.textContent = name.charAt(0).toUpperCase() + name.slice(1);
    btn.dataset.ds = name;
    btn.addEventListener('click', function() {
      state.dataset = name; state.net = null;
      updateUI(); draw();
    });
    dsEl.appendChild(btn);
  });

  var addNeuronBtn = document.createElement('button');
  addNeuronBtn.className = 'expl-arch-btn'; addNeuronBtn.textContent = 'Add neuron';
  addNeuronBtn.addEventListener('click', function() { state.width++; state.net = null; updateUI(); draw(); });
  var addLayerBtn = document.createElement('button');
  addLayerBtn.className = 'expl-arch-btn'; addLayerBtn.textContent = 'Add layer';
  addLayerBtn.addEventListener('click', function() { state.depth++; state.net = null; updateUI(); draw(); });
  var resetBtn = document.createElement('button');
  resetBtn.className = 'expl-arch-btn expl-reset'; resetBtn.textContent = 'Reset';
  resetBtn.addEventListener('click', function() {
    state.width = 1; state.depth = 1; state.dataset = 'linear'; state.net = null; updateUI(); draw();
  });
  archEl.appendChild(addNeuronBtn);
  archEl.appendChild(addLayerBtn);
  archEl.appendChild(resetBtn);

  function updateUI() {
    host.querySelectorAll('.expl-ds-btn').forEach(function(b) {
      b.classList.toggle('active', b.dataset.ds === state.dataset);
    });
    var totalUnits = state.width * state.depth;
    var boundaryType = totalUnits === 1 ? 'a straight line' : 'a non-linear region';
    readout.innerHTML = 'Architecture: <strong>' + state.depth + '</strong> hidden layer' +
      (state.depth === 1 ? '' : 's') + ' &times; <strong>' + state.width + '</strong> neuron' +
      (state.width === 1 ? '' : 's') + ' — ' + boundaryType +
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

  function draw() {
    var f = fitCanvas(canvas);
    if (!f) return;
    var ctx = f.ctx, W = f.w, H = f.h;
    ctx.clearRect(0, 0, W, H);

    var data = generateData(state.dataset);
    if (!state.net) {
      state.net = train(data, state.width, state.depth, TRAIN_STEPS);
      state.acc = computeAccuracy(data, state.net);
      updateUI();
    }

    var pad = 18;
    var rightPanel = 170;
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

    // 0.5 contour via marching squares (correct traversal order)
    var CRES = 32;
    var cvals = [];
    for (var ci = 0; ci <= CRES; ci++) {
      cvals[ci] = [];
      for (var cj = 0; cj <= CRES; cj++) {
        cvals[ci][cj] = predictOne([ci / CRES, cj / CRES], state.net);
      }
    }
    ctx.strokeStyle = SECONDARY; ctx.lineWidth = 2.2; ctx.beginPath();
    for (var i = 0; i < CRES; i++) {
      for (var j = 0; j < CRES; j++) {
        // Corners in CCW edge order: bl, br, tr, tl
        // Edges: 0-1 bottom, 1-2 right, 2-3 top, 3-0 left
        var bl = [i,   j,   cvals[i][j]],
            br = [i+1, j,   cvals[i+1][j]],
            tr = [i+1, j+1, cvals[i+1][j+1]],
            tl = [i,   j+1, cvals[i][j+1]];
        var cell = [bl, br, tr, tl];
        for (var e = 0; e < 4; e++) {
          var a = cell[e], b = cell[(e + 1) % 4];
          if ((a[2] - 0.5) * (b[2] - 0.5) < 0) {
            var t = (0.5 - a[2]) / (b[2] - a[2]);
            var gx = a[0] + t * (b[0] - a[0]);
            var gy = a[1] + t * (b[1] - a[1]);
            var sx = ox + (gx / CRES) * side;
            var sy = oy + (1 - gy / CRES) * side;
            ctx.moveTo(sx, sy);
            for (var e2 = e + 1; e2 < 4; e2++) {
              var a2 = cell[e2], b2 = cell[(e2 + 1) % 4];
              if ((a2[2] - 0.5) * (b2[2] - 0.5) < 0) {
                var t2 = (0.5 - a2[2]) / (b2[2] - a2[2]);
                var gx2 = a2[0] + t2 * (b2[0] - a2[0]);
                var gy2 = a2[1] + t2 * (b2[1] - a2[1]);
                var sx2 = ox + (gx2 / CRES) * side;
                var sy2 = oy + (1 - gy2 / CRES) * side;
                ctx.lineTo(sx2, sy2);
                break;
              }
            }
          }
        }
      }
    }
    ctx.stroke();

    // Data scatter
    data.forEach(function(p) {
      var px = ox + p.x * side, py = oy + (1 - p.y) * side;
      ctx.beginPath(); ctx.arc(px, py, 4.5, 0, 6.2832);
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
      if (state.width > perLayer) {
        ctx.fillStyle = MUTED; ctx.font = '13px Inter, sans-serif'; ctx.textAlign = 'center';
        for (var d2 = 0; d2 < state.depth; d2++) ctx.fillText('⋮', colX(d2 + 1), bot + 12);
      }
    }
  }

  // Apply mode presets
  if (mode === 'depth') {
    state.width = 2; state.depth = 1; state.dataset = 'xor';
  }
  updateUI();
  return { resize: draw };
};
// ---- Single Perceptron widget: adjustable line on data, no training ----
INTERACTIVE_WIDGETS.singlePerceptron = function(host) {
  var GREEN = '#3fb950', RED = '#e74c3c', MUTED = '#8892a4',
      LINEC = '#2a3450', TEXT = '#e8eaf0', PRIMARY = '#4a9eff', SECONDARY = '#f5a623';
  var state = { dataset: 'linear', w1: 0.5, w2: 0.5, b: -0.5 };

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
    } else if (name === 'moons') {
      for (var i = 0; i < 35; i++) {
        var t = Math.PI * i / 35;
        pts.push({ x: Math.cos(t) + (rnd() - 0.5) * 0.18, y: Math.sin(t) + (rnd() - 0.5) * 0.18, cls: 1 });
        pts.push({ x: 1.1 - Math.cos(t) + (rnd() - 0.5) * 0.18, y: 0.4 - Math.sin(t) + (rnd() - 0.5) * 0.18, cls: 0 });
      }
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

  var datasets = ['linear', 'xor', 'moons'];
  datasets.forEach(function(name) {
    var btn = document.createElement('button');
    btn.className = 'perc-ds-btn';
    btn.textContent = name.charAt(0).toUpperCase() + name.slice(1);
    btn.dataset.ds = name;
    btn.addEventListener('click', function() {
      state.dataset = name; updateUI(); draw();
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
    var z = state.w1 * x + state.w2 * y + state.b;
    return z >= 0 ? 1 : 0;
  }

  function draw() {
    var f = fitCanvas(canvas);
    if (!f) return;
    var ctx = f.ctx, W = f.w, H = f.h;
    ctx.clearRect(0, 0, W, H);

    var pad = 18;
    var side = Math.min(W - 2 * pad, H - 2 * pad);
    var ox = pad + (W - 2 * pad - side) / 2;
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

    // Frame
    ctx.strokeStyle = LINEC; ctx.lineWidth = 1;
    ctx.strokeRect(ox, oy, side, side);

    var correct = 0;
    data.forEach(function(p) { if (predict(p.x, p.y) === p.cls) correct++; });
    var acc = data.length ? correct / data.length : 0;
    readout.innerHTML =
      'Classifier: ' + state.w1.toFixed(1) + 'x₁ + ' + state.w2.toFixed(1) + 'x₂ + ' + state.b.toFixed(1) + ' = 0' +
      ' &mdash; Accuracy: <strong>' + (acc * 100).toFixed(0) + '%</strong>';
  }

  buildSliders();
  updateUI();
  return { resize: draw };
};

// ---- Loss landscape widget: 3D draggable surface with height coloring ----
INTERACTIVE_WIDGETS.lossLandscape = function(host) {
  var TEXT = '#e8eaf0', MUTED = '#8892a4', PRIMARY = '#4a9eff',
      SECONDARY = '#f5a623', RED = '#e74c3c', LINEC = '#2a3450';
  var state = { w1: -1.0, w2: 2.0, eta: 0.5, lastStep: null,
                az: -0.8, el: 0.65, zoom: 1.0, panX: 0, panY: 0,
                dragging: false, dragType: null, lastMX: 0, lastMY: 0 };
  var WMIN = -3, WMAX = 5;
  var INIT_W1 = -1.0, INIT_W2 = 2.0, INIT_ETA = 0.5;
  var INIT_AZ = -0.8, INIT_EL = 0.65, INIT_ZOOM = 1.0;

  function loss(w1, w2) {
    return 0.5 * ((w1 - 1.5) * (w1 - 1.5) + (w2 + 1.0) * (w2 + 1.0));
  }
  function grad(w1, w2) {
    return { g1: w1 - 1.5, g2: w2 + 1.0 };
  }

  host.innerHTML =
    '<div class="land-widget">' +
      '<div class="land-main">' +
        '<canvas class="land-canvas"></canvas>' +
      '</div>' +
      '<div class="land-controls">' +
        '<div class="mlp-slider"><label>w₁</label><input data-k="w1" type="range" min="-3" max="5" step="0.1" value="-1.0"><p data-r="w1">-1.0</p></div>' +
        '<div class="mlp-slider"><label>w₂</label><input data-k="w2" type="range" min="-3" max="5" step="0.1" value="2.0"><p data-r="w2">2.0</p></div>' +
        '<div class="mlp-slider"><label>η</label><input data-k="eta" type="range" min="0.1" max="1.5" step="0.1" value="0.5"><p data-r="eta">0.5</p></div>' +
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

  host.querySelectorAll('input').forEach(function(inp) {
    inp.addEventListener('input', function() {
      state[inp.dataset.k] = parseFloat(inp.value);
      state.lastStep = null;
      host.querySelector('[data-r="' + inp.dataset.k + '"]').textContent = state[inp.dataset.k].toFixed(1);
      draw();
    });
  });
  host.querySelector('.land-step').addEventListener('click', function() {
    var g = grad(state.w1, state.w2);
    var oldW1 = state.w1, oldW2 = state.w2;
    state.w1 += -state.eta * g.g1;
    state.w2 += -state.eta * g.g2;
    state.lastStep = { oldW1: oldW1, oldW2: oldW2, g1: g.g1, g2: g.g2, eta: state.eta, newW1: state.w1, newW2: state.w2 };
    host.querySelector('[data-k="w1"]').value = state.w1;
    host.querySelector('[data-k="w2"]').value = state.w2;
    host.querySelector('[data-r="w1"]').textContent = state.w1.toFixed(2);
    host.querySelector('[data-r="w2"]').textContent = state.w2.toFixed(2);
    draw();
  });
  resetBtn.addEventListener('click', function() {
    state.w1 = INIT_W1; state.w2 = INIT_W2; state.eta = INIT_ETA;
    state.lastStep = null;
    state.az = INIT_AZ; state.el = INIT_EL; state.zoom = INIT_ZOOM;
    state.panX = 0; state.panY = 0;
    host.querySelector('[data-k="w1"]').value = state.w1;
    host.querySelector('[data-k="w2"]').value = state.w2;
    host.querySelector('[data-k="eta"]').value = state.eta;
    host.querySelector('[data-r="w1"]').textContent = state.w1.toFixed(1);
    host.querySelector('[data-r="w2"]').textContent = state.w2.toFixed(1);
    host.querySelector('[data-r="eta"]').textContent = state.eta.toFixed(1);
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
    var zr = z;
    var yr2 = yr * ce - zr * se;
    var zr2 = yr * se + zr * ce;
    var scale = Math.min(W, H) * 0.16 * state.zoom;
    var px = W * 0.5 + state.panX + xr * scale;
    var py = H * 0.55 + state.panY - yr2 * scale * 0.72;
    return { x: px, y: py, depth: zr2 };
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

    var g = grad(state.w1, state.w2);
    if (state.lastStep) {
      readout.innerHTML =
        'Step: [' + state.lastStep.newW1.toFixed(2) + ', ' + state.lastStep.newW2.toFixed(2) +
        '] = [' + state.lastStep.oldW1.toFixed(2) + ', ' + state.lastStep.oldW2.toFixed(2) +
        '] + (-' + state.lastStep.eta.toFixed(1) + ' &middot; [' +
        state.lastStep.g1.toFixed(3) + ', ' + state.lastStep.g2.toFixed(3) +
        ']). Current loss: <strong>' + curL.toFixed(3) + '</strong>.';
    } else {
      readout.innerHTML =
        'Gradient: [' + g.g1.toFixed(3) + ', ' + g.g2.toFixed(3) +
        ']. Next: w_new = [' + state.w1.toFixed(2) + ', ' + state.w2.toFixed(2) +
        '] + (-' + state.eta.toFixed(1) + ' &middot; [' + g.g1.toFixed(3) + ', ' + g.g2.toFixed(3) + ']).';
    }
  }
  return { resize: draw };
};

// ---- Adam Landscape widget: loss surface + Adam terms (m, v, m_hat, v_hat) ----
INTERACTIVE_WIDGETS.adamLandscape = function(host) {
  var TEXT = '#e8eaf0', MUTED = '#8892a4', PRIMARY = '#4a9eff',
      SECONDARY = '#f5a623', RED = '#e74c3c', LINEC = '#2a3450', GREEN = '#3fb950';
  var state = { w1: -1.0, w2: 2.0, eta: 0.3, beta1: 0.9, beta2: 0.999,
                m1: 0, m2: 0, v1: 0, v2: 0, t: 0,
                az: -0.8, el: 0.65, zoom: 1.0, panX: 0, panY: 0,
                dragging: false, dragType: null, lastMX: 0, lastMY: 0 };
  var WMIN = -3, WMAX = 5;

  function loss(w1, w2) {
    return 0.5 * ((w1 - 1.5) * (w1 - 1.5) + (w2 + 1.0) * (w2 + 1.0));
  }
  function grad(w1, w2) {
    return { g1: w1 - 1.5, g2: w2 + 1.0 };
  }

  host.innerHTML =
    '<div class="land-widget">' +
      '<div class="land-main">' +
        '<canvas class="land-canvas"></canvas>' +
      '</div>' +
      '<div class="land-controls">' +
        '<div class="mlp-slider"><label>η</label><input data-k="eta" type="range" min="0.05" max="1.5" step="0.05" value="0.3"><p data-r="eta">0.3</p></div>' +
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

  host.querySelectorAll('input').forEach(function(inp) {
    inp.addEventListener('input', function() {
      state[inp.dataset.k] = parseFloat(inp.value);
      host.querySelector('[data-r="' + inp.dataset.k + '"]').textContent = state[inp.dataset.k].toFixed(inp.dataset.k === 'beta2' ? 3 : 2);
      draw();
    });
  });

  host.querySelector('.land-step').addEventListener('click', function() {
    var g = grad(state.w1, state.w2);
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
    draw();
  });

  host.querySelector('.land-reset-btn').addEventListener('click', function() {
    state.w1 = -1.0; state.w2 = 2.0; state.eta = 0.3;
    state.beta1 = 0.9; state.beta2 = 0.999;
    state.m1 = 0; state.m2 = 0; state.v1 = 0; state.v2 = 0; state.t = 0;
    state.az = -0.8; state.el = 0.65; state.zoom = 1.0; state.panX = 0; state.panY = 0;
    host.querySelector('[data-k="eta"]').value = state.eta;
    host.querySelector('[data-k="beta1"]').value = state.beta1;
    host.querySelector('[data-k="beta2"]').value = state.beta2;
    host.querySelector('[data-r="eta"]').textContent = state.eta.toFixed(2);
    host.querySelector('[data-r="beta1"]').textContent = state.beta1.toFixed(2);
    host.querySelector('[data-r="beta2"]').textContent = state.beta2.toFixed(3);
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
    var yr2 = yr * ce - z * se;
    var zr2 = yr * se + z * ce;
    var scale = Math.min(W, H) * 0.16 * state.zoom;
    return { x: W * 0.5 + state.panX + xr * scale, y: H * 0.55 + state.panY - yr2 * scale * 0.72, depth: zr2 };
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

    var curL = loss(state.w1, state.w2);
    var curP = project3D(state.w1, state.w2, curL, W, H);
    ctx.shadowColor = RED; ctx.shadowBlur = 14;
    ctx.beginPath(); ctx.arc(curP.x, curP.y, 8, 0, 6.2832);
    ctx.fillStyle = RED; ctx.fill(); ctx.shadowBlur = 0;

    ctx.fillStyle = MUTED; ctx.font = '12px Inter, sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('Drag: rotate | Shift+drag: pan | Scroll: zoom', 10, H - 10);

    var g = grad(state.w1, state.w2);
    var eps = 1e-8;
    if (state.t > 0) {
      var m1h = state.m1 / (1 - Math.pow(state.beta1, state.t));
      var m2h = state.m2 / (1 - Math.pow(state.beta1, state.t));
      var v1h = state.v1 / (1 - Math.pow(state.beta2, state.t));
      var v2h = state.v2 / (1 - Math.pow(state.beta2, state.t));
      readout.innerHTML =
        'Step ' + state.t + ': w=[' + state.w1.toFixed(2) + ', ' + state.w2.toFixed(2) +
        '] loss=<strong>' + curL.toFixed(3) + '</strong> | ' +
        '<span style="color:' + PRIMARY + '">m̂</span>=[' + m1h.toFixed(3) + ', ' + m2h.toFixed(3) + '] ' +
        '<span style="color:' + SECONDARY + '">v̂</span>=[' + v1h.toFixed(3) + ', ' + v2h.toFixed(3) + '] ' +
        '| <span style="color:' + GREEN + '">step</span>=[' +
        (state.eta * m1h / (Math.sqrt(v1h) + eps)).toFixed(4) + ', ' +
        (state.eta * m2h / (Math.sqrt(v2h) + eps)).toFixed(4) + ']';
    } else {
      readout.innerHTML =
        'w=[' + state.w1.toFixed(2) + ', ' + state.w2.toFixed(2) +
        '] loss=<strong>' + curL.toFixed(3) + '</strong> | Press Step to begin Adam descent.';
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
