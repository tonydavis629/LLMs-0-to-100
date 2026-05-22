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
      advanceManimStepper(scene);
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

// Intercept right arrow / space / enter on manim slides to step through sections
document.addEventListener('keydown', function(e) {
  if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'Enter') {
    var currentSlide = Reveal.getCurrentSlide();
    var stepper = currentSlide ? currentSlide.querySelector('.manim-stepper') : null;
    if (stepper) {
      var scene = stepper.getAttribute('data-manim-scene');
      if (advanceManimStepper(scene)) {
        e.stopImmediatePropagation();
        e.preventDefault();
      }
    }
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
      LINEC = '#2a3450', TEXT = '#e8eaf0', SECON = '#f5a623';
  var NEURON_COLORS = ['#4a9eff', '#f5a623', '#3fb950'];

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
      { w1: -1, w2: 1, b: 0.5 },
      { w1: 1, w2: 1, b: -1.0 }
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
        '<figure class="fold-view"><figcaption>Hidden space &mdash; h = ReLU(Wx + b)</figcaption>' +
          '<canvas class="fold-canvas" data-pane="hidden"></canvas></figure>' +
      '</div>' +
      '<div class="fold-controls">' +
        '<div class="fold-modes">' +
          '<button data-neurons="1">1 neuron</button>' +
          '<button data-neurons="2">2 neurons</button>' +
          '<button data-neurons="3">3 neurons</button>' +
          '<button class="fold-reset">Reset to XOR solution</button>' +
        '</div>' +
        '<div class="fold-sliders"></div>' +
        '<p class="fold-readout"></p>' +
      '</div>' +
    '</div>';

  var inCanvas = host.querySelector('[data-pane="input"]');
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
      });
      data.forEach(function(p) {
        var q = m(p.x, p.y);
        dot(ctx, q[0], q[1], p.cls ? GREEN : RED);
      });
      ctx.restore();
    }

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

    // Pocket perceptron (Rosenblatt): guaranteed to find a separating
    // line when one exists, and keeps the best line seen otherwise --
    // exactly what "linearly separable" means, and on-topic for this
    // module. Updates are scaled by `sc` only for numeric stability.
    var oneD = (k === 1);
    var sc = 1;
    pts.forEach(function(p) {
      sc = Math.max(sc, Math.abs(p.x), oneD ? 0 : Math.abs(p.y));
    });
    function lineAcc(W) {
      var c = 0;
      pts.forEach(function(p) {
        var s = (W.wx * p.x + (oneD ? 0 : W.wy * p.y) + W.b) >= 0 ? 1 : 0;
        if (s === p.cls) c++;
      });
      return c;
    }
    var cur = { wx: 0, wy: 0, b: 0 };
    var sw = { wx: 0, wy: 0, b: 0 };
    var best = lineAcc(sw);
    for (var ep = 0; ep < 400 && best < pts.length; ep++) {
      pts.forEach(function(p) {
        var s = (cur.wx * p.x + (oneD ? 0 : cur.wy * p.y) + cur.b) >= 0 ? 1 : 0;
        var e = p.cls - s; // -1, 0, or +1
        if (e !== 0) {
          cur.wx += e * p.x / sc;
          if (!oneD) cur.wy += e * p.y / sc;
          cur.b += e / sc;
        }
      });
      var s2 = lineAcc(cur);
      if (s2 > best) { best = s2; sw.wx = cur.wx; sw.wy = cur.wy; sw.b = cur.b; }
    }
    var acc = best / pts.length;

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

// ---- Per-module hooks (from MODULE_CONFIG) ----
if (MODULE_CONFIG.title) document.title = MODULE_CONFIG.title;
if (MODULE_CONFIG.widgets) Object.assign(INTERACTIVE_WIDGETS, MODULE_CONFIG.widgets);
if (typeof MODULE_CONFIG.onReady === 'function') {
  Reveal.on('ready', function() { MODULE_CONFIG.onReady(Reveal); });
}
if (typeof MODULE_CONFIG.onSlideChanged === 'function') {
  Reveal.on('slidechanged', function(event) { MODULE_CONFIG.onSlideChanged(event); });
}
