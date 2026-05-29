// =====================================================================
// Core deck infrastructure: Manim steppers, Reveal.js init, syntax
// highlighting, interactive widget plumbing, and per-module hooks.
// Loaded before any widget definitions.
// =====================================================================

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

// ---- Per-module hooks (from MODULE_CONFIG) ----
if (MODULE_CONFIG.title) document.title = MODULE_CONFIG.title;
if (MODULE_CONFIG.widgets) Object.assign(INTERACTIVE_WIDGETS, MODULE_CONFIG.widgets);
if (typeof MODULE_CONFIG.onReady === 'function') {
  Reveal.on('ready', function() { MODULE_CONFIG.onReady(Reveal); });
}
if (typeof MODULE_CONFIG.onSlideChanged === 'function') {
  Reveal.on('slidechanged', function(event) { MODULE_CONFIG.onSlideChanged(event); });
}
