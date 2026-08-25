// =====================================================================
// Slide annotation / drawing overlay.
//
// A full-viewport <canvas> the presenter can draw on top of any slide.
// Strokes are stored per slide (keyed by Reveal indices, in normalized
// coordinates) so they survive navigation and window resizes. A small
// floating toolbar exposes pen colors, an eraser, undo, and clear.
//
// Keyboard:
//   d       toggle draw mode on/off
//   e       eraser (while in draw mode)
//   c       clear the current slide's drawing
//   u       undo the last stroke
//   Escape  exit draw mode
//
// While draw mode is active, Reveal's own keyboard/touch navigation is
// suspended so drawing gestures never flip the slide.
// =====================================================================
(function () {
  'use strict';

  var PALETTE = ['#ff5252', '#ffd740', '#69f0ae', '#40c4ff', '#ffffff'];
  var PEN_WIDTH = 3;
  var ERASER_WIDTH = 28;

  // strokesBySlide[key] = [ {color, width, erase, points:[{x,y}, ...]}, ... ]
  // Points are normalized to [0,1] against the canvas so they rescale on resize.
  var strokesBySlide = {};

  var canvas, ctx, toolbar;
  var drawMode = false;
  var erasing = false;
  var color = PALETTE[0];
  var drawing = false;
  var currentStroke = null;

  // ---- slide identity + canvas geometry ----
  function slideKey() {
    var idx = (typeof Reveal !== 'undefined' && Reveal.getIndices) ? Reveal.getIndices() : { h: 0, v: 0 };
    return idx.h + '-' + (idx.v || 0);
  }

  function strokes() {
    var k = slideKey();
    if (!strokesBySlide[k]) strokesBySlide[k] = [];
    return strokesBySlide[k];
  }

  // Size the backing store to device pixels for crisp lines, keep the CSS
  // box at viewport size, then repaint the current slide's strokes.
  function resizeCanvas() {
    if (!canvas) return;
    var dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(window.innerWidth * dpr);
    canvas.height = Math.round(window.innerHeight * dpr);
    canvas.style.width = window.innerWidth + 'px';
    canvas.style.height = window.innerHeight + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    redraw();
  }

  function drawStroke(s) {
    if (!s.points.length) return;
    var w = window.innerWidth, h = window.innerHeight;
    ctx.globalCompositeOperation = s.erase ? 'destination-out' : 'source-over';
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(s.points[0].x * w, s.points[0].y * h);
    for (var i = 1; i < s.points.length; i++) {
      ctx.lineTo(s.points[i].x * w, s.points[i].y * h);
    }
    // A single tap should still leave a visible dot.
    if (s.points.length === 1) ctx.lineTo(s.points[0].x * w + 0.01, s.points[0].y * h);
    ctx.stroke();
  }

  function redraw() {
    if (!ctx) return;
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    strokes().forEach(drawStroke);
  }

  // ---- mode toggling ----
  // While drawing, the canvas captures pointer events. Arrow/space keys still
  // navigate slides (they never conflict with pointer drawing), so keyboard
  // nav stays on; only touch/swipe is suspended so finger-drawing on a
  // touchscreen can't flip the slide out from under the pen.
  function setDrawMode(on) {
    drawMode = on;
    if (!on) erasing = false;
    document.body.classList.toggle('draw-mode', on);
    if (canvas) canvas.style.pointerEvents = on ? 'auto' : 'none';
    if (typeof Reveal !== 'undefined' && Reveal.configure) {
      Reveal.configure({ touch: !on });
    }
    updateToolbar();
  }

  function setColor(c) {
    color = c;
    erasing = false;
    if (!drawMode) setDrawMode(true);
    updateToolbar();
  }

  function setEraser() {
    erasing = true;
    if (!drawMode) setDrawMode(true);
    updateToolbar();
  }

  function clearSlide() {
    strokesBySlide[slideKey()] = [];
    redraw();
  }

  function undo() {
    strokes().pop();
    redraw();
  }

  function injectStyles() {
    var css = [
      '#draw-canvas{position:fixed;inset:0;z-index:40;pointer-events:none;touch-action:none;}',
      'body.draw-mode #draw-canvas{cursor:crosshair;}',
      // Hidden by default; slides up into view only while the pointer is in
      // the bottom hover band (see the mousemove handler that toggles .visible).
      '#draw-toolbar{position:fixed;bottom:14px;left:50%;',
      '  transform:translateX(-50%) translateY(24px);',
      '  z-index:60;display:flex;align-items:center;gap:8px;padding:7px 10px;',
      '  background:rgba(20,20,24,0.92);border:1px solid rgba(255,255,255,0.18);',
      '  border-radius:10px;opacity:0;pointer-events:none;',
      '  transition:opacity 0.15s,transform 0.15s;font-family:sans-serif;}',
      '#draw-toolbar.visible{opacity:1;pointer-events:auto;transform:translateX(-50%) translateY(0);}',
      '.draw-btn{width:26px;height:26px;border-radius:6px;border:1px solid rgba(255,255,255,0.25);',
      '  background:transparent;cursor:pointer;padding:0;display:flex;align-items:center;',
      '  justify-content:center;color:#fff;font-size:14px;line-height:1;}',
      '.draw-btn:hover{background:rgba(255,255,255,0.15);}',
      '.draw-swatch{width:22px;height:22px;border-radius:50%;border:2px solid transparent;cursor:pointer;padding:0;}',
      '.draw-swatch.active{border-color:#fff;}',
      '.draw-btn.active{background:rgba(255,255,255,0.28);}',
      '.draw-sep{width:1px;height:22px;background:rgba(255,255,255,0.2);}'
    ].join('\n');
    var style = document.createElement('style');
    style.textContent = css;
    document.head.appendChild(style);
  }

  function buildToolbar() {
    toolbar = document.createElement('div');
    toolbar.id = 'draw-toolbar';

    // Pen toggle
    var penBtn = document.createElement('button');
    penBtn.className = 'draw-btn';
    penBtn.dataset.role = 'pen';
    penBtn.title = 'Draw (d)';
    penBtn.textContent = '✎'; // pencil
    penBtn.addEventListener('click', function () { setDrawMode(!drawMode); });
    toolbar.appendChild(penBtn);

    toolbar.appendChild(sep());

    // Color swatches
    PALETTE.forEach(function (c) {
      var b = document.createElement('button');
      b.className = 'draw-swatch';
      b.dataset.color = c;
      b.style.background = c;
      b.title = 'Pen color';
      b.addEventListener('click', function () { setColor(c); });
      toolbar.appendChild(b);
    });

    toolbar.appendChild(sep());

    // Eraser / undo / clear
    toolbar.appendChild(iconBtn('⊘', 'Eraser (e)', 'eraser', setEraser));
    toolbar.appendChild(iconBtn('↶', 'Undo (u)', 'undo', undo));
    toolbar.appendChild(iconBtn('✕', 'Clear slide (c)', 'clear', clearSlide));

    document.body.appendChild(toolbar);
    updateToolbar();
  }

  function sep() {
    var s = document.createElement('span');
    s.className = 'draw-sep';
    return s;
  }

  function iconBtn(glyph, title, role, handler) {
    var b = document.createElement('button');
    b.className = 'draw-btn';
    b.dataset.role = role;
    b.title = title;
    b.textContent = glyph;
    b.addEventListener('click', handler);
    return b;
  }

  function updateToolbar() {
    if (!toolbar) return;
    toolbar.querySelectorAll('.draw-swatch').forEach(function (b) {
      b.classList.toggle('active', drawMode && !erasing && b.dataset.color === color);
    });
    var pen = toolbar.querySelector('[data-role="pen"]');
    if (pen) pen.classList.toggle('active', drawMode);
    var er = toolbar.querySelector('[data-role="eraser"]');
    if (er) er.classList.toggle('active', drawMode && erasing);
  }

  // ---- pointer drawing ----
  function point(e) {
    return { x: e.clientX / window.innerWidth, y: e.clientY / window.innerHeight };
  }

  function startStroke(e) {
    if (!drawMode) return;
    drawing = true;
    currentStroke = {
      color: color,
      width: erasing ? ERASER_WIDTH : PEN_WIDTH,
      erase: erasing,
      points: [point(e)]
    };
    strokes().push(currentStroke);
    if (canvas.setPointerCapture && e.pointerId != null) {
      try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
    }
    e.preventDefault();
  }

  function extendStroke(e) {
    if (!drawing || !currentStroke) return;
    currentStroke.points.push(point(e));
    redraw();
    e.preventDefault();
  }

  function endStroke() {
    drawing = false;
    currentStroke = null;
  }

  function bindCanvasEvents() {
    canvas.addEventListener('pointerdown', startStroke);
    canvas.addEventListener('pointermove', extendStroke);
    canvas.addEventListener('pointerup', endStroke);
    canvas.addEventListener('pointercancel', endStroke);
    canvas.addEventListener('pointerleave', endStroke);
  }

  function bindKeys() {
    // Capture phase so we act before Reveal, but only for our bare shortcuts
    // (ignore when a modifier is held or focus is in a text field).
    document.addEventListener('keydown', function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      var tag = (e.target && e.target.tagName) || '';
      if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target && e.target.isContentEditable)) return;

      if (e.key === 'd' || e.key === 'D') {
        setDrawMode(!drawMode);
        e.preventDefault(); e.stopImmediatePropagation();
      } else if (!drawMode) {
        return; // remaining shortcuts only apply while drawing
      } else if (e.key === 'e' || e.key === 'E') {
        setEraser(); e.preventDefault(); e.stopImmediatePropagation();
      } else if (e.key === 'c' || e.key === 'C') {
        clearSlide(); e.preventDefault(); e.stopImmediatePropagation();
      } else if (e.key === 'u' || e.key === 'U') {
        undo(); e.preventDefault(); e.stopImmediatePropagation();
      } else if (e.key === 'Escape') {
        setDrawMode(false); e.preventDefault(); e.stopImmediatePropagation();
      }
    }, true);
  }

  // Reveal the toolbar only while the pointer is within a band along the
  // bottom of the viewport (or hovering the toolbar itself).
  function bindToolbarReveal() {
    var BAND = 90; // px from the bottom edge
    window.addEventListener('mousemove', function (e) {
      var near = e.clientY >= window.innerHeight - BAND;
      toolbar.classList.toggle('visible', near);
    });
    // Keep it up while the cursor is actually on the toolbar (drawing gestures
    // never fire mousemove there, but plain hover to click a tool should hold).
    toolbar.addEventListener('mouseenter', function () { toolbar.classList.add('visible'); });
  }

  function init() {
    injectStyles();

    canvas = document.createElement('canvas');
    canvas.id = 'draw-canvas';
    document.body.appendChild(canvas);
    ctx = canvas.getContext('2d');

    buildToolbar();
    bindCanvasEvents();
    bindKeys();
    bindToolbarReveal();
    resizeCanvas();

    window.addEventListener('resize', resizeCanvas);
    if (typeof Reveal !== 'undefined' && Reveal.on) {
      // Repaint the destination slide's stored strokes after navigation.
      Reveal.on('slidechanged', redraw);
      Reveal.on('ready', resizeCanvas);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
