// Shared helpers for the :::interactive widgets defined in each module's
// config.js. Loaded after config.js, so factories must reference WIDGET_UTIL
// from inside their bodies (which run at DOM-ready), never at definition time.
var WIDGET_UTIL = (function () {
  var COL = {
    text: '#e8eaf0', muted: '#8892a4', line: '#2a3450',
    primary: '#4a9eff', secondary: '#f5a623', green: '#3fb950',
    red: '#e74c3c', purple: '#c792ea', bg: '#0d1225'
  };

  // Size the backing store to the device pixel ratio so canvas text stays sharp.
  function fit(canvas) {
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return null;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h };
  }

  // Keep slider and button interaction from reaching reveal's navigation keys.
  function stop(host) {
    ['pointerdown', 'keydown', 'click'].forEach(function (ev) {
      host.addEventListener(ev, function (e) { e.stopPropagation(); });
    });
  }

  function rounded(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  // Markup for one labelled range slider inside an .iw-sliders grid.
  function sliderHTML(key, label, min, max, step, value) {
    return '<div class="iw-slider">' +
      '<label for="s-' + key + '">' + label + '</label>' +
      '<input id="s-' + key + '" type="range" data-key="' + key + '" min="' + min +
      '" max="' + max + '" step="' + step + '" value="' + value + '">' +
      '<span class="iw-val" data-val="' + key + '"></span>' +
      '</div>';
  }

  // Wire every slider in `host` to one callback; returns a read of all values.
  function bindSliders(host, onChange) {
    var inputs = Array.prototype.slice.call(host.querySelectorAll('input[type="range"]'));
    inputs.forEach(function (el) {
      el.addEventListener('input', function () { onChange(el.getAttribute('data-key'), parseFloat(el.value)); });
    });
    return function () {
      var out = {};
      inputs.forEach(function (el) { out[el.getAttribute('data-key')] = parseFloat(el.value); });
      return out;
    };
  }

  function setVal(host, key, text) {
    var el = host.querySelector('[data-val="' + key + '"]');
    if (el) el.textContent = text;
  }

  // Exclusive button group: marks the clicked button active, calls back with its value.
  function bindToggle(host, selector, onPick) {
    var btns = Array.prototype.slice.call(host.querySelectorAll(selector));
    btns.forEach(function (b) {
      b.addEventListener('click', function () {
        btns.forEach(function (o) { o.classList.remove('active'); });
        b.classList.add('active');
        onPick(b.getAttribute('data-val'), b);
      });
    });
  }

  function softmax(logits, temperature) {
    var T = temperature || 1;
    var scaled = logits.map(function (z) { return z / T; });
    var m = Math.max.apply(null, scaled);
    var ex = scaled.map(function (z) { return Math.exp(z - m); });
    var s = ex.reduce(function (a, b) { return a + b; }, 0);
    return ex.map(function (v) { return v / s; });
  }

  // Deterministic PRNG so a widget redraws identically on resize.
  function rng(seed) {
    var s = seed >>> 0;
    return function () {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }

  // Decimal units throughout (1 GB = 1000 MB), matching how model and cache
  // sizes are quoted in the deck.
  function fmtBytes(gb) {
    if (gb >= 1000) return (gb / 1000).toFixed(2) + ' TB';
    if (gb >= 1) return gb.toFixed(gb >= 10 ? 1 : 2) + ' GB';
    var mb = gb * 1000;
    if (mb >= 0.1) return mb.toFixed(mb >= 100 ? 0 : mb >= 10 ? 1 : 2) + ' MB';
    return (mb * 1000).toFixed(0) + ' KB';
  }

  // Compact SI-ish formatting for parameter and token counts.
  function fmtCount(n) {
    if (n >= 1e12) return (n / 1e12).toFixed(2) + 'T';
    if (n >= 1e9) return (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(0) + 'K';
    return String(Math.round(n));
  }

  return {
    COL: COL, fit: fit, stop: stop, rounded: rounded,
    sliderHTML: sliderHTML, bindSliders: bindSliders, setVal: setVal,
    bindToggle: bindToggle, softmax: softmax, rng: rng,
    fmtBytes: fmtBytes, fmtCount: fmtCount
  };
}());
