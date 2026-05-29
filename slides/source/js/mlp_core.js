// =====================================================================
// MLP Core — Shared neural-net library consumed by multiple widgets.
// Deterministic data generation, preset lookup-model geometry, and the
// Adam trainer (used by the debug harness; the deck uses lookupModel).
// =====================================================================

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

  function classifierBoundaryLines(data, count, dataset, green, red) {
    var centerCount = dataset === 'spiral' ? 24 : Math.max(8, count * 2);
    green = green || sampleClassPath(data, 1, centerCount);
    red = red || sampleClassPath(data, 0, centerCount);
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

  function geometryForModel(dataset, data, count, depth, band, solved, centerline, redCenterline) {
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
      var tube = tubePolygon(dataset, data, count, band);
      if (tube) return { polygon: tube, lines: polygonEdges(tube) };
      return { polygon: null, lines: classifierBoundaryLines(data, count, dataset, centerline, redCenterline) };
    }
    return { polygon: null, lineMode: 'halfPlane', lines: [{ x1: 0.04, y1: 0.96, x2: 0.96, y2: 0.04 }] };
  }

  function lookupModel(dataset, data, width, depth) {
    var pieces = effectivePieces(dataset, width, depth);
    var solved = pieces >= requiredPieces(dataset);
    var display = displayPieces(width, depth);
    var band = dataset === 'spiral' ? 0.088 : 0.16;
    var centerCount = Math.max(3, Math.ceil(pieces / 2) + 1);
    if (solved) {
      centerCount = dataset === 'spiral'
        ? Math.max(12, display + 1)
        : Math.max(8, display * 2);
    }
    var centerline = (dataset === 'moons' || dataset === 'spiral') ? sampleClassPath(data, 1, centerCount) : null;
    var redCenterline = (dataset === 'moons' || dataset === 'spiral') ? sampleClassPath(data, 0, centerCount) : null;
    var lookupPoints = (dataset === 'moons' || dataset === 'spiral') && solved ? data : pickPrototypes(data, pieces);
    var geom = geometryForModel(dataset, data, display, depth, band, solved, centerline, redCenterline);
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
