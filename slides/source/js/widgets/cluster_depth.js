// =====================================================================
// CLUSTER_MLP — a small real ReLU MLP trained in the browser (seeded, so
// the result is repeatable), plus a marching-squares contour helper.
// Used by the "Clusters" tab of the mlpBoundary widget: four green
// clusters on a red background. Each hidden neuron's zero-line in input
// space is a straight line for layer 1 and a piecewise-linear curve for
// deeper layers; the final boundary is the output neuron's zero-line.
// =====================================================================
var CLUSTER_MLP = (function() {
  function mulberry32(a) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function gauss(rnd) {
    var u = 1 - rnd(), v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.2831853 * v);
  }
  var CENTERS = [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]];

  function generateData() {
    var rnd = mulberry32(7);
    var pts = [];
    CENTERS.forEach(function(c) {
      for (var i = 0; i < 22; i++) {
        pts.push({ x: c[0] + gauss(rnd) * 0.045, y: c[1] + gauss(rnd) * 0.045, cls: 1 });
      }
    });
    var reds = 0;
    while (reds < 150) {
      var x = 0.03 + rnd() * 0.94, y = 0.03 + rnd() * 0.94;
      var near = CENTERS.some(function(c) { return Math.hypot(x - c[0], y - c[1]) < 0.17; });
      if (near) continue;
      pts.push({ x: x, y: y, cls: 0 });
      reds++;
    }
    return pts;
  }

  // ---- network: [2 -> W]*D (ReLU) -> 1 (sigmoid), full-batch Adam ----
  function makeNet(width, depth, seed) {
    var rnd = mulberry32(seed);
    var layers = [];
    var fanIn = 2;
    for (var d = 0; d < depth; d++) {
      layers.push(initLayer(fanIn, width, rnd));
      fanIn = width;
    }
    layers.push(initLayer(fanIn, 1, rnd));
    return { width: width, depth: depth, layers: layers };
  }
  function initLayer(nIn, nOut, rnd) {
    var W = [], b = [];
    var scale = Math.sqrt(2 / nIn);
    for (var o = 0; o < nOut; o++) {
      var row = [];
      for (var i = 0; i < nIn; i++) row.push(gauss(rnd) * scale);
      W.push(row);
      b.push(gauss(rnd) * 0.1);
    }
    return { W: W, b: b, mW: zeros(nOut, nIn), vW: zeros(nOut, nIn), mb: new Array(nOut).fill(0), vb: new Array(nOut).fill(0) };
  }
  function zeros(r, c) {
    var a = [];
    for (var i = 0; i < r; i++) a.push(new Array(c).fill(0));
    return a;
  }

  // Forward pass returning every layer's pre-activations and activations.
  // Inputs are centered so the first layer's lines can pass near the middle.
  function forward(net, x, y) {
    var act = [x - 0.5, y - 0.5];
    var pres = [], acts = [act];
    for (var l = 0; l < net.layers.length; l++) {
      var L = net.layers[l];
      var pre = [], out = [];
      var last = l === net.layers.length - 1;
      for (var o = 0; o < L.W.length; o++) {
        var s = L.b[o];
        for (var i = 0; i < act.length; i++) s += L.W[o][i] * act[i];
        pre.push(s);
        out.push(last ? s : (s > 0 ? s : 0));
      }
      pres.push(pre); acts.push(out); act = out;
    }
    return { pres: pres, acts: acts, logit: act[0] };
  }
  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  function trainSteps(net, data, steps, lr) {
    var b1 = 0.9, b2 = 0.999, eps = 1e-8;
    net.t = net.t || 0;
    for (var s = 0; s < steps; s++) {
      net.t++;
      var grads = net.layers.map(function(L) {
        return { W: zeros(L.W.length, L.W[0].length), b: new Array(L.W.length).fill(0) };
      });
      for (var n = 0; n < data.length; n++) {
        var p = data[n];
        var f = forward(net, p.x, p.y);
        var delta = [sigmoid(f.logit) - p.cls];
        for (var l = net.layers.length - 1; l >= 0; l--) {
          var L = net.layers[l], inp = f.acts[l], g = grads[l];
          var prev = new Array(inp.length).fill(0);
          for (var o = 0; o < L.W.length; o++) {
            g.b[o] += delta[o];
            for (var i = 0; i < inp.length; i++) {
              g.W[o][i] += delta[o] * inp[i];
              prev[i] += delta[o] * L.W[o][i];
            }
          }
          if (l > 0) {
            for (var k = 0; k < prev.length; k++) if (f.pres[l - 1][k] <= 0) prev[k] = 0;
          }
          delta = prev;
        }
      }
      var N = data.length;
      var c1 = 1 - Math.pow(b1, net.t), c2 = 1 - Math.pow(b2, net.t);
      net.layers.forEach(function(L, l) {
        var g = grads[l];
        for (var o = 0; o < L.W.length; o++) {
          for (var i = 0; i < L.W[o].length; i++) {
            var gw = g.W[o][i] / N;
            L.mW[o][i] = b1 * L.mW[o][i] + (1 - b1) * gw;
            L.vW[o][i] = b2 * L.vW[o][i] + (1 - b2) * gw * gw;
            L.W[o][i] -= lr * (L.mW[o][i] / c1) / (Math.sqrt(L.vW[o][i] / c2) + eps);
          }
          var gb = g.b[o] / N;
          L.mb[o] = b1 * L.mb[o] + (1 - b1) * gb;
          L.vb[o] = b2 * L.vb[o] + (1 - b2) * gb * gb;
          L.b[o] -= lr * (L.mb[o] / c1) / (Math.sqrt(L.vb[o] / c2) + eps);
        }
      });
    }
  }

  function accuracy(net, data) {
    var ok = 0;
    data.forEach(function(p) { if ((forward(net, p.x, p.y).logit > 0 ? 1 : 0) === p.cls) ok++; });
    return ok / data.length;
  }

  // Train with a few seeds and keep the best, so each (width, depth)
  // shows what that architecture can do rather than one unlucky init.
  function train(data, width, depth, opts) {
    opts = opts || {};
    var seeds = opts.seeds || 4, steps = opts.steps || 400, lr = opts.lr || 0.03;
    var best = null;
    for (var s = 0; s < seeds; s++) {
      var net = makeNet(width, depth, 1000 + s * 17 + width * 3 + depth * 101);
      trainSteps(net, data, steps, lr);
      var acc = accuracy(net, data);
      if (!best || acc > best.acc) best = { net: net, acc: acc };
    }
    return best;
  }

  // ---- marching squares: zero-contour of a scalar field on a grid ----
  function contour(field, n) {
    // field[i][j] sampled at x=i/n, y=j/n for i,j in 0..n
    var segs = [];
    function lerp(a, b, va, vb) { return a + (0 - va) / (vb - va) * (b - a); }
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        var v00 = field[i][j], v10 = field[i + 1][j], v01 = field[i][j + 1], v11 = field[i + 1][j + 1];
        var x0 = i / n, x1 = (i + 1) / n, y0 = j / n, y1 = (j + 1) / n;
        var pts = [];
        if ((v00 > 0) !== (v10 > 0)) pts.push([lerp(x0, x1, v00, v10), y0]);
        if ((v10 > 0) !== (v11 > 0)) pts.push([x1, lerp(y0, y1, v10, v11)]);
        if ((v01 > 0) !== (v11 > 0)) pts.push([lerp(x0, x1, v01, v11), y1]);
        if ((v00 > 0) !== (v01 > 0)) pts.push([x0, lerp(y0, y1, v00, v01)]);
        if (pts.length === 2) segs.push([pts[0], pts[1]]);
        else if (pts.length === 4) { segs.push([pts[0], pts[1]]); segs.push([pts[2], pts[3]]); }
      }
    }
    return segs;
  }

  return { generateData: generateData, train: train, forward: forward, accuracy: accuracy, contour: contour, sigmoid: sigmoid };
}());
if (typeof module !== "undefined" && module.exports) module.exports = CLUSTER_MLP;

