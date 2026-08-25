(function () {
  // Module 11 config. This deck has no Manim clips: the pipelines and loops in
  // this module are sequential diagrams, which reveal.js fragments and the
  // shared card/table styles handle without spatial animation.
  // ===================================================================
  // WIDGET: retrievalCompare — the same query scored by word overlap and
  // by meaning, then fused, over one small document set.
  // ===================================================================
  function retrievalCompare(host) {
    var U = WIDGET_UTIL;
    // Scores are hand-set to show where each retriever is strong: sparse
    // wins on exact rare strings, dense wins on paraphrase.
    var DOCS = [
      { id: 'D1', text: 'Refund policy: orders may be returned within 30 days of delivery.' },
      { id: 'D2', text: 'Our money-back window closes one month after the parcel arrives.' },
      { id: 'D3', text: 'Error code SKU-4417 means the warehouse could not reserve stock.' },
      { id: 'D4', text: 'Inventory reservation failures are logged by the fulfillment service.' },
      { id: 'D5', text: 'Shipping times vary by region and carrier.' },
      { id: 'D6', text: 'The 30-day guarantee does not apply to custom engraving.' }
    ];
    var QUERIES = {
      exact: {
        text: 'What does SKU-4417 mean?',
        sparse: { D3: 9.4, D6: 0.15, D1: 0.12, D4: 0.05, D2: 0.03, D5: 0.01 },
        dense: { D4: 0.74, D5: 0.55, D2: 0.51, D3: 0.49, D1: 0.44, D6: 0.30 },
        gold: 'D3'
      },
      para: {
        text: 'Can I get my money back after three weeks?',
        sparse: { D1: 1.2, D6: 1.0, D5: 0.35, D2: 0.30, D3: 0.05, D4: 0.02 },
        dense: { D2: 0.84, D1: 0.79, D6: 0.61, D5: 0.33, D4: 0.19, D3: 0.15 },
        gold: 'D2'
      },
      mixed: {
        text: 'Does the 30-day refund window cover engraved items?',
        sparse: { D1: 6.4, D6: 5.9, D2: 0.80, D5: 0.20, D3: 0.05, D4: 0.02 },
        dense: { D6: 0.83, D2: 0.74, D1: 0.70, D5: 0.31, D4: 0.18, D3: 0.14 },
        gold: 'D6'
      }
    };
    var state = { q: 'exact', mode: 'sparse' };

    host.innerHTML =
      '<div class="iw">' +
        '<div class="iw-controls">' +
          '<span class="iw-label">query:</span>' +
          '<button class="iw-btn active" data-q data-val="exact">rare exact term</button>' +
          '<button class="iw-btn" data-q data-val="para">paraphrase</button>' +
          '<button class="iw-btn" data-q data-val="mixed">both at once</button>' +
        '</div>' +
        '<p class="iw-readout" data-el="qtext" style="min-height:0;"></p>' +
        '<div class="iw-body iw-cols iw-hug">' +
          '<div class="iw-panel">' +
            '<h4>Ranking</h4>' +
            '<div class="iw-rows" data-el="rank"></div>' +
          '</div>' +
          '<div class="iw-panel">' +
            '<h4>Retriever</h4>' +
            '<div class="iw-controls" style="flex-direction:column; align-items:stretch; gap:8px;">' +
              '<button class="iw-btn active" data-mode data-val="sparse">Sparse (BM25)</button>' +
              '<button class="iw-btn" data-mode data-val="dense">Dense (embeddings)</button>' +
              '<button class="iw-btn" data-mode data-val="hybrid">Hybrid (RRF fusion)</button>' +
            '</div>' +
            '<h4 style="margin-top:14px;">Gold document</h4>' +
            '<div class="iw-tokens" data-el="gold"></div>' +
          '</div>' +
        '</div>' +
        '<p class="iw-readout" data-el="verdict"></p>' +
      '</div>';
    U.stop(host);
    var el = {};
    host.querySelectorAll('[data-el]').forEach(function (n) { el[n.getAttribute('data-el')] = n; });

    function ranked(scores) {
      return DOCS.map(function (d) { return d.id; })
        .sort(function (a, b) { return scores[b] - scores[a]; });
    }

    // Reciprocal rank fusion, the standard cheap way to combine two rankings.
    function rrf(q) {
      var rs = ranked(q.sparse), rd = ranked(q.dense), K = 60, out = {};
      DOCS.forEach(function (d) {
        out[d.id] = 1 / (K + rs.indexOf(d.id) + 1) + 1 / (K + rd.indexOf(d.id) + 1);
      });
      return out;
    }

    function draw() {
      var q = QUERIES[state.q];
      var scores = state.mode === 'sparse' ? q.sparse : state.mode === 'dense' ? q.dense : rrf(q);
      var order = ranked(scores);

      el.qtext.innerHTML = '<strong>' + q.text + '</strong>';
      el.gold.innerHTML = '<span class="iw-chip good">' + q.gold + '</span>';

      el.rank.innerHTML = order.map(function (id, i) {
        var doc = DOCS.filter(function (d) { return d.id === id; })[0];
        var top = i < 3;
        var isGold = id === q.gold;
        var val = state.mode === 'hybrid' ? scores[id].toFixed(4)
          : state.mode === 'dense' ? scores[id].toFixed(2) : scores[id].toFixed(1);
        return '<div class="iw-row" style="grid-template-columns: 74px 1fr;">' +
          '<p class="iw-key"><span class="iw-chip ' + (isGold ? 'good' : top ? '' : 'dim') + '">' + id + '</span></p>' +
          '<div style="font-size:12pt; color:' + (top ? 'var(--text-color)' : 'var(--muted-color)') + ';">' +
          doc.text + ' <span style="font-family:monospace; color:var(--secondary-color);">' + val + '</span></div>' +
          '</div>';
      }).join('');

      var goldRank = order.indexOf(q.gold) + 1;
      var sparseRank = ranked(q.sparse).indexOf(q.gold) + 1;
      var denseRank = ranked(q.dense).indexOf(q.gold) + 1;
      var msg;
      if (state.mode === 'hybrid') {
        msg = 'Fusion puts the answer at rank <strong>' + goldRank + '</strong>, from <strong>' +
          sparseRank + '</strong> on word overlap and <strong>' + denseRank +
          '</strong> on meaning. Neither retriever has to be right alone.';
      } else if (state.mode === 'dense' && denseRank > 2) {
        msg = 'The answer sits at rank <strong>' + denseRank +
          '</strong>. An embedding has no special respect for a rare identifier, so documents that merely sound related outrank the exact match.';
      } else if (state.mode === 'sparse' && sparseRank > 2) {
        msg = 'The answer sits at rank <strong>' + sparseRank +
          '</strong>. The query and the answer share almost no words, and word overlap cannot see past that.';
      } else {
        msg = 'The answer is at rank <strong>' + goldRank + '</strong>. This is the query type this retriever handles well.';
      }
      el.verdict.innerHTML = msg;
    }

    U.bindToggle(host, '[data-q]', function (val) { state.q = val; draw(); });
    U.bindToggle(host, '[data-mode]', function (val) { state.mode = val; draw(); });
    draw();
    return { resize: draw };
  }

  window.MODULE_CONFIG = {
    title: 'LLMs 0 to 100 - Module 11',
    manimSections: {},
    widgets: {
      retrievalCompare: retrievalCompare
    }
  };
}());
