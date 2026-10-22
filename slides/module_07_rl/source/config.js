(function () {
  function dpoTrainingDiagram() {
    return String.raw`<div class="loop-diagram dpo-diagram" role="img" aria-label="DPO: fixed preference pair, trainable LLM and frozen reference log-probabilities, preference margin, loss, and optimizer update">
      <svg class="loop-wires" viewBox="0 0 1120 430" aria-hidden="true">
        <defs><marker id="dpo-detail-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="none" stroke="currentColor" stroke-width="1.5"/></marker></defs>
        <g fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#dpo-detail-arrow)">
          <path class="flow-sample" d="M205 93 H255"/>
          <path class="flow-score" d="M465 93 H525"/>
          <path class="flow-score" d="M465 300 H495 V180 H525"/>
          <path class="flow-advantage" d="M660 220 V270"/>
          <path class="flow-advantage" d="M795 320 H845"/>
          <path class="flow-gradient" d="M970 371 V410 H125 V368"/>
          <path class="flow-update" d="M125 270 V205 H360 V170"/>
        </g>
      </svg>
      <div class="loop-node dpo-pair"><p class="loop-label">Fixed dataset pair</p><p>Prompt: reverse: cat</p><p><b class="positive">tac: chosen</b><br/><b class="negative">tca: rejected</b></p></div>
      <div class="loop-node loop-model dpo-policy"><p class="loop-label">LLM policy π<sub>θ</sub></p><p>Trainable weights θ</p><div class="weight-grid" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div><p class="loop-small loop-direction">Push ↑ chosen; ↓ rejected</p></div>
      <div class="loop-node dpo-reference"><p class="loop-label">Frozen reference π<sub>ref</sub></p><p>Scores both responses</p><p class="loop-small">No weight updates</p></div>
      <div class="loop-node dpo-logprobs"><p class="loop-label">Score the fixed responses under both models</p>
        <table><thead><tr><th>Response</th><th>log π<sub>θ</sub><br/>before → after</th><th>log π<sub>ref</sub><br/>frozen</th><th class="loop-math" data-tex="s=\log\frac{\pi_\theta}{\pi_{\rm ref}}"></th></tr></thead><tbody>
        <tr><td class="positive">tac</td><td>−1.386 → −0.511</td><td>−1.386</td><td>0 → +0.875</td></tr>
        <tr><td class="negative">tca</td><td>−0.288 → −0.916</td><td>−0.288</td><td>0 → −0.629</td></tr></tbody></table>
        <p class="loop-small">Toy probabilities: chosen 0.25 → 0.60; rejected 0.75 → 0.40</p>
      </div>
      <div class="loop-node loop-advantage dpo-margin"><p class="loop-label">Preference margin · β = 1</p><p class="loop-math" data-tex="m=\beta(s_w-s_l)"></p><p>0 → 1.504</p></div>
      <div class="loop-node loop-loss dpo-loss"><p class="loop-label">Minimize the DPO loss</p><p class="loop-math" data-tex="\mathcal L=-\log\sigma(m)"></p><p>0.693 → 0.201</p></div>
      <div class="loop-node loop-optimizer dpo-optimizer"><p class="loop-label">Optimizer updates θ</p><p class="loop-math" data-tex="\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L"></p><p class="loop-small">Policy gradients only</p></div>
      <p class="loop-edge-label dpo-gradient-label">Gradient returns to the LLM; the reference stays frozen</p>
    </div>`;
  }

  // All algorithms use the same layout so only the learning signal changes.
  function mountTrainingLoops(reveal) {
    var added = false;
    document.querySelectorAll('.training-loop:not([data-mounted])').forEach(function (host) {
      var algorithm = host.dataset.algorithm;
      var dpo = algorithm === 'dpo';
      var ppo = algorithm === 'ppo';
      var grpo = algorithm === 'grpo';
      var baseline = grpo ? 'Group: mean 0.5; std 0.5'
        : ppo ? 'Critic Vψ(x) predicts 0.5' : 'Running baseline b = 0.5';
      var advantages = grpo ? 'A = [+1, −1, +1, −1]'
        : 'A = [+0.5, −0.5, +0.5, −0.5]';
      var stages = dpo
        ? ['Load fixed pair', 'Score log-probs', 'Compare preferences', 'Update weights', 'Load next pair']
        : ['Sample; hold θ fixed', 'Score; hold θ fixed', 'Compute advantages', 'Update weights', 'Sample with new θ'];

      host.dataset.mounted = 'yes';
      host.dataset.phase = '0';
      host.innerHTML =
        '<div class="loop-diagram" role="img" aria-label="' + algorithm.toUpperCase() + ' training loop: inputs, LLM weights, responses, scoring, learning signal, and optimizer feedback">' +
          '<svg class="loop-wires" viewBox="0 0 1120 380" aria-hidden="true">' +
            '<defs><marker id="loop-arrow-' + algorithm + '" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="none" stroke="currentColor" stroke-width="1.5"/></marker></defs>' +
            '<g fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#loop-arrow-' + algorithm + ')">' +
              '<path class="flow-sample" d="M174 82 H230"/>' +
              '<path class="flow-sample" d="M430 82 H490"/>' +
              '<path class="flow-score" d="M690 82 H782"/>' +
              '<path class="flow-advantage" d="M940 164 V232"/>' +
              '<path class="flow-gradient" d="M782 286 H706"/>' +
              '<path class="flow-gradient" d="M486 286 H430"/>' +
              '<path class="flow-update" d="M330 236 V182"/>' +
              (grpo ? '<path class="flow-kl" d="M87 340 V365 H596 V340"/>' : '') +
            '</g>' +
            '<path class="critic-learning" d="M330 340 V365 H940 V340" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="6 6" marker-end="url(#loop-arrow-' + algorithm + ')"/>' +
          '</svg>' +
          '<div class="loop-node loop-prompt"><p class="loop-label">Prompt x</p><p>reverse: cat</p></div>' +
          '<div class="loop-node loop-model"><p class="loop-label">LLM policy π<sub>θ</sub></p><p>Trainable weights θ</p><div class="weight-grid" aria-hidden="true">' + '<i></i>'.repeat(15) + '</div><p class="loop-small loop-direction">Push ↑ tac; ↓ tca, cta</p></div>' +
          '<div class="loop-node loop-responses"><p class="loop-label">' + (dpo ? 'Fixed preference pair' : 'Responses y') + '</p>' +
            (dpo ? '<p class="loop-completions">tac <b class="positive">chosen</b><br/>tca <b class="negative">rejected</b></p>'
              : '<p class="loop-completions">tac &nbsp; tca &nbsp; tac &nbsp; cta</p><p class="loop-small">Same prompt, four attempts</p>') + '</div>' +
          '<div class="loop-node loop-checker"><p class="loop-label">' + (dpo ? 'Frozen reference πref' : 'Fixed Python checker') + '</p>' +
            (dpo ? '<p>Compare policy and reference<br/>log-probabilities for both answers</p><p class="loop-small">Preferences were collected earlier</p>'
              : '<p>answer == input[::-1]</p><p class="loop-rewards">Rewards R = [1, 0, 1, 0]</p><p class="loop-small">The checker determines the score</p>') + '</div>' +
          '<div class="loop-node loop-advantage"><p class="loop-label">' + (dpo ? 'Preference margin' : baseline) + '</p>' +
            (dpo ? '<p>Chosen vs rejected<br/>relative to the reference</p>'
              : '<p>' + advantages + '</p><p class="loop-small">' + (ppo ? 'Toy value; full PPO uses token states' : grpo ? 'Population std shown; no critic' : 'Estimate from earlier batches') + '</p>') + '</div>' +
          '<div class="loop-node loop-loss"><p class="loop-label">' + (dpo ? 'Preference loss' : 'Policy loss') + '</p>' +
            (grpo ? String.raw`<p class="loop-math" data-tex="\begin{gathered}\mathcal L=-J_{\mathrm{clip}}\\+\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})\end{gathered}"></p>`
              : '<p>' + (dpo ? '−log σ(margin)' : ppo ? 'Clipped-ratio objective' : '−A log πθ(y | x)') + '</p><p class="loop-small">LLM forward + backprop<br/>on the fixed response tokens</p>') + '</div>' +
          (grpo ? '<div class="loop-node loop-reference"><p class="loop-label">Frozen π<sub>ref</sub></p><p>Same tokens</p><p class="loop-small">No weight updates</p></div><p class="loop-edge-label edge-kl">Reference scores for KL</p>' : '') +
          '<div class="loop-node loop-optimizer"><p class="loop-label">Optimizer step</p><p>θ ← θ − η ∇<sub>θ</sub>L</p><p class="loop-small">' + (ppo ? 'Actor and critic both learn' : 'Update LLM weights here') + '</p></div>' +
          '<p class="loop-edge-label edge-gradient">gradient</p>' +
          '<p class="loop-edge-label edge-critic">Critic also trains on observed returns</p>' +
        '</div>' +
        '<div class="loop-stages">' + stages.map(function (stage, i) {
          return '<p class="fragment loop-stage" data-fragment-index="' + i + '"><b>' + (i + 1) + '</b>' + stage + '</p>';
        }).join('') + '</div>' +
        '<p class="loop-note">' + (dpo ? 'Fixed pairs; no online generation and no reward checker in this update. The reference stays frozen.'
          : grpo ? 'J<sub>clip</sub> compares πθ with the sampling snapshot πold. KL penalizes drift from frozen πref; β sets its strength. Only θ is updated.'
          : ppo ? 'Toy rewards for comparison. Full PPO uses clipped ratios to the sampling snapshot πold; LLM post-training can add KL to a frozen πref.'
          : 'Toy rewards for comparison. A is detached: no gradient through the checker or discrete sampling. Each click traces one phase.') + '</p>';
      if (dpo) host.querySelector('.loop-diagram').outerHTML = dpoTrainingDiagram();
      host.querySelectorAll('[data-tex]').forEach(function (node) {
        if (window.katex) katex.render(node.dataset.tex, node, { throwOnError: false });
        else node.textContent = '$' + node.dataset.tex + '$';
      });
      added = true;
    });
    if (added) reveal.sync();
    updateTrainingLoops();
  }

  function updateTrainingLoops() {
    document.querySelectorAll('.training-loop').forEach(function (host) {
      host.dataset.phase = String(host.querySelectorAll('.loop-stage.visible').length);
    });
  }

  function trainingLoopsReady(reveal) {
    mountTrainingLoops(reveal);
    reveal.on('fragmentshown', updateTrainingLoops);
    reveal.on('fragmenthidden', updateTrainingLoops);
    reveal.on('slidechanged', function () { mountTrainingLoops(reveal); });
    document.querySelectorAll('.training-loop').forEach(function (host) {
      host.addEventListener('click', function () { reveal.nextFragment(); });
    });
  }

  // ===================================================================
  // WIDGET: grpoGroup — set the rewards for a group of completions and
  // read off the advantages the group baseline produces.
  // ===================================================================
  function grpoGroup(host) {
    var U = WIDGET_UTIL, COL = WIDGET_UTIL.COL;
    var rewards = [1, 1, 0, 0, 1, 0, 0, 0];
    var LABELS = ['completion 1', '2', '3', '4', '5', '6', '7', '8'];

    host.innerHTML =
      '<div class="iw">' +
        '<div class="iw-canvas-wrap"><canvas class="iw-canvas"></canvas></div>' +
        '<div class="iw-stats">' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="mean">0</span><span class="iw-stat-lab">group mean</span></div>' +
          '<div class="iw-stat"><span class="iw-stat-num" data-el="std">0</span><span class="iw-stat-lab">group std</span></div>' +
          '<div class="iw-stat" data-el="sigbox"><span class="iw-stat-num" data-el="sig">0</span><span class="iw-stat-lab">gradient signal</span></div>' +
        '</div>' +
        '<div class="iw-controls">' +
          '<span class="iw-label">preset:</span>' +
          '<button class="iw-btn active" data-pick data-val="mixed">mixed group</button>' +
          '<button class="iw-btn" data-pick data-val="allright">all correct</button>' +
          '<button class="iw-btn" data-pick data-val="allwrong">all wrong</button>' +
          '<button class="iw-btn" data-pick data-val="one">one lucky solve</button>' +
        '</div>' +
        '<p class="iw-readout">Click a bar to flip that completion between correct and incorrect.</p>' +
      '</div>';
    U.stop(host);
    var canvas = host.querySelector('.iw-canvas');
    var readout = host.querySelector('.iw-readout');
    var el = {};
    host.querySelectorAll('[data-el]').forEach(function (n) { el[n.getAttribute('data-el')] = n; });

    function stats() {
      var G = rewards.length;
      var mean = rewards.reduce(function (a, b) { return a + b; }, 0) / G;
      var varr = rewards.reduce(function (a, b) { return a + (b - mean) * (b - mean); }, 0) / G;
      var std = Math.sqrt(varr);
      // With a degenerate group the denominator is guarded; every advantage is 0.
      var adv = rewards.map(function (r) { return std < 1e-8 ? 0 : (r - mean) / std; });
      return { mean: mean, std: std, adv: adv };
    }

    var bars = [];

    function draw() {
      var f = U.fit(canvas); if (!f) return;
      var ctx = f.ctx, W = f.w, H = f.h;
      ctx.clearRect(0, 0, W, H);
      var s = stats(), G = rewards.length;

      var padL = 30, padR = 20, padT = 26, padB = 34;
      var pw = W - padL - padR, ph = H - padT - padB;
      var mid = padT + ph / 2;
      var step = pw / G, bw = Math.min(56, step - 16);
      var scale = (ph / 2) / 2.0;   // advantages of +/-2 fill the half-height

      ctx.textAlign = 'center';
      ctx.font = '12px Inter, sans-serif';
      ctx.fillStyle = COL.muted;
      ctx.fillText('advantage per completion, measured from the group mean (click a bar to flip it)', W / 2, 16);

      ctx.strokeStyle = COL.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, mid); ctx.lineTo(W - padR, mid); ctx.stroke();
      ctx.textAlign = 'left';
      ctx.fillText('group mean', padL, mid - 8);
      ctx.textAlign = 'center';

      bars = [];
      for (var i = 0; i < G; i++) {
        var cx = padL + step * i + step / 2;
        var a = s.adv[i];
        var h = Math.abs(a) * scale;
        var y = a >= 0 ? mid - h : mid;
        ctx.fillStyle = a > 0.001 ? 'rgba(63,185,80,0.75)' : a < -0.001 ? 'rgba(231,76,60,0.7)' : 'rgba(136,146,164,0.35)';
        if (h < 3) { ctx.fillRect(cx - bw / 2, mid - 1.5, bw, 3); }
        else { U.rounded(ctx, cx - bw / 2, y, bw, h, 3); ctx.fill(); }

        ctx.fillStyle = COL.text;
        ctx.font = '600 12px Inter, sans-serif';
        ctx.fillText(a.toFixed(2), cx, a >= 0 ? y - 6 : y + h + 15);

        ctx.fillStyle = rewards[i] ? COL.green : COL.muted;
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText('R = ' + rewards[i], cx, H - 12);

        bars.push({ x: cx - step / 2, w: step, i: i });
      }

      el.mean.textContent = s.mean.toFixed(3);
      el.std.textContent = s.std.toFixed(3);
      var dead = s.std < 1e-8;
      el.sig.textContent = dead ? 'none' : 'yes';
      el.sigbox.className = 'iw-stat ' + (dead ? 'warn' : 'good');

      readout.innerHTML = dead
        ? 'Every completion scored the same, so the mean equals every reward and <strong>every advantage is zero</strong>. This group contributes <strong>no gradient at all</strong> &mdash; wasted samples, and the reason prompt difficulty has to be tuned.'
        : 'The baseline is just the group mean, so a completion is rewarded only for <strong>beating its peers on the same prompt</strong>. No value network computed any of this.';
    }

    canvas.addEventListener('click', function (e) {
      var rect = canvas.getBoundingClientRect();
      var x = e.clientX - rect.left;
      for (var b = 0; b < bars.length; b++) {
        if (x >= bars[b].x && x < bars[b].x + bars[b].w) {
          rewards[bars[b].i] = rewards[bars[b].i] ? 0 : 1;
          draw();
          return;
        }
      }
    });

    U.bindToggle(host, '[data-pick]', function (val) {
      if (val === 'mixed') rewards = [1, 1, 0, 0, 1, 0, 0, 0];
      else if (val === 'allright') rewards = [1, 1, 1, 1, 1, 1, 1, 1];
      else if (val === 'allwrong') rewards = [0, 0, 0, 0, 0, 0, 0, 0];
      else rewards = [0, 0, 0, 0, 0, 0, 0, 1];
      draw();
    });

    draw();
    return { resize: draw };
  }

  window.MODULE_CONFIG = {
    title: 'LLMs 0 to 100 - Module 7',
    onReady: trainingLoopsReady,
    manimSections: {
      'passk': [
        'PassKScene_0000_axes.mp4',
        'PassKScene_0001_base.mp4',
        'PassKScene_0002_rl.mp4',
        'PassKScene_0003_cross.mp4'
      ]
    },
    widgets: {
      grpoGroup: grpoGroup
    }
  };
}());
