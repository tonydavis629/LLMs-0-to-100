(function () {
  // Module 7 config. Manim steppers are keyed by the `scene` slug used in the
  // :::manim fences; each maps to its ordered list of section clips.
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
    manimSections: {
      'reinforce': [
        'REINFORCEScene_0000_samples.mp4',
        'REINFORCEScene_0001_reward.mp4',
        'REINFORCEScene_0002_weight.mp4',
        'REINFORCEScene_0003_estimator.mp4'
      ],
      'ppo': [
        'PPOScene_0000_actor.mp4',
        'PPOScene_0001_critic.mp4',
        'PPOScene_0002_advantage.mp4',
        'PPOScene_0003_clip.mp4'
      ],
      'dpo': [
        'DPOScene_0000_pair.mp4',
        'DPOScene_0001_reward.mp4',
        'DPOScene_0002_margin.mp4',
        'DPOScene_0003_update.mp4'
      ],
      'grpo': [
        'GRPOScene_0000_prompt.mp4',
        'GRPOScene_0001_group.mp4',
        'GRPOScene_0002_reward.mp4',
        'GRPOScene_0003_advantage.mp4',
        'GRPOScene_0004_update.mp4'
      ],
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
