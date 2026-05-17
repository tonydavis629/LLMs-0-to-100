# Module 2 Revision — Handover

Status snapshot for the Module 2 (Perceptrons & Optimization) overhaul. The
slide content, lecture notes, and coding exercise are **done and verified**.
The remaining work is the **Manim animation domain** (rewrite `scenes.py`,
render scene-by-scene, wire media + `index.html`, rebuild, visual QA).

Repo root: `/Users/davisac1/github/LLMs-0-to-100`

---

## 1. Original request (the user's change list)

Lecture: remove filler ("Today..", "Last class.."); simpler cross-entropy-as-loss
wording; biological-analogy diagram; activation-function diagrams; fix unrendered
"why linearity matters" LaTeX; trim universal-approximation; matrix-graph diagram;
linear-separability video must show AND + OR separable **before** XOR fails and
show a perceptron plotted as a line/decision classifier; XOR-hidden-layer diagram
+ space out the output-neuron text; rewrite the folding animation so it correctly
shows the **W matrix as the fold/transformation** with a line that actually
separates the classes; new animation showing an MLP boundary is made of multiple
lines; remove the "Perceptron vs Cortex" slide; backprop slide needs a diagram
(express path Loss → Output → chain rule → Weight → Δw), remove the BCE/sigmoid
block, mention the chain rule; move "Without Backprop" before the backprop slide;
move computation graphs after gradient descent; SGD animation; one unified
animation of a 2-weight perceptron's 2D loss landscape (loss as 3D height+color,
perceptron diagram color-coded to the axes) showing GD, SGD, and Adam; fix the
"best model" marker on the overfitting chart + show 1D underfit/correct/overfit
fits; remove the embedding-arithmetic slide. Exercise: use ReLU as an explicit
activation in the MLP (not sigmoid), update extra credit too.

---

## 2. DONE (do not redo — verify by reading the files if unsure)

### Exercise (`exercises/module_02_perceptrons/`)
- `exercise.py` and `solution/exercise.py`: added explicit `relu()` (student
  fills `np.maximum(0.0, z)`). `mlp_forward` now `relu` hidden + `sigmoid`
  output. `mlp_gradients` (extra credit / solution) backprops through ReLU
  (`relu_grad = (z1 > 0).astype(float)`). Single-neuron steps 1–4 unchanged
  (still sigmoid + BCE).
- `src/main.py` and `solution/src/main.py`: section headers renamed
  `STEP 1-5/6/7` → `PART 1/2/3` (decoupled from walkthrough step numbers).
  Runner still skips `NotImplementedError` gracefully.
- `README.md`: step table + prose updated for ReLU.
- **Verified by running the solution:** PART 1 single neuron linear data
  `149/150 (99.3%)`; PART 2 single neuron XOR `80/160 (50.0%)`, loss plateau
  `0.6926`; PART 3 ReLU MLP `160/160 (100.0%)`, loss → `0.0024`
  (epochs 100→500: 0.0114, 0.0059, 0.0040, 0.0030, 0.0024).
  These exact numbers are already baked into the slide terminal mockups.
  Output plots `output/step7_*.png` were regenerated.

### Slides (`slides/module_02_perceptrons/source/slides.md`)
All edits applied as small surgical Edits and sanity-checked (no stale refs):
- Trimmed filler; simplified cross-entropy review; trimmed universal-approx;
  trimmed folding text; trimmed loss-functions intro.
- Fixed the "Why Nonlinearity Matters" KaTeX (removed `\underbrace`, now two
  clean `$$` lines).
- Added inline **SVG** diagrams: biological neuron + artificial neuron;
  4 activation curves (step/sigmoid/tanh/relu); matrix↔graph equivalence;
  XOR hidden-layer network (output neuron now in its own `highlight-box`);
  backprop chain-rule pipeline diagram (forward blue arrows, backward orange
  arrows, ends in Δw = w − η∂L/∂w).
- Restructured backprop: "Why We Need Backprop" now **before** "The
  Backpropagation Algorithm" (which now has the diagram, no BCE/sigmoid block,
  mentions the chain rule + update-rule path). "Computation Graphs" moved to
  **after** the gradient-descent section.
- Removed "Perceptron vs Cortex" and "Embedding Arithmetic" slides.
- Added new slide "Many Lines Make a Curve" + its animation slide.
- Removed the standalone gradient-descent animation and the separate
  loss-landscape animation; the unified optimizer animation replaces both.
- Exercise walkthrough rewritten: Step 6 = `relu()`, Step 7 = ReLU MLP
  `mlp_forward`, Step 5 = "Why the Single Neuron Fails" concept slide,
  terminal mockups use real PART 1/2/3 output, extra credit = ReLU backprop.
  Code blocks match `exercise.py`.

### Lecture notes (`slides/module_02_perceptrons/notes.md`)
Synced: ReLU activation note + course pairing; new "Multi-Layer Decision
Boundaries" section; backprop chain-rule path + update rule; 2-weight
loss-landscape visualization note; removed both side-quest sections; new
"Exercise Structure" + "MLP with ReLU Hidden Layer" subsections.

---

## 3. REMAINING WORK (the Manim domain)

> A background subagent was tried for this and **died on `ECONNRESET` ~25 min
> in, making zero file changes** (`scenes.py` still has the original 6 classes,
> mtime Apr 6). The lesson: **do not** render all scenes in one long run.
> Author `scenes.py` once, then render **one scene at a time** in separate
> short commands, QA each, fix, move on. Each scene render is an independent,
> recoverable unit.

`scenes.py` currently still has the OLD classes:
`XORVisualizationScene`, `FoldingScene`, `GradientDescentScene`,
`LossLandscapeScene`, `OverfittingCurveScene`, `DataVisualizationScene`.

### Scene → slide key mapping (slides.md ALREADY uses these — do NOT change slides)

| New class | `data-manim-scene` key | Replaces |
|---|---|---|
| `LinearSeparabilityScene` | `linsep-viz` | XORVisualizationScene |
| `FoldingScene` (rewrite) | `folding-viz` | FoldingScene |
| `MLPBoundaryScene` (new) | `mlp-boundary-viz` | — |
| `OptimizerLandscapeScene` (new, 3D) | `optimizer-viz` | GradientDescentScene + LossLandscapeScene |
| `OverfittingCurveScene` (fix + extend) | `overfit-viz` | OverfittingCurveScene |
| `DataVisualizationScene` (UNCHANGED) | `data-viz` | — keep class & existing media as-is |

Delete `XORVisualizationScene`, `GradientDescentScene`, `LossLandscapeScene`.
**Do not modify or re-render `DataVisualizationScene`** — its existing
`media/sections/DataVisualizationScene_0000_data_linear.mp4` and
`_0001_data_xor.mp4` are still valid.

### Render pipeline (facts gathered)

- Manim venv: `slides/module_02_perceptrons/manim/.venv`, **manim 0.20.1**.
  Run via `uv run --project manim ...` from inside the module dir.
- ffmpeg: `/opt/homebrew/bin/ffmpeg` (for last-frame QA).
- Quality: existing media is **1080p60** → use `-qh`. If a scene (esp. the 3D
  one) is too slow at `-qh`, fall back to `-qm`; mixed resolutions are fine
  (videos scale via CSS).
- Sections come from `--save_sections`; each `self.next_section("name")`
  becomes `media/videos/scenes/1080p60/sections/<Scene>_<NNNN>_<name>.mp4`.
- The deck plays from `media/sections/<same-basename>.mp4`.

Recommended per-scene command (verify with the first one):
```bash
cd /Users/davisac1/github/LLMs-0-to-100/slides/module_02_perceptrons
uv run --project manim manim render manim/scenes.py LinearSeparabilityScene -qh --save_sections
# confirm files land in media/videos/scenes/1080p60/sections/
```
After all scenes render, copy + clean stale:
```bash
cp media/videos/scenes/1080p60/sections/*.mp4 media/sections/
# remove old scene files no longer referenced:
rm media/sections/XORVisualizationScene_*.mp4 \
   media/sections/GradientDescentScene_*.mp4 \
   media/sections/LossLandscapeScene_*.mp4
```

### Verified manim 0.20.1 API
`ThreeDAxes.plot_surface` exists (the old `LossLandscapeScene` used it and
rendered fine — reuse its pattern: `colorscale=[PRIMARY,"#2a6bbf",SECONDARY,
RED_C], fill_opacity=0.75`). `Surface.set_fill_by_value`, `TracedPath`,
`Dot3D`, `VMobject.apply_function` all exist.

### Theme constants (already at top of scenes.py — reuse)
`BG="#0a0e1a" TEXT="#e8eaf0" MUTED="#8892a4" PRIMARY="#4a9eff"
SECONDARY="#f5a623" RED_C="#e74c3c" GREEN_C="#3fb950" LINE_C="#2a3450"`.

### CLAUDE.md constraints that apply
- No emojis. No decorative gradients (a 3D surface colored by loss height via
  `colorscale` IS allowed/desired — it's data encoding).
- Never use `--` in on-screen text; use `—` or rephrase.
- Step-through: every distinct step is its own `self.next_section(name,
  skip_animations=False)`. Keep `run_time`s short (0.3–1.5 s).
- Background `#0a0e1a` on every scene.

---

## 4. Per-scene specs (the design — so you don't re-derive it)

### LinearSeparabilityScene → `linsep-viz`
Sections in order:
1. `perceptron_as_line` — 2D axes (x1,x2). Show `y = step(w·x + b)`. Draw the
   line `w·x+b=0` (e.g. w=(1,1), b=−1 → x1+x2=1). Tint the two half-planes
   (a coarse ~18×18 mesh of low-opacity squares colored by sign of w·x+b is
   the most robust way; green = predict 1, red = predict 0). Label "the
   perceptron is a linear classifier; the line is its decision boundary".
2. `and_separable` — AND points (0,0)=0,(0,1)=0,(1,0)=0,(1,1)=1
   (green=1, red=0). Draw a separating line `x1+x2=1.5`. Label "AND: one line works".
3. `or_separable` — OR points (0,0)=0 else 1. Line `x1+x2=0.5`.
   Label "OR: one line works".
4. `xor_fails` — XOR points (0,0)=0,(0,1)=1,(1,0)=1,(1,1)=0. Rotate/translate
   a line through several positions, each visibly misclassifying a point;
   conclude "XOR: no single line works". (Adapt the rotating-line code from
   the old `XORVisualizationScene`.)

### FoldingScene → `folding-viz`
The old one was wrong (dots teleported; line didn't separate; no sense of W
as a transform). Use the **abs/ReLU fold** that genuinely separates XOR:

- XOR clusters (with noise) near unit-square corners: class 0 at (0,0) & (1,1);
  class 1 at (0,1) & (1,0).
- Fold map (data coords): `X = 1.3*(x1 − x2)`, `Y = 1.5*abs((x1+x2) − 1)`.
  Then class 1 (x1+x2≈1) → Y≈0; class 0 (x1+x2≈0 or 2) → Y≈1.5.
- Sections: `fold_input` (faint grid + points, "not linearly separable");
  `fold_transform` (apply the SAME map to a sampled grid VGroup **and** the
  points so the plane visibly folds along the crease x1+x2=1 — build grid
  lines as VMobjects of sampled points in data coords via `axes.c2p`, build
  target VMobjects from mapped points, `Transform`); `fold_separate` (draw a
  horizontal dashed line at Y=0.75 that **actually** separates the folded
  classes; caption "the weight matrix W folds the space"). Caption the
  transform arrow `\sigma(W\mathbf{x}+\mathbf{b})`.

### MLPBoundaryScene → `mlp-boundary-viz`
Show an MLP boundary = many straight lines. Easiest correct dataset: class 1 =
points inside a diamond/box, class 0 outside (so 4 hidden-neuron lines form a
closed piecewise-linear boundary — XOR's two separate blobs can't be enclosed
by one convex region, avoid that). Sections: `data` (scatter); `lines` (add
3–4 labeled hidden-neuron lines h₁…h₄ one at a time); `regions` (shade the
half-planes / their intersection); `boundary` (highlight the final polygon;
caption "the curved-looking boundary is straight lines stitched together").

### OptimizerLandscapeScene → `optimizer-viz` (ThreeDScene, heaviest)
Sections: `perceptron`, `surface`, `gradient_descent`, `sgd`, `adam`.
- `perceptron`: a 2-weight perceptron diagram (x1,x2 → neuron). Edge/label
  **w₁ in PRIMARY**, **w₂ in SECONDARY**. Text "(w₁,w₂) is one point in
  weight space". Use `add_fixed_in_frame_mobjects`.
- `surface`: synthesize a tiny fixed-seed 2-blob dataset. Loss(w1,w2) =
  logistic **BCE + 0.05·(w1²+w2²)** (the L2 term gives a clean finite bowl;
  pure logistic on separable data has no finite min). `axes.plot_surface`
  over w∈[−3,3]², height=loss, `colorscale=[PRIMARY,"#2a6bbf",SECONDARY,
  RED_C]`. Color x-axis PRIMARY, y-axis SECONDARY (match the diagram). Drop a
  `Dot3D` marker at a high-loss start e.g. (−2.5, 2.5); briefly move it to
  show "changing weights moves the point".
- `gradient_descent` / `sgd` / `adam`: real updates in numpy.
  `grad = mean((sigmoid(w·x) − y)·x) + 0.1·w`. GD = full batch (smooth path).
  SGD = minibatch, fixed seed (visibly noisier/zig-zag). Adam = m,v moments
  (faster, well-damped). For each: reset ball to the same start, build the
  path as a VMobject through `axes.c2p(w1,w2,loss)` points, then
  `self.play(Create(path), MoveAlongPath(ball, path), run_time≈2.5)`. Use a
  distinct trail color per optimizer; fade the previous trail on reset.
  Keep surface `resolution≈(28,28)`, ≤~15 steps, minimal/zero ambient
  rotation to keep `-qh` render time bounded.

### OverfittingCurveScene → `overfit-viz`
Keep the curve build-up from the old scene but **fix the Best-model marker**:
the old code hard-codes the dashed line at x=30; instead compute
`argmin(val_loss)` numerically over `np.linspace(1,100,2000)` (true min ≈
x≈32.5 for the existing `val_loss`) and place the dashed line + "Best model"
label there. Add a final section `fits`: 3 small side-by-side axes over a 1D
function (e.g. `f(x)=sin(1.2x)` on [0,6]) with the SAME noisy sample points in
all three; draw the fitted curve for **Underfit** (deg-1 polyfit), **Good fit**
(the true function / deg-5), **Overfit** (deg ≈ n−1 polyfit, wild
oscillation). Label each panel.

### DataVisualizationScene → `data-viz`
**Unchanged.** Keep the original class verbatim; keep its existing media files.

---

## 5. Final wiring + QA

1. After all scenes render and section mp4s are copied to `media/sections/`
   (stale old ones removed), update **only** the `MANIM_SECTIONS` JS object in
   `slides/module_02_perceptrons/source/index.html` (~lines 40–73): six keys
   `linsep-viz / folding-viz / mlp-boundary-viz / optimizer-viz / overfit-viz
   / data-viz`, each → ordered array of the real generated section basenames
   (order by the `_NNNN_` index). Change nothing else in index.html.
2. Build the bundle:
   `node slides/build.mjs slides/module_02_perceptrons`
   (bundles `source/slides.md` into `source/index.html` → `index.bundled.html`,
   repoints styles.css). Run this **after** index.html is updated.
3. Visual QA with the Playwright MCP: `python3 -m http.server` in
   `slides/module_02_perceptrons/`, navigate `index.bundled.html#/<slide-id>`,
   reveal fragments, screenshot. Check: SVG diagrams not clipped/overlapping,
   activation curves render, KaTeX renders (esp. `why-nonlinearity`),
   exercise code matches `exercise.py`, terminal numbers match section 2 above,
   each animation slide plays its sections in order and the final frames are
   clean (no label collisions; folding line actually separates; optimizer
   trails sit on the surface; Best-model line at the val-loss min).
4. For animation QA without a browser: extract last frames with
   `ffmpeg -y -sseof -0.2 -i <mp4> -frames:v 1 /tmp/qa.png` and read them.

---

## 6. Task list state

Open tasks (see the session task tool): write `scenes.py`; then
render+QA each of LinearSeparability / Folding / MLPBoundary /
OptimizerLandscape / Overfitting individually; then wire media + index.html +
build + final QA. Exercise + slides.md + notes.md tasks are completed.

## 7. Gotchas

- The subagent approach for Manim failed on a connection drop with no output.
  Prefer in-process, scene-by-scene renders (recoverable units).
- `slides.md` is one file the build inlines whole — cannot be split across
  parallel editors. It is already finished; don't reopen it for parallel work.
- `slides.md` already references the six new scene keys; rendering must
  produce sections under those exact class names or the deck shows blank
  videos.
- Don't run `build.mjs` until `index.html`'s `MANIM_SECTIONS` is updated, or
  you'll bundle a stale map and have to rebuild.
- Keep the exercise terminal numbers in the slides in sync with an actual
  `uv run python module_02_perceptrons/solution/src/main.py` (current numbers
  are correct as of this handover).
