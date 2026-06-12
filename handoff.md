# Module 6 (Finetuning) — Build Handoff

Status: outline complete (`course_outline/module_06_finetuning.md`), nothing built yet.
This document is the execution plan for building the Module 6 slides and exercise.

## Context

Module 5 (Pretraining) is finished: students train a tiny char-level NanoGPT from
scratch and end with a base model that *continues* text but does not *follow
intent*. Module 6 turns that base model into an instruction-following assistant
via supervised finetuning (SFT). The approved outline
(`course_outline/module_06_finetuning.md`, 165 lines) defines the lecture,
notable figures, side quests, resources, and exercise. The job now is to produce
the actual reveal.js deck and the runnable exercise (skeleton + solution),
matching the design, tone, and conventions of Modules 1&ndash;5.

## Decisions locked

- **Scope seam.** Module 6 owns *only* the SFT stage. The InstructGPT reward
  model and RLHF/PPO are previewed under section c but taught in Module 7. Do not
  implement RL here.
- **Exercise.** Finetune the *bundled Module 5 base checkpoint* into a toy
  instruct model on prompt&ndash;response pairs. LoRA is implemented from scratch
  (pure torch, no `peft`). The payoff is the visible before/after flip on the
  same prompt: "continues text" &rarr; "answers the instruction."
- **Manim: three step-through scenes** (confirmed): LoRA low-rank decomposition,
  loss masking, and chat-template serialization. Everything else is inline
  SVG/CSS/reveal fragments.
- **Notable figures: real sourced photos** (confirmed), photo-reveal format, for
  Long Ouyang, Amanda Askell, Edward Hu, Tim Dettmers, Jason Wei. Flag any
  identity/licensing uncertainty; fall back to a text card only if a portrait
  cannot be sourced confidently.

## Conventions and gotchas (verified this session)

Environment (one shared root `.venv`):
- Run: `uv run --no-sync python ...` and `uv run --no-sync manim ...` (plain
  `uv run` hangs when the corporate index is down).
- Only if deps change: `UV_NATIVE_TLS=true uv sync --no-config`. Do **not** create
  per-module venvs/pyprojects. The exercise must stay within current deps
  (torch, numpy, matplotlib) &mdash; implement LoRA by hand, no new libraries.

Slides build:
- Author per-section partials in `slides/module_06_finetuning/source/slides/NNN-*.md`,
  numbered by 10s. Build: `uv run python build_course.py slides/module_06_finetuning`
  (Python build; no node). Output: `index.bundled.html`.
- Canonical fences only (single-pass expander, **no nested fences** &mdash; keep
  `:::note`/`:::columns` as sibling top-level blocks): `:::divider`,
  `:::columns` (`cols=`/`grid=`, `+++` splits), `:::note` (`variant="hint"`,
  `reveal="fragment"`), `:::quiz` (prompt `+++` answer), `:::step` (code `+++`
  hint `+++` answer), `:::terminal`, `:::manim` (`scene=` slug), `:::figure`
  (`img=` `name=` `kicker=`), `:::interactive`, `:::video`.
- Verify the built bundle has zero lines starting with `:::` and no literal
  `</textarea>` (the build errors on the latter).

KaTeX (delimiters `$...$` inline, `$$...$$` display, auto-render runs after
markdown):
- Avoid two `}_{`-style subscripts in one math block (markdown pairs them into
  `<em>` and ships raw `$$`): write `\mathbf w_{\text{new}}`, not
  `\mathbf{w}_{\text{new}}`.
- Do not use spacing macros `\!` `\,` `\;` `\:` (markdown eats the backslash).
  Use a plain space; KaTeX collapses it.

Inline SVG and custom HTML:
- **No blank lines** between `<svg` and `</svg>` (a blank line ends the HTML
  block and leaks `<text>` as giant body text). Detect:
  `awk '/<svg/{i=1} i&&/^[[:space:]]*$/{print NR} /<\/svg>/{i=0}' file.md`.
- The `.reveal` root text color is **black**; any bare `<div>`/`<strong>` you add
  needs an explicit `color: var(--text-color)`. Inline SVG is safe (uses `fill=`).

Manim pipeline (see "Manim scenes" below for the full recipe). `media/` is
gitignored, so section mp4s are never committed &mdash; they are regenerated.
`images/` is **not** gitignored, so figure photos are committed.

Styling rules (apply to slides, code, docs, this file): no emojis; never `--` as
a dash (use `&mdash;`/`—`); no gradients; professional academic tone; color in
equations sparingly; put all slide text on at once (don't animate sentence by
sentence); animations get their own dedicated slide; timelines to scale.

## A. Exercise solution (build first &mdash; it produces the real slide outputs)

Mirror the Module 5 exercise layout exactly. Students edit only
`exercise.py` at the module root; `src/` is plumbing.

```
exercises/module_06_finetuning/
  README.md
  exercise.py                 <- THE ONLY student file (10 NotImplementedError TODOs)
  src/
    __init__.py
    model.py                  <- TinyGPT (copied from M5) + LoRALinear, inject/merge, loader
    tokenizer.py              <- char vocab + special tokens, encode/decode
    data.py                   <- builds the toy instruction dataset (provided)
    main.py                   <- runner: probe steps, stage-skip, print before/after
  data/
    base_model.pt             <- bundled frozen Module 5 base checkpoint (committed)
    sft_pairs.jsonl           <- bundled toy instruction-response pairs (committed)
  solution/
    exercise.py               <- full reference implementation (+ extra credit)
    src/
      __init__.py  model.py  tokenizer.py  data.py  main.py
      make_base_checkpoint.py <- offline generator for data/base_model.pt (committed)
```

### Model (`src/model.py`)
Copy `GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`, `TinyGPT` verbatim from
`exercises/module_05_pretraining/solution/src/model.py` (it keeps weight tying:
`lm_head.weight = token_embed.weight`). Add, as PROVIDED machinery except the one
blank noted per step:

- `class LoRALinear(nn.Module)` &mdash; wraps a frozen `nn.Linear` `base`; params
  `A` (shape `r x in`, normal init) and `B` (shape `out x r`, **zero init** so the
  adapter starts as a no-op); `scale = alpha / r`; optional dropout on the input.
  `forward` returns `base(x) + delta`. **Student step 5** fills `delta`.
- `inject_lora(model, r, alpha, targets=("c_attn","c_proj","fc","proj"))` &mdash;
  walk modules, replace each named `nn.Linear` with a `LoRALinear` wrapping it.
- `freeze_base_(model)` &mdash; provided helper, but **student step 6** writes the
  `p.requires_grad = False` line for base params (LoRA `A`/`B` stay trainable).
- `merge_lora_weight(base_W, A, B, scale)` &mdash; **student step 10** fills
  `base_W + scale * (B @ A)`; a provided `merge_lora(model)` swaps each
  `LoRALinear` for a plain `nn.Linear` using it.
- `load_base_model(ckpt_path, new_vocab_size)` &mdash; build `TinyGPT` at the
  enlarged vocab, load the base `state_dict`, copy the 65 trained token-embedding
  rows into the larger (tied) table, leave the special-token rows at init.
  PROVIDED (vocab expansion is not a student step).

### Tokenizer (`src/tokenizer.py`)
Base vocab = the Module 5 65 characters (carried in `base_model.pt`) plus four
**atomic special tokens** appended as their own IDs: `<|user|>`, `<|assistant|>`,
`<|end|>`, `<|pad|>` (vocab 65 &rarr; 69). `encode`/`decode` treat specials as
single IDs, not character runs (this is how real ChatML templates work, and is the
honest version to teach). Save `stoi`/`itos` in the checkpoint.

### Data (`src/data.py` + `data/sft_pairs.jsonl`)
Toy, deterministically learnable tasks so a 4-layer/width-128 char model shows a
crisp flip in a few hundred CPU steps. Weight toward the easy ones:
- **uppercase**: "uppercase: hello" &rarr; "HELLO"
- **fixed Q&A / lookup** (memorization): "capital of France?" &rarr; "Paris."
- **repeat/copy**: "repeat: cat" &rarr; "cat"
- **reverse** (minority/extra; may be imperfect &mdash; an honest teaching point):
  "reverse: cat" &rarr; "tac"
Generate a few hundred pairs from templates, bundle as `sft_pairs.jsonl`.

### Base checkpoint (`data/base_model.pt`)
Generate offline with `solution/src/make_base_checkpoint.py`: train the M5 TinyGPT
briefly (seeded) on tiny Shakespeare, save `{config, state_dict, stoi, itos}`.
~3 MB float32, committed (not gitignored). The runner loads it frozen; the
exercise never re-pretrains.

### The 10 student blanks (each a one-liner; `# TODO` / `# HINT` / `raise`)
| # | Function | Blank (solution) |
|---|----------|------------------|
| 1 | `format_example(prompt, response, special)` | assemble chat template ids: `[user] + enc(prompt) + [end] + [assistant] + enc(response) + [end]` |
| 2 | `build_targets(ids, prompt_span)` | next-token targets with prompt positions set to `-100` (mask), response positions kept |
| 3 | `masked_cross_entropy(logits, targets)` | `F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-100)` |
| 4 | optimizer over adapters | `torch.optim.AdamW(trainable, lr=SMALL_LR)` (load is provided; student picks the small LR / builds opt) |
| 5 | `LoRALinear.forward` delta | `self.scale * (self.dropout(x) @ self.A.t() @ self.B.t())` |
| 6 | `freeze_base_(model)` | `p.requires_grad = False` (loop provided) |
| 7 | `sft_train_step(...)` | `optimizer.zero_grad(set_to_none=True); loss.backward()` (clip+step provided) |
| 8 | `count_trainable_params(model)` | `sum(p.numel() for p in model.parameters() if p.requires_grad)` |
| 9 | `build_generation_prompt(prompt, special)` | template up to the assistant marker, no response: `[user] + enc(prompt) + [end] + [assistant]` |
| 10 | `merge_lora_weight(base_W, A, B, scale)` | `base_W + scale * (B @ A)` |

The autoregressive sampling loop (`generate`) is PROVIDED (reused from M5) so the
ten steps stay finetuning-focused.

### Runner (`src/main.py`)
Mirror M5's runner: `_probe_steps()` detects which of the 10 are implemented and
stage-skips on `NotImplementedError`. Hardcode bundled paths relative to the file.
Print, in order: base-model completion of a sample instruction (**before**),
trainable-vs-total parameter count, finetuning loss per checkpoint, finetuned
completion of the same instruction (**after**), and a **merge-equality check**
(merged model output == adapter model output). Run as
`uv run python module_06_finetuning/src/main.py`.

### Extra credit (answers in solution)
Full-FT vs LoRA (unfreeze all, compare counts/quality); vary rank `r`;
catastrophic-forgetting probe (base-style prompt after finetuning); loss-mask
ablation (train without the mask, watch it hallucinate prompts).

## B. Exercise skeleton (student-facing)
Same tree as the solution minus `solution/` and `make_base_checkpoint.py`. Copy
the solution files, then replace each of the 10 blanks with the three-line
`# TODO:` / `# HINT:` / `raise NotImplementedError("TODO: ...")` form (set missing
vars to `None` where a later line reads them, exactly like M5's `exercise.py`).
The bundled `data/` (checkpoint + jsonl) and all `src/` plumbing are identical and
provided. `# TODO` text must be verbatim-equal to the slide `:::step` blocks.

## C. Slides (`slides/module_06_finetuning/source/slides/`)

Create the partials below (numbered by 10s), plus `source/config.js` and a module
`source/styles.css`. Reuse M5's `styles.css` resources-grid and SVG-wrapper rules.
Use **real captured outputs** from the solution run in every `:::terminal`/sample.

| File | Content |
|------|---------|
| `000-title.md` | `:::divider` "LLMs 0 to 100", sub "Module 6: Finetuning"; tagline "From base model to assistant" |
| `010-review.md` | 2 review slides: base-vs-assistant; same optimizer/loss machinery (M2/M5), new data + behavioral target; callback to the M5 handoff |
| `020-what-is-finetuning.md` | divider + section a: definition; "same algorithm, almost" (smaller LR, fewer steps, less data); transfer-learning framing; taxonomy (continued pretraining / SFT / preference-opt forward ref); catastrophic forgetting |
| `030-sft.md` | section b: prompt&ndash;response pairs; instruction tuning generalizes (FLAN/T0/Super-NaturalInstructions); **figure: Jason Wei** |
| `040-chat-template.md` | chat templates + special tokens; **`:::manim scene="chat-template"`** (own slide); SVG static fallback ok elsewhere |
| `050-loss-masking.md` | loss masking; **`:::manim scene="loss-mask"`**; masked-CE equation `$$\mathcal{L} = -\frac{1}{|R|}\sum_{t \in R}\log p_\theta(x_t \mid x_{<t})$$`; **side quest: loss-mask ablation** |
| `060-instructgpt.md` | divider + section c: three-stage recipe (Module 6 owns stage 1); 1.3B InstructGPT preferred over 175B GPT-3; HHH; **figures: Long Ouyang, Amanda Askell**; **side quest: alignment tax** |
| `070-lora.md` | divider + section d: full-FT cost (optimizer state, per-task copies); LoRA insight + equation `$$W' = W + \frac{\alpha}{r} BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}, r \ll d$$`; **`:::manim scene="lora"`**; merge (zero latency) / swap adapters; QLoRA (fwd ref M9 quantization); PEFT family named (adapters, prefix/prompt tuning, IA^3); trade-off; **figures: Edward Hu, Tim Dettmers**; **side quest: adapters as diffs** |
| `080-data-eval.md` | section e: data quality &gt; quantity (LIMA ~1000); sources (human, distillation via Self-Instruct/Alpaca, filtered logs); hyperparams (small LR, 1&ndash;3 epochs, held-out set); evaluation is hard (fwd ref M11); **side quests: superficial alignment hypothesis, synthetic data / model collapse** |
| `090-limits.md` | section f: SFT only imitates; no negative signal; exposure bias / distribution shift; handoff to Module 7 (RLHF/DPO/GRPO) |
| `100-exercise.md` | divider + **running instructions first**; 10 `:::step` slides (code matches `exercise.py` verbatim; `# TODO` shown, `# HINT` as hint fragment, answer as nested fragment); `:::terminal` slides with **cumulative** real output; extra-credit slide |
| `110-quiz.md` | divider + the 7 `:::quiz` from the outline |
| `120-end.md` | `:::divider` "Questions?" sub "Next: Module 7 &mdash; Reinforcement Learning Post-Training"; Resources grid (two-column `resources-grid`, the outline's links) |

`config.js`: set `window.MODULE_CONFIG = { title: 'LLMs 0 to 100 - Module 6',
manimSections: { 'chat-template': [...], 'loss-mask': [...], 'lora': [...] },
widgets: {} }` &mdash; fill the section arrays after rendering (step D).

Notable figures interleave into the lecture (not a separate section); side quests
are inline (no divider). Photo-reveal format: image alone first, then name, then
animate aside and reveal kicker/contribution &mdash; the `:::figure` fence already
produces this fragment sequence.

## D. Manim scenes (`slides/module_06_finetuning/manim/scenes.py`)

Start `scenes.py` by **copying the CRF-15 partial-movie monkeypatch and the
`Text` font-shadow class from `slides/module_04_architectures/manim/scenes.py`**
(crisp kerning + sharp edges; LaTeX-free &mdash; use `Text` only). Author each
scene with `self.next_section("name")` boundaries for click-through.

- `ChatTemplateScene` (slug `chat-template`): raw user/assistant turns &rarr;
  insert role markers &rarr; flatten into one token stream &rarr; highlight the
  special tokens. Shows serialization of a multi-role conversation.
- `LossMaskScene` (slug `loss-mask`): a formatted example's token row &rarr; mark
  the prompt span as ignored (`-100`) &rarr; show only response tokens feeding the
  loss &rarr; arrow into `L`. Makes the mask concrete (pairs with the ablation).
- `LoRAScene` (slug `lora`): frozen `W` (`d x d` grid) &rarr; freeze it &rarr; add
  `B` (`d x r`) and `A` (`r x d`) &rarr; show `r << d` &rarr; parameter count
  `d^2` vs `2dr` &rarr; merge `BA` back into `W`. The one genuine spatial viz.

Render/copy/wire (per module pipeline):
1. `cd slides/module_06_finetuning/manim && uv run --no-sync manim -qp --save_sections --disable_caching scenes.py ChatTemplateScene LossMaskScene LoRAScene`
2. Copy section mp4s from `manim/media/videos/scenes/1440p60/sections/` to
   `slides/module_06_finetuning/media/sections/` (flat dir the deck serves).
   Verify one resolution: `for f in media/sections/*.mp4; do ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$f"; done | sort | uniq -c`.
3. Fill `config.js manimSections[slug] = ['<Scene>_0000_<name>.mp4', ...]` (slug =
   the fence `scene=`, filenames use the class name).
4. Rebuild the deck. Hard-reload (Cmd+Shift+R) to bust the browser video cache.
   Title-duplication: do not set `title=` on a `:::manim` fence if the scene draws
   its own title.

## E. Notable-figure photos (`slides/module_06_finetuning/images/`)

Source portraits (prefer official/institutional/freely-licensed): **Long Ouyang,
Amanda Askell, Edward Hu, Tim Dettmers, Jason Wei**. Save under `images/` (committed;
matches every prior module's `images/` dir). Reference via `:::figure
img="images/<file>" name="..." kicker="<paper, year>"`. Verify each identity;
flag uncertainty and fall back to a text contribution card only if a confident
portrait cannot be found. Kickers: Ouyang = InstructGPT (2022); Askell = HHH
"General Language Assistant as a Laboratory for Alignment" (2021); Hu = LoRA
(2021); Dettmers = QLoRA (2023); Wei = FLAN "Finetuned Language Models Are
Zero-Shot Learners" (2021).

## F. Lecture notes (`slides/module_06_finetuning/notes.md`)

Mirror M5's `notes.md`: an explanation + citation for every claim on the slides,
the two equations (masked CE, LoRA update) mapped to the visuals they appear on,
and historical context. Cover all Resources from the outline (InstructGPT, LoRA,
QLoRA, FLAN, T0, LIMA, Self-Instruct, Alpaca, HHH, adapters) plus the
1.3B&gt;175B result, LIMA's ~1000 examples, and the superficial-alignment and
model-collapse points.

## Recommended build order

1. **Exercise solution** (A) &mdash; build `make_base_checkpoint.py`, generate
   `base_model.pt` + `sft_pairs.jsonl`, write model/tokenizer/data/runner +
   `solution/exercise.py`. Run it; tune steps/LR/data until the before/after flip
   is crisp and trainable params are a small fraction of total. **Capture the real
   terminal output now** &mdash; the slides depend on it.
2. **Exercise skeleton** (B) + `README.md`.
3. **Manim scenes** (D) &mdash; author, render, copy, wire `config.js`. Do this
   before finalizing the slide partials that embed them.
4. **Figure photos** (E).
5. **Slide partials** (C) using captured outputs + rendered scenes; build the deck.
6. **notes.md** (F).
7. **Visual QA** (G, below).

## G. Verification / acceptance criteria

Exercise:
- `cd exercises && uv run --no-sync python module_06_finetuning/src/main.py`:
  before-sample continues text / ignores the instruction; after-sample answers it;
  finetuning loss falls; trainable params reported as a small fraction of total;
  merge-equality check passes.
- Skeleton: same command runs and **skips** unimplemented steps with a message
  (no crash); each implemented step unlocks the next.
- Solution `exercise.py` and the slide `:::step` code are character-identical.

Slides:
- `uv run python build_course.py slides/module_06_finetuning` succeeds; bundle has
  zero lines starting with `:::` and no literal `</textarea>`.
- Playwright QA (serve `python3 -m http.server`, navigate with a cache-buster:
  `index.bundled.html?v=2#/<slide-id>`, reveal fragments via JS, screenshot):
  no text overlaps, content fits the frame, **no raw `$$...$$`** (math rendered),
  code blocks are light-on-dark, the three manim steppers load and step, figure
  photos render and the name/kicker reveal in order.
- SVG leak check: `[...document.querySelectorAll('text')].filter(t=>!t.closest('svg')).length === 0`.

Style lint (slides, code, notes, this doc): no emojis; no `--` dashes; no
`\!`/`\,`/`\;` in math; no gradients.

## Risks and how to de-risk

- **Tiny model may not learn the toy tasks fast on CPU.** Weight the dataset toward
  uppercase + fixed Q&A (both highly learnable); keep reverse as a minority. Tune
  before capturing outputs. If reverse stays poor, present it honestly as a limit.
- **Vocab expansion of the tied embedding.** The provided `load_base_model` must
  copy the 65 trained rows and preserve weight tying; confirm the base completion
  still looks base-like and that merge-equality holds at vocab 69.
- **Manim cost (3 scenes + render + QA).** Iterate at `-ql`/`-qm`, render final at
  `-qp`; verify a single resolution across `media/sections/`; hard-reload to see
  re-renders.
- **Portrait identity/licensing.** Verify each face maps to the right name; flag
  and fall back to a text card if unsure.

## Final checklist

- [ ] `data/base_model.pt` generated and committed; `sft_pairs.jsonl` committed
- [ ] `solution/` runs end-to-end; real outputs captured for slides
- [ ] `exercise.py` skeleton: 10 TODO/HINT/raise blanks, runner stage-skips
- [ ] `README.md` (running, what-to-implement table, dataset, extra credit)
- [ ] 3 Manim scenes rendered, copied, wired in `config.js`; single resolution
- [ ] 5 figure photos sourced into `images/`
- [ ] 13 slide partials authored; deck builds clean; `:::step` == `exercise.py`
- [ ] `notes.md` with citations + both equations mapped to slides
- [ ] Playwright QA passed on every slide; style lint clean

