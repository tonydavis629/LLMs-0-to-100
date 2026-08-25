"""Generate an illustrative ViT patch-token norm heatmap for the registers side quest.

Left: without registers, a few patches develop very high norm (artifact / accidental
scratch space). Right: with dedicated register tokens, the dense features are clean.
This is a schematic illustration of Darcet et al., "Vision Transformers Need Registers."
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
GRID = 16
BG = "#111a2b"
TEXT = "#e8eaf0"
SUB = "#8fa0bd"

def base_field():
    # smooth-ish low-magnitude feature norm across patches
    f = 0.18 + 0.06 * rng.standard_normal((GRID, GRID))
    return np.clip(f, 0.05, 0.4)

# without registers: inject a handful of high-norm artifact patches
without = base_field()
artifacts = [(3, 12), (6, 4), (9, 9), (11, 13), (13, 2)]
for (r, c) in artifacts:
    without[r, c] = 1.0
    # slight bleed into neighbors
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < GRID and 0 <= cc < GRID:
            without[rr, cc] = max(without[rr, cc], 0.45)

# with registers: clean, uniform low norm
with_reg = base_field()

fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.0))
fig.patch.set_facecolor(BG)

titles = ["Without registers", "With register tokens"]
fields = [without, with_reg]
subs = ["high-norm scratch patches",
        "clean, interpretable features"]

for ax, field, title, sub in zip(axes, fields, titles, subs):
    im = ax.imshow(field, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title, color=TEXT, fontsize=15, pad=12)
    ax.set_xlabel(sub, color=SUB, fontsize=10.5, labelpad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3550")

# shared colorbar
cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
cbar.set_label("patch token norm", color=TEXT, fontsize=11)
cbar.ax.yaxis.set_tick_params(color=SUB)
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=SUB)
cbar.outline.set_edgecolor("#2a3550")

out = "../images/register_attention.png"
fig.savefig(out, dpi=140, facecolor=BG, bbox_inches="tight")
print("saved", out)
