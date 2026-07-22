#!/usr/bin/env python3
"""Figures for the LM-KBC 2026 paper. Matplotlib -> vector PDF (+ PNG preview).
Numbers are the validation macro-F1 scores computed by solution/eval.py on
preds_val.jsonl (Qwen2.5-14B) and preds_craftx_val.jsonl (Gemma-4-31B); the
0.5B abstention row is from the local dry run. Regenerate: python make_figures.py
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "savefig.dpi": 300, "figure.dpi": 130,
    "axes.grid": True, "grid.alpha": 0.30, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
# distinct palette (slate / purple / teal / warm-orange headline)
C = {"empty": "#9AA0A6", "base": "#8E44AD", "14b": "#16A085", "30b": "#E67E22"}
C3 = "#C0392B"  # third accent (crimson) for a per-relation line in the sweep

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig); print("wrote", name)

# ---- Fig: per-relation macro-F1, 4 systems ----
rels = ["borders", "company", "death", "area", "capacity", "award"]
empty = [0.265, 0.360, 0.390, 0.000, 0.000, 0.000]
base  = [0.665, 0.354, 0.210, 0.290, 0.180, 0.101]
m14   = [0.918, 0.609, 0.450, 0.330, 0.090, 0.096]
m30   = [0.971, 0.737, 0.420, 0.330, 0.170, 0.140]
x = np.arange(len(rels)); w = 0.2
fig, ax = plt.subplots(figsize=(7.0, 2.7))
for i, (vals, key, lab) in enumerate([(empty,"empty","Empty"),(base,"base","Baseline"),
                                      (m14,"14b","Qwen2.5-14B"),(m30,"30b","Gemma-4-31B")]):
    ax.bar(x + (i-1.5)*w, vals, w, label=lab, color=C[key], edgecolor="black", linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(rels); ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1.02)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False, columnspacing=1.1)
save(fig, "fig_bars")

# ---- Fig: abstention threshold sweep (14B, 3 null-heavy) ----
fig, a1 = plt.subplots(figsize=(3.4, 2.5))
th = [0, 0.34, 0.5, 0.67, 1.0]
a1.plot(th, [0.912,0.918,0.918,0.925,0.925], "o-", color=C["14b"], label="borders", ms=4)
a1.plot(th, [0.590,0.609,0.609,0.613,0.613], "s-", color=C["base"], label="company", ms=4)
a1.plot(th, [0.420,0.450,0.450,0.410,0.410], "^-", color=C3, label="death", ms=4)
a1.set_xlabel(r"voting threshold $\theta$"); a1.set_ylabel("macro-F1"); a1.set_ylim(0.38,0.98)
a1.legend(frameon=False, loc="center right")
save(fig, "fig_ablation")

# ---- Fig: model-scale (avg-of-6 macro-F1) ----
fig, ax = plt.subplots(figsize=(3.3, 2.4))
names = ["Empty", "Baseline", "0.5B", "14B", "30B"]
avg6  = [0.169, 0.300, 0.108, 0.416, 0.461]
cols  = [C["empty"], C["base"], "#C9CCD1", C["14b"], C["30b"]]
b = ax.bar(names, avg6, color=cols, edgecolor="black", linewidth=0.3)
ax.set_ylabel("avg-of-6 macro-F1"); ax.set_ylim(0, 0.52)
for r, v in zip(b, avg6): ax.text(r.get_x()+r.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontsize=7)
ax.set_xticklabels(names, rotation=15)
save(fig, "fig_scale")
print("done")
