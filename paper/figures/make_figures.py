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
empty = [0.25, 0.35, 0.45, 0.00, 0.00, 0.00]
base  = [0.679, 0.320, 0.160, 0.310, 0.100, 0.069]
m14   = [0.900, 0.616, 0.480, 0.330, 0.090, 0.081]
m30   = [0.953, 0.722, 0.460, 0.330, 0.170, 0.120]
x = np.arange(len(rels)); w = 0.2
fig, ax = plt.subplots(figsize=(7.0, 2.7))
for i, (vals, key, lab) in enumerate([(empty,"empty","Empty"),(base,"base","Baseline"),
                                      (m14,"14b","Qwen2.5-14B"),(m30,"30b","Gemma-4-31B")]):
    ax.bar(x + (i-1.5)*w, vals, w, label=lab, color=C[key], edgecolor="black", linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(rels); ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1.02)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False, columnspacing=1.1)
save(fig, "fig_bars")

# ---- Fig: two-panel ablations (threshold sweep + numeric n-sweep) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.5))
th = [0, 0.34, 0.5, 0.67, 1.0]
a1.plot(th, [0.902,0.900,0.900,0.907,0.907], "o-", color=C["14b"], label="borders", ms=4)
a1.plot(th, [0.603,0.616,0.616,0.621,0.621], "s-", color=C["base"], label="company", ms=4)
a1.plot(th, [0.450,0.480,0.480,0.470,0.470], "^-", color=C3, label="death", ms=4)
a1.set_xlabel(r"voting threshold $\theta$"); a1.set_ylabel("macro-F1"); a1.set_ylim(0.4,0.98)
a1.set_title("null-heavy relations"); a1.legend(frameon=False, loc="center right")
n = [1,2,3,4,5]
a2.plot(n, [0.30,0.28,0.31,0.30,0.33], "o-", color=C["30b"], label="hasArea", ms=4)
a2.plot(n, [0.10,0.04,0.08,0.06,0.09], "s-", color=C["base"], label="hasCapacity", ms=4)
a2.set_xlabel(r"# median samples $n$"); a2.set_ylim(0,0.4); a2.set_xticks(n)
a2.set_title("numeric relations"); a2.legend(frameon=False)
save(fig, "fig_ablation")

# ---- Fig: model-scale (avg-of-6 macro-F1) ----
fig, ax = plt.subplots(figsize=(3.3, 2.4))
names = ["Empty", "Baseline", "0.5B", "14B", "30B"]
avg6  = [0.175, 0.273, 0.108, 0.416, 0.459]
cols  = [C["empty"], C["base"], "#C9CCD1", C["14b"], C["30b"]]
b = ax.bar(names, avg6, color=cols, edgecolor="black", linewidth=0.3)
ax.set_ylabel("avg-of-6 macro-F1"); ax.set_ylim(0, 0.52)
for r, v in zip(b, avg6): ax.text(r.get_x()+r.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontsize=7)
ax.set_xticklabels(names, rotation=15)
save(fig, "fig_scale")
print("done")
