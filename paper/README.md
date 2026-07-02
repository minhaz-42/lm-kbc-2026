# LM-KBC 2026 system paper (`paper/`)

LaTeX source for *"Closed-Book Knowledge Base Construction with Calibrated
Abstention and Numeric Self-Consistency."* ACL format (`acl.sty`). Compiles to
`main.pdf`; the numbered body (§1–8) fits the **6-page** limit.

## Build
```bash
cd paper && tectonic main.tex        # self-contained, no TeX install needed
# or upload paper/ to Overleaf, root = main.tex
```
- `main.tex` shows the author (mode `[final]`). **For the double-blind OpenReview
  submission, change `[final]` → `[review]`** to re-anonymize.
- Do **not** add `\bibliographystyle` — `acl.sty` sets it (a duplicate errors).

## Layout (modular, one file per section)
```
main.tex             preamble, title, \input of sections, \bibliography
references.bib        ~36 entries
acl.sty, acl_natbib.bst   ACL style files (vendored)
sections/            00_abstract … 10_ethics  (one .tex per section)
figures/             make_figures.py (matplotlib) + fig_*.pdf/.png
```

## Figures
| File | Shows | Source |
|---|---|---|
| Fig 1 (in `04_method`) | system-pipeline architecture (numbered stages, per-relation branch, worked examples) | inline TikZ |
| `figures/fig_bars.pdf` | per-relation macro-F1: Empty / Baseline / Qwen2.5-14B / Gemma-4-31B | `make_figures.py` |
| `figures/fig_ablation.pdf` | abstention threshold sweep + numeric n-sweep (14B) | `make_figures.py` |
| `figures/fig_scale.pdf` | avg-of-6 macro-F1 across models (0.5B→14B→30B) | `make_figures.py` (spare) |

Regenerate the plots: `python figures/make_figures.py` (matplotlib; numbers are
the validation scores from `solution/eval.py`).

## Headline result (validation)
Gemma-4-31B: **0.459** relation-averaged macro-F1 / **0.490** micro — beats the
official baseline on all six relations (Qwen2.5-14B: 0.416 / 0.447).
