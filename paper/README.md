# Paper (6-page ACL, double-blind via OpenReview)

`main.tex` is the system-description paper. It is in **anonymous review mode**
(`\usepackage[review]{acl}`) and uses placeholder macros:

- `\result{...}` (red) — a number to fill after the final Kaggle run.
- `\TODO{...}` (blue) — prose/analysis to add once we have outputs.

Grep `result`/`TODO` before submitting; nothing red/blue should remain.

## What's already written (real, final)
Intro, task & **metric analysis** (the empty-baseline finding), method
(scope-grounded prompts, calibrated abstention, numeric self-consistency, robust
parsing), setup, limitations. Empty-baseline numbers in Table 2 are measured.

## What needs the final run
Model id + params, the `Ours`/`$\Delta$` columns of Table 2, the ablation table,
the error analysis, and the abstract/intro/conclusion `\result{XX.X}` figures.

## Build
Easiest: **Overleaf** → New Project → upload `main.tex` + `refs.bib`, then add
the official ACL style files (`acl.sty`, `acl_natbib.bst`) from the
[ACL template](https://github.com/acl-org/acl-style-files) or the EMNLP 2026
template (Overleaf has it under *Templates*). Or locally:

```bash
# place acl.sty + acl_natbib.bst next to main.tex, then:
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

For camera-ready: change `[review]` to `[]` and fill in author names.
