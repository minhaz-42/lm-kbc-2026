# Paper (6-page ACL, double-blind via OpenReview)

`main.tex` is the complete system-description paper, in **anonymous review mode**
(`\usepackage[review]{acl}`). **All numbers are filled in** from the real
Qwen2.5-14B validation run — no placeholders remain.

## Headline results (validation)
- **0.416** relation-averaged macro-F1 (**0.447** over all instances)
- vs empty baseline 0.175 / 0.203 and official baseline 0.284
- Abstention ablation: 0.5B 0.173→0.372; 14B 0.409→0.417 (the nice finding:
  abstention rescues weak models, refines strong ones)

## Before submission
- Switch `[review]` → `[]` and add author names for camera-ready.
- If a larger model / test-set run is done later, update Tables 2–3.

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
