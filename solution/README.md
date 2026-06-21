# LM-KBC 2026 — our solution

Closed-book knowledge-base construction: given `(subject, relation)`, predict the
complete set of object strings. Entry for the [AKBC/EMNLP 2026 shared task](https://www.akbc.ws/2026/shared-task.html).

## Approach (why it's built this way)

The single most important fact about the metric: **predicting nothing scores
macro-F1 = 0.203 on val**, and the official baseline only reaches 0.284. Points
come from (1) numeric factual recall (`hasCapacity`, `hasArea`), (2) *calibrated
abstention* on null-heavy relations, (3) precision on the rest. So the design is:

- **Per-relation prompts** (`relations.py`, `prompts.py`) that state the *exact*
  ground-truth scope (land-only borders, city granularity, max capacity, …) and
  force `{"answers": [...]}` JSON output.
- **Abstention as a feature**: null-possible relations get mixed empty/non-empty
  few-shot exemplars + an explicit "don't guess when unsure" rule.
- **Self-consistency for numerics**: sample N, take the median (robust under the
  5% tolerance metric).
- **Robust parser** (`parsing.py`) tolerant to code fences, prose, commas, units;
  normalization is byte-identical to the official `evaluate.py`.

## Layout

| file | role | needs GPU? |
|---|---|---|
| `relations.py` | per-relation config + scope definitions | no |
| `prompts.py` | builds chat prompts + few-shot | no |
| `parsing.py` | model-text → clean `List[str]` | no |
| `generators.py` | vLLM / HF-4bit / mock backends | GPU for real ones |
| `run.py` | end-to-end → `predictions.jsonl` **+ `*.raw.jsonl` cache** | depends on backend |
| `decoding.py` | raw samples → ObjectEntities (parse + numeric median) | no |
| `decode.py` | re-decode a raw cache → predictions | no |
| `eval.py` | per-relation macro-F1 vs empty floor | no |
| `test_parsing.py` | parser unit tests | no |
| `kaggle_lmkbc.ipynb` | run on Kaggle 2×T4 | yes |

## Develop locally (no GPU)

```bash
python solution/test_parsing.py                 # parser unit tests
# full pipeline plumbing with mock backends:
python solution/run.py --backend empty  -i dataset2026/data/val.jsonl -o /tmp/e.jsonl
python solution/run.py --backend oracle -i dataset2026/data/val.jsonl -o /tmp/o.jsonl
python solution/eval.py -p /tmp/o.jsonl -g dataset2026/data/val.jsonl   # oracle == 1.000
```

`empty` reproduces the 0.203 floor; `oracle` pushes gold through the real
parser and must score 1.000 — together they prove the plumbing before any GPU
time is spent. Iterate on prompts/parsing here for free.

## Run on Kaggle (the real model)

1. Push this repo to **public GitHub** (required for the submission anyway).
2. Open `kaggle_lmkbc.ipynb` on Kaggle → Accelerator **GPU T4 ×2**, Internet **On**.
3. Set `REPO_URL`, `MODEL`, `BACKEND`, `SPLIT` in cell 1; Run All.
4. `SPLIT='val'` prints the per-relation scoreboard; `SPLIT='test'` produces the
   submission file for Codabench.

### Iterate without burning GPU

`run.py` writes `*.raw.jsonl` — the model's raw outputs. Download it once, then
re-tune parsing/abstention/numeric aggregation locally for free:

```bash
python solution/decode.py -r raw_val.jsonl -o preds.jsonl   # no GPU
python solution/eval.py   -p preds.jsonl   -g dataset2026/data/val.jsonl
```

Only re-run the Kaggle notebook when you change the *prompts* or the *model*.

## Submission checklist (deadline Aug 15 2026)

- [ ] `submission_test.jsonl` → Codabench competition 16267
- [ ] public GitHub repo with this code
- [ ] 6-page ACL paper → OpenReview (LM-KBC Shared Task)
