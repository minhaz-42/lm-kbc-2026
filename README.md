# Closed-Book Knowledge Base Construction with Calibrated Abstention and Numeric Self-Consistency

System entry for the **AKBC @ EMNLP 2026 Shared Task** (5th LM-KBC edition):
given a `(subject, relation)` pair, predict the *complete* set of object strings
(zero, one, or many). Strictly **closed-book**, **open-weight ≤ 32B parameters**,
**no retrieval, no fine-tuning**.

> Task page: <https://www.akbc.ws/2026/shared-task.html> · Leaderboard: Codabench ·
> Paper: [`paper/`](paper/) (ACL format, ≤ 6 pages of main content; camera-ready)

## Headline results (validation, relation-averaged macro-F1 / micro-F1)

| System | avg-of-6 | micro | Notes |
|---|---|---|---|
| Empty baseline (predict nothing) | 0.169 | 0.195 | measured floor |
| Official baseline (Qwen3.5-9B) | 0.300 | 0.313 | organiser few-shot |
| **Qwen2.5-14B** (local, 4-bit) | 0.415 | 0.442 | with self-consistency voting |
| **Gemma-4-31B** (API, single sample) | **0.461** | **0.488** | **primary system** |

Gemma-4-31B beats the official baseline on **five of the six relations** (all but
hasCapacity). The main
empirical finding: at the ≤ 32B budget, **model scale outweighs decode-time
self-consistency** — a single-sample 30B beats a multi-sample 14B.

## Approach

Everything follows from one property of the metric: because a correct **empty**
prediction scores a perfect F1, abstention is high-value. The pipeline adds no
trainable parameters and combines standard components:

- **Scope-grounded prompts** (`relations.py`, `prompts.py`): each relation states its
  *exact* ground-truth scope and forces `{"answers": [...]}` JSON.
- **Calibrated abstention** (`decoding.py`): for null-heavy relations, sample `N=3`,
  keep an object only if it appears in ≥ `θ` of samples (fixed `θ=0.5`, majority),
  else emit `∅`.
- **Numeric self-consistency**: sample `n=5`, take the **median** (robust under the
  5% tolerance).
- **Robust parser** (`parsing.py`): normalization byte-identical to the official
  `evaluate.py`.

## Repository layout

```
solution/     pipeline: relations/prompts/parsing/decoding, run.py, decode.py, eval.py, tests
predictions/  validation outputs for both models (+ raw generation caches)
paper/         ACL LaTeX source, figures, and compiled PDF
colab/ kaggle/ notebook generators for GPU runs
```
Dataset and official `evaluate.py` are **not** redistributed here — get them from the
[organiser repository](https://www.akbc.ws/2026/shared-task.html) into `dataset2026/`.

## Reproduce

```bash
pip install -r solution/requirements.txt

# 1. Sanity-check the plumbing with no GPU (empty floor + oracle==1.000):
python solution/run.py  --backend empty  -i dataset2026/data/val.jsonl -o /tmp/e.jsonl
python solution/run.py  --backend oracle -i dataset2026/data/val.jsonl -o /tmp/o.jsonl
python solution/eval.py -p /tmp/o.jsonl  -g dataset2026/data/val.jsonl   # 1.000

# 2. Generate with an open-weight model.
#    Gemma-4-31B via an OpenAI-compatible API (set your key in the environment):
export LLM_API_KEY=...           # never commit keys
python solution/run.py --backend openai --base-url <api-base-url> \
       --model "<gemma-4-31b-id>" -i dataset2026/data/val.jsonl \
       -o predictions/val_gemma-4-31b.jsonl --string-samples 1
#    (or run Qwen2.5-14B locally in 4-bit on a T4 via kaggle_lmkbc.ipynb)

# 3. Re-decode the cached raw generations offline (NO GPU) and score:
python solution/decode.py -r predictions/val_gemma-4-31b.raw.jsonl -o /tmp/p.jsonl
python solution/eval.py   -p /tmp/p.jsonl -g dataset2026/data/val.jsonl
```

Every result in the paper is a deterministic, GPU-free function of the cached
`*.raw.jsonl` files in `predictions/`, so the decode stage reproduces without a GPU.

## Predictions

`predictions/` holds the validation outputs and their raw generation caches:

| File | Model | Purpose |
|---|---|---|
| `val_gemma-4-31b.jsonl` | Gemma-4-31B | primary submission-format predictions |
| `val_qwen2.5-14b.jsonl` | Qwen2.5-14B | self-consistency ablation model |
| `*.raw.jsonl` | both | raw samples for GPU-free re-decoding |

## Compliance

Open-weight models only (≤ 32B total parameters, counting quantization-free);
closed-book (no web search / RAG / external corpus / KB lookup); no fine-tuning or
continued pretraining. API hosting is used purely as compute for the open weights.
