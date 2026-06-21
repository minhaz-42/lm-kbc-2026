"""Generate kaggle_lmkbc.ipynb — a self-contained Kaggle notebook that clones
the solution repo + official data, runs inference on 2xT4, and scores val.
Run:  python solution/make_notebook.py
"""
import json, os

REPO_URL = "https://github.com/YOUR_USERNAME/lm-kbc-2026.git"  # <-- edit after you push

def md(*lines): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}
def code(*lines): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": [l + "\n" for l in lines]}

cells = [
    md("# LM-KBC 2026 — inference on Kaggle (2×T4)",
       "",
       "**Settings → Accelerator → GPU T4 ×2**, and **Internet → On**.",
       "",
       "Pipeline: clone solution + data → run `solution/run.py` → score on val with `solution/eval.py`.",
       "Edit `MODEL` and `BACKEND` below. `hf` (4-bit) is the most reliable on Kaggle; `vllm` is faster."),

    md("## 1. Config",
       "",
       "**Model menu** (all verified ≤32.0B total params — quantization does NOT reduce the counted size):",
       "- `Qwen/Qwen3-14B` or `Qwen/Qwen2.5-14B-Instruct` — **dev**: fits 1×T4, fast iteration.",
       "- `Qwen/Qwen3-30B-A3B-Instruct-2507` — **compete (default)**: 30.5B total / 3.3B active MoE, Apache-2.0, vLLM-proven, ~14B speed → lots of self-consistency. Needs 2×T4.",
       "- `google/gemma-4-31B-it` — **upgrade**: 30.7B dense, strongest knowledge + 140+ langs, but verify your vLLM/transformers version supports Gemma-4 first.",
       "- ❌ NOT `Qwen2.5-32B` (32.5B) / `Qwen3-32B` (32.8B) — both OVER the cap.",
       "- For fast vLLM on 2×T4, prefer a pre-quantized **AWQ/GPTQ** checkpoint of the chosen model if one exists."),
    code("MODEL   = 'Qwen/Qwen3-30B-A3B-Instruct-2507'   # see menu above; must be <=32.0B total params",
        "BACKEND = 'hf'        # 'hf' (transformers+bitsandbytes 4-bit, most reliable) or 'vllm' (faster)",
        "TP      = 2           # vllm tensor-parallel across the 2 T4s (ignored for hf; hf uses device_map=auto)",
        "SPLIT   = 'val'       # 'val' to score locally, 'test' for the real submission",
        "SC      = 5           # self-consistency samples for numeric relations",
        "REPO_URL = '" + REPO_URL + "'",
        "",
        "# Sanity-check the param cap before spending GPU time:",
        "from transformers import AutoConfig",
        "try:",
        "    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)",
        "    print('loaded config for', MODEL)",
        "except Exception as e:",
        "    print('!! could not load config — check the model id / version:', e)"),

    md("## 2. Install deps"),
    code("import sys",
        "!pip -q install 'transformers>=4.51.0' accelerate bitsandbytes loguru pyyaml 2>/dev/null",
        "if BACKEND == 'vllm':",
        "    !pip -q install vllm 2>/dev/null",
        "print('deps installed')"),

    md("## 3. Get code + data",
       "Clones your solution repo and the official dataset repo (always-current val/test)."),
    code("import os",
        "os.chdir('/kaggle/working')",
        "!rm -rf lm-kbc-2026 dataset2026",
        "!git clone -q $REPO_URL lm-kbc-2026 || echo 'EDIT REPO_URL (or upload solution/ as a Kaggle dataset)'",
        "!git clone -q --depth 1 https://github.com/lm-kbc/dataset2026.git",
        "# solution/ may live at repo root or under solution/ — find it",
        "import glob, shutil",
        "cand = glob.glob('/kaggle/working/lm-kbc-2026/**/run.py', recursive=True)",
        "SOL = os.path.dirname(cand[0]) if cand else '/kaggle/working/lm-kbc-2026/solution'",
        "print('solution dir:', SOL)",
        "!ls $SOL"),

    md("## 4. Run inference",
       "Writes `preds.jsonl`. For the real submission set `SPLIT='test'` (no gold to score against)."),
    code("INPUT = f'/kaggle/working/dataset2026/data/{SPLIT}.jsonl'",
        "TRAIN = '/kaggle/working/dataset2026/data/train.jsonl'",
        "OUT   = '/kaggle/working/preds.jsonl'",
        "args = f'--backend {BACKEND} --model {MODEL} -i {INPUT} --train {TRAIN} -o {OUT} --sc-samples {SC}'",
        "if BACKEND == 'vllm': args += f' --tp {TP}'",
        "!cd $SOL && python run.py {args}"),

    md("## 5. Score (val only)"),
    code("if SPLIT == 'val':",
        "    !cd $SOL && python eval.py -p $OUT -g $INPUT",
        "else:",
        "    print('test split — submit preds.jsonl to Codabench / save as the submission file')"),

    md("## 6. Save submission artifact"),
    code("import shutil",
        "shutil.copy(OUT, f'/kaggle/working/submission_{SPLIT}.jsonl')",
        "print('saved /kaggle/working/submission_%s.jsonl' % SPLIT)"),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(__file__), "kaggle_lmkbc.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
