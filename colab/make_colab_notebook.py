"""Generate a SELF-CONTAINED Google Colab notebook (embeds the solution code via
%%writefile, so no repo access needed). Open it in Colab on a T4 GPU and Run all.

    python colab/make_colab_notebook.py val  Qwen/Qwen2.5-14B-Instruct
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.join(HERE, "..", "solution")
MODULES = ["relations.py", "prompts.py", "parsing.py", "decoding.py",
           "generators.py", "run.py", "eval.py"]

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-14B-Instruct"
SC = 5


def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": [l + "\n" for l in src.split("\n")]}
def md(src): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in src.split("\n")]}


cells = [
    md(f"# LM-KBC 2026 — Colab run ({SPLIT}, {MODEL})\n"
       "**Before running:** Runtime → Change runtime type → **T4 GPU** → Save.\n"
       "Then **Runtime → Run all**. At the end it downloads `preds_{split}.jsonl` and "
       "`raw_{split}.jsonl` — send me **raw_{split}.jsonl**."),
    code("import torch, os\n"
         "assert torch.cuda.is_available(), 'No GPU! Runtime -> Change runtime type -> T4 GPU'\n"
         "print('GPU:', torch.cuda.get_device_name(0))\n"
         "os.makedirs('solution', exist_ok=True)"),
    code("!pip -q install 'transformers>=4.51.0' accelerate bitsandbytes 'numpy<2' 2>/dev/null\n"
         "print('deps ok')"),
]

for m in MODULES:
    with open(os.path.join(SOL, m)) as f:
        body = f.read()
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": [f"%%writefile solution/{m}\n"] + [body]})

cells += [
    code("!git clone -q --depth 1 https://github.com/lm-kbc/dataset2026.git\n"
         "print('data:', os.listdir('dataset2026/data'))"),
    code(f"MODEL={MODEL!r}; SPLIT={SPLIT!r}; SC={SC}\n"
         "INPUT=f'dataset2026/data/{SPLIT}.jsonl'; TRAIN='dataset2026/data/train.jsonl'; OUT=f'/content/preds_{SPLIT}.jsonl'"),
    # `!` streams output LIVE (download bar + per-relation progress) — never looks frozen.
    code("!cd solution && python run.py --backend hf --model {MODEL} -i ../{INPUT} --train ../{TRAIN} -o {OUT} --sc-samples {SC} --string-samples 3"),
    code("!cd solution && python eval.py -p {OUT} -g ../{INPUT}" if SPLIT == "val"
         else "print('test split: submit', OUT, 'to Codabench')"),
    code("raw=OUT.replace('.jsonl','.raw.jsonl')\n"
         "for f in (OUT, raw):\n"
         "    print(f, os.path.getsize(f) if os.path.exists(f) else 'MISSING', 'bytes')\n"
         "# Browser Colab: auto-download. VS Code Colab extension: right-click these\n"
         "# paths in the file panel -> Download (into the EMNLP folder).\n"
         "try:\n"
         "    from google.colab import files\n"
         "    files.download(OUT)\n"
         "    if os.path.exists(raw): files.download(raw)\n"
         "except Exception as e:\n"
         "    print('(auto-download unavailable here:', e, ')')\n"
         "    print('Download these two files manually:', OUT, raw)"),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = os.path.join(HERE, "lmkbc_colab.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, f"({len(cells)} cells, model={MODEL}, split={SPLIT})")
