"""Generate a SELF-CONTAINED Kaggle notebook that embeds the solution code
(via %%writefile cells) so it runs headless via the Kaggle API with NO private
repo clone. Re-run whenever solution/*.py changes.

    python kaggle/make_selfcontained_notebook.py val   Qwen/Qwen2.5-14B-Instruct
    python kaggle/make_selfcontained_notebook.py test  Qwen/Qwen3-30B-A3B-Instruct-2507
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.join(HERE, "..", "solution")
# runtime modules to embed (decode.py/test/make_* not needed on Kaggle)
MODULES = ["relations.py", "prompts.py", "parsing.py", "decoding.py",
           "generators.py", "run.py", "eval.py"]

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-14B-Instruct"
BACKEND = sys.argv[3] if len(sys.argv) > 3 else "hf"
SC = 5


def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": [l + "\n" for l in src.split("\n")]}
def md(src): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in src.split("\n")]}


cells = [
    md(f"# LM-KBC 2026 — self-contained run ({SPLIT}, {MODEL})\nEmbeds the solution code; no external repo needed."),
    code("import os, shutil, subprocess\n"
         "os.makedirs('solution', exist_ok=True)\n"
         "try:\n"
         "    import torch; print('cuda_available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\n"
         "    assert torch.cuda.is_available(), 'NO GPU attached — enable Accelerator (and phone-verify the Kaggle account)'\n"
         "except Exception as e:\n"
         "    print('GPU CHECK:', e)"),
    code("!pip -q install 'transformers>=4.51.0' accelerate bitsandbytes 'numpy<2' 2>/dev/null\nprint('deps ok')"),
]

# one %%writefile cell per module
for m in MODULES:
    with open(os.path.join(SOL, m)) as f:
        body = f.read()
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": [f"%%writefile solution/{m}\n"] + [body]})

cells += [
    code("!git clone -q --depth 1 https://github.com/lm-kbc/dataset2026.git\nprint('data:', os.listdir('dataset2026/data'))"),
    code(f"MODEL   = {MODEL!r}\nBACKEND = {BACKEND!r}\nSPLIT   = {SPLIT!r}\nSC      = {SC}"),
    code("from transformers import AutoConfig\n"
         "cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)\n"
         "print('config loaded for', MODEL)"),
    code("INPUT='dataset2026/data/%s.jsonl'%SPLIT; TRAIN='dataset2026/data/train.jsonl'; OUT='/kaggle/working/preds_%s.jsonl'%SPLIT\n"
         "cmd=f'cd solution && python run.py --backend {BACKEND} --model {MODEL} -i ../{INPUT} --train ../{TRAIN} -o {OUT} --sc-samples {SC}'\n"
         "print(cmd)\n"
         "import subprocess,sys\n"
         "p=subprocess.run(cmd,shell=True,capture_output=True,text=True)\n"
         "print(p.stdout[-3000:]); print('STDERR:', p.stderr[-3000:])"),
    code("if SPLIT=='val':\n"
         "    r=subprocess.run(f'cd solution && python eval.py -p {OUT} -g ../{INPUT}',shell=True,capture_output=True,text=True)\n"
         "    print(r.stdout); print(r.stderr[-1500:])"),
    code("# artifacts: /kaggle/working/preds_{split}.jsonl + .raw.jsonl are auto-saved as kernel output\n"
         "print(sorted(f for f in os.listdir('/kaggle/working') if f.endswith('.jsonl')))"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(HERE, "kernel.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, f"({len(cells)} cells, model={MODEL}, split={SPLIT}, backend={BACKEND})")
