"""Re-decode a raw generation cache into predictions — NO model / GPU needed.

Run the model once on Kaggle (run.py writes <output>.raw.jsonl), download that
cache, then iterate on parsing / abstention / numeric aggregation here for free:

    python solution/decode.py -r preds.raw.jsonl -o preds.jsonl
    python solution/eval.py   -p preds.jsonl -g dataset2026/data/val.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoding import decode_cache          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--raw", required=True, help="raw generation cache (.raw.jsonl)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--no-numeric-sc", action="store_true",
                    help="disable median self-consistency for numeric relations")
    args = ap.parse_args()

    with open(args.raw) as f:
        raw_rows = [json.loads(l) for l in f if l.strip()]
    preds = decode_cache(raw_rows, numeric_self_consistency=not args.no_numeric_sc)
    with open(args.output, "w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"decoded {len(preds)} rows -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
