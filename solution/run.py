"""End-to-end runner: (subject, relation) -> predictions.jsonl

    # local plumbing (no GPU): oracle should ~match gold, empty -> the 0.203 floor
    python solution/run.py --backend oracle --input dataset2026/data/val.jsonl -o /tmp/oracle.jsonl
    python solution/run.py --backend empty  --input dataset2026/data/val.jsonl -o /tmp/empty.jsonl

    # Kaggle (2xT4) with vLLM:
    python solution/run.py --backend vllm --model Qwen/Qwen2.5-14B-Instruct \
        --tp 2 --input dataset2026/data/val.jsonl -o preds_val.jsonl

Numeric relations use self-consistency (sample N, take the median) — robust to
outliers and well-suited to the 5% tolerance metric. String relations default
to greedy decoding.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import defaultdict
from statistics import median
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relations import RELATIONS, ALL_RELATIONS          # noqa: E402
from prompts import build_messages, gold_answer_list     # noqa: E402
from decoding import decode_samples                       # noqa: E402


def read_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_generator(args, gold_map):
    if args.backend == "oracle":
        from generators import OracleMock
        return OracleMock(gold_map)
    if args.backend == "empty":
        from generators import EmptyMock
        return EmptyMock()
    if args.backend == "vllm":
        from generators import VLLMGenerator
        return VLLMGenerator(args.model, tensor_parallel_size=args.tp,
                             quantization=args.quantization,
                             max_model_len=args.max_model_len)
    if args.backend == "hf":
        from generators import HFGenerator
        return HFGenerator(args.model, load_in_4bit=not args.no_4bit)
    raise ValueError(args.backend)


def run(args):
    rows = read_jsonl(args.input)
    train = read_jsonl(args.train)
    if args.relations:
        keep = set(args.relations.split(","))
        rows = [r for r in rows if r["Relation"] in keep]
    if args.limit:
        # keep a balanced-ish slice: first N per relation
        per = defaultdict(int); sliced = []
        for r in rows:
            if per[r["Relation"]] < args.limit:
                sliced.append(r); per[r["Relation"]] += 1
        rows = sliced

    gold_map = {(r["SubjectEntity"], r["Relation"]): gold_answer_list(r)
                for r in rows} if args.backend == "oracle" else {}

    gen = build_generator(args, gold_map)

    by_rel: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_rel[r["Relation"]].append(r)

    raw_rows: List[Dict] = []          # the expensive-to-produce generation cache
    predictions: List[Dict] = []
    for rel, rel_rows in by_rel.items():
        spec = RELATIONS[rel]
        prompts = [gen.apply_template(build_messages(spec, r["SubjectEntity"], train))
                   for r in rel_rows]

        numeric_sc = spec.kind == "numeric" and args.sc_samples > 1 and args.backend in ("vllm", "hf")
        if numeric_sc:
            comps = gen.generate(prompts, spec.max_new_tokens,
                                 temperature=args.sc_temperature, n=args.sc_samples)
        else:
            comps = gen.generate(prompts, spec.max_new_tokens, temperature=0.0, n=1)

        for r, sample_list in zip(rel_rows, comps):
            raw_rows.append({
                "SubjectEntity": r["SubjectEntity"],
                "Relation": rel,
                "RawSamples": sample_list,
            })
            predictions.append({
                "SubjectEntity": r["SubjectEntity"],
                "Relation": rel,
                "ObjectEntities": decode_samples(rel, sample_list, numeric_self_consistency=numeric_sc),
            })
        print(f"  [{rel}] {len(rel_rows)} rows done", file=sys.stderr)

    # raw cache: re-decode locally with decode.py instead of re-running the model
    raw_path = args.raw_out or (os.path.splitext(args.output)[0] + ".raw.jsonl")
    with open(raw_path, "w") as f:
        for r in raw_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.output, "w") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(predictions)} predictions -> {args.output}", file=sys.stderr)
    print(f"wrote raw generation cache -> {raw_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["vllm", "hf", "oracle", "empty"])
    ap.add_argument("--model", default=None)
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("--train", default=None)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--raw-out", default=None, dest="raw_out",
                    help="path for the raw generation cache (default: <output>.raw.jsonl)")
    ap.add_argument("--tp", type=int, default=1, help="vLLM tensor_parallel_size")
    ap.add_argument("--quantization", default=None, help="e.g. awq, gptq, bitsandbytes")
    ap.add_argument("--max-model-len", type=int, default=4096, dest="max_model_len")
    ap.add_argument("--no-4bit", action="store_true", help="(hf) disable 4-bit")
    ap.add_argument("--sc-samples", type=int, default=5, help="self-consistency samples for numeric relations")
    ap.add_argument("--sc-temperature", type=float, default=0.7)
    ap.add_argument("--relations", default=None, help="comma-separated subset")
    ap.add_argument("--limit", type=int, default=0, help="max rows per relation (debug)")
    args = ap.parse_args()
    if args.train is None:
        args.train = os.path.join(os.path.dirname(args.input), "train.jsonl")
    run(args)


if __name__ == "__main__":
    main()
