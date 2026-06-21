"""Score a predictions file against gold, with a per-relation breakdown and a
comparison against the 'predict nothing' floor.

    python solution/eval.py -p preds_val.jsonl -g dataset2026/data/val.jsonl
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os

REPO_EVAL = os.path.join(os.path.dirname(__file__), "..", "dataset2026", "evaluate.py")


def load_evaluator():
    spec = importlib.util.spec_from_file_location("ev", REPO_EVAL)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def macro_table(ev, pred_rows, gt_rows):
    sc = ev.evaluate_per_sr_pair(pred_rows, gt_rows, ev.RELATION_TYPE, 0.05)
    return ev.macro_average_per_relation(sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--predictions", required=True)
    ap.add_argument("-g", "--ground_truth", required=True)
    args = ap.parse_args()

    ev = load_evaluator()
    gt = ev.read_jsonl_file(args.ground_truth)
    pred = ev.read_jsonl_file(args.predictions)

    # empty-prediction floor on the same gold
    empty = [{"SubjectEntity": r["SubjectEntity"], "Relation": r["Relation"],
              "ObjectEntities": []} for r in gt]

    me = macro_table(ev, empty, gt)
    mp = macro_table(ev, pred, gt)

    per_rels = [k for k in mp if k != "*** All Relations ***"]
    print(f"{'relation':32} {'macro-f1':>9} {'(empty)':>9} {'Δ':>7}   {'P':>5} {'R':>5}")
    print("-" * 74)
    for rel in per_rels:
        f1 = mp[rel]["macro-f1"]; ef1 = me[rel]["macro-f1"]
        p = mp[rel]["macro-p"]; r = mp[rel]["macro-r"]
        flag = "  <-- below empty!" if f1 + 1e-9 < ef1 else ""
        print(f"{rel:32} {f1:>9.3f} {ef1:>9.3f} {f1-ef1:>+7.3f}   {p:>5.2f} {r:>5.2f}{flag}")
    print("-" * 74)
    # Two summaries. The 2025 edition ranked by AVG-OF-RELATIONS (each relation
    # 1/6 weight) — treat that as the headline. All-pairs is what evaluate.py prints.
    avg6 = sum(mp[r]["macro-f1"] for r in per_rels) / len(per_rels)
    eavg6 = sum(me[r]["macro-f1"] for r in per_rels) / len(per_rels)
    ap = mp["*** All Relations ***"]["macro-f1"]
    eap = me["*** All Relations ***"]["macro-f1"]
    print(f"{'AVG of relations (likely rank)':32} {avg6:>9.3f} {eavg6:>9.3f} {avg6-eavg6:>+7.3f}")
    print(f"{'All-pairs (evaluate.py)':32} {ap:>9.3f} {eap:>9.3f} {ap-eap:>+7.3f}")


if __name__ == "__main__":
    main()
