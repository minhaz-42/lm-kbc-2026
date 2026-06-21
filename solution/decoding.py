"""Turn cached raw model samples into final ObjectEntities.

Kept separate from generation so the EXPENSIVE step (calling the model on
Kaggle) runs once and writes a raw-output cache, while the CHEAP step (parsing,
abstention, numeric aggregation) can be re-run locally for free as we iterate.

    raw cache row:  {"SubjectEntity", "Relation", "RawSamples": [str, ...]}
    prediction row: {"SubjectEntity", "Relation", "ObjectEntities": [str, ...]}
"""
from __future__ import annotations
from statistics import median
from typing import List

from relations import RELATIONS
from parsing import parse_prediction


def _fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return repr(x).rstrip("0").rstrip(".")


def decode_samples(relation: str, samples: List[str],
                   numeric_self_consistency: bool = True) -> List[str]:
    """Aggregate one (subject, relation)'s sampled completions into a final
    answer list. Numeric relations with >1 sample use the median (robust under
    the 5% tolerance metric); everything else decodes the first sample."""
    spec = RELATIONS[relation]
    if spec.kind == "numeric" and numeric_self_consistency and len(samples) > 1:
        nums = []
        for s in samples:
            p = parse_prediction(relation, s)
            if p:
                try:
                    nums.append(float(p[0]))
                except ValueError:
                    pass
        return [_fmt_num(median(nums))] if nums else []
    return parse_prediction(relation, samples[0] if samples else "")


def decode_cache(raw_rows: List[dict], numeric_self_consistency: bool = True) -> List[dict]:
    out = []
    for r in raw_rows:
        objs = decode_samples(r["Relation"], r.get("RawSamples", []), numeric_self_consistency)
        out.append({
            "SubjectEntity": r["SubjectEntity"],
            "Relation": r["Relation"],
            "ObjectEntities": objs,
        })
    return out
