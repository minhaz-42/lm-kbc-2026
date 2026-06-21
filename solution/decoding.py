"""Turn cached raw model samples into final ObjectEntities.

Kept separate from generation so the EXPENSIVE step (calling the model on
Kaggle) runs once and writes a raw-output cache, while the CHEAP step (parsing,
abstention, numeric aggregation) can be re-run locally for free as we iterate.

    raw cache row:  {"SubjectEntity", "Relation", "RawSamples": [str, ...]}
    prediction row: {"SubjectEntity", "Relation", "ObjectEntities": [str, ...]}
"""
from __future__ import annotations
from collections import Counter
from statistics import median
from typing import List

from relations import RELATIONS
from parsing import normalize_string, parse_prediction


def _fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return repr(x).rstrip("0").rstrip(".")


def _string_consistency(relation: str, samples: List[str], threshold: float) -> List[str]:
    """Calibrated abstention via self-consistency across N string samples.

    Each sample is parsed to a set of answers. An answer is kept only if it
    appears in >= `threshold` fraction of samples (the model is consistent about
    it); otherwise it is dropped. For single-valued relations we keep only the
    top answer, and abstain entirely (-> []) if even it is below threshold.
    This directly attacks the precision-killing over-answering seen on the
    null-heavy relations: when the model is guessing, samples disagree -> empty."""
    spec = RELATIONS[relation]
    n = len(samples)
    # map normalized form -> (count, a representative surface form)
    counts: Counter = Counter()
    surface: dict = {}
    for s in samples:
        seen = set()
        for ans in parse_prediction(relation, s):
            key = normalize_string(ans)
            if not key or key in seen:
                continue
            seen.add(key)
            counts[key] += 1
            surface.setdefault(key, ans)
    if not counts:
        return []
    if spec.multi_valued:
        keep = [k for k, c in counts.items() if c / n >= threshold]
        return [surface[k] for k in keep]
    # single-valued: top answer, but only if consistent enough
    key, c = counts.most_common(1)[0]
    return [surface[key]] if c / n >= threshold else []


def decode_samples(relation: str, samples: List[str],
                   numeric_self_consistency: bool = True,
                   string_consistency_threshold: float | None = None) -> List[str]:
    """Aggregate one (subject, relation)'s sampled completions into a final
    answer list.
      * numeric relations w/ >1 sample -> median (robust under 5% tolerance);
      * string relations w/ >1 sample AND a threshold -> consistency abstention;
      * otherwise -> decode the first sample."""
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
    if spec.kind == "string" and string_consistency_threshold is not None and len(samples) > 1:
        return _string_consistency(relation, samples, string_consistency_threshold)
    return parse_prediction(relation, samples[0] if samples else "")


def decode_cache(raw_rows: List[dict], numeric_self_consistency: bool = True,
                 string_consistency_threshold: float | None = None) -> List[dict]:
    out = []
    for r in raw_rows:
        objs = decode_samples(r["Relation"], r.get("RawSamples", []),
                              numeric_self_consistency, string_consistency_threshold)
        out.append({
            "SubjectEntity": r["SubjectEntity"],
            "Relation": r["Relation"],
            "ObjectEntities": objs,
        })
    return out
