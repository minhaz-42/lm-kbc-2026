"""Tests for self-consistency decoding (numeric median + string abstention).
    python solution/test_decoding.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from decoding import decode_samples

def J(*xs):  # build sample completions
    import json
    return [json.dumps({"answers": list(x)}) for x in xs]

CASES = [
    # numeric median over 3 samples
    ("hasCapacity", J(["35000"], ["36000"], ["34000"]), {"numeric_self_consistency": True}, ["35000"]),
    ("hasArea",     J(["100"], ["102"], ["5000"]),       {"numeric_self_consistency": True}, ["102"]),  # median rejects outlier

    # single-valued string: consistent -> answer; inconsistent -> abstain
    ("personHasCityOfDeath", J(["Paris"], ["Paris"], ["Paris"]), {"string_consistency_threshold": 0.5}, ["Paris"]),
    ("personHasCityOfDeath", J(["Paris"], ["Lyon"], ["Berlin"]), {"string_consistency_threshold": 0.5}, []),  # all disagree -> abstain
    ("personHasCityOfDeath", J(["London"], ["London"], ["Oxford"]), {"string_consistency_threshold": 0.5}, ["London"]),  # 2/3
    ("personHasCityOfDeath", J([], [], ["NY"]), {"string_consistency_threshold": 0.5}, []),  # mostly empty -> abstain

    # multi-valued string: keep answers above threshold
    ("countryLandBordersCountry",
     J(["France","Spain"], ["France","Spain","Italy"], ["France","Spain"]),
     {"string_consistency_threshold": 0.5}, ["France","Spain"]),   # Italy only 1/3 -> dropped

    # single sample falls back to plain parse
    ("personHasCityOfDeath", ['{"answers": ["Rome"]}'], {"string_consistency_threshold": 0.5}, ["Rome"]),
]

def run():
    fails = 0
    for rel, samples, kw, expected in CASES:
        got = decode_samples(rel, samples, **kw)
        # order-independent compare for multi-valued
        ok = sorted(map(str.lower, got)) == sorted(map(str.lower, expected))
        if not ok:
            fails += 1
            print(f"FAIL [{rel}] kw={kw}\n   expected {expected}\n   got      {got}")
    print(f"\ndecoding cases: {len(CASES)-fails}/{len(CASES)} passed")
    return fails

if __name__ == "__main__":
    sys.exit(1 if run() else 0)
