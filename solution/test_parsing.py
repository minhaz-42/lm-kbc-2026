"""Local sanity tests for parsing — runs with NO model/GPU.
    python solution/test_parsing.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util

from parsing import parse_prediction, normalize_string, parse_number

CASES = [
    # (relation, raw_model_text, expected_parsed)
    ("countryLandBordersCountry", '{"answers": ["Haiti"]}', ["Haiti"]),
    ("countryLandBordersCountry", '```json\n{"answers": ["Guinea", "Liberia"]}\n```', ["Guinea", "Liberia"]),
    ("countryLandBordersCountry", 'Sure! {"answers": ["France", "France", "Spain"]} hope that helps', ["France", "Spain"]),
    ("countryLandBordersCountry", '{"answers": []}', []),
    ("countryLandBordersCountry", 'The answer is none.', []),
    ("countryLandBordersCountry", '["Austria", "Belgium"]', ["Austria", "Belgium"]),
    ("personHasCityOfDeath", '{"answers": ["Groningen"]}', ["Groningen"]),
    ("personHasCityOfDeath", '{"answers": []}', []),
    ("personHasCityOfDeath", '{"answers": ["Paris", "Lyon"]}', ["Paris"]),  # single-valued -> keep 1
    ("personHasCityOfDeath", 'still alive', []),
    ("companyTradesAtStockExchange", '{"answers": ["Tokyo Stock Exchange", "Nagoya Stock Exchange"]}',
     ["Tokyo Stock Exchange", "Nagoya Stock Exchange"]),
    ("companyTradesAtStockExchange", '{"answers": ["not publicly traded"]}', []),
    ("hasCapacity", '{"answers": ["35000"]}', ["35000"]),
    ("hasCapacity", '{"answers": ["35,000 people"]}', ["35000"]),
    ("hasCapacity", 'The capacity is about 35000.', ["35000"]),
    ("hasCapacity", '{"answers": ["18 thousand"]}', ["18000"]),
    ("hasArea", '{"answers": ["15.4"]}', ["15.4"]),
    ("hasArea", '{"answers": ["20,770 km2"]}', ["20770"]),
    ("hasArea", '{"answers": ["123.8"]}', ["123.8"]),
    ("awardWonBy", '{"answers": ["Max Theiler", "Ivan Pavlov", "Max Theiler"]}', ["Max Theiler", "Ivan Pavlov"]),
]


def run():
    failed = 0
    for rel, raw, expected in CASES:
        got = parse_prediction(rel, raw)
        ok = got == expected
        if not ok:
            failed += 1
            print(f"FAIL [{rel}] raw={raw!r}\n      expected={expected}\n      got     ={got}")
    print(f"\nparse cases: {len(CASES)-failed}/{len(CASES)} passed")

    # Confirm our normalize_string matches the official evaluator's exactly.
    repo_eval = os.path.join(os.path.dirname(__file__), "..", "dataset2026", "evaluate.py")
    if os.path.exists(repo_eval):
        spec = importlib.util.spec_from_file_location("ev", repo_eval)
        ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
        probe = ["Côte d'Ivoire", "  São Paulo!! ", "NEW YORK", "Ås, Norway", "naïve  café"]
        mismatch = [p for p in probe if ev.normalize_string(p) != normalize_string(p)]
        print(f"normalize_string matches evaluate.py: {'YES' if not mismatch else 'NO ' + str(mismatch)}")
    else:
        print("(evaluate.py not found — skipped normalize parity check)")

    return failed


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
