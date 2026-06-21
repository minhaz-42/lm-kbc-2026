"""Per-relation configuration for the LM-KBC 2026 shared task.

Everything that differs between the six relations lives here: the precise scope
definition (copied from the official relation definitions — the model is told
these), how the answer should be shaped, how many objects to expect, generation
budget, and whether an empty answer is plausible (abstention lever).

This module has NO heavy dependencies so it can be imported and unit-tested on
a laptop with no GPU.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class RelationSpec:
    name: str
    kind: str                      # "numeric" | "string"
    # One-sentence task shown to the model, phrased as an instruction.
    instruction: str
    # The official scope definition the ground truth was built against.
    definition: str
    # Natural-language description of the expected answer cardinality / shape.
    answer_shape: str
    allow_empty: bool              # can the correct answer legitimately be empty?
    multi_valued: bool             # can there be many objects?
    max_new_tokens: int            # generation budget (awardWonBy needs a lot)
    # how many few-shot exemplars to include from train (mix of empty/non-empty)
    few_shot: int = 6
    # unit hint for numeric relations
    unit: str = ""
    # use self-consistency voting to abstain (set on null-heavy relations)
    consistency_abstain: bool = False


RELATIONS: dict[str, RelationSpec] = {
    "countryLandBordersCountry": RelationSpec(
        name="countryLandBordersCountry",
        kind="string",
        instruction="List every country that shares a LAND border with the subject country.",
        definition=(
            "Countries (or comparable territories) that share a LAND border with the "
            "subject. Maritime/sea borders are EXCLUDED (e.g. Russia-Japan, Samoa-USA "
            "do NOT count). An island country with no land border has an EMPTY answer. "
            "Only currently-recognised states count."
        ),
        answer_shape="A list of country names (English common name). Empty list if the country is an island with no land neighbour.",
        allow_empty=True,
        multi_valued=True,
        consistency_abstain=True,
        max_new_tokens=128,
        few_shot=8,
    ),
    "personHasCityOfDeath": RelationSpec(
        name="personHasCityOfDeath",
        kind="string",
        instruction="Give the CITY where the subject person died.",
        definition=(
            "The CITY where the person died (city granularity, not country or region). "
            "If the person is still alive, or the city of death is unknown, the answer "
            "is EMPTY."
        ),
        answer_shape="Exactly one city name, or an empty list if the person is alive / city unknown.",
        allow_empty=True,
        multi_valued=False,
        consistency_abstain=True,
        max_new_tokens=48,
        few_shot=8,
    ),
    "companyTradesAtStockExchange": RelationSpec(
        name="companyTradesAtStockExchange",
        kind="string",
        instruction="List the stock exchange(s) on which the subject company's shares are publicly traded.",
        definition=(
            "The stock exchange(s) on which the company's shares are publicly traded. "
            "Multiple listings are possible. A private company or a subsidiary that is "
            "NOT separately listed has an EMPTY answer."
        ),
        answer_shape="A list of stock exchange names (e.g. 'New York Stock Exchange', 'Tokyo Stock Exchange'). Empty if not publicly listed.",
        allow_empty=True,
        multi_valued=True,
        consistency_abstain=True,
        max_new_tokens=64,
        few_shot=8,
    ),
    "awardWonBy": RelationSpec(
        name="awardWonBy",
        kind="string",
        instruction=("Exhaustively list EVERY recipient/winner of the subject award you can "
                     "recall — aim for as many as possible (dozens; these awards often have "
                     "tens to hundreds of recipients). Do NOT stop at the few most famous; "
                     "keep going until you genuinely cannot recall more. Only include names "
                     "you are reasonably confident actually received THIS specific award."),
        definition=(
            "Entities (people or organisations) that have received the SPECIFIC award "
            "named by the subject. Predecessor/successor awards with different names are "
            "DISTINCT and must not be bundled. Some awards have hundreds of recipients."
        ),
        answer_shape="A long list of recipient names. List every winner you can recall.",
        allow_empty=False,
        multi_valued=True,
        max_new_tokens=2048,
        few_shot=2,
    ),
    "hasCapacity": RelationSpec(
        name="hasCapacity",
        kind="numeric",
        instruction="Give the maximum spectator capacity of the subject venue, as a single integer number of people.",
        definition=(
            "The MAXIMUM spectator capacity of the venue, as an integer number of people "
            "(Wikidata P1083). When several capacities exist (seated vs total, "
            "before/after renovation), use the HIGHEST published capacity."
        ),
        answer_shape="A single integer (number of people), no commas, no units.",
        allow_empty=False,
        multi_valued=False,
        max_new_tokens=32,
        unit="people",
    ),
    "hasArea": RelationSpec(
        name="hasArea",
        kind="numeric",
        instruction="Give the total surface area of the subject in square kilometres (km²).",
        definition=(
            "The total surface area of the geographic entity in SQUARE KILOMETRES (km²). "
            "For countries use TOTAL area (land + inland water), Wikidata P2046. Convert "
            "from hectares/sq-miles to km² if needed."
        ),
        answer_shape="A single number in km² (may be a decimal), no units, no commas.",
        allow_empty=False,
        multi_valued=False,
        max_new_tokens=32,
        unit="km^2",
    ),
}

ALL_RELATIONS: List[str] = list(RELATIONS.keys())
