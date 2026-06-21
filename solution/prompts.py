"""Prompt construction for the LM-KBC 2026 task.

Strategy:
  * One chat prompt per (subject, relation).
  * System message states the precise relation definition + scope so the model
    targets the same notion the ground truth was built against.
  * We force a small JSON output `{"answers": [...]}` so parsing is robust and
    the empty case is unambiguous (`{"answers": []}`).
  * Few-shot exemplars are drawn from the TRAIN split. For null-possible
    relations we deliberately mix empty and non-empty exemplars so the model
    learns that abstaining is a valid, expected answer (abstention is worth
    real points — predicting [] scores macro-F1 0.20 on val by itself).

No GPU / model dependencies here: a prompt is just a list of {role, content}
dicts, ready for `tokenizer.apply_chat_template`.
"""
from __future__ import annotations
import json
from typing import Dict, List

from relations import RelationSpec, RELATIONS


def _first_alias(obj_aliases: List[str]) -> str:
    return obj_aliases[0] if obj_aliases else ""


def gold_answer_list(row: Dict) -> List[str]:
    """Canonical (first-alias) answer strings for a train/val row."""
    return [_first_alias(o) for o in row["ObjectEntities"] if o]


def select_few_shot(spec: RelationSpec, train_rows: List[Dict], k: int) -> List[Dict]:
    """Pick up to k exemplars for this relation. For null-possible relations,
    interleave empty and non-empty examples so both behaviours are demonstrated.
    Deterministic (no RNG) for reproducibility."""
    rel_rows = [r for r in train_rows if r["Relation"] == spec.name]
    if not spec.allow_empty:
        return rel_rows[:k]
    empties = [r for r in rel_rows if len(r["ObjectEntities"]) == 0]
    nonempties = [r for r in rel_rows if len(r["ObjectEntities"]) > 0]
    out, i, j = [], 0, 0
    # alternate non-empty / empty, starting with non-empty
    while len(out) < k and (i < len(nonempties) or j < len(empties)):
        if i < len(nonempties):
            out.append(nonempties[i]); i += 1
        if len(out) < k and j < len(empties):
            out.append(empties[j]); j += 1
    return out


def _exemplar_answer_json(spec: RelationSpec, row: Dict) -> str:
    answers = gold_answer_list(row)
    if spec.name == "awardWonBy":
        # keep exemplar short: show a handful of winners, not hundreds
        answers = answers[:12]
    return json.dumps({"answers": answers}, ensure_ascii=False)


def system_message(spec: RelationSpec) -> str:
    lines = [
        "You are a precise knowledge-base construction engine. You answer purely "
        "from your own parametric knowledge (closed book).",
        "",
        f"TASK: {spec.instruction}",
        f"DEFINITION (the exact scope you are graded on): {spec.definition}",
        f"ANSWER SHAPE: {spec.answer_shape}",
        "",
        "Rules:",
        '- Reply with ONLY a single JSON object: {"answers": [...]}. No prose, no markdown.',
    ]
    if spec.kind == "numeric":
        lines.append(
            '- The list holds exactly one element: the number as a string, e.g. {"answers": ["35000"]}. '
            "No commas, no units, no ranges."
        )
    else:
        lines.append('- Each list element is one object string, e.g. {"answers": ["Haiti"]}.')
    if spec.allow_empty:
        lines.append(
            '- If the correct answer is genuinely empty (e.g. ' +
            ("an island country with no land border" if spec.name == "countryLandBordersCountry"
             else "the person is still alive or the city is unknown" if spec.name == "personHasCityOfDeath"
             else "the company is private / not separately listed") +
            '), reply {"answers": []}. Do NOT guess when unsure — a wrong guess is '
            "penalised, an empty answer is not."
        )
    else:
        lines.append("- Always give your single best answer; do not return an empty list.")
    return "\n".join(lines)


def user_message(spec: RelationSpec, subject: str) -> str:
    return f"Subject: {subject}\nRelation: {spec.name}\nAnswer:"


def build_messages(spec: RelationSpec, subject: str, train_rows: List[Dict]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_message(spec)}]
    for ex in select_few_shot(spec, train_rows, spec.few_shot):
        msgs.append({"role": "user", "content": user_message(spec, ex["SubjectEntity"])})
        msgs.append({"role": "assistant", "content": _exemplar_answer_json(spec, ex)})
    msgs.append({"role": "user", "content": user_message(spec, subject)})
    return msgs


def build_all(spec_name: str, subject: str, train_rows: List[Dict]) -> List[Dict[str, str]]:
    return build_messages(RELATIONS[spec_name], subject, train_rows)
