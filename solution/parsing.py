"""Robustly turn raw model text into a clean List[str] of object strings.

The model is asked for `{"answers": [...]}`, but real models drift: markdown
fences, trailing prose, bare lists, "None", numbers with commas/units, etc.
This module recovers a clean answer list and normalises numeric relations.

Pure-python, no model deps — fully unit-testable on a laptop.
"""
from __future__ import annotations
import json
import re
import string
import unicodedata
from typing import List, Optional

from relations import RelationSpec, RELATIONS

# Tokens that signal an explicit empty / abstain answer.
_EMPTY_TOKENS = {
    "", "none", "n/a", "na", "null", "nil", "unknown", "no answer", "empty",
    "not applicable", "no", "no land border", "still alive", "alive",
    "not publicly traded", "not listed", "private",
}


# --- normalization identical to evaluate.py (keep in sync) -------------------
def normalize_string(s: str) -> str:
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    for p in string.punctuation:
        s = s.replace(p, " ")
    return " ".join(s.split())


# --- JSON / list recovery ----------------------------------------------------
def _extract_answers_field(text: str) -> Optional[List]:
    """Find a JSON object containing an "answers" list, scanning all
    balanced-brace candidates and taking the first that parses."""
    # Strip code fences.
    text = re.sub(r"```(?:json)?", "", text)
    # Find every {...} candidate (greedy-ish, brace-balanced scan).
    for m in re.finditer(r"\{", text):
        start = m.start()
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start:i + 1]
                    try:
                        obj = json.loads(blob)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and "answers" in obj and isinstance(obj["answers"], list):
                        return obj["answers"]
                    break
    return None


def _extract_bare_list(text: str) -> Optional[List]:
    text2 = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\[.*\]", text2, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _fallback_lines(text: str) -> List[str]:
    """Last-resort: split prose into candidate items (comma / newline / bullets)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = re.sub(r"^(the answer is|answer:|answers:)\s*", "", text, flags=re.I)
    parts = re.split(r"[\n,;]+|\s+and\s+", text)
    return [p.strip(" .-*•\t") for p in parts if p.strip(" .-*•\t")]


def raw_to_items(text: str) -> List[str]:
    """Recover a list of raw string items from any model output."""
    if text is None:
        return []
    items = _extract_answers_field(text)
    if items is None:
        items = _extract_bare_list(text)
    if items is None:
        items = _fallback_lines(text)
    out: List[str] = []
    for it in items:
        if isinstance(it, (int, float)):
            out.append(str(it))
        elif isinstance(it, str):
            out.append(it.strip())
        # ignore nested/other types
    return out


# --- numeric handling --------------------------------------------------------
_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def parse_number(s: str) -> Optional[str]:
    """Extract the first number from a string, handling commas, scaling words,
    and unit suffixes. Returns a canonical numeric string or None."""
    if not isinstance(s, str):
        s = str(s)
    low = s.lower().replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+", low)
    if not m:
        return None
    val = float(m.group(0))
    tail = low[m.end():]
    # scaling words right after the number
    if re.match(r"\s*(thousand|k)\b", tail):
        val *= 1e3
    elif re.match(r"\s*(million|mn|m)\b", tail):
        val *= 1e6
    # hectare -> km^2 conversion only handled upstream via prompt; we trust km^2.
    # render: int if integral, else trimmed float
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return repr(val).rstrip("0").rstrip(".") if "." in repr(val) else str(val)


# --- main entrypoint ---------------------------------------------------------
def parse_prediction(relation: str, raw_text: str) -> List[str]:
    spec: RelationSpec = RELATIONS[relation]
    items = raw_to_items(raw_text)

    # drop explicit empty markers
    items = [it for it in items if normalize_string(it) not in _EMPTY_TOKENS]

    if spec.kind == "numeric":
        for it in items:
            n = parse_number(it)
            if n is not None:
                return [n]            # numeric relations want exactly one value
        return []

    # string relations: dedup by normalized form, preserve order & casing
    seen, out = set(), []
    for it in items:
        key = normalize_string(it)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)

    if not spec.multi_valued:
        out = out[:1]                 # single-valued (cityOfDeath): keep best one
    return out
