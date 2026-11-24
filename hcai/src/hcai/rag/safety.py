"""Emergency red-flag gate. Runs BEFORE retrieval or generation; no LLM is involved.

Crude on purpose (keyword rules, documented in LIMITATIONS.md). Bias: false alarms are cheaper
than missed emergencies, except for clearly informational questions ("what are the symptoms of
a stroke?"), which are answered from the documents."""

from __future__ import annotations

import re
from dataclasses import dataclass

EMERGENCY_MESSAGE = (
    "These symptoms can be signs of a medical emergency. Do not wait for this tool: call your "
    "local emergency number now (112 in India, 911 in the US, 999 in the UK) or go to the "
    "nearest emergency department. If someone else has these symptoms, stay with them and call "
    "for help."
)

_RED_FLAGS = [
    r"chest (pain|pressure|tightness|discomfort)|pain in (my |the )?chest|tight(ness)? in (my |the )?chest",
    r"heart attack|cardiac arrest",
    r"(can'?t|cannot|unable to|trouble|difficulty) breath|short(ness)? of breath|struggling to breathe",
    r"faint(ed|ing)?|pass(ed|ing) out|lost consciousness|collaps(e|ed|ing)",
    r"stroke|face (is )?droop|slurred speech|can'?t speak|(sudden|one[- ]sided) (numbness|weakness)|arm weakness",
    r"sudden (severe )?headache|worst headache",
    r"crushing pain|pain (radiating|spreading) (to|down)",
]  # fmt: skip
_RED_FLAG_RE = re.compile("|".join(f"(?:{p})" for p in _RED_FLAGS), re.I)

_INFORMATIONAL_RE = re.compile(
    r"^\s*(what|which|why|how|when|who|can you explain|explain|define|list|tell me about|are|is)\b",
    re.I,
)
_PERSONAL_RE = re.compile(
    r"\b(i|i'm|im|i've|ive|my|me|we|he|she|they|someone|dad|mom|mother|father|wife|husband|"
    r"friend|patient|right now|currently|having|just)\b",
    re.I,
)


@dataclass(frozen=True)
class SafetyAlert:
    matched: str
    message: str = EMERGENCY_MESSAGE


def check_emergency(question: str) -> SafetyAlert | None:
    """Return an alert if the text describes a possible emergency happening to someone."""
    m = _RED_FLAG_RE.search(question)
    if not m:
        return None
    informational = bool(_INFORMATIONAL_RE.match(question)) and not _PERSONAL_RE.search(question)
    return None if informational else SafetyAlert(matched=m.group(0).lower())
