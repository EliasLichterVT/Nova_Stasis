"""Build weighted negative conditioning from visible anatomy cues."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AnatomyMode = Literal["high", "off"]

CORE = (
    ("malformed anatomy", 1.25),
    ("extra limbs", 1.30),
    ("fused limbs", 1.25),
    ("distorted joints", 1.20),
    ("malformed hands", 1.25),
    ("distorted facial features", 1.20),
)
REGIONS = (
    (
        re.compile(r"\b(?:hands?|fingers?|palms?|wrists?|holding|holds?|held|grasp\w*|"
                   r"waving|waves?|rais\w*|reach\w*|shak\w*|rings?|gloves?)\b", re.IGNORECASE),
        (("malformed hands", 1.30), ("extra fingers", 1.30), ("fused fingers", 1.25), ("missing fingers", 1.20)),
    ),
    (
        re.compile(r"\b(?:portraits?|faces?|eyes?|close[ -]?ups?|headshots?|"
                   r"smil\w*|teeth|mouth|expression\w*)\b", re.IGNORECASE),
        (("distorted facial features", 1.25), ("misaligned eyes", 1.20), ("extra eyes", 1.25)),
    ),
    (
        re.compile(r"\b(?:full[ -]?(?:body|length)|feet|foot|toes?|ankles?|"
                   r"barefoot|walk\w*|run\w*|danc\w*)\b", re.IGNORECASE),
        (("malformed feet", 1.25), ("extra toes", 1.25), ("twisted ankles", 1.20)),
    ),
)


class AnatomyPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    prompt: str = Field(default="", max_length=3000)
    negative: str = Field(default="", max_length=2000)
    anatomy: AnatomyMode = "high"


def effective_negative(prompt, negative, anatomy):
    if anatomy == "off":
        return negative

    # Recognize repeated phrases, including terms that the user weighted manually.
    existing = {
        " ".join(re.sub(r":\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*\)*$", "", term)
                 .strip(" ()").casefold().split())
        for term in negative.split(",")
    }
    terms = dict(CORE)
    for pattern, region_terms in REGIONS:
        if pattern.search(prompt):
            terms.update(region_terms)
    automatic = [f"({term}:{weight:.2f})" for term, weight in terms.items() if term not in existing]
    if negative:
        automatic.append(negative)
    return ", ".join(automatic)
