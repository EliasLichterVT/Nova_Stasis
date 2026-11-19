"""Local LoRA trigger-word catalog and story matching."""

import re
import unicodedata
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


Term = Annotated[str, Field(max_length=120)]
ModelName = Annotated[str, Field(min_length=1, max_length=300)]
STOP_WORDS = set(
    "a an and are at b by for from in is it of on or the to with adult adults character characters "
    "distinct frame image lora lycoris matching model part reference safetensors sd sdxl single "
    "two upper lower left right style version".split()
)


class KeywordEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: ModelName
    triggers: list[Term] = Field(max_length=40)
    keywords: list[Term] = Field(max_length=40)


class SuggestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    prompt: str = Field(min_length=1, max_length=3000)
    selected: list[ModelName] = Field(default_factory=list, max_length=2)


def clean_terms(values):
    terms = {}
    for value in values if isinstance(values, list) else []:
        if not isinstance(value, str):
            continue
        term = " ".join(value.split()).strip(", ")
        if term and len(term) <= 120 and term.casefold() not in terms:
            terms[term.casefold()] = term
            if len(terms) == 40:
                break
    return list(terms.values())


def catalog_entry(db, name):
    model = db.get("lora_source", name) or {}
    custom = db.get("lora_keywords", name)
    return {
        "name": name,
        "title": model.get("title", ""),
        "base": model.get("base", ""),
        "triggers": custom["triggers"] if custom else model.get("triggers", []),
        "keywords": custom["keywords"] if custom else [],
        "source_triggers": model.get("triggers", []),
    }


def words(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[^\W_]+", normalized)


def suggest_loras(db, installed, prompt, selected):
    tokens = words(prompt)
    source = f" {' '.join(tokens)} "
    visible_words = set(tokens) - STOP_WORDS
    ranked = []
    for name in dict.fromkeys(installed):
        entry = catalog_entry(db, name)
        stem = re.split(r"[/\\]", name)[-1].rsplit(".", 1)[0]
        terms = dict.fromkeys([*entry["keywords"], *entry["triggers"], entry["title"], stem])
        matches = []
        score = 0
        for term in terms:
            phrase = words(term)
            meaningful = {word for word in phrase if len(word) >= 3 and word not in STOP_WORDS
                          and not word.isdecimal() and not re.fullmatch(r"v\d+", word)}
            if not meaningful:
                continue
            if f" {' '.join(phrase)} " in source:
                matches.append(term)
                score += 4 + len(meaningful)
            elif term in (stem, entry["title"]):
                overlap = meaningful & visible_words
                if overlap:
                    matches.extend(sorted(overlap))
                    score += len(overlap)
        active = name in selected
        if active or score:
            entry.update(matches=list(dict.fromkeys(matches)), selected=active)
            ranked.append((not active, -score, name.casefold(), entry))
    ranked.sort(key=lambda item: item[:3])
    return [item[3] for item in ranked[:5]]
