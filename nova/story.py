"""Extract visible story beats into single-image prompts."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MAX_SCENES = 12
SENTENCES_PER_BEAT = 5
DETAIL_LIMIT = 700
PARAGRAPHS = re.compile(r"\n\s*\n|^\s*(?:\*\s*){3,}$|^\s*-{3,}\s*$", re.MULTILINE)
DIALOGUE = re.compile(r'"[^"\n]*"|“[^”\n]*”|«[^»\n]*»')
SENTENCES = re.compile(r"(?<=[.!?。！？])\s+|(?<=[。！？])")
VISUAL = re.compile(
    r"\b(?:wear\w*|dress\w*|hold\w*|held|carry\w*|carried|stand\w*|stood|sit\w*|sat|"
    r"walk\w*|step\w*|reach\w*|rais\w*|cross\w*|turn\w*|lean\w*|open\w*|look\w*|"
    r"rain\w*|snow\w*|fog\w*|mist\w*|light\w*|dawn|dusk|sunset|sunrise|night|shadow\w*|"
    r"glow\w*|bright|dark|gold\w*|silver|red|orange|blue|green|white|black|"
    r"forest|street|room|bridge|river|tower|hall|door\w*|window\w*)\b", re.IGNORECASE,
)
SPEECH_TAG = re.compile(
    r"(?:[\w'’-]+ ){0,3}(?:said|asked|replied|whispered|answered)"
    r"(?: [\w'’-]+){0,2}[.!?]?", re.IGNORECASE,
)
REFLECTION = re.compile(
    r"\b(?:thought|wondered|remembered|believed|felt|hoped|knew|wished|decided|"
    r"said|asked|replied|whispered|answered)\b", re.IGNORECASE,
)


class StoryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    story: str = Field(min_length=1, max_length=12000)
    name_a: str = Field(default="", max_length=60)
    name_b: str = Field(default="", max_length=60)
    layout: Literal["horizontal", "vertical"] = "horizontal"

    @model_validator(mode="after")
    def distinct_names(self):
        if self.name_a and self.name_a.casefold() == self.name_b.casefold():
            raise ValueError("Use different names for Characters A and B.")
        return self


def char_pattern(name):
    return re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE) if name else None


def shorten(text, limit):
    if len(text) <= limit:
        return text
    prefix = text[:limit].rsplit(maxsplit=1)
    return (prefix[0] if len(prefix) == 2 else text[:limit]).rstrip(" ,;:.") + "…"


def scene_details(sentences, names):
    # Prefer visible actions and setting while retaining the original narrative order.
    def score(sentence):
        named = sum(bool(pattern and pattern.search(sentence)) for pattern in names)
        return 2 * len(VISUAL.findall(sentence)) + named - 2 * len(REFLECTION.findall(sentence))

    ranked = sorted(enumerate(sentences), key=lambda row: (-score(row[1]), row[0]))
    selected = []
    remaining = DETAIL_LIMIT
    for index, sentence in ranked:
        if selected and (len(selected) == 3 or remaining < 80):
            break
        compact = shorten(sentence, min(remaining, 300))
        selected.append((index, compact))
        remaining -= len(compact) + 2
    return " ".join(text for _, text in sorted(selected))


def translate_story(request: StoryRequest):
    names = (char_pattern(request.name_a), char_pattern(request.name_b))
    positions = ("on the left", "on the right") if request.layout == "horizontal" else (
        "in the upper part of the frame", "in the lower part of the frame",
    )
    subjects = [
        f"{label} matching Reference {slot} {position}"
        for slot, label, position in zip(
            ("A", "B"), (request.name_a or "Character A", request.name_b or "Character B"), positions,
        )
    ]
    opening = f"Single image with two distinct adult characters. {subjects[0]}. {subjects[1]}."
    scenes = []
    truncated = False
    for paragraph in PARAGRAPHS.split(request.story.replace("\r\n", "\n").replace("\r", "\n")):
        has_dialogue = bool(DIALOGUE.search(paragraph))
        narration = DIALOGUE.sub(" ", paragraph)
        narration = re.sub(r"\s+([,.!?])", r"\1", narration)
        sentences = [
            segment.strip(" \t\r\n,;:-—")
            for segment in SENTENCES.split(narration)
            if any(character.isalnum() for character in segment)
        ]
        sentences = [" ".join(sentence.split()) for sentence in sentences]
        if has_dialogue:
            sentences = [sentence for sentence in sentences if not SPEECH_TAG.fullmatch(sentence)]
        for start in range(0, len(sentences), SENTENCES_PER_BEAT):
            if len(scenes) == MAX_SCENES:
                truncated = True
                break
            detail = scene_details(sentences[start:start + SENTENCES_PER_BEAT], names)
            scenes.append({"title": f"Scene {len(scenes) + 1}", "prompt": f"{opening}\n{detail}"})
        if truncated:
            break
    return {"scenes": scenes, "truncated": truncated}
