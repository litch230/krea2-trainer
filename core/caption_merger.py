import html
import re
from typing import Optional


def _tag_key(value: str) -> str:
    """Curator-compatible key used to deduplicate caption tag entries."""
    value = html.unescape(str(value)).replace("\\(", "(").replace("\\)", ")")
    value = value.replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", value)


def _tag_entries(value: str):
    text = str(value or "").replace("\r\n", "\n").replace("\n", ",").strip()
    if text.endswith("."):
        text = text[:-1]
    seen = set()
    for part in text.split(","):
        display = html.unescape(part).replace("\\(", "(").replace("\\)", ")").strip()
        key = _tag_key(display)
        if key and key not in seen:
            seen.add(key)
            yield key, display

def merge_tags_and_nl_captions(tags: Optional[str], nl: Optional[str]) -> Optional[str]:
    """
    Merges tags and NL captions with proper punctuation.
    """
    nl_body = str(nl or "").strip()
    # Curator keeps the natural-language prose before the final sentence.
    sentence_start = 0
    for match in re.finditer(r"[.!?](?:\s+|$)", nl_body):
        if match.end() < len(nl_body):
            sentence_start = match.end()
    prose = nl_body[:sentence_start]
    final_nl_sentence = nl_body[sentence_start:]

    merged = []
    seen = set()
    for raw in (final_nl_sentence, tags or ""):
        for key, display in _tag_entries(raw):
            if key not in seen:
                seen.add(key)
                merged.append(display)

    if not merged:
        return nl_body or (str(tags).strip() if tags else None)
    return f"{prose}{', '.join(merged)}."
