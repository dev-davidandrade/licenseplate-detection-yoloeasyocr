import re
from typing import Literal


PlateFormat = Literal["old", "mercosul"]

OLD_PLATE_PATTERN = re.compile(r"^[A-Z]{3}[0-9]{4}$")
MERCOSUL_PLATE_PATTERN = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")

LETTER_FROM_DIGIT = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}

DIGIT_FROM_LETTER = {
    "A": "4",
    "B": "8",
    "D": "0",
    "G": "6",
    "I": "1",
    "L": "1",
    "O": "0",
    "Q": "0",
    "S": "5",
    "T": "7",
    "Z": "2",
}


def clean_text(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def plate_format(text: str) -> PlateFormat | None:
    cleaned = clean_text(text)
    if OLD_PLATE_PATTERN.fullmatch(cleaned):
        return "old"
    if MERCOSUL_PLATE_PATTERN.fullmatch(cleaned):
        return "mercosul"
    return None


def is_valid_plate(text: str) -> bool:
    return plate_format(text) is not None


def _coerce_to_mask(text: str, mask: str) -> tuple[str, int] | None:
    if len(text) != len(mask):
        return None

    result = []
    substitutions = 0
    for character, expected_type in zip(text, mask, strict=True):
        if expected_type == "L":
            if character.isalpha():
                result.append(character)
            elif character in LETTER_FROM_DIGIT:
                result.append(LETTER_FROM_DIGIT[character])
                substitutions += 1
            else:
                return None
        else:
            if character.isdigit():
                result.append(character)
            elif character in DIGIT_FROM_LETTER:
                result.append(DIGIT_FROM_LETTER[character])
                substitutions += 1
            else:
                return None
    return "".join(result), substitutions


def normalize_plate_text(
    raw: str,
    preferred_format: PlateFormat | None = None,
) -> str:
    cleaned = clean_text(raw)
    if len(cleaned) != 7:
        return cleaned

    masks: dict[PlateFormat, str] = {
        "old": "LLLDDDD",
        "mercosul": "LLLDLDD",
    }
    formats = [preferred_format] if preferred_format else ["old", "mercosul"]
    candidates = []
    for format_name in formats:
        coerced = _coerce_to_mask(cleaned, masks[format_name])
        if coerced is not None:
            normalized, substitutions = coerced
            candidates.append((substitutions, format_name != preferred_format, normalized))

    if not candidates:
        return cleaned
    return min(candidates)[2]
