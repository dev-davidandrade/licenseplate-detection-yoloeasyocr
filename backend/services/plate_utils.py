import re


def clean_text(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())
