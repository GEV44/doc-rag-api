"""Script-based language detection for Armenian / Russian / English (no dependencies)."""


def detect_language(text: str) -> str:
    """Return 'hy', 'ru', 'en' or 'unknown' from the dominant alphabet of the letters in text."""
    hy = ru = en = 0
    for ch in text:
        o = ord(ch)
        if 0x0530 <= o <= 0x058F:
            hy += 1
        elif 0x0400 <= o <= 0x04FF:
            ru += 1
        elif ch.isalpha() and o < 0x250:
            en += 1
    best = max((hy, "hy"), (ru, "ru"), (en, "en"))
    return best[1] if best[0] else "unknown"
