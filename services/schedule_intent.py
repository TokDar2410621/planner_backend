"""
Cheap schedule-intent gate.

A fast, dependency-free pre-filter that scores how likely a piece of text is a
schedule/agenda (day names + clock times + schedule vocabulary). It runs BEFORE
any expensive LLM call so we don't pay Gemini for obviously non-schedule text,
and it feeds the confidence/confirmation layer so aggressive capture does not
pollute the planning with false positives.

Text-only: images have no text to gate on before vision, so the gate applies to
OCR/pdfplumber text and to plain chat text. Deterministic and unit-testable.
"""
import re
import unicodedata

# Day names across FR/EN, full + common abbreviations (accents already stripped).
_DAY_WORDS = {
    'lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche',
    'lun', 'mar', 'mer', 'jeu', 'ven', 'sam', 'dim',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
}

# Schedule vocabulary (accent-stripped, matched as substrings).
_KEYWORDS = (
    'cours', 'salle', 'local', 'horaire', 'prof', 'semestre', 'session',
    'agenda', 'planning', 'rendez', 'reunion', 'examen', 'labo', 'periode',
    'emploi du temps', 'matiere', 'schedule', 'room', 'lecture', 'class',
)

# Clock times: 9h, 9h30, 09:00, 14 h 30, 8h-10h. Requires an h/: separator so
# bare numbers (years, prices) don't trigger.
_TIME_RE = re.compile(r'\b\d{1,2}\s*[h:]\s*\d{0,2}\b', re.IGNORECASE)


def _normalize(text: str) -> str:
    """Lowercase and strip accents so 'Matière'/'réunion' match the keyword set."""
    text = text.lower()
    decomposed = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in decomposed if not unicodedata.combining(c))


def signals(text: str) -> dict:
    """Return raw signal counts: {'days', 'times', 'keywords'}."""
    if not text:
        return {'days': 0, 'times': 0, 'keywords': 0}
    norm = _normalize(text)
    words = set(re.findall(r'[a-z]+', norm))
    return {
        'days': len(words & _DAY_WORDS),
        'times': len(_TIME_RE.findall(text)),
        'keywords': sum(1 for k in _KEYWORDS if k in norm),
    }


def intent_score(text: str) -> float:
    """Score 0..1 that `text` describes a schedule. Higher = more schedule-like."""
    s = signals(text)
    score = 0.0
    # A day alone (0.35) or a time alone (0.35) stays below the 0.4 gate; it
    # takes a real combination (day+time, or a signal plus vocab) to pass.
    if s['days'] >= 1:
        score += 0.35
    if s['days'] >= 2:
        score += 0.10
    if s['times'] >= 1:
        score += 0.35
    if s['times'] >= 3:
        score += 0.10
    if s['keywords'] >= 1:
        score += 0.20
    return round(min(score, 1.0), 3)


def has_schedule_signal(text: str, threshold: float = 0.4) -> bool:
    """True when `text` clears the intent threshold (day + time, or strong vocab)."""
    return intent_score(text) >= threshold
