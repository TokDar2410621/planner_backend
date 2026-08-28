"""
Mesures non bloquantes du canal libre de DIRE.

La garantie structurelle protege les actions referencees. Elle ne protege pas
les champs libres ouverture et suite: ce module les observe sans les reecrire,
sans les filtrer et sans journaliser leur contenu.
"""
from __future__ import annotations

import re
import unicodedata

from services.agent_v2.redaction import ReponseDire

_VERBE_ACTION = (
    r"(?:organis\w*|reorganis\w*|supprim\w*|effac\w*|vid\w*|enlev\w*|"
    r"retir\w*|ajout\w*|cre\w*|deplac\w*|modifi\w*|annul\w*|"
    r"planifi\w*|programm\w*|cal\w*|plac\w*|optimis\w*|"
    # L'INFINITIF autant que le participe: « je suis en train de METTRE a
    # jour » est l'une des trois phrases exactes du 18 aout, et une premiere
    # version de ce module ne connaissait que « mis a jour », donc la ratait.
    r"restaur\w*|termin\w*|complet\w*|mis\s+a\s+jour|mettre\s+a\s+jour)"
)
_OPTIONNEL = r"(?:bien\s+|deja\s+|tout\s+|te\s+|tes?\s+|ton\s+|ta\s+|mes?\s+)*"

_REGLES = (
    ("passe", re.compile(rf"\bj\s*'?\s*ai\s+{_OPTIONNEL}{_VERBE_ACTION}\b")),
    ("futur", re.compile(rf"\bje\s+vais\s+{_OPTIONNEL}{_VERBE_ACTION}\b")),
    (
        "en_cours",
        re.compile(rf"\bje\s+suis\s+en\s+train\s+de\s+{_OPTIONNEL}{_VERBE_ACTION}\b"),
    ),
    ("prise_en_charge", re.compile(r"\bje\s+m\s*'?\s*occupe\s+de\b")),
)

_APOSTROPHES = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201b": "'",
    "\u02bc": "'",
    "`": "'",
    "\u00b4": "'",
})


def _normaliser(texte) -> str:
    if not texte or not isinstance(texte, str):
        return ""
    texte = texte.translate(_APOSTROPHES)
    plat = (
        unicodedata.normalize("NFKD", texte)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return re.sub(r"[^a-z0-9']+", " ", plat)


def fuite_lexicale(texte) -> list[str]:
    """Detecte une affirmation d'action dans un champ libre de DIRE."""
    plat = _normaliser(texte)
    if not plat:
        return []
    return [nom for nom, regle in _REGLES if regle.search(plat)]


def fuites_reponse(reponse: ReponseDire) -> list[str]:
    """Observe seulement ouverture et suite, pas les actions structurees."""
    fuites: list[str] = []
    for champ in ("ouverture", "suite"):
        for fuite in fuite_lexicale(getattr(reponse, champ, "")):
            fuites.append(f"{champ}:{fuite}")
    return fuites
