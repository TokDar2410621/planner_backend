"""
Interactive UI tools - Let the AI present structured forms and choices to the user.
"""
import math
import re
import unicodedata
from datetime import date, timedelta

from django.contrib.auth.models import User
from django.utils import timezone

from .base import BaseTool, ToolResult

FIELD_TYPES = ("time_range", "time", "number", "checkbox", "select", "radio", "duration", "date")
CHOICE_TYPES = ("checkbox", "select", "radio")
DEFAULT_DURATION_PRESETS = (30, 60, 90, 120)
WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
WEEKDAY_ABBREVIATIONS = ("lun", "mar", "mer", "jeu", "ven", "sam", "dim")
DAY_LABELS = ("Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche")
SHORT_MONTHS = ("janv.", "févr.", "mars", "avr.", "mai", "juin",
                "juil.", "août", "sept.", "oct.", "nov.", "déc.")
CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def _fold(text) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip().lower()


def _weekday_index(option: dict):
    label = _fold(option.get("label", "")).rstrip(".")
    if label in WEEKDAYS:
        return WEEKDAYS.index(label)
    if label in WEEKDAY_ABBREVIATIONS:
        return WEEKDAY_ABBREVIATIONS.index(label)
    value = str(option.get("value", "")).strip()
    if value in ("0", "1", "2", "3", "4", "5", "6"):
        return int(value)
    return None


def weekday_presets(options) -> list:
    """Raccourcis Lun-ven / Tous quand toutes les options sont des jours de
    semaine (au moins 5). Le doublon Lun-ven == Tous (options = lun..ven
    exactement) n'est pas emis."""
    if not isinstance(options, list) or len(options) < 5:
        return []
    indexed = []
    for opt in options:
        if not isinstance(opt, dict):
            return []
        idx = _weekday_index(opt)
        if idx is None:
            return []
        indexed.append((idx, opt.get("value")))
    workdays = [value for idx, value in indexed if idx < 5]
    all_values = [value for _, value in indexed]
    presets = []
    if workdays and workdays != all_values:
        presets.append({"label": "Lun-ven", "values": workdays})
    presets.append({"label": "Tous", "values": all_values})
    return presets


def date_presets() -> list:
    """Aujourd'hui, demain, puis le prochain samedi et le prochain dimanche
    (strictement a venir), en date murale. Un week-end deja couvert par
    aujourd'hui/demain n'est pas repete."""
    today = timezone.localdate()
    presets = [
        {"label": "Aujourd'hui", "value": today.isoformat()},
        {"label": "Demain", "value": (today + timedelta(days=1)).isoformat()},
    ]
    for weekday in (5, 6):
        ahead = (weekday - today.weekday()) % 7 or 7
        day = today + timedelta(days=ahead)
        value = day.isoformat()
        if any(p["value"] == value for p in presets):
            continue
        label = f"{DAY_LABELS[weekday]} {day.day} {SHORT_MONTHS[day.month - 1]}"
        presets.append({"label": label, "value": value})
    return presets


def _clock(value):
    """'9:00' -> '09:00'; None si ce n'est pas une heure HH:MM."""
    if not isinstance(value, str):
        return None
    match = CLOCK_RE.match(value.strip())
    if not match:
        return None
    hours, minutes = int(match.group(1)), int(match.group(2))
    if hours > 23 or minutes > 59:
        return None
    return f"{hours:02d}:{minutes:02d}"


def _time_range(value):
    if not isinstance(value, dict):
        return None
    start, end = _clock(value.get("start")), _clock(value.get("end"))
    if start is None or end is None:
        return None
    return {"start": start, "end": end}


def _iso_date(value):
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError:
        return None


def _positive_int(value):
    """Entier strictement positif; 2.0 passe (vaut 2), 1.5 et '60' non."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        if value <= 0 or int(value) != value:
            return None
    except (ValueError, OverflowError):
        return None
    return int(value)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # json.loads accepte NaN et Infinity, json.dumps les reemet tels quels,
    # et la trame SSE devient illisible pour le client.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _label(value):
    if isinstance(value, str) and value.strip():
        return value
    return None


def _unique(values) -> list:
    seen = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


def _options(raw) -> list:
    """Options d'un champ a choix, values forcees en chaines: le frontend
    compare des chaines, un default ou un preset entier ne cochait rien."""
    if not isinstance(raw, list):
        return []
    options = []
    for opt in raw:
        if not isinstance(opt, dict) or opt.get("value") is None:
            continue
        value = str(opt["value"])
        if not value.strip():
            continue
        label = opt.get("label")
        options.append({"value": value, "label": str(label) if label not in (None, "") else value})
    return options


def _default(kind: str, raw, known: set):
    if kind == "time_range":
        return _time_range(raw)
    if kind == "time":
        return _clock(raw)
    if kind == "number":
        return _number(raw)
    if kind == "checkbox":
        if not isinstance(raw, list):
            return None
        return _unique(str(v) for v in raw if str(v) in known) or None
    if kind in ("select", "radio"):
        value = str(raw)
        return value if value in known else None
    if kind == "duration":
        return _positive_int(raw)
    if kind == "date":
        return _iso_date(raw)
    return None


def _presets(kind: str, raw, known: set) -> list:
    """Presets valides pour le type, entree par entree; ce qui est mal forme
    est jete (le rendu du chat plantait sur un preset checkbox sans 'values')."""
    if not isinstance(raw, list):
        raw = []
    kept = []
    if kind == "checkbox":
        for p in raw:
            if not isinstance(p, dict) or _label(p.get("label")) is None:
                continue
            values = p.get("values")
            if not isinstance(values, list):
                continue
            values = _unique(str(v) for v in values if str(v) in known)
            if values:
                kept.append({"label": p["label"], "values": values})
    elif kind == "time_range":
        for p in raw:
            if not isinstance(p, dict) or _label(p.get("label")) is None:
                continue
            plage = _time_range(p)
            if plage:
                kept.append({"label": p["label"], **plage})
    elif kind == "date":
        for p in raw:
            if not isinstance(p, dict) or _label(p.get("label")) is None:
                continue
            value = _iso_date(p.get("value"))
            if value:
                kept.append({"label": p["label"], "value": value})
    elif kind == "duration":
        kept = sorted({m for m in (_positive_int(x) for x in raw) if m is not None})
    return kept


def _int_bound(value):
    number = _number(value)
    if number is None:
        return None
    try:
        return int(number)
    except (ValueError, OverflowError):
        return None


class PresentFormTool(BaseTool):
    """
    Present an interactive form to the user with various input types.

    The frontend renders these as rich UI components (time pickers, checkboxes, etc.)
    The user's selections are sent back as the next message.
    """

    name = "present_form"
    description = (
        "Présente un formulaire interactif à l'utilisateur avec des champs structurés. "
        "Utilise cet outil quand tu as besoin que l'utilisateur fasse des choix précis "
        "(horaires, jours de la semaine, options multiples, etc.) au lieu de taper du texte libre. "
        "Les types disponibles : time_range (plage horaire), time (heure), number (nombre), "
        "checkbox (choix multiples), select (liste déroulante), radio (choix unique avec option 'Autre'), "
        "duration (durée en minutes), date (jour). "
        "Pré-remplis ce que tu peux (default) et offre des raccourcis en un tap (presets): "
        "l'utilisateur ajuste, il ne saisit pas."
    )
    parameters = {
        "type": "object",
        "properties": {
            "inputs": {
                "type": "array",
                "description": "Liste des champs du formulaire",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Identifiant unique du champ (ex: 'sleep_time', 'work_days')"
                        },
                        "type": {
                            "type": "string",
                            "enum": ["time_range", "time", "number", "checkbox", "select", "radio", "duration", "date"],
                            "description": (
                                "Type de champ: time_range (plage horaire start/end), time (heure unique), "
                                "number (nombre), checkbox (choix multiples), select (dropdown), radio (choix unique), "
                                "duration (durée: entiers, en minutes; pastilles 30 min / 1 h / 1 h 30 / 2 h + saisie libre), "
                                "date (jour: pastilles aujourd'hui / demain / week-end + calendrier)"
                            )
                        },
                        "label": {
                            "type": "string",
                            "description": "Label court du champ (ex: 'Heures de sommeil')"
                        },
                        "question": {
                            "type": "string",
                            "description": "Question affichée au-dessus du champ (ex: 'À quelle heure tu te couches et te réveilles ?')"
                        },
                        "options": {
                            "type": "array",
                            "description": "Options pour checkbox/select/radio (obligatoires pour ces trois types). Chaque option a un value (chaîne) et un label.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "string"},
                                    "label": {"type": "string"}
                                },
                                "required": ["value", "label"]
                            }
                        },
                        "default": {
                            "description": (
                                "Valeur par défaut. time_range: {start: '23:00', end: '07:00'}. time: '18:00' (HH:MM). "
                                "number: 8. checkbox: liste des values pré-cochées (ex: ['0','1','2','3','4'] pour lun-ven). "
                                "select/radio: la value choisie. duration: entier, en minutes (ex: 90). date: 'YYYY-MM-DD'."
                            )
                        },
                        "presets": {
                            "description": (
                                "Raccourcis en un tap, selon le type. "
                                "time_range: [{label: '22h-6h', start: '22:00', end: '06:00'}, ...] (2-3 plages, heures HH:MM). "
                                "checkbox: [{label: 'Lun-ven', values: [...]}] (ajouté automatiquement quand les options sont les jours de la semaine). "
                                "duration: liste d'entiers, en minutes, défaut [30, 60, 90, 120]. "
                                "date: [{label: 'Demain', value: 'YYYY-MM-DD'}], défaut calculé: aujourd'hui, demain, prochain samedi, prochain dimanche."
                            )
                        },
                        "min": {
                            "type": "number",
                            "description": "Valeur minimum (pour number; pour duration: entier, en minutes)"
                        },
                        "max": {
                            "type": "number",
                            "description": "Valeur maximum (pour number; pour duration: entier, en minutes)"
                        },
                        "allow_other": {
                            "type": "boolean",
                            "description": "Pour radio: afficher une option 'Autre' avec texte libre (défaut: true)"
                        },
                        "other_placeholder": {
                            "type": "string",
                            "description": "Placeholder pour l'option 'Autre' du radio"
                        }
                    },
                    "required": ["id", "type", "label", "question"]
                }
            }
        },
        "required": ["inputs"]
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        inputs = kwargs.get("inputs", [])

        if not inputs:
            return ToolResult(
                success=False,
                message="Aucun champ spécifié pour le formulaire."
            )

        # Validate and normalize inputs
        normalized = []
        for inp in inputs:
            if not isinstance(inp, dict):
                inp = {}
            kind = inp.get("type")
            field_id = inp.get("id") or "?"
            if kind not in FIELD_TYPES:
                return ToolResult(
                    success=False,
                    message=(
                        f"Type de champ inconnu '{kind}' pour le champ '{field_id}'. "
                        f"Types valides: {', '.join(FIELD_TYPES)}."
                    )
                )
            if not all(isinstance(inp.get(k), str) and inp[k].strip()
                       for k in ("id", "label", "question")):
                return ToolResult(
                    success=False,
                    message=f"Le champ '{field_id}' doit avoir id, label et question."
                )

            field = {
                "id": inp["id"],
                "type": kind,
                "label": inp["label"],
                "question": inp["question"],
            }

            known = set()
            if kind in CHOICE_TYPES:
                field["options"] = _options(inp.get("options"))
                if not field["options"]:
                    return ToolResult(
                        success=False,
                        message=f"Le champ '{inp['id']}' de type {kind} nécessite des options."
                    )
                known = {opt["value"] for opt in field["options"]}

            if "default" in inp:
                default = _default(kind, inp["default"], known)
                if default is not None:
                    field["default"] = default

            presets = _presets(kind, inp.get("presets"), known)
            if kind == "checkbox":
                presets = presets or weekday_presets(field["options"])
            elif kind == "duration":
                # Une liste partiellement invalide (0.5, 1, 1.5: le modele
                # pensait en heures) n'est pas fiable: on repart des defauts.
                brut = inp.get("presets")
                fiable = (isinstance(brut, list) and bool(brut)
                          and all(_positive_int(x) is not None for x in brut))
                presets = presets if fiable else list(DEFAULT_DURATION_PRESETS)
            elif kind == "date":
                presets = presets or date_presets()
            if presets:
                field["presets"] = presets

            # Number and duration constraints
            if kind in ("number", "duration"):
                cast = _int_bound if kind == "duration" else _number
                for bound in ("min", "max"):
                    if bound in inp:
                        value = cast(inp[bound])
                        if value is not None:
                            field[bound] = value

            # Radio specific
            if kind == "radio":
                if isinstance(inp.get("allow_other"), bool):
                    field["allowOther"] = inp["allow_other"]
                if isinstance(inp.get("other_placeholder"), str):
                    field["otherPlaceholder"] = inp["other_placeholder"]

            normalized.append(field)

        return ToolResult(
            success=True,
            data={"interactive_inputs": normalized},
            message=f"Formulaire avec {len(normalized)} champ(s) présenté à l'utilisateur. Attends sa réponse."
        )


class PresentQuickRepliesTool(BaseTool):
    """
    Present quick reply buttons to the user.
    """

    name = "present_quick_replies"
    description = (
        "Présente des boutons de réponse rapide à l'utilisateur. "
        "Utilise cet outil quand tu veux proposer 2-4 actions rapides après ta réponse. "
        "Chaque bouton a un label (texte affiché) et une value (message envoyé au clic)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "replies": {
                "type": "array",
                "description": "Liste des boutons (max 4)",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Texte affiché sur le bouton (court, avec emoji optionnel)"
                        },
                        "value": {
                            "type": "string",
                            "description": "Message envoyé quand l'utilisateur clique"
                        }
                    },
                    "required": ["label", "value"]
                }
            }
        },
        "required": ["replies"]
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        replies = kwargs.get("replies", [])

        if not replies:
            return ToolResult(
                success=False,
                message="Aucun bouton spécifié."
            )

        # Limit to 4 buttons
        normalized = replies[:4]

        return ToolResult(
            success=True,
            data={"quick_replies": normalized},
            message=f"{len(normalized)} bouton(s) de réponse rapide présenté(s)."
        )
