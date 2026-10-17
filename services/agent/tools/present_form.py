"""Outil present_form."""
import math
import re
from datetime import timedelta
from typing import Annotated, Any, Literal

from django.contrib.auth.models import User
from django.utils import timezone
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from . import commun
from .base import ToolResult
from .commun import _fold, _iso_date

FIELD_TYPES = ("time_range", "time", "number", "checkbox", "select", "radio", "duration", "date")
CHOICE_TYPES = ("checkbox", "select", "radio")
DEFAULT_DURATION_PRESETS = (30, 60, 90, 120)
WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
WEEKDAY_ABBREVIATIONS = ("lun", "mar", "mer", "jeu", "ven", "sam", "dim")
DAY_LABELS = ("Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche")
SHORT_MONTHS = ("janv.", "févr.", "mars", "avr.", "mai", "juin",
                "juil.", "août", "sept.", "oct.", "nov.", "déc.")
CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})$")

# Source de verite : FIELD_TYPES. Le Literal ci-dessous est le miroir type
# du schema ; l'assertion garde les deux synchronises.
TypeChamp = Literal["time_range", "time", "number", "checkbox", "select", "radio", "duration", "date"]
assert set(TypeChamp.__args__) == set(FIELD_TYPES), \
    "derive entre le schema type et FIELD_TYPES"


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


NAME = "present_form"
DESCRIPTION = (
    "Présente un formulaire interactif à l'utilisateur avec des champs structurés. "
    "Utilise cet outil quand tu as besoin que l'utilisateur fasse des choix précis "
    "(horaires, jours de la semaine, options multiples, etc.) au lieu de taper du texte libre. "
    "Les types disponibles : time_range (plage horaire), time (heure), number (nombre), "
    "checkbox (choix multiples), select (liste déroulante), radio (choix unique avec option 'Autre'), "
    "duration (durée en minutes), date (jour). "
    "Pré-remplis ce que tu peux (default) et offre des raccourcis en un tap (presets): "
    "l'utilisateur ajuste, il ne saisit pas."
)


def _chaine_souple(v):
    """Coercition historique des options : le frontend compare des chaines,
    un entier brut ne cocherait rien. None reste None (l'appelant decide :
    _options ignore les values vides)."""
    if v is None or isinstance(v, str):
        return v
    return str(v)


ChaineSouple = Annotated[Any, BeforeValidator(_chaine_souple)]


class OptionFormulaire(BaseModel):
    """Option d'un champ a choix. Les values sont forcees en chaines des la
    validation (le frontend compare des chaines, un default ou un preset
    entier ne cochait rien)."""
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _non_dict_vaut_vide(cls, v):
        # Une option qui n'est pas un objet est ignoree par _options
        # dans execute(), pas rejetee par la validation.
        return v if isinstance(v, dict) else {"value": None}

    value: ChaineSouple = Field(
        default=None,
        description="Valeur de l'option (chaîne; un entier est casté en chaîne)",
    )
    label: ChaineSouple = Field(
        default=None,
        description="Label affiché (défaut: la valeur)",
    )


class ChampFormulaire(BaseModel):
    """Champ du formulaire. Le schema reste volontairement permissif : la
    validation metier (type inconnu, id/label/question manquants, options
    requises) vit dans execute() avec ses messages historiques, et la
    coercition (entiers -> chaines, bornes, presets) aussi. Le schema ne
    fait que documenter la forme attendue."""
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _non_dict_vaut_vide(cls, v):
        # Un champ qui n'est pas un objet est refuse proprement par
        # execute() ("Type de champ inconnu 'None' pour le champ '?'"),
        # pas par une erreur de validation brute.
        return v if isinstance(v, dict) else {}

    id: Any = Field(
        default=None,
        description="Identifiant unique du champ (ex: 'sleep_time', 'work_days')",
        json_schema_extra={"type": "string"},
    )
    type: Any = Field(
        default=None,
        description=(
            "Type de champ: time_range (plage horaire start/end), time (heure unique), "
            "number (nombre), checkbox (choix multiples), select (dropdown), radio (choix unique), "
            "duration (durée: entier, en minutes; pastilles 30 min / 1 h / 1 h 30 / 2 h + saisie libre), "
            "date (jour: pastilles aujourd'hui / demain / week-end + calendrier)"
        ),
        json_schema_extra={"type": "string", "enum": list(FIELD_TYPES)},
    )
    label: Any = Field(
        default=None,
        description="Label court du champ (ex: 'Heures de sommeil')",
        json_schema_extra={"type": "string"},
    )
    question: Any = Field(
        default=None,
        description="Question affichée au-dessus du champ (ex: 'À quelle heure tu te couches et te réveilles ?')",
        json_schema_extra={"type": "string"},
    )
    options: list[OptionFormulaire] | None = Field(
        default=None,
        description="Options pour checkbox/select/radio (obligatoires pour ces trois types). Chaque option a un value (chaîne) et un label.",
    )
    default: Any = Field(
        default=None,
        description=(
            "Valeur par défaut. time_range: {start: '23:00', end: '07:00'}. time: '18:00' (HH:MM). "
            "number: 8. checkbox: liste des values pré-cochées (ex: ['0','1','2','3','4'] pour lun-ven). "
            "select/radio: la value choisie. duration: entier, en minutes (ex: 90). date: 'YYYY-MM-DD'."
        ),
    )
    presets: Any = Field(
        default=None,
        description=(
            "Raccourcis en un tap, selon le type. "
            "time_range: [{label: '22h-6h', start: '22:00', end: '06:00'}, ...] (2-3 plages, heures HH:MM). "
            "checkbox: [{label: 'Lun-ven', values: [...]}] (ajouté automatiquement quand les options sont les jours de la semaine). "
            "duration: liste d'entiers, en minutes, défaut [30, 60, 90, 120]. "
            "date: [{label: 'Demain', value: 'YYYY-MM-DD'}], défaut calculé: aujourd'hui, demain, prochain samedi, prochain dimanche."
        ),
    )
    min: Any = Field(
        default=None,
        description="Valeur minimum (pour number: décimal accepté; pour duration: entier, en minutes)",
        json_schema_extra={"type": "number"},
    )
    max: Any = Field(
        default=None,
        description="Valeur maximum (pour number: décimal accepté; pour duration: entier, en minutes)",
        json_schema_extra={"type": "number"},
    )
    allow_other: Any = Field(
        default=None,
        description="Pour radio: afficher une option 'Autre' avec texte libre (défaut: true)",
        json_schema_extra={"type": "boolean"},
    )
    other_placeholder: Any = Field(
        default=None,
        description="Placeholder pour l'option 'Autre' du radio",
        json_schema_extra={"type": "string"},
    )


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    inputs: list[ChampFormulaire] = Field(
        description="Liste des champs du formulaire",
    )


PARAMETERS = commun.inline_refs(Parametres.model_json_schema())


def execute(user: User, **kwargs) -> ToolResult:
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
