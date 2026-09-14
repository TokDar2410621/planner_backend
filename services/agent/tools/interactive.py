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


SOURCES_CHOIX = ("creneaux", "blocs", "taches", "jours")
MAX_QUESTION_CHOIX = 140
MAX_LIBELLE_CHOIX = 40
MAX_VALEUR_CHOIX = 200
MAX_OPTIONS_CHOIX = 4
MIN_OPTIONS_CHOIX = 2
DUREE_CRENEAU_PAR_DEFAUT = 30
MOIS_COURTS_PLATS = ("janv", "fevr", "mars", "avr", "mai", "juin",
                     "juil", "aout", "sept", "oct", "nov", "dec")
_APOSTROPHES_CHOIX = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_HEURE_CHOIX = re.compile(r"(?<!\d)(\d{1,2})\s*(?:h|:)\s*(\d{2})?(?!\d)|\b(midi|minuit)\b")
_JOUR_CHOIX = re.compile(
    r"\b(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|demain|apres-demain"
    r"|aujourd'hui|aujourdhui)\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}\s+(?:%s)" % "|".join(MOIS_COURTS_PLATS)
    + r"|\b\d{1,2}/\d{1,2}\b")


def _maintenant():
    """L'instant mural, isole pour que les tests le figent sans toucher au
    module timezone de Django."""
    return timezone.localtime()


def _plat(texte) -> str:
    """Minuscules, accents retires, apostrophes droites, espaces reduits."""
    return " ".join(_fold(str(texte).translate(_APOSTROPHES_CHOIX)).split())


def _texte_court(valeur, maximum: int):
    if not isinstance(valeur, str):
        return None
    propre = " ".join(valeur.split())
    if not propre or len(propre) > maximum:
        return None
    return propre


def _titre_contenu(libelle_plat: str, titres: dict):
    """Le titre reel (le plus long) que le libelle contient en mots entiers."""
    trouve = None
    for plat, original in titres.items():
        if not plat:
            continue
        if re.search(r"(?<!\w)" + re.escape(plat) + r"(?!\w)", libelle_plat):
            if trouve is None or len(plat) > len(_plat(trouve)):
                trouve = original
    return trouve


def _minutes_du_libelle(libelle: str):
    """(debut, fin) en minutes lus dans un libelle de creneau, fin par defaut
    debut + 30 min. None si le libelle ne porte pas d'heure valide."""
    heures = []
    for trouve in _HEURE_CHOIX.finditer(_plat(libelle)):
        if trouve.group(3):
            heures.append(720 if trouve.group(3) == "midi" else 0)
            continue
        h, m = int(trouve.group(1)), int(trouve.group(2) or 0)
        if h > 23 or m > 59:
            return None
        heures.append(h * 60 + m)
        if len(heures) == 2:
            break
    if not heures:
        return None
    debut = heures[0]
    if len(heures) == 1:
        return debut, debut + DUREE_CRENEAU_PAR_DEFAUT
    fin = heures[1]
    if fin == 0:
        fin = 24 * 60  # « 23 h a minuit »
    if fin <= debut:
        return None
    return debut, fin


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _demande_de_choix(question: str, source: str, options: list, parametres: dict,
                      cible: dict) -> dict:
    """La DEMANDE de motif choix_modele, meme forme que les demandes des gardes."""
    import hashlib

    return {
        "type": "choix",
        "motif": "choix_modele",
        "cle": "choix:" + hashlib.sha1(question.encode("utf-8")).hexdigest()[:12],
        "outil": "present_choices",
        "parametres": parametres,
        "cible": cible,
        "options": [
            {"id": f"o{rang}", "effet": None, "cible": option["cible"],
             "libelle": option["label"], "valeur": option["value"]}
            for rang, option in enumerate(options, start=1)
        ],
        "question": question,
        "source": source,
        "emise_le": timezone.now().isoformat(),
    }


class PresentChoicesTool(BaseTool):
    """
    Une question a 2-4 reponses en un tap, ancrees dans le planning reel.

    Reserve a v2 (V2_SEULEMENT dans tools/__init__.py): la reponse part dans
    done.quick_replies par la DEMANDE rangee dans data. Chaque option doit
    exister (bloc, tache, creneau libre, jour) et aucune ne peut affirmer une
    action: une question n'est pas un canal pour dire « j'ai deplace ».
    """

    name = "present_choices"
    description = (
        "Pose UNE question courte avec 2 à 4 réponses en un tap, tirées du planning réel. "
        "Utilise-le quand la réponse est bornée: lequel de plusieurs blocs ou tâches existants "
        "(source blocs ou taches), quel créneau libre (source creneaux, avec date; lis d'abord "
        "find_free_slots), quel jour (source jours). Le code rejette toute option qui n'existe "
        "pas et tout ce qui affirme une action. N'affirme rien: la question se pose avant d'agir."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "La question, courte (140 caractères max), qui finit par « ? ». Ex: 'Lequel de tes cours de chimie ?'",
            },
            "options": {
                "type": "array",
                "description": "2 à 4 réponses. Chaque option a un label court (40 caractères max) et une value: la phrase complète envoyée au tap (200 caractères max).",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Texte du bouton, ex: '13 h à 14 h' ou 'Chimie générale'"},
                        "value": {"type": "string", "description": "Message envoyé au tap, ex: 'Va pour 13 h à 14 h jeudi.'"},
                    },
                    "required": ["label", "value"],
                },
            },
            "source": {
                "type": "string",
                "enum": list(SOURCES_CHOIX),
                "description": "D'où viennent les options: creneaux (créneaux libres d'une date), blocs, taches, jours.",
            },
            "date": {
                "type": "string",
                "description": "Date des créneaux au format YYYY-MM-DD. Obligatoire quand source vaut creneaux.",
            },
        },
        "required": ["question", "options", "source"],
    }

    @staticmethod
    def _refus(message: str, **data) -> ToolResult:
        return ToolResult(success=False, data=data, message=message)

    def execute(self, user: User, **kwargs) -> ToolResult:
        from services.agent_v2.mesure import fuite_question

        question = _texte_court(kwargs.get("question"), MAX_QUESTION_CHOIX)
        if question is None or not question.endswith("?"):
            return self._refus(
                "Choix non présenté : la question doit être courte (140 caractères max) "
                "et finir par « ? ».")
        source = kwargs.get("source")
        if source not in SOURCES_CHOIX:
            return self._refus(
                f"Choix non présenté : source inconnue '{source}'. "
                f"Sources valides : {', '.join(SOURCES_CHOIX)}.")
        if fuite_question(question):
            return self._refus(
                "Choix non présenté : la question affirme une action. "
                "Pose-la sans rien raconter de ce qui a été fait.")

        jour = None
        brute = kwargs.get("date")
        if brute not in (None, ""):
            iso = _iso_date(brute)
            jour = date.fromisoformat(iso) if iso else None
        if source == "creneaux" and jour is None:
            return self._refus(
                "Choix non présenté : une date YYYY-MM-DD est obligatoire pour des créneaux.")

        candidates, vus, rejetees = [], set(), []
        brutes = kwargs.get("options")
        for option in brutes if isinstance(brutes, list) else []:
            if not isinstance(option, dict):
                continue
            label = _texte_court(option.get("label"), MAX_LIBELLE_CHOIX)
            value = _texte_court(option.get("value"), MAX_VALEUR_CHOIX)
            if label is None or value is None:
                continue
            if _plat(label) in vus:
                continue
            if fuite_question(label) or fuite_question(value):
                rejetees.append(label)
                continue
            vus.add(_plat(label))
            candidates.append({"label": label, "value": value})

        ancrees = self._ancrer(user, source, jour, candidates, rejetees)
        ancrees = ancrees[:MAX_OPTIONS_CHOIX]
        if len(ancrees) < MIN_OPTIONS_CHOIX:
            detail = f" Options écartées : {', '.join(rejetees)}." if rejetees else ""
            seul = self._seul_element_reel(user, source, brutes)
            if seul is not None:
                # Banc r8, s03-2: un seul cours proche plus « Un autre cours »
                # inventé. « Pose plutôt une question courte » faisait
                # redemander au lieu de créer un cours dont l'utilisateur
                # venait de donner les jours et les heures.
                return self._refus(
                    f"Choix non présenté : un seul élément réel correspond (« {seul} »), "
                    "il n'y a donc rien à choisir. Si l'utilisateur ajoute un cours ou une "
                    "activité avec ses jours et ses heures, crée-le (create_block) sans "
                    "demander lequel. Sinon, vise cet élément." + detail,
                    options_ecartees=rejetees)
            return self._refus(
                "Choix non présenté : il faut au moins 2 options réelles (créneaux libres, "
                "blocs, tâches ou jours existants). Pose plutôt une question courte." + detail,
                options_ecartees=rejetees)

        parametres = {
            "question": question,
            "options": [{"label": o["label"], "value": o["value"]} for o in ancrees],
            "source": source,
        }
        cible = {}
        if jour is not None:
            parametres["date"] = jour.isoformat()
            cible["date"] = jour.isoformat()
        demande = _demande_de_choix(question, source, ancrees, parametres, cible)
        return ToolResult(
            success=True,
            data={"demande": demande},
            message=(
                f"Choix présenté à l'utilisateur ({len(ancrees)} options). "
                "Attends sa réponse, ne pose pas d'autre question."
            ),
        )

    def _ancrer(self, user, source: str, jour, candidates: list, rejetees: list) -> list:
        """Garde les options qui designent une vraie entite, avec leur cible."""
        if source in ("blocs", "taches"):
            titres = self._titres(user, source)
            gardees = []
            for option in candidates:
                titre = _titre_contenu(_plat(option["label"]), titres)
                if titre is None:
                    rejetees.append(option["label"])
                    continue
                gardees.append({**option, "cible": {"titre": titre}})
            return gardees

        if source == "jours":
            gardees = []
            for option in candidates:
                trouve = _JOUR_CHOIX.search(_plat(option["label"]))
                if trouve is None:
                    rejetees.append(option["label"])
                    continue
                cible = {}
                mot = trouve.group(1)
                if mot in WEEKDAYS:
                    cible["jour"] = WEEKDAYS.index(mot)
                elif _iso_date(trouve.group(0)):
                    cible["date"] = trouve.group(0)
                gardees.append({**option, "cible": cible})
            return gardees

        # creneaux: chaque plage doit tenir ENTIERE dans un trou libre du jour,
        # et ne pas etre deja passee.
        from services.scheduling.placement import open_intervals

        maintenant = _maintenant()
        aujourdhui = maintenant.date()
        if jour < aujourdhui:
            rejetees.extend(option["label"] for option in candidates)
            return []
        plancher = maintenant.hour * 60 + maintenant.minute if jour == aujourdhui else 0
        libres = open_intervals(user, jour, 0, 24 * 60)
        gardees = []
        for option in candidates:
            plage = _minutes_du_libelle(option["label"])
            if plage is None:
                rejetees.append(option["label"])
                continue
            debut, fin = plage
            if debut < plancher or not any(s <= debut and fin <= e for s, e in libres):
                rejetees.append(option["label"])
                continue
            gardees.append({**option, "cible": {
                "date": jour.isoformat(), "debut": _hhmm(debut),
                "fin": _hhmm(fin % (24 * 60))}})
        return gardees

    def _seul_element_reel(self, user, source: str, brutes) -> str | None:
        """Le titre du SEUL bloc ou tache reel que nomment les options, sinon None.

        Lit les libelles bruts: une option reelle ecartee parce que sa valeur
        raconte une action compte encore comme element reel.
        """
        if source not in ("blocs", "taches") or not isinstance(brutes, list):
            return None
        titres = self._titres(user, source)
        reels = set()
        for option in brutes:
            if not isinstance(option, dict):
                continue
            label = _texte_court(option.get("label"), MAX_LIBELLE_CHOIX)
            if label is None:
                continue
            titre = _titre_contenu(_plat(label), titres)
            if titre is not None:
                reels.add(titre)
        return reels.pop() if len(reels) == 1 else None

    @staticmethod
    def _titres(user, source: str) -> dict:
        if source == "taches":
            from core.models import Task
            noms = Task.objects.filter(user=user).values_list("title", flat=True)
        else:
            from core.models import RecurringBlock, ScheduledBlock
            noms = list(RecurringBlock.objects.filter(user=user, active=True)
                        .values_list("title", flat=True))
            noms += list(ScheduledBlock.objects.filter(user=user)
                         .values_list("task__title", flat=True))
        return {_plat(nom): nom for nom in noms if nom and _plat(nom)}


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
