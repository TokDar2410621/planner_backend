"""
Ce que l'utilisateur lit, rendu par du CODE depuis les donnees des outils.

Jusqu'au 2026-09-14, le bloc factuel imprimait ToolResult.message, un texte
ecrit pour le modele: pluriels « (s) », dates ISO, « Refus: », « 5 x
create_block », exceptions brutes (enquete answer-rendering, formes S02 a
S19 du catalogue). Ici chaque phrase se construit depuis `Action.donnees`,
jamais depuis `message`, avec un seul formateur pour les heures (« 9 h 30 »),
les dates (« jeu. 24 sept. », « demain ») et les pluriels.

Module PUR: aucune requete en base, aucun appel de modele. `aujourdhui` se
lit a l'appel (import paresseux de django.utils.timezone), les tests passent
une date fixe.

La garantie de verite ne bouge pas: seules les actions du registre se
racontent, et une action retenue par une garde ne se raconte jamais comme
faite.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime

from services.agent_v2.registre import Registre

JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_JOURS_ABREGES = ("lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim.")
_MOIS_ABREGES = ("janv.", "févr.", "mars", "avr.", "mai", "juin",
                 "juil.", "août", "sept.", "oct.", "nov.", "déc.")
_JOURS_ANGLAIS = ("monday", "tuesday", "wednesday", "thursday",
                  "friday", "saturday", "sunday")
_RELATIFS = {0: "aujourd'hui", 1: "demain", -1: "hier"}

# Ordre de priorite des questions d'un tour (contrat, section 8). rendu ne
# pose que les motifs portes par une DEMANDE; les autres sont choisis par
# l'agent.
PRIORITE = [
    "portee_jour", "destructif", "heure_refusee", "creation_en_masse",
    "optimisation", "formulaire", "choix_modele", "chevauchement",
    "fin_recurrence", "creneaux", "dire",
]

# Une action retenue par une garde n'a PAS eu lieu. Elle se tait quand la
# question du tour la couvre, sinon une seule ligne dit qu'elle attend.
MOTIFS_RETENUS = frozenset({"portee_jour", "destructif", "creation_en_masse", "optimisation"})

LECTURES_RENDUES = ("get_week_schedule", "get_today_schedule", "list_blocks",
                    "find_free_slots", "list_tasks")

# Jamais racontes: un formulaire et un choix sont des questions, un import
# recent est du contexte pour DIRE.
_IGNORES = frozenset({"present_form", "present_choices", "import_recent"})

# Au-dela, les reussites d'une meme famille se regroupent en une ligne.
SEUIL_GROUPEMENT = 5

_DEJA_FAIT = "Deja fait par le code"


# ── Formateurs ──────────────────────────────────────────────────────────────

def _txt(valeur) -> str:
    return str(valeur).strip() if valeur is not None else ""


def _fin(texte: str) -> str:
    """Une abreviation finale (« sept. ») ne prend pas un second point."""
    return re.sub(r"\.\.$", ".", texte or "")


def _plat(texte: str) -> str:
    """Minuscules, sans accents: pour comparer, jamais pour afficher."""
    brut = unicodedata.normalize("NFKD", _txt(texte).lower())
    return brut.encode("ascii", "ignore").decode("ascii")


def _hm(valeur):
    """(heures, minutes) depuis « 09:30 », « 9h30 », un objet time; sinon None."""
    if valeur is None or isinstance(valeur, bool):
        return None
    if hasattr(valeur, "hour") and hasattr(valeur, "minute"):
        return valeur.hour, valeur.minute
    m = re.match(r"^\s*(\d{1,2})\s*(?::|h)\s*(\d{2})?", _txt(valeur))
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2) or 0)
    if h > 24 or mi > 59:
        return None
    return h, mi


def _minutes(valeur):
    t = _hm(valeur)
    return None if t is None else t[0] * 60 + t[1]


def heure(hhmm: str) -> str:
    """« 09:00 » -> « 9 h », « 09:30 » -> « 9 h 30 », « 00:00 » -> « minuit »."""
    t = _hm(hhmm)
    if t is None:
        return _txt(hhmm)
    h, m = t
    if m == 0 and h in (0, 24):
        return "minuit"
    return f"{h} h" if m == 0 else f"{h} h {m:02d}"


def plage(debut: str, fin: str) -> str:
    """« 10 h à 11 h 50 ». Un chevauchement de minuit reste tel quel (19 h à 2 h).

    23:59 est la fin technique d'un morceau du soir (schedule_task_at): pour
    l'utilisateur, c'est minuit.
    """
    fin_txt = "minuit" if _hm(fin) == (23, 59) else heure(fin)
    return f"{heure(debut)} à {fin_txt}"


def _fuseau():
    from zoneinfo import ZoneInfo
    try:
        from django.conf import settings
        return ZoneInfo(settings.TIME_ZONE or "America/Toronto")
    except Exception:  # noqa: BLE001 - hors Django, l'heure murale reste Toronto
        return ZoneInfo("America/Toronto")


def _date(valeur):
    """Une date locale depuis « 2026-09-24 », un datetime ISO ou un objet date."""
    if valeur is None:
        return None
    if isinstance(valeur, datetime):
        dt = valeur
    elif isinstance(valeur, date):
        return valeur
    else:
        brut = _txt(valeur)
        if not brut:
            return None
        try:
            dt = datetime.fromisoformat(brut.replace("Z", "+00:00"))
        except ValueError:
            try:
                return date.fromisoformat(brut[:10])
            except ValueError:
                return None
    if dt.tzinfo is not None:
        # Une echeance stockee en UTC tombe souvent le lendemain a 3 h 59:
        # sans conversion, « pour demain » deviendrait « pour après-demain ».
        dt = dt.astimezone(_fuseau())
    return dt.date()


def _aujourdhui(aujourdhui):
    if aujourdhui is not None:
        return aujourdhui
    from django.utils import timezone
    return timezone.localdate()


def _jour_mois(d: date) -> str:
    return f"{'1er' if d.day == 1 else d.day} {_MOIS_ABREGES[d.month - 1]}"


def date_courte(iso: str, aujourdhui: date | None = None) -> str:
    """« aujourd'hui », « demain », « hier », sinon « jeu. 24 sept. »."""
    d = _date(iso)
    if d is None:
        return _txt(iso)
    auj = _aujourdhui(aujourdhui)
    relatif = _RELATIFS.get((d - auj).days)
    if relatif:
        return relatif
    # Banc du 2026-09-14: une revision placee en 2025 s'affichait « jeu. 18
    # sept. », sans rien qui trahisse la mauvaise annee.
    annee = f" {d.year}" if d.year != auj.year else ""
    return f"{_JOURS_ABREGES[d.weekday()]} {_jour_mois(d)}{annee}"


def _quand(iso, aujourdhui) -> str:
    """« demain » ou « le jeu. 24 sept. », pour glisser dans une phrase."""
    if _date(iso) is None:
        return ""
    txt = date_courte(iso, aujourdhui)
    return txt if txt in _RELATIFS.values() else f"le {txt}"


def _dow(valeur):
    """Jour de semaine 0 = lundi, depuis un entier, un nom ou une abreviation."""
    if valeur is None or isinstance(valeur, bool):
        return None
    if isinstance(valeur, int):
        return valeur if 0 <= valeur <= 6 else None
    txt = _plat(valeur).strip(" .")
    if not txt:
        return None
    if txt.isdigit():
        n = int(txt)
        return n if 0 <= n <= 6 else None
    if len(txt) > 3 and txt.endswith("s"):
        txt = txt[:-1]  # « lundis »
    for i, nom in enumerate(JOURS):
        if len(txt) >= 3 and nom.startswith(txt):
            return i
    for i, nom in enumerate(_JOURS_ANGLAIS):
        if len(txt) >= 3 and nom.startswith(txt):
            return i
    return None


def jour(dow: int, pluriel: bool = False) -> str:
    """3 -> « jeudi », « jeudis » au pluriel."""
    d = _dow(dow)
    if d is None:
        return ""
    return JOURS[d] + ("s" if pluriel else "")


def _liste(elements) -> str:
    items = [e for e in elements if e]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " et " + items[-1]


def _jours_tries(dows) -> list[int]:
    return sorted({d for d in (_dow(x) for x in dows or []) if d is not None})


def _suite_continue(vus: list[int]) -> bool:
    return len(vus) >= 3 and vus == list(range(vus[0], vus[-1] + 1))


def jours(dows: list[int]) -> str:
    """[0,2,4] -> « lundi, mercredi et vendredi »; [0..4] -> « du lundi au vendredi »."""
    vus = _jours_tries(dows)
    if not vus:
        return ""
    if len(vus) == 7:
        return "tous les jours"
    if _suite_continue(vus):
        return f"du {JOURS[vus[0]]} au {JOURS[vus[-1]]}"
    return _liste([JOURS[d] for d in vus])


def _les_jours(dows) -> str:
    """Pour une habitude: « les lundis et mercredis », « du lundi au vendredi »."""
    vus = _jours_tries(dows)
    if not vus:
        return ""
    if len(vus) == 7:
        return "tous les jours"
    if _suite_continue(vus):
        return f"du {JOURS[vus[0]]} au {JOURS[vus[-1]]}"
    return "les " + _liste([JOURS[d] + "s" for d in vus])


def _le_jour(dows) -> str:
    """Pour un jour precis d'une habitude: « le lundi et le mercredi »."""
    vus = _jours_tries(dows)
    if _suite_continue(vus):
        return f"du {JOURS[vus[0]]} au {JOURS[vus[-1]]}"
    return _liste([f"le {JOURS[d]}" for d in vus])


def _jours_abreges(dows) -> str:
    vus = _jours_tries(dows)
    if not vus:
        return ""
    if len(vus) == 7:
        return "tous les jours"
    if _suite_continue(vus):
        return f"du {_JOURS_ABREGES[vus[0]]} au {_JOURS_ABREGES[vus[-1]]}"
    return _liste([_JOURS_ABREGES[d] for d in vus])


def pluriel(n: int, singulier: str, forme_plurielle: str | None = None) -> str:
    """« 1 cours », « 2 blocs ». En francais, 0 et 1 prennent le singulier."""
    forme = singulier if abs(n) <= 1 else (forme_plurielle or singulier + "s")
    return f"{n} {forme}"


# ── Marqueurs bruts: ce qui ne doit jamais atteindre l'utilisateur ─────────

_NOMS_D_OUTILS = re.compile(
    r"\b(?:get|list|create|update|delete|clear|skip|restore|schedule|cancel|"
    r"optimize|organize|find|check|complete|present|import|send|suggest|detect)"
    r"_[a-z_]+\b")
_MARQUEURS = (
    ("refus", re.compile(r"\brefus\b", re.IGNORECASE)),
    ("ecart", re.compile(r"\b[eé]carts?\b", re.IGNORECASE)),
    ("pluriel_machine", re.compile(r"\(s\)")),
    ("date_iso", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("heure_hhmm", re.compile(r"\b\d{2}:\d{2}\b")),
    ("nom_outil", _NOMS_D_OUTILS),
    ("ref_registre", re.compile(r"\(\s*[ae]\d+\s*\)")),
    ("compte_outil", re.compile(r"\b\d+ x [a-z_]+")),
    ("id_interne", re.compile(r"#\d+")),
    ("anglais", re.compile(r"\b(?:created|skipped|block|tool)\b", re.IGNORECASE)),
    # Les mots du systeme: l'utilisateur a des cours, des quarts et des
    # seances, pas des « blocs »; il voit des champs, pas un « formulaire »
    # (banc du 2026-09-14).
    ("vocabulaire_systeme", re.compile(r"\b(?:blocs?|formulaires?)\b", re.IGNORECASE)),
    # Le tiret long (U+2014) est banni de tout texte lu par l'utilisateur. Il
    # a atteint done.response au banc du round 3 sans que ce compteur le voie.
    ("tiret_long", re.compile("\u2014")),
)


def marqueurs_bruts(texte: str) -> list[str]:
    """Les formes machine presentes dans un texte montre a l'utilisateur."""
    if not texte:
        return []
    return sorted({nom for nom, motif in _MARQUEURS if motif.search(texte)})


# ── Petits lecteurs de donnees ─────────────────────────────────────────────

def _dict(valeur) -> dict:
    return valeur if isinstance(valeur, dict) else {}


def _dicts(valeur) -> list[dict]:
    return [v for v in (valeur or []) if isinstance(v, dict)] if isinstance(valeur, (list, tuple)) else []


def _demande(action):
    d = (action.donnees or {}).get("demande")
    return d if isinstance(d, dict) and d.get("motif") else None


def _titre_demande(demande) -> str:
    cible = _dict(demande.get("cible"))
    titre = _txt(cible.get("titre"))
    if titre:
        return titre
    titres = [t for t in (_txt(x) for x in cible.get("titres") or []) if t]
    if titres:
        return _liste(titres)
    return _txt(_dict(demande.get("parametres")).get("title"))


_RE_CHEVAUCHEMENT = re.compile(r"Chevauchement avec '(.+)' \((\d{1,2}:\d{2})-(\d{1,2}:\d{2})\)")


def _ressemble_sommeil(titre: str, block_type=None) -> bool:
    if block_type:
        return block_type == "sleep"
    return bool(re.search(r"\b(sommeil|dodo|dormir|nuit)\b", _plat(titre)))


# ── Faits: les reussites ───────────────────────────────────────────────────

@dataclass
class _Fait:
    texte: str
    famille: str = ""
    titre: str = ""
    nombre: int = 1


_FAMILLES = {
    "bloc_ajoute": ("créneau ajouté à ton horaire", "créneaux ajoutés à ton horaire"),
    "bloc_modifie": ("créneau modifié", "créneaux modifiés"),
    "bloc_supprime": ("créneau retiré de ton horaire", "créneaux retirés de ton horaire"),
    "occurrence_sautee": ("séance retirée pour une fois", "séances retirées pour une fois"),
    "occurrence_remise": ("séance remise", "séances remises"),
    "evenement_planifie": ("événement planifié", "événements planifiés"),
    "evenement_annule": ("événement annulé", "événements annulés"),
    "tache_ajoutee": ("tâche ajoutée à ta liste", "tâches ajoutées à ta liste"),
    "tache_modifiee": ("tâche modifiée", "tâches modifiées"),
    "tache_retiree": ("tâche retirée de ta liste", "tâches retirées de ta liste"),
    "tache_cochee": ("tâche cochée", "tâches cochées"),
    "objectif": ("objectif mis à jour", "objectifs mis à jour"),
}

_PREFERENCES = {
    "min_sleep_hours": "heures de sommeil",
    "peak_productivity_time": "moment où tu es le plus efficace",
    "transport_time_minutes": "temps de trajet",
    "max_deep_work_hours_per_day": "heures de travail concentré par jour",
    "onboarding_completed": "accueil terminé",
}


def _segments_horaire(creneaux) -> str:
    """{(dow, debut, fin)} -> « les lundis de 10 h à 11 h 50 et les jeudis de 8 h à 9 h »."""
    par_heure: dict[tuple, list[int]] = {}
    for dow, debut, fin in creneaux:
        par_heure.setdefault((debut, fin), []).append(dow)
    morceaux = []
    for (debut, fin), dows in sorted(par_heure.items(),
                                     key=lambda kv: (min(_jours_tries(kv[1]) or [9]), _minutes(kv[0][0]) or 0)):
        quand = _les_jours(dows)
        morceaux.append(f"{quand} de {plage(debut, fin)}" if quand else f"de {plage(debut, fin)}")
    return _liste(morceaux)


class _Narrateur:
    """Un passage sur le registre. Garde l'etat qui evite les doublons."""

    def __init__(self, registre: Registre, aujourdhui: date, cles_posees):
        self.registre = registre
        self.auj = aujourdhui
        self.cles_posees = cles_posees
        self.faits: list = []          # _Fait ou cle de creation fusionnee
        self.creations: dict = {}      # cle titre -> {"titre", "creneaux", "ids", "borne"}
        self.sections: list[str] = []  # blocs multi-lignes (import, plan de semaine)
        self.nb_imports = 0
        self.refus: list[str] = []
        self.retenues: list[str] = []
        self.couverts: set = set()     # (action_id, genre d'ecart deja dit)
        self.cles_heure: set = set()
        self.cles_retenues: set = set()

    # -- utilitaires

    def quand(self, iso) -> str:
        return _quand(iso, self.auj)

    def ajouter_fait(self, fait: _Fait) -> None:
        if not fait.texte:
            return
        fait.texte = _fin(fait.texte)
        for deja in self.faits:
            if isinstance(deja, _Fait) and deja.texte == fait.texte:
                return
        self.faits.append(fait)

    def ajouter_refus(self, texte: str) -> None:
        texte = _fin(texte)
        if texte and texte not in self.refus:
            self.refus.append(texte)

    # -- reussites

    def reussite(self, a) -> None:
        d = a.donnees or {}
        p = a.parametres or {}
        outil = a.outil
        methode = getattr(self, f"_ok_{outil}", None)
        if methode is not None:
            methode(a, d, p)

    def _ok_create_block(self, a, d, p):
        for c in _dicts(d.get("created")):
            titre = _txt(c.get("title")) or _txt(p.get("title"))
            cle = ("create_block", titre.casefold())
            acc = self.creations.get(cle)
            if acc is None:
                acc = {"titre": titre, "creneaux": set(), "borne": None}
                self.creations[cle] = acc
                self.faits.append(cle)
            dow = _dow(c.get("day_of_week"))
            if dow is None:
                dow = _dow(c.get("day_name"))
            acc["creneaux"].add((dow, _txt(c.get("start_time")) or _txt(p.get("start_time")),
                                 _txt(c.get("end_time")) or _txt(p.get("end_time"))))
            borne = _dict(d.get("borne_auto")).get("end_date")
            if borne:
                acc["borne"] = borne
        demande = _demande(a)
        # Sur l'heure dite par l'utilisateur, le chevauchement devient la ligne
        # « 10 h lundi, c'est pris par X » rendue par heure_refusee.
        self.sautes_create_block(
            a, d, p, ignorer_chevauchement=bool(demande and demande.get("motif") == "heure_refusee"))

    def _fait_de_creation(self, acc) -> _Fait:
        creneaux = {c for c in acc["creneaux"] if c[0] is not None}
        texte = f"Ajouté : {acc['titre']}, {_segments_horaire(creneaux)}."
        if acc["borne"]:
            jusqua = date_courte(acc["borne"], self.auj)
            jusqua = f"jusqu'à {jusqua}" if jusqua in _RELATIFS.values() else f"jusqu'au {jusqua}"
            texte = texte[:-1] + f", cette semaine seulement ({jusqua})."
        return _Fait(texte, "bloc_ajoute", acc["titre"], max(1, len(creneaux)))

    def _ok_schedule_task_at(self, a, d, p):
        sb = _dict(d.get("scheduled_block"))
        titre = _txt(sb.get("title")) or _txt(p.get("title"))
        quand = self.quand(sb.get("date") or p.get("date"))
        debut = sb.get("start_time") or p.get("start_time")
        fin = sb.get("end_time") or p.get("end_time")
        texte = f"Planifié : {titre}"
        if quand:
            texte += f", {quand}"
        if _hm(debut) and _hm(fin):
            texte += f" de {plage(debut, fin)}"
            if sb.get("overnight"):
                texte += " (fin le lendemain)"
        self.ajouter_fait(_Fait(texte + ".", "evenement_planifie", titre))

    def _ok_update_block(self, a, d, p):
        b = _dict(d.get("block"))
        titre = _txt(b.get("title")) or _txt(p.get("title")) or "Ce créneau"
        avant = _dict(d.get("avant"))
        dow = _dow(b.get("day_of_week"))
        maintenant = ""
        if dow is not None and _hm(b.get("start_time")) and _hm(b.get("end_time")):
            maintenant = f"{_les_jours([dow])} de {plage(b['start_time'], b['end_time'])}"
        morceaux = []
        ancien_titre = _txt(avant.get("title"))
        if ancien_titre and ancien_titre != titre:
            morceaux.append(f"{ancien_titre} s'appelle maintenant {titre}.")
        dow_avant = _dow(avant.get("day_of_week"))
        heures_changees = avant and (
            _txt(avant.get("start_time"))[:5] != _txt(b.get("start_time"))[:5]
            or _txt(avant.get("end_time"))[:5] != _txt(b.get("end_time"))[:5])
        jour_change = avant and dow_avant is not None and dow is not None and dow_avant != dow
        if maintenant and (heures_changees or jour_change):
            if jour_change:
                ancien = f"{_les_jours([dow_avant])} de {plage(avant.get('start_time'), avant.get('end_time'))}"
            else:
                ancien = f"de {plage(avant.get('start_time'), avant.get('end_time'))}"
            morceaux.append(f"{titre} : maintenant {maintenant} (avant {ancien}).")
        elif avant and avant.get("flexibility") and b.get("flexibility") \
                and avant.get("flexibility") != b.get("flexibility"):
            nature = "souple" if b.get("flexibility") == "flexible" else "fixe"
            morceaux.append(f"{titre} est maintenant {nature}" + (f", {maintenant}." if maintenant else "."))
        elif not morceaux:
            morceaux.append(f"{titre} mis à jour" + (f" : {maintenant}." if maintenant else "."))
        if p.get("end_date") and b.get("end_date"):
            morceaux.append(f"Dernière séance {self.quand(b.get('end_date'))}.")
        self.ajouter_fait(_Fait(" ".join(morceaux), "bloc_modifie", titre))

    def _ok_delete_block(self, a, d, p):
        b = _dict(d.get("block"))
        titre = _txt(b.get("title"))
        if not titre:
            self.ajouter_fait(_Fait("Un créneau est retiré de ton horaire.", "bloc_supprime", ""))
            return
        dow = _dow(b.get("day_of_week"))
        detail = ""
        if dow is not None and _hm(b.get("start_time")) and _hm(b.get("end_time")):
            detail = f", {_les_jours([dow])} de {plage(b['start_time'], b['end_time'])}"
        self.ajouter_fait(_Fait(f"Retiré de ton horaire : {titre}{detail}.", "bloc_supprime", titre))

    def _ok_clear_all_blocks(self, a, d, p):
        n = d.get("deleted_count")
        if not isinstance(n, int):
            self.ajouter_fait(_Fait("Ton planning est vidé."))
        elif n == 0:
            self.ajouter_fait(_Fait("Ton planning était déjà vide."))
        else:
            texte = f"Ton planning est vidé : {pluriel(n, 'créneau archivé', 'créneaux archivés')}"
            if d.get("reversible"):
                texte += ", tu peux les récupérer"
            self.ajouter_fait(_Fait(texte + "."))

    def _ok_skip_block_occurrence(self, a, d, p):
        titre = _txt(d.get("title")) or _txt(p.get("title")) or "Ce créneau"
        iso = d.get("date") or p.get("date")
        quand = self.quand(iso)
        jour_date = _date(iso)
        texte = f"{titre} n'a pas lieu {quand}".rstrip()
        if jour_date is not None:
            texte += f", les autres {JOURS[jour_date.weekday()]}s restent"
        self.ajouter_fait(_Fait(texte + ".", "occurrence_sautee", titre))

    def _ok_restore_block_occurrence(self, a, d, p):
        if d.get("restored") is False:
            self.dire_ecart(a, "rien_a_restaurer", {"titre": d.get("title"), "date": d.get("date")})
            return
        titre = _txt(d.get("title")) or _txt(p.get("title")) or "Ce créneau"
        quand = self.quand(d.get("date") or p.get("date"))
        self.ajouter_fait(_Fait(f"{titre} est de retour {quand}.".replace(" .", "."),
                                "occurrence_remise", titre))

    def _ok_create_task(self, a, d, p):
        t = _dict(d.get("task"))
        titre = _txt(t.get("title")) or _txt(p.get("title"))
        deja = d.get("deja_presente") is True or any(
            e.action_id == a.id and e.genre == "tache_existante" for e in self.registre.ecarts)
        if deja:
            self.dire_ecart(a, "tache_existante", {"titre": titre})
            return
        texte = f"Ajouté à ta liste : {titre}"
        if t.get("deadline") and _date(t.get("deadline")):
            texte += f" (pour {date_courte(t['deadline'], self.auj)})"
        self.ajouter_fait(_Fait(texte + ".", "tache_ajoutee", titre))

    def _ok_update_task(self, a, d, p):
        t = _dict(d.get("task"))
        titre = _txt(t.get("title")) or _txt(p.get("title")) or "Cette tâche"
        texte = f"Tâche mise à jour : {titre}"
        if t.get("deadline") and _date(t.get("deadline")):
            texte += f" (pour {date_courte(t['deadline'], self.auj)})"
        self.ajouter_fait(_Fait(texte + ".", "tache_modifiee", titre))

    def _ok_delete_task(self, a, d, p):
        titre = _txt(d.get("title"))
        if titre:
            self.ajouter_fait(_Fait(f"Retiré de ta liste : {titre}.", "tache_retiree", titre))
        else:
            self.ajouter_fait(_Fait("Une tâche est retirée de ta liste.", "tache_retiree", ""))

    def _ok_complete_task(self, a, d, p):
        titre = _txt(_dict(d.get("task")).get("title"))
        self.ajouter_fait(_Fait(f"Coché : {titre}." if titre else "Tâche cochée.",
                                "tache_cochee", titre))

    def _ok_cancel_scheduled_block(self, a, d, p):
        par_titre: dict[str, list[dict]] = {}
        for c in _dicts(d.get("cancelled")):
            par_titre.setdefault(_txt(c.get("title")) or _txt(p.get("title")), []).append(c)
        if not par_titre:
            self.ajouter_fait(_Fait("Événement annulé.", "evenement_annule", ""))
            return
        for titre, morceaux in par_titre.items():
            morceaux.sort(key=lambda c: (_txt(c.get("date")), _txt(c.get("start_time"))))
            premier, dernier = morceaux[0], morceaux[-1]
            texte = f"Annulé : {titre or 'ton événement'}"
            quand = self.quand(premier.get("date"))
            if quand:
                texte += f", {quand}"
            if _hm(premier.get("start_time")) and _hm(dernier.get("end_time")):
                texte += f" de {plage(premier['start_time'], dernier['end_time'])}"
            self.ajouter_fait(_Fait(texte + ".", "evenement_annule", titre))

    def _ok_organize_day(self, a, d, p):
        quand = self.quand(d.get("date") or p.get("date")) or "ta journée"
        places = [f"{_txt(r.get('title'))} de {plage(r.get('start_time'), r.get('end_time'))}"
                  for r in _dicts(d.get("moved" if d.get("applied") else "placed"))]
        if d.get("applied"):
            texte = (f"Journée réorganisée {quand} : {_liste(places)}." if places
                     else f"Rien à déplacer {quand} : c'est déjà bien placé.")
        else:
            texte = (f"Proposition pour {quand} : {_liste(places)}." if places
                     else f"Rien à replacer {quand}.")
            texte += " C'est une proposition, rien n'a changé."
            self.couverts.add((a.id, "plan_propose"))
        sans_place = [_txt(r.get("title")) for r in _dicts(d.get("skipped"))]
        if any(sans_place):
            texte += f" Pas de place pour {_liste(sans_place)}."
        self.ajouter_fait(_Fait(texte))

    def _ok_optimize_week(self, a, d, p):
        jours_data = _dicts(d.get("days"))
        if d.get("applied"):
            deplaces = []
            for j in jours_data:
                dj = _date(j.get("date"))
                for m in _dicts(j.get("moved")):
                    quand = _les_jours([dj.weekday()]) if dj else ""
                    deplaces.append(f"{_txt(m.get('title'))} {quand} de "
                                    f"{plage(m.get('start_time'), m.get('end_time'))}".replace("  ", " "))
            if not deplaces:
                self.ajouter_fait(_Fait("Rien à déplacer dans ta semaine : tout est déjà bien placé."))
                return
            texte = "Semaine réorganisée : " + _liste(deplaces[:6])
            if len(deplaces) > 6:
                texte += f", et {pluriel(len(deplaces) - 6, 'autre')}"
            self.ajouter_fait(_Fait(texte + "."))
            return
        lignes = []
        for j in jours_data:
            places = [f"{_txt(r.get('title'))} de {plage(r.get('start_time'), r.get('end_time'))}"
                      for r in _dicts(j.get("placed"))]
            sans_place = [_txt(r.get("title")) for r in _dicts(j.get("skipped"))]
            if not places and not sans_place:
                continue
            ligne = f"- {date_courte(j.get('date'), self.auj)} : " + (", ".join(places) or "rien à placer")
            if any(sans_place):
                ligne += f" (pas de place pour {_liste(sans_place)})"
            lignes.append(ligne)
        tete = "Proposition pour ta semaine. C'est une proposition, rien n'a changé."
        self.couverts.add((a.id, "plan_propose"))
        self.sections.append(tete + ("\n" + "\n".join(lignes) if lignes else ""))

    def _ok_update_preferences(self, a, d, p):
        champs = [c for c in d.get("updated_fields") or [] if c]
        if not champs:
            self.dire_ecart(a, "preferences_inchangees", {})
            return
        noms = []
        for c in champs:
            nom = _PREFERENCES.get(c, "une préférence")
            if nom not in noms:
                noms.append(nom)
        self.ajouter_fait(_Fait(f"Préférences mises à jour : {_liste(noms)}."))

    def _ok_create_goal(self, a, d, p):
        titre = _txt(_dict(d.get("goal")).get("title")) or _txt(p.get("title"))
        self.ajouter_fait(_Fait(f"Objectif ajouté : {titre}." if titre else "Objectif ajouté.",
                                "objectif", titre))

    def _ok_update_goal(self, a, d, p):
        g = _dict(d.get("goal"))
        titre = _txt(g.get("title")) or "Ton objectif"
        texte = f"Objectif mis à jour : {titre}"
        if isinstance(g.get("progress"), (int, float)):
            texte += f", {int(g['progress'])} % fait"
        self.ajouter_fait(_Fait(texte + ".", "objectif", titre))

    def _ok_import_document(self, a, d, p):
        recap = _recap_import(d, self.auj)
        if recap:
            # Les recaps d'import passent avant tout autre bloc multi-lignes.
            self.sections.insert(self.nb_imports, recap)
            self.nb_imports += 1

    # -- sauts de create_block (reussite partielle ou refus)

    def sautes_create_block(self, a, d, p, ignorer_chevauchement: bool = False):
        titre_defaut = _txt(p.get("title"))
        groupes: dict[tuple, dict] = {}
        for s in _dicts(d.get("skipped")):
            raison = _txt(s.get("reason"))
            motif = s.get("motif")
            avec = _dict(s.get("avec"))
            if not motif:
                if "existe déjà" in raison or "doublon" in _plat(raison):
                    motif = "doublon"
                elif raison.lower().startswith("chevauchement"):
                    motif = "chevauchement"
                    m = _RE_CHEVAUCHEMENT.search(raison)
                    if m and not avec:
                        avec = {"titre": m.group(1), "debut": m.group(2), "fin": m.group(3)}
                else:
                    motif = "jour_invalide"
            if motif == "chevauchement" and ignorer_chevauchement:
                continue
            titre = _txt(s.get("titre")) or titre_defaut
            debut = _txt(s.get("debut")) or _txt(p.get("start_time"))
            fin = _txt(s.get("fin")) or _txt(p.get("end_time"))
            cle = (motif, titre.casefold(), debut, fin, _txt(avec.get("titre")),
                   _txt(avec.get("debut")), _txt(avec.get("fin")))
            g = groupes.setdefault(cle, {"motif": motif, "titre": titre, "debut": debut,
                                         "fin": fin, "avec": avec, "jours": []})
            dow = _dow(s.get("day"))
            if dow is None:
                dow = _dow(s.get("day_name"))
            if dow is not None:
                g["jours"].append(dow)
        for g in groupes.values():
            self.ajouter_refus(self._phrase_saut(g))

    def _phrase_saut(self, g) -> str:
        titre = g["titre"] or "ce créneau"
        le_jour = _le_jour(g["jours"])
        if g["motif"] == "doublon":
            texte = f"{titre} est déjà à ton horaire"
            if le_jour:
                texte += f" {le_jour}"
            if _hm(g["debut"]):
                texte += f" à {heure(g['debut'])}"
            return texte + ", rien à ajouter."
        if g["motif"] == "chevauchement":
            texte = f"Je n'ai pas ajouté {titre}"
            if le_jour:
                texte += f" {le_jour}"
            if _hm(g["debut"]) and _hm(g["fin"]):
                texte += f" de {plage(g['debut'], g['fin'])}"
            avec = g["avec"]
            if _txt(avec.get("titre")):
                texte += f" : ça tombe en même temps que {avec['titre']}"
                if _hm(avec.get("debut")) and _hm(avec.get("fin")):
                    texte += f" ({plage(avec['debut'], avec['fin'])})"
            else:
                texte += " : ce moment est déjà pris"
            return texte + "."
        return f"Un des jours demandés pour {titre} n'existe pas, je l'ai laissé de côté."

    # -- refus

    def refus_de(self, a, index: int) -> None:
        d = a.donnees or {}
        p = a.parametres or {}
        outil = a.outil
        demande = _demande(a)
        motif = demande.get("motif") if demande else None

        if motif in MOTIFS_RETENUS:
            self.retenue(a, demande)
            return
        if motif == "heure_refusee":
            self.heure_refusee(a, demande)
            return
        if motif == "choix_modele" and outil != "present_choices":
            # Un ajout retenu par le code en attendant un choix (jour a
            # choisir): la question le couvre, sinon une ligne d'attente.
            self.retenue(a, demande)
            return
        if d.get("needs_confirmation") or d.get("requires_confirmation"):
            objet = _objet_retenu(outil, _txt(p.get("title")))
            self.ajouter_refus(f"Je n'ai pas encore {objet} : il me faut ton accord d'abord.")
            return

        if outil == "create_block" and (d.get("skipped") or d.get("created") is not None):
            if _dicts(d.get("skipped")):
                self.sautes_create_block(a, d, p)
                return

        if outil == "schedule_task_at" and _dict(d.get("conflict")):
            c = _dict(d["conflict"])
            titre = _txt(p.get("title")) or "cet événement"
            texte = f"Je n'ai pas planifié {titre}"
            quand = self.quand(p.get("date"))
            if quand:
                texte += f" {quand}"
            if _hm(p.get("start_time")) and _hm(p.get("end_time")):
                texte += f" de {plage(p['start_time'], p['end_time'])}"
            if c.get("sommeil"):
                texte += " : ça tombe pendant ton sommeil"
            elif _txt(c.get("titre")):
                texte += f" : c'est pris par {c['titre']}"
                if _hm(c.get("start_time")) and _hm(c.get("end_time")):
                    texte += f" ({plage(c['start_time'], c['end_time'])})"
            elif _hm(c.get("start_time")) and _hm(c.get("end_time")):
                texte += f" : ce moment est déjà pris ({plage(c['start_time'], c['end_time'])})"
            else:
                texte += " : ce moment est déjà pris"
            self.ajouter_refus(texte + ".")
            return

        if outil == "update_block" and _dict(d.get("conflit")):
            c = _dict(d["conflit"])
            titre = (_titre_demande(demande) if demande else "") or _txt(p.get("title")) or "ce créneau"
            texte = f"Je n'ai pas modifié {titre}"
            if _txt(c.get("titre")):
                texte += f" : ça tombe en même temps que {c['titre']}"
                if _hm(c.get("debut")) and _hm(c.get("fin")):
                    texte += f" ({plage(c['debut'], c['fin'])})"
            else:
                texte += " : ce moment est déjà pris"
            self.ajouter_refus(texte + ".")
            return

        if outil in ("skip_block_occurrence", "restore_block_occurrence") and d.get("candidates"):
            verbe = "retirer" if outil == "skip_block_occurrence" else "remettre"
            noms = [_txt(c.get("title")) for c in _dicts(d.get("candidates"))]
            quand = self.quand(p.get("date"))
            self.ajouter_refus(
                f"Je n'ai rien pu {verbe} {quand} : il y a plusieurs créneaux ce jour-là "
                f"({_liste(noms)}), dis-moi lequel.".replace("  ", " "))
            return

        if outil == "cancel_scheduled_block" and d.get("candidates"):
            noms = [_txt(c) for c in d.get("candidates") or [] if _txt(c)]
            quand = self.quand(p.get("date"))
            self.ajouter_refus(
                f"Je n'ai rien annulé {quand} : il y a plusieurs événements ce jour-là "
                f"({_liste(noms)}), dis-moi lequel.".replace("  ", " "))
            return

        # Echec sans donnee exploitable: on ne recopie JAMAIS le message (il
        # peut porter une exception brute ou un ordre au modele). Si la meme
        # intention a reussi plus loin dans le tour, l'echec etait une
        # tentative corrigee: le raconter contredirait la ligne de succes.
        if _reussie_plus_loin(self.registre.actions, index, a):
            return
        self.ajouter_refus(_echec_generique(outil, p))

    def retenue(self, a, demande) -> None:
        cle = _txt(demande.get("cle")) or f"{a.outil}:{a.id}"
        if self.cles_posees is None or cle in self.cles_posees or cle in self.cles_retenues:
            return
        self.cles_retenues.add(cle)
        motif = demande.get("motif")
        titre = _titre_demande(demande) or _txt((a.parametres or {}).get("title"))
        if motif == "optimisation":
            objet = "appliqué le plan de la semaine"
        elif motif == "creation_en_masse":
            objet = f"ajouté {titre}" if titre else "ajouté le reste"
        else:
            objet = _objet_retenu(_txt(demande.get("outil")) or a.outil, titre)
        self.retenues.append(f"Je n'ai pas encore {objet} : redemande-le-moi après ta réponse.")

    def heure_refusee(self, a, demande) -> None:
        d = a.donnees or {}
        # Une creation partielle garde ses jours reussis; seul le chevauchement
        # sur l'heure dite devient la ligne « c'est pris ».
        if a.outil == "create_block" and a.succes is False and _dicts(d.get("skipped")):
            self.sautes_create_block(a, d, a.parametres or {}, ignorer_chevauchement=True)
        cle = _txt(demande.get("cle")) or f"{a.outil}:{a.id}"
        if cle in self.cles_heure:
            return
        self.cles_heure.add(cle)
        cible = _dict(demande.get("cible"))
        p = a.parametres or {}
        debut = cible.get("debut") or p.get("start_time")
        moment = heure(debut) if _hm(debut) else ""
        if cible.get("recurrent") and _dow(cible.get("jour")) is not None:
            moment = f"{moment} le {jour(cible['jour'])}".strip()
        elif cible.get("date") and _date(cible.get("date")):
            moment = f"{moment} {date_courte(cible['date'], self.auj)}".strip()
        elif _dow(cible.get("jour")) is not None:
            moment = f"{moment} le {jour(cible['jour'])}".strip()
        occupant, sommeil = _occupant(d, debut)
        if sommeil:
            raison = "c'est pendant ton sommeil"
        elif occupant:
            raison = f"c'est pris par {occupant}"
        else:
            raison = "c'est déjà pris"
        texte = f"{moment}, {raison}." if moment else f"{raison[0].upper()}{raison[1:]}."
        self.ajouter_refus(texte)

    # -- ecarts

    def dire_ecart(self, a, genre: str, donnees: dict) -> None:
        if (a.id, genre) in self.couverts:
            return
        self.couverts.add((a.id, genre))
        texte = _phrase_ecart(genre, donnees, self.auj)
        if texte:
            self.ajouter_fait(_Fait(texte))

    # -- assemblage

    def lignes_de_faits(self) -> list[str]:
        faits: list[_Fait] = []
        for f in self.faits:
            if isinstance(f, _Fait):
                faits.append(f)
            else:
                faits.append(self._fait_de_creation(self.creations[f]))
        if len(faits) <= SEUIL_GROUPEMENT:
            return [f.texte for f in faits]
        par_famille: dict[str, list[_Fait]] = {}
        for f in faits:
            if f.famille:
                par_famille.setdefault(f.famille, []).append(f)
        lignes: list[str] = []
        vues: set = set()
        for f in faits:
            groupe = par_famille.get(f.famille) if f.famille else None
            if not groupe or len(groupe) < 2:
                lignes.append(f.texte)
                continue
            if f.famille in vues:
                continue
            vues.add(f.famille)
            singulier, plurielle = _FAMILLES[f.famille]
            total = sum(g.nombre for g in groupe)
            titres = []
            for g in groupe:
                if g.titre and g.titre not in titres:
                    titres.append(g.titre)
            texte = pluriel(total, singulier, plurielle)
            lignes.append(texte + (f" : {_liste(titres)}." if titres else "."))
        return lignes


def _occupant(donnees: dict, debut) -> tuple[str, bool]:
    """Qui occupe l'heure refusee, lu dans les donnees de l'outil."""
    conflit = _dict(donnees.get("conflict")) or _dict(donnees.get("conflit"))
    if conflit.get("sommeil"):
        return "", True
    if _txt(conflit.get("titre")):
        return _txt(conflit["titre"]), False
    for s in _dicts(donnees.get("skipped")):
        avec = _dict(s.get("avec"))
        if _txt(avec.get("titre")) and (not debut or not s.get("debut")
                                         or _txt(s.get("debut"))[:5] == _txt(debut)[:5]):
            return _txt(avec["titre"]), False
        m = _RE_CHEVAUCHEMENT.search(_txt(s.get("reason")))
        if m:
            return m.group(1), False
    return "", False


def _objet_abandonne(demande: dict) -> str:
    """Ce que le code laisse tomber, au nom: « la suppression de Gym »."""
    motif = demande.get("motif")
    outil = _txt(demande.get("outil"))
    titre = _titre_demande(demande) if demande else ""
    if motif == "optimisation" or outil == "optimize_week":
        return "le nouveau plan de ta semaine"
    if motif == "creation_en_masse":
        return "le reste des ajouts"
    if outil == "clear_all_blocks":
        return "le vidage de ton planning"
    if outil == "delete_task":
        return f"la suppression de la tâche {titre}" if titre else "la suppression de cette tâche"
    if outil == "cancel_scheduled_block":
        return f"l'annulation de {titre}" if titre else "l'annulation de cet événement"
    if outil == "update_block":
        return f"l'arrêt de {titre}" if titre else "l'arrêt de ce créneau"
    return f"la suppression de {titre}" if titre else "la suppression de ce créneau"


_SUPPRESSION_DE = "la suppression de "


def _objets_abandonnes(objets: list[str]) -> str:
    """« la suppression de A et de B », ou « X, Y et Z » quand les motifs
    different."""
    if len(objets) == 1:
        return objets[0]
    if all(o.startswith(_SUPPRESSION_DE) for o in objets):
        noms = [o[len(_SUPPRESSION_DE):] for o in objets]
        return _SUPPRESSION_DE + ", de ".join(noms[:-1]) + " et de " + noms[-1]
    return ", ".join(objets[:-1]) + " et " + objets[-1]


def _objet_retenu(outil: str, titre: str) -> str:
    if outil == "clear_all_blocks":
        return "vidé ton planning"
    if outil == "cancel_scheduled_block":
        return f"annulé {titre}" if titre else "annulé cet événement"
    if outil == "update_block":
        return f"arrêté {titre}" if titre else "arrêté ce créneau"
    if outil == "delete_task":
        return f"supprimé {titre}" if titre else "supprimé cette tâche"
    if outil == "optimize_week":
        return "appliqué le plan de la semaine"
    if outil == "schedule_task_at":
        return f"planifié {titre}" if titre else "planifié cet événement"
    return f"supprimé {titre}" if titre else "supprimé ce créneau"


_VERBES_ECHEC = {
    "create_block": ("ajouter", "ce créneau"),
    "update_block": ("modifier", "ce créneau"),
    "delete_block": ("supprimer", "ce créneau"),
    "clear_all_blocks": ("vider ton planning", ""),
    "skip_block_occurrence": ("retirer", "ce créneau pour une fois"),
    "restore_block_occurrence": ("remettre", "ce créneau"),
    "create_task": ("ajouter", "cette tâche"),
    "update_task": ("modifier", "cette tâche"),
    "delete_task": ("supprimer", "cette tâche"),
    "complete_task": ("cocher", "cette tâche"),
    "schedule_task_at": ("planifier", "cet événement"),
    "cancel_scheduled_block": ("annuler", "cet événement"),
    "optimize_week": ("réorganiser ta semaine", ""),
    "organize_day": ("réorganiser ta journée", ""),
    "update_preferences": ("changer tes préférences", ""),
    "create_goal": ("ajouter", "cet objectif"),
    "update_goal": ("modifier", "cet objectif"),
    "import_document": ("importer ton document", ""),
}


# Outils dont le parametre `title` NOMME la cible. Pour update_*, `title` est
# le NOUVEAU nom: « Je n'ai pas pu modifier X » designerait la mauvaise chose.
_TITRE_EST_LA_CIBLE = frozenset({
    "create_block", "create_task", "create_goal", "schedule_task_at",
    "cancel_scheduled_block", "skip_block_occurrence", "restore_block_occurrence",
})


def _echec_generique(outil: str, parametres: dict) -> str:
    verbe, objet = _VERBES_ECHEC.get(outil, ("faire cette modification", ""))
    titre = _txt(parametres.get("title")) if outil in _TITRE_EST_LA_CIBLE else ""
    if objet:
        objet = titre or objet
    return f"Je n'ai pas pu {verbe}{' ' + objet if objet else ''}, rien n'a changé."


def _identite(action) -> tuple:
    p = action.parametres or {}
    for cle in ("block_id", "task_id", "goal_id"):
        if p.get(cle) is not None:
            return (action.outil, cle, _txt(p[cle]))
    titre = _txt(p.get("title"))
    if titre:
        return (action.outil, "titre", titre.casefold())
    return (action.outil,)


def _reussie_plus_loin(actions, index: int, action) -> bool:
    ident = _identite(action)
    return any(b.succes and _identite(b) == ident for b in actions[index + 1:])


def _phrase_ecart(genre: str, donnees: dict, auj: date) -> str:
    d = donnees or {}
    titre = _txt(d.get("titre"))
    if genre == "passe":
        quand = _quand(d.get("date"), auj)
        if _hm(d.get("debut")):
            moment = f"{quand} à {heure(d['debut'])}".strip()
        elif _hm(d.get("fin")):
            moment = f"{quand}, jusqu'à {heure(d['fin'])}".strip(" ,")
        else:
            moment = quand
        sujet = f"{titre} : prévu" if titre else "C'était prévu"
        return f"{sujet} {moment}, c'est déjà passé.".replace("  ", " ")
    if genre == "date_differente":
        obtenue = _quand(d.get("obtenue"), auj)
        demandee = _quand(d.get("demandee"), auj)
        if not obtenue:
            return ""
        sujet = f"{titre} : placé" if titre else "Attention, c'est placé"
        suite = f", pas {demandee} comme demandé" if demandee else ""
        return f"{sujet} {obtenue}{suite}."
    if genre == "tache_existante":
        return f"{titre} était déjà dans ta liste." if titre else "Cette tâche était déjà dans ta liste."
    if genre == "plan_propose":
        return "C'est une proposition, rien n'a changé."
    if genre == "preferences_inchangees":
        return "Aucune de tes préférences n'a changé."
    if genre == "rien_a_restaurer":
        quand = _quand(d.get("date"), auj)
        if titre:
            return f"{titre} n'était pas retiré {quand}, rien à remettre.".replace("  ", " ").replace(" ,", ",")
        return "Il n'y avait rien à remettre."
    return ""


# ── Recap d'import ─────────────────────────────────────────────────────────

_RE_EXAMEN = re.compile(r"\b(examen|exam|intra|final|quiz|test|evaluation|partiel)\b")
_NOMS_IMPORT = {
    "work": ("quart", "quarts"),
    "sport": ("entraînement", "entraînements"),
}


def _recap_import(donnees: dict, auj: date) -> str:
    blocs = _dicts(donnees.get("blocs"))
    dates = _dicts(donnees.get("dates"))
    ignores = _dicts(donnees.get("ignores"))
    try:
        en_attente = int(donnees.get("en_attente") or 0)
    except (TypeError, ValueError):
        en_attente = 0

    # Un cours qui revient deux fois par semaine reste UN cours.
    titres_par_nom: dict[tuple, list[str]] = {}
    groupes: dict[str, dict] = {}
    for b in blocs:
        titre = _txt(b.get("titre"))
        if not titre:
            continue
        nom = _NOMS_IMPORT.get(_txt(b.get("type")), ("cours", "cours"))
        vus = titres_par_nom.setdefault(nom, [])
        if titre.casefold() not in vus:
            vus.append(titre.casefold())
        g = groupes.setdefault(titre.casefold(), {"titre": titre, "creneaux": {}})
        g["creneaux"].setdefault((_txt(b.get("debut")), _txt(b.get("fin"))), []).append(_dow(b.get("jour")))

    parties = [pluriel(len(titres), sing, plur) for (sing, plur), titres in titres_par_nom.items()]
    examens = sum(1 for d in dates if _RE_EXAMEN.search(_plat(d.get("titre"))))
    autres = len(dates) - examens
    if examens:
        parties.append(pluriel(examens, "examen"))
    if autres:
        parties.append(pluriel(autres, "événement"))

    if parties:
        tete = f"C'est importé : {_liste(parties)}."
    elif en_attente or ignores:
        tete = "C'est lu, mais rien n'est ajouté directement."
    else:
        return ""

    lignes = []
    for g in groupes.values():
        morceaux = []
        for (debut, fin), dows in g["creneaux"].items():
            quand = _jours_abreges([x for x in dows if x is not None])
            heures = plage(debut, fin) if _hm(debut) and _hm(fin) else ""
            morceaux.append(", ".join(m for m in (quand, heures) if m))
        lignes.append(f"{g['titre']} · " + " ; ".join(m for m in morceaux if m))
    for d in dates:
        moment = date_courte(d.get("date"), auj) if _date(d.get("date")) else ""
        if _hm(d.get("debut")) and _hm(d.get("fin")):
            moment = ", ".join(m for m in (moment, plage(d["debut"], d["fin"])) if m)
        lignes.append(f"{_txt(d.get('titre'))} · {moment}".rstrip(" ·"))

    verifier = []
    for i in ignores:
        dow = _dow(i.get("jour"))
        moment = ", ".join(m for m in (
            _JOURS_ABREGES[dow] if dow is not None else "",
            plage(i.get("debut"), i.get("fin")) if _hm(i.get("debut")) and _hm(i.get("fin")) else "",
        ) if m)
        titre = _txt(i.get("titre")) or "Un cours"
        verifier.append(f"{titre}{f' ({moment})' if moment else ''} n'est pas ajouté : "
                        "il tombe en même temps qu'un créneau déjà en place.")
    if en_attente:
        pronom = "le" if en_attente == 1 else "les"
        verifier.append(f"{pluriel(en_attente, 'créneau lu', 'créneaux lus')} avec un doute : "
                        f"confirme-{pronom} dans ton planning.")

    texte = tete
    if lignes:
        texte += "\n" + "\n".join(f"- {l}" for l in lignes)
    if verifier:
        texte += "\n\n**À vérifier**\n" + "\n".join(f"- {v}" for v in verifier)
    return texte


# ── API: les faits ─────────────────────────────────────────────────────────

def _deja_fait(action) -> bool:
    return (action.message or "").startswith(_DEJA_FAIT)


def rendre_faits(registre: Registre, aujourdhui: date | None = None,
                 cles_posees: set[str] | None = None) -> str:
    """Le compte rendu de ce que le tour a CHANGE, en francais lisible.

    Rien de ce qui sort d'ici ne vient de ToolResult.message. Les lectures
    ratees se taisent (un « Format de date invalide » ne concerne pas
    l'utilisateur), les actions retenues par une garde aussi quand la
    question du tour les couvre.
    """
    auj = _aujourdhui(aujourdhui)
    n = _Narrateur(registre, auj, cles_posees)
    abandonnes: list[str] = []

    for index, a in enumerate(registre.actions):
        if a.outil in _IGNORES or _deja_fait(a):
            continue
        # Round 6 (D2): une demande laissee tombee par le code se dit en UNE
        # ligne, meme si la question du tour porte la meme cle et quel que
        # soit l'outil sous lequel les gardes l'ont consignee. Plusieurs
        # demandes abandonnees partagent cette ligne unique.
        if (a.donnees or {}).get("abandonnee_par_le_code"):
            objet = _objet_abandonne(_dict((a.donnees or {}).get("demande")))
            if objet not in abandonnes:
                abandonnes.append(objet)
            continue
        if not a.est_mutation:
            continue
        if a.succes:
            n.reussite(a)
            demande = _demande(a)
            if demande and demande.get("motif") == "heure_refusee":
                n.heure_refusee(a, demande)
        else:
            n.refus_de(a, index)

    if abandonnes:
        n.ajouter_refus(f"Je laisse tomber {_objets_abandonnes(abandonnes)}. "
                        "Redis-le si tu veux toujours.")

    for e in registre.ecarts:
        if not e.genre:
            continue
        action = registre.par_id(e.action_id)
        if action is not None and _deja_fait(action):
            continue
        if (e.action_id, e.genre) in n.couverts:
            continue
        n.couverts.add((e.action_id, e.genre))
        texte = _phrase_ecart(e.genre, e.donnees, auj)
        if texte:
            n.ajouter_fait(_Fait(texte))

    lignes = n.lignes_de_faits() + n.refus + n.retenues
    if registre.budget_epuise:
        lignes.append("Je me suis arrêté avant la fin : il restait trop d'étapes.")
    if getattr(registre, "boucle_interrompue", False):
        lignes.append("Je me suis arrêté : je répétais la même étape.")

    sections = list(n.sections)
    if lignes:
        if len(lignes) == 1 and not sections:
            sections.append(lignes[0])
        else:
            sections.append("\n".join(f"- {l}" for l in lignes))
    return "\n\n".join(s for s in sections if s)


# ── API: les lectures ──────────────────────────────────────────────────────

_RE_BLOC_TEXTE = re.compile(r"^(.*)\s\((\d{1,2}:\d{2})-(\d{1,2}:\d{2})\)$")


def _items_semaine(jour_data: dict) -> list[dict]:
    items = []
    detail = _dicts(jour_data.get("detail"))
    if detail:
        for b in detail:
            items.append({"titre": _txt(b.get("title")), "debut": b.get("start_time"),
                          "fin": b.get("end_time"),
                          "sommeil": _ressemble_sommeil(_txt(b.get("title")), b.get("block_type"))})
        return items
    for brut in jour_data.get("blocks") or []:
        if isinstance(brut, dict):
            items.append({"titre": _txt(brut.get("title")), "debut": brut.get("start_time"),
                          "fin": brut.get("end_time"),
                          "sommeil": _ressemble_sommeil(_txt(brut.get("title")), brut.get("block_type"))})
            continue
        m = _RE_BLOC_TEXTE.match(_txt(brut))
        if m:
            items.append({"titre": m.group(1), "debut": m.group(2), "fin": m.group(3),
                          "sommeil": _ressemble_sommeil(m.group(1))})
    return items


def _ligne_item(item: dict) -> str:
    if _hm(item["debut"]) and _hm(item["fin"]):
        return f"- {plage(item['debut'], item['fin'])} · {item['titre']}"
    return f"- {item['titre']}"


def _ordre(item: dict) -> int:
    m = _minutes(item["debut"])
    return m if m is not None else 24 * 60


def _duree(item: dict) -> int:
    s, e = _minutes(item["debut"]), _minutes(item["fin"])
    if s is None or e is None:
        return 0
    return e - s if e > s else 24 * 60 - s + e


def _rendre_jours(jours_data: list[tuple], sujet: str, dire_vides: bool) -> str:
    """jours_data: [(dow, [items])] dans l'ordre de la semaine."""
    if not jours_data:
        return ""
    visibles = [(dow, [i for i in items if not i["sommeil"]]) for dow, items in jours_data]
    sommeils = [[i for i in items if i["sommeil"]] for _, items in jours_data]
    total = sum(len(v) for _, v in visibles)

    ligne_sommeil = ""
    if sommeils and all(len(s) == 1 for s in sommeils):
        heures = {(_txt(s[0]["debut"])[:5], _txt(s[0]["fin"])[:5]) for s in sommeils}
        if len(heures) == 1 and len(jours_data) == 7:
            debut, fin = next(iter(heures))
            if _hm(debut) and _hm(fin):
                ligne_sommeil = f"Sommeil : {plage(debut, fin)}"

    if total == 0:
        tete = f"Rien de prévu {sujet[1]}." if dire_vides else f"Rien {sujet[1]}."
        return tete + (f"\n\n{ligne_sommeil}" if ligne_sommeil else "")

    if len(jours_data) == 1 and not dire_vides:
        # Une lecture filtree sur un seul jour: le jour en tete suffit. Un
        # compte (« Ton horaire compte 2 blocs ») y passait pour tout l'horaire
        # et sautait le sommeil (banc du 2026-09-14).
        dow, items = visibles[0]
        lignes = [_ligne_item(i) for i in sorted(items, key=_ordre)]
        sections = [f"**{JOURS[dow].capitalize()}**\n" + "\n".join(lignes)]
        if ligne_sommeil:
            sections.append(ligne_sommeil)
        return "\n\n".join(sections)

    # Pas de compte en tete: « Ton horaire compte 10 blocs » parlait la langue
    # de l'outil (banc du 2026-09-14, s03-1, s04-1, s10-1bis). La tete dit ce
    # qui aide a lire: la journee la plus chargee.
    # Une semaine datee dit « mardi »; un horaire recurrent dit « le mardi ».
    article = "" if dire_vides else "le "
    tete = f"{sujet[0]}, jour par jour"
    charges = [(sum(_duree(i) for i in items), dow) for dow, items in visibles if items]
    if len(charges) > 1:
        maximum = max(c for c, _ in charges)
        plus = [dow for c, dow in charges if c == maximum and maximum > 0]
        if len(plus) == 1:
            tete = f"{sujet[0]} : la journée la plus chargée est {article}{JOURS[plus[0]]}"
        elif len(plus) == 2:
            tete = (f"{sujet[0]} : les plus chargées sont {article}{JOURS[plus[0]]} "
                    f"et {article}{JOURS[plus[1]]}")
    sections = [tete + "."]

    vides = []
    for dow, items in visibles:
        if not items:
            vides.append(dow)
            continue
        lignes = [_ligne_item(i) for i in sorted(items, key=_ordre)]
        sections.append(f"**{JOURS[dow].capitalize()}**\n" + "\n".join(lignes))
    if vides and dire_vides:
        sections.append(f"Rien de prévu {_liste([JOURS[d] for d in vides])}.")
    if ligne_sommeil:
        sections.append(ligne_sommeil)
    return "\n\n".join(sections)


def _lecture_semaine(d: dict, auj: date) -> str:
    jours_data = []
    for i, j in enumerate(_dicts(d.get("days"))):
        dj = _date(j.get("date"))
        dow = dj.weekday() if dj else _dow(j.get("day_name"))
        if dow is None:
            dow = i % 7
        jours_data.append((dow, _items_semaine(j)))
    return _rendre_jours(jours_data, ("Ta semaine", "cette semaine"), dire_vides=True)


def _lecture_horaire(d: dict, auj: date) -> str:
    par_jour: dict[int, list[dict]] = {}
    for b in _dicts(d.get("blocks")):
        dow = _dow(b.get("day_of_week"))
        if dow is None:
            dow = _dow(b.get("day_name"))
        if dow is None:
            continue
        par_jour.setdefault(dow, []).append({
            "titre": _txt(b.get("title")), "debut": b.get("start_time"), "fin": b.get("end_time"),
            "sommeil": _ressemble_sommeil(_txt(b.get("title")), b.get("block_type"))})
    if not par_jour:
        return "Rien à ton horaire pour l'instant." if "blocks" in d else ""
    jours_data = [(dow, par_jour[dow]) for dow in sorted(par_jour)]
    return _rendre_jours(jours_data, ("Ton horaire", "à ton horaire"), dire_vides=False)


def _libre(creneaux) -> str:
    morceaux = [plage(c.get("start_time"), c.get("end_time")) for c in _dicts(creneaux)
                if _hm(c.get("start_time")) and _hm(c.get("end_time"))]
    return ", ".join(morceaux)


def _lecture_journee(d: dict, auj: date) -> str:
    dj = _date(d.get("date"))
    dow = dj.weekday() if dj else _dow(d.get("day_name"))
    if dow is not None:
        tete = f"**{JOURS[dow].capitalize()}"
        if dj is not None:
            relatif = _RELATIFS.get((dj - auj).days)
            tete += f"** ({relatif})" if relatif else f" {_jour_mois(dj)}**"
        else:
            tete += "**"
    else:
        tete = "**Ta journée**"
    items = []
    for b in _dicts(d.get("blocks")):
        titre = _txt(b.get("title"))
        if _ressemble_sommeil(titre, b.get("block_type")):
            continue
        items.append({"titre": titre, "debut": b.get("start_time"), "fin": b.get("end_time")})
    corps = "\n".join(_ligne_item(i) for i in sorted(items, key=_ordre)) or "Rien de prévu."
    texte = f"{tete}\n{corps}"
    if "free_slots" in d:
        libre = _libre(d.get("free_slots"))
        texte += f"\n\nLibre : {libre}" if libre else "\n\nAucun moment libre dans la journée."
    return texte


def _lecture_creneaux(d: dict, auj: date) -> str:
    quand = date_courte(d.get("date"), auj) if _date(d.get("date")) else ""
    libre = _libre(d.get("free_slots"))
    if libre:
        return f"Libre {quand} : {libre}".replace("  ", " ")
    return f"Aucun créneau libre {_quand(d.get('date'), auj)}.".replace(" .", ".")


def _lecture_taches(d: dict, auj: date) -> str:
    taches = _dicts(d.get("tasks"))
    if not taches:
        return "Ta liste de tâches est vide."
    lignes = []
    for t in sorted(taches, key=lambda t: bool(t.get("completed"))):
        ligne = f"- {_txt(t.get('title'))}"
        if t.get("deadline") and _date(t.get("deadline")):
            ligne += f" (pour {date_courte(t['deadline'], auj)})"
        if t.get("completed"):
            ligne += " (fait)"
        lignes.append(ligne)
    return "Dans ta liste :\n" + "\n".join(lignes)


_LECTEURS = {
    "get_week_schedule": _lecture_semaine,
    "get_today_schedule": _lecture_journee,
    "list_blocks": _lecture_horaire,
    "find_free_slots": _lecture_creneaux,
    "list_tasks": _lecture_taches,
}


def rendre_lecture(registre: Registre, aujourdhui: date | None = None) -> str:
    """Ce que l'agent a VU, sur un tour qui n'a rien change.

    La derniere lecture reussie fait foi: si l'agent a relu, c'est la vue la
    plus recente qui compte.
    """
    if any(a.succes and a.est_mutation for a in registre.actions):
        return ""
    lectures = [a for a in registre.actions if a.succes and a.outil in _LECTEURS]
    if not lectures:
        return ""
    auj = _aujourdhui(aujourdhui)
    for action in reversed(lectures):
        texte = _LECTEURS[action.outil](action.donnees or {}, auj)
        if texte:
            return texte
    return ""


# ── API: les questions du code ─────────────────────────────────────────────

def _chip(label: str, valeur: str, option: str) -> dict:
    return {"label": label, "value": valeur, "option": option}


def _ids_options(demandes) -> set | None:
    ids = set()
    for d in demandes:
        for o in _dicts(d.get("options")):
            if _txt(o.get("id")):
                ids.add(_txt(o["id"]))
    return ids or None


def _garder(chips: list[dict], ids) -> list[dict]:
    if ids is None:
        return chips
    return [c for c in chips if c["option"] in ids]


def _titres(demandes) -> list[str]:
    titres = []
    for d in demandes:
        t = _titre_demande(d)
        if t and t not in titres:
            titres.append(t)
    return titres


def _dow_cible(cible: dict):
    dow = _dow(cible.get("jour"))
    if dow is None and _date(cible.get("date")):
        dow = _date(cible["date"]).weekday()
    return dow


def _question_portee_jour(demandes, auj):
    titres = _titres(demandes)
    cibles = [_dict(d.get("cible")) for d in demandes]
    dows = {_dow_cible(c) for c in cibles}
    dates = {_date(c.get("date")) for c in cibles if _date(c.get("date"))}
    nommes = _liste(titres) or "ce créneau"
    if len(dows) == 1 and None not in dows:
        j = JOURS[next(iter(dows))]
        date_txt = f" {_jour_mois(next(iter(dates)))}" if len(dates) == 1 else ""
        question = f"Tu veux enlever {nommes} seulement ce {j}{date_txt} ou tous les {j}s ?"
        chips = [
            _chip(f"Seulement ce {j}", f"Seulement ce {j}{date_txt} (sauter l'occurrence).", "occurrence"),
            _chip(f"Tous les {j}s", f"Tous les {j}s (supprimer la série).", "serie"),
        ]
    else:
        question = f"Tu veux enlever {nommes} seulement cette fois ou pour de bon ?"
        chips = [
            _chip("Seulement cette fois", "Seulement cette fois (sauter l'occurrence).", "occurrence"),
            _chip("Toute la série", "Toute la série (supprimer la série).", "serie"),
        ]
    chips.append(_chip("Non, garde tout", "Non, ne change rien.", "annuler"))
    return question, _garder(chips, _ids_options(demandes))


def _objet_destructif(demande, auj) -> str:
    outil = _txt(demande.get("outil"))
    cible = _dict(demande.get("cible"))
    titre = _titre_demande(demande)
    if outil == "clear_all_blocks":
        return "vider tout ton planning"
    if outil == "delete_task":
        return f"supprimer la tâche {titre}" if titre else "supprimer cette tâche"
    if outil == "cancel_scheduled_block":
        quand = _quand(cible.get("date"), auj)
        return f"annuler {titre or 'cet événement'}{' ' + quand if quand else ''}"
    if outil == "update_block":
        return f"arrêter {titre}" if titre else "arrêter ce créneau"
    detail = ""
    dow = _dow_cible(cible)
    if dow is not None and _hm(cible.get("debut")) and _hm(cible.get("fin")):
        detail = f" ({_les_jours([dow])} de {plage(cible['debut'], cible['fin'])})"
    return f"supprimer {titre}{detail}" if titre else "supprimer ce créneau"


def _question_destructif(demandes, auj):
    objets = []
    for d in demandes:
        o = _objet_destructif(d, auj)
        if o not in objets:
            objets.append(o)
    question = f"Tu veux vraiment {_liste(objets)} ?"
    chips = [
        _chip("Oui, confirme", "Oui, je confirme.", "confirmer"),
        _chip("Non, garde tout", "Non, ne change rien.", "annuler"),
    ]
    return question, _garder(chips, _ids_options(demandes))


def _question_heure_refusee(demandes, auj):
    titres = _titres(demandes)
    question = f"Quelle heure te va pour {_liste(titres)} ?" if titres else "Quelle heure te va plutôt ?"
    chips: list[dict] = []
    valeurs: set = set()
    autre_jour = False
    for d in demandes:
        cible_demande = _dict(d.get("cible"))
        for o in _dicts(d.get("options")):
            ident = _txt(o.get("id"))
            if ident == "autre_jour":
                autre_jour = True
                continue
            if not ident.startswith("creneau"):
                continue
            c = _dict(o.get("cible"))
            debut, fin = c.get("debut"), c.get("fin")
            if not (_hm(debut) and _hm(fin)):
                continue
            iso = c.get("date") or cible_demande.get("date")
            etiquette = plage(debut, fin)
            dow_option = _dow(c.get("jour") if c.get("jour") is not None else cible_demande.get("jour"))
            if (c.get("recurrent") or cible_demande.get("recurrent")) and dow_option is not None:
                valeur = f"Va pour {etiquette} le {JOURS[dow_option]}."
            elif _date(iso):
                valeur = _fin(f"Va pour {etiquette} {date_courte(iso, auj)}.")
            elif _dow(c.get("jour") if c.get("jour") is not None else cible_demande.get("jour")) is not None:
                j = _dow(c.get("jour") if c.get("jour") is not None else cible_demande.get("jour"))
                valeur = f"Va pour {etiquette} le {JOURS[j]}."
            else:
                valeur = f"Va pour {etiquette}."
            if valeur in valeurs:
                continue
            valeurs.add(valeur)
            chips.append(_chip(etiquette, valeur, ident))
    if autre_jour:
        chips.append(_chip("Un autre jour", "Je préfère un autre jour.", "autre_jour"))
    return question, chips


def _question_creation_en_masse(demandes, auj):
    titres = []
    deja = 0
    for d in demandes:
        cible = _dict(d.get("cible"))
        for t in [cible.get("titre")] + list(cible.get("titres") or []):
            if _txt(t) and _txt(t) not in titres:
                titres.append(_txt(t))
        try:
            deja = max(deja, int(cible.get("deja") or 0))
        except (TypeError, ValueError):
            pass
    if not titres:
        titres = [t for t in _titres(demandes)]
    suite = f"avec {_liste(titres)}" if titres else "les ajouts"
    tete = f"Ça fait déjà {pluriel(deja, 'ajout')} d'un coup. " if deja else ""
    question = f"{tete}Je continue {suite} ?"
    chips = [
        _chip("Oui, continue", "Oui, continue les ajouts.", "confirmer"),
        _chip("Non, arrête là", "Non, arrête là.", "annuler"),
    ]
    return question, _garder(chips, _ids_options(demandes))


def _question_optimisation(demandes, auj):
    chips = [
        _chip("Applique le plan", "Oui, applique le plan.", "confirmer"),
        _chip("Montre d'abord", "Montre-moi d'abord la proposition.", "annuler"),
    ]
    return "Je réorganise ta semaine selon ce plan ?", _garder(chips, _ids_options(demandes))


def _question_choix_modele(demandes, auj):
    d = demandes[0]
    question = _txt(d.get("question"))
    if question and not question.endswith("?"):
        question = question.rstrip(" .") + " ?"
    chips = []
    for o in _dicts(d.get("options")):
        libelle, valeur = _txt(o.get("libelle")), _txt(o.get("valeur"))
        if libelle and valeur:
            chips.append(_chip(libelle, valeur, _txt(o.get("id"))))
    return question, chips


def _question_chevauchement(demandes, auj):
    titres = _titres(demandes)
    nommes = _liste(titres)
    if nommes:
        verbe = "tombent" if len(titres) > 1 else "tombe"
        question = f"{nommes} {verbe} en même temps qu'autre chose. Je trouve une autre heure ?"
        valeur = f"Trouve une autre heure pour {nommes}."
    else:
        question = "Ça tombe en même temps qu'autre chose. Je trouve une autre heure ?"
        valeur = "Trouve une autre heure."
    chips = [
        _chip("Trouve une autre heure", valeur, "autre_heure"),
        _chip("Laisse faire", "Non, laisse faire.", "annuler"),
    ]
    return question, _garder(chips, _ids_options(demandes))


_QUESTIONS = {
    "portee_jour": _question_portee_jour,
    "destructif": _question_destructif,
    "heure_refusee": _question_heure_refusee,
    "creation_en_masse": _question_creation_en_masse,
    "optimisation": _question_optimisation,
    "choix_modele": _question_choix_modele,
    "chevauchement": _question_chevauchement,
}


def rendre_demandes(demandes: list[dict], aujourdhui: date | None = None) -> tuple[str, list[dict], list[str]]:
    """Une seule question pour le tour: le motif le plus prioritaire present.

    Rend (question, chips [{label, value, option}], cles rendues). Les
    demandes d'un motif moins prioritaire ne sont PAS rendues: leurs cles
    n'apparaissent pas, et rendre_faits dira qu'elles attendent.
    """
    valides = [d for d in demandes or [] if isinstance(d, dict) and d.get("motif") in _QUESTIONS]
    if not valides:
        return "", [], []
    motif = next(m for m in PRIORITE if any(d["motif"] == m for d in valides))
    # Toutes les demandes du motif entrent dans la question, meme celles qui
    # partagent une cle: deux ajouts retenus sous « creation_en_masse » doivent
    # etre nommes tous les deux (revue du 2026-09-14). Chaque fonction de
    # question dedoublonne ses objets; les cles rendues restent uniques.
    choisies = [d for d in valides if d["motif"] == motif]
    if motif == "choix_modele":
        # Deux questions du modele ne fusionnent pas: la premiere seule.
        choisies = choisies[:1]
    question, chips = _QUESTIONS[motif](choisies, _aujourdhui(aujourdhui))
    if not question:
        return "", [], []
    cles: list[str] = []
    for d in choisies:
        cle = _txt(d.get("cle"))
        if cle and cle not in cles:
            cles.append(cle)
    return question, chips, cles
