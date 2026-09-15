"""
Les regles ou la lecture typee DECIDE (LIRE_REGLES).

Principes du fondateur, tenus par chaque regle:
- le modele lit la langue, le code decide sur des champs types: aucune liste
  de phrases, aucune regex sur le texte de l'utilisateur;
- une regle ajoute une question ou un formulaire, jamais une ecriture;
- sans lecture utilisable (absente, tardive, LIRE ou regle coupee), le tour
  est celui de main;
- les lecteurs regex geles restent le repli: une regle ne retire rien de ce
  que main aurait montre.

Regle formulaire_cours (bug « mets mon cours de maths », 2026-09-15): AGIR
voyait « Calcul differentiel », concluait qu'il n'y avait rien a faire, et
DIRE ecrivait une absence fausse avec des jours inventes en puces.
Regle creneaux (bug « aujourdui »): la jambe regex de v1 ne lit que
aujourd'hui, demain ou une date ISO; voir boutons.creneaux_types.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from services.agent.tools.base import ToolResult
from services.agent_v2 import lecture as _lecture
from services.agent_v2 import lecture_schema as schema

FORMULAIRE_COURS = "formulaire_cours"
CRENEAUX = "creneaux"

# L'operation ne sert qu'a EXCLURE: Gemini la lit souvent de travers, et une
# exclusion de trop ne coute qu'une regle qui ne se declenche pas.
OPERATIONS_EXCLUES = frozenset({"deplacer", "supprimer", "sauter_une_fois", "consulter",
                                "modifier", "restaurer"})
# Un element fixe par un tiers: un cours (labo compris) ou un quart de travail.
# Une revision ou une seance d'etude est souple: elle n'en fait pas partie.
GENRES_FIXES = frozenset({"course", "work"})
TITRES_MAX = 6
JOURS = ("Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche")


@dataclass(frozen=True)
class LectureDuTour:
    """La lecture validee d'un tour, avec ce qu'il faut pour la resoudre."""
    lecture: object
    refs: dict
    aujourdhui: date
    origine: str = "tape"


def lecture_du_tour(suivi, resultat) -> Optional[LectureDuTour]:
    """Une lecture ok ou partielle, preparee ce tour; sinon None (main decide)."""
    prep = getattr(suivi, "preparation", None)
    if prep is None or resultat is None or getattr(resultat, "lecture", None) is None:
        return None
    if resultat.statut not in (_lecture.OK, _lecture.PARTIELLE):
        return None
    return LectureDuTour(lecture=resultat.lecture, refs=prep.refs or {},
                         aujourdhui=prep.aujourdhui, origine=getattr(prep, "origine", "tape"))


def element_demande(lecture):
    """L'unique element demande, hors deplacement, retrait, lecture et
    modification; None des qu'il y en a zero ou plusieurs."""
    demandes = [e for e in lecture.elements if e.polarite == "demande"]
    if len(demandes) != 1 or demandes[0].operation in OPERATIONS_EXCLUES:
        return None
    return demandes[0]


def _nom(element) -> str:
    """Le nom de l'element: son extrait ancre, jamais un titre propose."""
    return " ".join((element.mention or "").split())


def _tour_sans_suite(registre, reemises) -> bool:
    """Ni mutation reussie, ni demande, ni formulaire, ni choix ce tour."""
    if reemises:
        return False
    for action in registre.actions:
        if action.succes and action.est_mutation:
            return False
        if isinstance((action.donnees or {}).get("demande"), dict):
            return False
        if action.outil in ("present_form", "present_choices"):
            return False
    return True


# ----------------------------------------------- regle 1: formulaire du code

def _titres_de_la_semaine(element, refs: dict) -> list[str]:
    semaine = (refs or {}).get("s") or {}
    titres: list[str] = []
    for ref in element.candidats:
        groupe = semaine.get(ref) if isinstance(ref, str) else None
        titre = " ".join(str((groupe or {}).get("titre") or "").split())
        if titre and titre not in titres:
            titres.append(titre)
    return titres[:TITRES_MAX]


def _formulaire(nom: str) -> ToolResult:
    """Construit par l'outil present_form lui-meme: meme normalisation qu'un
    formulaire du modele. Plage horaire sans defaut ni raccourci: la regle ne
    part que si la lecture ne dit ni jour ni heure, rien n'est a pre-remplir."""
    from services.agent.tools.interactive import PresentFormTool

    return PresentFormTool().execute(None, inputs=[
        {"id": "jours", "type": "checkbox", "label": f"Jours de « {nom} »",
         "question": f"Quels jours as-tu « {nom} » ?",
         "options": [{"value": str(i), "label": jour} for i, jour in enumerate(JOURS)]},
        # « Plage horaire », pas « Heures de ... »: la reponse du frontend
        # (« <libelle>: 16:00 - 17:50 ») passerait sinon pour une duree aux
        # lecteurs geles (_LIBELLE_DUREE), qui ne gardaient que 17:50, et la
        # garde des heures dites imposait ce mauvais debut (banc du 2026-09-15).
        {"id": "heures", "type": "time_range", "label": "Plage horaire",
         "question": f"À quelle heure commence et finit « {nom} » ?"},
    ])


def appliquer_formulaire_cours(ldt: Optional[LectureDuTour], registre, *, attachment,
                               reemises) -> str:
    """Pose le formulaire du code au registre et rend la prose du tour, ou "".

    Le formulaire entre par le chemin de _dernier_formulaire (un present_form
    reussi, marque par_le_code): il atteint done et les metadonnees comme un
    formulaire du modele, et la reponse « Voici mes réponses » porte le nom."""
    if ldt is None or attachment is not None or ldt.origine == "formulaire":
        return ""
    element = element_demande(ldt.lecture)
    if element is None or element.genre not in GENRES_FIXES:
        return ""
    if element.dates or element.jours_semaine or element.heures:
        return ""
    nom = _nom(element)
    if not nom or not _tour_sans_suite(registre, reemises):
        return ""
    # Un message en plusieurs parties garde main (relecture Codex: « mets mon
    # cours de maths et montre-moi mon horaire demain » perdait la lecture de
    # l'horaire sous le formulaire): un seul element lu, aucune reponse a une
    # question, et aucune consultation au registre autre que list_blocks, la
    # verification silencieuse d'AGIR. get_week_schedule affiche la semaine
    # demandee: le formulaire la cacherait si la lecture ratait cette partie.
    if len(ldt.lecture.elements) != 1 or ldt.lecture.reponses:
        return ""
    from services.agent_v2.rendu import LECTURES_RENDUES

    if any(a.succes and a.outil in LECTURES_RENDUES and a.outil != "list_blocks"
           for a in registre.actions):
        return ""
    resultat = _formulaire(nom)
    if not resultat.success:
        return ""
    lignes = []
    titres = _titres_de_la_semaine(element, ldt.refs)
    if titres:
        lignes.append(f"Déjà à ton horaire : {', '.join(titres)}")
    lignes.append(f"Quels jours et à quelles heures as-tu « {nom} » ?")
    donnees = {**(resultat.data or {}), "par_le_code": True, "regle": FORMULAIRE_COURS}
    registre.ajouter("present_form", {"regle": FORMULAIRE_COURS},
                     ToolResult(success=True, data=donnees, message=resultat.message))
    return "\n".join(lignes)


# --------------------------------------------------- regle 2: creneaux types

def _heure_ferme(element, role: str) -> Optional[int]:
    heures = [h for h in element.heures if h.role == role]
    if len(heures) != 1 or heures[0].genre != "ferme" or len(heures[0].lectures) != 1:
        return None
    try:
        hh, mm = heures[0].lectures[0].split(":")
        return int(hh) * 60 + int(mm)
    except (ValueError, AttributeError):
        return None


def appel_bloque_type(ldt: Optional[LectureDuTour]) -> Optional[dict]:
    """La fenetre qu'une demande de placement vise, si tout est tranche:
    {"titre", "date", "debut_min", "fin_min"}, sinon None.

    Une seule date, decidee par le resolveur du code (une question ouverte ne
    donne rien: c'est la regle 3, inactive); une heure de debut et une de fin,
    fermes et a une seule lecture."""
    if ldt is None:
        return None
    element = element_demande(ldt.lecture)
    if element is None or len(element.dates) != 1 or element.dates[0].genre != "placement":
        return None
    resolution = schema.resoudre_jour(element.dates[0], ldt.aujourdhui)
    if (resolution.question or resolution.rejet or resolution.fenetre
            or len(resolution.dates) != 1 or resolution.dates[0] < ldt.aujourdhui):
        return None
    debut, fin = _heure_ferme(element, "debut"), _heure_ferme(element, "fin")
    if debut is None or fin is None or len(element.heures) != 2 or fin <= debut:
        return None
    from services.agent.tools.schedule import DAY_END_MIN, DAY_START_MIN

    # Hors de la journee ou cherchent les creneaux libres (7 h a 23 h), une
    # fenetre libre passerait pour occupee (relecture Codex, 23 h a 23 h 30).
    if debut < DAY_START_MIN or fin > DAY_END_MIN:
        return None
    return {"titre": _nom(element) or "cet événement", "date": resolution.dates[0],
            "debut_min": debut, "fin_min": fin}
