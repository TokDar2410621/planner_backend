"""Le chargeur d'outils: chercher un outil rare, puis l'appeler.

Mesure du 2026-10-01 sur 600 tours reels: 75 % des tours n'appellent AUCUN
outil, et les 33 outils pesaient 10 400 jetons a chaque tour. Dix-sept d'entre
eux (3 243 jetons) ne servent presque jamais, dont neuf jamais sur
l'echantillon. Ils passent derriere deux outils: on cherche, puis on appelle.

Regle de securite non negociable: `appeler_outil` ne reimplemente RIEN. Il
fabrique l'executeur du vrai outil, exactement comme `outils_pour`, et le
laisse passer par le meme chemin: verrou du tour, gardes destructives,
registre, idempotence, reconciliation. Un appel indirect qui court-circuiterait
ce chemin supprimerait un bloc sans confirmation.
"""
from __future__ import annotations

import json
import re
import unicodedata

MAX_RESULTATS = 4
# Au-dela, le modele recoit la liste des noms seuls: lui servir vingt schemas
# couterait ce que ce chargeur economise.
MAX_NOMS = 20


def _plat(texte: str) -> str:
    sans = unicodedata.normalize("NFKD", texte or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", sans.lower())


def _mots(texte: str) -> list[str]:
    # Les mots de deux lettres ou moins ne discriminent rien.
    return [m for m in _plat(texte).split() if len(m) > 2]


def _score(besoin: list[str], nom: str, description: str) -> int:
    """Combien de mots du besoin se retrouvent dans l'outil.

    Le nom compte double: « supprime une tache » doit trouver delete_task
    avant un outil dont la description parle de taches en passant. Ce n'est
    pas de la comprehension d'intention (le besoin est ecrit par le modele,
    pas par la personne): c'est une recherche documentaire.
    """
    cible_nom = f" {_plat(nom)} "
    cible_desc = f" {_plat(description)} "
    total = 0
    for mot in besoin:
        if mot in cible_nom:
            total += 2
        elif mot in cible_desc:
            total += 1
    return total


def _schema_compact(parametres: dict) -> str:
    """Le schema sans sa prose: noms, types, requis, enums et valeurs par defaut.

    Les descriptions de parametres restent dans l'outil lui-meme; les relayer
    ici annulerait l'economie qu'on cherche.
    """
    props = (parametres or {}).get("properties") or {}
    requis = set((parametres or {}).get("required") or [])
    morceaux = []
    for nom, regle in props.items():
        bout = f"{nom}: {regle.get('type') or 'string'}"
        if regle.get("enum"):
            bout += " = " + "|".join(str(v) for v in regle["enum"])
        if nom in requis:
            bout += " (requis)"
        morceaux.append(bout)
    return ", ".join(morceaux) or "aucun parametre"


def chercher(besoin: str, outils: list, description_de) -> str:
    """Rend les outils qui repondent au besoin, prets a etre appeles.

    `outils` sont les outils CACHES (les exposes s'appellent directement), et
    `description_de` rend la description v2 d'un outil.
    """
    termes = _mots(besoin)
    if not termes:
        noms = ", ".join(sorted(o.name for o in outils)[:MAX_NOMS])
        return (f"Dis ce que tu cherches a faire. Outils disponibles ici: {noms}.")

    classes = sorted(((_score(termes, o.name, description_de(o)), o.name, o)
                      for o in outils), key=lambda t: (-t[0], t[1]))
    # Seuls les outils proches du meilleur: servir trois schemas hors sujet
    # couterait ce que ce chargeur economise. Mesure du 2026-10-01:
    # check_feasibility remontait sur presque tous les besoins.
    meilleur = classes[0][0] if classes else 0
    seuil = max(1, (meilleur + 1) // 2)
    trouves = [(n, o) for s, n, o in classes if s >= seuil][:MAX_RESULTATS]
    if not trouves:
        noms = ", ".join(sorted(o.name for o in outils)[:MAX_NOMS])
        return (f"Aucun outil ne correspond a « {besoin} ». "
                f"Outils disponibles ici: {noms}.")

    lignes = [f"{len(trouves)} outil(s). Appelle-les avec "
              f"appeler_outil(nom=..., parametres={{...}}):"]
    for nom, outil in trouves:
        lignes.append(f"\n{nom}")
        lignes.append(f"  {description_de(outil)}")
        lignes.append(f"  parametres: {_schema_compact(outil.parameters)}")
    return "\n".join(lignes)


def requis_manquants(parametres_schema: dict, donnes: dict) -> list[str]:
    requis = list((parametres_schema or {}).get("required") or [])
    fournis = donnes or {}
    return [c for c in requis if c not in fournis
            or (isinstance(fournis.get(c), str) and not fournis[c].strip())]


def lire_parametres(brut) -> tuple[dict | None, str]:
    """Le modele envoie parfois un objet JSON, parfois sa chaine.

    Rend (parametres, erreur). Une erreur non vide est le message a lui rendre.
    """
    if brut is None or brut == "":
        return {}, ""
    if isinstance(brut, dict):
        return dict(brut), ""
    if isinstance(brut, str):
        try:
            lu = json.loads(brut)
        except (TypeError, ValueError):
            return None, ("parametres illisibles: envoie un objet JSON, "
                          "par exemple {\"block_id\": 12}.")
        if isinstance(lu, dict):
            return lu, ""
    return None, "parametres doit etre un objet, pas une liste ni une valeur seule."
