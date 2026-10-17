"""Harnais d'execution des outils, style OpenCode.

Un outil est un module qui expose NAME, DESCRIPTION, PARAMETERS et
``execute(user, **kwargs) -> ToolResult``. PARAMETERS est genere depuis
``module.Parametres`` (un modele pydantic, source de verite typee),
comme OpenCode genere son JSON Schema depuis son schema type.

Autour de chaque appel, le harnais applique, dans l'ordre :

1. crochets avant execution (style OpenCode ``tool.execute.before``) :
   la v2 y branche sa chaine de gardes ; la v1 n'en met aucun ;
2. validation des arguments contre le schema type de l'outil, AVANT
   execute (style OpenCode ``decodeInput``/``validateInput`` dans
   ``tool/runtime.ts``) : l'outil recoit des donnees deja valides ;
3. contrat d'erreur type a la frontiere : aucune exception ne remonte,
   tout devient un ToolResult (issue/cause/public) ;
4. pas de reemission aveugle : sur panne transitoire, c'est le chemin
   reconciliation de la v2 qui verifie avant de rejouer (jamais de
   retry aveugle, meme borne) ;
5. boucles : detection de repetition a l'identique (registre,
   REPETITIONS_TOLEREES) et plafond de pas par tour.

References : sst/opencode, packages/core/src/tool/runtime.ts,
packages/core/src/session/model-request.ts,
packages/core/src/session/runner/{max-steps,retry}.ts.
"""
import json
import logging

from pydantic import ValidationError

from .base import ISSUE_BESOIN_PRECISION, ISSUE_ECHEC, ToolResult

logger = logging.getLogger(__name__)


# Crochets "avant execution", style OpenCode tool.execute.before.
# Chaque crochet recoit (nom, kwargs) et rend soit None (laisse passer),
# soit un ToolResult de refus qui court-circuite l'appel.
# La v2 y branche sa chaine de gardes ; la v1 n'en met aucun.
crochets_avant: list = []


def _erreur_validation(nom: str, erreur: ValidationError, recus: dict) -> ToolResult:
    """Rend une erreur de validation actionnable, style OpenCode.

    OpenCode (tool/runtime.ts, formatInputIssues) liste les champs en
    faute, montre les arguments recus et dit au modele de corriger et
    rappeler l'outil. Meme contrat ici.
    """
    details = []
    fautes = erreur.errors()
    for faute in fautes[:5]:
        chemin = ".".join(str(p) for p in faute["loc"]) or "racine"
        details.append(f"- {chemin} : {faute['msg']}")
    if len(fautes) > 5:
        details.append(f"- ... et {len(fautes) - 5} autres")
    try:
        recus_str = json.dumps(recus, ensure_ascii=False, default=str)[:2000]
    except Exception:  # noqa: BLE001
        recus_str = "<illisible>"
    message = (
        f'Arguments invalides pour l\'outil "{nom}" :\n'
        + "\n".join(details)
        + f"\n\nArguments recus :\n{recus_str}"
        "\n\nCorrige les arguments et rappelle l'outil."
    )
    return ToolResult(
        success=False,
        data={},
        message=message,
        issue=ISSUE_BESOIN_PRECISION,
        cause="parametres_invalides",
        public="Je n'ai pas pu lancer l'outil : un paramètre est invalide.",
    )


def valider(module, kwargs: dict) -> tuple:
    """Valide kwargs contre module.Parametres.

    Rend (kwargs_valides, None) ou (None, ToolResult d'erreur).
    Les kwargs prives (prefixe ``_``, injectes par les gardes de la v2 :
    _arrangements, _expected_entries...) ne font pas partie du schema
    du modele : ils traversent sans validation.
    """
    modele = getattr(module, "Parametres", None)
    if modele is None:
        return dict(kwargs or {}), None
    kwargs = dict(kwargs or {})
    prives = {k: v for k, v in kwargs.items() if k.startswith("_")}
    publics = {k: v for k, v in kwargs.items() if not k.startswith("_")}
    try:
        valide = modele.model_validate(publics)
    except ValidationError as e:
        return None, _erreur_validation(module.NAME, e, publics)
    # exclude_unset : un champ non fourni reste absent, comme avant
    # (l'ancien code testait `field in kwargs`).
    fusion = valide.model_dump(exclude_unset=True)
    fusion.update(prives)
    return fusion, None


def appeler(module, user, kwargs: dict) -> ToolResult:
    """Chemin complet d'un appel d'outil : crochets, validation, execution.

    Ne leve jamais : toute exception devient un ToolResult typé
    (contrat a la frontiere, comme OpenCode qui convertit toute erreur
    etrangere en Tool.Error dans tool/runtime.ts).
    """
    nom = module.NAME
    for crochet in crochets_avant:
        try:
            refus = crochet(nom, kwargs)
        except Exception:  # noqa: BLE001
            logger.error("Crochet avant execution en panne sur %s", nom, exc_info=True)
            continue
        if refus is not None:
            return refus

    valides, erreur = valider(module, kwargs)
    if erreur is not None:
        return erreur

    try:
        resultat = module.execute(user, **valides)
    except Exception as e:  # noqa: BLE001
        logger.error("Outil %s a leve : %s", nom, e, exc_info=True)
        return ToolResult(
            success=False,
            data={},
            message=f"Erreur de l'outil : {e}",
            issue=ISSUE_ECHEC,
            cause="exception_outil",
            public="L'outil a rencontré une erreur.",
        )
    if not isinstance(resultat, ToolResult):
        logger.error("Outil %s : retour inattendu %r", nom, type(resultat))
        return ToolResult(
            success=False,
            data={},
            message=f"Erreur de l'outil : retour inattendu ({type(resultat).__name__})",
            issue=ISSUE_ECHEC,
            cause="retour_inattendu",
            public="L'outil a rencontré une erreur.",
        )
    return resultat


def en_claude(module) -> dict:
    """Definition d'un outil au format attendu par le modele."""
    return {
        "name": module.NAME,
        "description": module.DESCRIPTION,
        "input_schema": module.PARAMETERS,
    }
