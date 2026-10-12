"""Memoire durable des preferences: capture, gestion, injection.

Capture = l'ecriture (le message « souviens-toi que... » devient une ligne en
base). Injection = la lecture (chaque tour glisse les preferences dans le
brief d'AGIR et de DIRE). Les deux sont deterministes: aucun appel au modele
n'est necessaire pour memoriser, oublier ou lister.

Voies de capture:
1. Explicite: « souviens-toi que... », « rappelle-toi que... », etc.
2. Inferee: `chip_inference` propose une chip « Memoriser ? » quand le message
   porte un marqueur de durabilite (« toujours », « jamais »...). Le tap sur
   la chip rejoue la voie explicite: la confirmation reste a l'utilisateur.
3. Gestion: « oublie ... » desactive (jamais de hard delete),
   « que sais-tu de moi ? » liste.
"""
import re
import unicodedata

LIMITE_INJECTION = 15
LIMITE_STOCKAGE = 30

# -- Reconnaissance ---------------------------------------------------------

_CAPTURE = re.compile(
    r"^\s*(?:souviens?\s*-?\s*toi\s+que|rappelle\s*-?\s*toi\s+que"
    r"|retiens\s+que|note\s+que|[aà]\s+partir\s+de\s+maintenant[,:]?"
    r"|d[eé]sormais[,:]?)\s+(.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_OUBLI = re.compile(
    r"^\s*(?:oublie\s+(?:que\s+)?|ne\s+(?:te\s+)?souviens\s+plus\s+de\s+"
    r"|supprime\s+de\s+ta\s+memoire\s+)(.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_LISTE = re.compile(
    r"^\s*(?:qu['e]\s*est-ce\s+que\s+tu\s+sais\s+(?:de|sur)\s+moi"
    r"|que\s+sais\s*-?\s*tu\s+de\s+moi|mes\s+pr[ée]f[ée]rences"
    r"|ce\s+que\s+tu\s+as\s+retenu(?:\s+sur\s+moi)?)\s*[?!.]?\s*$",
    re.IGNORECASE,
)
# Marqueurs de durabilite pour la voie inferee (chip de confirmation).
_DURABLE = re.compile(
    r"\b(toujours|jamais|chaque\s+fois|tous\s+les|toutes\s+les"
    r"|je\s+pr[ée]f[èe]re|je\s+d[ée]teste|j['e]\s*aime\s+pas"
    r"|je\s+n['e]\s*aime\s+pas|d['e]\s*habitude|en\s+g[ée]n[ée]ral)\b",
    re.IGNORECASE,
)

def _normaliser(texte: str) -> str:
    decompose = unicodedata.normalize("NFKD", texte or "")
    return "".join(c for c in decompose if not unicodedata.combining(c)).lower()


# -- Interpretation ----------------------------------------------------------

class CommandeMemoire:
    """Commande detectee dans le message tape (tap exclu par l'appelant)."""

    def __init__(self, genre: str, enonce: str = "", reste: str = ""):
        # genre: memoriser | oublier | lister
        self.genre = genre
        self.enonce = enonce.strip()
        # Ce qui reste du message pour AGIR quand la commande est melangee
        # a une vraie demande (« souviens-toi que X. Planifie Y »).
        self.reste = reste.strip()

    @property
    def pure(self) -> bool:
        return not self.reste


def interpreter(message: str) -> "CommandeMemoire | None":
    """Detecte une commande memoire en tete de message. Rend None sinon."""
    if not message or not message.strip():
        return None
    m = _CAPTURE.match(message)
    if m:
        enonce = m.group(1).strip()
        # « Souviens-toi que X. Planifie Y »: la premiere phrase est la
        # preference, le reste va a AGIR.
        morceaux = re.split(r"(?<=[.!?])\s+", enonce, maxsplit=1)
        if len(morceaux) == 2 and len(morceaux[1]) > 3:
            preference = morceaux[0].rstrip(".!?").strip()
            return CommandeMemoire("memoriser", preference, morceaux[1])
        return CommandeMemoire("memoriser", enonce.rstrip(".!?").strip())
    m = _OUBLI.match(message)
    if m:
        return CommandeMemoire("oublier", m.group(1))
    if _LISTE.match(message):
        return CommandeMemoire("lister")
    return None


# -- Execution ----------------------------------------------------------------

def _categorie(enonce: str) -> str:
    n = _normaliser(enonce)
    if re.search(r"\b(heure|matin|soir|midi|lundi|mardi|mercredi|jeudi|vendredi"
                 r"|samedi|dimanche|semaine|jour|nuit|tard|tot|avant|apres)\b", n):
        return "horaire"
    if re.search(r"\b(adresse|chez|au\s|a\s+la|bureau|maison|appartement"
                 r"|centre[-\s]ville|gym|salle)\b", n):
        return "lieu"
    if re.search(r"\b(avec|docteur|dr\.?|monsieur|madame|mme?\.?|collegue"
                 r"|ami|famille)\b", n):
        return "personne"
    if re.search(r"\b(toujours|jamais|habitude|prefere|deteste)\b", n):
        return "habitude"
    return "autre"


def _actives(user):
    from core.models import PreferenceUtilisateur
    return list(PreferenceUtilisateur.objects
                .filter(user=user, actif=True)
                .order_by("-updated_at"))


def _cle_exacte(enonce: str) -> str:
    """Normalisation de forme seulement : conserve jours, nombres et négation."""
    return " ".join(unicodedata.normalize("NFC", enonce or "").casefold().split()).rstrip(".!?")


def memoriser(user, enonce: str, source: str = "dite",
              confiance: float = 1.0, *, remplace_id: int | None = None) -> str:
    """Ajoute une préférence ; seul un ID explicite autorise un remplacement.

    Le recouvrement lexical ne prouve ni identité ni contradiction. À la
    limite de stockage, demande un choix au lieu de désactiver une donnée.
    """
    from django.contrib.auth import get_user_model
    from django.db import transaction
    from core.models import PreferenceUtilisateur
    enonce = " ".join(enonce.split())
    if not enonce:
        return "Rien a memoriser: la preference est vide."
    if len(enonce) > 500:
        return "Cette préférence dépasse 500 caractères. Raccourcis-la avant de la mémoriser."
    with transaction.atomic():
        # Une ligne utilisateur existe même quand sa mémoire est vide.
        get_user_model().objects.select_for_update().get(pk=user.pk)
        actives = _actives(user)
        cible = next((p for p in actives if p.pk == remplace_id), None)
        if remplace_id is not None and cible is None:
            return "Préférence à remplacer introuvable. Rien n'a changé."
        doublon = next((p for p in actives if _cle_exacte(p.enonce) == _cle_exacte(enonce)), None)
        if doublon is not None:
            return f"Déjà mémorisé : « {doublon.enonce} »."
        if len(actives) >= LIMITE_STOCKAGE and cible is None:
            return "La mémoire a atteint sa limite. Choisis une préférence à oublier ou à remplacer."
        PreferenceUtilisateur.objects.create(
            user=user, enonce=enonce, categorie=_categorie(enonce), source=source, confiance=confiance)
        if cible is not None:
            cible.actif = False
            cible.save(update_fields=["actif", "updated_at"])
    if cible is not None:
        return (f"Preference memorisee: « {enonce} ». "
                f"Remplace: « {cible.enonce[:80]} ».")
    return f"Preference memorisee: « {enonce} »."


def oublier(user, recherche: str = "", *, preference_id: int | None = None) -> str:
    """Désactive un ID ou un énoncé exact, jamais une proximité de vocabulaire."""
    recherche = " ".join((recherche or "").split())
    if not recherche and preference_id is None:
        return "Precise ce que je dois oublier."
    candidates = [p for p in _actives(user)
                  if (p.pk == preference_id if preference_id is not None
                      else _cle_exacte(p.enonce) == _cle_exacte(recherche))]
    if not candidates:
        return "Précise l'énoncé exact de la préférence à oublier. Rien n'a changé."
    if len(candidates) > 1:
        liste = "; ".join(f"« {c.enonce[:60]} »" for c in candidates[:4])
        return (f"J'ai plusieurs souvenirs proches: {liste}. "
                "Precise lequel oublier.")
    cible = candidates[0]
    cible.actif = False
    cible.save(update_fields=["actif", "updated_at"])
    return f"Oublie: « {cible.enonce} »."


def lister(user) -> str:
    """Rend la liste des preferences actives, pour DIRE."""
    actives = _actives(user)
    if not actives:
        return "Je n'ai encore rien memorise sur tes preferences."
    lignes = [f"- {p.enonce}" for p in actives[:LIMITE_STOCKAGE]]
    return "Ce que j'ai retenu sur tes preferences:\n" + "\n".join(lignes)


# -- Injection -----------------------------------------------------------------

def section_memoire(user, limite: int = LIMITE_INJECTION) -> str:
    """Le texte glisse dans le brief d'AGIR et de DIRE. Vide si rien."""
    actives = _actives(user)[:limite]
    if not actives:
        return ""
    lignes = [f"  - {p.enonce}" for p in actives]
    return "MEMOIRE (preferences durables: applique-les sans les redemander):\n" \
        + "\n".join(lignes)


# -- Inference (chip de confirmation) ------------------------------------------

def deja_connue(user, message: str) -> bool:
    """La preference est-elle deja en memoire ? Evite les chips redondantes."""
    return any(_cle_exacte(p.enonce) == _cle_exacte(message) for p in _actives(user))


def chip_inference(user, message: str) -> "dict | None":
    """Propose « Memoriser ? » quand le message sonne comme une preference
    durable. Le tap rejoue la voie explicite: confirmation par l'utilisateur,
    zero ecriture silencieuse."""
    if not message or not message.strip():
        return None
    if interpreter(message) is not None:
        return None  # deja une commande memoire
    if "?" in message:
        return None  # une question n'est pas une preference
    if not _DURABLE.search(message):
        return None
    if deja_connue(user, message):
        return None
    court = " ".join(message.split())
    if len(court) > 120:
        return None  # trop long pour une chip fiable
    return {"label": "Mémoriser ?",
            "value": f"souviens-toi que {court}"}
