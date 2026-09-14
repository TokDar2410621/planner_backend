"""Suggestions differees: les chips proposees APRES la reponse.

Le client appelle /chat/quick-replies/ une fois le tour termine. Ces
suggestions sont generees par un second appel au modele, sans registre ni
garde: elles ne doivent donc jamais proposer une action destructive. Un tap
sur « Supprime tout mon jeudi » enverrait la demande destructive comme si
l'utilisateur l'avait tapee, alors qu'il ne l'a jamais formulee.

Elles ne doivent pas non plus concurrencer une question: quand le tour vient
de poser une question (chips de choix, formulaire), des suggestions
generiques en dessous brouillent la reponse attendue.
"""
import re
import unicodedata

# Accents retires et minuscules AVANT la recherche: « Enlève », « Réinitialise »
# et « EFFACE » tombent tous sous la meme regle.
_DESTRUCTIF = re.compile(
    r"\b(supprim\w*|effac\w*|vide[rz]?|enleve\w*|retir\w*|annul\w*|archiv\w*"
    r"|reset|recommenc\w*|reinitialis\w*)\b"
)

MAX_SUGGESTIONS = 3


def _normaliser(texte: str) -> str:
    decompose = unicodedata.normalize("NFKD", texte)
    return "".join(c for c in decompose if not unicodedata.combining(c)).lower()


def _destructive(texte: str) -> bool:
    return bool(_DESTRUCTIF.search(_normaliser(texte)))


def filtrer_suggestions(replies: list[dict]) -> list[dict]:
    """Garde au plus 3 suggestions bien formees et non destructives.

    Bien formee: un dictionnaire dont label et value sont des chaines non
    vides. Une suggestion dont le libelle OU la valeur touche au lexique
    destructif est retiree en entier: le libelle est ce que l'utilisateur
    lit, la valeur est ce qui part au tap.
    """
    gardees: list[dict] = []
    if not isinstance(replies, (list, tuple)):
        return gardees
    for item in replies:
        if not isinstance(item, dict):
            continue
        label, value = item.get("label"), item.get("value")
        if not isinstance(label, str) or not isinstance(value, str):
            continue
        if not label.strip() or not value.strip():
            continue
        if _destructive(label) or _destructive(value):
            continue
        gardees.append(item)
        if len(gardees) >= MAX_SUGGESTIONS:
            break
    return gardees


def tour_a_pose_une_question(user) -> bool:
    """Le dernier message de la conversation est-il une question de l'assistant ?

    Lu sur la metadonnee ecrite par l'agent v2 au moment de sauver sa reponse
    (question_posee, quick_replies, interactive_inputs). Le dernier message
    par pk seulement: si l'utilisateur a deja repondu, la question est close
    et les suggestions redeviennent utiles.
    """
    from core.models import ConversationMessage

    dernier = (ConversationMessage.objects
               .filter(user=user)
               .order_by("-pk")
               .only("role", "metadata", "content")
               .first())
    if dernier is None or dernier.role != "assistant":
        return False
    meta = dernier.metadata if isinstance(dernier.metadata, dict) else {}
    if (meta.get("question_posee") or meta.get("quick_replies")
            or meta.get("interactive_inputs")):
        return True
    # Filet (banc du round 3, s05-2): une question ecrite par v2 hors de son
    # champ laissait question_posee a faux, et des puces generiques venaient
    # contredire une question a deux issues.
    return meta.get("agent") == "v2" and bool(_QUESTION_DANS_LE_TEXTE.search(dernier.content or ""))


_QUESTION_DANS_LE_TEXTE = re.compile(r"\?(?:[\s\"'»)\]]|$)")
