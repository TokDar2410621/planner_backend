"""
Transcription vocale : l'audio du bouton micro devient du texte de chat.

La voix n'est qu'une PORTE D'ENTREE vers le pipeline texte existant : le
texte transcrit atterrit dans le champ de saisie du front, l'utilisateur le
relit et l'envoie comme s'il l'avait tape. Transcrire n'autorise rien :
aucune action, aucun tour d'agent ne part d'ici.

Gemini transcrit (le fournisseur deja en place, audio natif) : pas de
nouveau vendor, pas de SDK embarque, le francais quebecois est bien gere.
"""
import logging

from django.conf import settings

logger = logging.getLogger(__name__)

# Safari enregistre en audio/mp4 (AAC), Chrome et Firefox en audio/webm.
# Le reste couvre les enregistreurs natifs usuels.
MIMES_ACCEPTES = frozenset({
    "audio/webm", "audio/mp4", "audio/mpeg", "audio/mp3", "audio/wav",
    "audio/x-wav", "audio/ogg", "audio/aac", "audio/m4a", "audio/x-m4a",
})
# Une dictee de chat, pas un podcast: ~90 s d'audio compresse tiennent
# largement ici, et la borne coupe les uploads accidentels.
TAILLE_MAX_OCTETS = 5 * 1024 * 1024

_CONSIGNE = (
    "Transcris mot à mot, en français, ce que dit la personne dans cet "
    "audio. Rends UNIQUEMENT la transcription, sans guillemets, sans "
    "commentaire, sans ponctuation inventée en fin de phrase. Si l'audio "
    "est vide, silencieux ou inintelligible, rends une chaîne vide."
)


class TranscriptionIndisponible(Exception):
    """Le fournisseur n'est pas configure ou a echoue: le client recoit un
    message generique, jamais le detail (une HTTPError porte l'URL entiere)."""


def transcrire_audio(donnees: bytes, mime_type: str) -> str:
    """Le texte dit dans `donnees`, ou une chaine vide si rien d'audible."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover - dependance du deploiement
        raise TranscriptionIndisponible("google-genai absent") from e
    if not getattr(settings, "GEMINI_API_KEY", ""):
        raise TranscriptionIndisponible("GEMINI_API_KEY absente")

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    try:
        reponse = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=donnees, mime_type=mime_type),
                _CONSIGNE,
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                # Budget explicite, jamais le mode dynamique: gemini-2.5-flash
                # rend parfois un candidat VIDE en dynamique (voir
                # GeminiProvider.THINKING_BUDGET). Transcrire ne demande pas
                # de raisonnement.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    except Exception as e:  # noqa: BLE001 - le detail reste au serveur
        logger.error("Transcription en echec: %s", type(e).__name__, exc_info=True)
        raise TranscriptionIndisponible("appel Gemini en echec") from e

    texte = (getattr(reponse, "text", None) or "").strip()
    logger.info("Transcription: %d octets audio -> %d caracteres", len(donnees), len(texte))
    return texte
