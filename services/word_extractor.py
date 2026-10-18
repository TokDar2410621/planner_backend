"""Extract schedule text from modern and legacy Word documents."""

import logging
import os
import subprocess

from docx import Document
from docx.table import Table

MAX_WORD_TEXT_CHARS = 15_000
logger = logging.getLogger(__name__)


class WordExtractionError(ValueError):
    """The document contains no readable text or could not be parsed."""


class WordServiceUnavailable(WordExtractionError):
    """The legacy .doc converter is missing from the runtime."""


def extract_word_text(path: str, kind: str) -> str:
    """Read a .docx or .doc as text without sending its raw file to an LLM."""
    if kind == 'docx':
        try:
            document = Document(path)
            lignes = []
            for bloc in document.iter_inner_content():
                if isinstance(bloc, Table):
                    for ligne in bloc.rows:
                        cellules = [cellule.text.replace('\n', ' ').strip()
                                    for cellule in ligne.cells]
                        if any(cellules):
                            lignes.append(' | '.join(cellules))
                elif bloc.text.strip():
                    lignes.append(bloc.text.strip())
            texte = '\n'.join(lignes).strip()
        except Exception as exc:
            logger.warning('Lecture DOCX échouée (%s)', type(exc).__name__)
            raise WordExtractionError('Document Word illisible.') from None
    elif kind == 'doc':
        try:
            resultat = subprocess.run(
                ['antiword', '-w', '0', path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
                check=False,
                env={**os.environ, 'LC_ALL': 'C.UTF-8'},
            )
        except FileNotFoundError:
            raise WordServiceUnavailable('Convertisseur .doc indisponible.') from None
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning('Lecture DOC échouée (%s)', type(exc).__name__)
            raise WordExtractionError('Document Word illisible.') from None
        if resultat.returncode != 0:
            logger.warning('Lecture DOC échouée (code %s)', resultat.returncode)
            raise WordExtractionError('Document Word illisible.')
        texte = resultat.stdout.decode('utf-8', errors='replace').strip()
    else:
        raise ValueError('Format Word inconnu.')

    if not texte:
        raise WordExtractionError('Document Word sans texte lisible.')
    return texte[:MAX_WORD_TEXT_CHARS]
