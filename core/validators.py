"""
Upload validation helpers for Planner AI backend.

S4: uploads must be constrained by size, extension allowlist AND magic
bytes so an attacker cannot smuggle an executable/SVG/HTML payload past a
simple extension check (stored-XSS / DoS).
"""
import os
from zipfile import BadZipFile, ZipFile

from rest_framework.exceptions import ValidationError

# Maximum allowed upload size (8 MB).
MAX_UPLOAD_SIZE = 8 * 1024 * 1024

# Allowed file extensions (lowercase, without the leading dot).
# heic/heif: photos iPhone. Acceptees depuis le fix extraction-horaires:
# avant, une photo iPhone etait rejetee en 400 alors que c'est le cas
# d'usage principal de l'envoi d'horaires en photo.
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'webp', 'heic', 'heif', 'doc', 'docx'}

WORD_OLE_HEADER = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'
MAX_WORD_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_WORD_ARCHIVE_ENTRIES = 512

# Marques (brands) ISO BMFF reconnues comme HEIC/HEIF: les 4 octets a
# l'offset 8 quand les octets 4-8 valent 'ftyp'.
HEIC_BRANDS = (
    b'heic', b'heix', b'hevc', b'hevx', b'heim', b'heis', b'hevm', b'hevs',
    b'mif1', b'msf1',
)

# Magic byte signatures keyed by canonical type. Each entry is a list of
# (offset, signature_bytes) tuples that must ALL match for the file to be
# accepted as that type.
MAGIC_SIGNATURES = {
    'pdf': [(0, b'%PDF-')],
    'png': [(0, b'\x89PNG\r\n\x1a\n')],
    'jpg': [(0, b'\xff\xd8\xff')],
    # WEBP is a RIFF container: "RIFF"...."WEBP".
    'webp': [(0, b'RIFF'), (8, b'WEBP')],
    # HEIC/HEIF is an ISO BMFF container: "ftyp" at offset 4, brand at
    # offset 8. La signature accepte une ALTERNATIVE (tuple): au moins une
    # des marques doit correspondre.
    'heic': [(4, b'ftyp'), (8, HEIC_BRANDS)],
}

# Map an extension to the magic signature group that validates it.
EXTENSION_TO_MAGIC = {
    'pdf': 'pdf',
    'png': 'png',
    'jpg': 'jpg',
    'jpeg': 'jpg',
    'webp': 'webp',
    'heic': 'heic',
    'heif': 'heic',
}


def _read_header(file, size=16):
    """Read the first bytes of an uploaded file without consuming it."""
    pos = None
    try:
        pos = file.tell()
    except (AttributeError, OSError):
        pos = None

    try:
        file.seek(0)
    except (AttributeError, OSError):
        pass

    header = file.read(size) or b''

    # Restore the original cursor position so downstream reads still work.
    try:
        file.seek(pos if pos is not None else 0)
    except (AttributeError, OSError):
        pass

    if isinstance(header, str):
        header = header.encode('latin-1', errors='ignore')
    return header


def _matches_signature(header, signatures):
    for offset, sig in signatures:
        if isinstance(sig, (tuple, list)):
            # Alternative: au moins une des signatures candidates doit
            # correspondre (utilise pour les marques HEIC/HEIF).
            width = max(len(s) for s in sig)
            chunk = header[offset:offset + width]
            if not any(chunk[:len(s)] == s for s in sig):
                return False
        elif header[offset:offset + len(sig)] != sig:
            return False
    return True


def is_heic_header(header: bytes) -> bool:
    """True si l'en-tete correspond a un conteneur HEIC/HEIF."""
    return _matches_signature(header, MAGIC_SIGNATURES['heic'])


def sniff_kind(header: bytes) -> str:
    """Return the extraction path from a file's leading magic bytes.

    Documents are stored on Cloudinary with a neutral extension (a real .pdf
    public_id is blocked from delivery), so the processor can no longer trust
    the file suffix and must decide the extraction path from CONTENT. GIF is
    accepted here (vision-readable) even though it is not in the upload
    allowlist. Anything unrecognised defaults to 'pdf', whose pipeline itself
    falls back to vision.
    """
    if header[:5] == b'%PDF-':
        return 'pdf'
    if header[:4] == b'PK\x03\x04':
        return 'docx'
    if header[:8] == WORD_OLE_HEADER:
        return 'doc'
    if (header[:8] == b'\x89PNG\r\n\x1a\n'
            or header[:3] == b'\xff\xd8\xff'
            or (header[:4] == b'RIFF' and header[8:12] == b'WEBP')
            or header[:6] in (b'GIF87a', b'GIF89a')
            or is_heic_header(header)):
        return 'image'
    return 'pdf'


def validate_upload_file(file):
    """Validate an uploaded file by size, extension and magic bytes.

    Raises rest_framework.exceptions.ValidationError on any violation.
    Returns the canonical type string (e.g. 'pdf', 'png') on success.
    """
    if file is None:
        raise ValidationError("Aucun fichier fourni.")

    # --- Size ---
    size = getattr(file, 'size', None)
    if size is None:
        try:
            pos = file.tell()
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(pos)
        except (AttributeError, OSError):
            size = None
    if size is not None and size > MAX_UPLOAD_SIZE:
        raise ValidationError(
            f"Fichier trop volumineux ({size} octets). "
            f"Maximum autorisé : {MAX_UPLOAD_SIZE} octets."
        )
    if size == 0:
        raise ValidationError("Le fichier est vide.")

    # --- Extension allowlist ---
    name = getattr(file, 'name', '') or ''
    ext = os.path.splitext(name)[1].lower().lstrip('.')
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            "Type de fichier non autorisé. "
            f"Extensions acceptées : {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    # --- Magic bytes ---
    header = _read_header(file)
    if ext == 'doc':
        if not header.startswith(WORD_OLE_HEADER):
            raise ValidationError("Le contenu du fichier ne correspond pas à son extension.")
        return 'doc'
    if ext == 'docx':
        if not header.startswith(b'PK\x03\x04'):
            raise ValidationError("Le contenu du fichier ne correspond pas à son extension.")
        position = file.tell()
        try:
            file.seek(0)
            with ZipFile(file) as archive:
                infos = archive.infolist()
                noms = {info.filename for info in infos}
                if ('[Content_Types].xml' not in noms or 'word/document.xml' not in noms
                        or len(infos) > MAX_WORD_ARCHIVE_ENTRIES
                        or sum(info.file_size for info in infos) > MAX_WORD_ARCHIVE_BYTES
                        or any(info.flag_bits & 1 for info in infos)):
                    raise ValidationError("Document Word invalide ou trop volumineux.")
        except (BadZipFile, OSError, ValueError):
            raise ValidationError("Document Word invalide.") from None
        finally:
            file.seek(position)
        return 'docx'
    magic_key = EXTENSION_TO_MAGIC[ext]
    if not _matches_signature(header, MAGIC_SIGNATURES[magic_key]):
        raise ValidationError(
            "Le contenu du fichier ne correspond pas à son extension."
        )

    return magic_key
