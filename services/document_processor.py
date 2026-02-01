"""
Document processing service using pdfplumber text extraction + LLM structuring.
Fallback to Gemini Vision for scanned PDFs.
Includes caching to avoid re-processing identical documents.
"""
import base64
import hashlib
import json
import logging
import re
from io import BytesIO
from typing import Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image
from django.conf import settings

# Import our new PDF text extractor
from services.pdf_extractor import PDFTextExtractor

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from datetime import time as dt_time
from core.models import UploadedDocument, RecurringBlock
from utils.helpers import retry_with_backoff, run_in_background

logger = logging.getLogger(__name__)


def parse_time_string(time_str: str) -> Optional[dt_time]:
    """
    Parse a time string like "08:00" or "14:30" to a time object.

    Args:
        time_str: Time string in HH:MM format

    Returns:
        time object or None if parsing fails
    """
    if not time_str:
        return None
    try:
        parts = time_str.strip().split(':')
        if len(parts) >= 2:
            hour = int(parts[0])
            minute = int(parts[1])
            # Normalize to quarter hours (00, 15, 30, 45)
            minute = round(minute / 15) * 15
            if minute == 60:
                minute = 0
                hour = (hour + 1) % 24
            return dt_time(hour, minute)
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not parse time string '{time_str}': {e}")
    return None


def is_night_shift(start_time: dt_time, end_time: dt_time) -> bool:
    """
    Determine if a work shift is a night shift.

    A night shift is when:
    - Work starts after 20:00 OR
    - Work ends before 08:00 (next day) OR
    - End time is before start time (crosses midnight)

    Args:
        start_time: Shift start time
        end_time: Shift end time

    Returns:
        bool: True if it's a night shift
    """
    if start_time is None or end_time is None:
        return False

    # Crosses midnight (end time < start time)
    if end_time < start_time:
        return True

    # Starts after 20:00
    if start_time.hour >= 20:
        return True

    # Ends very early (before 8:00) but doesn't cross midnight
    # This would be unusual, but let's handle it
    if end_time.hour < 8 and start_time.hour < 8:
        return True

    return False


class DocumentProcessor:
    """
    Process uploaded documents (PDF, images) using Gemini Vision.

    Extracts schedule information from course schedules and work schedules.
    """

    # Unified extraction prompt that extracts all types of schedule data
    UNIFIED_EXTRACTION_PROMPT = """Tu es un expert en extraction de données d'emploi du temps. Analyse cette image avec une EXTRÊME PRÉCISION.

=== ÉTAPE 1: IDENTIFIER LA STRUCTURE ===
Regarde attentivement l'image et identifie:
- Est-ce une GRILLE (tableau avec jours en colonnes et heures en lignes)?
- Est-ce une LISTE (événements listés un par un)?
- Où sont les JOURS? (en haut? sur le côté?)
- Où sont les HEURES? (à gauche? dans chaque case?)

=== ÉTAPE 2: LIRE CHAQUE CELLULE/BLOC ===
Pour CHAQUE bloc coloré ou case remplie:
1. Note le JOUR de la colonne
2. Note l'HEURE DE DÉBUT (regarder l'axe des heures à gauche)
3. Note l'HEURE DE FIN (fin du bloc sur l'axe des heures)
4. Note le CONTENU (nom du cours, code, salle, prof)

=== FORMATS DE PLANNING COURANTS ===

FORMAT GRILLE UNIVERSITAIRE:
- Colonnes = Jours (Lun, Mar, Mer, Jeu, Ven)
- Lignes = Créneaux horaires (8h, 9h, 10h...)
- Cases = Cours avec code + nom + salle

FORMAT PLANNING TRAVAIL:
- Peut être hebdomadaire ou mensuel
- Cherche les horaires de shift (ex: "9h-17h", "14h-22h")

FORMAT LISTE/AGENDA:
- Événements listés avec date et heure

=== CONVERSION DES JOURS ===
TOUJOURS convertir en français minuscule:
- Lun/L/Mon/Monday/Lu → "lundi"
- Mar/Ma/Tue/Tuesday → "mardi"
- Mer/Me/Wed/Wednesday → "mercredi"
- Jeu/Je/Thu/Thursday → "jeudi"
- Ven/Ve/Fri/Friday → "vendredi"
- Sam/Sa/Sat/Saturday → "samedi"
- Dim/Di/Sun/Sunday → "dimanche"

=== CONVERSION DES HEURES ===
TOUJOURS format HH:MM (24h):
- "8h" ou "8:00" → "08:00"
- "8h30" ou "8:30" → "08:30"
- "14h" → "14:00"
- "2pm" → "14:00"
- "midi" → "12:00"

=== FORMAT JSON REQUIS ===
{
    "detected_type": "course_schedule|work_schedule|mixed",
    "schedule_structure": "grid|list|other",
    "courses": [
        {
            "name": "Nom LISIBLE du cours (si le code est illisible, utilise juste le nom simple ex: 'Programmation', 'Mathématiques', 'Projets')",
            "day": "lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche",
            "start_time": "HH:MM",
            "end_time": "HH:MM",
            "location": "Salle/Local (laisser null si pas visible ou illisible)",
            "professor": "Prof si visible",
            "course_code": "Code du cours si présent (ex: INF1120)"
        }
    ],
    "shifts": [
        {
            "day": "lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche",
            "start_time": "HH:MM",
            "end_time": "HH:MM",
            "is_night_shift": false,
            "role": "Poste si mentionné",
            "notes": ""
        }
    ],
    "events": []
}

=== RÈGLES ABSOLUES ===
1. UN BLOC = UNE ENTRÉE (si "Math" apparaît lundi ET mercredi = 2 entrées séparées)
2. Si un bloc s'étend sur plusieurs heures, calcule la durée totale (ex: bloc de 8h à 10h = start 08:00, end 10:00)
3. Si l'heure de fin n'est pas visible, estime selon la taille du bloc ou ajoute 1h30 par défaut
4. Extrait TOUS les blocs visibles, même partiellement lisibles
5. Arrays vides [] si aucune donnée de ce type
6. UNIQUEMENT le JSON, aucun texte avant/après
7. is_night_shift = true SEULEMENT si shift après 20h ou finit après minuit

=== IMPORTANT ===
- Lis TRÈS attentivement chaque case de la grille
- Ne rate aucun cours/bloc
- Vérifie que chaque jour de la semaine a été parcouru
- Pour le champ "name": préfère un nom COURT et LISIBLE. Ex: "Programmation" plutôt que "420KWAJQ-00001 Programmation I"
- Si le code est incompréhensible (ex: 420KWAJQ), mets-le dans "course_code" et utilise juste le nom du cours dans "name"
- Si location est illisible ou absent, mets null (pas "None" ou "-")"""

    EXTRACTION_PROMPTS = {
        'course_schedule': UNIFIED_EXTRACTION_PROMPT,
        'work_schedule': UNIFIED_EXTRACTION_PROMPT,
        'other': UNIFIED_EXTRACTION_PROMPT,
    }

    # New prompt for TEXT-BASED extraction (pdfplumber output)
    TEXT_STRUCTURING_PROMPT = """Tu es un expert en extraction de données d'emploi du temps.
Je te donne le TEXTE BRUT extrait d'un PDF d'horaire. Analyse-le et structure les données en JSON.

=== RÈGLES D'ANALYSE DU TEXTE ===

1. STRUCTURE GRILLE (tableau horaire):
   - Les jours sont généralement en colonnes: Lundi | Mardi | Mercredi | Jeudi | Vendredi
   - Les heures sont à gauche: 8h, 9h, 10h, etc.
   - Chaque cellule contient: [Code cours] [Nom] [Salle]

2. LECTURE DES HORAIRES:
   - "8h à 9h" ou "8h-9h" = start_time: "08:00", end_time: "09:00"
   - Si un bloc couvre "8h à 10h" = 2 heures de cours
   - Les créneaux qui se répètent (ex: même cours 8h-9h ET 9h-10h) = UN SEUL bloc de 8h à 10h

3. ASSOCIATION JOUR/HORAIRE:
   - Dans une grille, le contenu sous "Mardi" appartient au mardi
   - Cherche la légende/tableau des cours pour les noms complets et profs

=== CONVERSION DES JOURS ===
- Lun/L/Mon → "lundi"
- Mar/Ma/Tue → "mardi"
- Mer/Me/Wed → "mercredi"
- Jeu/Je/Thu → "jeudi"
- Ven/Ve/Fri → "vendredi"
- Sam/Sa/Sat → "samedi"
- Dim/Di/Sun → "dimanche"

=== FORMAT JSON REQUIS ===
{{
    "detected_type": "course_schedule|work_schedule|mixed",
    "courses": [
        {{
            "name": "Nom du cours (ex: Programmation avancée)",
            "day": "lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche",
            "start_time": "HH:MM",
            "end_time": "HH:MM",
            "location": "Salle (ex: 316.1)",
            "professor": "Nom du prof si trouvé",
            "course_code": "Code du cours (ex: 420KWAJQ)"
        }}
    ],
    "shifts": [
        {{
            "day": "lundi|mardi|...",
            "start_time": "HH:MM",
            "end_time": "HH:MM",
            "role": "Poste si mentionné"
        }}
    ],
    "events": []
}}

=== RÈGLES ABSOLUES ===
1. UN CRÉNEAU PAR JOUR = UNE ENTRÉE (même cours lundi et mardi = 2 entrées)
2. FUSIONNER les créneaux consécutifs du même cours le même jour (8h-9h + 9h-10h = 8h-10h)
3. Extraire TOUS les cours/shifts visibles
4. Si "Cours inclus à l'horaire" ou légende présente, utiliser les VRAIS NOMS des cours
5. UNIQUEMENT le JSON, aucun texte avant/après

=== TEXTE EXTRAIT DU PDF ===
{text}

Analyse ce texte et retourne le JSON structuré:"""

    # Extended day mapping with various abbreviations and formats
    DAY_MAPPING = {
        # French full names
        'lundi': 0, 'mardi': 1, 'mercredi': 2, 'jeudi': 3,
        'vendredi': 4, 'samedi': 5, 'dimanche': 6,
        # French abbreviations
        'lun': 0, 'lun.': 0, 'lu': 0,
        'mar': 1, 'mar.': 1, 'ma': 1,
        'mer': 2, 'mer.': 2,
        'jeu': 3, 'jeu.': 3, 'je': 3,
        'ven': 4, 'ven.': 4, 've': 4,
        'sam': 5, 'sam.': 5, 'sa': 5,
        'dim': 6, 'dim.': 6, 'di': 6,
        # Single letters (French) - careful with 'm' which could be mardi or mercredi
        'l': 0, 'j': 3, 'v': 4, 's': 5, 'd': 6,
        # English full names
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6,
        # English abbreviations
        'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6,
    }

    def __init__(self):
        """Initialize the document processor with pdfplumber + Gemini."""
        # PDF Text Extractor (pdfplumber + OCR fallback)
        self.pdf_extractor = PDFTextExtractor()

        # Primary: Gemini (for text structuring + vision fallback)
        if GEMINI_AVAILABLE and settings.GEMINI_API_KEY:
            self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self.gemini_model = 'gemini-2.5-flash'
        else:
            self.gemini_client = None
            self.gemini_model = None

        # Fallback: Hugging Face
        if HF_AVAILABLE and settings.HF_API_KEY:
            self.hf_client = InferenceClient(token=settings.HF_API_KEY)
            self.hf_model = settings.HF_MODEL
        else:
            self.hf_client = None
            self.hf_model = None

        # Legacy alias
        self.client = self.gemini_client
        self.model_name = self.gemini_model

        if not self.gemini_client and not self.hf_client:
            logger.warning("No AI API configured. Document processing will be limited.")

    def _compute_file_hash(self, file_path: str) -> str:
        """
        Compute SHA-256 hash of file content.

        Args:
            file_path: Path to the file

        Returns:
            str: Hex digest of the hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _check_cache(self, user, content_hash: str) -> Optional[dict]:
        """
        Check if we've already processed a document with the same content.

        Args:
            user: The user who owns the document
            content_hash: SHA-256 hash of the file content

        Returns:
            dict or None: Cached extracted_data if found, None otherwise
        """
        if not content_hash:
            return None

        # Look for a previously processed document with same hash for this user
        cached_doc = UploadedDocument.objects.filter(
            user=user,
            content_hash=content_hash,
            processed=True,
            processing_error__isnull=True
        ).exclude(extracted_data={}).first()

        if cached_doc:
            logger.info(f"Cache HIT: Found previously processed document with hash {content_hash[:16]}...")
            return cached_doc.extracted_data

        logger.info(f"Cache MISS: No cached document found for hash {content_hash[:16]}...")
        return None

    @retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=30.0)
    def _call_gemini(self, contents):
        """
        Call Gemini API with retry logic for rate limits.

        Args:
            contents: Content to send to Gemini (prompt + images)

        Returns:
            Response from Gemini
        """
        return self.gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=contents
        )

    @retry_with_backoff(max_retries=2, base_delay=3.0, max_delay=30.0)
    def _call_hf_vision(self, prompt: str, images: list) -> str:
        """
        Call Hugging Face vision model as fallback.

        Args:
            prompt: Text prompt for the model
            images: List of PIL Image objects

        Returns:
            str: Model response text
        """
        if not self.hf_client:
            raise RuntimeError("Hugging Face client not configured")

        # Prepare messages with images
        content = [{"type": "text", "text": prompt}]

        for img in images:
            # Convert image to base64
            img_bytes = BytesIO()
            img.save(img_bytes, format='JPEG', quality=85)
            img_b64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })

        messages = [{"role": "user", "content": content}]

        response = self.hf_client.chat_completion(
            model=self.hf_model,
            messages=messages,
            max_tokens=4096,
        )

        return response.choices[0].message.content

    def process_document_async(self, document_id: int) -> None:
        """
        Process a document asynchronously in background.

        Args:
            document_id: ID of the UploadedDocument to process
        """
        run_in_background(self._process_document_by_id, document_id)

    def _process_document_by_id(self, document_id: int) -> dict:
        """
        Process a document by its ID (for background processing).

        Args:
            document_id: ID of the document

        Returns:
            dict: Extracted data
        """
        from django.db import connection
        # Close old connection to avoid issues in new thread
        connection.close()

        document = UploadedDocument.objects.get(id=document_id)
        return self.process_document(document)

    def process_document(self, document: UploadedDocument) -> dict:
        """
        Process an uploaded document and extract schedule data.

        NEW PIPELINE:
        0. Check cache - if same file was processed before, reuse the result
        1. PDF → pdfplumber text extraction (preserves structure)
        2. If text found → send TEXT to Gemini for structuring
        3. If no text (scanned PDF) → fallback to Gemini Vision

        Args:
            document: The UploadedDocument instance to process

        Returns:
            dict: Extracted data from the document
        """
        try:
            file_path = document.file.path
            file_ext = file_path.lower().split('.')[-1]

            # Step 0: Compute file hash and check cache
            content_hash = self._compute_file_hash(file_path)
            document.content_hash = content_hash
            document.file_name = document.file.name.split('/')[-1]
            document.save(update_fields=['content_hash', 'file_name'])

            cached_data = self._check_cache(document.user, content_hash)
            if cached_data:
                # Use cached result - much faster!
                extracted_data = cached_data.copy()
                extracted_data['from_cache'] = True
                logger.info(f"Using cached extraction for document {document.id}")

                # Update document with cached data
                document.extracted_data = extracted_data
                document.processed = True
                document.processing_error = None
                document.save()

                # Still create recurring blocks from cached data
                self._create_recurring_blocks(document, extracted_data)

                return extracted_data

            response_text = None
            extraction_method = None

            if file_ext == 'pdf':
                # NEW: Try text extraction first (pdfplumber)
                response_text, extraction_method = self._process_pdf_smart(file_path, document.document_type)

            elif file_ext in ['png', 'jpg', 'jpeg', 'webp', 'gif']:
                # Images always use vision
                response_text, extraction_method = self._process_image(file_path, document.document_type)

            else:
                raise ValueError(f"Type de fichier non supporté: {file_ext}")

            if response_text is None:
                raise RuntimeError("Aucun service AI disponible")

            # Parse response
            logger.debug(f"Raw response ({extraction_method}): {response_text[:500]}...")
            extracted_data = self._parse_response(response_text)

            # Add extraction method to data
            extracted_data['extraction_method'] = extraction_method

            # Log extraction results
            courses_count = len(extracted_data.get('courses', []))
            shifts_count = len(extracted_data.get('shifts', []))
            events_count = len(extracted_data.get('events', []))
            logger.info(f"Extracted ({extraction_method}): {courses_count} courses, {shifts_count} shifts, {events_count} events")

            if courses_count == 0 and shifts_count == 0:
                logger.warning(f"No schedule data extracted from document {document.id}")
                if 'parse_error' in extracted_data:
                    logger.error(f"Parse error in response: {extracted_data.get('raw_response', '')[:300]}")

            # Update document
            document.extracted_data = extracted_data
            document.processed = True
            document.processing_error = None
            document.save()

            # Create recurring blocks from extracted data
            self._create_recurring_blocks(document, extracted_data)

            return extracted_data

        except Exception as e:
            logger.error(f"Error processing document {document.id}: {str(e)}")
            document.processing_error = str(e)
            document.processed = False
            document.save()
            raise

    def _process_pdf_smart(self, file_path: str, doc_type: str) -> Tuple[Optional[str], str]:
        """
        Smart PDF processing: text extraction first, vision fallback.

        Args:
            file_path: Path to PDF file
            doc_type: Document type for prompt selection

        Returns:
            Tuple of (response_text, extraction_method)
        """
        # Step 1: Try pdfplumber text extraction
        logger.info(f"Extracting text from PDF with pdfplumber...")
        extraction_result = self.pdf_extractor.extract(file_path)

        extracted_text = extraction_result.get('text', '')
        tables = extraction_result.get('tables', [])
        method = extraction_result.get('method', 'unknown')

        logger.info(f"PDF extraction method: {method}, text length: {len(extracted_text)}, tables: {len(tables)}")

        # Step 2: Check if we got meaningful text
        # Minimum threshold: at least 100 chars and contains some schedule-related words
        has_meaningful_text = (
            len(extracted_text.strip()) > 100 and
            any(word in extracted_text.lower() for word in ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'monday', 'tuesday', 'cours', 'horaire', 'schedule', '8h', '9h', '10h', ':00'])
        )

        if has_meaningful_text:
            # Use TEXT-BASED extraction (much more accurate!)
            logger.info("Using TEXT-BASED extraction (pdfplumber + LLM structuring)")
            return self._structure_text_with_llm(extracted_text, tables), f"text_{method}"
        else:
            # Fallback to VISION-BASED extraction
            logger.info("Text extraction insufficient, falling back to VISION-BASED extraction")
            return self._process_pdf_vision(file_path, doc_type), "vision"

    def _structure_text_with_llm(self, text: str, tables: list = None) -> Optional[str]:
        """
        Send extracted text to LLM for structuring into JSON.

        Args:
            text: Extracted text from PDF
            tables: Extracted tables (optional)

        Returns:
            LLM response with structured JSON
        """
        # Build the prompt with extracted text
        prompt = self.TEXT_STRUCTURING_PROMPT.format(text=text[:15000])  # Limit to 15k chars

        # Add tables if available
        if tables:
            table_text = "\n\n=== TABLEAUX EXTRAITS ===\n"
            for i, table in enumerate(tables[:5]):  # Max 5 tables
                table_text += f"\nTableau {i+1}:\n"
                for row in table[:20]:  # Max 20 rows per table
                    table_text += " | ".join(str(cell) for cell in row) + "\n"
            prompt += table_text

        if self.gemini_client:
            try:
                response = self._call_gemini(prompt)
                return response.text
            except Exception as e:
                logger.error(f"Gemini text structuring failed: {e}")
                raise

        raise RuntimeError("No LLM available for text structuring")

    def _process_pdf_vision(self, file_path: str, doc_type: str) -> Optional[str]:
        """
        Process PDF using vision (for scanned PDFs).

        Args:
            file_path: Path to PDF file
            doc_type: Document type

        Returns:
            LLM response text
        """
        # Extract PDF as images
        images = self._extract_pdf_content(file_path)

        # Get vision prompt
        prompt = self._get_extraction_prompt(doc_type)

        if len(images) > 1:
            full_prompt = f"""Ce document contient {len(images)} page(s). Analyse TOUTES les pages et combine les informations.

{prompt}

IMPORTANT: Si l'emploi du temps s'étend sur plusieurs pages, fusionne toutes les données dans une seule réponse JSON."""
        else:
            full_prompt = prompt

        # Use Gemini Vision
        if self.gemini_client:
            try:
                parts = [full_prompt]
                for idx, img in enumerate(images):
                    img_bytes = BytesIO()
                    if img.width > 2000 or img.height > 2000:
                        img.save(img_bytes, format='JPEG', quality=95)
                        mime_type = 'image/jpeg'
                    else:
                        img.save(img_bytes, format='PNG')
                        mime_type = 'image/png'
                    logger.info(f"Vision: page {idx + 1}: {img.width}x{img.height}px as {mime_type}")
                    parts.append(types.Part.from_bytes(data=img_bytes.getvalue(), mime_type=mime_type))
                response = self._call_gemini(parts)
                return response.text
            except Exception as e:
                logger.warning(f"Gemini vision failed: {e}")
                # Try HF fallback
                if self.hf_client:
                    return self._call_hf_vision(full_prompt, images)
                raise

        # HF fallback
        if self.hf_client:
            return self._call_hf_vision(full_prompt, images)

        return None

    def _process_image(self, file_path: str, doc_type: str) -> Tuple[Optional[str], str]:
        """
        Process image file using vision.

        Args:
            file_path: Path to image file
            doc_type: Document type

        Returns:
            Tuple of (response_text, extraction_method)
        """
        image = self._load_image(file_path)
        prompt = self._get_extraction_prompt(doc_type)

        if self.gemini_client:
            try:
                img_bytes = BytesIO()
                image.save(img_bytes, format='PNG')
                response = self._call_gemini([
                    prompt,
                    types.Part.from_bytes(data=img_bytes.getvalue(), mime_type='image/png')
                ])
                return response.text, "vision_gemini"
            except Exception as e:
                logger.warning(f"Gemini vision failed: {e}")
                if self.hf_client:
                    return self._call_hf_vision(prompt, [image]), "vision_hf"
                raise

        if self.hf_client:
            return self._call_hf_vision(prompt, [image]), "vision_hf"

        return None, "none"

    def _extract_pdf_content(self, file_path: str) -> list:
        """
        Extract content from PDF file as high-resolution images.

        Args:
            file_path: Path to the PDF file

        Returns:
            list: List of PIL Image objects (one per page)
        """
        images = []
        doc = fitz.open(file_path)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Render page to image with HIGH resolution for better OCR
            # 3x zoom = ~216 DPI, good balance between quality and file size
            mat = fitz.Matrix(3, 3)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Log image dimensions for debugging
            logger.info(f"PDF page {page_num + 1}: {img.width}x{img.height}px")

            images.append(img)

        doc.close()
        logger.info(f"Extracted {len(images)} page(s) from PDF")
        return images

    def _load_image(self, file_path: str) -> Image.Image:
        """
        Load an image file.

        Args:
            file_path: Path to the image file

        Returns:
            PIL.Image: The loaded image
        """
        return Image.open(file_path)

    def _get_extraction_prompt(self, doc_type: str) -> str:
        """
        Get the appropriate extraction prompt for document type.

        Args:
            doc_type: Type of document

        Returns:
            str: The extraction prompt
        """
        return self.EXTRACTION_PROMPTS.get(doc_type, self.EXTRACTION_PROMPTS['other'])

    def _parse_response(self, response_text: str) -> dict:
        """
        Parse the Gemini response to extract JSON data.

        Args:
            response_text: Raw response text from Gemini

        Returns:
            dict: Parsed JSON data
        """
        # Try to extract JSON from response
        # Sometimes the model adds markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning(f"Could not find JSON in response: {response_text[:200]}")
                return {"raw_response": response_text, "parse_error": True}

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            return {"raw_response": response_text, "parse_error": True}

    def _create_recurring_blocks(self, document: UploadedDocument, data: dict) -> list:
        """
        Create RecurringBlock instances from ALL extracted data types.

        Args:
            document: The source document
            data: Extracted data dictionary

        Returns:
            list: List of created RecurringBlock instances
        """
        created_blocks = []
        user = document.user

        # Check if blocks already exist for this document (prevent duplicates)
        existing_blocks = RecurringBlock.objects.filter(source_document=document)
        if existing_blocks.exists():
            logger.info(f"Blocks already exist for document {document.id}, skipping creation")
            return list(existing_blocks)

        # Update document type based on detected type
        if 'detected_type' in data:
            detected = data['detected_type']
            if detected in ['course_schedule', 'work_schedule', 'mixed']:
                document.document_type = detected
                document.save(update_fields=['document_type'])

        # Create blocks from courses
        if 'courses' in data and data['courses']:
            for course in data['courses']:
                day_str = course.get('day', '')
                if day_str:
                    day = self.DAY_MAPPING.get(day_str.lower().strip())
                else:
                    day = None

                if day is None:
                    logger.warning(f"Invalid day '{day_str}' for course: {course.get('name', 'Unknown')}")
                    continue

                # Parse time strings to time objects
                start_time = parse_time_string(course.get('start_time', ''))
                end_time = parse_time_string(course.get('end_time', ''))

                if start_time is None:
                    start_time = dt_time(9, 0)  # Default 09:00
                    logger.warning(f"Using default start time for course: {course.get('name', 'Unknown')}")
                if end_time is None:
                    end_time = dt_time(10, 0)  # Default 10:00
                    logger.warning(f"Using default end time for course: {course.get('name', 'Unknown')}")

                try:
                    # Handle None values from JSON (including string "None" from Gemini)
                    location = course.get('location') or ''
                    if location.lower() in ('none', 'n/a', 'null', '-'):
                        location = ''

                    block = RecurringBlock.objects.create(
                        user=user,
                        title=course.get('name') or 'Cours',
                        block_type='course',
                        day_of_week=day,
                        start_time=start_time,
                        end_time=end_time,
                        location=location,
                        source_document=document,
                    )
                    created_blocks.append(block)
                    logger.info(f"Created course block: {block.title} on day {day}")
                except Exception as e:
                    logger.error(f"Error creating course block '{course.get('name', 'Unknown')}': {e}")

        # Create blocks from work shifts
        if 'shifts' in data and data['shifts']:
            for shift in data['shifts']:
                day_str = shift.get('day', '')
                if day_str:
                    day = self.DAY_MAPPING.get(day_str.lower().strip())
                else:
                    day = None

                if day is None:
                    logger.warning(f"Invalid day '{day_str}' for shift")
                    continue

                # Parse time strings to time objects
                start_time = parse_time_string(shift.get('start_time', ''))
                end_time = parse_time_string(shift.get('end_time', ''))

                if start_time is None:
                    start_time = dt_time(9, 0)
                if end_time is None:
                    end_time = dt_time(17, 0)

                try:
                    # Handle None values from JSON (including string "None" from Gemini)
                    role = shift.get('role') or ''
                    if role.lower() in ('none', 'n/a', 'null', '-'):
                        role = ''
                    title = f"Travail - {role}" if role else 'Travail'

                    # Auto-detect night shift based on times (more reliable than Gemini)
                    is_night = is_night_shift(start_time, end_time)
                    if is_night:
                        logger.info(f"Auto-detected night shift: {start_time}-{end_time}")

                    block = RecurringBlock.objects.create(
                        user=user,
                        title=title,
                        block_type='work',
                        day_of_week=day,
                        start_time=start_time,
                        end_time=end_time,
                        is_night_shift=is_night,
                        source_document=document,
                    )
                    created_blocks.append(block)
                    logger.info(f"Created work block: {title} on day {day}")
                except Exception as e:
                    logger.error(f"Error creating work block: {e}")

        # Create blocks from events (as 'other' type)
        if 'events' in data and data['events']:
            for event in data['events']:
                day_str = event.get('day', '')
                if day_str:
                    day = self.DAY_MAPPING.get(day_str.lower().strip())
                else:
                    day = None

                if day is None:
                    logger.warning(f"Invalid day '{day_str}' for event: {event.get('title', 'Unknown')}")
                    continue

                # Parse time strings to time objects
                start_time = parse_time_string(event.get('start_time', ''))
                end_time = parse_time_string(event.get('end_time', ''))

                if start_time is None:
                    start_time = dt_time(9, 0)
                if end_time is None:
                    end_time = dt_time(10, 0)

                try:
                    # Handle None values from JSON (including string "None" from Gemini)
                    title = event.get('title') or ''
                    if not title or title.lower() in ('none', 'n/a', 'null', '-'):
                        title = 'Événement'

                    block = RecurringBlock.objects.create(
                        user=user,
                        title=title,
                        block_type='other',
                        day_of_week=day,
                        start_time=start_time,
                        end_time=end_time,
                        source_document=document,
                    )
                    created_blocks.append(block)
                    logger.info(f"Created event block: {block.title} on day {day}")
                except Exception as e:
                    logger.error(f"Error creating event block '{event.get('title', 'Unknown')}': {e}")

        return created_blocks
