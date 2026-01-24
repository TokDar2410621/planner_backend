"""
Document processing service using Google Gemini Vision.
"""
import json
import logging
import re
from io import BytesIO
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image
from django.conf import settings

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from datetime import time as dt_time
from core.models import UploadedDocument, RecurringBlock

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
            "name": "Nom COMPLET (inclure le code si présent, ex: INF1120 - Programmation)",
            "day": "lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche",
            "start_time": "HH:MM",
            "end_time": "HH:MM",
            "location": "Salle/Local si visible",
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
- Vérifie que chaque jour de la semaine a été parcouru"""

    EXTRACTION_PROMPTS = {
        'course_schedule': UNIFIED_EXTRACTION_PROMPT,
        'work_schedule': UNIFIED_EXTRACTION_PROMPT,
        'other': UNIFIED_EXTRACTION_PROMPT,
    }

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
        """Initialize the document processor."""
        if GEMINI_AVAILABLE and settings.GEMINI_API_KEY:
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self.model_name = 'gemini-2.0-flash'
        else:
            self.client = None
            self.model_name = None
            logger.warning("Gemini API not configured. Document processing will be limited.")

    def process_document(self, document: UploadedDocument) -> dict:
        """
        Process an uploaded document and extract schedule data.

        Args:
            document: The UploadedDocument instance to process

        Returns:
            dict: Extracted data from the document
        """
        try:
            file_path = document.file.path
            file_ext = file_path.lower().split('.')[-1]

            # Extract content based on file type
            if file_ext == 'pdf':
                content = self._extract_pdf_content(file_path)
            elif file_ext in ['png', 'jpg', 'jpeg', 'webp', 'gif']:
                content = self._load_image(file_path)
            else:
                raise ValueError(f"Type de fichier non supporté: {file_ext}")

            # Get appropriate prompt
            prompt = self._get_extraction_prompt(document.document_type)

            # Process with Gemini
            if self.client is None:
                raise RuntimeError("Service Gemini non configuré")

            if isinstance(content, list):
                # Multiple images from PDF - convert PIL images to parts
                # Add context about multiple pages
                multi_page_prompt = f"""Ce document contient {len(content)} page(s). Analyse TOUTES les pages et combine les informations.

{prompt}

IMPORTANT: Si l'emploi du temps s'étend sur plusieurs pages, fusionne toutes les données dans une seule réponse JSON."""

                parts = [multi_page_prompt]
                for idx, img in enumerate(content):
                    img_bytes = BytesIO()
                    # Use higher quality JPEG for large images to reduce size while keeping quality
                    if img.width > 2000 or img.height > 2000:
                        img.save(img_bytes, format='JPEG', quality=95)
                        mime_type = 'image/jpeg'
                    else:
                        img.save(img_bytes, format='PNG')
                        mime_type = 'image/png'
                    logger.info(f"Sending page {idx + 1}: {img.width}x{img.height}px as {mime_type}")
                    parts.append(types.Part.from_bytes(data=img_bytes.getvalue(), mime_type=mime_type))

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=parts
                )
            elif isinstance(content, Image.Image):
                # Single image - convert to bytes
                img_bytes = BytesIO()
                content.save(img_bytes, format='PNG')
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=img_bytes.getvalue(), mime_type='image/png')
                    ]
                )
            else:
                # Text content
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=f"{prompt}\n\nContenu du document:\n{content}"
                )

            # Parse response
            logger.info(f"Raw Gemini response length: {len(response.text)} chars")
            logger.debug(f"Raw response: {response.text[:500]}...")

            extracted_data = self._parse_response(response.text)

            # Log extraction results
            courses_count = len(extracted_data.get('courses', []))
            shifts_count = len(extracted_data.get('shifts', []))
            events_count = len(extracted_data.get('events', []))
            logger.info(f"Extracted: {courses_count} courses, {shifts_count} shifts, {events_count} events")

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
                    # Handle None values from JSON
                    location = course.get('location') or ''

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
                    # Handle None values from JSON
                    role = shift.get('role') or ''
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
                    # Handle None values from JSON
                    title = event.get('title') or 'Événement'

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
