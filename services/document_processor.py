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
            return dt_time(hour, minute)
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not parse time string '{time_str}': {e}")
    return None


class DocumentProcessor:
    """
    Process uploaded documents (PDF, images) using Gemini Vision.

    Extracts schedule information from course schedules and work schedules.
    """

    # Unified extraction prompt that extracts all types of schedule data
    UNIFIED_EXTRACTION_PROMPT = """Tu es un expert en extraction de données d'emploi du temps. Analyse cette image TRÈS ATTENTIVEMENT et extrait TOUTES les informations au format JSON.

TYPES DE DOCUMENTS À DÉTECTER:
- Emploi du temps de COURS (université, école, CÉGEP, UQAC, etc.)
- Planning de TRAVAIL (horaires de travail, shifts, jobs)
- Calendrier mixte (cours + travail)

COMMENT LIRE L'IMAGE:
1. Identifie la structure: grille avec jours en colonnes/lignes, ou liste
2. Repère les HEURES sur les axes (souvent à gauche ou en haut)
3. Repère les JOURS (Lun/Mon, Mar/Tue, Mer/Wed, Jeu/Thu, Ven/Fri, Sam/Sat, Dim/Sun)
4. Extrait CHAQUE bloc/case avec son jour, heure début, heure fin, et titre

CONVERSION DES ABRÉVIATIONS DE JOURS:
- "Lun", "L", "Mon", "Monday" → "lundi"
- "Mar", "M", "Tue", "Tuesday" → "mardi"
- "Mer", "Me", "Wed", "Wednesday" → "mercredi"
- "Jeu", "J", "Thu", "Thursday" → "jeudi"
- "Ven", "V", "Fri", "Friday" → "vendredi"
- "Sam", "S", "Sat", "Saturday" → "samedi"
- "Dim", "D", "Sun", "Sunday" → "dimanche"

Format de réponse JSON:
{
    "detected_type": "course_schedule|work_schedule|mixed",
    "courses": [
        {
            "name": "Nom COMPLET du cours/matière",
            "day": "lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche",
            "start_time": "HH:MM",
            "end_time": "HH:MM",
            "location": "Salle/Local/Bâtiment",
            "professor": "Nom du prof si visible"
        }
    ],
    "shifts": [
        {
            "day": "lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche",
            "start_time": "HH:MM",
            "end_time": "HH:MM",
            "is_night_shift": true|false,
            "role": "Poste/Rôle",
            "notes": ""
        }
    ],
    "events": [
        {
            "title": "Titre",
            "day": "jour",
            "start_time": "HH:MM",
            "end_time": "HH:MM",
            "notes": ""
        }
    ]
}

RÈGLES CRITIQUES:
- Jours TOUJOURS en français minuscules (lundi, mardi, mercredi, jeudi, vendredi, samedi, dimanche)
- Heures au format 24h avec deux chiffres (ex: "08:00", "14:30", "19:30", "23:00")
- Si "8h" → "08:00", si "14h30" → "14:30"
- Si une plage horaire est floue: "Matin" = "08:00"-"12:00", "Après-Midi" = "14:00"-"18:00", "Soir" = "18:00"-"22:00"
- is_night_shift = true si le shift commence après 20h OU finit après minuit
- UN COURS = UNE ENTRÉE SÉPARÉE (même si le même cours se répète plusieurs jours)
- Si le même cours apparaît Lundi et Mercredi, crée 2 entrées distinctes
- Arrays vides [] si aucune donnée de ce type trouvée
- Retourne UNIQUEMENT le JSON valide, AUCUN texte avant ou après

SOIS EXHAUSTIF: Extrait TOUS les cours/shifts visibles, même partiellement."""

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
                parts = [prompt]
                for img in content:
                    img_bytes = BytesIO()
                    img.save(img_bytes, format='PNG')
                    parts.append(types.Part.from_bytes(data=img_bytes.getvalue(), mime_type='image/png'))
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
            extracted_data = self._parse_response(response.text)

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
        Extract content from PDF file as images.

        Args:
            file_path: Path to the PDF file

        Returns:
            list: List of PIL Image objects (one per page)
        """
        images = []
        doc = fitz.open(file_path)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Render page to image with good resolution
            mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        doc.close()
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
                    is_night = shift.get('is_night_shift') or False

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
