"""
PDF Text Extraction Service using pdfplumber with OCR fallback.

This module provides reliable text extraction from PDFs:
1. Primary: pdfplumber for native PDFs (preserves structure/tables)
2. Fallback: Tesseract OCR for scanned PDFs (images)
"""
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any

import pdfplumber
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import OCR dependencies (optional)
try:
    import pytesseract
    from pdf2image import convert_from_path, convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR dependencies not available. Install pytesseract and pdf2image for scanned PDF support.")


class PDFTextExtractor:
    """
    Extract text from PDFs using pdfplumber with OCR fallback.

    Usage:
        extractor = PDFTextExtractor()
        result = extractor.extract(pdf_path_or_bytes)

        # result = {
        #     'text': 'Full extracted text...',
        #     'pages': ['Page 1 text', 'Page 2 text', ...],
        #     'tables': [[table1_data], [table2_data], ...],
        #     'method': 'pdfplumber' | 'ocr',
        #     'page_count': 2
        # }
    """

    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize the extractor.

        Args:
            tesseract_cmd: Path to tesseract executable (optional, auto-detected)
        """
        if tesseract_cmd and OCR_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract(self, source: str | bytes | BytesIO) -> Dict[str, Any]:
        """
        Extract text from a PDF file.

        Args:
            source: File path (str), bytes, or BytesIO object

        Returns:
            Dictionary with extracted text, tables, and metadata
        """
        result = {
            'text': '',
            'pages': [],
            'tables': [],
            'method': 'pdfplumber',
            'page_count': 0,
            'error': None
        }

        try:
            # Try pdfplumber first (native text extraction)
            result = self._extract_with_pdfplumber(source)

            # Check if we got meaningful text
            if not result['text'].strip() or len(result['text'].strip()) < 50:
                logger.info("pdfplumber returned minimal text, trying OCR fallback...")
                if OCR_AVAILABLE:
                    result = self._extract_with_ocr(source)
                else:
                    logger.warning("OCR not available, returning minimal text")
                    result['error'] = "PDF appears to be scanned but OCR is not available"

        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            result['error'] = str(e)

            # Try OCR as last resort
            if OCR_AVAILABLE:
                try:
                    result = self._extract_with_ocr(source)
                except Exception as ocr_error:
                    logger.error(f"OCR fallback also failed: {ocr_error}")
                    result['error'] = f"Both pdfplumber and OCR failed: {e}, {ocr_error}"

        return result

    def _extract_with_pdfplumber(self, source: str | bytes | BytesIO) -> Dict[str, Any]:
        """Extract text using pdfplumber (best for native PDFs)."""
        pages_text = []
        all_tables = []

        # Open PDF based on source type
        if isinstance(source, str):
            pdf = pdfplumber.open(source)
        elif isinstance(source, bytes):
            pdf = pdfplumber.open(BytesIO(source))
        else:
            pdf = pdfplumber.open(source)

        with pdf:
            for page in pdf.pages:
                # Extract text with layout preservation
                text = page.extract_text(
                    layout=True,  # Preserve layout/columns
                    x_tolerance=3,
                    y_tolerance=3
                )
                if text:
                    pages_text.append(text)

                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        # Clean up table data
                        cleaned_table = []
                        for row in table:
                            cleaned_row = [cell.strip() if cell else '' for cell in row]
                            if any(cleaned_row):  # Skip empty rows
                                cleaned_table.append(cleaned_row)
                        if cleaned_table:
                            all_tables.append(cleaned_table)

            page_count = len(pdf.pages)

        full_text = '\n\n--- PAGE BREAK ---\n\n'.join(pages_text)

        return {
            'text': full_text,
            'pages': pages_text,
            'tables': all_tables,
            'method': 'pdfplumber',
            'page_count': page_count,
            'error': None
        }

    def _extract_with_ocr(self, source: str | bytes | BytesIO) -> Dict[str, Any]:
        """Extract text using Tesseract OCR (for scanned PDFs)."""
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR dependencies not installed")

        pages_text = []

        # Convert PDF pages to images
        if isinstance(source, str):
            images = convert_from_path(source, dpi=300)
        elif isinstance(source, bytes):
            images = convert_from_bytes(source, dpi=300)
        else:
            source.seek(0)
            images = convert_from_bytes(source.read(), dpi=300)

        for i, image in enumerate(images):
            # Run OCR on each page
            text = pytesseract.image_to_string(
                image,
                lang='fra+eng',  # French + English
                config='--psm 6'  # Assume uniform block of text
            )
            if text.strip():
                pages_text.append(text)

            logger.info(f"OCR processed page {i+1}/{len(images)}")

        full_text = '\n\n--- PAGE BREAK ---\n\n'.join(pages_text)

        return {
            'text': full_text,
            'pages': pages_text,
            'tables': [],  # OCR doesn't extract tables well
            'method': 'ocr',
            'page_count': len(images),
            'error': None
        }

    def extract_tables_only(self, source: str | bytes | BytesIO) -> List[List[List[str]]]:
        """
        Extract only tables from a PDF.

        Returns:
            List of tables, where each table is a list of rows (list of cells)
        """
        result = self._extract_with_pdfplumber(source)
        return result.get('tables', [])


def extract_schedule_text(pdf_path: str) -> str:
    """
    Convenience function to extract schedule text from a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text string
    """
    extractor = PDFTextExtractor()
    result = extractor.extract(pdf_path)
    return result['text']


def extract_schedule_with_tables(pdf_path: str) -> Dict[str, Any]:
    """
    Extract both text and tables from a schedule PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with 'text', 'tables', and metadata
    """
    extractor = PDFTextExtractor()
    return extractor.extract(pdf_path)
