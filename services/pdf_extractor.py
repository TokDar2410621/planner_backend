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


# --- Resource caps for PDF rasterization (S4: prevent PDF-bomb DoS) ---
# A malicious PDF can declare thousands of pages or huge page dimensions.
# Rendering those at 3x zoom / 300 DPI would exhaust memory. These caps bound
# both the vision rasterization (PyMuPDF, in document_processor) and the OCR
# rasterization (pdf2image, here) before any pixels are produced.
MAX_PDF_PAGES = 30
MAX_RENDER_PIXELS = 25_000_000  # per rendered page (~25 megapixels)
POINTS_PER_INCH = 72.0
DEFAULT_OCR_DPI = 300
MIN_OCR_DPI = 72


def rendered_pixel_count(width_pt: float, height_pt: float, zoom: float) -> int:
    """
    Pixel count produced by rendering a page at a given zoom factor.

    Args:
        width_pt: Page width in points (1/72 inch) at zoom=1.
        height_pt: Page height in points at zoom=1.
        zoom: Render scale factor (fitz Matrix zoom, or dpi/72 for rasterizers).

    Returns:
        Number of pixels the rendered page would occupy.
    """
    w = int(round(max(width_pt, 0.0) * zoom))
    h = int(round(max(height_pt, 0.0) * zoom))
    return w * h


def max_zoom_for_page(width_pt: float, height_pt: float,
                      max_pixels: int = MAX_RENDER_PIXELS) -> float:
    """
    Largest zoom factor that keeps a single page under ``max_pixels``.

    Used to downscale rendering instead of rejecting outright.
    """
    if width_pt <= 0 or height_pt <= 0:
        return 1.0
    return (max_pixels / (width_pt * height_pt)) ** 0.5


def exceeds_render_caps(page_count: int, page_dimensions, zoom: float, *,
                        max_pages: int = MAX_PDF_PAGES,
                        max_pixels: int = MAX_RENDER_PIXELS) -> bool:
    """
    Whether rendering this PDF would blow the page-count or per-page pixel caps.

    Args:
        page_count: Number of pages in the PDF.
        page_dimensions: Iterable of (width_pt, height_pt) at zoom=1 (points).
        zoom: Render scale factor that would be applied.
        max_pages: Maximum allowed page count.
        max_pixels: Maximum allowed pixels per rendered page.

    Returns:
        True if the document is oversized and must be rejected/downscaled.
    """
    if page_count > max_pages:
        return True
    for width_pt, height_pt in page_dimensions:
        if rendered_pixel_count(width_pt, height_pt, zoom) > max_pixels:
            return True
    return False


def safe_dpi_for_pages(page_dimensions, requested_dpi: int = DEFAULT_OCR_DPI,
                       max_pixels: int = MAX_RENDER_PIXELS) -> int:
    """
    Downscaled DPI so the largest page stays under ``max_pixels`` when rasterized.

    Never returns more than ``requested_dpi`` nor less than ``MIN_OCR_DPI``.
    """
    dpi = float(requested_dpi)
    for width_pt, height_pt in page_dimensions:
        zoom_cap = max_zoom_for_page(width_pt, height_pt, max_pixels)
        dpi = min(dpi, zoom_cap * POINTS_PER_INCH)
    return int(max(MIN_OCR_DPI, min(dpi, requested_dpi)))


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

    def _pdf_page_geometry(self, source: str | bytes | BytesIO):
        """
        Inspect page count and per-page dimensions WITHOUT rasterizing.

        Returns:
            Tuple of (page_count, [(width_pt, height_pt), ...]).
        """
        if isinstance(source, str):
            pdf = pdfplumber.open(source)
        elif isinstance(source, bytes):
            pdf = pdfplumber.open(BytesIO(source))
        else:
            try:
                source.seek(0)
            except Exception:
                pass
            pdf = pdfplumber.open(source)

        with pdf:
            dims = [(float(p.width), float(p.height)) for p in pdf.pages]
        return len(dims), dims

    def _extract_with_ocr(self, source: str | bytes | BytesIO) -> Dict[str, Any]:
        """Extract text using Tesseract OCR (for scanned PDFs)."""
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR dependencies not installed")

        pages_text = []

        # S4: bound the rasterization before rendering at 300 DPI.
        # Reject PDF bombs (too many pages) and downscale DPI so no single page
        # exceeds the per-page pixel cap.
        dpi = DEFAULT_OCR_DPI
        try:
            page_count, dims = self._pdf_page_geometry(source)
            if page_count > MAX_PDF_PAGES:
                raise ValueError(
                    f"PDF has too many pages for OCR ({page_count} > {MAX_PDF_PAGES})"
                )
            dpi = safe_dpi_for_pages(dims, DEFAULT_OCR_DPI)
            if dpi < DEFAULT_OCR_DPI:
                logger.info(f"OCR DPI downscaled to {dpi} to bound rasterization")
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"Could not inspect PDF geometry ({e}); using conservative DPI")
            dpi = min(DEFAULT_OCR_DPI, 150)

        # Convert PDF pages to images (page count is also capped defensively)
        if isinstance(source, str):
            images = convert_from_path(source, dpi=dpi, last_page=MAX_PDF_PAGES)
        elif isinstance(source, bytes):
            images = convert_from_bytes(source, dpi=dpi, last_page=MAX_PDF_PAGES)
        else:
            source.seek(0)
            images = convert_from_bytes(source.read(), dpi=dpi, last_page=MAX_PDF_PAGES)

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
