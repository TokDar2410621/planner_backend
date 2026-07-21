"""
Regression tests for group A9 fixes (S4, resource side).

Document-processing hardening against PDF-bomb / DoS:
- Page-count and per-page pixel caps before rasterizing.
- OCR DPI downscaling so huge pages never blow the pixel cap.
- Vision-render zoom downscaling (same intent, PyMuPDF path).
- Generic processing-error message (no raw exception leaked to clients).

These tests are offline: no network, no LLM, no fitz/pdf2image rasterization.
They unit-test the pure guard functions and the module-level generic error.
"""
import pytest

from services.pdf_extractor import (
    MAX_PDF_PAGES,
    MAX_RENDER_PIXELS,
    MIN_OCR_DPI,
    DEFAULT_OCR_DPI,
    POINTS_PER_INCH,
    rendered_pixel_count,
    max_zoom_for_page,
    exceeds_render_caps,
    safe_dpi_for_pages,
)


# A4 page at 72 pt/inch: 8.27 x 11.69 in -> ~595 x 842 pt.
A4 = (595.0, 842.0)


# --------------------------------------------------------------------------
# rendered_pixel_count
# --------------------------------------------------------------------------

def test_rendered_pixel_count_scales_with_zoom():
    # 100x200 pt at zoom 3 -> 300 x 600 = 180000 px
    assert rendered_pixel_count(100, 200, 3) == 300 * 600


def test_rendered_pixel_count_clamps_negative():
    assert rendered_pixel_count(-5, 200, 3) == 0


# --------------------------------------------------------------------------
# exceeds_render_caps  (the "does this PDF exceed the caps" helper)
# --------------------------------------------------------------------------

def test_normal_a4_pdf_within_caps():
    # A handful of A4 pages at zoom 3 must be accepted.
    dims = [A4] * 5
    assert exceeds_render_caps(len(dims), dims, zoom=3) is False


def test_too_many_pages_is_oversized():
    dims = [A4] * (MAX_PDF_PAGES + 1)
    assert exceeds_render_caps(len(dims), dims, zoom=3) is True


def test_oversized_page_dimensions_is_oversized():
    # One monstrous page that blows the per-page pixel cap at zoom 3.
    huge = (20000.0, 20000.0)  # 20000*3 * 20000*3 px >> MAX_RENDER_PIXELS
    assert rendered_pixel_count(*huge, 3) > MAX_RENDER_PIXELS
    assert exceeds_render_caps(1, [huge], zoom=3) is True


def test_page_within_caps_at_low_zoom_but_oversized_at_high_zoom():
    # Dimensions that are fine at zoom 1 but oversized at zoom 3.
    # pixels at zoom = (w*zoom)*(h*zoom); pick so zoom=3 exceeds, zoom=1 doesn't.
    w = h = 3000.0
    assert exceeds_render_caps(1, [(w, h)], zoom=1) is False
    assert exceeds_render_caps(1, [(w, h)], zoom=3) is True


# --------------------------------------------------------------------------
# max_zoom_for_page  (downscale factor)
# --------------------------------------------------------------------------

def test_max_zoom_keeps_page_under_cap():
    w = h = 5000.0
    z = max_zoom_for_page(w, h, MAX_RENDER_PIXELS)
    # Rendering at that zoom must land at (or just under) the cap.
    assert rendered_pixel_count(w, h, z) <= MAX_RENDER_PIXELS
    # And a hair more zoom would exceed it.
    assert rendered_pixel_count(w, h, z * 1.01) > MAX_RENDER_PIXELS


def test_max_zoom_zero_dimensions_is_safe():
    assert max_zoom_for_page(0, 0, MAX_RENDER_PIXELS) == 1.0


# --------------------------------------------------------------------------
# safe_dpi_for_pages  (OCR rasterization bound)
# --------------------------------------------------------------------------

def test_safe_dpi_unchanged_for_normal_pages():
    # Normal A4 pages at 300 DPI are ~2480x3508 ~= 8.7 MP, under the cap.
    assert safe_dpi_for_pages([A4], DEFAULT_OCR_DPI) == DEFAULT_OCR_DPI


def test_safe_dpi_downscaled_for_huge_page():
    # 3000x3000 pt is oversized at 300 DPI but downscalable to a valid DPI
    # that stays above the floor and under the pixel cap.
    huge = (3000.0, 3000.0)
    dpi = safe_dpi_for_pages([huge], DEFAULT_OCR_DPI)
    assert dpi < DEFAULT_OCR_DPI
    assert dpi > MIN_OCR_DPI
    # At the returned DPI, the page stays under the pixel cap.
    zoom = dpi / POINTS_PER_INCH
    assert rendered_pixel_count(*huge, zoom) <= MAX_RENDER_PIXELS


def test_safe_dpi_never_below_floor():
    absurd = (1_000_000.0, 1_000_000.0)
    assert safe_dpi_for_pages([absurd], DEFAULT_OCR_DPI) == MIN_OCR_DPI


def test_safe_dpi_never_exceeds_requested():
    # Even tiny pages should not bump DPI above the requested value.
    tiny = (10.0, 10.0)
    assert safe_dpi_for_pages([tiny], DEFAULT_OCR_DPI) == DEFAULT_OCR_DPI


# --------------------------------------------------------------------------
# Module still imports; generic error message is used (no raw exception leak)
# --------------------------------------------------------------------------

def test_document_processor_imports_and_uses_generic_error():
    from services import document_processor as dp

    assert isinstance(dp.GENERIC_PROCESSING_ERROR, str)
    assert dp.GENERIC_PROCESSING_ERROR.strip()
    # Download and render caps are wired in.
    assert dp.MAX_DOWNLOAD_BYTES > 0
    assert dp.PDF_RENDER_ZOOM == 3
    assert dp.MAX_PDF_PAGES == MAX_PDF_PAGES
