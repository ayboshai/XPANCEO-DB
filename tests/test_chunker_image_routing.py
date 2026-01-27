"""
Chunker image routing tests (no mocks, real code paths).
"""

from ingestion.chunker import Chunker
from ingestion.models import OCRResult


def test_text_scan_ocr_failed_keeps_both_chunks():
    """OCR-failed text_scan keeps OCR text and Vision caption."""
    chunker = Chunker()
    ocr_result = OCRResult(
        text="low quality",
        confidence=0.0,
        word_count=1,
        char_count=3,
        alpha_ratio=0.1,
    )

    chunks = chunker.create_image_chunks(
        doc_id="doc",
        page=1,
        image_index=0,
        ocr_text="low quality",
        ocr_result=ocr_result,
        caption="Caption fallback",
        image_path="image.png",
        image_hash="hash",
        source_path="source.pdf",
        config=None,
        image_type="text_scan",
    )

    types = {c.type for c in chunks}
    assert "image_ocr" in types
    assert "image_caption" in types
    assert all(c.metadata.processing_status == "success" for c in chunks)


def test_visual_caption_success_has_no_ocr_failure():
    """Visual type with Vision caption should not flag OCR failure."""
    chunker = Chunker()

    chunks = chunker.create_image_chunks(
        doc_id="doc",
        page=2,
        image_index=1,
        ocr_text="",
        ocr_result=None,
        caption="Chart caption",
        image_path="image.png",
        image_hash="hash",
        source_path="source.pdf",
        config=None,
        image_type="chart",
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.type == "image_caption"
    assert chunk.metadata.ocr_failed is False
    assert chunk.metadata.vision_used is True


def test_visual_total_failure_creates_failed_chunk():
    """Visual type with no caption and no OCR creates failed chunk."""
    chunker = Chunker()

    chunks = chunker.create_image_chunks(
        doc_id="doc",
        page=3,
        image_index=2,
        ocr_text="",
        ocr_result=None,
        caption=None,
        image_path="image.png",
        image_hash="hash",
        source_path="source.pdf",
        config=None,
        image_type="photo",
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.metadata.processing_status == "failed"
