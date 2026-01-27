from __future__ import annotations

from pathlib import Path

from PIL import Image

from ingestion.pipeline import IngestionPipeline


def _make_blank_image(path: Path, size: int = 512) -> None:
    Image.new("RGB", (size, size), color=(255, 255, 255)).save(path)


def _build_config(tmp_path: Path) -> dict:
    cache_dir = tmp_path / "cache"
    data_dir = tmp_path / "data"
    index_dir = tmp_path / "index"
    return {
        "openai_api_key": "test-key",
        "data_dir": str(data_dir),
        "cache_dir": str(cache_dir),
        "index_dir": str(index_dir),
        "ocr_cache_dir": str(cache_dir / "ocr"),
        "vision_cache_dir": str(cache_dir / "vision"),
        "embedding_cache_dir": str(cache_dir / "embeddings"),
        "model_vision": "gpt-4o-mini",
        "model_embeddings": "text-embedding-3-small",
        "chunk_size_tokens": 128,
        "chunk_overlap_tokens": 0,
        "top_k": 5,
        "hybrid_enabled": True,
    }


def test_blank_uniform_image_skips_vision_and_audits(tmp_path: Path) -> None:
    image_path = tmp_path / "blank.png"
    _make_blank_image(image_path)

    config = _build_config(tmp_path)
    pipeline = IngestionPipeline(config)

    chunks = pipeline._process_image(
        doc_id="doc-blank",
        page=1,
        image_index=0,
        image_path=str(image_path),
        source_path="dummy.pdf",
        low_info_hashes=None,
    )

    assert chunks, "Pipeline must always return at least one chunk"
    assert all(not c.metadata.vision_used for c in chunks)
    assert any(c.metadata.processing_status == "failed" for c in chunks)
    assert any(c.metadata.vision_error == "Vision skipped: blank/uniform image" for c in chunks)

    # Defense-in-depth: Vision cache should remain empty when Vision is skipped.
    vision_cache_dir = Path(config["vision_cache_dir"])
    vision_cache_files = list(vision_cache_dir.glob("*.json")) if vision_cache_dir.exists() else []
    assert not vision_cache_files

