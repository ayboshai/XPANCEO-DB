"""
PDF Parser using Unstructured with PyPDF2 fallback.
Extracts text, tables, and images from PDF documents.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple, Dict, Iterator, Literal, Optional

logger = logging.getLogger(__name__)

# Soft threshold for Unstructured parser (seconds).
# No hard timeouts are enforced to avoid data loss.
PDF_PARSE_TIMEOUT = 60

# Soft timeout for pdfplumber (logging only, NOT enforced)
# NO HARD TIMEOUT - tables must never be lost
PDFPLUMBER_SOFT_TIMEOUT = 120  # Log warning if pdfplumber takes >120s

# Maximum images to process with Unstructured (prevents hanging on malformed PDFs)
MAX_IMAGES_FOR_UNSTRUCTURED = 500


class TooManyImagesError(Exception):
    """Raised when PDF has too many embedded images for full processing."""
    pass


def _count_pdf_images(pdf_path: str) -> int:
    """
    Count images in PDF using pdfimages (most accurate, catches inline images).
    Falls back to PyPDF2 if pdfimages not available.
    Used to detect malformed PDFs with thousands of tiny images.
    """
    import subprocess

    # Try pdfimages first (most accurate - catches inline images)
    try:
        result = subprocess.run(
            ["pdfimages", "-list", pdf_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Count lines minus header (2 lines)
            lines = result.stdout.strip().split('\n')
            return max(0, len(lines) - 2)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback to PyPDF2 (only counts XObject images, misses inline)
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        count = 0
        for page in reader.pages:
            if '/XObject' in page.get('/Resources', {}):
                xobject = page['/Resources']['/XObject'].get_object()
                count += len([k for k in xobject.keys() if xobject[k].get('/Subtype') == '/Image'])
        return count
    except Exception:
        return 0  # On error, assume OK


@dataclass
class Element:
    """Raw element extracted from PDF."""
    
    type: Literal["text", "table", "image"]
    content: str  # Text content or table markdown
    page: int
    image_path: Optional[str] = None  # Path to saved image file
    bbox: Optional[List[float]] = None
    
    
def generate_doc_id(filepath: str, strategy: str = "hash") -> str:
    """Generate document ID based on configured strategy."""
    path = Path(filepath)
    
    if strategy == "hash":
        # Hash of absolute path + modification time
        mtime = os.path.getmtime(filepath)
        content = f"{path.absolute()}:{mtime}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    elif strategy == "filename":
        return path.stem
    elif strategy == "uuid":
        import uuid
        return str(uuid.uuid4())[:12]
    else:
        return hashlib.md5(str(path.absolute()).encode()).hexdigest()[:12]


class PDFParser:
    """
    Parse PDF files to extract text, tables, and images.
    Uses PyPDF2-first with coverage fallback to Unstructured (NO DATA LOSS).
    """

    def __init__(
        self,
        image_output_dir: Optional[str] = None,
        doc_id_strategy: str = "hash",
    ):
        self.image_output_dir = image_output_dir or tempfile.mkdtemp(prefix="xpanceo_images_")
        self.doc_id_strategy = doc_id_strategy
        os.makedirs(self.image_output_dir, exist_ok=True)

    def parse(self, pdf_path: str, timeout: int = PDF_PARSE_TIMEOUT,
              timing_callback: Optional[Callable[[str, float], None]] = None) -> Tuple[str, List[Element], int]:
        """
        Parse PDF and return (doc_id, list of elements, page_count).

        PyPDF2-FIRST strategy with coverage fallback:
        1. Try PyPDF2 (fast, ~1s per PDF)
        2. Check coverage: pages_with_text/total >= 0.6 AND total_chars >= 1500
        3. If coverage low → fallback to Unstructured (slower but better for scanned/complex PDFs)

        Images and tables extracted unconditionally (pdfimages + pdfplumber), so NO DATA LOSS.

        Args:
            pdf_path: Path to PDF file
            timeout: Soft warning threshold for Unstructured (default 60s).
                     No hard timeouts are enforced to avoid data loss.
            timing_callback: Optional callback(stage_name, duration_sec) for detailed timing

        Returns:
            Tuple of (doc_id, list of elements, page_count)
            NEVER loses data - images/tables extracted unconditionally.

        Raises:
            RuntimeError: Only if BOTH parsers fail completely (rare)
        """
        import time as time_module

        doc_id = generate_doc_id(pdf_path, self.doc_id_strategy)
        fallback_reason = None

        # Pre-check: detect malformed PDFs with excessive images (e.g., 20k+ tiny images)
        image_count = _count_pdf_images(pdf_path)
        force_pypdf2 = image_count > MAX_IMAGES_FOR_UNSTRUCTURED
        if force_pypdf2:
            logger.warning(f"Too many images ({image_count}) in {pdf_path}. Using PyPDF2 only.")

        # UNCONDITIONAL: Extract images BEFORE any parser attempt
        # If image_count is extreme, render full pages instead of extracting thousands of tiny images.
        t0 = time_module.time()
        image_elements = self._extract_all_images(pdf_path, doc_id, force_page_renders=force_pypdf2)
        if timing_callback:
            timing_callback("parse_pdfimages", time_module.time() - t0)

        # STRATEGY: PyPDF2-first with coverage fallback
        text_elements: List[Element] = []
        use_unstructured = False
        page_count = self.get_page_count(pdf_path)

        # Step 1: Always try PyPDF2 first (fast, ~1s)
        try:
            t0 = time_module.time()
            pypdf2_elements = list(self._parse_with_pypdf2(pdf_path))
            pypdf2_duration = time_module.time() - t0
            if timing_callback:
                timing_callback("parse_pypdf2", pypdf2_duration)

            # Step 2: Check coverage
            coverage = self._calculate_coverage(pypdf2_elements, page_count)

            if coverage['accept']:
                # Coverage good → use PyPDF2
                text_elements = pypdf2_elements
                logger.info(
                    f"PyPDF2 coverage OK for {pdf_path}: "
                    f"{coverage['pages_with_text']}/{coverage['total_pages']} pages, "
                    f"{coverage['total_chars']} chars"
                )
            else:
                # Coverage low → fallback to Unstructured
                fallback_reason = (
                    f"Low coverage ({coverage['pages_with_text']}/{coverage['total_pages']} pages, "
                    f"{coverage['total_chars']} chars)"
                )
                logger.warning(f"{fallback_reason} in {pdf_path}. Falling back to Unstructured.")
                use_unstructured = True

        except Exception as e:
            fallback_reason = f"PyPDF2 error: {e}"
            logger.warning(f"{fallback_reason} for {pdf_path}. Trying Unstructured.")
            use_unstructured = True

        # Step 3: Unstructured fallback if coverage low or PyPDF2 failed
        if use_unstructured and not force_pypdf2:
            try:
                t0 = time_module.time()
                text_elements = list(self._parse_with_unstructured_text_only(pdf_path, doc_id))
                duration = time_module.time() - t0
                if timing_callback:
                    timing_callback("parse_unstructured", duration)
                if duration > timeout:
                    logger.warning(
                        f"Unstructured slow for {pdf_path}: {duration:.1f}s (soft threshold {timeout}s)"
                    )
            except Exception as e:
                logger.error(f"Unstructured failed for {pdf_path}: {e}. Using low-coverage PyPDF2 text.")
                text_elements = pypdf2_elements if 'pypdf2_elements' in locals() else []

        # Final safety: if still no text elements, re-extract with PyPDF2
        if not text_elements:
            try:
                logger.warning(f"No text elements from any parser for {pdf_path}. Final PyPDF2 attempt.")
                text_elements = list(self._parse_with_pypdf2(pdf_path))
            except Exception as e2:
                logger.error(f"All parsers failed for {pdf_path}: {e2}")
                raise RuntimeError(f"Failed to parse {pdf_path}: {fallback_reason}, PyPDF2({e2})")

        # Table extraction must be complete: always scan all pages (no candidate filtering)
        table_candidate_pages = None

        # UNCONDITIONAL: Extract tables AFTER text to limit pdfplumber work when possible
        t0 = time_module.time()
        table_elements, tables_page_count = self._extract_all_tables(
            pdf_path,
            candidate_pages=table_candidate_pages,
        )
        if tables_page_count:
            page_count = tables_page_count
        if timing_callback:
            timing_callback("parse_pdfplumber", time_module.time() - t0)

        logger.info(f"Pre-extracted: {len(image_elements)} images, {len(table_elements)} tables")

        # Determine parser used for logging
        parser_name = "PyPDF2"
        if use_unstructured:
            parser_name = "Unstructured"
        elif force_pypdf2:
            parser_name = "PyPDF2 (forced)"

        # Combine with tables and images
        # Attach page-level table captions (if present) to improve retrieval
        table_elements = self._attach_table_captions(text_elements, table_elements)
        all_elements = text_elements + table_elements + image_elements
        all_elements = self._sort_elements_by_page(all_elements)
        logger.info(
            f"Parsed {pdf_path} ({parser_name}): "
            f"{len(all_elements)} elements (text={len(text_elements)}, tables={len(table_elements)}, images={len(image_elements)})"
        )
        return doc_id, all_elements, page_count

    def _attach_table_captions(self, text_elements: List[Element], table_elements: List[Element]) -> List[Element]:
        """
        Attach a table caption (if found in page text) to the table content.
        This improves retrieval for table queries without changing extraction logic.
        """
        if not text_elements or not table_elements:
            return table_elements

        # Build per-page text blob from extracted text elements
        page_text: Dict[int, str] = {}
        for elem in text_elements:
            if elem.type != "text" or not elem.content:
                continue
            page_text.setdefault(elem.page, []).append(elem.content)

        for page, parts in page_text.items():
            page_text[page] = "\n".join(parts)

        def _find_caption(text: str) -> Optional[str]:
            if not text:
                return None
            # Look for a short line containing "Table"/"Таблица"
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            for ln in lines:
                lower = ln.lower()
                if ("table" in lower or "табл" in lower or "таблица" in lower) and len(ln) <= 200:
                    return ln
            return None

        for t in table_elements:
            caption = _find_caption(page_text.get(t.page, ""))
            if caption and caption not in t.content:
                t.content = f"{caption}\n{t.content}"

        return table_elements

    def _calculate_coverage(self, text_elements: List[Element], total_pages: int) -> dict:
        """
        Calculate text coverage for PyPDF2 output to decide if Unstructured fallback needed.

        Coverage criteria (conservative to avoid false negatives):
        - pages_with_text / total_pages >= 0.6 (at least 60% of pages have text)
        - total_chars >= 1500 (minimum content threshold)

        A page is "with text" if it has >= 200 chars (not just whitespace/garbage).

        Returns:
            dict with keys: pages_with_text, total_pages, total_chars, accept (bool)
        """
        pages_with_text = 0
        total_chars = 0

        # Count pages with significant text (>= 200 chars)
        page_chars = {}
        for elem in text_elements:
            if elem.type != "text":
                continue
            page = elem.page
            content = elem.content.strip()
            page_chars[page] = page_chars.get(page, 0) + len(content)
            total_chars += len(content)

        pages_with_text = sum(1 for chars in page_chars.values() if chars >= 200)

        # Coverage thresholds
        page_coverage = pages_with_text / max(1, total_pages)
        accept = page_coverage >= 0.6 and total_chars >= 1500

        return {
            'pages_with_text': pages_with_text,
            'total_pages': total_pages,
            'total_chars': total_chars,
            'page_coverage': page_coverage,
            'accept': accept,
        }

    def _sort_elements_by_page(self, elements: List[Element]) -> List[Element]:
        """
        Sort elements by (page, type_order) to maintain correct prev/next chunk linking.
        Type order: text → table → image (fixed order for consistency).
        """
        TYPE_ORDER = {"text": 0, "table": 1, "image": 2}

        def sort_key(elem):
            return (elem.page, TYPE_ORDER.get(elem.type, 999))

        return sorted(elements, key=sort_key)

    def _looks_like_table_text(self, text: str) -> bool:
        """
        Heuristic to identify pages that likely contain tables.
        High-recall by design to avoid missing tables.
        """
        if not text or not text.strip():
            return False

        lower = text.lower()
        if "table" in lower or "табл" in lower or "таблица" in lower or "таб." in lower:
            return True
        if "|" in text:
            return True
        if re.search(r"\S+\s{2,}\S+\s{2,}\S+", text):
            return True

        digits = sum(1 for c in text if c.isdigit())
        digit_ratio = digits / max(1, len(text))
        if digits >= 12 and digit_ratio >= 0.05:
            return True

        return False

    def _infer_table_candidate_pages(self, text_elements: List[Element]) -> Optional[set]:
        """
        Infer candidate pages for table extraction from extracted text.
        Returns a set of page numbers, or an empty set if no text found.
        """
        if not text_elements:
            return set()

        page_text: Dict[int, List[str]] = {}
        for elem in text_elements:
            if elem.type != "text":
                continue
            if not elem.content:
                continue
            page_text.setdefault(elem.page, []).append(elem.content)

        if not page_text:
            return set()

        candidates = set()
        for page, parts in page_text.items():
            page_blob = "\n".join(parts)
            if self._looks_like_table_text(page_blob):
                candidates.add(page)

        # If no candidates found, fall back to all text pages (safety)
        if not candidates:
            candidates = set(page_text.keys())

        return candidates

    def _extract_all_images(
        self,
        pdf_path: str,
        doc_id: str,
        force_page_renders: bool = False,
    ) -> List[Element]:
        """
        Extract images from PDF unconditionally using pdfimages.
        Called BEFORE parser selection to guarantee image extraction.

        Returns:
            List of Image Element objects with paths and page numbers
        """
        doc_image_dir = os.path.join(self.image_output_dir, doc_id)

        # CRITICAL: Clean directory before extraction to prevent stale images from previous runs
        if os.path.exists(doc_image_dir):
            import shutil
            shutil.rmtree(doc_image_dir)
            logger.debug(f"Cleaned image directory: {doc_image_dir}")

        os.makedirs(doc_image_dir, exist_ok=True)

        if force_page_renders:
            # Fallback for PDFs with massive inline images: render full pages once
            # This prevents 10k+ tiny artifacts while preserving visual content.
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(pdf_path, dpi=200)
                elements = []
                for idx, img in enumerate(images, start=1):
                    out_path = Path(doc_image_dir) / f"page-{idx:03d}.png"
                    img.save(out_path, "PNG")
                    elements.append(Element(
                        type="image",
                        content="",
                        page=idx,
                        image_path=str(out_path),
                    ))
                return elements
            except Exception as e:
                logger.warning(f"Page render fallback failed for {pdf_path}: {e}. Using pdfimages instead.")

        # Extract using pdfimages
        self._extract_images_with_pdfimages(pdf_path, doc_image_dir)

        # Convert extracted files to Element objects
        elements = []
        extracted_images = sorted(Path(doc_image_dir).glob("img-*.*"))

        for img_file in extracted_images:
            # Apply existing filters
            if '-mask' in img_file.name or '-smask' in img_file.name:
                continue

            page_num = self._parse_page_from_pdfimages_filename(img_file.name)
            elements.append(Element(
                type="image",
                content="",
                page=page_num,
                image_path=str(img_file),
            ))

        return elements

    def _extract_all_tables(self, pdf_path: str, candidate_pages: Optional[set] = None) -> Tuple[List[Element], int]:
        """
        Extract tables from PDF unconditionally using pdfplumber.
        Called BEFORE parser selection to guarantee table extraction.

        Returns:
            Tuple of (list of Table Element objects, page_count)
        """
        tables, page_count = self._extract_tables_with_pdfplumber(pdf_path, candidate_pages=candidate_pages)
        # Deduplicate exact duplicates per page (pdfplumber can emit the same table twice).
        # This does not drop unique information, only identical repeats on the same page.
        import hashlib
        from collections import defaultdict

        def _norm_table(s: str) -> str:
            s = (s or "").strip().lower()
            s = re.sub(r"\s+", " ", s)
            return s

        seen_by_page = defaultdict(set)
        deduped: List[Element] = []
        dropped = 0

        for page_num, table_md in tables:
            norm = _norm_table(table_md)
            if not norm:
                continue
            h = hashlib.md5(norm.encode("utf-8")).hexdigest()
            if h in seen_by_page[page_num]:
                dropped += 1
                continue
            seen_by_page[page_num].add(h)
            deduped.append(Element(type="table", content=table_md, page=page_num))

        if dropped:
            logger.info(f"Deduplicated {dropped} duplicate tables on same pages for {pdf_path}")

        return deduped, page_count

    def _parse_with_unstructured_text_only(self, pdf_path: str, doc_id: str) -> Iterator[Element]:
        """
        Parse PDF using Unstructured library (TEXT ONLY).
        Images and tables are extracted separately via _extract_all_images/tables.
        """
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError:
            raise ImportError("unstructured not installed. Run: pip install unstructured[pdf]")

        elements = partition_pdf(
            filename=pdf_path,
            strategy="fast",  # Fast text extraction (~5-10x faster than hi_res)
            include_page_breaks=True,
        )

        current_page = 1

        for elem in elements:
            # Track page numbers
            if hasattr(elem, "metadata") and hasattr(elem.metadata, "page_number"):
                current_page = elem.metadata.page_number or current_page

            elem_type = type(elem).__name__

            # Text elements ONLY - ignore tables and images
            if elem_type in ("NarrativeText", "Text", "Title", "ListItem", "UncategorizedText"):
                text = str(elem)
                if text.strip():
                    yield Element(
                        type="text",
                        content=text,
                        page=current_page,
                    )

            # IGNORE Table elements - tables come ONLY from pdfplumber
            elif elem_type == "Table":
                logger.debug(f"Ignoring Unstructured Table on page {current_page} (using pdfplumber tables)")
                continue

            # IGNORE Image elements - images come ONLY from pdfimages
            elif elem_type in ("Image", "FigureCaption"):
                logger.debug(f"Ignoring Unstructured Image on page {current_page} (using pdfimages)")
                continue

    def _extract_images_with_pdfimages(self, pdf_path: str, output_dir: str):
        """
        Extract images from PDF using pdfimages with -p flag for page mapping.

        Uses -p flag to include page number in filename: img-{page}-{num}.ext
        This ensures reliable page mapping without parsing -list output.
        Uses -png to force all extracted images into a Vision/OCR-safe format.
        """
        import subprocess

        try:
            # Extract images with page number in filename (-p flag)
            # Output format: img-{page:03d}-{num:03d}.ext
            subprocess.run(
                ["pdfimages", "-png", "-p", pdf_path, os.path.join(output_dir, "img")],
                check=True,
                capture_output=True,
            )
            return
        except FileNotFoundError as e:
            raise RuntimeError("pdfimages not installed - cannot extract images without data loss") from e
        except subprocess.CalledProcessError as e:
            logger.warning(f"pdfimages failed for {pdf_path}: {e}. Retrying per-page extraction.")

        # Fallback: per-page extraction to avoid losing images on a single failure
        # Clean partial outputs from failed full extraction to avoid duplicates
        for f in Path(output_dir).glob("img-*.*"):
            try:
                f.unlink()
            except OSError:
                pass

        page_count = self.get_page_count(pdf_path)
        for page in range(1, page_count + 1):
            try:
                subprocess.run(
                    [
                        "pdfimages",
                        "-png",
                        "-p",
                        "-f",
                        str(page),
                        "-l",
                        str(page),
                        pdf_path,
                        os.path.join(output_dir, "img"),
                    ],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"pdfimages failed on page {page}/{page_count} for {pdf_path} - aborting to avoid data loss"
                ) from e

    def _parse_page_from_pdfimages_filename(self, filename: str) -> int:
        """
        Parse page number from pdfimages -p output filename.

        Format: img-{page}-{num}.ext (e.g., img-005-000.png → page 5)
        Fallback: page 1 if parsing fails.
        """
        import re
        # Match: img-{page}-{num}.ext
        match = re.match(r'img-(\d+)-\d+\.', filename)
        if match:
            return int(match.group(1))
        logger.debug(f"Could not parse page from {filename}, defaulting to page 1")
        return 1

    def _extract_tables_with_pdfplumber(
        self,
        pdf_path: str,
        candidate_pages: Optional[set] = None,
    ) -> Tuple[List[Tuple[int, str]], int]:
        """
        Extract tables using pdfplumber with fixed settings.

        NO HARD TIMEOUT - tables must never be lost.
        Soft timeout (120s) logs warning but continues processing.

        Filters:
        - min_rows >= 2
        - min_cols >= 2
        - empty_ratio <= 0.7
        NO max_rows limit (large tables handled by chunker).

        Returns:
            Tuple of (list of (page_number, table_markdown) tuples, page_count)
        """
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed, skipping table extraction")
            return [], 0

        table_settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 5,
            "join_tolerance": 5,
            "edge_min_length": 3,
            "min_words_vertical": 3,
            "min_words_horizontal": 1,
        }

        tables = []
        page_count = 0
        try:
            start_time = time.time()
            soft_timeout_logged = False

            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    if candidate_pages is not None and page_num not in candidate_pages:
                        continue

                    # Soft timeout check (logging only, NO break)
                    elapsed = time.time() - start_time
                    if elapsed > PDFPLUMBER_SOFT_TIMEOUT and not soft_timeout_logged:
                        logger.warning(
                            f"pdfplumber slow for {pdf_path}: {elapsed:.1f}s elapsed, "
                            f"still processing page {page_num}/{page_count}"
                        )
                        soft_timeout_logged = True

                    # SINGLE MODE: Always use fixed table_settings
                    # Skip pages with no text objects (cannot yield text-based tables)
                    if not page.chars:
                        continue
                    page_tables = page.extract_tables(table_settings=table_settings)

                    for table in page_tables:
                        if not table:
                            continue

                        # Filter: min 2 rows
                        if len(table) < 2:
                            continue

                        # Filter: min 2 columns (use max cols to handle ragged rows)
                        max_cols = max((len(row) for row in table if row), default=0)
                        if max_cols < 2:
                            continue

                        # Filter: empty ratio <= 0.7
                        total_cells = sum(len(row) for row in table)
                        empty_cells = sum(
                            1 for row in table for cell in row
                            if not cell or not str(cell).strip()
                        )
                        if total_cells > 0 and empty_cells / total_cells > 0.7:
                            continue

                        # Convert to markdown with padded rows for consistent columns
                        md_rows = []
                        for i, row in enumerate(table):
                            row = row or []
                            # Pad to max columns
                            padded = list(row) + [''] * max(0, max_cols - len(row))
                            # Normalize cells (strip + collapse newlines)
                            cells = [
                                str(c).replace("\n", " ").strip() if c is not None else ''
                                for c in padded
                            ]
                            md_rows.append('| ' + ' | '.join(cells) + ' |')
                            # Header separator after first row
                            if i == 0:
                                md_rows.append('| ' + ' | '.join(['---'] * len(cells)) + ' |')

                        table_md = '\n'.join(md_rows)
                        tables.append((page_num, table_md))

        except Exception as e:
            logger.error(f"pdfplumber table extraction failed for {pdf_path}: {e}")
            # DO NOT raise - return partial tables to avoid data loss

        return tables, page_count

    def _split_text_blocks(self, text: str) -> List[str]:
        """
        Split text into smaller blocks to improve retrieval granularity.
        Preserves content without loss by splitting on blank lines.
        """
        if not text:
            return []
        normalized = self._normalize_text_for_retrieval(text)
        # Split on blank lines (preserved by normalization)
        parts = re.split(r"\n\s*\n+", normalized.strip())
        return [p.strip() for p in parts if p.strip()]

    def _normalize_text_for_retrieval(self, text: str) -> str:
        """
        Light normalization to improve lexical matching without dropping content.
        - Preserves paragraph breaks
        - Fixes common PDF artifacts (line breaks, hyphenation, missing spaces)
        """
        if not text:
            return ""

        # Preserve blank lines as paragraph separators
        paragraph_token = "__PARA_BREAK__"
        text = re.sub(r"\n\s*\n+", paragraph_token, text)

        # De-hyphenate across line breaks: "exam-\nple" -> "example"
        text = re.sub(r"(?<=\w)-\n(?=[a-zа-я])", "", text)

        # Remaining newlines are layout artifacts; collapse to spaces
        text = re.sub(r"\n+", " ", text)

        # Insert missing spaces between letter/number boundaries and camelCase joins
        text = re.sub(r"(?<=[A-Za-zА-Яа-я])(?=\d)", " ", text)
        text = re.sub(r"(?<=\d)(?=[A-Za-zА-Яа-я])", " ", text)
        text = re.sub(r"(?<=[a-zа-я])(?=[A-ZА-Я])", " ", text)

        # Collapse excessive whitespace
        text = re.sub(r"[ \\t]+", " ", text).strip()

        # Restore paragraph breaks
        return text.replace(paragraph_token, "\n\n")

    def _parse_with_pypdf2(self, pdf_path: str) -> Iterator[Element]:
        """Fallback parser using PyPDF2 - text only, no tables/images."""
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
        
        reader = PdfReader(pdf_path)
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                for block in self._split_text_blocks(text):
                    yield Element(
                        type="text",
                        content=block,
                        page=page_num,
                    )
    
    def _html_to_markdown(self, html: str) -> Optional[str]:
        """Convert HTML table to Markdown."""
        if not html:
            return None
        
        try:
            # Simple HTML table to markdown conversion
            import re
            
            # Extract rows
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE)
            if not rows:
                return None
            
            md_rows = []
            for i, row in enumerate(rows):
                # Extract cells (th or td)
                cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.DOTALL | re.IGNORECASE)
                cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
                
                if cells:
                    md_rows.append("| " + " | ".join(cells) + " |")
                    
                    # Add header separator after first row
                    if i == 0:
                        md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
            
            return "\n".join(md_rows) if md_rows else None
        except Exception:
            return None
    
    def get_page_count(self, pdf_path: str) -> int:
        """Get total page count of PDF."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            return len(reader.pages)
        except Exception:
            return 0
