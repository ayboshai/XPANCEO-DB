"""
PDF Parser using Unstructured with PyPDF2 fallback.
Extracts text, tables, and images from PDF documents.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Element:
    """Raw element extracted from PDF."""
    
    type: Literal["text", "table", "image"]
    content: str  # Text content or table markdown
    page: int
    image_path: Optional[str] = None  # Path to saved image file
    bbox: Optional[list[float]] = None
    
    
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
    Uses Unstructured as primary parser with PyPDF2 fallback.
    """
    
    def __init__(
        self,
        image_output_dir: Optional[str] = None,
        doc_id_strategy: str = "hash",
    ):
        self.image_output_dir = image_output_dir or tempfile.mkdtemp(prefix="xpanceo_images_")
        self.doc_id_strategy = doc_id_strategy
        os.makedirs(self.image_output_dir, exist_ok=True)
    
    def parse(self, pdf_path: str) -> tuple[str, list[Element]]:
        """
        Parse PDF and return (doc_id, list of elements).
        Tries Unstructured first, falls back to PyPDF2 for text-only.
        """
        doc_id = generate_doc_id(pdf_path, self.doc_id_strategy)
        
        try:
            elements = list(self._parse_with_unstructured(pdf_path, doc_id))
            logger.info(f"Parsed {pdf_path} with Unstructured: {len(elements)} elements")
            return doc_id, elements
        except Exception as e:
            logger.warning(f"Unstructured failed for {pdf_path}: {e}. Trying PyPDF2 fallback.")
            try:
                elements = list(self._parse_with_pypdf2(pdf_path))
                logger.info(f"Parsed {pdf_path} with PyPDF2 fallback: {len(elements)} elements")
                return doc_id, elements
            except Exception as e2:
                logger.error(f"Both parsers failed for {pdf_path}: {e2}")
                raise RuntimeError(f"Failed to parse {pdf_path}: Unstructured({e}), PyPDF2({e2})")
    
    def _parse_with_unstructured(self, pdf_path: str, doc_id: str) -> Iterator[Element]:
        """Parse PDF using Unstructured library."""
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError:
            raise ImportError("unstructured not installed. Run: pip install unstructured[pdf]")
        
        # Create image output directory for this document
        doc_image_dir = os.path.join(self.image_output_dir, doc_id)
        os.makedirs(doc_image_dir, exist_ok=True)
        
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",  # High resolution for better table/image extraction
            extract_images_in_pdf=True,
            extract_image_block_output_dir=doc_image_dir,
            include_page_breaks=True,
        )
        
        current_page = 1
        image_counter = 0
        
        for elem in elements:
            # Track page numbers
            if hasattr(elem, "metadata") and hasattr(elem.metadata, "page_number"):
                current_page = elem.metadata.page_number or current_page
            
            elem_type = type(elem).__name__
            
            # Text elements
            if elem_type in ("NarrativeText", "Text", "Title", "ListItem", "UncategorizedText"):
                text = str(elem)
                if text.strip():
                    yield Element(
                        type="text",
                        content=text,
                        page=current_page,
                    )
            
            # Table elements
            elif elem_type == "Table":
                # Convert to markdown if possible
                table_text = str(elem)
                if hasattr(elem, "metadata") and hasattr(elem.metadata, "text_as_html"):
                    # Try to use HTML representation
                    table_text = self._html_to_markdown(elem.metadata.text_as_html) or table_text
                
                yield Element(
                    type="table",
                    content=table_text,
                    page=current_page,
                )
            
            # Image elements
            elif elem_type in ("Image", "FigureCaption"):
                # Look for extracted image files
                image_files = list(Path(doc_image_dir).glob(f"*{image_counter}*.*"))
                if not image_files:
                    image_files = list(Path(doc_image_dir).glob("*.png"))
                    image_files.extend(Path(doc_image_dir).glob("*.jpg"))
                
                image_path = None
                if image_files:
                    # Use the first matching image
                    for img_file in sorted(image_files):
                        if img_file.is_file():
                            image_path = str(img_file)
                            break
                
                yield Element(
                    type="image",
                    content=str(elem) if str(elem).strip() else "",
                    page=current_page,
                    image_path=image_path,
                )
                image_counter += 1
        
        # Also check for any images not captured through elements
        self._extract_additional_images(pdf_path, doc_image_dir, current_page)
    
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
                yield Element(
                    type="text",
                    content=text.strip(),
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
    
    def _extract_additional_images(self, pdf_path: str, output_dir: str, max_page: int) -> None:
        """Extract images using pdf2image if Unstructured didn't catch them."""
        try:
            from pdf2image import convert_from_path
            
            existing_images = list(Path(output_dir).glob("*.png"))
            if len(existing_images) >= max_page:
                return  # Already have enough images
            
            # Convert PDF pages to images (for OCR of entire page if needed)
            # This is a fallback - only used if no images extracted
            pass  # Skip for now to avoid duplication
        except ImportError:
            pass  # pdf2image not installed, skip
    
    def get_page_count(self, pdf_path: str) -> int:
        """Get total page count of PDF."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            return len(reader.pages)
        except Exception:
            return 0
