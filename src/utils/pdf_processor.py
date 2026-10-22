from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import pdfplumber
from tqdm import tqdm
import sys

from config.config import LOGGER, CHUNK_SIZE, CHUNK_OVERLAP, PROCESSED_DIR
from .validator import safe_write_json


def is_valid_table(table: List[List[Any]], min_rows: int = 2, min_cols: int = 2, min_cell_fill_ratio: float = 0.3) -> bool:
    """
    Validate if a detected table is actually a real table.
    
    Args:
        table: Raw table data (list of rows)
        min_rows: Minimum number of rows to be considered a table
        min_cols: Minimum number of columns to be considered a table
        min_cell_fill_ratio: Minimum ratio of non-empty cells
        
    Returns:
        True if this appears to be a valid table
    """
    if not table or len(table) < min_rows:
        return False
    
    # Check if all rows have consistent column count
    col_counts = [len(row) for row in table if row]
    if not col_counts:
        return False
    
    max_cols = max(col_counts)
    min_cols_in_table = min(col_counts)
    
    # Reject if too few columns
    if max_cols < min_cols:
        return False
    
    # Reject if column count is too inconsistent (likely not a real table)
    if max_cols > 1 and min_cols_in_table < max_cols * 0.5:
        return False
    
    # Count non-empty cells
    total_cells = 0
    filled_cells = 0
    
    for row in table:
        for cell in row:
            total_cells += 1
            if cell and str(cell).strip():
                filled_cells += 1
    
    # Check cell fill ratio
    if total_cells > 0:
        fill_ratio = filled_cells / total_cells
        if fill_ratio < min_cell_fill_ratio:
            return False
    
    # Additional check: reject tables with only one column of text (likely formatted text, not table)
    if max_cols == 1:
        return False
    
    # Check if table has meaningful content structure
    # If most rows have only 1 filled cell, it's likely not a table
    single_cell_rows = 0
    for row in table:
        filled_in_row = sum(1 for cell in row if cell and str(cell).strip())
        if filled_in_row <= 1:
            single_cell_rows += 1
    
    if len(table) > 0 and single_cell_rows / len(table) > 0.7:
        return False
    
    return True


def extract_tables_with_bbox(page, min_rows: int = 2, min_cols: int = 2) -> List[Dict[str, Any]]:
    """Extract tables from a page with validation and bounding box info.
    
    Args:
        page: pdfplumber page object
        min_rows: Minimum rows for a valid table
        min_cols: Minimum columns for a valid table
    Returns:
        List of table dicts with keys: table_index, headers, rows, raw_data, bbox
    """
    tables = []
    try:
        # Configure table settings for better detection
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "explicit_vertical_lines": [],
            "explicit_horizontal_lines": [],
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
            "min_words_vertical": 3,
            "min_words_horizontal": 1,
        }
        
        # Find tables with bounding boxes
        found_tables = page.find_tables(table_settings)
        
        if found_tables:
            valid_table_idx = 0
            for idx, table_obj in enumerate(found_tables):
                table_data = table_obj.extract()
                
                if table_data and len(table_data) > 0:
                    # Validate the table
                    if not is_valid_table(table_data, min_rows=min_rows, min_cols=min_cols):
                        LOGGER.debug(f"Skipping invalid table {idx} on page (too few rows/cols or inconsistent structure)")
                        continue
                    
                    # First row as headers (if available)
                    headers = table_data[0] if table_data else []
                    rows = table_data[1:] if len(table_data) > 1 else []
                    
                    # Get bounding box
                    bbox = table_obj.bbox  # (x0, top, x1, bottom)
                    
                    tables.append({
                        "table_index": valid_table_idx,
                        "headers": headers,
                        "rows": rows,
                        "raw_data": table_data,
                        "num_rows": len(table_data),
                        "num_cols": len(table_data[0]) if table_data else 0,
                        "bbox": bbox,
                    })
                    valid_table_idx += 1
    except Exception as exc:
        LOGGER.warning("Failed to extract tables from page: %s", exc)
    
    return tables


def format_table_as_text(table: Dict[str, Any]) -> str:
    """Convert table dict to readable text format.
    
    Args:
        table: Table dictionary with headers and rows
    Returns:
        Formatted string representation of the table
    """
    lines = []
    lines.append(f"\n[TABLE {table['table_index']}]")
    
    # Add headers
    if table['headers']:
        headers = [str(h).strip() if h else '' for h in table['headers']]
        lines.append(" | ".join(headers))
        lines.append("-" * 80)
    
    # Add rows
    for row in table['rows']:
        row_str = [str(cell).strip() if cell else '' for cell in row]
        lines.append(" | ".join(row_str))
    
    lines.append("[END TABLE]")
    return "\n".join(lines)


def get_text_in_bbox(page, bbox: Tuple[float, float, float, float]) -> str:
    """Extract text within a bounding box.
    
    Args:
        page: pdfplumber page object
        bbox: (x0, top, x1, bottom)
    Returns:
        Text within the bounding box
    """
    try:
        cropped = page.within_bbox(bbox)
        return cropped.extract_text() or ""
    except:
        return ""


def replace_tables_in_text(page, text: str, tables: List[Dict[str, Any]]) -> str:
    """Replace table regions in text with formatted table content.
    
    Args:
        page: pdfplumber page object
        text: Original extracted text
        tables: List of table dicts with bbox information
    Returns:
        Text with tables replaced by formatted versions
    """
    if not tables:
        return text
    
    # Extract text for each table region to find what to replace
    replacements = []
    for table in tables:
        if 'bbox' not in table:
            continue
        
        # Get the text that was in the table area
        table_region_text = get_text_in_bbox(page, table['bbox'])
        
        if table_region_text.strip():
            # Format the table
            formatted_table = format_table_as_text(table)
            replacements.append((table_region_text.strip(), formatted_table))
    
    # Apply replacements
    modified_text = text
    for original, formatted in replacements:
        # Try to find and replace the table text
        if original in modified_text:
            modified_text = modified_text.replace(original, formatted, 1)
    
    return modified_text


def extract_pages(pdf_path: str, include_tables: bool = True, replace_with_tables: bool = False, 
                  min_table_rows: int = 2, min_table_cols: int = 2) -> List[Dict[str, Any]]:
    """Extract text and tables per page with metadata.

    Args:
        pdf_path: path to a PDF file
        include_tables: whether to extract tables (default: True)
        replace_with_tables: if True, replace table portions in text with formatted tables (default: False)
        min_table_rows: minimum rows for valid table (default: 2)
        min_table_cols: minimum columns for valid table (default: 2)
    Returns:
        List of page dicts with keys: page_number, text, tables, metadata
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: List[Dict[str, Any]] = []
    with pdfplumber.open(str(path)) as pdf:
        for idx, page in enumerate(tqdm(pdf.pages, desc=f"Reading {path.name}"), start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                LOGGER.warning("Failed to extract text from page %s of %s: %s", idx, path.name, exc)
                text = ""
            
            # Extract tables if enabled
            tables = []
            if include_tables:
                tables = extract_tables_with_bbox(page, min_rows=min_table_rows, min_cols=min_table_cols)
            
            # Replace table portions in text if requested and tables exist
            if replace_with_tables and tables:
                text = replace_tables_in_text(page, text, tables)
            elif not replace_with_tables and tables:
                # Append mode: add formatted tables at the end
                for table in tables:
                    text += "\n\n" + format_table_as_text(table)
            
            # Only add page if it has content (text or tables)
            if text.strip() or tables:
                page_data = {
                    "page_number": idx,
                    "text": text,
                    "tables": tables,
                    "has_tables": len(tables) > 0,
                    "replaced_with_tables": replace_with_tables and len(tables) > 0,
                    "metadata": {
                        "pdf_name": path.name,
                        "total_pages": len(pdf.pages),
                        "num_tables": len(tables),
                    },
                }
                pages.append(page_data)
    
    return pages


def combine_text_and_tables(page: Dict[str, Any], replace_with_tables: bool = False) -> str:
    """Get the combined text (already processed in extract_pages).
    
    Args:
        page: Page dictionary with text and tables
        replace_with_tables: Not used here, processing done in extract_pages
    Returns:
        The text field which already has tables processed
    """
    return page["text"]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks for long contexts."""
    if chunk_size <= 0:
        return [text]
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = max(end - overlap, 0)
    return chunks


def extract_sections(pdf_path: str, min_len: int = 50, include_tables: bool = True, 
                    replace_with_tables: bool = False, min_table_rows: int = 2, 
                    min_table_cols: int = 2) -> List[Dict[str, Any]]:
    """Extract sections by page-level, including tables.
    Returns non-empty pages as sections.
    """
    sections = []
    for page in extract_pages(pdf_path, include_tables=include_tables, 
                              replace_with_tables=replace_with_tables,
                              min_table_rows=min_table_rows, 
                              min_table_cols=min_table_cols):
        combined_text = combine_text_and_tables(page, replace_with_tables=replace_with_tables)
        if len(combined_text.strip()) >= min_len:
            page["combined_text"] = combined_text
            sections.append(page)
    return sections


def extract_pdf_chunks(
    pdf_path: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_len: int = 50,
    include_tables: bool = True,
    replace_with_tables: bool = False,
    min_table_rows: int = 2,
    min_table_cols: int = 2,
) -> List[Dict[str, Any]]:
    """Extract pages (with tables) and return chunked contexts with metadata.

    Args:
        pdf_path: path to PDF file
        chunk_size: size of each chunk in characters
        overlap: overlap between chunks
        min_len: minimum text length to include
        include_tables: whether to extract tables
        replace_with_tables: if True, replace table portions in text with formatted tables
        min_table_rows: minimum rows for valid table
        min_table_cols: minimum columns for valid table
        
    Each item has: {"pdf_name", "page_number", "chunk_index", "text", "tables", "metadata"}
    """
    chunks: List[Dict[str, Any]] = []
    pages = extract_pages(pdf_path, include_tables=include_tables, 
                         replace_with_tables=replace_with_tables,
                         min_table_rows=min_table_rows,
                         min_table_cols=min_table_cols)
    
    for page in pages:
        # Get the processed text (tables already replaced/appended in extract_pages)
        combined_text = combine_text_and_tables(page, replace_with_tables=replace_with_tables)
        
        if len(combined_text.strip()) < min_len:
            continue
        
        page_chunks = chunk_text(combined_text, chunk_size=chunk_size, overlap=overlap)
        
        for cidx, ctext in enumerate(page_chunks):
            if not ctext.strip():
                continue
            
            chunk_data = {
                "pdf_name": page["metadata"]["pdf_name"],
                "page_number": page["page_number"],
                "chunk_index": cidx,
                "text": ctext,
                "has_tables": page["has_tables"],
                "replaced_with_tables": page.get("replaced_with_tables", False),
                "metadata": page["metadata"],
            }
            
            # Include table data in first chunk of the page
            if cidx == 0 and page.get("tables"):
                # Remove bbox from tables before saving (not JSON serializable)
                tables_without_bbox = []
                for table in page["tables"]:
                    table_copy = table.copy()
                    if 'bbox' in table_copy:
                        del table_copy['bbox']
                    tables_without_bbox.append(table_copy)
                chunk_data["tables"] = tables_without_bbox
            
            chunks.append(chunk_data)
    
    return chunks


def save_chunks_json(chunks: List[Dict[str, Any]], output_path: str) -> str:
    """Save chunk list to JSON and return the path."""
    safe_write_json(chunks, output_path)
    return output_path


def output_path_for_pdf(pdf_path: str, output_dir: Optional[str] = None) -> Path:
    """Derive default output path for chunk file from pdf name."""
    out_dir = Path(output_dir) if output_dir else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_path).stem
    return out_dir / f"{stem}.chunks.json"


def process_pdf_to_chunks(
    pdf_path: str,
    output_dir: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_len: int = 50,
    include_tables: bool = True,
    replace_with_tables: bool = False,
    min_table_rows: int = 2,
    min_table_cols: int = 2,
) -> str:
    """Process a single PDF into chunk JSON. Returns output JSON path."""
    mode = "replace mode (inline)" if replace_with_tables else "append mode"
    LOGGER.info("Processing PDF to chunks: %s (tables: %s, %s, min_rows=%d, min_cols=%d)", 
                pdf_path, include_tables, mode, min_table_rows, min_table_cols)
    chunks = extract_pdf_chunks(
        pdf_path, 
        chunk_size=chunk_size, 
        overlap=overlap, 
        min_len=min_len,
        include_tables=include_tables,
        replace_with_tables=replace_with_tables,
        min_table_rows=min_table_rows,
        min_table_cols=min_table_cols,
    )
    out_path = output_path_for_pdf(pdf_path, output_dir)
    
    replaced_count = sum(1 for c in chunks if c.get("replaced_with_tables"))
    LOGGER.info("Extracted %d chunks (%d with tables, %d with inline replacements)", 
                len(chunks), 
                sum(1 for c in chunks if c.get("has_tables")),
                replaced_count)
    return save_chunks_json(chunks, str(out_path))


def process_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    pattern: str = "*.pdf",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_len: int = 50,
    include_tables: bool = True,
    replace_with_tables: bool = False,
    min_table_rows: int = 2,
    min_table_cols: int = 2,
) -> List[str]:
    """Process all PDFs in a directory. Returns list of output JSON paths."""
    in_dir = Path(input_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")
    pdfs = sorted(in_dir.glob(pattern))
    if not pdfs:
        LOGGER.warning("No PDFs found in %s matching %s", input_dir, pattern)
        return []
    outputs: List[str] = []
    for pdf in tqdm(pdfs, desc="Processing PDFs"):
        try:
            out = process_pdf_to_chunks(
                str(pdf),
                output_dir=output_dir,
                chunk_size=chunk_size,
                overlap=overlap,
                min_len=min_len,
                include_tables=include_tables,
                replace_with_tables=replace_with_tables,
                min_table_rows=min_table_rows,
                min_table_cols=min_table_cols,
            )
            outputs.append(out)
        except Exception as exc:
            LOGGER.error("Failed processing %s: %s", pdf.name, exc)
    return outputs