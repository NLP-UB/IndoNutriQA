#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Add parent directory to path to allow imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from utils.pdf_processor import process_pdf_to_chunks, process_directory

def main():
    parser = argparse.ArgumentParser(description='Extract and chunk PDF documents (including tables)')
    parser.add_argument('input', help='PDF file or directory path')
    parser.add_argument('-o', '--output', default='data/1-run_extraction', 
                       help='Output directory (default: data/1-run_extraction)')
    parser.add_argument('-c', '--chunk-size', type=int, default=1000,
                       help='Chunk size in characters (default: 1000)')
    parser.add_argument('--overlap', type=int, default=200,
                       help='Chunk overlap (default: 200)')
    parser.add_argument('--min-len', type=int, default=50,
                       help='Minimum text length (default: 50)')
    parser.add_argument('--replace-with-tables', action='store_true',
                       help='Replace text content with table content when tables exist')
    parser.add_argument('--no-tables', action='store_true',
                       help='Disable table extraction (default: tables are extracted)')
    parser.add_argument('--min-table-rows', type=int, default=1,
                       help='Minimum rows for valid table (default: 2)')
    parser.add_argument('--min-table-cols', type=int, default=1,
                       help='Minimum columns for valid table (default: 2)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    include_tables = not args.no_tables
    replace_with_tables = args.replace_with_tables
    
    # Validate: can't replace with tables if tables are disabled
    if replace_with_tables and not include_tables:
        print("Warning: --replace-with-tables requires table extraction. Enabling table extraction.")
        include_tables = True
    
    if input_path.is_file() and input_path.suffix == '.pdf':
        # Process single PDF
        output = process_pdf_to_chunks(
            str(input_path),
            output_dir=args.output,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            min_len=args.min_len,
            include_tables=include_tables,
            replace_with_tables=replace_with_tables,
            min_table_rows=args.min_table_rows,
            min_table_cols=args.min_table_cols,
        )
        print(f"✓ Processed: {output}")
        if replace_with_tables:
            print("✓ Mode: Tables replaced inline (preserving non-table text)")
        elif include_tables:
            print("✓ Mode: Tables appended to text")
        print(f"✓ Table validation: min {args.min_table_rows} rows, {args.min_table_cols} cols")
    elif input_path.is_dir():
        # Process directory
        outputs = process_directory(
            str(input_path),
            output_dir=args.output,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            min_len=args.min_len,
            include_tables=include_tables,
            replace_with_tables=replace_with_tables,
            min_table_rows=args.min_table_rows,
            min_table_cols=args.min_table_cols,
        )
        print(f"✓ Processed {len(outputs)} files")
        if replace_with_tables:
            print("✓ Mode: Tables replaced inline (preserving non-table text)")
        elif include_tables:
            print("✓ Mode: Tables appended to text")
        print(f"✓ Table validation: min {args.min_table_rows} rows, {args.min_table_cols} cols")
    else:
        print(f"Error: {args.input} is not a valid PDF or directory")

if __name__ == "__main__":
    main()

# Replace text with tables (inline replacement, preserving non-table text)
# python src/pipelines/1-run_extraction.py data/0-raw/ -o data/1-run_extraction --replace-with-tables

# Usage examples:
# Default: Extract tables and append to text (min 2 rows, 2 cols)
# python src/pipelines/1-run_extraction.py data/0-raw/ -o data/1-run_extraction

# With stricter table validation (min 3 rows, 3 cols)
# python src/pipelines/1-run_extraction.py data/0-raw/ -o data/1-run_extraction --min-table-rows 3 --min-table-cols 3

# Process single file with custom table validation
# python src/pipelines/1-run_extraction.py data/raw/document.pdf --min-table-rows 4 --min-table-cols 2

# Skip table extraction
# python src/pipelines/1-run_extraction.py data/raw/ --no-tables