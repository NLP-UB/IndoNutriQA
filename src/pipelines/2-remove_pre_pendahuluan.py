#!/usr/bin/env python3
"""
Remove chunks that appear before pages containing 'BAB 1\nPENDAHULUAN' in JSON files
within data/1-run_extraction directory.

This script:
1. Scans all .chunks.json files in data/1-run_extraction/
2. For each file, finds the first page that contains "PENDAHULUAN"
3. Removes all chunks with page_number less than that page
4. Saves the filtered chunks back to the same file
"""

import json
import glob
from pathlib import Path
from typing import List, Dict, Any

def find_pendahuluan_page(chunks: List[Dict[str, Any]]) -> int:
    """
    Find the first page number whose text STARTS with one of the target patterns.
    Returns the page number, or -1 if not found.
    
    Checks for patterns at the start of text (not in the middle):
    - 'BAB I\nPENDAHULUAN\n', 'BAB 1\nPENDAHULUAN\n'
    - 'BABI\nPENDAHULUAN\n', 'BAB I. PENDAHULUAN\n', 'BAB 1. PENDAHULUAN\n'
    - 'BAB I\n', 'BAB 1\n'
    - 'BAB 1\nPENDAHULUAN', 'PENDAHULUAN\n', 'SKRINING GIZI\n'
    """
    patterns = [
        'BAB I\nPENDAHULUAN\n',
        'BAB 1\nPENDAHULUAN\n',
        'BABI\nPENDAHULUAN\n',
        'BAB I. PENDAHULUAN\n',
        'BAB 1. PENDAHULUAN\n',
        'BAB I\n',
        'BAB 1\nPENDAHULUAN',
        'BAB 1\n',
        'PENDAHULUAN\n',
        'A. IDENTITAS PASIEN\n',
        'SKRINING GIZI\n',
        'DATA DASAR\n'
    ]
    
    for chunk in chunks:
        text = chunk.get('text', '').upper()
        # Check if text starts with any of the patterns
        for pattern in patterns:
            if text.startswith(pattern):
                return int(chunk.get('page_number', -1))
    return -1

def remove_chunks_before_pendahuluan(chunks: List[Dict[str, Any]]) -> tuple:
    """
    Remove chunks that appear before PENDAHULUAN page.
    Returns (filtered_chunks, removed_count, pendahuluan_page).
    """
    pendahuluan_page = find_pendahuluan_page(chunks)
    
    if pendahuluan_page == -1:
        print(f"'PENDAHULUAN' not found in this file")
        return chunks, 0, -1
    
    # Filter out chunks before PENDAHULUAN page
    filtered = [c for c in chunks if c.get('page_number', 0) >= pendahuluan_page]
    removed = len(chunks) - len(filtered)
    
    return filtered, removed, pendahuluan_page

def process_file(filepath: str, dry_run: bool = False) -> dict:
    """
    Process a single .chunks.json file.
    Returns stats dict with file info and changes made.
    """
    try:
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        if not isinstance(chunks, list):
            return {
                'file': Path(filepath).name,
                'status': 'SKIP',
                'reason': 'Not a list structure',
                'chunks_before': len(chunks) if isinstance(chunks, list) else 'N/A',
                'chunks_after': 'N/A',
                'removed': 0,
                'pendahuluan_page': -1
            }
        
        original_count = len(chunks)
        filtered_chunks, removed_count, pendahuluan_page = remove_chunks_before_pendahuluan(chunks)
        
        if removed_count == 0:
            return {
                'file': Path(filepath).name,
                'status': 'NO_CHANGE',
                'reason': 'No chunks before PENDAHULUAN',
                'chunks_before': original_count,
                'chunks_after': len(filtered_chunks),
                'removed': 0,
                'pendahuluan_page': pendahuluan_page
            }
        
        # Write to data/output if not dry run
        output_dir = Path('data/2-remove_pre_pendahuluan')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / Path(filepath).name
        if not dry_run:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_chunks, f, ensure_ascii=False, indent=2)
        
        return {
            'file': Path(filepath).name,
            'output': str(output_path),
            'status': 'UPDATED' if not dry_run else 'WOULD_UPDATE',
            'chunks_before': original_count,
            'chunks_after': len(filtered_chunks),
            'removed': removed_count,
            'pendahuluan_page': pendahuluan_page
        }
    
    except json.JSONDecodeError as e:
        return {
            'file': Path(filepath).name,
            'status': 'ERROR',
            'reason': f'JSON decode error: {e}',
            'chunks_before': 'N/A',
            'chunks_after': 'N/A',
            'removed': 0,
            'pendahuluan_page': -1
        }
    except Exception as e:
        return {
            'file': Path(filepath).name,
            'status': 'ERROR',
            'reason': str(e),
            'chunks_before': 'N/A',
            'chunks_after': 'N/A',
            'removed': 0,
            'pendahuluan_page': -1
        }

def main():
    import sys
    
    # Check for dry-run flag
    dry_run = '--dry-run' in sys.argv or '-d' in sys.argv
    
    # Find all .chunks.json files
    data_dir = Path('data/1-run_extraction')
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return
    
    chunk_files = sorted(data_dir.glob('*.chunks.json'))
    
    if not chunk_files:
        print(f"No .chunks.json files found in {data_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Processing {len(chunk_files)} chunk files...")
    print(f"Dry run: {dry_run}")
    print(f"{'='*80}\n")
    
    results = []
    total_removed = 0
    
    for filepath in chunk_files:
        result = process_file(str(filepath), dry_run=dry_run)
        results.append(result)
        total_removed += result.get('removed', 0)
        
        # Print status
        status = result['status']
        file_short = result['file'][:50] + '...' if len(result['file']) > 50 else result['file']
        
        if status == 'UPDATED' or status == 'WOULD_UPDATE':
            symbol = '✓' if status == 'UPDATED' else '→'
            print(f"{symbol} {file_short}")
            print(f"  Pages removed: {result['removed']} | PENDAHULUAN starts page: {result['pendahuluan_page']}")
            print(f"  Before: {result['chunks_before']} chunks | After: {result['chunks_after']} chunks")
        elif status == 'NO_CHANGE':
            print(f"{file_short} (no changes needed)")
        elif status == 'SKIP':
            print(f"{file_short} (skipped: {result['reason']})")
        elif status == 'ERROR':
            print(f"{file_short} (error: {result['reason']})")
        print()
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    updated = sum(1 for r in results if r['status'] in ['UPDATED', 'WOULD_UPDATE'])
    skipped = sum(1 for r in results if r['status'] == 'SKIP')
    errors = sum(1 for r in results if r['status'] == 'ERROR')
    no_change = sum(1 for r in results if r['status'] == 'NO_CHANGE')
    
    print(f"Total files: {len(results)}")
    print(f"  Updated: {updated}")
    print(f"  No changes: {no_change}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print(f"\nTotal chunks removed: {total_removed}")
    
    if dry_run:
        print(f"\nRun without --dry-run to apply changes")
    else:
        print(f"\nProcessing complete!")

if __name__ == '__main__':
    main()

# python src/pipelines/2-remove_pre_pendahuluan.py --dry-run 2>&1 | head -80
# python src/pipelines/2-remove_pre_pendahuluan.py