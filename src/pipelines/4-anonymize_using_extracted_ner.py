#!/usr/bin/env python3
"""
Anonymize clinical case chunks using extracted NER entities and output only selected fields.

This script:
1. Reads extracted_ner.json containing medical entities (names, dates, IDs, addresses) grouped by chunk index
2. For each .chunks.json file in data/2-remove_pre_pendahuluan/, replaces extracted entities (applied per chunk using chunk_index AND page_number) with placeholders
3. Saves anonymized chunks to data/4-anonymize_using_extracted_ner/ with ONLY pdf_name, page_number, chunk_index, and text fields
4. Tracks anonymization statistics (entities replaced, files processed)

Mapping of NER entities to Indonesian placeholders:
- name -> <NAMA>
- tanggal_lahir -> <TGL>
- no_rm -> <NO_RM>
- no_mrs -> <NO_MRS>
- address -> <LOKASI>
- no_registrasi -> <NO_REG>
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


# Mapping from NER entity types to placeholder replacements
NER_TO_PLACEHOLDER = {
    "name": "<NAMA>",
    "tanggal_lahir": "<TGL>",
    "no_rm": "<NO_RM>",
    "no_mrs": "<NO_MRS>",
    "address": "<LOKASI>",
    "no_registrasi": "<NO_REG>",
}


def escape_regex_special_chars(text: str) -> str:
    """Escape regex special characters in text for safe replacement."""
    return re.escape(text)


def replace_entity_in_text(
    text: str, 
    entity_value: str, 
    placeholder: str, 
    entity_type: str = "name", 
    case_insensitive: bool = True
) -> Tuple[str, int]:
    """
    Replace an entity value with placeholder in text, handling variations.
    
    Args:
        text: Input text
        entity_value: Entity value to replace (e.g., "An. K")
        placeholder: Replacement placeholder (e.g., "<NAMA>")
        entity_type: Type of entity for smart matching (name, no_mrs, etc.)
        case_insensitive: Whether to perform case-insensitive replacement
    
    Returns:
        Tuple of (modified_text, replacement_count)
    """
    if not entity_value or not entity_value.strip():
        return text, 0
    
    # Escape special regex characters
    entity_value = entity_value.strip()
    escaped_entity = escape_regex_special_chars(entity_value)
    
    # For numeric entities like no_mrs, add word boundary to avoid partial matches
    # e.g., "2" in "2020" should not be replaced
    if entity_type in ["no_mrs", "no_rm", "no_registrasi"] and entity_value.isdigit():
        # Use word boundary for pure numeric IDs
        pattern = rf"\b{escaped_entity}\b"
    else:
        # For names and dates, use regular pattern
        pattern = escaped_entity
    
    flags = re.IGNORECASE if case_insensitive else 0
    
    # Count replacements
    matches = re.finditer(pattern, text, flags)
    count = len(list(matches))
    
    # Perform replacement
    modified_text = re.sub(pattern, placeholder, text, flags=flags)
    
    return modified_text, count


def load_extracted_ner(ner_json_path: Path) -> Dict[str, Dict[Tuple[int, int], Dict[str, List[str]]]]:
    """
    Load extracted NER JSON and create a mapping from filename to entities grouped by (page_number, chunk_index).

    The returned structure is:
        {
            "some_file.chunks.json": {
                (5, 0): {"name": ["An. K"], "address": ["Probolinggo"]},
                (5, 1): {"name": [...], ...},
            },
            ...
        }
    This lets anonymization operate on a per-chunk basis using both page_number and chunk_index
    to match the exact chunk from the NER extraction step.
    """
    with ner_json_path.open("r", encoding="utf-8") as f:
        ner_data = json.load(f)
    
    # Create filename -> ((page_number, chunk_index) -> entity_type -> [values]) mapping
    ner_map: Dict[str, Dict[Tuple[int, int], Dict[str, List[str]]]] = {}

    for entry in ner_data:
        # Only include successful entries
        if entry.get("status") != "SUCCESS":
            continue

        filename = entry.get("file")
        if filename:
            filename = filename.strip()
        entities = entry.get("entities", {}) or {}

        per_chunk: Dict[Tuple[int, int], Dict[str, List[str]]] = {}

        for entity_type, items in entities.items():
            # In the extraction output, each entity type maps to a list of dicts with keys: value, chunk_index, page_number
            for item in items:
                # item may be a dict or a simple string (be defensive)
                if isinstance(item, dict):
                    value = item.get("value")
                    chunk_index = item.get("chunk_index", 0)
                    page_number = item.get("page_number", 0)
                else:
                    value = item
                    chunk_index = 0
                    page_number = 0

                try:
                    chunk_index = int(chunk_index)
                except Exception:
                    chunk_index = 0
                
                try:
                    page_number = int(page_number)
                except Exception:
                    page_number = 0

                if not value:
                    continue

                # Use (page_number, chunk_index) as the key
                chunk_key = (page_number, chunk_index)
                per_chunk.setdefault(chunk_key, {}).setdefault(entity_type, []).append(value)

        ner_map[filename] = per_chunk

    return ner_map


def anonymize_chunk(
    chunk: Dict[str, Any], 
    entities: Dict[str, List[str]]
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Anonymize a single chunk by replacing extracted entities.
    Returns only pdf_name, page_number, chunk_index, and text fields.
    
    Args:
        chunk: Single chunk dict with "text" key
        entities: Dict of entity_type -> entity_value mappings
    
    Returns:
        Tuple of (anonymized_chunk, replacement_stats)
    """
    original_text = chunk.get("text", "")
    modified_text = original_text
    replacement_stats = defaultdict(int)
    
    # Replace each entity
    # Process ID numbers first to avoid conflicts (e.g., no_mrs should not replace partial matches in text)
    priority_order = ["no_rm", "no_mrs", "no_registrasi", "name", "tanggal_lahir", "address"]
    
    for entity_type in priority_order:
        if entity_type in entities and entities[entity_type] and entity_type in NER_TO_PLACEHOLDER:
            placeholder = NER_TO_PLACEHOLDER[entity_type]
            # Entities may be a list of values for this type; replace each occurrence
            for entity_value in entities[entity_type]:
                modified_text, count = replace_entity_in_text(
                    modified_text,
                    entity_value,
                    placeholder,
                    entity_type=entity_type,
                )
                if count > 0:
                    replacement_stats[entity_type] += count
    
    # Handle any remaining entity types not in priority order
    for entity_type, values in entities.items():
        if entity_type in priority_order:
            continue
        if entity_type in NER_TO_PLACEHOLDER and values:
            placeholder = NER_TO_PLACEHOLDER[entity_type]
            for entity_value in values:
                modified_text, count = replace_entity_in_text(
                    modified_text,
                    entity_value,
                    placeholder,
                    entity_type=entity_type,
                )
                if count > 0:
                    replacement_stats[entity_type] += count
    
    # Return only the selected fields
    anonymized_chunk = {
        "pdf_name": chunk.get("pdf_name", ""),
        "page_number": chunk.get("page_number", 0),
        "chunk_index": chunk.get("chunk_index", 0),
        "text": modified_text
    }
    
    return anonymized_chunk, dict(replacement_stats)


def anonymize_chunks_file(
    input_path: Path,
    output_path: Path,
    entities_by_chunk: Dict[Tuple[int, int], Dict[str, List[str]]],
) -> Dict[str, Any]:
    """
    Anonymize all chunks in a file and output only selected fields.
    
    Args:
        input_path: Path to input .chunks.json file
        output_path: Path to output anonymized file
        entities_by_chunk: Dict of (page_number, chunk_index) -> entities for this file
    
    Returns:
        Stats dict with anonymization results
    """
    try:
        # Load chunks
        with input_path.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        if not isinstance(chunks, list):
            return {
                "file": input_path.name,
                "status": "ERROR",
                "reason": "Not a list structure",
                "chunks_processed": 0,
                "total_replacements": 0,
            }
        
        if len(chunks) == 0:
            return {
                "file": input_path.name,
                "status": "SKIP",
                "reason": "Empty chunk list",
                "chunks_processed": 0,
                "total_replacements": 0,
            }
        
        # Anonymize chunks (apply entities per (page_number, chunk_index) key)
        anonymized_chunks = []
        total_replacements = 0
        entity_type_counts = defaultdict(int)
        found_entities: Dict[str, List[str]] = {}
        matched_chunks = 0
        unmatched_chunks = 0

        for chunk in chunks:
            # Extract page_number and chunk_index from the chunk
            try:
                page_number = int(chunk.get("page_number", 0))
            except Exception:
                page_number = 0
            
            try:
                chunk_index = int(chunk.get("chunk_index", 0))
            except Exception:
                chunk_index = 0

            # Create the lookup key
            chunk_key = (page_number, chunk_index)
            
            # Get entities for this specific chunk
            chunk_entities = entities_by_chunk.get(chunk_key, {}) if entities_by_chunk else {}

            if chunk_entities:
                matched_chunks += 1
                # Track found entities for reporting
                for etype, values in chunk_entities.items():
                    found_entities.setdefault(etype, []).extend(values)
            else:
                unmatched_chunks += 1

            anon_chunk, stats = anonymize_chunk(chunk, chunk_entities)
            anonymized_chunks.append(anon_chunk)
            total_replacements += sum(stats.values())
            for entity_type, count in stats.items():
                entity_type_counts[entity_type] += count
        
        # Save anonymized chunks
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(anonymized_chunks, f, ensure_ascii=False, indent=2)
        
        # Deduplicate found entities
        entities_found_unique = {k: sorted(set(v)) for k, v in found_entities.items()}

        return {
            "file": input_path.name,
            "status": "SUCCESS",
            "output": str(output_path),
            "chunks_processed": len(anonymized_chunks),
            "chunks_with_entities": matched_chunks,
            "chunks_without_entities": unmatched_chunks,
            "total_replacements": total_replacements,
            "entity_replacements": dict(entity_type_counts),
            "entities_found": entities_found_unique,
        }
    
    except json.JSONDecodeError as e:
        return {
            "file": input_path.name,
            "status": "ERROR",
            "reason": f"JSON decode error: {e}",
            "chunks_processed": 0,
            "total_replacements": 0,
        }
    except Exception as e:
        return {
            "file": input_path.name,
            "status": "ERROR",
            "reason": str(e),
            "chunks_processed": 0,
            "total_replacements": 0,
        }


def process_all_chunks(
    input_dir: Path,
    output_dir: Path,
    ner_map: Dict[str, Dict[Tuple[int, int], Dict[str, List[str]]]],
) -> List[Dict[str, Any]]:
    """
    Process all .chunks.json files in input directory.
    
    Args:
        input_dir: Directory containing .chunks.json files
        output_dir: Directory to save anonymized files
        ner_map: Mapping from filename to (page_number, chunk_index) -> entities
    
    Returns:
        List of result dicts
    """
    chunk_files = sorted(input_dir.glob("*.chunks.json"))
    results = []
    
    for chunk_file in chunk_files:
        entities = ner_map.get(chunk_file.name, {})
        output_file = output_dir / chunk_file.name
        
        result = anonymize_chunks_file(chunk_file, output_file, entities)
        results.append(result)
    
    return results


def print_results_summary(results: List[Dict[str, Any]]) -> None:
    """Print formatted summary of anonymization results."""
    print("\n" + "=" * 80)
    print("ANONYMIZATION SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r.get("status") == "SUCCESS"]
    skipped = [r for r in results if r.get("status") == "SKIP"]
    errors = [r for r in results if r.get("status") == "ERROR"]
    
    print(f"\nTotal files processed: {len(results)}")
    print(f"Successfully anonymized: {len(successful)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Errors: {len(errors)}")
    
    # Chunk matching statistics
    total_chunks = sum(r.get("chunks_processed", 0) for r in successful)
    matched_chunks = sum(r.get("chunks_with_entities", 0) for r in successful)
    unmatched_chunks = sum(r.get("chunks_without_entities", 0) for r in successful)
    
    print(f"\nTotal chunks processed: {total_chunks}")
    print(f"Chunks with entities matched: {matched_chunks}")
    print(f"Chunks without entities: {unmatched_chunks}")
    
    # Entity replacement statistics
    total_replacements = sum(r.get("total_replacements", 0) for r in successful)
    print(f"\nTotal entity replacements: {total_replacements}")
    
    # Aggregate entity-type statistics
    entity_type_totals = defaultdict(int)
    for result in successful:
        for entity_type, count in result.get("entity_replacements", {}).items():
            entity_type_totals[entity_type] += count
    
    if entity_type_totals:
        print("\nReplacements by entity type:")
        for entity_type, count in sorted(entity_type_totals.items()):
            print(f"  {entity_type}: {count}")
    
    # Show sample results
    if successful:
        print("\n" + "-" * 80)
        print("Sample Results (first 5):")
        print("-" * 80)
        for result in successful[:5]:
            print(f"\nFile: {result['file']}")
            print(f"  Chunks processed: {result['chunks_processed']}")
            print(f"  Chunks with entities: {result.get('chunks_with_entities', 0)}")
            print(f"  Total replacements: {result['total_replacements']}")
            if result.get("entity_replacements"):
                print(f"  Entity replacements: {result['entity_replacements']}")
            if result.get("entities_found"):
                print(f"  Unique entities found: {result['entities_found']}")
    
    # Show errors if any
    if errors:
        print("\n" + "-" * 80)
        print("Errors:")
        print("-" * 80)
        for result in errors[:5]:
            print(f"\nFile: {result['file']}")
            print(f"  Error: {result.get('reason', 'Unknown error')}")


def save_results_summary(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Save anonymization results summary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anonymize clinical case chunks using extracted NER entities and output only selected fields"
    )
    parser.add_argument(
        "--ner-file",
        "-n",
        default="data/3-extract_medical_ner/extracted_ner.json",
        help="Path to extracted_ner.json file",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        default="data/2-remove_pre_pendahuluan",
        help="Directory containing .chunks.json files to anonymize",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/4-anonymize_using_extracted_ner",
        help="Directory to save anonymized chunks",
    )
    parser.add_argument(
        "--results",
        "-r",
        default="data/anonymization_results.json",
        help="File to save anonymization results summary",
    )
    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help="Print summary to console",
    )
    args = parser.parse_args()
    
    # Validate paths
    ner_file = Path(args.ner_file)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    results_file = Path(args.results)
    
    if not ner_file.exists():
        print(f"Error: NER file not found: {ner_file}")
        return
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Load NER data
    print(f"Loading extracted NER from: {ner_file}")
    ner_map = load_extracted_ner(ner_file)
    print(f"Loaded {len(ner_map)} files with extracted entities")
    
    # Find chunk files
    chunk_files = list(input_dir.glob("*.chunks.json"))
    if not chunk_files:
        print(f"No .chunks.json files found in {input_dir}")
        return
    
    print(f"\nProcessing {len(chunk_files)} chunk files...")
    print(f"Output directory: {output_dir}\n")
    
    # Process all chunks
    results = process_all_chunks(input_dir, output_dir, ner_map)
    
    # Save results
    save_results_summary(results, results_file)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary if requested
    if args.summary:
        print_results_summary(results)


if __name__ == "__main__":
    main()


# python src/pipelines/4-anonymize_using_extracted_ner.py