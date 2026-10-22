import json
import os
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from difflib import SequenceMatcher

def find_answer_position(answer_text, context, _recursion_depth=0):
    """
    Find the position of answer text in context using multiple strategies.
    Returns the character position where the answer starts, or -1 if not found.
    """
    if not answer_text or not context:
        return -1

    # Prevent infinite recursion
    if _recursion_depth > 1:
        return -1

    # Strategy 1: Exact match
    pos = context.find(answer_text)
    if pos >= 0:
        return pos

    # Strategy 2: Case-insensitive match
    pos = context.lower().find(answer_text.lower())
    if pos >= 0:
        return pos

    # Strategy 3: Normalized whitespace matching
    answer_normalized = re.sub(r'\s+', ' ', answer_text.strip())
    context_normalized = re.sub(r'\s+', ' ', context)
    norm_pos = context_normalized.find(answer_normalized)

    if norm_pos >= 0:
        # Map normalized position back to original text position
        char_count = 0
        for i, char in enumerate(context):
            if char_count == norm_pos:
                return i
            if not char.isspace() or (i > 0 and not context[i-1].isspace()):
                char_count += 1

    # Strategy 4: Case-insensitive normalized whitespace
    answer_norm_lower = answer_normalized.lower()
    context_norm_lower = context_normalized.lower()
    norm_pos = context_norm_lower.find(answer_norm_lower)

    if norm_pos >= 0:
        # Map back to original position
        char_count = 0
        for i, char in enumerate(context):
            if char_count == norm_pos:
                return i
            if not char.isspace() or (i > 0 and not context[i-1].isspace()):
                char_count += 1

    # Strategy 5: Remove punctuation and try again
    def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)

    answer_no_punct = remove_punctuation(answer_text)
    context_no_punct = remove_punctuation(context)

    pos = context_no_punct.lower().find(answer_no_punct.lower())
    if pos >= 0:
        # Map back approximately (may not be exact)
        # Count non-space characters to find position
        char_count = 0
        for i, char in enumerate(context):
            cleaned_char = remove_punctuation(char)
            if cleaned_char and not cleaned_char.isspace():
                if char_count == pos:
                    return i
                char_count += 1

    # Strategy 6: Try finding answer fragments (start, middle, end)
    answer_clean = re.sub(r'\s+', ' ', answer_text.strip())

    # Try progressively smaller fragments from the start
    for fragment_len in [min(len(answer_clean), l) for l in [150, 100, 80, 60, 40, 30, 20]]:
        if fragment_len < 10:
            break

        fragment = answer_clean[:fragment_len]
        fragment_norm = re.sub(r'\s+', ' ', fragment).lower()
        context_norm = re.sub(r'\s+', ' ', context).lower()
        norm_pos = context_norm.find(fragment_norm)

        if norm_pos >= 0:
            # Map back to original position
            char_count = 0
            for i, char in enumerate(context):
                if char_count == norm_pos:
                    return i
                if not char.isspace() or (i > 0 and not context[i-1].isspace()):
                    char_count += 1

    # Strategy 7: Fuzzy matching using SequenceMatcher for similar substrings
    # Find the best matching substring in context
    best_ratio = 0.0
    best_pos = -1
    answer_lower = answer_text.lower()
    context_lower = context.lower()

    # Only do fuzzy matching for shorter answers (< 200 chars) to save time
    if len(answer_text) < 200:
        window_size = len(answer_text)
        for i in range(len(context) - window_size + 1):
            window = context_lower[i:i + window_size]
            ratio = SequenceMatcher(None, answer_lower, window).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i

        # If we found a good match (> 80% similarity), use it
        if best_ratio > 0.8:
            return best_pos

    # Strategy 8: Word-based matching - find position where most answer words appear
    answer_words = answer_normalized.lower().split()
    if len(answer_words) >= 3:
        # Look for sequences of at least 3 consecutive answer words
        for i in range(len(answer_words) - 2):
            phrase = ' '.join(answer_words[i:i+3])
            pos = context_normalized.lower().find(phrase)
            if pos >= 0:
                # Map back to original position
                char_count = 0
                for j, char in enumerate(context):
                    if char_count == pos:
                        return j
                    if not char.isspace() or (j > 0 and not context[j-1].isspace()):
                        char_count += 1

    # Strategy 9: Try first sentence of answer if answer is long
    if len(answer_text) > 100 and _recursion_depth == 0:
        # Split by period, exclamation, or question mark
        sentences = re.split(r'[.!?]+', answer_text)
        if sentences and len(sentences[0].strip()) > 20:
            first_sentence = sentences[0].strip()
            pos = find_answer_position(first_sentence, context, _recursion_depth + 1)
            if pos >= 0:
                return pos

    # All strategies failed
    return -1

def calculate_similarity(text1, text2):
    """Calculate similarity between two texts using TF-IDF and cosine similarity."""
    if not text1 or not text2:
        return 0.0
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        # Fallback to simple comparison if vectorization fails
        return 0.0

def find_matching_json(file_name, json_path):
    """Find the JSON file that matches the given file name."""
    json_path = Path(json_path)
    
    # Try exact match first
    exact_match = json_path / file_name
    if exact_match.exists():
        return exact_match
    
    # Try with .json extension
    if not file_name.endswith('.json'):
        json_match = json_path / f"{file_name}.json"
        if json_match.exists():
            return json_match
    
    # Search for similar filenames
    for json_file in json_path.glob('*.json'):
        if file_name.replace('.json', '') in json_file.stem:
            return json_file
    
    return None

def match_and_add_chunk_index(file_a_path, json_files_path, output_path):
    """
    Match records from File A with JSON files and add chunk_index.
    Always returns the chunk with the best similarity score.
    
    Parameters:
    - file_a_path: Path to File A (JSON with 6 columns)
    - json_files_path: Directory containing JSON files with 4 columns
    - output_path: Path to save the updated File A
    """
    
    # Load File A
    with open(file_a_path, 'r', encoding='utf-8') as f:
        file_a_data = json.load(f)

    # Filter out rows where answer is "-"
    file_a_data = [record for record in file_a_data if record.get('answer', '').strip() != '-']
    
    print(f"Processing {len(file_a_data)} records (rows with answer='-' have been removed)...\n")
    
    # Process each record in File A
    for record in file_a_data:
        file_name = record.get('file', '')
        answer_text = record.get('answer', '')
        
        # Initialize chunk_index, page_number, and context as None
        record['page_number'] = None
        record['chunk_index'] = None
        record['context'] = None
        
        if not file_name or not answer_text:
            continue
        
        # Find matching JSON file
        json_file_path = find_matching_json(file_name, json_files_path)
        
        if json_file_path is None:
            print(f"Warning: No JSON file found for '{file_name}'")
            continue
        
        # Load the matching JSON file
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"Error loading {json_file_path}: {e}")
            continue
        
        # Improved Strategy: First find high-similarity chunks, then check for answer
        chunks_all = []

        # Step 1: Calculate TF-IDF similarity for all chunks
        for chunk in json_data:
            chunk_text = chunk.get('text', '')
            chunks_all.append({
                'chunk_index': chunk.get('chunk_index'),
                'page_number': chunk.get('page_number'),
                'text': chunk_text,
                'similarity': calculate_similarity(answer_text, chunk_text),
                'answer_pos': -1
            })

        # Step 2: Sort by similarity and check top candidates for answer position
        chunks_all.sort(key=lambda x: x['similarity'], reverse=True)

        # Check top 5 candidates (or all if fewer than 5)
        top_candidates = chunks_all[:min(5, len(chunks_all))]
        chunks_with_answer = []

        for chunk_info in top_candidates:
            # Check if this chunk contains the answer
            answer_pos = find_answer_position(answer_text, chunk_info['text'])
            chunk_info['answer_pos'] = answer_pos

            if answer_pos >= 0:
                chunks_with_answer.append(chunk_info)

        # Select best chunk using prioritization
        best_chunk = None

        if chunks_with_answer:
            # Priority 1: Choose chunk with answer and highest similarity
            best_chunk = max(chunks_with_answer, key=lambda x: x['similarity'])
        elif chunks_all:
            # Priority 2: Use highest TF-IDF similarity chunk (already first in sorted list)
            best_chunk = chunks_all[0]

        # Extract best chunk info (including already-calculated answer position)
        if best_chunk:
            best_similarity = best_chunk['similarity']
            best_chunk_index = best_chunk['chunk_index']
            best_page_number = best_chunk['page_number']
            best_chunk_text = best_chunk['text']
            answer_start = best_chunk['answer_pos']  # Already calculated during chunk selection
        else:
            best_similarity = 0
            best_chunk_index = None
            best_page_number = None
            best_chunk_text = None
            answer_start = -1

        # Always add the best matching chunk_index, page_number, and context
        if best_chunk_index is not None:
            record['page_number'] = best_page_number
            record['chunk_index'] = best_chunk_index
            record['context'] = best_chunk_text

            # Convert answer to SQuAD format
            record['answer'] = {
                'text': [answer_text],
                'answer_start': [answer_start]
            }

            print(f"Match found for '{file_name}': chunk_index={best_chunk_index}, page_number={best_page_number}, similarity={best_similarity:.4f}, answer_start={answer_start}")
        else:
            print(f"No chunks found in '{file_name}'")
    
    # Restructure records with id and reordered columns
    output_data = []
    for idx, record in enumerate(file_a_data):
        # Create new record with specified column order
        new_record = {
            'id': idx,
            'report_title': record.get('judul_laporan', ''),
            'file': record.get('file', ''),
            'page_number': record.get('page_number'),
            'chunk_index': record.get('chunk_index'),
            'anthropometric_assessment': record.get('assessment_antropometri', ''),
            'medical_diagnosis': record.get('diagnosis_medis', ''),
            'context': record.get('context'),
            'question': record.get('question', ''),
            'answer': record.get('answer', '')            
        }
        output_data.append(new_record)

    # Save updated File A
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nProcessing complete. Output saved to: {output_path}")

    # Print statistics
    matched_count = sum(1 for r in output_data if r['chunk_index'] is not None)
    print(f"Records matched: {matched_count}/{len(output_data)}")

if __name__ == "__main__":
    FILE_A_PATH = "data/6-xlsx_to_qa_json/qa_pairs.json"
    JSON_FILES_PATH = "data/4-anonymize_using_extracted_ner"
    OUTPUT_PATH = "data/dataset/dataset.json"  # Output file path
    
    match_and_add_chunk_index(
        file_a_path=FILE_A_PATH,
        json_files_path=JSON_FILES_PATH,
        output_path=OUTPUT_PATH
    )