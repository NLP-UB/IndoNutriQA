import pandas as pd
import os
import re
from pathlib import Path

def normalize_text_preserve_plus(text):
    """Normalize text but preserve + symbol to distinguish it from commas"""
    if pd.isna(text):
        return ""
    text = str(text).upper()
    # Remove most punctuation but KEEP + symbol
    # Replace / with space, remove commas, periods, etc., but preserve +
    text = re.sub(r'[/\-_.,;:!?(){}[\]"\'`~@#$%^&*=|\\<>]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def normalize_text_no_space(text):
    """Normalize text and remove all spaces"""
    normalized = normalize_text_preserve_plus(text)
    return normalized.replace(' ', '')

def calculate_match_score(filename_normalized, judul_normalized, filename_original):
    """
    Calculate a match score based on how well the filename matches the judul.
    Higher score = better match.
    Returns (score, match_type) tuple.
    """
    # Check if filename is contained in judul
    if filename_normalized in judul_normalized:
        # Score based on length and position
        score = len(filename_normalized) * 100
        
        # Bonus for matching + symbols (exact punctuation match)
        plus_count_file = filename_normalized.count('+')
        plus_count_judul = judul_normalized.count('+')
        
        # If file has + symbols and they match in judul context, give bonus
        if plus_count_file > 0:
            # Check if the context around + matches
            if filename_normalized in judul_normalized:
                score += plus_count_file * 1000  # Big bonus for + matching
        
        # Penalty if judul has + but filename doesn't (likely wrong match)
        # Extract the portion of judul that matches filename length
        start_idx = judul_normalized.find(filename_normalized)
        if start_idx >= 0:
            end_idx = start_idx + len(filename_normalized)
            # Look for + in nearby context (before/after match)
            context_before = judul_normalized[max(0, start_idx-10):start_idx]
            context_after = judul_normalized[end_idx:min(len(judul_normalized), end_idx+10)]
            
            if '+' in context_before or '+' in context_after:
                if '+' not in filename_normalized:
                    score -= 500  # Penalty for missing + when context has it
        
        return (score, 'with_space')
    
    return (0, 'no_match')

def match_files_with_excel(excel_path, folder_path, output_path):
    """
    Match 'Judul Laporan' column with filenames in a folder.
    """
    
    df = pd.read_excel(excel_path)
    
    if 'Judul Laporan' not in df.columns:
        raise ValueError("Column 'Judul Laporan' not found in the Excel file")
    
    files = os.listdir(folder_path)
    print(f"Found {len(files)} files in folder")
    
    # Prepare filename data
    file_data = []
    for file in files:
        file_base = file
        if file_base.endswith('.json'):
            file_base = file_base[:-5]
        if file_base.endswith('.chunks'):
            file_base = file_base[:-7]
        
        file_normalized = normalize_text_preserve_plus(file_base)
        file_normalized_no_space = normalize_text_no_space(file_base)
        
        file_data.append({
            'original': file,
            'base': file_base,
            'normalized': file_normalized,
            'normalized_no_space': file_normalized_no_space,
            'length': len(file_normalized),
            'has_plus': '+' in file_normalized
        })
    
    # Sort by length descending
    file_data.sort(key=lambda x: x['length'], reverse=True)
    
    print(f"\nSample normalized filenames (first 5):")
    for fd in file_data[:5]:
        print(f"  {fd['original']}")
        print(f"    -> {fd['normalized'][:80]}")
        print(f"    Has +: {fd['has_plus']}")
        print()
    
    def find_match(judul, row_idx):
        if pd.isna(judul):
            return None
        
        judul_normalized = normalize_text_preserve_plus(judul)
        judul_normalized_no_space = normalize_text_no_space(judul)
        
        # Try exact match first
        for fd in file_data:
            if judul_normalized == fd['normalized']:
                return fd['original']
        
        # Find all potential matches with scores
        matches = []
        
        for fd in file_data:
            if len(fd['normalized']) >= 5:
                # Try with-space matching
                score_space, match_type_space = calculate_match_score(
                    fd['normalized'], judul_normalized, fd['original']
                )
                if score_space > 0:
                    matches.append((fd['original'], score_space, match_type_space, fd))
                
                # Try no-space matching
                if fd['normalized_no_space'] in judul_normalized_no_space:
                    score_no_space = len(fd['normalized']) * 50  # Lower score than with-space
                    # Apply same + symbol logic
                    if fd['has_plus'] and '+' in judul_normalized:
                        score_no_space += fd['normalized'].count('+') * 1000
                    matches.append((fd['original'], score_no_space, 'no_space', fd))
        
        # Sort by score descending and pick best match
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            best_match = matches[0]
            
            # Debug output for first few rows
            if row_idx < 5:
                print(f"\n{'='*70}")
                print(f"Row {row_idx}:")
                print(f"  Judul: {str(judul)[:120]}...")
                print(f"  Judul normalized: {judul_normalized[:120]}...")
                print(f"  Best match: {best_match[0]} (score: {best_match[1]}, type: {best_match[2]})")
                if len(matches) > 1:
                    print(f"  Other candidates:")
                    for m in matches[1:4]:  # Show top 3 alternatives
                        print(f"    - {m[0]} (score: {m[1]})")
            
            return best_match[0]
        
        return None
    
    # Apply matching
    df['File'] = [find_match(judul, idx) for idx, judul in enumerate(df['Judul Laporan'])]
    
    # Save results
    df.to_excel(output_path, index=False)
    
    # Print summary
    matched = df['File'].notna().sum()
    total = len(df)
    print(f"\n{'='*70}")
    print(f"Matching complete!")
    print(f"Matched: {matched}/{total} items ({matched/total*100:.1f}%)")
    print(f"Output saved to: {output_path}")
    
    # Show unmatched
    unmatched = df[df['File'].isna()]
    if len(unmatched) > 0:
        print(f"\nUnmatched items: {len(unmatched)}")
        print("First 5 unmatched:")
        for idx, judul in enumerate(unmatched['Judul Laporan'].head(5)):
            print(f"  {idx+1}. {str(judul)[:100]}...")
    
    return df

if __name__ == "__main__":
    # Set your file paths here
    excel_file = "data/5-match_qna_to_chunks/QnA.xlsx"  # Your input Excel file
    folder = "data/2-remove_pre_pendahuluan"  # Folder containing files to match
    output_file = "data/5-match_qna_to_chunks/QnA.with_matches.xlsx"  # Output Excel file
    
    # Run the matching
    result_df = match_files_with_excel(excel_file, folder, output_file)
    
    # Display results
    print("\nPreview of results:")
    print(result_df[['Judul Laporan', 'File']].head(10))