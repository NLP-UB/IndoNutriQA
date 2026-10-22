import pandas as pd
import json

def xlsx_to_qa_json(xlsx_file, output_file):
    """
    Convert XLSX file to QA pairs JSON format.
    
    Args:
        xlsx_file: Path to input XLSX file
        output_file: Path to output JSON file
    """
    # Read the Excel file
    df = pd.read_excel(xlsx_file)
    
    # Define the question columns
    question_columns = [
        'Apa metode skrining status gizi yang direkomendasikan?',
        'Bagaimana status gizi pada pasien ini?',
        'Bagaimana data biokimia/parameter lab pada pasien (yg abnormal)?',
        'Bagaimana menyusun diagnosis gizi dengan format PES pada pasien ini?',
        'Apa tujuan diet pada pasien ini?',
        'Bagaimana prinsip diet yang tepat sesuai dengan kondisi pasien ini?',
        'Bagaimana syarat diet yang harus dipenuhi dalam pengaturan diet pasien ini?',
        'Apa pedoman diet yang digunakan dalam perencanaan diet pasien ini?'
    ]
    
    # List to store all QA pairs
    qa_pairs = []
    
    # Process each row in the dataframe
    for _, row in df.iterrows():
        judul_laporan = row['Judul Laporan']
        assessment = row['Assessment Antropometri (BB, TB, Umur, Jenis Kelamin, dll)']
        diagnosis = row['Diagnosis Medis']
        file_name = row['File']
        
        # Create QA pair for each question that has an answer
        for question_col in question_columns:
            answer = row[question_col]
            
            # Skip if answer is empty or NaN
            if pd.isna(answer) or str(answer).strip() == '':
                continue
            
            # Create QA pair object
            qa_pair = {
                'judul_laporan': judul_laporan,
                'file': file_name,
                'assessment_antropometri': assessment,
                'diagnosis_medis': diagnosis,
                'question': question_col,
                'answer': answer
            }
            
            qa_pairs.append(qa_pair)
    
    # Write to JSON file as array of objects
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)
        
    
    print(f"Successfully converted {len(qa_pairs)} QA pairs to {output_file}")
    return qa_pairs


# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    input_file = "data/5-match_qna_to_chunks/QnA.with_matches.xlsx"
    output_file = "data/6-xlsx_to_qa_json/qa_pairs.json"
    
    try:
        qa_pairs = xlsx_to_qa_json(input_file, output_file)
        print(f"\nTotal QA pairs created: {len(qa_pairs)}")
        
        # Display first QA pair as example
        if qa_pairs:
            print("\nExample QA pair:")
            print(json.dumps(qa_pairs[0], indent=2, ensure_ascii=False))
            
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"Error: {e}")