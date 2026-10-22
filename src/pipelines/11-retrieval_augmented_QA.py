"""
Retrieval-Augmented Question Answering (RAG) System

This module implements a RAG pipeline for Indonesian clinical nutrition QA:
1. Embed all context chunks using Sentence-BERT
2. Retrieve top-k most relevant chunks for each question
3. Generate answer using the retrieved contexts

Pipeline:
- Dense Retriever: Sentence-BERT (paraphrase-multilingual-MiniLM-L12-v2)
- Generator: mT5 for Indonesian text generation

This demonstrates dataset scalability by retrieving from ALL chunks,
not just the pre-matched ones.
"""

import json
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from tqdm import tqdm
import os

# ============================================================================
# Configuration
# ============================================================================

# Paths
CHUNKS_DIR = "data/4-anonymize_using_extracted_ner"
QA_PAIRS_FILE = "data/dataset/dataset.json"
OUTPUT_DIR = "data/11-retrieval_augmented_QA"
EMBEDDINGS_CACHE = os.path.join(OUTPUT_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_CACHE = os.path.join(OUTPUT_DIR, "chunk_metadata.json")

# Model configuration
RETRIEVER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GENERATOR_MODEL = "google/mt5-base"

# Retrieval configuration
TOP_K_CHUNKS = 5  # Retrieve top 5 most relevant chunks
MAX_CONTEXT_LENGTH = 512  # Maximum tokens for combined context

# Generation configuration
MAX_ANSWER_LENGTH = 150
NUM_BEAMS = 4

# ============================================================================
# Step 1: Load and Embed All Context Chunks
# ============================================================================

def load_all_chunks(chunks_dir):
    """
    Load all chunks from anonymized JSON files.
    Returns list of chunks with metadata.
    """
    chunks_dir = Path(chunks_dir)
    all_chunks = []

    print("Loading chunks from JSON files...")
    json_files = list(chunks_dir.glob("*.json"))

    for json_file in tqdm(json_files, desc="Loading chunk files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            for chunk in chunks:
                all_chunks.append({
                    'file': json_file.name,
                    'page_number': chunk.get('page_number'),
                    'chunk_index': chunk.get('chunk_index'),
                    'text': chunk.get('text', '')
                })
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")

    print(f"Loaded {len(all_chunks)} chunks from {len(json_files)} files")
    return all_chunks


def embed_chunks(chunks, retriever_model, cache_path=None, metadata_cache=None):
    """
    Embed all chunks using Sentence-BERT.
    Cache embeddings to disk for faster reuse.
    """
    # Check if embeddings are cached
    if cache_path and os.path.exists(cache_path) and metadata_cache and os.path.exists(metadata_cache):
        print(f"Loading cached embeddings from {cache_path}...")
        embeddings = np.load(cache_path)
        with open(metadata_cache, 'r', encoding='utf-8') as f:
            cached_chunks = json.load(f)
        print(f"Loaded {len(embeddings)} cached embeddings")
        return embeddings, cached_chunks

    print("Embedding chunks (this may take a while)...")
    chunk_texts = [chunk['text'] for chunk in chunks]

    # Embed in batches for efficiency
    batch_size = 32
    embeddings = retriever_model.encode(
        chunk_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=False
    )

    # Cache embeddings
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"Embeddings cached to {cache_path}")

    if metadata_cache:
        with open(metadata_cache, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Chunk metadata cached to {metadata_cache}")

    return embeddings, chunks


# ============================================================================
# Step 2: Retrieve Top-K Relevant Chunks
# ============================================================================

def retrieve_relevant_chunks(question, chunk_embeddings, chunks, retriever_model, top_k=5):
    """
    Retrieve top-k most relevant chunks for a question.
    Uses cosine similarity between question and chunk embeddings.
    """
    # Embed the question
    question_embedding = retriever_model.encode(question, convert_to_tensor=True)

    # Convert chunk embeddings to tensor on same device as question embedding
    device = question_embedding.device
    chunk_embeddings_tensor = torch.tensor(chunk_embeddings, device=device)

    # Calculate cosine similarity
    similarities = util.cos_sim(question_embedding, chunk_embeddings_tensor)[0]

    # Get top-k indices
    top_k_indices = torch.topk(similarities, k=min(top_k, len(similarities))).indices.tolist()

    # Retrieve chunks with scores
    retrieved_chunks = []
    for idx in top_k_indices:
        retrieved_chunks.append({
            'chunk': chunks[idx],
            'score': float(similarities[idx])
        })

    return retrieved_chunks


def combine_retrieved_contexts(retrieved_chunks, max_length=512):
    """
    Combine retrieved chunks into a single context.
    Truncate if necessary to fit max_length.
    """
    combined_text = ""

    for i, item in enumerate(retrieved_chunks):
        chunk_text = item['chunk']['text']
        score = item['score']

        # Add chunk with metadata
        chunk_header = f"\n[Chunk {i+1} - Score: {score:.3f}]\n"
        chunk_content = chunk_text.strip() + "\n"

        # Check if adding this chunk exceeds max_length (rough estimation)
        if len(combined_text) + len(chunk_header) + len(chunk_content) > max_length * 4:
            break

        combined_text += chunk_header + chunk_content

    return combined_text.strip()


# ============================================================================
# Step 3: Generate Answer
# ============================================================================

class RAGGenerator:
    """Generator using mT5 for answer generation."""

    def __init__(self, model_name="google/mt5-base"):
        print(f"Loading generator model: {model_name}...")
        self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Generator loaded on {self.device}")

    def generate_answer(self, question, context, max_length=150, num_beams=4):
        """
        Generate answer given question and retrieved context.
        """
        # Format input for mT5
        input_text = f"question: {question}\ncontext: {context}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


# ============================================================================
# Main RAG Pipeline
# ============================================================================

def run_rag_pipeline(qa_pairs, chunk_embeddings, chunks, retriever_model, generator, top_k=5):
    """
    Run the complete RAG pipeline:
    1. For each question, retrieve top-k chunks
    2. Combine retrieved contexts
    3. Generate answer
    """
    results = []

    print(f"\nRunning RAG pipeline on {len(qa_pairs)} questions...")
    print(f"Retrieving top-{top_k} chunks per question\n")

    for i, qa_pair in enumerate(tqdm(qa_pairs, desc="RAG Pipeline")):
        question = qa_pair['question']
        ground_truth = qa_pair['answer_text']

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = retrieve_relevant_chunks(
            question,
            chunk_embeddings,
            chunks,
            retriever_model,
            top_k=top_k
        )

        # Step 2: Combine contexts
        combined_context = combine_retrieved_contexts(retrieved_chunks, MAX_CONTEXT_LENGTH)

        # Step 3: Generate answer
        generated_answer = generator.generate_answer(
            question,
            combined_context,
            max_length=MAX_ANSWER_LENGTH,
            num_beams=NUM_BEAMS
        )

        # Store results
        results.append({
            'id': i,
            'question': question,
            'ground_truth': ground_truth,
            'generated_answer': generated_answer,
            'retrieved_chunks': [
                {
                    'file': rc['chunk']['file'],
                    'page': rc['chunk']['page_number'],
                    'chunk_idx': rc['chunk']['chunk_index'],
                    'score': rc['score'],
                    'text_preview': rc['chunk']['text'][:100] + "..."
                }
                for rc in retrieved_chunks
            ],
            'num_chunks_retrieved': len(retrieved_chunks)
        })

    return results


# ============================================================================
# Evaluation Metrics
# ============================================================================

def calculate_metrics(results):
    """
    Calculate evaluation metrics for RAG system.
    """
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []

    for result in results:
        generated = result['generated_answer']
        ground_truth = result['ground_truth']

        # ROUGE scores
        rouge_scores = scorer.score(ground_truth, generated)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

        # BLEU score
        reference = [ground_truth.split()]
        hypothesis = generated.split()
        bleu = sentence_bleu(reference, hypothesis, smoothing_function=smoothing)
        bleu_scores.append(bleu)

    metrics = {
        'rouge1': np.mean(rouge1_scores) * 100,
        'rouge2': np.mean(rouge2_scores) * 100,
        'rougeL': np.mean(rougeL_scores) * 100,
        'bleu': np.mean(bleu_scores) * 100,
        'total_questions': len(results)
    }

    return metrics


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("="*80)
    print("RETRIEVAL-AUGMENTED QA (RAG) SYSTEM")
    print("="*80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========================================================================
    # Load Data
    # ========================================================================

    print("\n1. LOADING DATA")
    print("-"*80)

    # Load QA pairs
    print(f"Loading QA pairs from {QA_PAIRS_FILE}...")
    with open(QA_PAIRS_FILE, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    # Extract questions and answers
    qa_pairs = []
    for item in qa_data:
        answer = item.get('answer', '')
        if isinstance(answer, dict):
            answer_text = answer.get('text', [''])[0]
        else:
            answer_text = str(answer)

        qa_pairs.append({
            'question': item.get('question', ''),
            'answer_text': answer_text
        })

    print(f"Loaded {len(qa_pairs)} QA pairs")

    # Load all chunks
    all_chunks = load_all_chunks(CHUNKS_DIR)

    # ========================================================================
    # Initialize Models
    # ========================================================================

    print("\n2. INITIALIZING MODELS")
    print("-"*80)

    # Load retriever
    print(f"Loading retriever: {RETRIEVER_MODEL}...")
    retriever = SentenceTransformer(RETRIEVER_MODEL)
    print("✓ Retriever loaded")

    # Load generator
    generator = RAGGenerator(GENERATOR_MODEL)
    print("✓ Generator loaded")

    # ========================================================================
    # Embed Chunks
    # ========================================================================

    print("\n3. EMBEDDING CONTEXT CHUNKS")
    print("-"*80)

    chunk_embeddings, chunks = embed_chunks(
        all_chunks,
        retriever,
        cache_path=EMBEDDINGS_CACHE,
        metadata_cache=CHUNK_METADATA_CACHE
    )

    print(f"Embedding shape: {chunk_embeddings.shape}")
    print(f"Total chunks available for retrieval: {len(chunks)}")

    # ========================================================================
    # Run RAG Pipeline
    # ========================================================================

    print("\n4. RUNNING RAG PIPELINE")
    print("-"*80)

    # Process all questions in the dataset
    test_qa_pairs = qa_pairs  # Process all 911 questions

    results = run_rag_pipeline(
        test_qa_pairs,
        chunk_embeddings,
        chunks,
        retriever,
        generator,
        top_k=TOP_K_CHUNKS
    )

    # ========================================================================
    # Evaluate Results
    # ========================================================================

    print("\n5. EVALUATION")
    print("-"*80)

    metrics = calculate_metrics(results)

    print("\nRAG System Performance:")
    print(f"  ROUGE-1:  {metrics['rouge1']:.2f}%")
    print(f"  ROUGE-2:  {metrics['rouge2']:.2f}%")
    print(f"  ROUGE-L:  {metrics['rougeL']:.2f}%")
    print(f"  BLEU:     {metrics['bleu']:.2f}%")

    # ========================================================================
    # Save Results
    # ========================================================================

    print("\n6. SAVING RESULTS")
    print("-"*80)

    # Save detailed results
    results_file = os.path.join(OUTPUT_DIR, "rag_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ Detailed results saved to {results_file}")

    # Save metrics
    metrics_file = os.path.join(OUTPUT_DIR, "rag_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {metrics_file}")

    # ========================================================================
    # Show Examples
    # ========================================================================

    print("\n7. EXAMPLE OUTPUTS")
    print("-"*80)

    for i in range(min(3, len(results))):
        result = results[i]
        print(f"\nExample {i+1}:")
        print(f"Question: {result['question'][:100]}...")
        print(f"Generated Answer: {result['generated_answer'][:150]}...")
        print(f"Ground Truth: {result['ground_truth'][:150]}...")
        print(f"Retrieved {result['num_chunks_retrieved']} chunks:")
        for j, chunk in enumerate(result['retrieved_chunks'][:3]):
            print(f"  [{j+1}] Score: {chunk['score']:.3f}, File: {chunk['file']}, Page: {chunk['page']}")

    print("\n" + "="*80)
    print("RAG PIPELINE COMPLETE")
    print("="*80)
    print(f"\nDataset Scalability Demonstrated:")
    print(f"  • Total chunks indexed: {len(chunks)}")
    print(f"  • Questions processed: {len(results)}")
    print(f"  • Average retrieval per question: {TOP_K_CHUNKS} chunks")
    print(f"  • Total retrievals performed: {len(results) * TOP_K_CHUNKS}")
    print("\nThis demonstrates the system can scale to retrieve from")
    print("the entire document collection, not just pre-matched chunks!")


if __name__ == "__main__":
    main()
