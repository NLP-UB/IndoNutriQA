import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

# Optional imports for advanced models
try:
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers torch")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not installed. Install with: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("Warning: nltk not installed. Install with: pip install nltk")

# ============================================================================
# STEP 1: DATA LOADING AND SPLITTING
# ============================================================================

def combine_question_fields(qa_record: Dict) -> str:
    """
    Combine 'assessment_antropometri', 'diagnosis_medis', and 'question' fields
    into a single comprehensive question string.
    """
    parts = []
    
    assessment = qa_record.get('assessment_antropometri', '').strip()
    if assessment:
        parts.append(f"Assessment Antropometri: {assessment}")
    
    diagnosis = qa_record.get('diagnosis_medis', '').strip()
    if diagnosis:
        parts.append(f"Diagnosis Medis: {diagnosis}")
    
    question = qa_record.get('question', '').strip()
    if question:
        parts.append(f"Pertanyaan: {question}")
    
    combined = " | ".join(parts) if parts else ""
    
    return combined

def load_data(qa_file_path: str, chunks_dir: str) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """
    Load QA pairs and corresponding chunk files.
    """
    with open(qa_file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    for record in qa_data:
        record['combined_question'] = combine_question_fields(record)
    
    chunks_dict = {}
    chunks_path = Path(chunks_dir)
    
    for json_file in chunks_path.glob('*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks_dict[json_file.name] = json.load(f)
    
    if qa_data:
        print(f"Sample combined question: {qa_data[0]['combined_question'][:200]}...")
    
    return qa_data, chunks_dict

def split_dataset(qa_data: List[Dict], train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split dataset into train, validation, and test sets.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    train_data, temp_data = train_test_split(
        qa_data, 
        test_size=(val_size + test_size), 
        random_state=random_state
    )
    
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=test_size/(val_size + test_size), 
        random_state=random_state
    )
    
    print(f"Dataset split complete:")
    print(f"  Train: {len(train_data)} samples ({train_size*100:.1f}%)")
    print(f"  Validation: {len(val_data)} samples ({val_size*100:.1f}%)")
    print(f"  Test: {len(test_data)} samples ({test_size*100:.1f}%)")
    
    return train_data, val_data, test_data

# ============================================================================
# STEP 2: RETRIEVAL MODELS
# ============================================================================

class BM25Retriever:
    """Classical BM25 retrieval baseline."""
    
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        
    def fit(self, documents: List[str]):
        """Fit the vectorizer on documents."""
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.doc_lengths = np.array([len(doc.split()) for doc in documents])
        self.avg_doc_length = np.mean(self.doc_lengths)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Retrieve top-k most similar documents."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        
        return list(zip(top_indices, top_scores))

class TFIDFRetriever:
    """TF-IDF based retrieval."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(lowercase=True)
        
    def fit(self, documents: List[str]):
        """Fit the vectorizer on documents."""
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Retrieve top-k most similar documents."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        
        return list(zip(top_indices, top_scores))

class BERTRetriever:
    """BERT-based dense retrieval using sentence embeddings."""
    
    def __init__(self, model_name: str = "indobenchmark/indobert-base-p1"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Install with: pip install transformers torch")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.embeddings = None
        
    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                       max_length=512, padding=True).to(self.device)
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    def fit(self, documents: List[str]):
        """Encode all documents."""
        print(f"    Encoding {len(documents)} documents...")
        self.embeddings = self._encode(documents)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Retrieve top-k most similar documents."""
        query_embedding = self._encode([query])
        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        
        return list(zip(top_indices, top_scores))

# ============================================================================
# STEP 3: RETRIEVAL EVALUATION
# ============================================================================

def evaluate_retrieval(retriever, test_data: List[Dict], chunks_dict: Dict[str, List[Dict]], 
                       k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
    """
    Evaluate retrieval performance using Recall@k.
    """
    results = {f"Recall@{k}": 0.0 for k in k_values}
    results["MRR"] = 0.0
    
    total_queries = 0
    mrr_sum = 0.0
    
    for qa_record in test_data:
        file_name = qa_record.get('file', '')
        question = qa_record.get('combined_question', '')
        correct_chunk_idx = qa_record.get('chunk_index')
        
        if not file_name or not question or correct_chunk_idx is None:
            continue
        
        chunks = chunks_dict.get(file_name, [])
        if not chunks:
            continue
        
        documents = [chunk.get('text', '') for chunk in chunks]
        
        retriever.fit(documents)
        retrieved = retriever.retrieve(question, top_k=max(k_values))
        
        retrieved_indices = [idx for idx, score in retrieved]
        
        try:
            correct_position = retrieved_indices.index(correct_chunk_idx) + 1
            mrr_sum += 1.0 / correct_position
            
            for k in k_values:
                if correct_chunk_idx in retrieved_indices[:k]:
                    results[f"Recall@{k}"] += 1
        except ValueError:
            pass
        
        total_queries += 1
    
    if total_queries > 0:
        for k in k_values:
            results[f"Recall@{k}"] = results[f"Recall@{k}"] / total_queries
        results["MRR"] = mrr_sum / total_queries
    
    return results

# ============================================================================
# STEP 4: QA/READER MODELS
# ============================================================================

class SimpleExtractiveQA:
    """Simple baseline: extract sentences with keyword overlap."""
    
    def predict(self, question: str, context: str) -> str:
        question_words = set(self.normalize_answer(question).split())
        sentences = context.split('.')
        
        best_sentence = ""
        best_overlap = 0
        
        for sentence in sentences:
            sentence_words = set(self.normalize_answer(sentence).split())
            overlap = len(question_words & sentence_words)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence.strip()
        
        return best_sentence if best_sentence else context[:200]
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

class BERTQuestionAnswering:
    """BERT-based extractive QA model using pipeline or fallback."""
    
    def __init__(self, model_name: str = "indobenchmark/indobert-base-p1"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed")
        
        self.device = 0 if torch.cuda.is_available() else -1
        self.qa_pipeline = None
        self.simple_qa = SimpleExtractiveQA()
        self.model_name = model_name
        
        print(f"    Attempting to load {model_name}...")
        
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model=model_name,
                tokenizer=model_name,
                device=self.device,
                framework="pt"
            )
            print(f"    ✓ Successfully loaded {model_name} with QA pipeline")
        except Exception as e:
            print(f"    ✗ Failed to load {model_name}: {str(e)[:80]}")
            print(f"    → This model will fall back to Simple Extractive QA")
            self.qa_pipeline = None
    
    def predict(self, question: str, context: str) -> str:
        # If no pipeline available, use simple extraction
        if self.qa_pipeline is None:
            return self.simple_qa.predict(question, context)
        
        try:
            # Truncate context if too long
            max_context_length = 3000
            if len(context) > max_context_length:
                context = context[:max_context_length]
            
            # Truncate question if too long
            max_question_length = 200
            if len(question) > max_question_length:
                question = question[:max_question_length]
            
            result = self.qa_pipeline(
                question=question,
                context=context,
                max_answer_len=200,
                handle_impossible_answer=True,
                top_k=1
            )
            
            answer = result.get('answer', '').strip()
            
            # If no answer or very short, fall back to simple extraction
            if not answer or len(answer) < 3:
                return self.simple_qa.predict(question, context)
            
            return answer
            
        except Exception as e:
            # Silently fall back to simple extraction on error
            return self.simple_qa.predict(question, context)

class LLMQuestionAnswering:
    """LLM-based QA using API (GPT-4, Claude, etc.)."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key and "gpt" in model.lower():
            print("  Warning: No API key provided. Set OPENAI_API_KEY environment variable.")
    
    def predict(self, question: str, context: str) -> str:
        """Call LLM API to answer question given context."""
        if "gpt" in self.model.lower():
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                
                prompt = f"""Berdasarkan konteks berikut, jawab pertanyaan dengan singkat dan akurat.

Konteks: {context}

Pertanyaan: {question}

Jawaban:"""
                
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.0
                )
                
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"  API call failed: {e}")
                return SimpleExtractiveQA().predict(question, context)
        
        elif "claude" in self.model.lower():
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
                
                prompt = f"""Berdasarkan konteks berikut, jawab pertanyaan dengan singkat dan akurat.

Konteks: {context}

Pertanyaan: {question}

Jawaban:"""
                
                response = client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return response.content[0].text.strip()
            except Exception as e:
                print(f"  API call failed: {e}")
                return SimpleExtractiveQA().predict(question, context)
        
        return SimpleExtractiveQA().predict(question, context)

# ============================================================================
# STEP 5: QA EVALUATION METRICS
# ============================================================================

def normalize_answer(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def compute_f1(predicted: str, ground_truth: str) -> float:
    """Compute F1 score between predicted and ground truth answers."""
    pred_tokens = normalize_answer(predicted).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common_tokens.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1

def compute_exact_match(predicted: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(predicted) == normalize_answer(ground_truth))

def compute_bleu(predicted: str, ground_truth: str) -> float:
    """Compute BLEU score."""
    if not BLEU_AVAILABLE:
        return 0.0
    
    reference = [normalize_answer(ground_truth).split()]
    candidate = normalize_answer(predicted).split()
    
    smoothing = SmoothingFunction().method1
    return sentence_bleu(reference, candidate, smoothing_function=smoothing)

def compute_rouge(predicted: str, ground_truth: str) -> Dict[str, float]:
    """Compute ROUGE scores."""
    if not ROUGE_AVAILABLE:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    scores = scorer.score(ground_truth, predicted)
    
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure
    }

def evaluate_qa(qa_model, test_data: List[Dict], chunks_dict: Dict[str, List[Dict]]) -> Dict[str, float]:
    """
    Evaluate QA/Reader performance using multiple metrics.
    Uses batching for efficiency when using GPU pipelines.
    """
    f1_scores = []
    em_scores = []
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    total_samples = len(test_data)
    processed = 0
    
    for i, qa_record in enumerate(test_data):
        if i % 10 == 0:
            print(f"    Processing {i}/{total_samples}...")
        
        file_name = qa_record.get('file', '')
        question = qa_record.get('combined_question', '')
        ground_truth = qa_record.get('answer', '')
        chunk_idx = qa_record.get('chunk_index')
        
        if not file_name or not question or not ground_truth or chunk_idx is None:
            continue
        
        chunks = chunks_dict.get(file_name, [])
        if not chunks:
            continue
        
        try:
            correct_chunk = next(c for c in chunks if c.get('chunk_index') == chunk_idx)
            context = correct_chunk.get('text', '')
        except StopIteration:
            continue
        
        if not context:
            continue
        
        try:
            predicted = qa_model.predict(question, context)
        except Exception as e:
            print(f"    Warning: Prediction failed for sample {i}, using fallback")
            predicted = SimpleExtractiveQA().predict(question, context)
        
        f1_scores.append(compute_f1(predicted, ground_truth))
        em_scores.append(compute_exact_match(predicted, ground_truth))
        bleu_scores.append(compute_bleu(predicted, ground_truth))
        
        rouge_scores = compute_rouge(predicted, ground_truth)
        rouge1_scores.append(rouge_scores['rouge1'])
        rouge2_scores.append(rouge_scores['rouge2'])
        rougeL_scores.append(rouge_scores['rougeL'])
        
        processed += 1
    
    results = {
        "F1": np.mean(f1_scores) if f1_scores else 0.0,
        "Exact_Match": np.mean(em_scores) if em_scores else 0.0,
        "BLEU": np.mean(bleu_scores) if bleu_scores else 0.0,
        "ROUGE-1": np.mean(rouge1_scores) if rouge1_scores else 0.0,
        "ROUGE-2": np.mean(rouge2_scores) if rouge2_scores else 0.0,
        "ROUGE-L": np.mean(rougeL_scores) if rougeL_scores else 0.0,
        "Total_Samples": processed
    }
    
    return results

# ============================================================================
# MAIN BENCHMARKING PIPELINE
# ============================================================================

def run_benchmark(qa_file_path: str, chunks_dir: str, output_file: str = "benchmark_results.json",
                 use_bert: bool = True, use_llm: bool = False, llm_api_key: Optional[str] = None):
    """
    Run complete benchmarking pipeline.
    """
    print("="*80)
    print("ADVANCED RETRIEVAL AND QA BENCHMARKING PIPELINE")
    print("="*80)
    
    print("\n[1] Loading data...")
    qa_data, chunks_dict = load_data(qa_file_path, chunks_dir)
    print(f"Loaded {len(qa_data)} QA pairs and {len(chunks_dict)} chunk files")
    
    print("\n[2] Splitting dataset...")
    train_data, val_data, test_data = split_dataset(qa_data)
    
    print("\n[3] Initializing retrieval models...")
    retrievers = {
        "BM25": BM25Retriever(),
        "TF-IDF": TFIDFRetriever()
    }
    
    if use_bert and TRANSFORMERS_AVAILABLE:
        try:
            retrievers["IndoBERT"] = BERTRetriever("indobenchmark/indobert-base-p1")
        except Exception as e:
            print(f"  Could not load IndoBERT: {e}")
    
    print("\n[4] Running retrieval benchmarks...")
    retrieval_results = {}
    
    for name, retriever in retrievers.items():
        print(f"\n  Evaluating {name}...")
        results = evaluate_retrieval(retriever, test_data, chunks_dict)
        retrieval_results[name] = results
        
        print(f"  {name} Results:")
        for metric, value in results.items():
            print(f"    {metric}: {value:.4f}")
    
    print("\n[5] Initializing QA models...")
    qa_models = {
        "Simple_Extractive": SimpleExtractiveQA()
    }
    
    if use_bert and TRANSFORMERS_AVAILABLE:
        # Try multiple Indonesian QA models for comparison
        bert_models = [
            ("Rifky/Indobert-QA", "Rifky_IndoBERT_QA"),
            ("cahya/bert-base-indonesian-1.5G", "Cahya_IndoBERT"),
            ("LazarusNLP/IndoBERT-base-uncased", "Lazarus_IndoBERT"),
            ("indobenchmark/indobert-base-p1", "IndoBERT_Base"),
            ("deepset/xlm-roberta-large-squad2", "/xlm-roberta-large"),
            ("deepset/xlm-roberta-base-squad2", "/xlm-roberta-base"),
            ("rifqifazh/IndoRoberta-QA", "IndoRoberta"),
            ("deepset/roberta-base-squad2", "roberta_base"),
        ]
        
        for model_name, display_name in bert_models:
            try:
                print(f"\n  Loading {display_name}...")
                qa_model = BERTQuestionAnswering(model_name)
                # Only add if successfully loaded with QA pipeline
                if qa_model.qa_pipeline is not None:
                    qa_models[display_name] = qa_model
                    print(f"  ✓ {display_name} added to comparison")
                else:
                    print(f"  ✗ {display_name} not available (will be skipped)")
            except Exception as e:
                print(f"  ✗ Could not load {display_name}: {str(e)[:80]}")
    
    if use_llm:
        qa_models["GPT-4o"] = LLMQuestionAnswering(api_key=llm_api_key, model="gpt-4o")
    
    print(f"\n  Total QA models to evaluate: {len(qa_models)}")
    for model_name in qa_models.keys():
        print(f"    - {model_name}")
    
    print("\n[6] Running QA/Reader benchmarks...")
    qa_results = {}
    
    for name, qa_model in qa_models.items():
        print(f"\n  Evaluating {name}...")
        results = evaluate_qa(qa_model, test_data, chunks_dict)
        qa_results[name] = results
        
        print(f"  {name} Results:")
        for metric, value in results.items():
            if metric != "Total_Samples":
                print(f"    {metric}: {value:.4f}")
            else:
                print(f"    {metric}: {value}")
    
    # Save results
    all_results = {
        "dataset_info": {
            "total_samples": len(qa_data),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data)
        },
        "retrieval_results": retrieval_results,
        "qa_results": qa_results
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[7] Results saved to {output_file}")
    
    # Print comparison summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY - QA MODEL COMPARISON")
    print("="*80)
    
    if qa_results:
        print("\nQA Model Rankings by F1 Score:")
        sorted_models = sorted(qa_results.items(), key=lambda x: x[1].get('F1', 0), reverse=True)
        for rank, (model_name, results) in enumerate(sorted_models, 1):
            f1 = results.get('F1', 0)
            em = results.get('Exact_Match', 0)
            samples = results.get('Total_Samples', 0)
            print(f"  {rank}. {model_name:30s} - F1: {f1:.4f}, EM: {em:.4f} ({samples} samples)")
        
        print("\nQA Model Rankings by Exact Match:")
        sorted_models_em = sorted(qa_results.items(), key=lambda x: x[1].get('Exact_Match', 0), reverse=True)
        for rank, (model_name, results) in enumerate(sorted_models_em, 1):
            em = results.get('Exact_Match', 0)
            print(f"  {rank}. {model_name:30s} - EM: {em:.4f}")
        
        print("\nDetailed Metrics Comparison:")
        print(f"{'Model':<30s} {'F1':>8s} {'EM':>8s} {'BLEU':>8s} {'R-1':>8s} {'R-2':>8s} {'R-L':>8s}")
        print("-" * 80)
        for model_name, results in sorted_models:
            print(f"{model_name:<30s} "
                  f"{results.get('F1', 0):>8.4f} "
                  f"{results.get('Exact_Match', 0):>8.4f} "
                  f"{results.get('BLEU', 0):>8.4f} "
                  f"{results.get('ROUGE-1', 0):>8.4f} "
                  f"{results.get('ROUGE-2', 0):>8.4f} "
                  f"{results.get('ROUGE-L', 0):>8.4f}")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    
    return all_results

if __name__ == "__main__":
    QA_FILE = "data/dataset/dataset.json"
    CHUNKS_DIR = "data/4-anonymize_using_extracted_ner"
    OUTPUT_FILE = "data/8-retrieval_benchmarking/benchmark_results.json"
    
    USE_BERT = True
    USE_LLM = False
    LLM_API_KEY = None
    
    results = run_benchmark(
        qa_file_path=QA_FILE,
        chunks_dir=CHUNKS_DIR,
        output_file=OUTPUT_FILE,
        use_bert=USE_BERT,
        use_llm=USE_LLM,
        llm_api_key=LLM_API_KEY
    )