"""
Context-based Clinical Question Answering System

This module implements two baseline models for Indonesian clinical QA:
1. IndoBERT (extractive) - Extracts answer spans from context
2. mT5 (generative) - Generates answers based on context

Evaluation metrics:
- Exact Match (EM)
- F1 Score
- BLEU
- ROUGE-1, ROUGE-2, ROUGE-L
"""

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import numpy as np
from collections import Counter
import string
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_answer(s):
    """Normalize answer text for evaluation."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, ground_truth):
    """Compute exact match score."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction, ground_truth):
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common_tokens.values())

    if num_common == 0:
        return 0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def compute_bleu(prediction, ground_truth):
    """Compute BLEU score."""
    # Tokenize
    reference = [ground_truth.split()]
    hypothesis = prediction.split()

    # Use smoothing function to handle edge cases
    smoothing = SmoothingFunction().method1

    try:
        bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing)
    except:
        bleu_score = 0.0

    return bleu_score


def compute_rouge_scores(prediction, ground_truth):
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)

    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


# ============================================================================
# Data Loading
# ============================================================================

def load_qa_data(json_path):
    """Load QA data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter out records without context or answer
    filtered_data = []
    for record in data:
        context = record.get('context')
        question = record.get('question')
        answer = record.get('answer')

        # Handle both old format (string) and new format (dict)
        if isinstance(answer, dict):
            answer_text = answer.get('text', [''])[0] if answer.get('text') else ''
            answer_start = answer.get('answer_start', [-1])[0] if answer.get('answer_start') else -1
        else:
            answer_text = str(answer)
            answer_start = -1

        if context and question and answer_text:
            filtered_data.append({
                'id': record.get('id'),
                'context': context,
                'question': question,
                'answer_text': answer_text,
                'answer_start': answer_start
            })

    return filtered_data


def prepare_extractive_features(examples, tokenizer, max_length=384):
    """Prepare features for extractive QA (IndoBERT)."""
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation="only_second",
        max_length=max_length,
        stride=256,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Map answer positions to tokenized context
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answer_start = examples['answer_start'][sample_index]
        answer_text = examples['answer_text'][sample_index]

        if answer_start == -1 or not answer_text:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            answer_end = answer_start + len(answer_text)

            # Find token start and end positions
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Check if answer is in this context window
            if not (offsets[token_start_index][0] <= answer_start and
                    offsets[token_end_index][1] >= answer_end):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Find start and end token positions
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= answer_start:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= answer_end:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_generative_features(examples, tokenizer, max_length=512):
    """Prepare features for generative QA (mT5)."""
    inputs = [f"question: {q} context: {c}" for q, c in zip(examples['question'], examples['context'])]

    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )

    # Tokenize targets
    labels = tokenizer(
        examples['answer_text'],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ============================================================================
# IndoBERT Extractive QA Model
# ============================================================================

class IndoBERTExtractiveQA:
    """IndoBERT-based extractive QA model."""

    def __init__(self, model_name="indolem/indobert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, train_data, val_data=None, output_dir="models/indobert_qa", epochs=3, batch_size=8, learning_rate=3e-5):
        """Train the extractive QA model."""
        # Prepare training dataset
        train_dataset = Dataset.from_dict({
            'question': [d['question'] for d in train_data],
            'context': [d['context'] for d in train_data],
            'answer_text': [d['answer_text'] for d in train_data],
            'answer_start': [d['answer_start'] for d in train_data]
        })

        # Tokenize training data
        tokenized_train = train_dataset.map(
            lambda x: prepare_extractive_features(x, self.tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names
        )

        # Prepare validation dataset if provided
        tokenized_val = None
        if val_data:
            val_dataset = Dataset.from_dict({
                'question': [d['question'] for d in val_data],
                'context': [d['context'] for d in val_data],
                'answer_text': [d['answer_text'] for d in val_data],
                'answer_start': [d['answer_start'] for d in val_data]
            })
            tokenized_val = val_dataset.map(
                lambda x: prepare_extractive_features(x, self.tokenizer),
                batched=True,
                remove_columns=val_dataset.column_names
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch" if val_data else "no",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True if val_data else False,
            metric_for_best_model="eval_loss" if val_data else None,
            logging_steps=10,
            warmup_ratio=0.1,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def predict(self, question, context):
        """Predict answer for a given question and context."""
        inputs = self.tokenizer(
            question,
            context,
            max_length=512,
            truncation="only_second",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1

        answer = self.tokenizer.decode(
            inputs["input_ids"][0][answer_start:answer_end],
            skip_special_tokens=True
        )

        return answer.strip()


# ============================================================================
# mT5 Generative QA Model
# ============================================================================

class MT5GenerativeQA:
    """mT5-based generative QA model."""

    def __init__(self, model_name="google/mt5-base"):
        self.model_name = model_name
        self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, train_data, val_data=None, output_dir="models/mt5_qa", epochs=3, batch_size=4, learning_rate=5e-5):
        """Train the generative QA model."""
        # Prepare training dataset
        train_dataset = Dataset.from_dict({
            'question': [d['question'] for d in train_data],
            'context': [d['context'] for d in train_data],
            'answer_text': [d['answer_text'] for d in train_data]
        })

        # Tokenize training data
        tokenized_train = train_dataset.map(
            lambda x: prepare_generative_features(x, self.tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names
        )

        # Prepare validation dataset if provided
        tokenized_val = None
        if val_data:
            val_dataset = Dataset.from_dict({
                'question': [d['question'] for d in val_data],
                'context': [d['context'] for d in val_data],
                'answer_text': [d['answer_text'] for d in val_data]
            })
            tokenized_val = val_dataset.map(
                lambda x: prepare_generative_features(x, self.tokenizer),
                batched=True,
                remove_columns=val_dataset.column_names
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch" if val_data else "no",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True if val_data else False,
            metric_for_best_model="eval_loss" if val_data else None,
            logging_steps=10,
            warmup_ratio=0.1,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def predict(self, question, context):
        """Generate answer for a given question and context."""
        input_text = f"question: {question} context: {context}"
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, test_data, model_name="Model"):
    """Evaluate a QA model on test data."""
    em_scores = []
    f1_scores = []
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    predictions = []
    ground_truths = []

    print(f"\nEvaluating {model_name}...")

    for i, example in enumerate(test_data):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(test_data)}")

        question = example['question']
        context = example['context']
        ground_truth = example['answer_text']

        # Get prediction
        prediction = model.predict(question, context)

        predictions.append(prediction)
        ground_truths.append(ground_truth)

        # Compute metrics
        em = compute_exact_match(prediction, ground_truth)
        f1 = compute_f1(prediction, ground_truth)
        bleu = compute_bleu(prediction, ground_truth)
        rouge_scores_dict = compute_rouge_scores(prediction, ground_truth)

        em_scores.append(em)
        f1_scores.append(f1)
        bleu_scores.append(bleu)
        rouge1_scores.append(rouge_scores_dict['rouge1'])
        rouge2_scores.append(rouge_scores_dict['rouge2'])
        rougeL_scores.append(rouge_scores_dict['rougeL'])

    # Calculate average metrics
    avg_em = np.mean(em_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100
    avg_bleu = np.mean(bleu_scores) * 100
    avg_rouge1 = np.mean(rouge1_scores) * 100
    avg_rouge2 = np.mean(rouge2_scores) * 100
    avg_rougeL = np.mean(rougeL_scores) * 100

    results = {
        'model': model_name,
        'exact_match': avg_em,
        'f1_score': avg_f1,
        'bleu': avg_bleu,
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL,
        'predictions': predictions,
        'ground_truths': ground_truths
    }

    return results


def error_analysis(predictions, ground_truths, contexts, questions, top_k=10):
    """Perform error analysis on predictions."""
    errors = []

    for i, (pred, truth, ctx, q) in enumerate(zip(predictions, ground_truths, contexts, questions)):
        em = compute_exact_match(pred, truth)
        f1 = compute_f1(pred, truth)

        if em == 0:  # Incorrect prediction
            errors.append({
                'index': i,
                'question': q,
                'context': ctx[:200] + "...",  # Truncate for readability
                'prediction': pred,
                'ground_truth': truth,
                'f1_score': f1
            })

    # Sort by F1 score (lowest first - worst errors)
    errors.sort(key=lambda x: x['f1_score'])

    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    print(f"\nTotal errors: {len(errors)} out of {len(predictions)} ({len(errors)/len(predictions)*100:.2f}%)")

    print(f"\nTop {top_k} worst predictions:")
    for i, error in enumerate(errors[:top_k], 1):
        print(f"\n--- Error {i} (F1: {error['f1_score']:.3f}) ---")
        print(f"Question: {error['question']}")
        print(f"Context: {error['context']}")
        print(f"Predicted: {error['prediction']}")
        print(f"Ground Truth: {error['ground_truth']}")

    # Categorize errors
    print("\n" + "="*80)
    print("ERROR CATEGORIZATION")
    print("="*80)

    # Length analysis
    empty_predictions = sum(1 for p in predictions if not p.strip())
    too_short = sum(1 for p in predictions if len(p.split()) < 3)
    too_long = sum(1 for p in predictions if len(p.split()) > 50)

    print(f"\nEmpty predictions: {empty_predictions} ({empty_predictions/len(predictions)*100:.2f}%)")
    print(f"Too short (<3 words): {too_short} ({too_short/len(predictions)*100:.2f}%)")
    print(f"Too long (>50 words): {too_long} ({too_long/len(predictions)*100:.2f}%)")

    # Partial match analysis
    partial_matches = sum(1 for pred, truth in zip(predictions, ground_truths)
                         if compute_f1(pred, truth) > 0 and compute_exact_match(pred, truth) == 0)

    print(f"\nPartial matches (F1>0 but EM=0): {partial_matches} ({partial_matches/len(predictions)*100:.2f}%)")

    return errors


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("CONTEXT-BASED CLINICAL QA SYSTEM")
    print("="*80)

    # Load data
    print("\nLoading data...")
    data_path = "data/dataset/dataset.json"
    all_data = load_qa_data(data_path)

    print(f"Total examples: {len(all_data)}")

    # Split data (70% train, 15% validation, 15% test)
    train_idx = int(len(all_data) * 0.7)
    val_idx = int(len(all_data) * 0.85)

    train_data = all_data[:train_idx]
    val_data = all_data[train_idx:val_idx]
    test_data = all_data[val_idx:]

    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Test examples: {len(test_data)}")

    # ========================================================================
    # OPTION: Fine-tune models (set to True to enable training)
    # ========================================================================
    ENABLE_FINETUNING = True  # Change to True to enable fine-tuning
    # When True, models whose saved-checkpoint dir already exists are loaded
    # (not re-trained). Set to False (or `rm -rf models/<name>`) to force a
    # fresh fine-tune of an individual model.
    SKIP_IF_TRAINED = True
    # When True, the mT5 generative section is skipped entirely. Useful when
    # only re-evaluating the BERT-family additions (XLM-R, mDeBERTa) without
    # re-running the expensive mT5 fine-tune.
    SKIP_MT5 = True

    if ENABLE_FINETUNING:
        print("\n" + "="*80)
        print("FINE-TUNING MODE ENABLED")
        print("="*80)
        print("\nNote: Fine-tuning will take significant time depending on your hardware.")
        print("GPU recommended for faster training.")
        print("Models will be saved to './models/' directory.\n")

    # ========================================================================
    # BERT-based Extractive QA Models
    # ========================================================================

    print("\n" + "="*80)
    print("BASELINE 1: BERT-based Extractive QA Models")
    print("="*80)

    # Indonesian models + multilingual encoders (XLM-R-large and mDeBERTa-v3-base
    # added as multilingual baselines per reviewer request; both handle
    # extractive QA via AutoModelForQuestionAnswering, same path as IndoBERT).
    bert_models = [
        ("Rifky/Indobert-QA", "Rifky_IndoBERT_QA"),
        ("cahya/bert-base-indonesian-1.5G", "Cahya_IndoBERT"),
        ("indolem/indobert-base-uncased", "Indolem_IndoBERT"),
        ("FacebookAI/xlm-roberta-large", "XLM-R_large"),
        ("microsoft/mdeberta-v3-base", "mDeBERTa-v3_base"),
    ]

    # Store results for all BERT models
    bert_results_all = []

    import os
    for model_name, display_name in bert_models:
        print("\n" + "-"*80)
        print(f"Evaluating: {display_name}")
        print("-"*80)

        try:
            output_dir = f"models/{display_name.lower().replace(' ', '_')}_finetuned"
            already_trained = SKIP_IF_TRAINED and os.path.isdir(output_dir)

            if already_trained:
                # Load the saved fine-tuned checkpoint and skip training.
                print(f"  found existing checkpoint at {output_dir}; loading instead of re-training.")
                print(f"  (delete the dir or set SKIP_IF_TRAINED=False to force a re-train.)")
                bert_model = IndoBERTExtractiveQA(model_name=output_dir)
            else:
                # Initialize from the pristine HF model id and fine-tune.
                bert_model = IndoBERTExtractiveQA(model_name=model_name)
                if ENABLE_FINETUNING:
                    print(f"\nFine-tuning {display_name}...")
                    bert_model.train(
                        train_data=train_data,
                        val_data=val_data,
                        output_dir=output_dir,
                        epochs=3,
                        batch_size=8,
                        learning_rate=3e-5
                    )
                    print(f"Model saved to {output_dir}")

            # Evaluate
            results = evaluate_model(bert_model, test_data, display_name)

            print(f"\n{display_name} RESULTS:")
            print(f"Exact Match: {results['exact_match']:.2f}%")
            print(f"F1 Score: {results['f1_score']:.2f}%")
            print(f"BLEU: {results['bleu']:.2f}%")
            print(f"ROUGE-1: {results['rouge1']:.2f}%")
            print(f"ROUGE-2: {results['rouge2']:.2f}%")
            print(f"ROUGE-L: {results['rougeL']:.2f}%")

            bert_results_all.append({
                'model_name': display_name,
                'model_path': model_name,
                'results': results
            })

        except Exception as e:
            print(f"Error evaluating {display_name}: {e}")
            continue

    # Find best BERT model
    if bert_results_all:
        best_bert = max(bert_results_all, key=lambda x: x['results']['f1_score'])
        print("\n" + "="*80)
        print(f"BEST BERT MODEL: {best_bert['model_name']}")
        print(f"F1 Score: {best_bert['results']['f1_score']:.2f}%")
        print("="*80)

    # ========================================================================
    # mT5 Generative QA
    # ========================================================================

    if SKIP_MT5:
        print("\n" + "="*80)
        print("SKIP_MT5=True -- skipping mT5 baseline (and the downstream")
        print("BERT-vs-mT5 comparison section). Set SKIP_MT5=False to enable.")
        print("="*80)
        return

    print("\n" + "="*80)
    print("BASELINE 2: mT5 (Generative QA)")
    print("="*80)

    mt5_model = MT5GenerativeQA()

    # Fine-tune if enabled
    if ENABLE_FINETUNING:
        print("\nFine-tuning mT5...")
        output_dir = "models/mt5_finetuned"
        mt5_model.train(
            train_data=train_data,
            val_data=val_data,
            output_dir=output_dir,
            epochs=3,
            batch_size=4,
            learning_rate=5e-5
        )
        print(f"Model saved to {output_dir}")

    # Evaluate
    mt5_results = evaluate_model(mt5_model, test_data, "mT5 (Generative)")

    print("\n" + "-"*80)
    print("MT5 RESULTS:")
    print(f"Exact Match: {mt5_results['exact_match']:.2f}%")
    print(f"F1 Score: {mt5_results['f1_score']:.2f}%")
    print(f"BLEU: {mt5_results['bleu']:.2f}%")
    print(f"ROUGE-1: {mt5_results['rouge1']:.2f}%")
    print(f"ROUGE-2: {mt5_results['rouge2']:.2f}%")
    print(f"ROUGE-L: {mt5_results['rougeL']:.2f}%")

    # ========================================================================
    # Comparison Table - All Models
    # ========================================================================

    print("\n" + "="*80)
    print("MODEL COMPARISON - ALL BERT MODELS vs mT5")
    print("="*80)

    # Print comprehensive table header
    print("\n{:<25} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}".format(
        "Model", "EM (%)", "F1 (%)", "BLEU (%)", "R-1 (%)", "R-2 (%)", "R-L (%)"
    ))
    print("-" * 110)

    # Print all BERT model results
    for bert_result in bert_results_all:
        model_name = bert_result['model_name']
        results = bert_result['results']
        print("{:<25} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f}".format(
            model_name,
            results['exact_match'],
            results['f1_score'],
            results['bleu'],
            results['rouge1'],
            results['rouge2'],
            results['rougeL']
        ))

    # Print mT5 results
    print("{:<25} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f}".format(
        "mT5 (Generative)",
        mt5_results['exact_match'],
        mt5_results['f1_score'],
        mt5_results['bleu'],
        mt5_results['rouge1'],
        mt5_results['rouge2'],
        mt5_results['rougeL']
    ))
    print("-" * 110)

    # Highlight best models
    if bert_results_all:
        print(f"\nBest BERT model: {best_bert['model_name']} (F1: {best_bert['results']['f1_score']:.2f}%)")
        print(f"mT5 model: mT5 (Generative) (F1: {mt5_results['f1_score']:.2f}%)")

    # ========================================================================
    # Error Analysis
    # ========================================================================

    test_contexts = [d['context'] for d in test_data]
    test_questions = [d['question'] for d in test_data]

    # Error analysis for best BERT model
    if bert_results_all:
        print("\n" + "="*80)
        print(f"ERROR ANALYSIS - {best_bert['model_name']} (Best BERT Model)")
        print("="*80)

        best_bert_errors = error_analysis(
            best_bert['results']['predictions'],
            best_bert['results']['ground_truths'],
            test_contexts,
            test_questions,
            top_k=5
        )

    # Error analysis for mT5
    print("\n" + "="*80)
    print("ERROR ANALYSIS - mT5 (Generative)")
    print("="*80)

    mt5_errors = error_analysis(
        mt5_results['predictions'],
        mt5_results['ground_truths'],
        test_contexts,
        test_questions,
        top_k=5
    )

    # ========================================================================
    # Save Results
    # ========================================================================

    # Build results summary with all models
    results_summary = {
        'bert_models': {}
    }

    # Add all BERT model results
    for bert_result in bert_results_all:
        model_name = bert_result['model_name']
        results = bert_result['results']
        results_summary['bert_models'][model_name] = {
            'model_path': bert_result['model_path'],
            'exact_match': float(results['exact_match']),
            'f1_score': float(results['f1_score']),
            'bleu': float(results['bleu']),
            'rouge1': float(results['rouge1']),
            'rouge2': float(results['rouge2']),
            'rougeL': float(results['rougeL'])
        }

    # Add mT5 results
    results_summary['mt5'] = {
        'exact_match': float(mt5_results['exact_match']),
        'f1_score': float(mt5_results['f1_score']),
        'bleu': float(mt5_results['bleu']),
        'rouge1': float(mt5_results['rouge1']),
        'rouge2': float(mt5_results['rouge2']),
        'rougeL': float(mt5_results['rougeL'])
    }

    # Add best model info
    if bert_results_all:
        results_summary['best_bert_model'] = {
            'name': best_bert['model_name'],
            'model_path': best_bert['model_path'],
            'exact_match': float(best_bert['results']['exact_match']),
            'f1_score': float(best_bert['results']['f1_score']),
            'bleu': float(best_bert['results']['bleu']),
            'rouge1': float(best_bert['results']['rouge1']),
            'rouge2': float(best_bert['results']['rouge2']),
            'rougeL': float(best_bert['results']['rougeL'])
        }

    # Create output directory if it doesn't exist
    import os
    os.makedirs('data/9-context_based_QA', exist_ok=True)

    with open('data/9-context_based_QA/qa_baseline_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("Results saved to data/9-context_based_QA/qa_baseline_results.json")
    print("="*80)

if __name__ == "__main__":
    main()
