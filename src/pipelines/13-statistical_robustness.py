"""
Statistical Robustness Pipeline for IndoNutriQA

Implements the three robustness procedures referenced in §5.4 of the paper:

1. Bootstrap 95% confidence intervals
   - Cheap, post-hoc
   - Reuses LLM prediction cache from 12-llm_evaluation
   - Loads saved fine-tuned baselines from models/ for inference
   - Output: data/13-statistical_robustness/bootstrap_ci.json

2. 5-fold cross-validation (stratified by source report)
   - Re-trains each fine-tuned baseline 5 times
   - GroupKFold on `file` so chunks from one document never split across folds
   - Output: data/13-statistical_robustness/cv_results.json

3. Learning curve over training-set fractions {10%, 25%, 50%, 100%}
   - Re-trains each fine-tuned baseline 4 times
   - Validation and test splits held fixed; training subsampled with a fixed seed
   - Output: data/13-statistical_robustness/learning_curve.json
              + Paper/figures/learning_curve.pdf

Usage (cheap → expensive):
    # 1. Cheap: dump predictions from the LLM cache and bootstrap them
    python src/pipelines/13-statistical_robustness.py extract-llm-predictions
    python src/pipelines/13-statistical_robustness.py bootstrap

    # 2. Medium (~5 min/model): inference with saved fine-tuned baselines
    python src/pipelines/13-statistical_robustness.py extract-baseline-predictions
    python src/pipelines/13-statistical_robustness.py bootstrap

    # 3. Heavy (overnight): re-train under k-fold CV
    python src/pipelines/13-statistical_robustness.py cv \\
        --models rifky cahya indolem mt5 --folds 5

    # 4. Heavy (overnight): learning curve at 10/25/50/100% of training
    python src/pipelines/13-statistical_robustness.py learning-curve \\
        --models rifky cahya indolem mt5 --fractions 0.1 0.25 0.5 1.0
"""

import argparse
import hashlib
import json
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

DATA_PATH = "data/dataset/dataset.json"
OUTPUT_DIR = Path("data/13-statistical_robustness")
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
LLM_CACHE_DIR = Path("data/12-llm_evaluation/cache")
LLM_RESULTS_PATH = Path("data/12-llm_evaluation/llm_results.json")
MODELS_DIR = Path("models")
PAPER_FIGURES_DIR = Path("Paper/figures")

DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 42

# Paper §5.4 specifies these. Override via CLI if needed.
DEFAULT_CV_FOLDS = 5
DEFAULT_LEARNING_CURVE_FRACTIONS = [0.1, 0.25, 0.5, 1.0]

# Fine-tuned baselines — alias -> (saved model path, kind).
# `kind` selects between extractive (IndoBERT family) and generative (mT5).
BASELINE_REGISTRY: Dict[str, Dict[str, str]] = {
    "rifky":     {"path": "models/rifky_indobert_qa_finetuned", "kind": "extractive"},
    "cahya":     {"path": "models/cahya_indobert_finetuned",    "kind": "extractive"},
    "indolem":   {"path": "models/indolem_indobert_finetuned",  "kind": "extractive"},
    "xlmr":      {"path": "models/xlm-r_large_finetuned",       "kind": "extractive"},
    "mdeberta":  {"path": "models/mdeberta-v3_base_finetuned",  "kind": "extractive"},
    "mt5":       {"path": "models/mt5_finetuned",               "kind": "generative"},
}

# Pristine HF model ids used for re-training under CV / learning curve
# (the saved finetuned models can't be re-trained from scratch without these).
BASELINE_PRISTINE_HF_IDS: Dict[str, str] = {
    "rifky":     "Rifky/Indobert-QA",
    "cahya":     "cahya/bert-base-indonesian-1.5G",
    "indolem":   "indolem/indobert-base-uncased",
    "xlmr":      "FacebookAI/xlm-roberta-large",
    "mdeberta":  "microsoft/mdeberta-v3-base",
    "mt5":       "google/mt5-base",
}


# ============================================================================
# Evaluation metrics (mirrors 9-context_based_QA.py and 12-llm_evaluation.py)
# ============================================================================

def normalize_answer(s: str) -> str:
    def remove_articles(t):  return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t):  return " ".join(t.split())
    def remove_punc(t):
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_em(prediction: str, truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(truth))


def compute_f1(prediction: str, truth: str) -> float:
    p = normalize_answer(prediction).split()
    t = normalize_answer(truth).split()
    if not p or not t:
        return float(p == t)
    common = Counter(p) & Counter(t)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision = n / len(p)
    recall = n / len(t)
    return 2 * precision * recall / (precision + recall)


def compute_bleu(prediction: str, truth: str) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    try:
        return sentence_bleu([truth.split()], prediction.split(),
                             smoothing_function=SmoothingFunction().method1)
    except Exception:
        return 0.0


def compute_rouge_l(prediction: str, truth: str) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(truth, prediction)["rougeL"].fmeasure


METRIC_FNS = {
    "exact_match": compute_em,
    "f1_score":    compute_f1,
    "bleu":        compute_bleu,
    "rougeL":      compute_rouge_l,
}


# ============================================================================
# Data loading (identical 70/15/15 split as 9-context_based_QA.py)
# ============================================================================

def load_qa_data(path: str = DATA_PATH) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: List[Dict[str, Any]] = []
    for record in data:
        context = record.get("context")
        question = record.get("question")
        answer = record.get("answer")
        if isinstance(answer, dict):
            atxt = answer.get("text", [""])[0] if answer.get("text") else ""
            astart = answer.get("answer_start", [-1])[0] if answer.get("answer_start") else -1
        else:
            atxt = str(answer) if answer else ""
            astart = -1
        if context and question and atxt:
            out.append({
                "id": record.get("id"),
                "context": context,
                "question": question,
                "answer_text": atxt,
                "answer_start": astart,
                "anthropometric_assessment": record.get("anthropometric_assessment"),
                "medical_diagnosis": record.get("medical_diagnosis"),
                "file": record.get("file"),
                "report_title": record.get("report_title"),
            })
    return out


def split_data(all_data: List[Dict[str, Any]]) -> Tuple[List, List, List]:
    train_idx = int(len(all_data) * 0.7)
    val_idx = int(len(all_data) * 0.85)
    return all_data[:train_idx], all_data[train_idx:val_idx], all_data[val_idx:]


# ============================================================================
# Predictions on disk
#   Format: {"model": str, "shots": int, "n": int,
#            "predictions": [str, ...], "ground_truths": [str, ...]}
# ============================================================================

def save_predictions(name: str, payload: Dict[str, Any]) -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    p = PREDICTIONS_DIR / f"{name}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def load_predictions(name: str) -> Dict[str, Any]:
    p = PREDICTIONS_DIR / f"{name}.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Bootstrap CI
# ============================================================================

def bootstrap_metric(
    predictions: List[str],
    ground_truths: List[str],
    metric_fn,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Return (point_estimate_pct, ci_low_pct, ci_high_pct)."""
    n = len(predictions)
    rng = np.random.default_rng(seed)
    point = 100.0 * np.mean([metric_fn(p, t) for p, t in zip(predictions, ground_truths)])

    # Cache per-example scores so we can resample without recomputing the metric.
    per_example = np.array([metric_fn(p, t) for p, t in zip(predictions, ground_truths)])
    sample_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample_means[i] = per_example[idx].mean()
    lo = float(np.percentile(sample_means, 2.5) * 100.0)
    hi = float(np.percentile(sample_means, 97.5) * 100.0)
    return point, lo, hi


def bootstrap_all_metrics(
    predictions: List[str],
    ground_truths: List[str],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for metric_name, fn in METRIC_FNS.items():
        point, lo, hi = bootstrap_metric(predictions, ground_truths, fn, n_resamples, seed)
        out[metric_name] = {"point": point, "ci_low": lo, "ci_high": hi}
    return out


# ============================================================================
# Subcommand: extract-llm-predictions  (free — reads existing cache)
# ============================================================================

def cmd_extract_llm_predictions(args: argparse.Namespace) -> None:
    """Reconstruct per-example predictions from data/12-llm_evaluation/cache/.

    The cache is keyed on a SHA-256 of (model_id, shots, prompt_version,
    question, context, anthro, diagnosis), so we recompute the same hash for
    every (model, shots, test example) and look it up.
    """
    from importlib.util import spec_from_file_location, module_from_spec

    # Pull the LLM module to reuse its cache_key + sample_few_shot helpers.
    llm_path = Path(__file__).parent / "12-llm_evaluation.py"
    spec = spec_from_file_location("llm_eval", llm_path)
    llm = module_from_spec(spec)
    spec.loader.exec_module(llm)

    all_data = load_qa_data(args.data_path)
    train_data, _, test_data = split_data(all_data)

    results_path = Path(args.results_file) if args.results_file else LLM_RESULTS_PATH
    if not results_path.exists():
        raise SystemExit(f"{results_path} not found. Run 12-llm_evaluation.py first.")
    summary = json.loads(results_path.read_text())
    rows = summary.get("results", [])
    if not rows:
        raise SystemExit(
            f"No results in {results_path}. "
            f"Pass --results-file pointing to a populated file (e.g. llm_results_openai.json)."
        )

    fs_seed = summary.get("config", {}).get("few_shot_seed", llm.DEFAULT_FEW_SHOT_SEED)

    extracted = 0
    missing = 0
    for row in rows:
        alias = row["alias"]
        model_id = row["model_id"]
        shots = row["shots"]
        few_shot = llm.sample_few_shot(train_data, shots, seed=fs_seed)

        predictions: List[Optional[str]] = []
        for ex in test_data:
            key = llm.cache_key(model_id, shots, ex, fs_seed)
            cached = llm.cache_get(key)
            predictions.append(cached)

        n_missing = sum(1 for p in predictions if p is None)
        if n_missing > 0:
            print(f"  [warn] {alias} ({shots}-shot): {n_missing}/{len(test_data)} cache misses; skipping")
            missing += 1
            continue

        truths = [ex["answer_text"] for ex in test_data]
        name = f"llm__{alias}__{shots}shot"
        save_predictions(name, {
            "model": alias,
            "model_id": model_id,
            "kind": "llm",
            "shots": shots,
            "n": len(test_data),
            "predictions": predictions,
            "ground_truths": truths,
        })
        extracted += 1
        print(f"  saved {name}.json ({len(test_data)} examples)")

    print(f"\nExtracted: {extracted} | with missing cache: {missing}")


# ============================================================================
# Subcommand: extract-baseline-predictions
#   Loads each saved fine-tuned model from models/ and runs test-set inference.
# ============================================================================

def _predict_extractive(model_path: str, test_data: List[Dict[str, Any]]) -> List[str]:
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device).eval()

    preds: List[str] = []
    for ex in tqdm(test_data, desc=f"  {Path(model_path).name}"):
        inputs = tokenizer(
            ex["question"], ex["context"],
            max_length=512, truncation="only_second", return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        start = int(torch.argmax(out.start_logits))
        end = int(torch.argmax(out.end_logits)) + 1
        ans = tokenizer.decode(inputs["input_ids"][0][start:end], skip_special_tokens=True)
        preds.append(ans.strip())
    return preds


def _predict_generative(model_path: str, test_data: List[Dict[str, Any]]) -> List[str]:
    import torch
    from transformers import MT5ForConditionalGeneration, MT5Tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MT5Tokenizer.from_pretrained(model_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_path).to(device).eval()

    preds: List[str] = []
    for ex in tqdm(test_data, desc=f"  {Path(model_path).name}"):
        prompt = f"question: {ex['question']} context: {ex['context']}"
        inputs = tokenizer(prompt, max_length=512, truncation=True,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True).strip())
    return preds


def cmd_extract_baseline_predictions(args: argparse.Namespace) -> None:
    all_data = load_qa_data(args.data_path)
    _, _, test_data = split_data(all_data)
    truths = [ex["answer_text"] for ex in test_data]

    aliases = args.models or list(BASELINE_REGISTRY.keys())
    for alias in aliases:
        if alias not in BASELINE_REGISTRY:
            print(f"  [skip] unknown baseline alias: {alias}")
            continue
        spec = BASELINE_REGISTRY[alias]
        path = spec["path"]
        if not Path(path).exists():
            print(f"  [skip] {alias}: model dir {path} not found (re-run 9-context_based_QA.py)")
            continue
        print(f"\n--- {alias} ({spec['kind']}: {path}) ---")
        try:
            if spec["kind"] == "extractive":
                preds = _predict_extractive(path, test_data)
            else:
                preds = _predict_generative(path, test_data)
        except Exception as e:
            print(f"  [skip] inference failed for {alias}: {e}")
            continue

        name = f"baseline__{alias}"
        save_predictions(name, {
            "model": alias,
            "model_path": path,
            "kind": f"baseline-{spec['kind']}",
            "shots": None,
            "n": len(test_data),
            "predictions": preds,
            "ground_truths": truths,
        })
        print(f"  saved {name}.json")


# ============================================================================
# Subcommand: bootstrap
# ============================================================================

def cmd_bootstrap(args: argparse.Namespace) -> None:
    if not PREDICTIONS_DIR.exists() or not any(PREDICTIONS_DIR.glob("*.json")):
        raise SystemExit(
            f"No prediction files in {PREDICTIONS_DIR}. "
            f"Run extract-llm-predictions and/or extract-baseline-predictions first."
        )

    rows: List[Dict[str, Any]] = []
    for path in sorted(PREDICTIONS_DIR.glob("*.json")):
        if args.filter and args.filter not in path.stem:
            continue
        payload = json.loads(path.read_text())
        preds = payload["predictions"]
        truths = payload["ground_truths"]
        print(f"\n{path.stem}  (n={len(preds)})")
        ci = bootstrap_all_metrics(preds, truths,
                                   n_resamples=args.resamples, seed=args.seed)
        for metric, vals in ci.items():
            print(f"  {metric:12s} {vals['point']:6.2f}  [{vals['ci_low']:5.2f}, {vals['ci_high']:5.2f}]")
        rows.append({
            "name": path.stem,
            "model": payload.get("model"),
            "kind": payload.get("kind"),
            "shots": payload.get("shots"),
            "n": payload.get("n"),
            "metrics": ci,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "bootstrap_ci.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "n_resamples": args.resamples,
                "seed": args.seed,
                "metrics": list(METRIC_FNS.keys()),
            },
            "results": rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")


# ============================================================================
# Subcommand: cv  (5-fold cross-validation, stratified by source report)
# ============================================================================

def _train_and_eval_extractive(
    base_hf_id: str,
    train_data: List[Dict[str, Any]],
    eval_data: List[Dict[str, Any]],
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Dict[str, float]:
    """Fine-tune an extractive baseline and return point metrics on eval_data."""
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                              Trainer, TrainingArguments)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_hf_id)
    model = AutoModelForQuestionAnswering.from_pretrained(base_hf_id).to(device)

    def featurize(examples):
        tokenized = tokenizer(
            examples["question"], examples["context"],
            truncation="only_second", max_length=384, stride=256,
            return_overflowing_tokens=True, return_offsets_mapping=True,
            padding="max_length",
        )
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        offsets = tokenized.pop("offset_mapping")
        tokenized["start_positions"] = []
        tokenized["end_positions"] = []
        for i, off in enumerate(offsets):
            ids = tokenized["input_ids"][i]
            cls = ids.index(tokenizer.cls_token_id)
            seq = tokenized.sequence_ids(i)
            j = sample_map[i]
            astart = examples["answer_start"][j]
            atxt = examples["answer_text"][j]
            if astart == -1 or not atxt:
                tokenized["start_positions"].append(cls)
                tokenized["end_positions"].append(cls)
                continue
            aend = astart + len(atxt)
            ts, te = 0, len(ids) - 1
            while seq[ts] != 1: ts += 1
            while seq[te] != 1: te -= 1
            if not (off[ts][0] <= astart and off[te][1] >= aend):
                tokenized["start_positions"].append(cls)
                tokenized["end_positions"].append(cls)
            else:
                while ts < len(off) and off[ts][0] <= astart: ts += 1
                tokenized["start_positions"].append(ts - 1)
                while off[te][1] >= aend: te -= 1
                tokenized["end_positions"].append(te + 1)
        return tokenized

    train_ds = Dataset.from_dict({k: [d[k] for d in train_data]
                                  for k in ("question", "context", "answer_text", "answer_start")})
    train_tok = train_ds.map(featurize, batched=True, remove_columns=train_ds.column_names)

    args_ = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=50,
        warmup_ratio=0.1,
        report_to=[],
    )
    Trainer(model=model, args=args_, train_dataset=train_tok).train()

    model.eval()
    preds: List[str] = []
    for ex in eval_data:
        inputs = tokenizer(ex["question"], ex["context"], max_length=512,
                           truncation="only_second", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        s = int(torch.argmax(out.start_logits))
        e = int(torch.argmax(out.end_logits)) + 1
        preds.append(tokenizer.decode(inputs["input_ids"][0][s:e], skip_special_tokens=True).strip())

    truths = [ex["answer_text"] for ex in eval_data]
    return {m: 100.0 * float(np.mean([fn(p, t) for p, t in zip(preds, truths)]))
            for m, fn in METRIC_FNS.items()}


def _train_and_eval_generative(
    base_hf_id: str,
    train_data: List[Dict[str, Any]],
    eval_data: List[Dict[str, Any]],
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Dict[str, float]:
    import torch
    from datasets import Dataset
    from transformers import (MT5ForConditionalGeneration, MT5Tokenizer,
                              Trainer, TrainingArguments)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MT5Tokenizer.from_pretrained(base_hf_id)
    model = MT5ForConditionalGeneration.from_pretrained(base_hf_id).to(device)

    def featurize(examples):
        inputs = [f"question: {q} context: {c}"
                  for q, c in zip(examples["question"], examples["context"])]
        enc = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(examples["answer_text"], max_length=128,
                           truncation=True, padding="max_length")
        enc["labels"] = labels["input_ids"]
        return enc

    train_ds = Dataset.from_dict({k: [d[k] for d in train_data]
                                  for k in ("question", "context", "answer_text")})
    train_tok = train_ds.map(featurize, batched=True, remove_columns=train_ds.column_names)

    args_ = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=50,
        warmup_ratio=0.1,
        report_to=[],
    )
    Trainer(model=model, args=args_, train_dataset=train_tok).train()

    model.eval()
    preds: List[str] = []
    for ex in eval_data:
        prompt = f"question: {ex['question']} context: {ex['context']}"
        inputs = tokenizer(prompt, max_length=512, truncation=True,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True).strip())

    truths = [ex["answer_text"] for ex in eval_data]
    return {m: 100.0 * float(np.mean([fn(p, t) for p, t in zip(preds, truths)]))
            for m, fn in METRIC_FNS.items()}


def _train_and_eval(
    alias: str, train_data, eval_data, output_dir, epochs, batch_size, learning_rate
) -> Dict[str, float]:
    spec = BASELINE_REGISTRY[alias]
    base_hf = BASELINE_PRISTINE_HF_IDS[alias]
    if spec["kind"] == "extractive":
        return _train_and_eval_extractive(base_hf, train_data, eval_data,
                                          output_dir, epochs, batch_size, learning_rate)
    return _train_and_eval_generative(base_hf, train_data, eval_data,
                                      output_dir, epochs, batch_size, learning_rate)


def cmd_cv(args: argparse.Namespace) -> None:
    from sklearn.model_selection import GroupKFold

    all_data = load_qa_data(args.data_path)
    groups = [d["file"] or d["report_title"] or str(d["id"]) for d in all_data]
    gkf = GroupKFold(n_splits=args.folds)

    aliases = args.models or list(BASELINE_REGISTRY.keys())
    cv_results: Dict[str, Dict[str, Any]] = {}

    for alias in aliases:
        if alias not in BASELINE_REGISTRY:
            print(f"  [skip] unknown alias: {alias}")
            continue
        print(f"\n=== CV for {alias} ({args.folds} folds, GroupKFold by source report) ===")
        per_fold: List[Dict[str, float]] = []
        for fold_idx, (train_i, eval_i) in enumerate(gkf.split(all_data, groups=groups)):
            print(f"\n--- {alias} fold {fold_idx + 1}/{args.folds} "
                  f"(train={len(train_i)}, eval={len(eval_i)}) ---")
            train_data = [all_data[i] for i in train_i]
            eval_data = [all_data[i] for i in eval_i]
            out_dir = OUTPUT_DIR / "cv_runs" / alias / f"fold{fold_idx + 1}"
            try:
                metrics = _train_and_eval(
                    alias, train_data, eval_data, str(out_dir),
                    epochs=args.epochs,
                    batch_size=args.batch_size if BASELINE_REGISTRY[alias]["kind"] == "extractive" else max(1, args.batch_size // 2),
                    learning_rate=args.learning_rate,
                )
            except Exception as e:
                print(f"  [skip] fold failed: {e}")
                continue
            per_fold.append(metrics)
            print(f"  fold {fold_idx + 1} metrics: " +
                  ", ".join(f"{k}={v:.2f}" for k, v in metrics.items()))

        if not per_fold:
            cv_results[alias] = {"error": "all folds failed"}
            continue
        agg: Dict[str, Dict[str, float]] = {}
        for metric in METRIC_FNS:
            vals = [f[metric] for f in per_fold]
            agg[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                           "per_fold": vals}
        cv_results[alias] = {"folds": len(per_fold), "metrics": agg}
        print(f"\n{alias} aggregated:")
        for m, vals in agg.items():
            print(f"  {m:12s} {vals['mean']:.2f} ± {vals['std']:.2f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "cv_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"folds": args.folds, "epochs": args.epochs,
                       "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                       "stratification": "GroupKFold by source report (file)"},
            "results": cv_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")


# ============================================================================
# Subcommand: learning-curve
# ============================================================================

def cmd_learning_curve(args: argparse.Namespace) -> None:
    all_data = load_qa_data(args.data_path)
    train_data, val_data, test_data = split_data(all_data)
    rng = np.random.default_rng(args.seed)

    aliases = args.models or list(BASELINE_REGISTRY.keys())
    fractions = sorted(args.fractions)
    lc_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for alias in aliases:
        if alias not in BASELINE_REGISTRY:
            print(f"  [skip] unknown alias: {alias}")
            continue
        print(f"\n=== Learning curve for {alias} ===")
        per_frac: Dict[str, Dict[str, float]] = {}
        for frac in fractions:
            n = max(1, int(len(train_data) * frac))
            idx = rng.permutation(len(train_data))[:n]
            sub_train = [train_data[i] for i in idx]
            print(f"\n--- {alias} @ {frac:.0%} (n_train={n}) ---")
            out_dir = OUTPUT_DIR / "lc_runs" / alias / f"frac{int(frac * 100)}"
            try:
                metrics = _train_and_eval(
                    alias, sub_train, test_data, str(out_dir),
                    epochs=args.epochs,
                    batch_size=args.batch_size if BASELINE_REGISTRY[alias]["kind"] == "extractive" else max(1, args.batch_size // 2),
                    learning_rate=args.learning_rate,
                )
            except Exception as e:
                print(f"  [skip] fraction failed: {e}")
                continue
            per_frac[f"{frac:.2f}"] = metrics
            print(f"  metrics: " + ", ".join(f"{k}={v:.2f}" for k, v in metrics.items()))
        lc_results[alias] = per_frac

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "learning_curve.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"fractions": fractions, "epochs": args.epochs,
                       "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                       "seed": args.seed},
            "results": lc_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for alias, per_frac in lc_results.items():
                xs = sorted(float(f) for f in per_frac.keys())
                ys = [per_frac[f"{x:.2f}"]["f1_score"] for x in xs]
                ax.plot([x * 100 for x in xs], ys, marker="o", label=alias)
            ax.set_xlabel("Training-set fraction (%)")
            ax.set_ylabel("F1 (%)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig_path = PAPER_FIGURES_DIR / "learning_curve.pdf"
            fig.savefig(fig_path)
            print(f"Saved {fig_path}")
        except ImportError:
            print("  [warn] matplotlib not installed; skipping figure")


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Statistical robustness pipeline (Paper §5.4).")
    p.add_argument("--data-path", default=DATA_PATH)

    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("extract-llm-predictions",
                        help="Reconstruct per-example predictions from the LLM cache (free).")
    s1.add_argument("--results-file", default=None,
                    help="Path to an LLM results JSON (default: data/12-llm_evaluation/llm_results.json). "
                         "Use this to point at e.g. llm_results_openai.json.")

    s2 = sub.add_parser("extract-baseline-predictions",
                        help="Run inference with saved fine-tuned baselines (~5 min/model).")
    s2.add_argument("--models", nargs="+", default=None,
                    help=f"Baseline aliases (default: all). Available: {list(BASELINE_REGISTRY)}")

    s3 = sub.add_parser("bootstrap",
                        help="Bootstrap 95%% CIs over all prediction files in data/13-statistical_robustness/predictions/.")
    s3.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    s3.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    s3.add_argument("--filter", default=None,
                    help="Only bootstrap predictions whose filename contains this substring.")

    s4 = sub.add_parser("cv",
                        help="5-fold cross-validation with GroupKFold by source report (slow).")
    s4.add_argument("--models", nargs="+", default=None,
                    help=f"Baseline aliases (default: all). Available: {list(BASELINE_REGISTRY)}")
    s4.add_argument("--folds", type=int, default=DEFAULT_CV_FOLDS)
    s4.add_argument("--epochs", type=int, default=3)
    s4.add_argument("--batch-size", type=int, default=8)
    s4.add_argument("--learning-rate", type=float, default=3e-5)

    s5 = sub.add_parser("learning-curve",
                        help="Re-train at fractions {10,25,50,100}%% of training (slow).")
    s5.add_argument("--models", nargs="+", default=None,
                    help=f"Baseline aliases (default: all). Available: {list(BASELINE_REGISTRY)}")
    s5.add_argument("--fractions", nargs="+", type=float,
                    default=DEFAULT_LEARNING_CURVE_FRACTIONS)
    s5.add_argument("--epochs", type=int, default=3)
    s5.add_argument("--batch-size", type=int, default=8)
    s5.add_argument("--learning-rate", type=float, default=3e-5)
    s5.add_argument("--seed", type=int, default=42)
    s5.add_argument("--plot", action="store_true",
                    help="Also write Paper/figures/learning_curve.pdf")

    return p


DISPATCH = {
    "extract-llm-predictions": cmd_extract_llm_predictions,
    "extract-baseline-predictions": cmd_extract_baseline_predictions,
    "bootstrap": cmd_bootstrap,
    "cv": cmd_cv,
    "learning-curve": cmd_learning_curve,
}


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DISPATCH[args.cmd](args)


if __name__ == "__main__":
    main()
