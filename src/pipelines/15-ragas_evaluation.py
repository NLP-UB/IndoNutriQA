"""
RAGAS Evaluation Pipeline for IndoNutriQA

Adds RAG-specific evaluation metrics for the retrieval-augmented baseline
reported in Table 3 of the paper. Addresses the reviewer concern that lexical
metrics (the only RAG numbers currently in §5.3) cannot diagnose *why* RAG
performs poorly — whether the retrieval misses, the generator hallucinates,
or both.

We compute the four standard RAGAS metrics (faithfulness, answer relevancy,
context precision, context recall) on the same 137-example test split used
elsewhere in the paper. Each metric uses an LLM internally; we default to
gpt-4o-mini to stay consistent with §5.5 (LLM-as-judge) and to keep cost low.

Inputs (must already exist):
- data/11-retrieval_augmented_QA/rag_results.json   (per-example RAG outputs)
- data/11-retrieval_augmented_QA/chunk_metadata.json (full chunk text used to
                                                      hydrate truncated text_previews)

Output:
- data/15-ragas_evaluation/ragas_results.json
    {
      "config": {...},
      "n": 137,
      "per_example": [{"id": 774, "faithfulness": 0.8, ...}, ...],
      "metrics": {
        "faithfulness":      {"point": ..., "ci_low": ..., "ci_high": ...},
        "answer_relevancy":  {...},
        "context_precision": {...},
        "context_recall":    {...}
      }
    }

Usage:
    pip install ragas langchain-openai datasets
    python src/pipelines/15-ragas_evaluation.py --limit 5   # smoke test
    python src/pipelines/15-ragas_evaluation.py             # full test split
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# .env at project root
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass


# ============================================================================
# Configuration
# ============================================================================

QA_PAIRS_PATH = "data/dataset/dataset.json"
RAG_RESULTS_PATH = "data/11-retrieval_augmented_QA/rag_results.json"
CHUNK_METADATA_PATH = "data/11-retrieval_augmented_QA/chunk_metadata.json"
OUTPUT_DIR = Path("data/15-ragas_evaluation")
OUTPUT_PATH = OUTPUT_DIR / "ragas_results.json"

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 42


# ============================================================================
# Data assembly
# ============================================================================

def load_test_split_ids() -> List[int]:
    """Return the IDs that fall in the test split (last 15%) of qa_pairs_with_chunks.json,
    matching the convention used by 9-context_based_QA.py and 12-llm_evaluation.py."""
    with open(QA_PAIRS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    val_idx = int(len(data) * 0.85)
    return [int(record.get("id")) for record in data[val_idx:] if record.get("id") is not None]


def build_chunk_lookup() -> Dict[Tuple[str, int, int], str]:
    """Map (file, page_number, chunk_index) -> full chunk text."""
    with open(CHUNK_METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        (m["file"], int(m["page_number"]), int(m["chunk_index"])): m["text"]
        for m in meta
    }


def assemble_ragas_examples(limit: int = None) -> List[Dict[str, Any]]:
    """Join rag_results with full chunk text and filter to the test split.

    Returns a list of dicts with keys:
        id, question, contexts (list of full chunk strings), answer, ground_truth.
    """
    test_ids = set(load_test_split_ids())
    chunk_lookup = build_chunk_lookup()

    with open(RAG_RESULTS_PATH, "r", encoding="utf-8") as f:
        rag = json.load(f)

    examples: List[Dict[str, Any]] = []
    n_missing_chunks = 0
    for r in rag:
        rid = int(r.get("id"))
        if rid not in test_ids:
            continue

        contexts: List[str] = []
        for c in r.get("retrieved_chunks", []):
            key = (c["file"], int(c["page"]), int(c["chunk_idx"]))
            text = chunk_lookup.get(key)
            if text is None:
                n_missing_chunks += 1
                # Fall back to the truncated preview if metadata join fails.
                text = c.get("text_preview", "")
            contexts.append(text)

        examples.append({
            "id": rid,
            "question": r["question"],
            "contexts": contexts,
            "answer": r["generated_answer"],
            "ground_truth": r["ground_truth"],
        })

    if n_missing_chunks:
        print(f"[warn] {n_missing_chunks} retrieved chunks not found in metadata; used truncated preview")
    if limit is not None:
        examples = examples[:limit]
    return examples


# ============================================================================
# Bootstrap CI
# ============================================================================

def bootstrap_ci(per_example: np.ndarray, n_resamples: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(per_example)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(per_example.mean())
    sample_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample_means[i] = per_example[idx].mean()
    return point, float(np.percentile(sample_means, 2.5)), float(np.percentile(sample_means, 97.5))


# ============================================================================
# RAGAS evaluation
# ============================================================================

def run_ragas(
    examples: List[Dict[str, Any]],
    judge_model: str,
    embedding_model: str,
) -> Tuple["object", List[str]]:
    """Evaluate RAGAS metrics; return (raw_result, list_of_metric_names)."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency: {e}\n"
            f"Install with:\n"
            f"  pip install ragas langchain-openai datasets"
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (check .env).")

    print(f"\nBuilding HF dataset from {len(examples)} examples...")
    ds = Dataset.from_dict({
        "question":     [e["question"]     for e in examples],
        "contexts":     [e["contexts"]     for e in examples],
        "answer":       [e["answer"]       for e in examples],
        "ground_truth": [e["ground_truth"] for e in examples],
    })

    llm = ChatOpenAI(model=judge_model, temperature=0)
    emb = OpenAIEmbeddings(model=embedding_model)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print(f"Running RAGAS with judge={judge_model}, embeddings={embedding_model}...")
    result = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb)
    return result, metric_names


def to_per_example_table(result, metric_names: List[str], examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pull per-row metric values out of the RAGAS Result regardless of API version."""
    # RAGAS returns a Result that exposes a pandas DataFrame via .to_pandas()
    df = result.to_pandas()
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples):
        row: Dict[str, Any] = {"id": ex["id"]}
        for m in metric_names:
            v = df.iloc[i][m] if m in df.columns else None
            try:
                row[m] = float(v) if v is not None else None
            except (TypeError, ValueError):
                row[m] = None
        rows.append(row)
    return rows


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAGAS evaluation for the IndoNutriQA RAG baseline.")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                   help="LLM used internally by RAGAS (default: gpt-4o-mini).")
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                   help="Embedding model used by RAGAS for answer_relevancy.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of examples (smoke test).")
    p.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    p.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    p.add_argument("--output", default=str(OUTPUT_PATH),
                   help="Where to write the consolidated RAGAS results JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("INDONUTRIQA - RAGAS EVALUATION")
    print("=" * 70)

    examples = assemble_ragas_examples(limit=args.limit)
    print(f"Assembled {len(examples)} test-split examples with retrieved contexts.")
    if not examples:
        raise SystemExit("No examples assembled; check that rag_results.json contains test-split IDs.")

    result, metric_names = run_ragas(examples, args.judge_model, args.embedding_model)
    per_example = to_per_example_table(result, metric_names, examples)

    print("\n" + "=" * 70)
    print(f"RAGAS results (n={len(examples)})")
    print("=" * 70)

    metrics_block: Dict[str, Any] = {}
    for m in metric_names:
        values = np.array([r[m] for r in per_example if r.get(m) is not None], dtype=float)
        if len(values) == 0:
            metrics_block[m] = {"point": None, "ci_low": None, "ci_high": None,
                                "n_valid": 0, "n_missing": len(per_example)}
            print(f"  {m:20s} (no valid scores)")
            continue
        point, lo, hi = bootstrap_ci(values, args.resamples, args.seed)
        metrics_block[m] = {
            "point": point, "ci_low": lo, "ci_high": hi,
            "n_valid": int(len(values)),
            "n_missing": int(len(per_example) - len(values)),
        }
        # RAGAS metrics are 0-1; render as percent for table compatibility.
        print(f"  {m:20s} {point*100:6.2f}  [{lo*100:5.2f}, {hi*100:5.2f}]  (n={len(values)})")

    summary = {
        "config": {
            "judge_model": args.judge_model,
            "embedding_model": args.embedding_model,
            "n_resamples": args.resamples,
            "seed": args.seed,
            "rag_results_source": RAG_RESULTS_PATH,
        },
        "n": len(examples),
        "metrics": metrics_block,
        "per_example": per_example,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
