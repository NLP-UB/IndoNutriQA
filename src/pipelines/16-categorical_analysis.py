"""
Categorical Analysis Pipeline for IndoNutriQA

Strengthens the experimental analysis by stratifying performance along three
dimensions that reviewers commonly ask for:

1. question-breakdown   -- F1/EM by question category
                            (factual / inferential / recommendation), using
                            a deterministic rule-based mapping over the eight
                            standardized questions (Section 4.2 / Appendix A).

2. disease-breakdown    -- F1/EM by disease category (respiratory, cardiac,
                            nutrition/growth, infectious, neurologic, etc.).
                            Uses GPT-4o-mini to classify each unique
                            medical_diagnosis once; mapping is cached on disk.

3. error-taxonomy       -- LLM-as-judge classifies a sampled subset of model
                            outputs into a fixed taxonomy:
                            {correct, partial_answer, wrong_span,
                             terminology_mismatch, hallucination, noise}.
                            Reports per-category counts and percentages.

All three reuse the per-(model, shots) prediction files at
data/13-statistical_robustness/predictions/, so no re-inference is required.

Usage:
    python src/pipelines/16-categorical_analysis.py question-breakdown
    python src/pipelines/16-categorical_analysis.py disease-breakdown
    python src/pipelines/16-categorical_analysis.py error-taxonomy \\
        --models llm__llama-3.1-8b-instruct__5shot baseline__rifky --n-sample 50

Outputs land in data/16-categorical_analysis/.
"""

import argparse
import hashlib
import json
import os
import random
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass


# ============================================================================
# Configuration
# ============================================================================

QA_PAIRS_PATH = "data/dataset/dataset.json"
PREDICTIONS_DIR = Path("data/13-statistical_robustness/predictions")
OUTPUT_DIR = Path("data/16-categorical_analysis")
DISEASE_MAP_PATH = OUTPUT_DIR / "disease_category_map.json"
TAXONOMY_CACHE_DIR = OUTPUT_DIR / "taxonomy_cache"

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


# ============================================================================
# Question category mapping (deterministic, based on §4.2 / Appendix A)
# ============================================================================

QUESTION_TO_CATEGORY: Dict[str, str] = {
    "Apa metode skrining status gizi yang direkomendasikan?":
        "factual",
    "Bagaimana data biokimia/parameter lab pada pasien (yg abnormal)?":
        "factual",
    "Bagaimana status gizi pada pasien ini?":
        "inferential",
    "Bagaimana menyusun diagnosis gizi dengan format PES pada pasien ini?":
        "inferential",
    "Apa tujuan diet pada pasien ini?":
        "recommendation",
    "Bagaimana prinsip diet yang tepat sesuai dengan kondisi pasien ini?":
        "recommendation",
    "Bagaimana syarat diet yang harus dipenuhi dalam pengaturan diet pasien ini?":
        "recommendation",
    "Apa pedoman diet yang digunakan dalam perencanaan diet pasien ini?":
        "recommendation",
}

QUESTION_CATEGORIES = ["factual", "inferential", "recommendation"]


# ============================================================================
# Disease categories (mirrors the §4.3 treemap)
# ============================================================================

DISEASE_CATEGORIES = [
    "respiratory",
    "cardiac",
    "nutrition_growth",
    "infectious",
    "hematologic_oncologic",
    "neurologic",
    "renal_urologic",
    "gastrointestinal",
    "ent",
    "other",
]

DISEASE_CLASSIFY_PROMPT = (
    "Classify the following Indonesian clinical diagnosis text into exactly "
    "ONE of these categories: respiratory, cardiac, nutrition_growth, "
    "infectious, hematologic_oncologic, neurologic, renal_urologic, "
    "gastrointestinal, ent, other.\n\n"
    "Rules:\n"
    "- If multiple conditions are listed, pick the dominant/primary one.\n"
    "- 'respiratory' covers pneumonia, bronchopneumonia, distress nafas, ARDS, asthma, TB paru, etc.\n"
    "- 'cardiac' covers PJB, gagal jantung, ASD, VSD, perikard, etc.\n"
    "- 'nutrition_growth' covers malnutrition, marasmus, kwashiorkor, growth failure, gizi buruk/kurang.\n"
    "- 'infectious' covers sepsis, dengue (DHF/DBD), bacterial/viral infections not localized to one system.\n"
    "- 'hematologic_oncologic' covers anemia, leukemia, lymphoma, ITP, oncology.\n"
    "- 'neurologic' covers cerebral palsy, hydrocephalus, meningitis, epilepsy, spina bifida.\n"
    "- 'renal_urologic' covers kidney disease, nephrotic/nephritic, urinary.\n"
    "- 'gastrointestinal' covers GEA, gastritis, hematemesis, post GI surgery, hepatic.\n"
    "- 'ent' covers labiopalatoschisis, pharyngitis, tonsil, otitis.\n"
    "- 'other' only when truly unrelated.\n\n"
    "Respond with a JSON object: {\"category\": \"<one_of_the_categories_above>\"}"
)


# ============================================================================
# Error taxonomy
# ============================================================================

ERROR_TAXONOMY = [
    "correct",
    "partial_answer",
    "wrong_span",
    "terminology_mismatch",
    "hallucination",
    "noise",
]

TAXONOMY_PROMPT = (
    "Anda adalah ahli gizi klinis. Anda akan menerima pertanyaan klinis "
    "berbahasa Indonesia, jawaban referensi (RD), dan jawaban kandidat dari "
    "sistem komputer. Klasifikasikan jawaban kandidat ke dalam SATU dari "
    "kategori berikut:\n"
    "- 'correct': pada dasarnya benar dan setara dengan referensi.\n"
    "- 'partial_answer': sebagian benar, namun ada poin klinis kunci yang hilang.\n"
    "- 'wrong_span': kandidat mengekstrak teks yang relevan dengan topik tetapi salah secara faktual.\n"
    "- 'terminology_mismatch': artinya sama dengan referensi tetapi memakai istilah/sinonim berbeda.\n"
    "- 'hallucination': kandidat memuat fakta yang tidak didukung referensi atau berkontradiksi.\n"
    "- 'noise': kandidat kosong, gibberish, atau hanya token kontrol (misal '<extra_id_0>').\n"
    "Jawablah HANYA dengan JSON: {\"label\": \"<one_of_the_categories>\", \"rationale\": \"<short reason in English>\"}."
)


# ============================================================================
# Data loading
# ============================================================================

def load_test_data() -> List[Dict[str, Any]]:
    with open(QA_PAIRS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    val_idx = int(len(data) * 0.85)
    out = []
    for r in data[val_idx:]:
        ans = r.get("answer")
        atxt = ans.get("text", [""])[0] if isinstance(ans, dict) and ans.get("text") else (str(ans) if ans else "")
        if r.get("context") and r.get("question") and atxt:
            out.append({
                "id": r.get("id"),
                "question": r["question"],
                "answer_text": atxt,
                "medical_diagnosis": r.get("medical_diagnosis") or "",
            })
    return out


def load_prediction_files(filter_substring: Optional[str] = None,
                           explicit_models: Optional[List[str]] = None) -> List[Path]:
    if explicit_models:
        files = [PREDICTIONS_DIR / f"{m}.json" for m in explicit_models]
        for f in files:
            if not f.exists():
                raise SystemExit(f"Prediction file not found: {f}")
        return files
    files = sorted(PREDICTIONS_DIR.glob("*.json"))
    if filter_substring:
        files = [f for f in files if filter_substring in f.stem]
    if not files:
        raise SystemExit(f"No prediction files in {PREDICTIONS_DIR}.")
    return files


# ============================================================================
# F1 (mirrors lexical metric used elsewhere in the project)
# ============================================================================

def normalize_answer(s: str) -> str:
    def remove_articles(t):  return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t):  return " ".join(t.split())
    def remove_punc(t):
        ex = set(string.punctuation)
        return "".join(ch for ch in t if ch not in ex)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction: str, truth: str) -> float:
    p = normalize_answer(prediction).split()
    t = normalize_answer(truth).split()
    if not p or not t:
        return float(p == t)
    common = Counter(p) & Counter(t)
    n = sum(common.values())
    if n == 0:
        return 0.0
    pr = n / len(p)
    rc = n / len(t)
    return 2 * pr * rc / (pr + rc)


def compute_em(prediction: str, truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(truth))


# ============================================================================
# Subcommand: question-breakdown
# ============================================================================

def cmd_question_breakdown(args: argparse.Namespace) -> None:
    test = load_test_data()
    files = load_prediction_files(args.filter, args.models)

    out: Dict[str, Any] = {"categories": QUESTION_CATEGORIES, "results": {}}
    for path in files:
        payload = json.loads(path.read_text())
        preds = payload["predictions"]
        truths = payload["ground_truths"]
        if len(preds) != len(test):
            print(f"  [skip] {path.stem}: prediction count {len(preds)} != test count {len(test)}")
            continue

        per_cat_f1: Dict[str, List[float]] = defaultdict(list)
        per_cat_em: Dict[str, List[float]] = defaultdict(list)
        unmapped = 0
        for ex, pred, truth in zip(test, preds, truths):
            cat = QUESTION_TO_CATEGORY.get(ex["question"].strip())
            if cat is None:
                unmapped += 1
                continue
            per_cat_f1[cat].append(compute_f1(pred, truth))
            per_cat_em[cat].append(compute_em(pred, truth))

        block = {}
        for cat in QUESTION_CATEGORIES:
            f1s = per_cat_f1.get(cat, [])
            ems = per_cat_em.get(cat, [])
            block[cat] = {
                "n": len(f1s),
                "f1": float(np.mean(f1s) * 100) if f1s else None,
                "em": float(np.mean(ems) * 100) if ems else None,
            }
        out["results"][path.stem] = block
        print(f"\n{path.stem}")
        for cat in QUESTION_CATEGORIES:
            b = block[cat]
            if b["n"] == 0:
                continue
            print(f"  {cat:14s} (n={b['n']:3d})  F1={b['f1']:6.2f}  EM={b['em']:5.2f}")
        if unmapped:
            print(f"  [warn] {unmapped} examples had unmapped questions")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "question_breakdown.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved {out_path}")


# ============================================================================
# Subcommand: disease-breakdown
# ============================================================================

def _classify_disease(client, judge_model: str, diagnosis_text: str) -> str:
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": DISEASE_CLASSIFY_PROMPT},
            {"role": "user", "content": diagnosis_text},
        ],
        temperature=0.0,
        max_tokens=64,
        response_format={"type": "json_object"},
    )
    txt = (resp.choices[0].message.content or "").strip()
    try:
        obj = json.loads(txt)
        cat = obj.get("category", "").strip().lower()
        return cat if cat in DISEASE_CATEGORIES else "other"
    except Exception:
        return "other"


def build_disease_map(test: List[Dict[str, Any]], judge_model: str) -> Dict[str, str]:
    """One-time classification of each unique diagnosis. Cached on disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, str] = {}
    if DISEASE_MAP_PATH.exists():
        existing = json.loads(DISEASE_MAP_PATH.read_text())

    unique_diags = sorted({(ex["medical_diagnosis"] or "").strip() for ex in test if ex["medical_diagnosis"]})
    todo = [d for d in unique_diags if d not in existing]
    if todo:
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set (check .env).")
        client = OpenAI()
        print(f"Classifying {len(todo)} new diagnoses (already cached: {len(existing)})...")
        for diag in tqdm(todo, desc="  classify"):
            try:
                existing[diag] = _classify_disease(client, judge_model, diag)
            except Exception as e:
                print(f"  [warn] classification failed for {diag[:60]!r}: {e}")
                existing[diag] = "other"
        DISEASE_MAP_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
        print(f"Saved disease-category map to {DISEASE_MAP_PATH}")
    else:
        print(f"Using cached disease-category map ({len(existing)} entries) at {DISEASE_MAP_PATH}")
    return existing


def load_full_data() -> List[Dict[str, Any]]:
    """Every QA pair in the corpus, not just the test split.

    §4.3 of the paper describes the disease mix of the *whole* dataset, so the
    distribution reported there must be computed over all source reports --
    not over the test split, and not over the multi-label keyword matcher in
    notebooks/Data Visualization.ipynb, which assigned a mean of 4.22
    categories per diagnosis and therefore produced percentages summing to
    well over 100%.
    """
    with open(QA_PAIRS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for r in data:
        ans = r.get("answer")
        atxt = ans.get("text", [""])[0] if isinstance(ans, dict) and ans.get("text") else (str(ans) if ans else "")
        if r.get("context") and r.get("question") and atxt:
            out.append({
                "id": r.get("id"),
                "question": r["question"],
                "answer_text": atxt,
                "medical_diagnosis": r.get("medical_diagnosis") or "",
                "file": r.get("file") or "",
            })
    return out


def cmd_disease_distribution(args: argparse.Namespace) -> None:
    """Single-label disease distribution over the FULL corpus, by source report.

    Writes disease_distribution.json and prints the numbers in the form §4.3
    needs. The unit of the headline percentages is the source report, because
    that is what the paper claims to describe; the per-QA-pair distribution is
    reported alongside since the two differ when reports contribute unequal
    numbers of pairs.
    """
    full = load_full_data()
    disease_map = build_disease_map(full, args.judge_model)

    # One diagnosis per source report (a report has a single diagnosis string).
    per_report: Dict[str, str] = {}
    for ex in full:
        per_report.setdefault(ex["file"], (ex["medical_diagnosis"] or "").strip())

    rep_cats = Counter(disease_map.get(d, "other") for d in per_report.values())
    pair_cats = Counter(disease_map.get((ex["medical_diagnosis"] or "").strip(), "other")
                        for ex in full)
    n_rep, n_pair = sum(rep_cats.values()), sum(pair_cats.values())

    print(f"\n=== Disease distribution over ALL {n_rep} source reports "
          f"({n_pair} QA pairs) ===")
    print(f"{'category':<24}{'reports':>9}{'% reports':>11}{'pairs':>8}{'% pairs':>9}")
    rows = []
    for cat in DISEASE_CATEGORIES:
        r, q = rep_cats.get(cat, 0), pair_cats.get(cat, 0)
        if r == 0 and q == 0:
            continue
        pr, pq = 100.0 * r / n_rep, 100.0 * q / n_pair
        rows.append({"category": cat, "reports": r, "pct_reports": pr,
                     "pairs": q, "pct_pairs": pq})
        print(f"{cat:<24}{r:>9}{pr:>10.2f}%{q:>8}{pq:>8.2f}%")
    rows.sort(key=lambda x: -x["reports"])

    out = {
        "method": ("single-label GPT classification of each unique medical_diagnosis "
                   "(one dominant category per report); supersedes the multi-label "
                   "keyword matcher in notebooks/Data Visualization.ipynb"),
        "judge_model": args.judge_model,
        "n_source_reports": n_rep,
        "n_qa_pairs": n_pair,
        "n_unique_diagnoses": len({(e["medical_diagnosis"] or "").strip() for e in full}),
        "by_report": {r["category"]: r["reports"] for r in rows},
        "rows": rows,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "disease_distribution.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved {out_path}")


def cmd_disease_breakdown(args: argparse.Namespace) -> None:
    test = load_test_data()
    disease_map = build_disease_map(test, args.judge_model)

    # Distribution check
    cats = [disease_map.get((ex["medical_diagnosis"] or "").strip(), "other") for ex in test]
    dist = Counter(cats)
    print("\nDisease-category distribution in test set (n={}):".format(len(test)))
    for cat in DISEASE_CATEGORIES:
        n = dist.get(cat, 0)
        if n > 0:
            print(f"  {cat:25s} {n:3d}")

    files = load_prediction_files(args.filter, args.models)
    out: Dict[str, Any] = {"categories": DISEASE_CATEGORIES, "distribution": dict(dist), "results": {}}

    for path in files:
        payload = json.loads(path.read_text())
        preds = payload["predictions"]
        truths = payload["ground_truths"]
        if len(preds) != len(test):
            print(f"  [skip] {path.stem}: prediction count != test count")
            continue
        per_cat_f1: Dict[str, List[float]] = defaultdict(list)
        for ex, pred, truth in zip(test, preds, truths):
            cat = disease_map.get((ex["medical_diagnosis"] or "").strip(), "other")
            per_cat_f1[cat].append(compute_f1(pred, truth))

        block = {}
        for cat in DISEASE_CATEGORIES:
            f1s = per_cat_f1.get(cat, [])
            block[cat] = {
                "n": len(f1s),
                "f1": float(np.mean(f1s) * 100) if f1s else None,
            }
        out["results"][path.stem] = block

        print(f"\n{path.stem}")
        for cat in DISEASE_CATEGORIES:
            b = block[cat]
            if b["n"] == 0:
                continue
            print(f"  {cat:25s} (n={b['n']:3d})  F1={b['f1']:6.2f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "disease_breakdown.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved {out_path}")


# ============================================================================
# Subcommand: error-taxonomy
# ============================================================================

def _taxonomy_cache_key(judge_model: str, question: str, reference: str, prediction: str) -> str:
    h = hashlib.sha256()
    h.update(f"taxonomy|{judge_model}|".encode("utf-8"))
    for s in (question, reference, prediction):
        h.update(b"|||")
        h.update(s.encode("utf-8"))
    return h.hexdigest()


def _classify_error(client, judge_model: str, question: str, reference: str,
                    prediction: str) -> Optional[Tuple[str, str]]:
    user = (f"PERTANYAAN: {question}\n\n"
            f"JAWABAN REFERENSI: {reference}\n\n"
            f"JAWABAN KANDIDAT: {prediction}\n\n"
            f"Berikan JSON sekarang.")
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": TAXONOMY_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    txt = (resp.choices[0].message.content or "").strip()
    try:
        obj = json.loads(txt)
        label = obj.get("label", "").strip().lower()
        rationale = obj.get("rationale", "").strip()
        if label in ERROR_TAXONOMY:
            return label, rationale
    except Exception:
        pass
    return None


def cmd_error_taxonomy(args: argparse.Namespace) -> None:
    if not args.models:
        raise SystemExit("Pass --models <prediction_file_stem> [more...]")

    test = load_test_data()
    files = load_prediction_files(args.filter, args.models)

    rng = random.Random(args.seed)
    sample_idx = sorted(rng.sample(range(len(test)), min(args.n_sample, len(test))))
    print(f"Sampling {len(sample_idx)} examples (seed={args.seed}) from test split (n={len(test)})")

    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (check .env).")
    client = OpenAI()
    TAXONOMY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Any] = {"sampled_indices": sample_idx, "results": {}}
    for path in files:
        payload = json.loads(path.read_text())
        preds = payload["predictions"]
        truths = payload["ground_truths"]
        if len(preds) != len(test):
            print(f"  [skip] {path.stem}: prediction count != test count")
            continue

        labels: List[str] = []
        rationales: List[str] = []
        for i in tqdm(sample_idx, desc=f"  {path.stem}", leave=False):
            ex = test[i]
            pred = preds[i]
            truth = truths[i]
            key = _taxonomy_cache_key(args.judge_model, ex["question"], truth, pred)
            cache_file = TAXONOMY_CACHE_DIR / f"{key}.json"
            if cache_file.exists():
                cached = json.loads(cache_file.read_text())
                labels.append(cached["label"])
                rationales.append(cached.get("rationale", ""))
                continue
            res = _classify_error(client, args.judge_model, ex["question"], truth, pred)
            if res is None:
                labels.append("noise")
                rationales.append("(parse failed)")
                continue
            label, rationale = res
            cache_file.write_text(json.dumps(
                {"label": label, "rationale": rationale,
                 "judge_model": args.judge_model}, ensure_ascii=False))
            labels.append(label)
            rationales.append(rationale)

        counts = Counter(labels)
        n = len(labels)
        block = {
            "n": n,
            "counts": {c: counts.get(c, 0) for c in ERROR_TAXONOMY},
            "percentages": {c: 100.0 * counts.get(c, 0) / n for c in ERROR_TAXONOMY},
        }
        out["results"][path.stem] = block

        print(f"\n{path.stem}  (n={n})")
        for c in ERROR_TAXONOMY:
            print(f"  {c:22s} {block['counts'][c]:3d}  ({block['percentages'][c]:5.1f}%)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "error_taxonomy.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved {out_path}")


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Categorical analysis pipeline for IndoNutriQA.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("question-breakdown",
                        help="F1/EM by question category (factual / inferential / recommendation).")
    s1.add_argument("--filter", default=None)
    s1.add_argument("--models", nargs="+", default=None,
                    help="Explicit prediction-file stems (default: all in predictions/).")

    s2b = sub.add_parser("disease-distribution",
                         help="Single-label disease distribution over the FULL corpus, "
                              "by source report (the numbers section 4.3 needs).")
    s2b.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)

    s2 = sub.add_parser("disease-breakdown",
                        help="F1 by disease category (uses gpt-4o-mini to classify diagnoses; cached).")
    s2.add_argument("--filter", default=None)
    s2.add_argument("--models", nargs="+", default=None)
    s2.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)

    s3 = sub.add_parser("error-taxonomy",
                        help="LLM-judge classifies a sampled subset into 6 error types.")
    s3.add_argument("--models", nargs="+", required=True,
                    help="Prediction-file stems to analyse (e.g. baseline__rifky).")
    s3.add_argument("--filter", default=None)
    s3.add_argument("--n-sample", type=int, default=50)
    s3.add_argument("--seed", type=int, default=42)
    s3.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)

    return p


DISPATCH = {
    "question-breakdown": cmd_question_breakdown,
    "disease-distribution": cmd_disease_distribution,
    "disease-breakdown":  cmd_disease_breakdown,
    "error-taxonomy":     cmd_error_taxonomy,
}


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DISPATCH[args.cmd](args)


if __name__ == "__main__":
    main()
