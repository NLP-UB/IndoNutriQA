"""
RD / LLM-Judge Agreement Pipeline for IndoNutriQA

Addresses the Limitations item that the LLM-as-judge rubric of
14-semantic_evaluation.py is only cross-validated *judge-against-judge*
(GPT-4o-mini vs Claude Haiku 4.5, pooled kappa = 0.652) and never against a
Registered Dietitian. This pipeline builds the human-anchored arm:

1. sample          -- select N (test example, system prediction) units for RD
                      annotation, stratified by question category
                      (factual / inferential / recommendation) x system
                      quality tier, so the sample spans the full answer-quality
                      range. Emits a blinded Label Studio task file, an offline
                      CSV scoring sheet, and an internal keyfile holding the
                      cached judge scores.

2. export-tasks    -- regenerate the Label Studio task JSON from an existing
                      sample (e.g. after editing the labeling config).

3. import          -- parse a Label Studio export (JSON or JSON-MIN) back into
                      a tidy rd_annotations.json keyed by unit_id x rater.

4. agreement       -- compute inter-RD and judge-vs-RD agreement, stratified by
                      question category:
                        * Cohen's kappa (unweighted / linear / quadratic)
                        * binarised kappa at the "clinically acceptable" cut
                        * percent agreement and PABAK (prevalence-adjusted)
                        * Krippendorff's alpha (ordinal) for >= 2 raters
                        * mean-score difference (judge leniency/severity bias)
                      all with bootstrap 95% CIs over annotation units.

5. report          -- render the agreement JSON as a LaTeX table for the paper.

The judge scores are read from the existing on-disk cache at
data/14-semantic_evaluation/judge_cache/, so this pipeline costs ZERO API
calls: sampling only proposes units whose judge scores are already cached for
every judge under study.

Usage:
    # 1. Draw the 50-unit sample (writes tasks + sheet + keyfile)
    python src/pipelines/18-rd_judge_agreement.py sample --n 50

    # 2. (RDs annotate in Label Studio -- see docs/rd_annotation/README.md)

    # 3. Parse the export
    python src/pipelines/18-rd_judge_agreement.py import \\
        --export label_studio_export.json \\
        --rater-map rd1@hospital.id=RD1 rd2@hospital.id=RD2

    # 4. Agreement statistics
    python src/pipelines/18-rd_judge_agreement.py agreement

    # 5. LaTeX table for the appendix
    python src/pipelines/18-rd_judge_agreement.py report

Before the RDs have finished, the whole chain can be smoke-tested end to end:
    python src/pipelines/18-rd_judge_agreement.py agreement --simulate

Outputs land in data/18-rd_judge_agreement/.
"""

import argparse
import csv
import hashlib
import json
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================================
# Configuration
# ============================================================================

QA_PAIRS_PATH = "data/dataset/dataset.json"
PREDICTIONS_DIR = Path("data/13-statistical_robustness/predictions")
JUDGE_CACHE_DIR = Path("data/14-semantic_evaluation/judge_cache")
OUTPUT_DIR = Path("data/18-rd_judge_agreement")

SAMPLE_PATH = OUTPUT_DIR / "annotation_sample.json"          # internal (unblinded)
TASKS_PATH = OUTPUT_DIR / "label_studio_tasks.json"          # blinded, for LS
SHEET_PATH = OUTPUT_DIR / "scoring_sheet.csv"                # offline fallback
ANNOTATIONS_PATH = OUTPUT_DIR / "rd_annotations.json"        # parsed LS export
AGREEMENT_PATH = OUTPUT_DIR / "agreement.json"
LATEX_PATH = OUTPUT_DIR / "agreement_table.tex"

DIMENSIONS = ["factuality", "completeness", "safety"]
LIKERT = [1, 2, 3, 4, 5]

# "Clinically acceptable" cut for the binarised kappa. A 4 means the answer
# would pass RD review with at most cosmetic correction; 3 or below means it
# would not be released to a clinician as-is.
ACCEPTABLE_THRESHOLD = 4

# Judges whose scores must already be cached for a unit to be sampleable.
DEFAULT_JUDGES = ["gpt-4o-mini", "claude-haiku-4-5"]

DEFAULT_N_UNITS = 50
DEFAULT_SEED = 42
DEFAULT_BOOTSTRAP_RESAMPLES = 2000

# System quality tiers. Tiers are assigned data-driven at sample time by
# ranking every prediction file on mean token-F1 and cutting at tertiles, so
# the sample spans weak / medium / strong systems without hard-coding names.
TIERS = ["low", "mid", "high"]


# Mirrors 16-categorical_analysis.py exactly so the stratification in this
# pipeline is the same one used for the stratified-performance tables.
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
# Data loading (identical 70/15/15 split as 13-statistical_robustness.py)
# ============================================================================

def load_qa_data(path: str = QA_PAIRS_PATH) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: List[Dict[str, Any]] = []
    for record in data:
        context = record.get("context")
        question = record.get("question")
        answer = record.get("answer")
        if isinstance(answer, dict):
            atxt = answer.get("text", [""])[0] if answer.get("text") else ""
        else:
            atxt = str(answer) if answer else ""
        if context and question and atxt:
            out.append({
                "id": record.get("id"),
                "context": context,
                "question": question,
                "answer_text": atxt,
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
# Token-F1 (mirrors 13-statistical_robustness.py) -- used only for tiering
# ============================================================================

def normalize_answer(s: str) -> str:
    def remove_articles(t):  return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t):  return " ".join(t.split())
    def remove_punc(t):
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)
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
    precision = n / len(p)
    recall = n / len(t)
    return 2 * precision * recall / (precision + recall)


# ============================================================================
# Judge cache access (key derivation mirrors 14-semantic_evaluation.py)
# ============================================================================

def judge_cache_key(judge_model: str, reference: str, prediction: str) -> str:
    h = hashlib.sha256()
    h.update(f"judge|{judge_model}|".encode("utf-8"))
    h.update(reference.encode("utf-8"))
    h.update(b"|||")
    h.update(prediction.encode("utf-8"))
    return h.hexdigest()


def judge_cache_get(judge_model: str, reference: str, prediction: str) -> Optional[Dict[str, int]]:
    p = JUDGE_CACHE_DIR / f"{judge_cache_key(judge_model, reference, prediction)}.json"
    if not p.exists():
        return None
    try:
        scores = json.loads(p.read_text()).get("scores")
    except Exception:
        return None
    if not scores or any(d not in scores for d in DIMENSIONS):
        return None
    return {d: int(scores[d]) for d in DIMENSIONS}


# ============================================================================
# Subcommand: sample
# ============================================================================

def _load_prediction_files() -> Dict[str, Dict[str, Any]]:
    if not PREDICTIONS_DIR.exists():
        raise SystemExit(f"{PREDICTIONS_DIR} does not exist; run 13-statistical_robustness.py first.")
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(PREDICTIONS_DIR.glob("*.json")):
        out[path.stem] = json.loads(path.read_text())
    if not out:
        raise SystemExit(f"No prediction files in {PREDICTIONS_DIR}.")
    return out


def _assign_tiers(preds: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Rank systems by mean token-F1 and cut at tertiles into low/mid/high."""
    scores: List[Tuple[str, float]] = []
    for stem, payload in preds.items():
        f1s = [compute_f1(p, t) for p, t in zip(payload["predictions"], payload["ground_truths"])]
        scores.append((stem, float(np.mean(f1s)) if f1s else 0.0))
    scores.sort(key=lambda x: x[1])
    n = len(scores)
    tier_of: Dict[str, str] = {}
    for i, (stem, _) in enumerate(scores):
        # Tertile by rank position; the top third is "high".
        tier_of[stem] = TIERS[min(2, (i * 3) // n)]
    return tier_of


def _allocate(total: int, buckets: Sequence[str]) -> Dict[str, int]:
    """Largest-remainder allocation of `total` across `buckets`, as even as possible."""
    k = len(buckets)
    base, rem = divmod(total, k)
    return {b: base + (1 if i < rem else 0) for i, b in enumerate(buckets)}


def cmd_sample(args: argparse.Namespace) -> None:
    all_data = load_qa_data()
    _, _, test_data = split_data(all_data)
    preds = _load_prediction_files()
    tier_of = _assign_tiers(preds)
    judges = args.judges

    print(f"[sample] test split n={len(test_data)}, systems={len(preds)}, judges={judges}")
    tier_members = defaultdict(list)
    for stem, tier in tier_of.items():
        tier_members[tier].append(stem)
    for tier in TIERS:
        print(f"  tier {tier:<4} ({len(tier_members[tier])}): {', '.join(sorted(tier_members[tier]))}")

    # ---- Build the candidate pool -----------------------------------------
    # A candidate is a (test item, system) pair whose prediction is non-empty
    # and whose scores are already cached for EVERY judge under study.
    candidates: List[Dict[str, Any]] = []
    n_missing_judge = 0
    for stem, payload in preds.items():
        if len(payload["predictions"]) != len(test_data):
            print(f"  [warn] {stem}: n={len(payload['predictions'])} != test n={len(test_data)}; skipping")
            continue
        for i, (pred, truth) in enumerate(zip(payload["predictions"], payload["ground_truths"])):
            if not pred or not pred.strip():
                continue
            js = {j: judge_cache_get(j, truth, pred) for j in judges}
            if any(v is None for v in js.values()):
                n_missing_judge += 1
                continue
            item = test_data[i]
            candidates.append({
                "test_index": i,
                "system": stem,
                "tier": tier_of[stem],
                "question": item["question"],
                "question_category": QUESTION_TO_CATEGORY.get(item["question"], "unknown"),
                "file": item["file"],
                "qa_id": item["id"],
                "context": item["context"],
                "anthropometric_assessment": item["anthropometric_assessment"],
                "medical_diagnosis": item["medical_diagnosis"],
                "reference": truth,
                "prediction": pred,
                "judge_scores": js,
            })
    print(f"[sample] candidate pool: {len(candidates)} "
          f"({n_missing_judge} skipped for missing judge-cache entries)")
    if not candidates:
        raise SystemExit("Empty candidate pool. Run 14-semantic_evaluation.py llm-judge first.")

    # ---- Stratified allocation --------------------------------------------
    # Equal allocation across question categories maximises the minimum
    # per-stratum precision, which is the point of the study (the hypothesis
    # is that recommendation-style questions diverge most). Within a category,
    # units are spread evenly across the three system tiers so that the sample
    # covers the full quality range.
    #
    # NOTE: stratification is on question category and *aggregate* system
    # quality, never on the per-unit judge score. Conditioning on the judge
    # score would bias the very quantity being estimated.
    per_category = _allocate(args.n, QUESTION_CATEGORIES)
    plan: Dict[Tuple[str, str], int] = {}
    for cat, n_cat in per_category.items():
        for tier, n_cell in _allocate(n_cat, TIERS).items():
            plan[(cat, tier)] = n_cell

    by_cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_cell[(c["question_category"], c["tier"])].append(c)

    rng = random.Random(args.seed)
    selected: List[Dict[str, Any]] = []
    selected_keys: set = set()       # (test_index, system) already taken
    used_items: set = set()          # each unit uses a distinct test item
    file_usage: Counter = Counter()  # spread over source reports

    for cell in sorted(plan):
        need = plan[cell]
        pool = list(by_cell.get(cell, []))
        if len(pool) < need:
            print(f"  [warn] cell {cell} has only {len(pool)} candidates for {need} slots")
        rng.shuffle(pool)
        taken = 0
        # Two passes: first honour the distinct-item constraint, then relax it
        # only if a cell cannot otherwise be filled.
        for enforce_distinct in (True, False):
            if taken >= need:
                break
            # Prefer candidates from the least-used source report.
            pool.sort(key=lambda c: file_usage[c["file"]])
            for c in pool:
                if taken >= need:
                    break
                key = (c["test_index"], c["system"])
                if key in selected_keys:
                    continue
                if enforce_distinct and c["test_index"] in used_items:
                    continue
                selected.append(c)
                selected_keys.add(key)
                used_items.add(c["test_index"])
                file_usage[c["file"]] += 1
                taken += 1

    if len(selected) < args.n:
        print(f"  [warn] filled {len(selected)}/{args.n} slots")

    # Randomise presentation order so tier is uncorrelated with position in
    # the annotation queue (guards against annotator drift/fatigue effects).
    rng.shuffle(selected)
    for k, unit in enumerate(selected, start=1):
        unit["unit_id"] = f"U{k:03d}"

    # ---- Persist -----------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_payload = {
        "config": {
            "n_units": len(selected),
            "seed": args.seed,
            "judges": judges,
            "stratification": "question_category x system_tier (tertiles of mean token-F1)",
            "distinct_test_items": len(used_items),
            "distinct_source_reports": len(file_usage),
            "acceptable_threshold": ACCEPTABLE_THRESHOLD,
            "dimensions": DIMENSIONS,
        },
        "plan": {f"{c}|{t}": n for (c, t), n in sorted(plan.items())},
        "units": selected,
    }
    SAMPLE_PATH.write_text(json.dumps(sample_payload, indent=2, ensure_ascii=False))
    print(f"\n[sample] wrote {SAMPLE_PATH}  ({len(selected)} units, "
          f"{len(used_items)} distinct items, {len(file_usage)} source reports)")

    _write_tasks(selected, include_metadata=not args.no_context)
    _write_sheet(selected)

    # ---- Sanity report -----------------------------------------------------
    print("\n[sample] composition")
    comp = Counter((u["question_category"], u["tier"]) for u in selected)
    for cat in QUESTION_CATEGORIES:
        row = " ".join(f"{t}={comp[(cat, t)]}" for t in TIERS)
        print(f"  {cat:<15} {row}   total={sum(comp[(cat, t)] for t in TIERS)}")

    print("\n[sample] cached judge-score distribution over the drawn sample")
    for j in judges:
        for dim in DIMENSIONS:
            dist = Counter(u["judge_scores"][j][dim] for u in selected)
            spread = " ".join(f"{v}:{dist.get(v, 0)}" for v in LIKERT)
            print(f"  {j:<20} {dim:<13} {spread}")
    _warn_on_prevalence(selected, judges)


def _warn_on_prevalence(units: List[Dict[str, Any]], judges: List[str]) -> None:
    """Flag dimensions where the judge marginal is so skewed that kappa will be
    depressed by the prevalence problem regardless of true agreement."""
    flagged = []
    for j in judges:
        for dim in DIMENSIONS:
            dist = Counter(u["judge_scores"][j][dim] for u in units)
            top = max(dist.values()) / max(1, len(units))
            if top >= 0.80:
                flagged.append((j, dim, top))
    if flagged:
        print("\n[sample] PREVALENCE WARNING")
        for j, dim, top in flagged:
            print(f"  {j} / {dim}: {top:.0%} of units share one score. Cohen's kappa "
                  f"will be depressed here even under near-perfect agreement;")
        print("  report percent agreement and PABAK alongside kappa for these "
              "dimensions (the `agreement` command does this automatically).")


# ============================================================================
# Subcommand: export-tasks
# ============================================================================

def _write_tasks(units: List[Dict[str, Any]], include_metadata: bool) -> None:
    """Write the BLINDED Label Studio task file.

    Blinding is the point: the task carries no system name, no tier, and no
    judge score, so an RD cannot anchor on them. The join back to that
    metadata happens through unit_id against annotation_sample.json.

    The full source chunk is never emitted -- labeling_config.xml shows the
    structured clinical metadata (diagnosis, anthropometry) instead, which is
    what an RD actually needs to judge a nutrition answer. `include_metadata`
    False drops those two fields too, giving the RD exactly the judge's
    information (strict parity); pair it with the trimmed labeling config.
    """
    tasks = []
    for u in units:
        data = {
            "unit_id": u["unit_id"],
            "question": u["question"],
            "reference": u["reference"],
            "candidate": u["prediction"],
        }
        if include_metadata:
            data["medical_diagnosis"] = u["medical_diagnosis"] or "-"
            data["anthropometric_assessment"] = u["anthropometric_assessment"] or "-"
        tasks.append({"data": data})
    TASKS_PATH.write_text(json.dumps(tasks, indent=2, ensure_ascii=False))
    mode = ("with diagnosis + anthropometry" if include_metadata
            else "reference+candidate only (strict judge parity)")
    print(f"[sample] wrote {TASKS_PATH}  ({len(tasks)} tasks, {mode})")


def _write_sheet(units: List[Dict[str, Any]]) -> None:
    """Offline CSV scoring sheet: the same rubric fields, for RDs who would
    rather work in a spreadsheet than in Label Studio."""
    with open(SHEET_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "unit_id", "diagnosis_medis", "asesmen_antropometri", "pertanyaan",
            "jawaban_referensi_RD", "jawaban_kandidat_sistem",
            "faktualitas_1_5", "kelengkapan_1_5", "keamanan_klinis_1_5",
            "referensi_bermasalah_y_n", "catatan",
        ])
        for u in units:
            w.writerow([u["unit_id"], u["medical_diagnosis"] or "-",
                        u["anthropometric_assessment"] or "-", u["question"],
                        u["reference"], u["prediction"],
                        "", "", "", "", ""])
    print(f"[sample] wrote {SHEET_PATH}  (offline fallback sheet)")


def cmd_export_tasks(args: argparse.Namespace) -> None:
    if not SAMPLE_PATH.exists():
        raise SystemExit(f"{SAMPLE_PATH} not found; run `sample` first.")
    payload = json.loads(SAMPLE_PATH.read_text())
    _write_tasks(payload["units"], include_metadata=not args.no_context)
    _write_sheet(payload["units"])


# ============================================================================
# Subcommand: import (parse a Label Studio export)
# ============================================================================

def _parse_rater(annotation: Dict[str, Any], rater_map: Dict[str, str]) -> str:
    """Resolve an annotation to a rater label. Label Studio writes
    `completed_by` as either an int user id or a nested user object depending
    on the export setting; handle both, then apply --rater-map."""
    cb = annotation.get("completed_by")
    raw: Optional[str] = None
    if isinstance(cb, dict):
        raw = cb.get("email") or cb.get("username") or str(cb.get("id"))
    elif cb is not None:
        raw = str(cb)
    if raw is None:
        raw = str(annotation.get("id", "unknown"))
    return rater_map.get(raw, raw)


def _extract_scores(result: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pull the rubric fields out of a Label Studio `result` array."""
    out: Dict[str, Any] = {}
    for r in result:
        name = r.get("from_name")
        value = r.get("value", {})
        if name in DIMENSIONS:
            choices = value.get("choices") or []
            if choices:
                try:
                    out[name] = int(str(choices[0]).strip())
                except ValueError:
                    pass
        elif name == "ref_problem":
            out["ref_problem"] = bool(value.get("choices"))
        elif name == "comment":
            texts = value.get("text") or []
            if texts:
                out["comment"] = texts[0]
    return out


def cmd_import(args: argparse.Namespace) -> None:
    rater_map: Dict[str, str] = {}
    for spec in args.rater_map or []:
        if "=" not in spec:
            raise SystemExit(f"--rater-map entry must be key=label, got {spec!r}")
        k, v = spec.split("=", 1)
        rater_map[k.strip()] = v.strip()

    # Each --export entry is either `path` (attribute by `completed_by`, for a
    # single project annotated by everyone) or `LABEL=path` (attribute the
    # whole file to LABEL, for the one-project-per-RD layout that Label Studio
    # Community forces when you need blinding).
    sources: List[Tuple[Optional[str], Path]] = []
    for spec in args.export:
        if "=" in spec and not Path(spec).exists():
            label, path = spec.split("=", 1)
            sources.append((label.strip(), Path(path.strip())))
        else:
            sources.append((None, Path(spec)))

    # unit_id -> rater -> scores
    records: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    n_ann = 0
    collisions: List[Tuple[str, str]] = []

    for forced_label, path in sources:
        if not path.exists():
            raise SystemExit(f"export not found: {path}")
        export = json.loads(path.read_text())
        if not isinstance(export, list):
            raise SystemExit(f"{path}: expected a Label Studio export that is a JSON array of tasks.")
        n_file = 0

        for task in export:
            data = task.get("data", task)
            unit_id = data.get("unit_id")
            if unit_id is None:
                continue

            if "annotations" in task:                 # full JSON export
                for ann in task["annotations"]:
                    if ann.get("was_cancelled"):
                        continue
                    scores = _extract_scores(ann.get("result", []))
                    if not any(d in scores for d in DIMENSIONS):
                        continue
                    rater = forced_label or _parse_rater(ann, rater_map)
                    if rater in records[unit_id]:
                        collisions.append((unit_id, rater))
                    records[unit_id][rater] = scores
                    n_ann += 1
                    n_file += 1
            else:                                     # JSON-MIN export
                scores = {}
                for d in DIMENSIONS:
                    v = task.get(d)
                    if isinstance(v, dict):
                        v = (v.get("choices") or [None])[0]
                    if v is not None:
                        try:
                            scores[d] = int(str(v).strip())
                        except ValueError:
                            pass
                if not scores:
                    continue
                raw = str(task.get("annotator", "unknown"))
                rater = forced_label or rater_map.get(raw, raw)
                if rater in records[unit_id]:
                    collisions.append((unit_id, rater))
                records[unit_id][rater] = scores
                n_ann += 1
                n_file += 1

        label_note = f" as {forced_label}" if forced_label else ""
        print(f"[import] {path.name}: {n_file} annotation(s){label_note}")

    if collisions:
        print(f"  [warn] {len(collisions)} duplicate (unit, rater) pair(s) — later file won. "
              f"First few: {collisions[:5]}")
        print("         If you exported one project per RD, check you did not pass the "
              "same file twice or reuse a label.")

    raters = sorted({r for v in records.values() for r in v})
    payload = {
        "source_exports": [str(p) for _, p in sources],
        "n_units": len(records),
        "n_annotations": n_ann,
        "raters": raters,
        "annotations": records,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[import] {n_ann} annotations over {len(records)} units by raters {raters}")
    print(f"[import] wrote {ANNOTATIONS_PATH}")

    coverage = Counter(len(v) for v in records.values())
    for k in sorted(coverage):
        print(f"  {coverage[k]} unit(s) annotated by {k} rater(s)")
    if any(k < 2 for k in coverage):
        print("  [warn] units with <2 raters contribute to judge-vs-RD but not to inter-RD kappa.")


# ============================================================================
# Agreement statistics
# ============================================================================

def _confusion(a: Sequence[int], b: Sequence[int], categories: Sequence[int]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(categories)}
    m = np.zeros((len(categories), len(categories)), dtype=float)
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    return m


def cohen_kappa(a: Sequence[int], b: Sequence[int],
                categories: Sequence[int] = tuple(LIKERT),
                weights: Optional[str] = None) -> float:
    """Cohen's kappa with optional linear / quadratic weights.

    Implemented directly (rather than via sklearn) so that the category set is
    pinned to the full 1-5 Likert scale even when a rater never uses some
    level -- otherwise the expected-agreement term shifts between strata and
    the numbers stop being comparable across the table.
    """
    if len(a) == 0:
        return float("nan")
    k = len(categories)
    m = _confusion(a, b, categories)
    n = m.sum()
    observed = m / n
    row = observed.sum(axis=1)
    col = observed.sum(axis=0)
    expected = np.outer(row, col)

    if weights is None:
        w = 1.0 - np.eye(k)
    else:
        i = np.arange(k).reshape(-1, 1)
        j = np.arange(k).reshape(1, -1)
        if weights == "linear":
            w = np.abs(i - j) / (k - 1)
        elif weights == "quadratic":
            w = ((i - j) ** 2) / ((k - 1) ** 2)
        else:
            raise ValueError(f"unknown weights {weights!r}")

    denom = float((w * expected).sum())
    if denom == 0:
        # Perfect expected agreement (both raters constant on the same level).
        return float("nan")
    return float(1.0 - (w * observed).sum() / denom)


def percent_agreement(a: Sequence[int], b: Sequence[int]) -> float:
    if not len(a):
        return float("nan")
    return float(np.mean([x == y for x, y in zip(a, b)]))


def pabak(a: Sequence[int], b: Sequence[int], k: int = len(LIKERT)) -> float:
    """Prevalence-adjusted bias-adjusted kappa: (k*p_o - 1)/(k - 1).

    Reported alongside kappa because the safety dimension is extremely skewed
    (the judges score it 5 on the large majority of units), which drives
    Cohen's kappa toward 0 even when raters agree on nearly every unit.
    """
    po = percent_agreement(a, b)
    if np.isnan(po):
        return float("nan")
    return float((k * po - 1) / (k - 1))


def krippendorff_alpha_ordinal(matrix: np.ndarray,
                               categories: Sequence[int] = tuple(LIKERT)) -> float:
    """Krippendorff's alpha with the ordinal difference function.

    `matrix` is units x raters with np.nan for missing ratings. Handles any
    number of raters and unbalanced coverage, which Cohen's kappa cannot.
    """
    cats = list(categories)
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}

    # Coincidence matrix over units with >= 2 ratings: every ordered pair of
    # *distinct rater positions* within a unit, weighted 1/(m_u - 1).
    coincidence = np.zeros((k, k), dtype=float)
    for row in matrix:
        vals = [int(v) for v in row if not np.isnan(v)]
        mu = len(vals)
        if mu < 2:
            continue
        for p in range(mu):
            for q in range(mu):
                if p == q:
                    continue
                coincidence[idx[vals[p]], idx[vals[q]]] += 1.0 / (mu - 1)

    n_c = coincidence.sum(axis=1)
    n = float(n_c.sum())
    if n < 2:
        return float("nan")

    # Ordinal delta: squared sum of marginals strictly between c and k,
    # plus half of each endpoint marginal.
    delta = np.zeros((k, k), dtype=float)
    for c in range(k):
        for kk in range(k):
            lo, hi = min(c, kk), max(c, kk)
            s = n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2.0
            delta[c, kk] = s ** 2

    d_o = float((coincidence * delta).sum() / n)
    expected = np.outer(n_c, n_c) - np.diag(n_c)
    d_e = float((expected * delta).sum() / (n * (n - 1)))
    if d_e == 0:
        return float("nan")
    return float(1.0 - d_o / d_e)


def _bootstrap_ci(fn, arrays: Sequence[np.ndarray], n_resamples: int, seed: int
                  ) -> Tuple[float, float]:
    """Percentile bootstrap over annotation units (the resampling unit is the
    unit, so paired rater scores move together)."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    if n == 0:
        return float("nan"), float("nan")
    vals = []
    for _ in range(n_resamples):
        i = rng.integers(0, n, size=n)
        v = fn(*[a[i] for a in arrays])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _pair_stats(a: np.ndarray, b: np.ndarray, n_resamples: int, seed: int) -> Dict[str, Any]:
    """Every agreement statistic for one pair of raters on one dimension."""
    bin_a = (a >= ACCEPTABLE_THRESHOLD).astype(int)
    bin_b = (b >= ACCEPTABLE_THRESHOLD).astype(int)

    def _k_unw(x, y):  return cohen_kappa(x.astype(int), y.astype(int))
    def _k_lin(x, y):  return cohen_kappa(x.astype(int), y.astype(int), weights="linear")
    def _k_quad(x, y): return cohen_kappa(x.astype(int), y.astype(int), weights="quadratic")
    def _k_bin(x, y):  return cohen_kappa(x.astype(int), y.astype(int), categories=(0, 1))
    def _mean_diff(x, y): return float(np.mean(x - y))

    out: Dict[str, Any] = {"n": int(len(a))}
    for name, fn, args_ in [
        ("kappa_unweighted", _k_unw, (a, b)),
        ("kappa_linear", _k_lin, (a, b)),
        ("kappa_quadratic", _k_quad, (a, b)),
        ("kappa_binary_acceptable", _k_bin, (bin_a, bin_b)),
    ]:
        point = fn(*args_)
        lo, hi = _bootstrap_ci(fn, args_, n_resamples, seed)
        out[name] = {"point": point, "ci_low": lo, "ci_high": hi}

    out["percent_agreement"] = float(percent_agreement(a.astype(int), b.astype(int)))
    out["percent_agreement_within_1"] = float(np.mean(np.abs(a - b) <= 1))
    out["pabak"] = float(pabak(a.astype(int), b.astype(int)))
    out["percent_agreement_binary"] = float(percent_agreement(bin_a, bin_b))
    out["pabak_binary"] = float(pabak(bin_a, bin_b, k=2))

    md = _mean_diff(a, b)
    lo, hi = _bootstrap_ci(_mean_diff, (a, b), n_resamples, seed)
    out["mean_first"] = float(np.mean(a))
    out["mean_second"] = float(np.mean(b))
    out["mean_difference"] = {"point": md, "ci_low": lo, "ci_high": hi}
    return out


def _simulate_annotations(units: List[Dict[str, Any]], judges: List[str], seed: int
                          ) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Fabricate two RDs by perturbing the first judge's scores.

    For smoke-testing the statistics end to end before the RDs deliver. The
    output is NOT a research result and is written to a separate file.
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for u in units:
        base = u["judge_scores"][judges[0]]
        per_rater = {}
        for rater, noise in (("RD1", 0.7), ("RD2", 1.0)):
            scores = {}
            for d in DIMENSIONS:
                # RDs are modelled as systematically stricter than the judge.
                v = base[d] - abs(rng.normal(0.4, noise))
                scores[d] = int(np.clip(round(v), 1, 5))
            per_rater[rater] = scores
        out[u["unit_id"]] = per_rater
    return out


def cmd_agreement(args: argparse.Namespace) -> None:
    if not SAMPLE_PATH.exists():
        raise SystemExit(f"{SAMPLE_PATH} not found; run `sample` first.")
    sample = json.loads(SAMPLE_PATH.read_text())
    units = sample["units"]
    unit_by_id = {u["unit_id"]: u for u in units}
    judges = args.judges or sample["config"]["judges"]

    if args.simulate:
        print("[agreement] *** SIMULATION MODE -- fabricated RD scores, not a result ***")
        annotations = _simulate_annotations(units, judges, args.seed)
        raters = ["RD1", "RD2"]
    else:
        if not ANNOTATIONS_PATH.exists():
            raise SystemExit(
                f"{ANNOTATIONS_PATH} not found. Run `import --export <label studio export>` "
                f"first, or pass --simulate to smoke-test the statistics."
            )
        payload = json.loads(ANNOTATIONS_PATH.read_text())
        annotations = payload["annotations"]
        raters = payload["raters"]

    print(f"[agreement] raters={raters}  judges={judges}  units={len(annotations)}")

    def strata_units(stratum: str) -> List[str]:
        if stratum == "pooled":
            return [uid for uid in annotations if uid in unit_by_id]
        return [uid for uid in annotations
                if uid in unit_by_id and unit_by_id[uid]["question_category"] == stratum]

    results: Dict[str, Any] = {
        "config": {
            "raters": raters,
            "judges": judges,
            "dimensions": DIMENSIONS,
            "likert": LIKERT,
            "acceptable_threshold": ACCEPTABLE_THRESHOLD,
            "bootstrap_resamples": args.resamples,
            "seed": args.seed,
            "simulated": bool(args.simulate),
            "consensus_rule": "mean of RD scores, rounded half up, clipped to 1-5",
        },
        "inter_rd": {},
        "judge_vs_rd": {},
        "descriptives": {},
    }

    strata = ["pooled"] + QUESTION_CATEGORIES

    # ---- Inter-RD ----------------------------------------------------------
    rd_pairs = [(raters[i], raters[j])
                for i in range(len(raters)) for j in range(i + 1, len(raters))]
    for stratum in strata:
        uids = strata_units(stratum)
        block: Dict[str, Any] = {}
        for dim in DIMENSIONS:
            dim_block: Dict[str, Any] = {}
            for r1, r2 in rd_pairs:
                paired = [(annotations[u][r1][dim], annotations[u][r2][dim])
                          for u in uids
                          if r1 in annotations[u] and r2 in annotations[u]
                          and dim in annotations[u][r1] and dim in annotations[u][r2]]
                if not paired:
                    continue
                a = np.array([p[0] for p in paired], dtype=float)
                b = np.array([p[1] for p in paired], dtype=float)
                dim_block[f"{r1}_vs_{r2}"] = _pair_stats(a, b, args.resamples, args.seed)

            # Krippendorff's alpha over all raters at once.
            mat = []
            for u in uids:
                row = [annotations[u][r][dim] if r in annotations[u] and dim in annotations[u][r]
                       else np.nan for r in raters]
                mat.append(row)
            if mat:
                dim_block["krippendorff_alpha_ordinal"] = krippendorff_alpha_ordinal(
                    np.array(mat, dtype=float))
            block[dim] = dim_block
        results["inter_rd"][stratum] = block

    # ---- RD consensus ------------------------------------------------------
    consensus: Dict[str, Dict[str, int]] = {}
    for uid, per_rater in annotations.items():
        c = {}
        for dim in DIMENSIONS:
            vals = [per_rater[r][dim] for r in raters
                    if r in per_rater and dim in per_rater[r]]
            if vals:
                # Round half up (np.round is banker's rounding).
                c[dim] = int(np.clip(np.floor(np.mean(vals) + 0.5), 1, 5))
        if c:
            consensus[uid] = c

    # ---- Judge vs RD -------------------------------------------------------
    for stratum in strata:
        uids = strata_units(stratum)
        block: Dict[str, Any] = {}
        for judge in judges:
            jblock: Dict[str, Any] = {}
            for dim in DIMENSIONS:
                targets = [(r, lambda u, r=r: annotations[u].get(r, {}).get(dim))
                           for r in raters]
                targets.append(("consensus", lambda u: consensus.get(u, {}).get(dim)))
                dim_block: Dict[str, Any] = {}
                for label, getter in targets:
                    paired = []
                    for u in uids:
                        rv = getter(u)
                        if rv is None or u not in unit_by_id:
                            continue
                        paired.append((unit_by_id[u]["judge_scores"][judge][dim], rv))
                    if not paired:
                        continue
                    j = np.array([p[0] for p in paired], dtype=float)
                    r = np.array([p[1] for p in paired], dtype=float)
                    # Order (judge, rd) so mean_difference > 0 means the judge
                    # is more lenient than the human.
                    dim_block[f"judge_vs_{label}"] = _pair_stats(j, r, args.resamples, args.seed)
                jblock[dim] = dim_block
            block[judge] = jblock
        results["judge_vs_rd"][stratum] = block

    # ---- Descriptives ------------------------------------------------------
    for stratum in strata:
        uids = strata_units(stratum)
        d: Dict[str, Any] = {"n_units": len(uids)}
        for dim in DIMENSIONS:
            entry = {}
            for r in raters:
                vals = [annotations[u][r][dim] for u in uids
                        if r in annotations[u] and dim in annotations[u][r]]
                if vals:
                    entry[r] = {"mean": float(np.mean(vals)),
                                "distribution": {str(v): int(sum(1 for x in vals if x == v))
                                                 for v in LIKERT}}
            for judge in judges:
                vals = [unit_by_id[u]["judge_scores"][judge][dim] for u in uids if u in unit_by_id]
                if vals:
                    entry[judge] = {"mean": float(np.mean(vals)),
                                    "distribution": {str(v): int(sum(1 for x in vals if x == v))
                                                     for v in LIKERT}}
            d[dim] = entry
        results["descriptives"][stratum] = d

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / ("agreement_simulated.json" if args.simulate else "agreement.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[agreement] wrote {out_path}")
    _print_agreement_summary(results, raters, judges)


def _fmt(entry: Optional[Dict[str, Any]]) -> str:
    if not entry:
        return "     --      "
    p, lo, hi = entry["point"], entry["ci_low"], entry["ci_high"]
    if np.isnan(p):
        return "     nan     "
    return f"{p:.3f} [{lo:.2f},{hi:.2f}]"


def _print_agreement_summary(results: Dict[str, Any], raters: List[str], judges: List[str]) -> None:
    rd_pair_key = f"{raters[0]}_vs_{raters[1]}" if len(raters) >= 2 else None

    print("\n" + "=" * 78)
    print("INTER-RD AGREEMENT  (quadratic-weighted Cohen's kappa [95% CI])")
    print("=" * 78)
    if rd_pair_key:
        print(f"{'stratum':<16}" + "".join(f"{d:<21}" for d in DIMENSIONS))
        for stratum in ["pooled"] + QUESTION_CATEGORIES:
            cells = []
            for dim in DIMENSIONS:
                e = results["inter_rd"][stratum].get(dim, {}).get(rd_pair_key, {})
                cells.append(_fmt(e.get("kappa_quadratic")))
            print(f"{stratum:<16}" + "".join(f"{c:<21}" for c in cells))
    else:
        print("  (only one rater found -- inter-RD kappa needs at least two)")

    print("\n" + "=" * 78)
    print("JUDGE vs RD-CONSENSUS  (quadratic-weighted Cohen's kappa [95% CI])")
    print("=" * 78)
    for judge in judges:
        print(f"\n  judge = {judge}")
        print(f"  {'stratum':<14}" + "".join(f"{d:<21}" for d in DIMENSIONS))
        for stratum in ["pooled"] + QUESTION_CATEGORIES:
            cells = []
            for dim in DIMENSIONS:
                e = results["judge_vs_rd"][stratum][judge].get(dim, {}).get("judge_vs_consensus", {})
                cells.append(_fmt(e.get("kappa_quadratic")))
            print(f"  {stratum:<14}" + "".join(f"{c:<21}" for c in cells))

    print("\n" + "=" * 78)
    print("JUDGE LENIENCY  (mean judge score - mean RD-consensus score, [95% CI])")
    print("  positive = judge scores the system HIGHER than the dietitians do")
    print("=" * 78)
    for judge in judges:
        print(f"\n  judge = {judge}")
        print(f"  {'stratum':<14}" + "".join(f"{d:<21}" for d in DIMENSIONS))
        for stratum in ["pooled"] + QUESTION_CATEGORIES:
            cells = []
            for dim in DIMENSIONS:
                e = results["judge_vs_rd"][stratum][judge].get(dim, {}).get("judge_vs_consensus", {})
                cells.append(_fmt(e.get("mean_difference")))
            print(f"  {stratum:<14}" + "".join(f"{c:<21}" for c in cells))

    print("\n" + "=" * 78)
    print("PREVALENCE CHECK  (percent agreement / PABAK, judge vs RD-consensus, pooled)")
    print("=" * 78)
    for judge in judges:
        for dim in DIMENSIONS:
            e = results["judge_vs_rd"]["pooled"][judge].get(dim, {}).get("judge_vs_consensus", {})
            if not e:
                continue
            print(f"  {judge:<20} {dim:<13} exact={e['percent_agreement']:.2f}  "
                  f"within-1={e['percent_agreement_within_1']:.2f}  "
                  f"PABAK={e['pabak']:.3f}  binary-PABAK={e['pabak_binary']:.3f}")
    print()


# ============================================================================
# Subcommand: report (LaTeX)
# ============================================================================

def _tex_cell(entry: Optional[Dict[str, Any]]) -> str:
    if not entry or np.isnan(entry["point"]):
        return "--"
    return (f"${entry['point']:.3f}$ \\scriptsize$[{entry['ci_low']:.2f}, "
            f"{entry['ci_high']:.2f}]$")


def cmd_report(args: argparse.Namespace) -> None:
    path = OUTPUT_DIR / ("agreement_simulated.json" if args.simulate else "agreement.json")
    if not path.exists():
        raise SystemExit(f"{path} not found; run `agreement` first.")
    results = json.loads(path.read_text())
    raters = results["config"]["raters"]
    judges = results["config"]["judges"]
    rd_pair_key = f"{raters[0]}_vs_{raters[1]}" if len(raters) >= 2 else None

    lines: List[str] = []
    lines.append("% Generated by src/pipelines/18-rd_judge_agreement.py report")
    if results["config"].get("simulated"):
        lines.append("% WARNING: built from SIMULATED annotations -- do not publish.")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Comparison & Stratum & Factuality & Completeness & Safety \\")
    lines.append(r"\midrule")

    if rd_pair_key:
        lines.append(r"\multicolumn{5}{l}{\textit{Inter-RD (quadratic-weighted $\kappa$)}} \\")
        for stratum in ["pooled"] + QUESTION_CATEGORIES:
            cells = [_tex_cell(results["inter_rd"][stratum].get(d, {})
                               .get(rd_pair_key, {}).get("kappa_quadratic"))
                     for d in DIMENSIONS]
            lines.append(f"{raters[0]}--{raters[1]} & {stratum} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    for judge in judges:
        safe = judge.replace("_", r"\_")
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{safe} vs RD consensus "
                     rf"(quadratic-weighted $\kappa$)}}}} \\")
        for stratum in ["pooled"] + QUESTION_CATEGORIES:
            cells = [_tex_cell(results["judge_vs_rd"][stratum][judge].get(d, {})
                               .get("judge_vs_consensus", {}).get("kappa_quadratic"))
                     for d in DIMENSIONS]
            lines.append(f"{safe} & {stratum} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    lines.append(r"\multicolumn{5}{l}{\textit{Judge leniency (mean judge $-$ mean RD consensus)}} \\")
    for judge in judges:
        safe = judge.replace("_", r"\_")
        cells = [_tex_cell(results["judge_vs_rd"]["pooled"][judge].get(d, {})
                           .get("judge_vs_consensus", {}).get("mean_difference"))
                 for d in DIMENSIONS]
        lines.append(f"{safe} & pooled & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    n_pooled = results["descriptives"]["pooled"]["n_units"]
    lines.append(rf"\caption{{Agreement between Registered Dietitians and the LLM-as-judge "
                 rf"rubric on {n_pooled} annotation units, stratified by question category. "
                 rf"Brackets give percentile bootstrap $95\%$ CIs over annotation units. "
                 rf"Positive leniency means the judge scores system answers higher than "
                 rf"the dietitians do.}}")
    lines.append(r"\label{tab:rd_judge_agreement}")
    lines.append(r"\end{table*}")

    LATEX_PATH.write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {LATEX_PATH}")
    print("\n".join(lines))


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RD / LLM-judge agreement pipeline for IndoNutriQA.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("sample", help="Draw the stratified RD annotation sample.")
    s1.add_argument("--n", type=int, default=DEFAULT_N_UNITS)
    s1.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s1.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES,
                    help="Only sample units whose scores are cached for all of these judges.")
    s1.add_argument("--no-context", action="store_true",
                    help="Omit the clinical metadata (diagnosis, anthropometry) from the tasks, "
                         "giving the RD exactly the same information the judge had (strict "
                         "parity). Pair with the trimmed labeling config.")

    s2 = sub.add_parser("export-tasks", help="Regenerate Label Studio tasks from an existing sample.")
    s2.add_argument("--no-context", action="store_true")

    s3 = sub.add_parser("import", help="Parse Label Studio export(s) into rd_annotations.json.")
    s3.add_argument("--export", required=True, nargs="+",
                    help="Label Studio export JSON path(s). Use LABEL=path to attribute a "
                         "whole file to one rater (one project per RD), e.g. "
                         "RD1=rd1_export.json RD2=rd2_export.json")
    s3.add_argument("--rater-map", nargs="*", default=None,
                    help="Map LS users to labels for un-labelled exports, "
                         "e.g. rd1@x.id=RD1 rd2@x.id=RD2")

    s4 = sub.add_parser("agreement", help="Inter-RD and judge-vs-RD kappa, stratified.")
    s4.add_argument("--judges", nargs="+", default=None)
    s4.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    s4.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s4.add_argument("--simulate", action="store_true",
                    help="Fabricate RD scores from the judge to smoke-test the statistics.")

    s5 = sub.add_parser("report", help="Render the agreement JSON as a LaTeX table.")
    s5.add_argument("--simulate", action="store_true")

    return p


DISPATCH = {
    "sample":       cmd_sample,
    "export-tasks": cmd_export_tasks,
    "import":       cmd_import,
    "agreement":    cmd_agreement,
    "report":       cmd_report,
}


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DISPATCH[args.cmd](args)


if __name__ == "__main__":
    main()
