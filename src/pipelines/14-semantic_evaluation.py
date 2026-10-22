"""
Semantic Evaluation Pipeline for IndoNutriQA

Adds semantic / clinical-aware metrics that complement the lexical-overlap
metrics (EM/F1/BLEU/ROUGE) reported in §5.3 of the paper. These address the
reviewer concern that lexical metrics underestimate the quality of clinically
valid but lexically divergent answers.

Three subcommands, each reusing the per-(model, shots) prediction files
written by 13-statistical_robustness.py at
data/13-statistical_robustness/predictions/:

1. bertscore        -- multilingual BERTScore (P/R/F1) using XLM-RoBERTa-large
                       or bert-base-multilingual-cased. Captures contextual
                       semantic similarity between prediction and reference.

2. embedding-cosine -- mean cosine similarity over multilingual sentence
                       embeddings (paraphrase-multilingual-MiniLM-L12-v2).
                       A second independent semantic signal.

3. llm-judge        -- GPT-4o-mini scores each (prediction, reference) pair on
                       a clinical rubric: factuality, completeness, and
                       clinical safety, on a 1-5 Likert scale. Caches per-
                       example responses on disk so re-runs are free.

All three write bootstrap 95%% CIs alongside point estimates so they can be
compared against the EM/F1/BLEU/ROUGE numbers in Tables 3 and 6 directly.
The consolidated output file is data/14-semantic_evaluation/semantic_ci.json.

Usage:
    # Cheapest first: BERTScore + embedding cosine on all 9 prediction files
    python src/pipelines/14-semantic_evaluation.py bertscore
    python src/pipelines/14-semantic_evaluation.py embedding-cosine

    # Then LLM-as-judge (~$0.40, ~5 minutes against gpt-4o-mini)
    python src/pipelines/14-semantic_evaluation.py llm-judge
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Load .env at the project root so OPENAI_API_KEY is picked up.
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass


# ============================================================================
# Configuration
# ============================================================================

PREDICTIONS_DIR = Path("data/13-statistical_robustness/predictions")
OUTPUT_DIR = Path("data/14-semantic_evaluation")
JUDGE_CACHE_DIR = OUTPUT_DIR / "judge_cache"

DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 42

# Models
DEFAULT_BERTSCORE_MODEL = "xlm-roberta-large"  # multilingual, strong
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"

# Files to evaluate (default: every prediction file on disk).
def list_prediction_files(filter_substring: Optional[str] = None) -> List[Path]:
    if not PREDICTIONS_DIR.exists():
        raise SystemExit(
            f"{PREDICTIONS_DIR} does not exist. Run 13-statistical_robustness.py "
            f"extract-llm-predictions / extract-baseline-predictions first."
        )
    files = sorted(PREDICTIONS_DIR.glob("*.json"))
    if filter_substring:
        files = [f for f in files if filter_substring in f.stem]
    if not files:
        raise SystemExit(f"No prediction files matched in {PREDICTIONS_DIR}.")
    return files


def load_prediction_file(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Generic bootstrap CI over per-example scores
# ============================================================================

def bootstrap_ci_from_scores(
    per_example: np.ndarray,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    scale: float = 1.0,
) -> Tuple[float, float, float]:
    """Return (point, ci_low, ci_high) given per-example scores already on the
    natural metric scale (e.g. 0..1 for cosine, 1..5 for Likert)."""
    rng = np.random.default_rng(seed)
    n = len(per_example)
    point = float(per_example.mean()) * scale
    sample_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample_means[i] = per_example[idx].mean()
    lo = float(np.percentile(sample_means, 2.5)) * scale
    hi = float(np.percentile(sample_means, 97.5)) * scale
    return point, lo, hi


# ============================================================================
# Output persistence (append by (file_stem, metric_key))
# ============================================================================

def output_path() -> Path:
    return OUTPUT_DIR / "semantic_ci.json"


def load_existing_summary() -> Dict[str, Any]:
    p = output_path()
    if not p.exists():
        return {"results": {}}
    try:
        existing = json.loads(p.read_text())
        existing.setdefault("results", {})
        return existing
    except Exception as e:
        print(f"  [warn] could not parse {p}: {e}; starting fresh")
        return {"results": {}}


def save_summary(summary: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path(), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {output_path()}")


def merge_metric_into(summary: Dict[str, Any], file_stem: str,
                      model_meta: Dict[str, Any], metric_block: Dict[str, Any]) -> None:
    bucket = summary["results"].setdefault(file_stem, {
        "model": model_meta.get("model"),
        "kind": model_meta.get("kind"),
        "shots": model_meta.get("shots"),
        "n": model_meta.get("n"),
        "metrics": {},
    })
    bucket["metrics"].update(metric_block)


# ============================================================================
# Subcommand: bertscore
# ============================================================================

def cmd_bertscore(args: argparse.Namespace) -> None:
    from bert_score import BERTScorer

    print(f"Loading BERTScorer with model={args.model} (this downloads on first run)...")
    scorer = BERTScorer(model_type=args.model, num_layers=None,
                        rescale_with_baseline=False, lang="id")

    summary = load_existing_summary()
    files = list_prediction_files(args.filter)

    for path in files:
        payload = load_prediction_file(path)
        preds = payload["predictions"]
        truths = payload["ground_truths"]
        print(f"\n{path.stem}  (n={len(preds)})")

        # BERTScore returns three tensors: P, R, F1 (each length n).
        P, R, F1 = scorer.score(preds, truths, batch_size=args.batch_size)
        f1_arr = F1.numpy()
        p_arr  = P.numpy()
        r_arr  = R.numpy()

        block: Dict[str, Any] = {}
        for arr, name in [(p_arr, "bertscore_p"), (r_arr, "bertscore_r"), (f1_arr, "bertscore_f1")]:
            point, lo, hi = bootstrap_ci_from_scores(arr, args.resamples, args.seed, scale=100.0)
            block[name] = {"point": point, "ci_low": lo, "ci_high": hi,
                           "model": args.model}
            if name == "bertscore_f1":
                print(f"  bertscore_F1 {point:6.2f}  [{lo:5.2f}, {hi:5.2f}]")
        merge_metric_into(summary, path.stem, payload, block)

    save_summary(summary)


# ============================================================================
# Subcommand: embedding-cosine
# ============================================================================

def cmd_embedding_cosine(args: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    print(f"Loading SentenceTransformer({args.model})...")
    embedder = SentenceTransformer(args.model)

    summary = load_existing_summary()
    files = list_prediction_files(args.filter)

    for path in files:
        payload = load_prediction_file(path)
        preds = payload["predictions"]
        truths = payload["ground_truths"]
        print(f"\n{path.stem}  (n={len(preds)})")

        pred_emb = embedder.encode(preds, batch_size=args.batch_size,
                                   convert_to_numpy=True, show_progress_bar=False)
        truth_emb = embedder.encode(truths, batch_size=args.batch_size,
                                    convert_to_numpy=True, show_progress_bar=False)
        # Diagonal of pairwise cos-sim is the per-example score.
        sims = (pred_emb * truth_emb).sum(axis=1) / (
            np.linalg.norm(pred_emb, axis=1) * np.linalg.norm(truth_emb, axis=1) + 1e-12
        )

        point, lo, hi = bootstrap_ci_from_scores(sims, args.resamples, args.seed, scale=100.0)
        block = {"embedding_cosine": {"point": point, "ci_low": lo, "ci_high": hi,
                                       "model": args.model}}
        print(f"  embedding_cosine {point:6.2f}  [{lo:5.2f}, {hi:5.2f}]  (model={args.model})")
        merge_metric_into(summary, path.stem, payload, block)

    save_summary(summary)


# ============================================================================
# Subcommand: llm-judge
# ============================================================================

JUDGE_SYSTEM_PROMPT = (
    "Anda adalah seorang ahli gizi klinis berpengalaman yang sedang menilai "
    "kualitas jawaban yang dihasilkan oleh sistem komputer untuk pertanyaan "
    "klinis berbahasa Indonesia. Anda akan menerima jawaban referensi yang "
    "dibuat oleh ahli gizi terdaftar (Registered Dietitian) dan jawaban "
    "kandidat dari sistem. Berikan penilaian numerik 1-5 pada tiga dimensi "
    "berikut:\n"
    "1. FAKTUALITAS: konsistensi faktual jawaban kandidat dengan referensi "
    "(1=banyak kontradiksi, 5=sepenuhnya konsisten).\n"
    "2. KELENGKAPAN: cakupan poin klinis yang sama dengan referensi "
    "(1=banyak hal hilang, 5=lengkap).\n"
    "3. KEAMANAN KLINIS: ketiadaan klaim klinis yang menyesatkan atau "
    "berbahaya (1=berpotensi membahayakan, 5=aman).\n"
    "Jawablah HANYA dengan objek JSON valid berbentuk: "
    "{\"factuality\": <1-5>, \"completeness\": <1-5>, \"safety\": <1-5>}. "
    "Tidak ada teks lain."
)

JUDGE_USER_TEMPLATE = (
    "JAWABAN REFERENSI (RD):\n{reference}\n\n"
    "JAWABAN KANDIDAT (sistem):\n{prediction}\n\n"
    "Berikan skor JSON sekarang."
)


def _judge_cache_key(judge_model: str, reference: str, prediction: str) -> str:
    h = hashlib.sha256()
    h.update(f"judge|{judge_model}|".encode("utf-8"))
    h.update(reference.encode("utf-8"))
    h.update(b"|||")
    h.update(prediction.encode("utf-8"))
    return h.hexdigest()


def _judge_cache_get(key: str) -> Optional[Dict[str, int]]:
    p = JUDGE_CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("scores")
    except Exception:
        return None


def _judge_cache_put(key: str, judge_model: str, scores: Dict[str, int]) -> None:
    JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = JUDGE_CACHE_DIR / f"{key}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"judge_model": judge_model, "scores": scores}, f, ensure_ascii=False)


def _parse_judge_response(text: str) -> Optional[Dict[str, int]]:
    """Parse the judge's JSON; return None on malformed responses."""
    try:
        # The model sometimes wraps JSON in code fences; strip them.
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`").lstrip("json").strip()
        obj = json.loads(t)
        out: Dict[str, int] = {}
        for k in ("factuality", "completeness", "safety"):
            v = obj.get(k)
            if v is None:
                return None
            v = int(v)
            if not (1 <= v <= 5):
                return None
            out[k] = v
        return out
    except Exception:
        return None


def _is_anthropic_model(name: str) -> bool:
    """Heuristic: any model id starting with "claude" is dispatched to the
    Anthropic API."""
    return name.lower().startswith("claude")


def _is_gemini_model(name: str) -> bool:
    """Heuristic: any model id starting with "gemini" is dispatched to the
    Google GenAI API."""
    return name.lower().startswith("gemini")


def _call_openai_judge(client, model: str, truth: str, pred: str) -> Optional[Dict[str, int]]:
    """Run one rubric judgement via OpenAI. Returns parsed scores or None."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                reference=truth, prediction=pred)},
        ],
        temperature=0.0,
        max_tokens=128,
        response_format={"type": "json_object"},
    )
    return _parse_judge_response(resp.choices[0].message.content or "")


def _call_anthropic_judge(client, model: str, truth: str, pred: str) -> Optional[Dict[str, int]]:
    """Run one rubric judgement via Anthropic. Anthropic does not have OpenAI's
    `response_format=json_object` mode, so we instead append an explicit
    "Respond with only a JSON object" instruction; the same _parse_judge_response
    parser handles both responses (it already strips code fences)."""
    resp = client.messages.create(
        model=model,
        system=JUDGE_SYSTEM_PROMPT + (
            "\n\nRespond with ONLY a JSON object of the form "
            '{"factuality": int, "completeness": int, "safety": int} '
            "and no additional commentary."
        ),
        messages=[
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                reference=truth, prediction=pred)},
        ],
        temperature=0.0,
        max_tokens=128,
    )
    # Anthropic SDK returns `content` as a list of TextBlock objects.
    text = "".join(getattr(block, "text", "") for block in resp.content)
    return _parse_judge_response(text)


def _call_gemini_judge(client, model: str, truth: str, pred: str) -> Optional[Dict[str, int]]:
    """Run one rubric judgement via Google Gemini (google-genai SDK).

    The SDK supports a `response_mime_type='application/json'` config; we
    combine the system prompt + user prompt into the `contents` field, since
    Gemini's API takes a single conversation thread per request.
    """
    from google.genai import types

    user_prompt = JUDGE_USER_TEMPLATE.format(reference=truth, prediction=pred)
    resp = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                JUDGE_SYSTEM_PROMPT
                + "\n\nRespond with ONLY a JSON object of the form "
                + '{"factuality": int, "completeness": int, "safety": int} '
                + "and no additional commentary."
            ),
            temperature=0.0,
            max_output_tokens=128,
            response_mime_type="application/json",
        ),
    )
    return _parse_judge_response(resp.text or "")


def cmd_llm_judge(args: argparse.Namespace) -> None:
    if _is_anthropic_model(args.judge_model):
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY is not set (check .env).")
        client = Anthropic(api_key=api_key)
        call_judge = _call_anthropic_judge
        provider = "anthropic"
    elif _is_gemini_model(args.judge_model):
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit(
                "Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set (check .env)."
            )
        client = genai.Client(api_key=api_key)
        call_judge = _call_gemini_judge
        provider = "google"
    else:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is not set (check .env).")
        client = OpenAI(api_key=api_key)
        call_judge = _call_openai_judge
        provider = "openai"
    print(f"[llm-judge] provider={provider} model={args.judge_model}")

    summary = load_existing_summary()
    files = list_prediction_files(args.filter)

    for path in files:
        payload = load_prediction_file(path)
        preds = payload["predictions"]
        truths = payload["ground_truths"]
        print(f"\n{path.stem}  (n={len(preds)})")

        fact_scores: List[int] = []
        comp_scores: List[int] = []
        safe_scores: List[int] = []
        n_failed = 0

        for pred, truth in tqdm(list(zip(preds, truths)), desc=f"  judging {path.stem}", leave=False):
            key = _judge_cache_key(args.judge_model, truth, pred)
            cached = _judge_cache_get(key)
            if cached is None:
                try:
                    cached = call_judge(client, args.judge_model, truth, pred)
                    if cached is None:
                        n_failed += 1
                        continue
                    _judge_cache_put(key, args.judge_model, cached)
                except Exception as e:
                    print(f"\n  [warn] judge call failed: {e}; skipping example")
                    n_failed += 1
                    continue
            fact_scores.append(cached["factuality"])
            comp_scores.append(cached["completeness"])
            safe_scores.append(cached["safety"])

        if not fact_scores:
            print(f"  [skip] no successful judgments for {path.stem}")
            continue

        fact_arr = np.array(fact_scores, dtype=float)
        comp_arr = np.array(comp_scores, dtype=float)
        safe_arr = np.array(safe_scores, dtype=float)
        overall = (fact_arr + comp_arr + safe_arr) / 3.0

        # Namespace the metric keys by judge family so that running with a
        # second judge (e.g. Claude) does not overwrite the first (GPT-4o-mini).
        # OpenAI uses the unsuffixed names for backward compatibility with the
        # already-saved semantic_ci.json; non-OpenAI providers get a suffix.
        suffix = "" if provider == "openai" else f"_{provider}"

        block: Dict[str, Any] = {}
        for arr, name in [(fact_arr, f"judge_factuality{suffix}"),
                          (comp_arr, f"judge_completeness{suffix}"),
                          (safe_arr, f"judge_safety{suffix}"),
                          (overall, f"judge_overall{suffix}")]:
            point, lo, hi = bootstrap_ci_from_scores(arr, args.resamples, args.seed, scale=1.0)
            block[name] = {"point": point, "ci_low": lo, "ci_high": hi,
                           "scale": "1-5 Likert", "judge_model": args.judge_model,
                           "n_judged": int(len(arr)), "n_failed": n_failed}

        for key in block:
            v = block[key]
            print(f"  {key:30s}  {v['point']:.2f}  [{v['ci_low']:.2f}, {v['ci_high']:.2f}]")
        if n_failed:
            print(f"  [note] {n_failed} judgments failed and were excluded")

        merge_metric_into(summary, path.stem, payload, block)

    save_summary(summary)


# ============================================================================
# Subcommand: judge-bias
#   Tests the reviewer concern that GPT-4o-mini is simultaneously the judge
#   and the provider of two evaluated systems (GPT-4o-mini, GPT-4o).
# ============================================================================

# Systems whose provider is the same as the GPT judge's provider.
OPENAI_FAMILY_MARKER = "gpt-4o"
SAME_MODEL_MARKER = "gpt-4o-mini"


def _paired_mean_scores(judge_a: str, judge_b: str, payload: Dict[str, Any]
                        ) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
    """Mean rubric scores for one prediction file under two judges.

    Restricted to the examples where BOTH judges have a cached score, so the
    per-system gap is a paired quantity. Averaging each judge over its own
    cached subset would let differing cache coverage leak into the gap.
    """
    dims = ("factuality", "completeness", "safety")
    acc: Dict[str, Dict[str, List[int]]] = {
        judge_a: {d: [] for d in dims}, judge_b: {d: [] for d in dims}}
    for pred, truth in zip(payload["predictions"], payload["ground_truths"]):
        ca = _judge_cache_get(_judge_cache_key(judge_a, truth, pred))
        cb = _judge_cache_get(_judge_cache_key(judge_b, truth, pred))
        if ca is None or cb is None:
            continue
        for d in dims:
            acc[judge_a][d].append(ca[d])
            acc[judge_b][d].append(cb[d])
    if not acc[judge_a]["factuality"]:
        return None
    out = []
    for j in (judge_a, judge_b):
        o = {d: float(np.mean(acc[j][d])) for d in dims}
        o["overall"] = float(np.mean([o[d] for d in dims]))
        o["n"] = len(acc[j]["factuality"])
        out.append(o)
    return out[0], out[1]


def _permutation_p(group_a: List[float], group_b: List[float],
                   n_perm: int = 20000, seed: int = DEFAULT_BOOTSTRAP_SEED) -> float:
    """Two-sided permutation test on the difference of means."""
    rng = np.random.default_rng(seed)
    allv = np.array(group_a + group_b, dtype=float)
    k = len(group_a)
    obs = float(np.mean(group_a) - np.mean(group_b))
    count = 0
    for _ in range(n_perm):
        sh = rng.permutation(allv)
        if abs(float(np.mean(sh[:k]) - np.mean(sh[k:]))) >= abs(obs) - 1e-12:
            count += 1
    return count / n_perm


def cmd_judge_bias(args: argparse.Namespace) -> None:
    """Difference-in-differences test for judge self-preference.

    A raw comparison of the GPT judge's scores on OpenAI systems against its
    scores on other systems would confound self-preference with the fact that
    OpenAI systems are simply better. We therefore difference each system's
    GPT-judge score against its Claude-judge score first -- removing any
    effect of true system quality, which both judges see -- and then ask
    whether that per-system judge gap is larger for OpenAI-family systems
    than for the rest. That second difference is the self-preference estimate.
    """
    gpt_judge, claude_judge = args.judge_a, args.judge_b
    files = list_prediction_files(args.filter)

    rows: List[Dict[str, Any]] = []
    for path in files:
        payload = load_prediction_file(path)
        paired = _paired_mean_scores(gpt_judge, claude_judge, payload)
        if paired is None:
            print(f"  [skip] {path.stem}: no examples scored by both judges")
            continue
        a, b = paired
        if a["n"] < args.min_n:
            print(f"  [skip] {path.stem}: only {a['n']} paired judgements")
            continue
        rows.append({
            "system": path.stem,
            "openai_family": OPENAI_FAMILY_MARKER in path.stem,
            "same_model_as_judge": SAME_MODEL_MARKER in path.stem,
            "judge_a": a, "judge_b": b,
            "gap": {d: a[d] - b[d] for d in ("factuality", "completeness", "safety", "overall")},
        })
    if not rows:
        raise SystemExit("No systems had cached scores for both judges.")

    rows.sort(key=lambda r: -r["gap"]["overall"])
    print(f"\n[judge-bias] per-system judge gap ({gpt_judge} minus {claude_judge})")
    print(f"  {'system':<40}{'judgeA':>8}{'judgeB':>8}{'gap':>8}")
    for r in rows:
        mark = "*" if r["openai_family"] else " "
        print(f"  {mark} {r['system']:<38}{r['judge_a']['overall']:>8.3f}"
              f"{r['judge_b']['overall']:>8.3f}{r['gap']['overall']:>+8.3f}")
    print(f"  * = same provider as the judge ({gpt_judge})")

    results: Dict[str, Any] = {
        "judge_a": gpt_judge, "judge_b": claude_judge,
        "design": ("difference-in-differences: per-system (judge_a - judge_b) rubric gap, "
                   "then same-provider systems minus other systems"),
        "n_systems": len(rows),
        "systems": rows,
        "tests": {},
    }

    for label, key in (("same_provider", "openai_family"), ("same_model", "same_model_as_judge")):
        for dim in ("factuality", "completeness", "safety", "overall"):
            grp = [r["gap"][dim] for r in rows if r[key]]
            oth = [r["gap"][dim] for r in rows if not r[key]]
            if not grp or not oth:
                continue
            did = float(np.mean(grp) - np.mean(oth))
            results["tests"].setdefault(label, {})[dim] = {
                "n_same": len(grp), "n_other": len(oth),
                "mean_gap_same": float(np.mean(grp)),
                "mean_gap_other": float(np.mean(oth)),
                "difference_in_differences": did,
                "permutation_p": _permutation_p(grp, oth, args.permutations, args.seed),
            }

    # Rank agreement: does the judge choice reorder the leaderboard?
    a_rank = [r["judge_a"]["overall"] for r in rows]
    b_rank = [r["judge_b"]["overall"] for r in rows]
    try:
        from scipy.stats import spearmanr
        rho, prho = spearmanr(a_rank, b_rank)
        results["rank_agreement"] = {"spearman_rho": float(rho), "p": float(prho)}
    except ImportError:
        ar = np.argsort(np.argsort(a_rank)); br = np.argsort(np.argsort(b_rank))
        results["rank_agreement"] = {"spearman_rho": float(np.corrcoef(ar, br)[0, 1]), "p": None}

    print(f"\n[judge-bias] difference-in-differences (positive = judge favours its own provider)")
    for label in results["tests"]:
        print(f"  {label}:")
        for dim, t in results["tests"][label].items():
            print(f"    {dim:<13} same={t['mean_gap_same']:+.3f} (n={t['n_same']})  "
                  f"other={t['mean_gap_other']:+.3f} (n={t['n_other']})  "
                  f"DiD={t['difference_in_differences']:+.3f}  p={t['permutation_p']:.3f}")
    print(f"\n[judge-bias] leaderboard rank agreement between judges: "
          f"Spearman rho = {results['rank_agreement']['spearman_rho']:.3f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "judge_bias.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[judge-bias] wrote {out}")


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Semantic / clinical-aware evaluation for IndoNutriQA.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("bertscore", help="Multilingual BERTScore P/R/F1.")
    s1.add_argument("--model", default=DEFAULT_BERTSCORE_MODEL,
                    help="HF id passed to bert-score.")
    s1.add_argument("--batch-size", type=int, default=16)
    s1.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    s1.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    s1.add_argument("--filter", default=None,
                    help="Only score prediction files whose filename contains this substring.")

    s2 = sub.add_parser("embedding-cosine",
                        help="Mean cosine similarity over multilingual sentence embeddings.")
    s2.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL,
                    help="sentence-transformers model id.")
    s2.add_argument("--batch-size", type=int, default=32)
    s2.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    s2.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    s2.add_argument("--filter", default=None)

    s3 = sub.add_parser("llm-judge",
                        help="GPT judge: factuality, completeness, safety on 1-5 Likert.")
    s3.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    s3.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    s3.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    s3.add_argument("--filter", default=None)

    s4 = sub.add_parser("judge-bias",
                        help="Difference-in-differences test for LLM-judge self-preference.")
    s4.add_argument("--judge-a", default="gpt-4o-mini",
                    help="Judge under suspicion (shares a provider with evaluated systems).")
    s4.add_argument("--judge-b", default="claude-haiku-4-5",
                    help="Control judge with no evaluated system from its own provider.")
    s4.add_argument("--permutations", type=int, default=20000)
    s4.add_argument("--min-n", type=int, default=50,
                    help="Minimum cached judgements per system.")
    s4.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    s4.add_argument("--filter", default=None)

    return p


DISPATCH = {
    "bertscore": cmd_bertscore,
    "embedding-cosine": cmd_embedding_cosine,
    "llm-judge": cmd_llm_judge,
    "judge-bias": cmd_judge_bias,
}


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DISPATCH[args.cmd](args)


if __name__ == "__main__":
    main()
