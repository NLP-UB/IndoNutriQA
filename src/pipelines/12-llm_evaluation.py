"""
LLM Evaluation Pipeline for IndoNutriQA

This module benchmarks Large Language Models (LLMs) on the IndoNutriQA
clinical QA dataset to address the "no LLMs evaluated" reviewer concern.

Supported providers:
- OpenAI / OpenAI-compatible APIs (GPT-4o-mini, plus Together / Groq /
  Fireworks endpoints for open-source models via custom base_url)
- Anthropic (DISABLED — class and registry entries kept commented out so
  re-enabling is a one-line change; see AnthropicClient below)
- HuggingFace transformers (local inference for Llama, Qwen, Aya,
  Sahabat-AI, Merak, Cendol, Meditron, BioMistral, MedGemma, etc.)

Settings:
- Zero-shot
- Few-shot (k examples sampled deterministically from the training split)

Reuses the same train/val/test split (70/15/15) and evaluation metrics
(EM, F1, BLEU, ROUGE-1/2/L) as 9-context_based_QA.py so results plug
straight into the existing comparison table.

Per-example responses are cached on disk under
data/12-llm_evaluation/cache/, keyed on (model_id, prompt_setting,
question, context) so re-runs do not re-pay API costs.

Usage:
    # Zero-shot, single model, full test set
    python src/pipelines/12-llm_evaluation.py \\
        --models gpt-4o-mini

    # Zero-shot + 5-shot, several models, smoke test on 10 samples
    python src/pipelines/12-llm_evaluation.py \\
        --models gpt-4o-mini llama-3.1-8b-instruct \\
        --shots 0 5 \\
        --limit 10

    # OpenAI-compatible endpoint (e.g., Together AI for Llama-70B)
    OPENAI_API_KEY=$TOGETHER_API_KEY \\
    python src/pipelines/12-llm_evaluation.py \\
        --models llama-3.1-70b-instruct-together \\
        --openai-base-url https://api.together.xyz/v1
"""

import argparse
import datetime
import hashlib
import json
import os
import random
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from tqdm import tqdm

# Load API keys from .env at the project root before anything reads os.environ.
# Walks upward from this file so the script works no matter the cwd.
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass


# ============================================================================
# Configuration
# ============================================================================

DATA_PATH = "data/dataset/dataset.json"
OUTPUT_DIR = Path("data/12-llm_evaluation")
CACHE_DIR = OUTPUT_DIR / "cache"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_FEW_SHOT_SEED = 42

# Model registry. The "model_id" is what the provider expects on the wire.
# Edit/extend freely - any HuggingFace repo id works for the "huggingface" provider.
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    # ---- Closed-source: OpenAI -------------------------------------------
    "gpt-4o-mini": {"provider": "openai", "model_id": "gpt-4o-mini"},
    "gpt-4o": {"provider": "openai", "model_id": "gpt-4o"},

    # ---- Closed-source: Anthropic (DISABLED) -----------------------------
    # "claude-opus-4.5": {"provider": "anthropic", "model_id": "claude-opus-4-5"},
    # "claude-sonnet-4.5": {"provider": "anthropic", "model_id": "claude-sonnet-4-5"},
    # "claude-haiku-4.5": {"provider": "anthropic", "model_id": "claude-haiku-4-5"},

    # ---- Open-source multilingual (HuggingFace) --------------------------
    "llama-3.1-8b-instruct": {
        "provider": "huggingface",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "llama-3.1-70b-instruct": {
        "provider": "huggingface",
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
    },
    "qwen2.5-7b-instruct": {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
    },
    "qwen2.5-14b-instruct": {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
    },
    "aya-23-8b": {
        "provider": "huggingface",
        "model_id": "CohereForAI/aya-23-8B",
    },
    "aya-expanse-8b": {
        "provider": "huggingface",
        "model_id": "CohereForAI/aya-expanse-8b",
    },

    # ---- Indonesian-specific (HuggingFace) -------------------------------
    "sahabat-ai-llama3-8b": {
        "provider": "huggingface",
        "model_id": "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct",
    },
    "sahabat-ai-gemma2-9b": {
        "provider": "huggingface",
        "model_id": "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct",
    },
    "merak-7b-v4": {
        "provider": "huggingface",
        "model_id": "Ichsan2895/Merak-7B-v4",
    },
    "cendol-llama2-7b": {
        "provider": "huggingface",
        "model_id": "indonlp/cendol-llama2-7b-inst",
    },

    # ---- Medical LLMs (HuggingFace) --------------------------------------
    "meditron-7b": {
        "provider": "huggingface",
        "model_id": "epfl-llm/meditron-7b",
    },
    "biomistral-7b": {
        "provider": "huggingface",
        "model_id": "BioMistral/BioMistral-7B",
    },
    "medgemma-4b-it": {
        "provider": "huggingface",
        "model_id": "google/medgemma-4b-it",
    },

    # ---- OpenAI-compatible aliases (set --openai-base-url) ---------------
    # Together AI / Groq / Fireworks expose Llama, Qwen etc. via OpenAI-compat
    # endpoints. The model_id below matches Together's catalog; adjust per host.
    "llama-3.1-70b-instruct-together": {
        "provider": "openai",
        "model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    },
    "llama-3.1-8b-instruct-together": {
        "provider": "openai",
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    },
}


# ============================================================================
# Indonesian QA prompt
# ============================================================================

SYSTEM_PROMPT_ID = (
    "Anda adalah asisten klinis ahli gizi yang menjawab pertanyaan medis "
    "dan gizi berbahasa Indonesia. Jawablah secara ringkas, akurat, dan "
    "hanya berdasarkan konteks dan profil pasien yang diberikan. Jika "
    "informasi tidak tersedia dalam konteks, jawab \"Informasi tidak tersedia.\""
)

USER_PROMPT_TEMPLATE_ID = (
    "Konteks:\n{context}\n\n"
    "Profil pasien:\n"
    "Penilaian antropometri:\n{anthropometric_assessment}\n\n"
    "Diagnosis medis:\n{medical_diagnosis}\n\n"
    "Pertanyaan: {question}\n\n"
    "Jawaban:"
)

_MISSING_FIELD_PLACEHOLDER = "Tidak tersedia."


def _format_field(value: Optional[str]) -> str:
    """Normalize an optional clinical metadata field for prompt rendering."""
    if value is None:
        return _MISSING_FIELD_PLACEHOLDER
    text = str(value).strip()
    return text if text else _MISSING_FIELD_PLACEHOLDER


def render_user_prompt(example: Dict[str, Any]) -> str:
    """Render the user-turn prompt for an example, embedding patient metadata."""
    return USER_PROMPT_TEMPLATE_ID.format(
        context=example.get("context", ""),
        anthropometric_assessment=_format_field(example.get("anthropometric_assessment")),
        medical_diagnosis=_format_field(example.get("medical_diagnosis")),
        question=example.get("question", ""),
    )


def build_messages(
    example: Dict[str, Any],
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """Build a chat-style message list usable by OpenAI/Anthropic/HF chat templates."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT_ID}]

    for ex in few_shot_examples or []:
        messages.append({"role": "user", "content": render_user_prompt(ex)})
        messages.append({"role": "assistant", "content": ex["answer_text"]})

    messages.append({"role": "user", "content": render_user_prompt(example)})
    return messages


def sample_few_shot(
    train_data: List[Dict[str, Any]], k: int, seed: int = DEFAULT_FEW_SHOT_SEED
) -> List[Dict[str, Any]]:
    if k <= 0:
        return []
    rng = random.Random(seed)
    pool = [d for d in train_data if d["answer_text"] and d["context"]]
    return rng.sample(pool, min(k, len(pool)))


# ============================================================================
# Evaluation metrics (mirrors 9-context_based_QA.py)
# ============================================================================

def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_bleu(prediction: str, ground_truth: str) -> float:
    reference = [ground_truth.split()]
    hypothesis = prediction.split()
    smoothing = SmoothingFunction().method1
    try:
        return sentence_bleu(reference, hypothesis, smoothing_function=smoothing)
    except Exception:
        return 0.0


_ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_rouge_scores(prediction: str, ground_truth: str) -> Dict[str, float]:
    scores = _ROUGE_SCORER.score(ground_truth, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


# ============================================================================
# Data loading (mirrors 9-context_based_QA.py for an identical split)
# ============================================================================

def load_qa_data(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered: List[Dict[str, Any]] = []
    for record in data:
        context = record.get("context")
        question = record.get("question")
        answer = record.get("answer")

        if isinstance(answer, dict):
            answer_text = answer.get("text", [""])[0] if answer.get("text") else ""
            answer_start = answer.get("answer_start", [-1])[0] if answer.get("answer_start") else -1
        else:
            answer_text = str(answer) if answer else ""
            answer_start = -1

        if context and question and answer_text:
            filtered.append(
                {
                    "id": record.get("id"),
                    "context": context,
                    "question": question,
                    "answer_text": answer_text,
                    "answer_start": answer_start,
                    "anthropometric_assessment": record.get("anthropometric_assessment"),
                    "medical_diagnosis": record.get("medical_diagnosis"),
                }
            )
    return filtered


def split_data(all_data: List[Dict[str, Any]]):
    train_idx = int(len(all_data) * 0.7)
    val_idx = int(len(all_data) * 0.85)
    return all_data[:train_idx], all_data[train_idx:val_idx], all_data[val_idx:]


# ============================================================================
# LLM provider clients
# ============================================================================

class LLMClient:
    """Unified interface for an LLM. Subclasses implement _call()."""

    name: str
    model_id: str

    def __init__(self, name: str, model_id: str, **kwargs):
        self.name = name
        self.model_id = model_id
        self.temperature = kwargs.get("temperature", DEFAULT_TEMPERATURE)
        self.max_new_tokens = kwargs.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)

    def _call(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError

    def predict(
        self,
        example: Dict[str, Any],
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 4,
    ) -> str:
        messages = build_messages(example, few_shot_examples)
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                out = self._call(messages)
                return (out or "").strip()
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                print(f"  [warn] {self.name} call failed (attempt {attempt + 1}/{max_retries}): {e}; sleeping {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")


class OpenAIClient(LLMClient):
    def __init__(self, name: str, model_id: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(name, model_id, **kwargs)
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    def _call(self, messages: List[Dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        return resp.choices[0].message.content or ""


# ---- Anthropic provider DISABLED ---------------------------------------
# Re-enable by uncommenting this class, the "anthropic" branch in
# build_client(), the registry entries above, and ANTHROPIC_API_KEY in .env.
#
# class AnthropicClient(LLMClient):
#     def __init__(self, name: str, model_id: str, **kwargs):
#         super().__init__(name, model_id, **kwargs)
#         import anthropic
#
#         api_key = os.getenv("ANTHROPIC_API_KEY")
#         if not api_key:
#             raise RuntimeError("ANTHROPIC_API_KEY is not set")
#         self._client = anthropic.Anthropic(api_key=api_key)
#
#     def _call(self, messages: List[Dict[str, str]]) -> str:
#         system = ""
#         chat_messages: List[Dict[str, str]] = []
#         for m in messages:
#             if m["role"] == "system":
#                 system = m["content"]
#             else:
#                 chat_messages.append({"role": m["role"], "content": m["content"]})
#         resp = self._client.messages.create(
#             model=self.model_id,
#             system=system,
#             messages=chat_messages,
#             temperature=self.temperature,
#             max_tokens=self.max_new_tokens,
#         )
#         parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
#         return "".join(parts)


class HuggingFaceClient(LLMClient):
    def __init__(
        self,
        name: str,
        model_id: str,
        dtype: str = "auto",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__(name, model_id, **kwargs)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, "auto")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self._model.eval()
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def _call(self, messages: List[Dict[str, str]]) -> str:
        import torch

        # Some base models lack a chat template; fall back to flat concat.
        try:
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self._model.device)
        except Exception:
            flat = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            flat += "\n\nASSISTANT:"
            input_ids = self._tokenizer(flat, return_tensors="pt").input_ids.to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self._tokenizer.pad_token_id,
            )
        gen_tokens = output[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(gen_tokens, skip_special_tokens=True)


def build_client(
    name: str,
    spec: Dict[str, str],
    openai_base_url: Optional[str] = None,
    hf_dtype: str = "auto",
    hf_device_map: str = "auto",
) -> LLMClient:
    provider = spec["provider"]
    model_id = spec["model_id"]
    if provider == "openai":
        return OpenAIClient(name=name, model_id=model_id, base_url=openai_base_url)
    # if provider == "anthropic":
    #     return AnthropicClient(name=name, model_id=model_id)
    if provider == "huggingface":
        return HuggingFaceClient(
            name=name, model_id=model_id, dtype=hf_dtype, device_map=hf_device_map
        )
    raise ValueError(f"Unknown provider: {provider}")


# ============================================================================
# Disk cache (avoid re-paying API costs on re-runs)
# ============================================================================

def cache_key(
    model_id: str,
    shots: int,
    example: Dict[str, Any],
    fs_seed: int,
    prompt_version: str = "v2-with-metadata",
) -> str:
    """Cache key keyed on every prompt input. `prompt_version` invalidates old caches
    whenever the user-prompt template changes (e.g. v1 -> v2 added patient metadata)."""
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(f"|shots={shots}|seed={fs_seed}|prompt={prompt_version}|".encode("utf-8"))
    for field in ("question", "context", "anthropometric_assessment", "medical_diagnosis"):
        h.update(b"|||")
        h.update((str(example.get(field) or "")).encode("utf-8"))
    return h.hexdigest()


def cache_get(key: str) -> Optional[str]:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("response")
        except Exception:
            return None
    return None


def cache_put(key: str, model_id: str, prediction: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model_id": model_id, "response": prediction}, f, ensure_ascii=False)


# ============================================================================
# Evaluation loop
# ============================================================================

def evaluate_llm(
    client: LLMClient,
    test_data: List[Dict[str, Any]],
    few_shot_examples: List[Dict[str, Any]],
    shots: int,
    fs_seed: int,
    label: str,
) -> Dict[str, Any]:
    em_scores, f1_scores, bleu_scores = [], [], []
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    predictions, ground_truths = [], []

    for ex in tqdm(test_data, desc=f"  {label}", leave=False):
        truth = ex["answer_text"]

        key = cache_key(client.model_id, shots, ex, fs_seed)
        cached = cache_get(key)
        if cached is not None:
            prediction = cached
        else:
            prediction = client.predict(ex, few_shot_examples)
            cache_put(key, client.model_id, prediction)

        predictions.append(prediction)
        ground_truths.append(truth)

        em_scores.append(compute_exact_match(prediction, truth))
        f1_scores.append(compute_f1(prediction, truth))
        bleu_scores.append(compute_bleu(prediction, truth))
        rouge_d = compute_rouge_scores(prediction, truth)
        rouge1_scores.append(rouge_d["rouge1"])
        rouge2_scores.append(rouge_d["rouge2"])
        rougeL_scores.append(rouge_d["rougeL"])

    return {
        "label": label,
        "model_id": client.model_id,
        "shots": shots,
        "n_examples": len(test_data),
        "exact_match": float(np.mean(em_scores) * 100),
        "f1_score": float(np.mean(f1_scores) * 100),
        "bleu": float(np.mean(bleu_scores) * 100),
        "rouge1": float(np.mean(rouge1_scores) * 100),
        "rouge2": float(np.mean(rouge2_scores) * 100),
        "rougeL": float(np.mean(rougeL_scores) * 100),
        "predictions": predictions,
        "ground_truths": ground_truths,
    }


def print_results_table(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 110)
    print("LLM EVALUATION RESULTS")
    print("=" * 110)
    print(
        "{:<35} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
            "Model", "Shots", "EM (%)", "F1 (%)", "BLEU (%)", "R-1 (%)", "R-2 (%)", "R-L (%)"
        )
    )
    print("-" * 110)
    for r in results:
        print(
            "{:<35} {:>6d} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}".format(
                r["label"][:35],
                r["shots"],
                r["exact_match"],
                r["f1_score"],
                r["bleu"],
                r["rouge1"],
                r["rouge2"],
                r["rougeL"],
            )
        )
    print("-" * 110)


# ============================================================================
# Append-mode persistence
# ============================================================================

def load_existing_summary(path: Path) -> Dict[str, Any]:
    """Load the summary dict from a previous run, or return an empty skeleton."""
    if not path.exists():
        return {"config": {}, "results": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if not isinstance(existing, dict):
            raise ValueError("expected a JSON object at the top level")
        existing.setdefault("results", [])
        existing.setdefault("config", {})
        return existing
    except Exception as e:
        print(f"  [warn] could not parse existing {path}: {e}; starting fresh")
        return {"config": {}, "results": []}


def merge_results(
    existing_results: List[Dict[str, Any]],
    new_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge new results into existing, replacing rows that match on (model_id, shots)."""
    new_keys = {(r.get("model_id"), r.get("shots")) for r in new_results}
    kept = [
        r for r in existing_results
        if (r.get("model_id"), r.get("shots")) not in new_keys
    ]
    merged = kept + new_results
    merged.sort(key=lambda r: (str(r.get("alias", "")), int(r.get("shots", 0))))
    return merged


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM evaluation pipeline for IndoNutriQA.")
    parser.add_argument(
        "--data-path", default=DATA_PATH, help="Path to qa_pairs_with_chunks.json"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=f"Model aliases from MODEL_REGISTRY. Available: {', '.join(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[0],
        help="Few-shot k values to evaluate (e.g. --shots 0 5).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally cap test set size for smoke tests.",
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    parser.add_argument(
        "--few-shot-seed", type=int, default=DEFAULT_FEW_SHOT_SEED
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="Custom base URL for OpenAI-compatible endpoints "
        "(Together / Groq / Fireworks / vLLM serve).",
    )
    parser.add_argument(
        "--hf-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for HuggingFace models.",
    )
    parser.add_argument(
        "--hf-device-map", default="auto", help="HuggingFace device_map argument."
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "llm_results.json"),
        help="Where to write the consolidated results JSON.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Persist per-example predictions in the output JSON (large).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output JSON instead of merging with previous runs. "
             "Default behavior is append: re-running a (model, shots) pair updates "
             "its row in place; everything else is preserved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("INDONUTRIQA - LLM EVALUATION")
    print("=" * 80)

    print("\nLoading data from:", args.data_path)
    all_data = load_qa_data(args.data_path)
    train_data, val_data, test_data = split_data(all_data)
    print(f"  Total: {len(all_data)} | Train: {len(train_data)} | "
          f"Val: {len(val_data)} | Test: {len(test_data)}")

    if args.limit is not None:
        test_data = test_data[: args.limit]
        print(f"  --limit {args.limit} applied -> evaluating on {len(test_data)} samples")

    # Pre-sample one few-shot batch per k value so all models see identical exemplars.
    fewshot_pools: Dict[int, List[Dict[str, Any]]] = {
        k: sample_few_shot(train_data, k, seed=args.few_shot_seed) for k in args.shots
    }

    # Validate all model aliases up-front.
    unknown = [m for m in args.models if m not in MODEL_REGISTRY]
    if unknown:
        raise SystemExit(
            f"Unknown model aliases: {unknown}. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )

    all_results: List[Dict[str, Any]] = []
    for alias in args.models:
        spec = MODEL_REGISTRY[alias]
        print(f"\n--- {alias} ({spec['provider']}: {spec['model_id']}) ---")
        try:
            client = build_client(
                name=alias,
                spec=spec,
                openai_base_url=args.openai_base_url,
                hf_dtype=args.hf_dtype,
                hf_device_map=args.hf_device_map,
            )
            client.temperature = args.temperature
            client.max_new_tokens = args.max_new_tokens
        except Exception as e:
            print(f"  [skip] failed to initialize {alias}: {e}")
            continue

        for k in args.shots:
            label = f"{alias} ({k}-shot)"
            try:
                result = evaluate_llm(
                    client=client,
                    test_data=test_data,
                    few_shot_examples=fewshot_pools[k],
                    shots=k,
                    fs_seed=args.few_shot_seed,
                    label=label,
                )
            except Exception as e:
                print(f"  [skip] evaluation failed for {label}: {e}")
                continue

            all_results.append({"alias": alias, "provider": spec["provider"], **result})
            print(
                f"  {label}: EM={result['exact_match']:.2f} F1={result['f1_score']:.2f} "
                f"BLEU={result['bleu']:.2f} R-L={result['rougeL']:.2f}"
            )

    print_results_table(all_results)

    run_timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    new_results = [
        {
            **{k: v for k, v in r.items() if args.save_predictions or k not in ("predictions", "ground_truths")},
            "run_timestamp": run_timestamp,
        }
        for r in all_results
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        existing = {"config": {}, "results": []}
        merged_results = new_results
        print(f"\n--overwrite: writing fresh results file ({len(merged_results)} entries).")
    else:
        existing = load_existing_summary(out_path)
        merged_results = merge_results(existing.get("results", []), new_results)
        added = len(merged_results) - len(existing.get("results", []))
        replaced = len(new_results) - max(0, added)
        print(
            f"\nAppend mode: {len(existing.get('results', []))} prior + "
            f"{len(new_results)} new -> {len(merged_results)} total "
            f"(replaced {replaced}, added {max(0, added)})."
        )

    summary = {
        "config": {
            "data_path": args.data_path,
            "n_test": len(test_data),
            "shots": args.shots,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "few_shot_seed": args.few_shot_seed,
            "openai_base_url": args.openai_base_url,
        },
        "last_updated": run_timestamp,
        "results": merged_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
