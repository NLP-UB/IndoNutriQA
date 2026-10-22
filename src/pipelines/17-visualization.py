"""
Visualization Pipeline for IndoNutriQA

Generates publication-ready figures from existing analysis outputs.
All inputs are JSON files produced by earlier pipelines; this script
performs no inference or evaluation.

Outputs (Paper/figures/):
- main_results.pdf       3-panel main-paper figure:
                          (a) lexical F1 vs BERTScore F1 scatter
                          (b) error-taxonomy stacked bars
                          (c) few-shot lift bar chart for LLMs
- disease_heatmap.pdf    9-model x 7-disease F1 heatmap (appendix)

Run:
    python src/pipelines/17-visualization.py
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("Paper/figures")
BOOTSTRAP_PATH = Path("data/13-statistical_robustness/bootstrap_ci.json")
SEMANTIC_PATH = Path("data/14-semantic_evaluation/semantic_ci.json")
ERROR_TAX_PATH = Path("data/16-categorical_analysis/error_taxonomy.json")
DISEASE_PATH = Path("data/16-categorical_analysis/disease_breakdown.json")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================================
# Data assembly
# ============================================================================

# Friendly label per prediction-file stem + family classification.
MODEL_META = {
    "baseline__rifky":                    ("Rifky_IndoBERT",     "Fine-tuned"),
    "baseline__indolem":                  ("IndoLEM_IndoBERT",   "Fine-tuned"),
    "baseline__cahya":                    ("Cahya_IndoBERT",     "Fine-tuned"),
    "baseline__xlmr":                     ("XLM-R-large",        "Fine-tuned"),
    "baseline__mdeberta":                 ("mDeBERTa-v3-base",   "Fine-tuned"),
    "baseline__mt5":                      ("mT5",                "Fine-tuned"),
    "llm__gpt-4o-mini__0shot":            ("GPT-4o-mini (0)",    "Closed LLM"),
    "llm__gpt-4o-mini__5shot":            ("GPT-4o-mini (5)",    "Closed LLM"),
    "llm__gpt-4o__0shot":                 ("GPT-4o (0)",         "Closed LLM"),
    "llm__gpt-4o__5shot":                 ("GPT-4o (5)",         "Closed LLM"),
    "llm__llama-3.1-8b-instruct__0shot":  ("Llama-3.1-8B (0)",   "Open LLM"),
    "llm__llama-3.1-8b-instruct__5shot":  ("Llama-3.1-8B (5)",   "Open LLM"),
    "llm__qwen2.5-7b-instruct__0shot":    ("Qwen2.5-7B (0)",     "Open LLM"),
    "llm__qwen2.5-7b-instruct__5shot":    ("Qwen2.5-7B (5)",     "Open LLM"),
    "llm__sahabat-ai-llama3-8b__0shot":   ("Sahabat-AI (0)",     "Indonesian LLM"),
    "llm__sahabat-ai-llama3-8b__5shot":   ("Sahabat-AI (5)",     "Indonesian LLM"),
    "llm__merak-7b-v4__0shot":            ("Merak-7B-v4 (0)",    "Indonesian LLM"),
    "llm__merak-7b-v4__5shot":            ("Merak-7B-v4 (5)",    "Indonesian LLM"),
    "llm__biomistral-7b__0shot":          ("BioMistral-7B (0)",  "Medical LLM"),
    "llm__biomistral-7b__5shot":          ("BioMistral-7B (5)",  "Medical LLM"),
    "llm__medgemma-4b-it__0shot":         ("MedGemma-4B-IT (0)", "Medical LLM"),
    "llm__medgemma-4b-it__5shot":         ("MedGemma-4B-IT (5)", "Medical LLM"),
}

FAMILY_COLORS = {
    "Fine-tuned":     "#7f7f7f",
    "Closed LLM":     "#d95f02",
    "Open LLM":       "#1b9e77",
    "Indonesian LLM": "#7570b3",
    "Medical LLM":    "#e6ab02",
}

FAMILY_MARKERS = {
    "Fine-tuned":     "s",
    "Closed LLM":     "o",
    "Open LLM":       "^",
    "Indonesian LLM": "D",
    "Medical LLM":    "v",
}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def lexical_f1_by_stem() -> Dict[str, float]:
    """{stem: lexical_F1_point} from bootstrap_ci.json."""
    data = load_json(BOOTSTRAP_PATH)
    out = {}
    for row in data["results"]:
        out[row["name"]] = row["metrics"]["f1_score"]["point"]
    return out


def bertscore_by_stem() -> Dict[str, float]:
    """{stem: BERTScore_F1_point} from semantic_ci.json (already in percent)."""
    data = load_json(SEMANTIC_PATH)
    out = {}
    for stem, payload in data["results"].items():
        m = payload["metrics"].get("bertscore_f1")
        if m is not None:
            out[stem] = m["point"]
    return out


# ============================================================================
# Panel (a): Lexical F1 vs BERTScore F1 scatter
# ============================================================================

def panel_a_scatter(ax) -> None:
    lex = lexical_f1_by_stem()
    bert = bertscore_by_stem()

    # Draw the "metrics agree" diagonal first so it sits behind points.
    ax.plot([0, 100], [0, 100], "--", color="lightgray", linewidth=0.8, zorder=0,
            label="metrics agree")

    # Only label the diagnostic points; rely on color/marker legend for the rest.
    # Format: stem -> (text, dx, dy, ha, va)
    POINT_LABELS = {
        "baseline__mt5":                      ("mT5",                      2.0,  0.0,  "left",  "center"),
        "baseline__rifky":                    ("Rifky",                   -1.5,  1.5,  "right", "bottom"),
        "baseline__indolem":                  ("IndoLEM",                 -1.5,  1.5,  "right", "bottom"),
        "baseline__cahya":                    ("Cahya",                   -1.5, -1.5,  "right", "top"),
        "llm__llama-3.1-8b-instruct__5shot":  ("5-shot LLMs",              2.5,  0.0,  "left",  "center"),
    }

    seen_families = set()
    for stem, (label, family) in MODEL_META.items():
        if stem not in lex or stem not in bert:
            continue
        x, y = lex[stem], bert[stem]
        legend_label = family if family not in seen_families else None
        seen_families.add(family)
        ax.scatter(
            x, y,
            s=45, marker=FAMILY_MARKERS[family], color=FAMILY_COLORS[family],
            edgecolor="black", linewidth=0.6, zorder=3, label=legend_label,
        )
        if stem in POINT_LABELS:
            text, dx, dy, ha, va = POINT_LABELS[stem]
            ax.annotate(text, (x, y), xytext=(x + dx, y + dy),
                        fontsize=9.5, ha=ha, va=va)

    ax.set_xlim(0, 75)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Lexical F1 (%)")
    ax.set_ylabel("BERTScore F1 (%)")
    ax.set_title("(a) Lexical vs. semantic")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=7,
              handletextpad=0.3, borderpad=0.35, labelspacing=0.25, borderaxespad=0.3)


# ============================================================================
# Panel (b): Error taxonomy stacked bars
# ============================================================================

# Use a green→red diverging palette: green=correct, ... red=noise.
TAX_ORDER = ["correct", "partial answer", "wrong span",
             "terminology mismatch", "hallucination", "noise"]
TAX_COLORS = {
    "correct":              "#2ca02c",
    "partial answer":       "#a6d854",
    "wrong span":           "#ffd92f",
    "terminology mismatch": "#fc8d62",
    "hallucination":        "#e7298a",
    "noise":                "#7570b3",
}

# Map taxonomy JSON keys (snake_case) to display labels above.
TAX_JSON_TO_LABEL = {
    "correct":              "correct",
    "partial_answer":       "partial answer",
    "wrong_span":           "wrong span",
    "terminology_mismatch": "terminology mismatch",
    "hallucination":        "hallucination",
    "noise":                "noise",
}

# Models to plot, ordered top-to-bottom in the horizontal stacked bar.
TAX_MODEL_ORDER = [
    ("baseline__rifky",                   "Rifky\n(fine-tuned)"),
    ("llm__gpt-4o-mini__5shot",           "GPT-4o-mini\n(5-shot)"),
    ("llm__llama-3.1-8b-instruct__5shot", "Llama-3.1-8B\n(5-shot)"),
]


def panel_b_taxonomy(ax) -> None:
    data = load_json(ERROR_TAX_PATH)
    results = data["results"]
    y_labels = []
    counts = {cat: [] for cat in TAX_ORDER}
    for stem, label in TAX_MODEL_ORDER:
        if stem not in results:
            continue
        y_labels.append(label)
        block = results[stem]
        for jkey, dkey in TAX_JSON_TO_LABEL.items():
            counts[dkey].append(block["counts"].get(jkey, 0))

    y_pos = np.arange(len(y_labels))
    left = np.zeros(len(y_labels))
    n_total = sum(counts[c][0] for c in TAX_ORDER) if y_labels else 50

    for cat in TAX_ORDER:
        vals = np.array(counts[cat], dtype=float)
        pct = 100.0 * vals / n_total
        ax.barh(y_pos, pct, left=left, color=TAX_COLORS[cat],
                edgecolor="white", linewidth=0.6, label=cat, height=0.7)
        # Annotate slices >= 8% so labels don't crowd small segments.
        for yi, (l, p) in enumerate(zip(left, pct)):
            if p >= 8:
                ax.text(l + p / 2, yi, f"{int(round(p))}%",
                        ha="center", va="center", fontsize=9.5,
                        color="white" if cat in ("noise", "hallucination",
                                                  "terminology mismatch")
                        else "black")
        left = left + pct

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of sampled outputs (n=50)")
    ax.set_title("(b) Error taxonomy")
    ax.invert_yaxis()
    # Single legend below the chart (compact)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, frameon=False, fontsize=9.5,
              handlelength=1.0, columnspacing=0.8, handletextpad=0.4)


# ============================================================================
# Panel (c): Few-shot lift bar chart
# ============================================================================

LIFT_MODELS = [
    ("GPT-4o-mini",   "llm__gpt-4o-mini__0shot",            "llm__gpt-4o-mini__5shot",            "Closed LLM"),
    ("Llama-3.1-8B",  "llm__llama-3.1-8b-instruct__0shot",  "llm__llama-3.1-8b-instruct__5shot",  "Open LLM"),
]


def panel_c_lift(ax) -> None:
    lex = lexical_f1_by_stem()
    x = np.arange(len(LIFT_MODELS))
    width = 0.32
    for i, (label, stem0, stem5, family) in enumerate(LIFT_MODELS):
        s0 = lex.get(stem0)
        s5 = lex.get(stem5)
        if s0 is None or s5 is None:
            continue
        color = FAMILY_COLORS[family]
        ax.bar(i - width / 2, s0, width, color=color, alpha=0.5,
               edgecolor="black", linewidth=0.6,
               label="0-shot" if i == 0 else None)
        ax.bar(i + width / 2, s5, width, color=color,
               edgecolor="black", linewidth=0.6,
               label="5-shot" if i == 0 else None)
        ax.text(i - width / 2, s0 + 1, f"{s0:.1f}",
                ha="center", va="bottom", fontsize=9.5)
        ax.text(i + width / 2, s5 + 1, f"{s5:.1f}",
                ha="center", va="bottom", fontsize=9.5)
        # Annotate lift above each pair
        lift = s5 - s0
        ax.annotate(
            f"+{lift:.1f}",
            xy=(i + width / 2, s5),
            xytext=(i, max(s0, s5) + 7),
            fontsize=10, ha="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="black"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in LIFT_MODELS])
    ax.set_ylim(0, 75)
    ax.set_ylabel("Lexical F1 (%)")
    ax.set_title("(c) Few-shot lift")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)


# ============================================================================
# Disease heatmap (appendix figure)
# ============================================================================

DISEASE_ORDER = [
    ("respiratory",          "Resp."),
    ("gastrointestinal",     "Gastro."),
    ("neurologic",           "Neuro."),
    ("hematologic_oncologic","Hem./Onc."),
    ("nutrition_growth",     "Nutr./Gr."),
    ("infectious",           "Infect."),
    ("ent",                  "ENT"),
]

HEATMAP_MODEL_ORDER = [
    ("baseline__rifky",                   "Rifky_IndoBERT"),
    ("baseline__indolem",                 "IndoLEM_IndoBERT"),
    ("baseline__cahya",                   "Cahya_IndoBERT"),
    ("baseline__xlmr",                    "XLM-R-large"),
    ("baseline__mdeberta",                "mDeBERTa-v3-base"),
    ("baseline__mt5",                     "mT5"),
    ("llm__gpt-4o-mini__0shot",           "GPT-4o-mini (0)"),
    ("llm__gpt-4o-mini__5shot",           "GPT-4o-mini (5)"),
    ("llm__gpt-4o__0shot",                "GPT-4o (0)"),
    ("llm__gpt-4o__5shot",                "GPT-4o (5)"),
    ("llm__llama-3.1-8b-instruct__0shot", "Llama-3.1-8B (0)"),
    ("llm__llama-3.1-8b-instruct__5shot", "Llama-3.1-8B (5)"),
    ("llm__medgemma-4b-it__0shot",        "MedGemma-4B-IT (0)"),
    ("llm__medgemma-4b-it__5shot",        "MedGemma-4B-IT (5)"),
]


def fig_disease_heatmap() -> None:
    data = load_json(DISEASE_PATH)
    results = data["results"]

    matrix = np.full((len(HEATMAP_MODEL_ORDER), len(DISEASE_ORDER)), np.nan)
    for i, (stem, _) in enumerate(HEATMAP_MODEL_ORDER):
        if stem not in results:
            continue
        for j, (cat_key, _) in enumerate(DISEASE_ORDER):
            cell = results[stem].get(cat_key)
            if cell and cell.get("f1") is not None:
                matrix[i, j] = cell["f1"]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=70)

    ax.set_xticks(np.arange(len(DISEASE_ORDER)))
    ax.set_xticklabels([d[1] for d in DISEASE_ORDER], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(HEATMAP_MODEL_ORDER)))
    ax.set_yticklabels([m[1] for m in HEATMAP_MODEL_ORDER])

    # Annotate each cell with its F1 value.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                txt, color = "–", "black"
            else:
                txt = f"{v:.1f}"
                color = "white" if v > 38 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=10, color=color)

    # Visually separate the family bands.
    ax.axhline(3.5, color="black", linewidth=0.8)
    ax.axhline(5.5, color="gray", linewidth=0.5, linestyle="--")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("F1 (%)")
    ax.set_title("F1 by disease category")
    fig.tight_layout()
    out = OUTPUT_DIR / "disease_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ============================================================================
# 3-panel main figure
# ============================================================================

def fig_main_3panel() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.3))
    panel_a_scatter(axes[0])
    panel_b_taxonomy(axes[1])
    panel_c_lift(axes[2])
    fig.tight_layout()
    # Extra bottom room for the panel-(b) legend underneath.
    fig.subplots_adjust(bottom=0.28, wspace=0.30)
    out = OUTPUT_DIR / "main_results.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ============================================================================
# Entrypoint
# ============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_main_3panel()
    fig_disease_heatmap()


if __name__ == "__main__":
    main()
