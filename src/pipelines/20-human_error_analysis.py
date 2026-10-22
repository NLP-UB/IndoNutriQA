#!/usr/bin/env python3
"""Human error analysis of system outputs, coded from dietitian free text.

The error taxonomy in 14-semantic_evaluation.py is assigned by the LLM judge.
That makes it circular as evidence about the judge, and it inherits whatever
blind spots the judge has. This pipeline builds the independent human
counterpart: during the RD calibration study (18-rd_judge_agreement.py) the
three dietitians could leave a free-text note on any unit, and 53 such notes
were written over 41 of the 50 units. The notes are unprompted -- no taxonomy
was shown to the annotators -- so the error *identification* is human and
un-primed.

Coding procedure. One author open-coded the 53 notes into the nine categories
below, then a second pass re-applied the finalised codebook to every note from
scratch; CODES holds the result. A note can carry several codes (a dietitian
often names an omission and a format violation in one sentence), so codes sum
to more than 53. Assignment is stored per (rater, unit) rather than derived by
keyword matching, because the notes are short clinical Indonesian in which the
same word carries different codes depending on the object ("tidak lengkap" on
the candidate is OMISSION, on the reference is REF_PROBLEM).

Usage:
    python src/pipelines/20-human_error_analysis.py report
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

EXPORT_DIR = Path("data/18-rd_judge_agreement/exports")
SAMPLE_PATH = Path("data/18-rd_judge_agreement/annotation_sample.json")
OUT_DIR = Path("data/20-human_error_analysis")
OUT_PATH = OUT_DIR / "human_errors.json"

# Label Studio Community forces one project per annotator when the annotators
# must be blind to each other, so the project number is the rater identity.
PROJECT_TO_RATER = {"project-2": "RD1", "project-3": "RD2", "project-5": "RD3"}

DIMENSIONS = ("factuality", "completeness", "safety")

# ---------------------------------------------------------------------------
# Codebook. Keys are stable identifiers used in the paper table.
# ---------------------------------------------------------------------------
CODEBOOK = {
    "OMISSION": "Incomplete coverage: a clinical point present in the reference is missing",
    "PES_FORMAT": "NCP/PES structural violation: wrong expansion of P-E-S, or domains collapsed",
    "CONTRADICTION": "Claim contradicts the patient's own assessment data",
    "NUMERIC": "Numeric or arithmetic error in a prescription or interpretation",
    "UNVERIFIABLE": "Specifics added that the reference does not support and that cannot be checked",
    "NON_ANSWER": "Empty output, truncated mid-sentence, or does not address the question",
    "DEGENERATE": "Repetition, or source-document boilerplate and table markup leaking into the answer",
    "LANGUAGE": "Non-standard Indonesian or non-standard clinical terminology",
    "REF_PROBLEM": "The reference answer itself is wrong or contestable",
}

# (rater, unit) -> codes.  Derived from the dietitians' notes; see module docstring.
CODES: Dict[Tuple[str, str], Tuple[str, ...]] = {
    ("RD1", "U003"): ("NUMERIC",),
    ("RD1", "U004"): ("CONTRADICTION",),
    ("RD1", "U006"): ("PES_FORMAT",),
    ("RD1", "U013"): ("NUMERIC", "REF_PROBLEM"),
    ("RD1", "U016"): ("CONTRADICTION",),
    ("RD1", "U017"): ("PES_FORMAT",),
    ("RD1", "U029"): ("OMISSION", "REF_PROBLEM"),
    ("RD1", "U044"): ("PES_FORMAT",),
    ("RD1", "U046"): ("UNVERIFIABLE",),
    ("RD2", "U004"): ("CONTRADICTION",),
    ("RD2", "U007"): ("DEGENERATE",),
    ("RD2", "U009"): ("LANGUAGE",),
    ("RD2", "U014"): ("DEGENERATE",),
    ("RD2", "U025"): ("PES_FORMAT",),
    ("RD2", "U029"): ("UNVERIFIABLE",),
    ("RD2", "U038"): ("DEGENERATE",),
    ("RD2", "U040"): ("DEGENERATE",),
    ("RD3", "U001"): ("OMISSION", "CONTRADICTION"),
    ("RD3", "U002"): (),                                  # praise, not an error
    ("RD3", "U003"): ("OMISSION", "NUMERIC"),
    ("RD3", "U004"): ("CONTRADICTION",),
    ("RD3", "U005"): ("NON_ANSWER",),
    ("RD3", "U006"): ("PES_FORMAT", "OMISSION"),
    ("RD3", "U007"): ("DEGENERATE",),
    ("RD3", "U009"): ("REF_PROBLEM", "NUMERIC"),
    ("RD3", "U010"): ("NON_ANSWER",),
    ("RD3", "U011"): ("OMISSION",),
    ("RD3", "U012"): ("NON_ANSWER", "REF_PROBLEM"),
    ("RD3", "U013"): ("OMISSION", "REF_PROBLEM"),
    ("RD3", "U014"): ("NON_ANSWER", "UNVERIFIABLE"),
    ("RD3", "U016"): ("OMISSION",),
    ("RD3", "U017"): ("PES_FORMAT", "OMISSION"),
    ("RD3", "U019"): ("LANGUAGE",),
    ("RD3", "U020"): ("UNVERIFIABLE", "OMISSION"),
    ("RD3", "U021"): ("OMISSION",),
    ("RD3", "U023"): ("UNVERIFIABLE", "OMISSION"),
    ("RD3", "U024"): ("NON_ANSWER",),
    ("RD3", "U025"): ("PES_FORMAT",),
    ("RD3", "U027"): ("NON_ANSWER",),
    ("RD3", "U029"): ("OMISSION",),
    ("RD3", "U030"): ("REF_PROBLEM",),
    ("RD3", "U032"): ("OMISSION",),
    ("RD3", "U036"): ("OMISSION",),
    ("RD3", "U039"): ("OMISSION", "PES_FORMAT"),
    ("RD3", "U040"): ("LANGUAGE",),
    ("RD3", "U042"): ("OMISSION",),
    ("RD3", "U043"): ("OMISSION",),
    ("RD3", "U044"): ("PES_FORMAT",),
    ("RD3", "U045"): ("NON_ANSWER",),
    ("RD3", "U046"): ("UNVERIFIABLE",),
    ("RD3", "U048"): ("NON_ANSWER",),
    ("RD3", "U049"): ("OMISSION",),
    ("RD3", "U050"): ("OMISSION", "PES_FORMAT"),
}


def load_notes() -> List[dict]:
    """Pull every free-text note out of the per-rater Label Studio exports."""
    cats = {}
    if SAMPLE_PATH.exists():
        sample = json.loads(SAMPLE_PATH.read_text())
        units = sample.get("units", sample if isinstance(sample, list) else [])
        for u in units:
            cats[u["unit_id"]] = u.get("question_category", "?")

    notes: List[dict] = []
    for path in sorted(EXPORT_DIR.glob("*.json")):
        key = next((k for k in PROJECT_TO_RATER if k in path.name), None)
        if key is None:
            raise SystemExit(f"cannot map {path.name} to a rater; expected one of {list(PROJECT_TO_RATER)}")
        rater = PROJECT_TO_RATER[key]
        for task in json.loads(path.read_text()):
            unit = task["data"]["unit_id"]
            for ann in task.get("annotations", []):
                if ann.get("was_cancelled"):
                    continue
                scores, text, flag = {}, None, False
                for r in ann.get("result", []):
                    fn = r["from_name"]
                    if fn in DIMENSIONS:
                        scores[fn] = int(r["value"]["choices"][0])
                    elif fn == "comment":
                        text = " ".join(r["value"]["text"]).strip()
                    elif fn == "ref_problem":
                        flag = True
                if text:
                    notes.append({
                        "rater": rater, "unit_id": unit,
                        "question_category": cats.get(unit, "?"),
                        "scores": scores, "ref_problem_flag": flag, "note": text,
                        "codes": list(CODES.get((rater, unit), ())),
                    })
    return notes


def cmd_report(args: argparse.Namespace) -> None:
    notes = load_notes()

    missing = [(n["rater"], n["unit_id"]) for n in notes if (n["rater"], n["unit_id"]) not in CODES]
    if missing:
        raise SystemExit(f"uncoded notes (add them to CODES): {missing}")
    stale = [k for k in CODES if k not in {(n["rater"], n["unit_id"]) for n in notes}]
    if stale:
        raise SystemExit(f"CODES entries with no matching note: {stale}")
    bad = {c for n in notes for c in n["codes"]} - set(CODEBOOK)
    if bad:
        raise SystemExit(f"codes not in CODEBOOK: {sorted(bad)}")

    # A unit counts once per category even if two dietitians name it, so the
    # unit-level rate is not inflated by the uneven commenting rates below.
    units_by_code: Dict[str, set] = defaultdict(set)
    notes_by_code: Counter = Counter()
    for n in notes:
        for c in n["codes"]:
            units_by_code[c].add(n["unit_id"])
            notes_by_code[c] += 1

    commented_units = sorted({n["unit_id"] for n in notes})
    per_rater = Counter(n["rater"] for n in notes)

    # Mean dietitian score on units carrying each code, to show which error
    # types actually drive the score down.
    score_by_code: Dict[str, Dict[str, float]] = {}
    for c in CODEBOOK:
        rel = [n for n in notes if c in n["codes"]]
        if rel:
            score_by_code[c] = {
                d: round(sum(n["scores"][d] for n in rel if d in n["scores"]) / len(rel), 2)
                for d in DIMENSIONS
            }

    print("=" * 74)
    print("HUMAN ERROR ANALYSIS  (dietitian free-text notes, RD calibration sample)")
    print("=" * 74)
    print(f"  notes: {len(notes)}  over {len(commented_units)} of 50 units")
    print(f"  notes per dietitian: " + ", ".join(f"{r}={per_rater[r]}" for r in sorted(per_rater)))
    print()
    print(f"  {'category':16s} {'units':>6s} {'%of50':>7s} {'notes':>6s}   mean fact/compl/safety")
    for c, _ in notes_by_code.most_common():
        u = len(units_by_code[c])
        s = score_by_code[c]
        print(f"  {c:16s} {u:6d} {100*u/50:6.0f}% {notes_by_code[c]:6d}   "
              f"{s['factuality']:.2f} / {s['completeness']:.2f} / {s['safety']:.2f}")
    print()

    # Cross-rater confirmation: units where >1 dietitian independently gave the
    # same code. This is the one agreement statistic the notes can support.
    conf = {c: sorted(u for u in units_by_code[c]
                      if sum(1 for n in notes if n["unit_id"] == u and c in n["codes"]) > 1)
            for c in units_by_code}
    # Which question categories each error type attaches to. PES violations
    # only exist where a nutrition diagnosis is being written, so this is a
    # sanity check on the coding as much as a finding.
    by_cat: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for n in notes:
        for c in n["codes"]:
            by_cat[c][n["question_category"]].add(n["unit_id"])
    cats = ("factual", "inferential", "recommendation")
    print(f"  {'category':16s} " + " ".join(f"{c:>14s}" for c in cats) + "   (units)")
    for c, _ in notes_by_code.most_common():
        print(f"  {c:16s} " + " ".join(f"{len(by_cat[c].get(k, ())):14d}" for k in cats))
    print()

    print("  independently named by >1 dietitian:")
    for c in sorted(conf):
        if conf[c]:
            print(f"    {c:16s} {conf[c]}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "codebook": CODEBOOK,
        "n_notes": len(notes),
        "n_units_commented": len(commented_units),
        "n_units_total": 50,
        "notes_per_rater": dict(per_rater),
        "units_by_code": {c: sorted(v) for c, v in units_by_code.items()},
        "notes_by_code": dict(notes_by_code),
        "mean_scores_by_code": score_by_code,
        "cross_rater_confirmed": {c: v for c, v in conf.items() if v},
        "units_by_code_and_question_category": {
            c: {k: sorted(v) for k, v in d.items()} for c, d in by_cat.items()
        },
        "notes": notes,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"[report] wrote {OUT_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("report", help="code the notes and print the error table")
    s.set_defaults(func=cmd_report)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
