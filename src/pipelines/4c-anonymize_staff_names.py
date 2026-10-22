#!/usr/bin/env python3
"""
Staff-name anonymization for IndoNutriQA.

Pass 4 removes patient identifiers and pass 4b removes hospital and publisher
names. Neither touches the names of the staff who appear in the reports:

- the treating physician in the patient header ("DPJP : dr. <name>, Sp.A")
- clinical instructors, supervisors and dietitians in signature blocks
  ("Mengetahui, Clinical Instructor <name>, SST. Gz  NIK. 538/B")
- employee numbers (NIK / NIP)
- the report authors on cover pages and forms (name and student number, NIM)
- presenters of education sessions ("Pemateri | <name>")

This pass replaces names with `<NAMA>`, employee numbers with
`<NO_PEGAWAI>` and student numbers with `<NIM>`, and one remaining hospital name with `<RS>`.
Names of cited authors in reference lists carry no academic or professional
degree suffix in the corpus, so they are left unchanged.

Usage:
    python src/pipelines/4c-anonymize_staff_names.py --dry-run
    python src/pipelines/4c-anonymize_staff_names.py
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

NAME = "<NAMA>"
STAFF_ID = "<NO_PEGAWAI>"
STUDENT_ID = "<NIM>"
HOSPITAL = "<RS>"

_DEGREE = (
    r"(?:SST|Sst|S\.\s?ST|S\.?\s?Gz|S\.?\s?Tr\.?\s?Gz|M\.\s?Gz|M\.\s?Sc|M\.\s?Kes|M\.\s?Si|"
    r"M\.\s?Biomed|M\.\s?Pd|MPH|MARS|S\.?\s?KM|SKM|S\.\s?Kep|S\.\s?Ked|Ph\.?D|PhD|RDN|RD\b|"
    r"Sp\.\s?[A-Za-z]+|SpA|SpOG|dr\.|Ns\.)"
)

# (regex, replacement, label). Order matters: the physician line goes first.
PATTERNS = [
    # Treating physician in the patient header: keep the label, drop the rest of the line.
    (re.compile(r"(DPJ[PB]\s*:\s*)[^\n]*"), r"\g<1>" + NAME, "DPJP line"),
    # Report authors on cover pages and forms: "Nama Intern : X", "NAMA MAHASISWA: X".
    (re.compile(r"((?:Nama\s+Intern|NAMA\s+MAHASISWA|Nama\s+Mahasiswa)\s*:?\s*)[^\n]+"),
     r"\g<1>" + NAME, "author name label"),
    # A name on the line before a student number ("Oleh:\nX\n2401...", "X - 2401...").
    (re.compile(
        r"\b(?!NIM\b)[A-Z][A-Za-z.'`]+(?:[ \t]+[A-Za-z.'`]+){0,5}(?=[ \t]*[-\u2013]?[ \t]*\n?[ \t]*2[0-5]\d{13}\b)"
    ), NAME, "name before student number"),
    # Student numbers (NIM): 15 digits starting with 20-25.
    (re.compile(r"(?<![\d.])2[0-5]\d{13}(?!\d)"), STUDENT_ID, "NIM"),
    # All-caps names under the supervisor heading on cover pages.
    (re.compile(r"(DOSEN\s+PEMBIMBING\s*\n)[A-Z' .]+"), r"\g<1>" + NAME, "supervisor name"),
    # Presenters of education sessions.
    (re.compile(r"(Pemateri[ \t]*\|?[ \t]*)(?!<NAMA>)[A-Z][^\n]*"), r"\g<1>" + NAME, "presenter"),
    (re.compile(r"(Koordinasi\s+bersama[ \t]*\|?[ \t]*)(?!Ahli|Perawat|Petugas|Tim|Dokter|<NAMA>)[A-Z][^\n|]*"),
     r"\g<1>" + NAME, "coordinator"),
    # A name repeated in parentheses after an already replaced name.
    (re.compile(r"(<NAMA>\s*)\([A-Z][A-Za-z.' ]+\)"), r"\g<1>", "name in parentheses"),
    # Names that appear once in free text.
    (re.compile(r"Ariesta\s+Widya(?:\s+I\.)?"), NAME, "listed name"),
    # "<Name>, <degree> ..." in signature blocks.
    (re.compile(
        r"\b[A-Z][A-Za-z.'`]+(?:[ \t]+[A-Za-z.'`]+){0,5},\s*" + _DEGREE + r"[\w.,() \t-]{0,40}"
    ), NAME, "name with degree"),
    # Employee numbers, and digits left behind after a replaced number.
    (re.compile(r"\b(?:NIK|NIP)\.?\s*:?\s*[0-9][0-9 /A-Za-z.-]{0,60}"), STAFF_ID, "NIK/NIP"),
    (re.compile(r"(<NO_PEGAWAI>)[ \t]*\d[\d \t/]*"), r"\g<1>", "NIK/NIP tail"),
    # A hospital that is written as a person-style name.
    (re.compile(r"(?:\bdr\.?\s*)?\bSoekanto\s+Jahja"), HOSPITAL, "hospital name"),
]

def anonymize(text, stats):
    for regex, repl, label in PATTERNS:
        text, n = regex.subn(repl, text)
        if n:
            stats[label] += n
    return text


def walk(obj, stats, skip_keys=("file", "id", "pdf_name")):
    """Anonymize every string value except identifiers and file names."""
    if isinstance(obj, str):
        return anonymize(obj, stats)
    if isinstance(obj, list):
        return [walk(x, stats, skip_keys) for x in obj]
    if isinstance(obj, dict):
        return {k: (v if k in skip_keys else walk(v, stats, skip_keys)) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--chunks-dir", default="data/chunks")
    ap.add_argument("--qa-file", default="data/dataset/dataset.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    chunks_dir, qa_path = Path(args.chunks_dir), Path(args.qa_file)
    if not chunks_dir.is_dir() or not qa_path.is_file():
        sys.exit("chunks dir or qa file not found")

    stats = Counter()
    files = sorted(chunks_dir.glob("*.chunks.json"))
    for f in files + [qa_path]:
        data = json.loads(f.read_text(encoding="utf-8"))
        new = walk(data, stats)
        if new != data and not args.dry_run:
            f.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"processed {len(files)} chunk files + {qa_path}")
    for label, n in stats.most_common():
        print(f"{n:6d}  {label}")
    if args.dry_run:
        print("[dry-run] no files were modified.")


if __name__ == "__main__":
    main()
