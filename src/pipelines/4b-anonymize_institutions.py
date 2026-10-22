#!/usr/bin/env python3
"""
Institution-level anonymization for IndoNutriQA.

Pass 4 (the existing 4-anonymize_using_extracted_ner.py) removes per-patient PII
(names, dates, MRNs, addresses). This second pass strips identifying *institution*
mentions -- hospitals where the cases were treated, and publishers attached to
academic citations -- which are not in the per-chunk NER list but still identify
the host site if released as-is.

Citations are preserved: once a chunk contains a "Daftar Pustaka" / "REFERENSI"
/ "References" header, everything from that point onward in the document is
treated as a bibliography and is NOT anonymized. Hospital and publisher names
inside the bibliography are scholarly citations (other people's research
sites, publishers of books being cited), not identifiers of the host
institution. The bibliography flag propagates across subsequent chunks of the
same PDF after sorting by (page_number, chunk_index).

Strings replaced (deterministic, regex-based, longest-pattern-first). Tags
follow the Indonesian-first convention established by `<NAMA>` / `<TGL>` /
`<NO_RM>` / `<LOKASI>`:
- Hospitals (`<RS>` -- the standard Indonesian abbreviation for "Rumah Sakit"):
    * "Rumah Sakit Umum Daerah X", "Rumah Sakit Umum X", "Rumah Sakit X"
    * "RSUD <name>" with optional continuation across a newline
    * Bare acronyms: RSUD, RSPAL, RSAB, RSP, RSDS, RSSA, RSHS, RSCM, RSUI, RSUP,
      RSU (only when followed by a capitalized continuation or end-of-word)
    * Named hospitals harvested from the source corpus: Saiful Anwar, Soetomo,
      Cipto Mangunkusumo, Hasan Sadikin, Ramelan, Panti Waluya, Bhayangkara,
      Bangil, "Karsa Husada", "Abdoel Moeloek", "W. Z. Johannes Kupang", ...
- Publishers (`<PENERBIT>`):
    * "UB Press", "UGM Press", "UI Press", "USU Press", "Unair Press"
    * "Penerbit Buku Kedokteran EGC", "Penerbit FKUI", "Penerbit IDAI"
    * Fallback: "Penerbit <Title-Cased Words>"

Inputs:
    data/4-anonymize_using_extracted_ner/*.chunks.json   (text field)
    data/dataset/dataset.json
        (context, question, answer, report_title, medical_diagnosis,
        anthropometric_assessment, file)

Outputs (in-place by default, --output-dir to redirect):
    Same files, with institutional spans replaced by placeholders.
    Statistics printed to stdout: replacement counts per pattern + per file.

Filename renames:
    Any source filename that itself contains a hospital token (e.g.
    "THYPOID FEVER DI RUANG PAVILIUN THERESIA RS PANTI WALUYA SAWAHANMALANG.chunks.json"
    -> "THYPOID FEVER DI RUANG PAVILIUN THERESIA.chunks.json") is renamed in
    the chunks directory, and every matching `file` field in
    qa_pairs_with_chunks.json is updated.

Usage:
    python src/pipelines/4b-anonymize_institutions.py             # in-place
    python src/pipelines/4b-anonymize_institutions.py --dry-run   # report only
    python src/pipelines/4b-anonymize_institutions.py \
        --chunks-dir data/4-anonymize_using_extracted_ner \
        --qa-file data/dataset/dataset.json
"""

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Pattern registry. ORDER MATTERS: longest / most-specific first.
# ---------------------------------------------------------------------------

HOSPITAL_PLACEHOLDER = "<RS>"
PUBLISHER_PLACEHOLDER = "<PENERBIT>"

# Each tuple = (compiled regex, placeholder, friendly label for stats).
# `\s+` matches across newlines so the multi-line "RSUD Dr. Saiful Anwar\nMalang"
# variants get caught. Patterns are case-sensitive on purpose for the named
# entities (e.g. "Sawahan" the hospital ward, not "sawahan" the verb suffix).

# `_DR` matches all three casings of "dr." found in the corpus: `Dr.`, `dr.`,
# and `DR.`. The trailing `\s*` lets "dr.Soedono" (no space after the period)
# match too.
_DR = r"(?:DR|Dr|dr)\.\s*"

_HOSPITAL_PHRASES = [
    # Full "Rumah Sakit ..." phrases. The all-caps RUMAH SAKIT variant is
    # listed first so it is preferred over title-case in headers.
    r"RUMAH\s+SAKIT\s+PUSAT\s+ANGKATAN\s+LAUT(?:\s+(?:DR|Dr|dr)\.\s*[A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,3})?",
    r"Rumah\s+Sakit\s+Pusat\s+Angkatan\s+Laut(?:\s+" + _DR + r"[A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,3})?",
    # Title-case + all-caps variants of "Rumah Sakit Umum Daerah ..."
    r"Rumah\s+Sakit\s+Umum\s+Daerah(?:\s+[A-Z][\w.]*){1,4}",
    r"RUMAH\s+SAKIT\s+UMUM\s+DAERAH(?:\s+[A-Z][\w.]*){1,5}",
    r"Rumah\s+Sakit\s+Umum(?:\s+[A-Z][\w.]*){1,4}",
    r"RUMAH\s+SAKIT\s+UMUM(?:\s+[A-Z][\w.]*){1,5}",
    r"Rumah\s+Sakit(?:\s+[A-Z][\w.]*){1,5}",
    r"RUMAH\s+SAKIT(?:\s+[A-Z][\w.]*){1,5}",
    # RSUD / RSU acronyms with "dr." (any casing, optional space) and a
    # capitalized continuation. The trailing tokens must start with an
    # UPPERCASE letter -- this prevents absorbing Indonesian connector words
    # like "dengan", "untuk", "yang" into the hospital phrase (e.g.
    # "RSUD dr. Saiful Anwar Malang dengan keluhan" must match only the
    # "RSUD dr. Saiful Anwar Malang" prefix, leaving "dengan keluhan" intact).
    # `RSUD\.?` allows an optional period after the acronym
    # ("RSUD. DR. SYAIFUL ANWAR KOTA MALANG"). `{0,5}` is generous enough to
    # absorb "DR. NAME KOTA MALANG" or "DR. NAME PROVINSI JAWA TIMUR".
    # Handles all of:
    #   "RSUD Dr. Saiful Anwar Malang"
    #   "RSUD DR. SAIFUL ANWAR" (all caps)
    #   "RSUD dr. Soedono Madiun"
    #   "RSUD dr.Soedono Madiun"   (no space after period)
    #   "RSUD dr.\nSoedono Madiun" (linebreak inside)
    #   "RSUD. DR. SYAIFUL ANWAR KOTA MALANG"
    r"RSUD\.?\s+" + _DR + r"[A-Z][\w.]*(?:\s+(?:KOTA\s+|Kota\s+|Kabupaten\s+|KABUPATEN\s+|PROVINSI\s+|Provinsi\s+)?[A-Z][\w.]*){0,5}",
    r"RSUD\s+HAJI(?:\s+PROVINSI)?(?:\s+[A-Z][\w.]*){0,3}",
    r"RSUD\s+Haji(?:\s+Provinsi)?(?:\s+[A-Z][\w.]*){0,3}",
    r"RSUD\s+(?:Kabupaten\s+)?[A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,3}",
    r"RSU\s+[A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,3}",
    r"RSP\s+[A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,3}",
    # Specific named hospitals (harvested from the corpus).
    # These cover both the bare name and acronym-style mentions left over
    # after the broader patterns have already fired.
    r"\bRSPAL\b(?:\s+" + _DR + r"[A-Za-z][\w.]*(?:\s+[A-Za-z][\w.]*){0,2})?",
    r"\bRSAL\b(?:\s+" + _DR + r"[A-Za-z][\w.]*(?:\s+[A-Za-z][\w.]*){0,2})?",
    r"\bRSAB\b(?:\s+[A-Z][\w.]*){0,2}",
    r"\bRSCM\b",
    r"\bRSSA\b",
    r"\bRSDS\b",
    r"\bRSHS\b",
    r"\bRSUI\b",
    r"\bRSUP\b(?:\s+[A-Z][\w.]*){0,3}",
    r"\bRSUD\b",
    # Title-case named hospitals (most common form in narrative).
    r"\bRS\s+(?:Panti\s+Waluya|Persada|Bhayangkara|Bangil|Sidoarjo|Sahabat|Wava(?:\s+Husada)?|Oepomo|Haji|Ngudi(?:\s+Waluyo(?:\s+Wlingi)?)?|Probolinggo|Kediri|Aura(?:\s+Syifa)?|Karsa(?:\s+Husada)?|Soedono|Soepraoen)(?:\s+[A-Z][\w.]*){0,3}",
    # All-caps variants (common in `report_title` cover pages). The newline
    # inside "RS PANTI \nWALUYA" / "PANTI \nWALUYA" is absorbed by `\s+`.
    r"\bRS\s+(?:PANTI\s+WALUYA|PERSADA|BHAYANGKARA|BANGIL|SIDOARJO|SAHABAT|WAVA(?:\s+HUSADA)?|OEPOMO|HAJI|NGUDI(?:\s+WALUYO(?:\s+WLINGI)?)?|PROBOLINGGO|KEDIRI|AURA(?:\s+SYIFA)?|KARSA(?:\s+HUSADA)?|SOEDONO|SOEPRAOEN)(?:\s+[A-Z][\w.]*){0,3}",
    # Bare hospital names without the "RS" prefix (PDF artifact where "RS\n"
    # got detached, or already-replaced "<RS>" precedes them).
    r"\bWava\s+Husada(?:\s+(?:Kepanjen|KEPANJEN))?(?:\s+(?:Malang|MALANG))?",
    r"\bWAVA\s+HUSADA(?:\s+(?:Kepanjen|KEPANJEN))?(?:\s+(?:Malang|MALANG))?",
    r"\b(?:Kepanjen|KEPANJEN)(?:\s+(?:Malang|MALANG))?",
    # Indonesian military hospitals: "RST Tk II dr. <Name>" or "RS TK II Dr. <Name>".
    # Both "RST" and bare "RS" prefixes appear; TK/Tk/tk and any-case "dr.".
    r"\bRST\s+(?:TK|Tk|tk)\s+[IVX]+\s+" + _DR + r"[A-Z][\w.]*",
    r"\bRS\s+(?:TK|Tk|tk)\s+[IVX]+\s+" + _DR + r"[A-Z][\w.]*",
    r"\bRST\s+" + _DR + r"[A-Z][\w.]*",
    r"\bRST\b",
    r"\bSoepraoen",
    r"\bSOEPRAOEN",
    # The host university (other universities appearing in references are
    # protected by the bibliography-skip logic).
    r"\bUNIVERSITAS\s+BRAWIJAYA(?:\s+MALANG)?",
    r"\bUniversitas\s+Brawijaya(?:\s+Malang)?",
    # Named hospitals (the most-cited in the corpus). `_DR` accepts any case.
    r"\b" + _DR + r"Saiful\s+Anwar(?:\s+Malang)?",
    r"\bSaiful\s+Anwar(?:\s+Malang)?",
    r"\b" + _DR + r"SAIFUL\s+ANWAR(?:\s+(?:MALANG|KOTA\s+MALANG|PROVINSI\s+JAWA\s+TIMUR))?",
    r"\bSAIFUL\s+ANWAR(?:\s+(?:MALANG|KOTA\s+MALANG|PROVINSI\s+JAWA\s+TIMUR))?",
    # SYAIFUL spelling variant (some reports use this transliteration).
    r"\b" + _DR + r"SYAIFUL\s+ANWAR(?:\s+(?:MALANG|KOTA\s+MALANG|PROVINSI\s+JAWA\s+TIMUR))?",
    r"\bSYAIFUL\s+ANWAR(?:\s+(?:MALANG|KOTA\s+MALANG|PROVINSI\s+JAWA\s+TIMUR))?",
    r"\b" + _DR + r"Syaiful\s+Anwar(?:\s+Malang)?",
    r"\bSyaiful\s+Anwar(?:\s+Malang)?",
    r"\b" + _DR + r"Soetomo(?:\s+Surabaya)?",
    r"\bSoetomo(?:\s+Surabaya)?",
    r"\b" + _DR + r"SOETOMO(?:\s+SURABAYA)?",
    r"\bSOETOMO(?:\s+SURABAYA)?",
    r"\b" + _DR + r"S[yY]aiful\s+Anwar(?:\s+Malang)?",
    r"\b" + _DR + r"Soedono(?:\s*,?\s*Madiun)?",
    r"\bSoedono(?:\s*,?\s*Madiun)?",
    r"\b" + _DR + r"Doris\s+Sylvanus",
    r"\bCipto\s+Mangunkusumo",
    r"\bHasan\s+Sadikin",
    r"\b" + _DR + r"Ramelan(?:\s+Surabaya)?",
    r"\bRamelan(?:\s+Surabaya)?",
    r"\b" + _DR + r"RAMELAN(?:\s+SURABAYA)?",
    r"\bRAMELAN(?:\s+SURABAYA)?",
    r"\bPanti\s+Waluya(?:\s+Sawahan(?:\s*Malang)?)?",
    r"\bPANTI\s+WALUYA(?:\s+SAWAHAN(?:\s*MALANG)?)?",
    r"\bSawahan\s*Malang",
    r"\bSAWAHAN\s*MALANG",
    r"\bKarsa\s+Husada",
    r"\bAbdoel\s+Moeloek",
    r"\bW\.\s*Z\.\s*Johannes(?:\s+Kupang)?",
    r"\bBhayangkara(?:\s+Manado)?",
    # Lowercase and bare variants left over in free-text notes.
    r"\b[Ww]ava\s+[Hh]usada(?:\s+[Kk]epanjen)?(?:\s+[Mm]alang)?",
    r"\b(?:soetomo|SOETOMO|Soetomo)\b(?:\s+[Ss]urabaya)?",
    r"\b[Ss]oepraoen",
    r"\b[Rr]amelan",
    r"\b[Pp]anti\s+[Ww]aluya",
    r"\bRumkital\b",
    # Generic placeholder used in the corpus.
    r"Rumah\s+Sakit\s+X(?:\s+[A-Z][\w.]*){0,3}",
]

# Personal names that survived the NER pass (presenter names in appendix forms).
PERSON_PLACEHOLDER = "<NAMA>"
_PERSON_PHRASES = [
    r"Sekar\s+Ramadhanti\s+Asyahir",
    r"\bAsyahir\b",
]

_PUBLISHER_PHRASES = [
    # Specific named publishers (longest first).
    r"Penerbit\s+Buku\s+Kedokteran\s+EGC",
    r"Penerbit\s+FKUI",
    r"Penerbit\s+IDAI",
    r"\bUB\s+Press\b",
    r"\bUGM\s+Press\b",
    r"\bUI\s+Press\b",
    r"\bUSU\s+Press\b",
    r"\bUnair\s+Press\b",
    # Generic fallback - "Penerbit X" for any title-cased X.
    r"Penerbit\s+[A-Z][\w]*(?:\s+[A-Z][\w]*){0,2}",
]


def build_patterns() -> List[Tuple[re.Pattern, str, str]]:
    """Compile all patterns. Returned in (longest-first) order."""
    compiled: List[Tuple[re.Pattern, str, str]] = []
    for phrase in _HOSPITAL_PHRASES:
        compiled.append((re.compile(phrase), HOSPITAL_PLACEHOLDER, phrase))
    for phrase in _PUBLISHER_PHRASES:
        compiled.append((re.compile(phrase), PUBLISHER_PLACEHOLDER, phrase))
    for phrase in _PERSON_PHRASES:
        compiled.append((re.compile(phrase), PERSON_PLACEHOLDER, phrase))
    return compiled


# ---------------------------------------------------------------------------
# Replacement core
# ---------------------------------------------------------------------------

# --- Orphan cleanup ----------------------------------------------------------
# These patterns mop up tails like "<RS> DR. RAMELAN SURABAYA" or
# "<RS> JAWA TIMUR" that get left behind when the leading "RSUD"/"RSPAL"
# acronym was matched by a generic pattern before a fully-qualified
# "RSUD DR. X Y" pattern could fire. Apply AFTER all main patterns.

# (a) Doctor suffix in any case: "<RS> DR. RAMELAN", "<RS> Dr. Doris Sylvanus",
# "<RS> dr.Soedono Madiun", "<RS>. DR. SYAIFUL ANWAR KOTA MALANG".
# Trailing tokens MUST start with uppercase to avoid swallowing Indonesian
# connector words like "dengan", "untuk", "yang" (which were eating "keluhan"
# in earlier iterations). Allows an optional `.` or `,` between the
# placeholder and the doctor title (artifact of "RSUD." vs "RSUD " in PDFs).
_ORPHAN_DR_SUFFIX = re.compile(
    r"<RS>\s*[,.]?\s*(?:DR|Dr|dr)\.?\s*[A-Z][\w.]*"
    r"(?:\s*,?\s*(?:KOTA\s+|Kota\s+|Kabupaten\s+|KABUPATEN\s+|PROVINSI\s+|Provinsi\s+)?[A-Z][\w.]*){0,5}"
)

# (b) Known Indonesian place/region tokens that commonly trail a hospital
# acronym ("RSUD MALANG", "RSUD JAWA TIMUR", "RSPAL ... SURABAYA"). The
# alternation is explicit -- bare "<RS>\s+[A-Z][\w.]+" would absorb unrelated
# capitalized continuations like "<RS> PADA PASIEN".
_PLACE_TOKEN = (
    r"(?:Surabaya|SURABAYA|Malang|MALANG|Madiun|MADIUN|Jakarta|JAKARTA|"
    r"Sidoarjo|SIDOARJO|Bangil|BANGIL|Kupang|KUPANG|Manado|MANADO|"
    r"Yogyakarta|YOGYAKARTA|Semarang|SEMARANG|Probolinggo|PROBOLINGGO|"
    r"Salatiga|SALATIGA|Tangerang|TANGERANG|Lampung|LAMPUNG|"
    r"Bandung|BANDUNG|Jawa\s+Timur|JAWA\s+TIMUR|Jawa\s+Tengah|JAWA\s+TENGAH|"
    r"Jawa\s+Barat|JAWA\s+BARAT|Sumatera\s+Utara|SUMATERA\s+UTARA|"
    r"Provinsi\s+Jawa\s+Timur|PROVINSI\s+JAWA\s+TIMUR|"
    r"Pusat\s+Angkatan\s+Laut|PUSAT\s+ANGKATAN\s+LAUT|"
    r"Haji|HAJI)"
)
_ORPHAN_PLACE_SUFFIX = re.compile(
    # "<RS>, Madiun" -- a comma between the tag and the place name -- is a
    # common artifact of "RSUD X, Madiun" where only "RSUD X" was matched.
    # Allow optional ",." and any administrative prefix in either casing.
    r"<RS>\s*[,.]?\s+"
    r"(?:Kota\s+|KOTA\s+|Kabupaten\s+|KABUPATEN\s+|Provinsi\s+|PROVINSI\s+)?"
    + _PLACE_TOKEN + r"(?:\s+[A-Z][\w.]*){0,2}\.?"
)


def collapse_repeated_placeholders(text: str) -> str:
    """Clean up redundant placeholders and absorb orphan hospital tails:

    1. "<RS> dr. Name Place" left behind when "RSUD" was matched but the
       trailing doctor name was not.
    2. "<RS> SURABAYA" / "<RS> JAWA TIMUR" / "<RS> MALANG" -- the city or
       province following the bare acronym in the source.
    3. Adjacent runs of "<RS><RS>" or "<PENERBIT><PENERBIT>" produced by
       overlapping pattern matches.

    Iterates twice so a multi-stage cleanup ("<RS> dr. X SURABAYA"
    -> "<RS> SURABAYA" -> "<RS>") converges in one call.
    """
    for _ in range(2):
        text = _ORPHAN_DR_SUFFIX.sub("<RS>", text)
        text = _ORPHAN_PLACE_SUFFIX.sub("<RS>", text)
    text = re.sub(r"(?:<RS>\s*){2,}", "<RS>", text)
    text = re.sub(r"(?:<PENERBIT>\s*){2,}", "<PENERBIT>", text)
    return text


# Bibliography-section header markers. Once one of these appears as a section
# heading (start-of-line, optionally indented), everything from that point
# onward is treated as a reference list and is NOT anonymized -- hospital and
# publisher names there are citations, not identifiers of the host
# institution.
#
# Important: standalone "Referensi" without the line-start anchor is NOT
# matched, because it is also a common table column header in the clinical
# narrative ("Indeks | Hasil | Interpretasi | Referensi" -- the right-most
# column lists clinical reference values, not academic references).
_BIBLIOGRAPHY_HEADER = re.compile(
    r"(?:^|\n)\s*(?:DAFTAR\s+PUSTAKA|Daftar\s+Pustaka|REFERENSI|References)\b"
)


# When True, reference lists are anonymized as well. Appendices (forms, rotation
# logs) can follow the reference list in the same PDF, so skipping everything
# after the header leaves institution names in the released text.
ANONYMIZE_BIBLIOGRAPHY = False


def anonymize_text(
    text: str,
    patterns: List[Tuple[re.Pattern, str, str]],
    stats: Counter,
    in_bibliography: bool = False,
) -> Tuple[str, bool]:
    """Apply every compiled pattern, in order, accumulating replacement counts.

    Returns (modified_text, in_bibliography_after). The boolean lets the caller
    keep track of whether the bibliography section has started in this PDF;
    once it has, subsequent chunks in the same PDF are returned unchanged.
    """
    if in_bibliography and not ANONYMIZE_BIBLIOGRAPHY:
        return text, True

    m = None if ANONYMIZE_BIBLIOGRAPHY else _BIBLIOGRAPHY_HEADER.search(text)
    if m is None:
        # No bibliography header in this chunk -- anonymize the whole thing.
        for regex, placeholder, label in patterns:
            text, n = regex.subn(placeholder, text)
            if n:
                stats[label] += n
        return collapse_repeated_placeholders(text), False

    # Bibliography starts inside this chunk. Anonymize only the prefix.
    cut = m.start()
    prefix, suffix = text[:cut], text[cut:]
    for regex, placeholder, label in patterns:
        prefix, n = regex.subn(placeholder, prefix)
        if n:
            stats[label] += n
    return collapse_repeated_placeholders(prefix) + suffix, True


# ---------------------------------------------------------------------------
# Filename rename logic
# ---------------------------------------------------------------------------

# Substrings inside filenames that flag the file as institution-revealing.
# Lowercased substring match; the rename keeps the surrounding diagnosis text.
_FILENAME_HOSPITAL_MARKERS = [
    "rs panti waluya",
    "panti waluya",
    "sawahanmalang",
    "sawahan malang",
    "rspal",
    "rsud dr.",
    "rsud dr ",
    "saiful anwar",
    "soetomo",
    "ramelan",
    "cipto mangunkusumo",
    "hasan sadikin",
    "bhayangkara",
]


def compute_renamed(stem: str) -> str:
    """Strip hospital markers from a filename stem.

    Returns `stem` unchanged when no hospital marker is present (we don't want
    to incidentally rename files whose only sin is trailing whitespace -- that
    is a separate cleanup concern outside the scope of institution
    anonymization).

    For files that DO have a marker (e.g. "THYPOID FEVER DI RUANG PAVILIUN
    THERESIA RS PANTI WALUYA SAWAHANMALANG"), trim from the first marker
    onward, then drop any orphaned connector words like "DI RUANG"/"DI RS".
    """
    low = stem.lower()
    cut_at = len(stem)
    for marker in _FILENAME_HOSPITAL_MARKERS:
        idx = low.find(marker)
        if idx != -1 and idx < cut_at:
            cut_at = idx
    if cut_at == len(stem):
        return stem  # no marker found - leave the filename alone
    new_stem = stem[:cut_at].rstrip(" ,.;:-_\n\t")
    new_stem = re.sub(
        r"\s+(?:DI\s+RUANG|DI\s+PAVILIUN|DI\s+RS|DI\s+RSUD)\s*$",
        "",
        new_stem,
        flags=re.IGNORECASE,
    ).rstrip(" ,.;:-_")
    return new_stem


def plan_renames(filenames: List[str]) -> Dict[str, str]:
    """Return {old_basename: new_basename} for every file that needs renaming.
    The `.chunks.json` extension is preserved."""
    renames: Dict[str, str] = {}
    for name in filenames:
        if not name.endswith(".chunks.json"):
            continue
        stem = name[: -len(".chunks.json")]
        new_stem = compute_renamed(stem)
        if new_stem != stem and new_stem:  # ignore no-ops + empty results
            renames[name] = f"{new_stem}.chunks.json"
    return renames


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

def process_chunks(
    chunks_dir: Path,
    patterns: List[Tuple[re.Pattern, str, str]],
    dry_run: bool,
) -> Tuple[Counter, Dict[str, str]]:
    """Anonymize the `text` field of every chunk file in `chunks_dir`.

    Returns (replacement_stats, file_renames).
    """
    stats: Counter = Counter()
    file_stats: Counter = Counter()
    filenames = sorted(p.name for p in chunks_dir.iterdir() if p.suffix == ".json")
    renames = plan_renames(filenames)

    for fn in filenames:
        path = chunks_dir / fn
        chunks = json.loads(path.read_text(encoding="utf-8"))
        n_before = sum(stats.values())
        # Sort chunks by (page_number, chunk_index) so that bibliography
        # state propagates correctly across the document.
        chunks.sort(key=lambda c: (c.get("page_number", 0), c.get("chunk_index", 0)))
        in_biblio = False
        for ch in chunks:
            if "text" in ch and isinstance(ch["text"], str):
                ch["text"], in_biblio = anonymize_text(
                    ch["text"], patterns, stats, in_bibliography=in_biblio
                )
            # The `pdf_name` field embeds the original filename. Update so it
            # stays in lockstep with the renamed file on disk.
            if "pdf_name" in ch and fn in renames:
                old_pdf = ch["pdf_name"]
                # The pdf_name uses ".pdf"; the chunks file uses ".chunks.json".
                new_basename = renames[fn][: -len(".chunks.json")]
                ch["pdf_name"] = re.sub(
                    r"^.*\.pdf$",
                    f"{new_basename}.pdf",
                    old_pdf,
                )
        file_stats[fn] = sum(stats.values()) - n_before

        if not dry_run:
            new_path = chunks_dir / renames.get(fn, fn)
            new_path.write_text(
                json.dumps(chunks, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # If we renamed, delete the original (write went to new path).
            if new_path != path:
                path.unlink()

    return stats, renames


def _anonymize_value(
    value: Any,
    patterns: List[Tuple[re.Pattern, str, str]],
    stats: Counter,
) -> Any:
    """Recurse through strings / lists / dicts, anonymizing every string
    leaf. Needed because the `answer` field is structured as
    `{"text": [str, ...], "answer_start": [int, ...]}`; without this recursive
    descent, hospital mentions inside the answer-span text never get touched."""
    if isinstance(value, str):
        new_value, _ = anonymize_text(value, patterns, stats)
        return new_value
    if isinstance(value, list):
        return [_anonymize_value(v, patterns, stats) for v in value]
    if isinstance(value, dict):
        return {k: _anonymize_value(v, patterns, stats) for k, v in value.items()}
    return value


def process_qa_pairs(
    qa_path: Path,
    patterns: List[Tuple[re.Pattern, str, str]],
    renames: Dict[str, str],
    dry_run: bool,
) -> Counter:
    """Anonymize text fields in qa_pairs_with_chunks.json and update `file`
    references for renamed sources. `answer` is a structured dict whose
    `.text[]` list holds the span strings -- we descend into it via
    _anonymize_value()."""
    stats: Counter = Counter()
    data = json.loads(qa_path.read_text(encoding="utf-8"))

    text_fields = (
        "context",
        "answer",
        "question",
        "report_title",
        "medical_diagnosis",
        "anthropometric_assessment",
    )

    for ex in data:
        for k in text_fields:
            if k in ex:
                # Use the recursive helper so structured fields like
                # `answer = {"text": [str, ...], "answer_start": [int, ...]}`
                # get their string leaves anonymized too.
                ex[k] = _anonymize_value(ex[k], patterns, stats)
        # Rewire `file` to the new filename if its source was renamed.
        if "file" in ex and ex["file"] in renames:
            ex["file"] = renames[ex["file"]]

    if not dry_run:
        qa_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--chunks-dir",
        default="data/4-anonymize_using_extracted_ner",
        help="Directory holding *.chunks.json files (modified in place).",
    )
    parser.add_argument(
        "--qa-file",
        default="data/dataset/dataset.json",
        help="Path to the consolidated QA pairs JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing any files.",
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="If set, copy each input file to this dir before modifying.",
    )
    parser.add_argument(
        "--anonymize-bibliography",
        action="store_true",
        help="Also anonymize reference lists and any text after them.",
    )
    args = parser.parse_args()
    global ANONYMIZE_BIBLIOGRAPHY
    ANONYMIZE_BIBLIOGRAPHY = args.anonymize_bibliography

    chunks_dir = Path(args.chunks_dir)
    qa_path = Path(args.qa_file)
    if not chunks_dir.is_dir():
        sys.exit(f"chunks dir not found: {chunks_dir}")
    if not qa_path.is_file():
        sys.exit(f"qa file not found: {qa_path}")

    if args.backup_dir and not args.dry_run:
        backup = Path(args.backup_dir)
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copytree(chunks_dir, backup / chunks_dir.name, dirs_exist_ok=True)
        shutil.copy2(qa_path, backup / qa_path.name)
        print(f"[backup] copied originals to {backup}/")

    patterns = build_patterns()

    print(f"[chunks] processing {chunks_dir}/ ...")
    chunk_stats, renames = process_chunks(chunks_dir, patterns, args.dry_run)
    print(f"[chunks] {sum(chunk_stats.values())} replacements across "
          f"{len(list(chunks_dir.iterdir()))} files")
    if renames:
        print(f"[chunks] would rename {len(renames)} file(s):")
        for old, new in renames.items():
            print(f"   {old!r} -> {new!r}")

    print(f"\n[qa-pairs] processing {qa_path} ...")
    qa_stats = process_qa_pairs(qa_path, patterns, renames, args.dry_run)
    print(f"[qa-pairs] {sum(qa_stats.values())} replacements in QA pairs")

    # Combined per-pattern summary, sorted by frequency.
    combined = Counter()
    for c in (chunk_stats, qa_stats):
        combined.update(c)
    print("\n[summary] replacements per pattern (top 30):")
    for pattern, n in combined.most_common(30):
        print(f"  {n:5d}  {pattern}")

    if args.dry_run:
        print("\n[dry-run] no files were modified.")


if __name__ == "__main__":
    main()
