#!/usr/bin/env python3
"""
Extract medical NER entities from clinical case chunks.

Extracts:
- No. RM (Nomor Rekam Medis / Medical Record Number)
- Name (Nama Pasien / Patient Name)
- Alamat (Address / Alamat)
- Tanggal Lahir (Date of Birth / DOB)
- No MRS (Nomor Masuk Rumah Sakit / Hospital Admission Number)

Uses regex patterns tailored to Indonesian clinical documents.
Processes .chunks.json files and outputs extracted NER to JSON.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

URL_PATTERN = re.compile(
    r"https?://[^\s]+",
    flags=re.IGNORECASE
)

# Regex patterns for NER extraction
PATTERNS = {
        "name": [
        # Nama: atau Nama Pasien:
        r"(?:Nama|Nama\s+Pasien)\s*:\s*([A-ZÀ-ÖØ-öø-ÿ\.\- \t]{2,120})(?=\r?\n|,|RM|No\.|Umur|Age|Jenis|Kelamin|TTL|DOB|$)",

        # Pasien:
        r"Pasien\s*:\s*([A-ZÀ-ÖØ-öø-ÿ\.\- \t]{2,120})(?=\r?\n|,|RM|No\.|Umur|$)",
    
        # Prefix An., By., Ny. + 1 kata
        r"Nama\s*:\s*([A-ZÀ-ÖØ-öø-ÿ\.\- \t]{2,120}?)(?=\r?\n|,|RM|No\.|Umur|Age|Jenis|Kelamin|TTL|DOB|Tgl|Tanggal|MRS|Masuk|$)",

        # Names followed by salutations after a comma. Examples:
        #   "Nadira Choirun\nNisa, An."
        #   "Mustofa, Ny."
        #   "Siska hidayah, sdri."
        r"\b([A-ZÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\.\-]{0,60}(?:[ \t\r\n]+[A-Za-zÀ-ÖØ-öø-ÿ\.\-]{1,60}){0,6})\s*,\s*(?:An\.|Ny\.|sdri\.|Tn\.)",

        r"\b((?:An\.|By\.|Ny\.)\s+[A-ZÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\.\-]{0,60})",
        # Multi-line name enclosed between pipes, preceding an RM: e.g.
        # '| JHANUAR\nRIFQI\nANTONIO | 2145971 |' -> capture the multi-line name
        r"\|\s*([A-ZÀ-ÖØ-öø-ÿ0-9][A-ZÀ-ÖØ-öø-ÿ0-9\.\-\s\n]{2,400}?)\s*(?=\|\s*[0-9]{5,12}\s*\|)",
            # Name immediately followed by a pipe and registration number, e.g.:
            # 'Zaky | 11617360' or 'Zahro | 11455206' — capture the name on the left
            r"\b([A-ZÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\.\-]{1,60}(?:[ \t]+[A-Za-zÀ-ÖØ-öø-ÿ\.\-]{1,60}){0,3})\s*(?=\|\s*[0-9]{5,12})",
    
    ],

    "address": [
        # Capture address blocks including internal newlines. Allow optional
        # leading lettered labels like 'd. Alamat:'. Exclude 'Alamat korespondensi'
        # or other correspondence-address labels via negative lookahead so we do
        # not capture correspondence addresses as primary addresses.
        r"(?:[A-Za-z]\.\s*)?(?:Alamat|Address)(?!\s+(?:korespondensi|koresponden|correspondence|korrespondensi))\s*[:\-]?\s*([\s\S]{1,400}?)\s*(?=\n\s*\n|$)",
    ],
    "tanggal_lahir": [

        # Label + tanggal format huruf (08 Mei 2022)
        r"(?:Tanggal\s*Lahir|TTL|DOB|Date\s+of\s+Birth)[\s:]*([0-3]?\d\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+(?:19|20)\d{2})",

        # Label + tanggal format numerik — NO MORE 3.4.11
        r"(?:Tanggal\s*Lahir|TTL|DOB|Date\s+of\s+Birth)[\s:]*([0-3]?\d[\/\-.][0-1]?\d[\/\-.](?:19|20)\d{2})",

        # Tanpa label, format huruf
        r"([0-3]?\d\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+(?:19|20)\d{2})",

        # Tanpa label, format numerik — NO MORE 3.4.11
        r"\b([0-3]?\d[\/\-.][0-1]?\d[\/\-.](?:19|20)\d{2})\b",
    ],


    "no_registrasi": [
    # 1. RM
    # Support 'No.Rekam Medis', 'No Rekam Medis' and 'No. RM' forms. Allow '*'
    # as a mask character in addition to 'x/X'. Accept optional '.' after 'No'
    # with optional whitespace before 'Rekam' to match 'No.Rekam Medis'.
    r"(?:No\.?\s*RM|No\.?\s*Rekam\s+Medis|No\s+Rekam\s+Medis|Nomor\s+Rekam\s+Medis|RM\s*No\.?|No\.?\s*RM\.)"
    r"(?:\s*\|\s*|\s*:\s*|\s*)+"
    r"([0-9xX\*\-\.\s]+)",
    r"RM\s*[:\-]\s*([0-9xX\*\-\.\s]+)",

    # 2. MRS
    r"(?:No\.?\s*MRS|Nomor\s+MRS|No\.?\s*Masuk)[\s:]*([0-9xX\-\.\s]+)",
    r"MRS\s*[:\-]\s*([0-9xX\-\.\s]+)",
    r"(?:Nomor\s+Masuk|No\s+Masuk)[\s:]*([0-9xX\-\.\s]+)",

    # 3. Registrasi
    r"(?:No\.?\s*Registrasi|Nomor\s+Registrasi|No\.?\s*Reg)[\s:]*([0-9xX\*\-\.\s]+)",
    r"Registrasi\s*[:\-]\s*([0-9xX\*\-\.\s]+)",
    r"(?:No\s+Registrasi|Nomor\s+Reg)[\s:]*([0-9xX\*\-\.\s]+)",

    # 4. Strict masked: 6 angka + 2 X (bukan tanggal)
    # Avoid matching when preceding character is '.' or ',' (likely decimal separator)
    r"(?<![0-9\.,])\b(\d{6}[xX]{2})\b",

    # 5. 8-digit or masked — Block dates (08/05/2022 → 08052022)
    # Avoid matching numbers that are part of decimals like '9,73443983' by ensuring
    # the match is not immediately preceded by a digit, comma, or dot.
    r"(?<![0-9\.,])\b(?!\d{2}(?:19|20)\d{2})([0-9]{6}[0-9xX]{2})\b",

    # 6. With prefix
    r"(?:Registrasi|No|Nomor)\s*[:\-\s]+(?!\d{2}(?:19|20)\d{2})([0-9]{6}[0-9xX]{2})\b",

    # 7. Isolated 8-char number
    r"(?:^|\s)(?!\d{2}(?:19|20)\d{2})([0-9]{6}[0-9xX]{2})(?!,\d)(?:\s|$)",

    # Numbers appearing after a pipe separator used in compact patient listings
    # e.g., 'Zaky | 11617360' → capture '11617360'
    r"\|\s*([0-9]{5,12})(?!,\d)",

]

}


def is_valid_registration_number(value: str) -> bool:
    """
    Filter out false positives for registration numbers.
    Exclude: birth dates (19XX/20XX format), student IDs (long sequences), NIP numbers.
    
    Args:
        value: The extracted value to validate
        
    Returns:
        True if this looks like a valid registration number
    """
    if not value or len(value.strip()) < 4:
        return False
    
    # Clean value
    value = value.strip()
    low_val = value.lower()
    
    # Remove common separators to get pure digits
    digits_only = re.sub(r'[xX\-\.\/\s]', '', value)
    
    # Filter out birth dates (starting with 19XX or 20XX)
    if re.match(r'^(19|20)\d{2}', digits_only):
        return False
    
    # Filter out very long sequences (likely student IDs, NIP, etc.)
    # Registration numbers are typically 8-10 digits
    if len(digits_only) > 12:
        return False
    
    x_count = low_val.count('x')
    if x_count == 1 and low_val.endswith('x'):
        return False

    if not re.search(r'\d', value):
        return False
    
    # Filter out sequences with excessive masking (likely redaction). Allow short
    # masked tails such as '1153****' or '1162xxxx' but reject very long masks
    # (6+ mask characters) which are unlikely to be valid registration numbers.
    if re.search(r'[xX\*]{6,}', value):  # Too many mask chars in a row
        return False

    # Ensure there's at least one digit present (avoid matching purely masked strings)
    if not re.search(r'\d', value):
        return False

    # Remove mask/breaker characters when analyzing digit length
    # (include '*' as a mask char to remove)
    digits_only = re.sub(r'[xX\*\-\.\/\s]', '', value)

    # Filter out birth dates (starting with 19XX or 20XX)
    if re.match(r'^(19|20)\d{2}', digits_only):
        return False

    # Filter out very long sequences (likely student IDs, NIP, etc.)
    # Registration numbers are typically 6-10 digits after removing masks
    if len(digits_only) > 12:
        return False
    
    return True


def is_valid_name(value: str, is_labeled: bool = False) -> bool:
    """
    Heuristic checks to filter out false-positive name extractions.

    Reject if the value contains certain non-name keywords (Indonesian/English),
    conjunctions indicating list/phrase (e.g., 'dan', 'atau'), or
    appears to be a longer descriptive phrase (too many words).
    """
    if not value or not value.strip():
        return False

    val = value.strip()
    low = val.lower()

    # Blocklist tokens that indicate the phrase is not a name
    BLOCKLIST = [
        r"\bmenggunakan\b",
        r"\bmetode\b",
        r"\bskrining\b",
        r"\bscreening\b",
        r"\bmenurut\b",
        r"\bhasil\b",
        r"\bpenggunaan\b",
        r"\bdiagnos(?:a|is)\b",
        r"\bpenilaian\b",
        r"\bpyms\b",
        r"\bpnst\b",
        r"\bstrong\b",
        r"\bstrong kids\b",
        # Additional blocks: ICU phrases, temporal phrases, transitional phrases, and units
        r"\bpada\s+saat\s+di\s+icu\b",
        r"\bhari\s+berturut-?turut\b",
        r"\bselain\s+itu\b",
        r"\bcm\b",
        # Monitoring / evaluation phrases, allow newline in whitespace
        r"\bselama\s+di\s+rumah\s+sakit\b",
        r"\bpada\s+monev\s+terakhir\b",
        # 'Berdasarkan data biokimia' phrase
        r"\bbiokimia\b",
        r"\bkanker\b",
        r"\bdefisit\b",
        r"\bideal\b",
        r"\bpasien\b",
        r"\bmengkaji\b",
        r"\bdiketahui\b",
        r"\bberdasarkan\b",
        r"\bmakan\b",
        r"\basupan\b",
        r"\bumum\b",
        r"\bnormal\b",
        r"\bpertanyaan\b",
        r"\basi\b",
        r"\bkasus\b",
        r"\bdietary\b",
        r"\bbaik\b",
        r"\bjam\b",
    ]

    for pat in BLOCKLIST:
        if re.search(pat, low):
            return False

    # If the phrase contains conjunctions (and/or) and is longer than a short name
    if (" dan " in low or " atau " in low or " & " in low) and len(low.split()) > 2:
        return False

    # Too long to be a name (likely descriptive phrase)
    if len(low.split()) > 6:
        return False

    # If it contains comma-separated clauses that look like sentence fragments, reject
    if "," in val and len(val.split()) > 4:
        return False

    # Reject any phrase that contains the token 'hari' (temporal words are not names)
    if "hari" in low:
        return False

    # If the value begins with an 'an.' prefix but the prefix is not capitalized ("An."),
    # reject to avoid picking up lowercase 'an.' that are not intended salutations.
    if re.match(r'(?i)^an\.?\s', val) and not re.match(r'^An\.?\s', value):
        return False

    # Remove common salutations/prefixes for the purpose of length/content checks
    prefix_re = re.compile(r'^(?:An\.?|Ny\.?|By\.?|Tn\.?|Bayi)\s+', flags=re.IGNORECASE)
    stripped = prefix_re.sub('', val).strip()

    # Remove punctuation and spaces to assess core character length
    core_chars = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ0-9]', '', stripped)
    # Reject if core is only a single character (e.g., 'An. C' -> core='C')
    # However, allow single-character cores when the name is explicitly labeled
    # (e.g., matched from a 'Nama:' label) or when a common salutational prefix
    # is present (An./Ny./Tn.). This handles cases like 'Nama : An. C'.
    if len(core_chars) <= 1:
        if not (is_labeled and re.match(r'^(?:An\.?|Ny\.?|By\.?|Tn\.?)\s+[A-Za-z]$', value.strip())):
            return False

    return True


def extract_entity(text: str, entity_type: str):
    """
    Extract all matches of an entity from text using regex patterns.
    Returns list of strings or None.
    """

    if entity_type not in PATTERNS:
        return None

    patterns = PATTERNS[entity_type]
    results = []

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL):

            # Raw captured text (may be multi-line for some patterns)
            raw = match.group(1)

            # === BLOCK: numbers inside URLs ===
            match_start = match.start(1)
            match_end = match.end(1)

            for url_match in URL_PATTERN.finditer(text):
                if url_match.start() <= match_start <= url_match.end():
                    raw = None
                    break

            if raw is None:
                continue

            # If this is an address, try to extend the capture using subsequent lines
            # when they look like continuation lines (RT, RW, Kota, Kelurahan, numeric lines, or indented lines).
            if entity_type == "address" and raw is not None:
                # Determine text after the captured group
                end_pos = match.end(1)
                following = text[end_pos:]
                appended_lines = []
                for line in following.splitlines(True):
                    # Skip leading blank lines; stop at blank line after we've started appending
                    if line.strip() == "":
                        if appended_lines:
                            break
                        else:
                            continue

                    # Stop if the line looks like a new labeled section (e.g., 'Diagnosa:', 'Tanggal Lahir:')
                    if re.match(r"^[A-Z][A-Za-z\s]{1,40}\s*:", line):
                        break

                    # If the following line begins with a bullet/symbol and then a labeled field
                    # (e.g., '▪ Tanggal MRS : 08 Mei 2022' or '▪ Ruang : ...'), treat it as
                    # the start of a new labeled section and stop extension.
                    clean_line = re.sub(r"^[\s•▪·\u2022\-\*]+", "", line)
                    # If the cleaned line starts with a capitalized label followed by ':' then stop
                    if re.match(r"^[A-Z][A-Za-z\s]{1,40}\s*:", clean_line):
                        break
                    # If the cleaned line contains explicit labels (Tanggal, Tgl, Ruang, Tanggal MRS, Tanggal Pengamatan,
                    # Agama, Suku, Registrasi etc.), stop
                    if re.search(r"\b(Tanggal|Tgl|Ruang|Tanggal\s+MRS|Tanggal\s+Pengamatan|Registrasi|No\.?\s*Registrasi|No\.?\s*RM|No\.?\s*MRS|Nomor|Nomor\s+Registrasi|Agama|Suku|Pekerjaan|Umur|Usia|Jenis\s+Kelamin)\b", clean_line, re.IGNORECASE):
                        break
                    # Also stop if the line begins with a lettered bullet/section like 'e. Agama :' or 'e. Suku :'
                    if re.match(r"^\s*[a-zA-Z]\.?\s+[A-Za-z\s]{1,40}\s*:\s*", clean_line):
                        break
                    # Also stop if the line begins with an explicit registration-like label
                    if re.match(r'^(?:No\.?\s*(?:RM|MRS|Registrasi|Reg|Masuk)|Nomor\s+(?:RM|MRS|Registrasi|Masuk)|Registrasi)\b', clean_line, re.IGNORECASE):
                        break

                    # Include common address continuation prefixes. For 'No'/'No.' only
                    # include if it is followed by a digit (e.g., 'No. 2A'), to avoid
                    # capturing labeled fields like 'No. Registrasi : ...'.
                    if re.match(r"^\s*(?:RT\b|RW\b|Kota\b|Kab\b|Kelurahan\b|Kec\b|Kecamatan\b|Provinsi\b|Dusun\b|Desa\b)", line, re.IGNORECASE) or re.match(r"^\s*(?:No\b|No\.)\s*\d", line, re.IGNORECASE):
                        appended_lines.append(line)
                        continue

                    # Include indented or lowercase continuation lines (but only after
                    # ensuring they are not actually labeled sections as checked above)
                    if re.match(r"^\s", line) or re.match(r"^[a-z0-9]", line):
                        appended_lines.append(line)
                        continue

                    # Include bullet or symbol-only lines (e.g., '▪', '•', '·', '*', '-') as continuations
                    if re.match(r"^\s*[•▪·\u2022\-\*]+\s*$", line):
                        appended_lines.append(line)
                        continue

                    # Heuristic: include lines that contain digits or commas *only if* they look
                    # like address fragments — i.e., they contain address keywords (RT, RW, No, Jl, Jalan, Kota, Desa, Dusun)
                    # or they contain a comma and letters (likely multi-part address).
                    if re.search(r"[0-9,]", line):
                        # Only accept numeric lines as address fragments when they
                        # contain address keywords or when 'No' is followed by digits
                        if re.search(r"\b(?:RT|RW|Jl\.?|Jalan|Kota|Provinsi|Desa|Dusun)\b", line, re.IGNORECASE) or re.search(r"\b(?:No\.?|No)\s*\d", line, re.IGNORECASE) or ("," in line and re.search(r"[A-Za-z]", line)):
                            appended_lines.append(line)
                            continue

                    # Otherwise stop extension
                    break

                if appended_lines:
                    raw = raw + "\n" + "".join(appended_lines)

            # Ambil group pertama → pastikan string
            extracted = str(raw) if raw is not None else ""

            # Normalize line endings and preserve internal newlines.
            extracted = extracted.replace("\r\n", "\n").replace("\r", "\n")

            # Collapse multiple spaces/tabs into one space (but do not remove newlines)
            extracted = re.sub(r"[ \t]+", " ", extracted)

            # Normalize spaces around newlines and collapse multiple newlines into one
            extracted = re.sub(r" *\n *", "\n", extracted)
            extracted = re.sub(r"\n{2,}", "\n", extracted)

            # Trim leading/trailing whitespace/newlines
            extracted = extracted.strip()

            # For addresses, remove trailing punctuation like '.' or ',' that often
            # end the address line in enumerated lists (e.g., '...Kab. Malang.').
            if entity_type == "address":
                extracted = re.sub(r"[\s\t\.,;:]+$", "", extracted)

            # === CLEANUP: Remove any standalone 'Jenis' tokens from extracted names
            # Examples:
            #  - 'An. John Doe Jenis' -> 'An. John Doe'
            #  - 'Bayi Ny. SLW Jenis' -> 'Bayi Ny. SLW'
            if entity_type == "name":
                # Remove any standalone 'Jenis' (case-insensitive), optionally followed by a period
                extracted = re.sub(r"\bJenis\b\.?", "", extracted, flags=re.IGNORECASE)
                # Remove standalone 'No' tokens when they are not followed by digits (e.g., 'An. C No' -> 'An. C')
                extracted = re.sub(r"\bNo\b\.?\b(?!\s*\d)", "", extracted, flags=re.IGNORECASE)
                # Collapse multiple spaces/tabs into a single space
                extracted = re.sub(r"[ \t]{2,}", " ", extracted)
                # Strip trailing commas/periods and surrounding whitespace
                extracted = re.sub(r"[ ,\.]+$", "", extracted).strip()

            # Apply name validation heuristics to reduce false positives
            if entity_type == "name":
                # Determine whether this match came from an explicit labeled pattern
                # (e.g., 'Nama : ...' or 'Pasien : ...'). If so, allow looser
                # validation such as single-letter cores with salutations.
                match_text = match.group(0) if match is not None else ""
                is_labeled = bool(re.search(r'^(?:Nama|Nama\s+Pasien|Pasien)\b', match_text, re.IGNORECASE))
                if not is_valid_name(extracted, is_labeled):
                    continue

            if not extracted:
                continue

            # Validation khusus registrasi
            if entity_type == "no_registrasi" and not is_valid_registration_number(extracted):
                continue

            results.append(extracted)

    # === FIX: remove duplicates safely ===
    # Convert items to string first → guaranteed hashable
    if results:
        # Remove exact duplicates while preserving order
        items = list(dict.fromkeys([str(item) for item in results]))

        # For names: prefer longer multi-line/compound matches over short substrings
        # (e.g., prefer 'JHANUAR\nRIFQI\nANTONIO' and drop standalone 'ANTONIO').
        if entity_type == "name" and len(items) > 1:
            filtered = []
            for it in items:
                # If 'it' is a substring of another longer item, drop it
                if any((it != other and it in other) for other in items):
                    continue
                filtered.append(it)
            items = filtered

        return items

    return None



def extract_all_entities(text: str) -> Dict[str, Optional[str]]:
    """
    Extract all medical NER entities from text.
    
    Returns:
        Dict with keys: no_rm, name, address, tanggal_lahir, no_mrs, no_registrasi
    """
    return {
        "name": extract_entity(text, "name"),
        "address": extract_entity(text, "address"),
        "tanggal_lahir": extract_entity(text, "tanggal_lahir"),
        "no_registrasi": extract_entity(text, "no_registrasi"),
    }


def process_chunks_file(filepath: Path) -> Dict[str, Any]:
    """
    Process a .chunks.json file and extract ALL NER entities from every chunk/page.
    
    Returns:
        Dict with extracted entities (all occurrences) and metadata
    """
    try:
        with filepath.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
    except json.JSONDecodeError as e:
        return {
            "file": filepath.name,
            "status": "ERROR",
            "reason": f"JSON decode error: {e}",
            "entities": {},
            "unique_entities": {},
        }
    
    if not isinstance(chunks, list) or len(chunks) == 0:
        return {
            "file": filepath.name,
            "status": "SKIP",
            "reason": "Not a list or empty",
            "entities": {},
            "unique_entities": {},
        }
    
    # Initialize entity collections: map entity type -> list of occurrences
    extracted_entities: Dict[str, List[Dict[str, Any]]] = {
        "name": [],
        "address": [],
        "tanggal_lahir": [],
        "no_registrasi": [],
    }
    first_chunk_with_entity: Optional[int] = None
    
    # Loop through ALL chunks (not just first 10) to find all entities on every page
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        page_number = chunk.get("page_number")
        chunk_index = chunk.get("chunk_index")

        # if page_number!=50:
        #     continue
        
        # Extract entities from this chunk
        entities = extract_all_entities(text)
        
        # Collect all occurrences
        for key, values in entities.items():
            if values:
                for val in values:    # flatten item per item
                    occurrence = {
                        "value": val,  # val adalah STRING → AMAN
                        "chunk_index": chunk_index,
                        "page_number": page_number,
                    }
                    extracted_entities[key].append(occurrence)
                
                # Track first chunk with any entity
                if first_chunk_with_entity is None:
                    first_chunk_with_entity = i
    
    # Create unique values lists (preserving order, removing duplicates)
    unique_entities: Dict[str, List[str]] = {}
    for key, occurrences in extracted_entities.items():
        seen = set()
        unique_list = []
        for occ in occurrences:
            val = occ.get("value")
            if val and val not in seen:
                seen.add(val)
                unique_list.append(val)
        unique_entities[key] = unique_list
    
    return {
        "file": filepath.name,
        "status": "SUCCESS",
        "entities": extracted_entities,
        "unique_entities": unique_entities,
        "first_entity_chunk": first_chunk_with_entity,
        "total_chunks": len(chunks),
    }


def process_directory(input_dir: Path) -> List[Dict[str, Any]]:
    """
    Process all .chunks.json files in a directory.
    
    Returns:
        List of result dicts for each file
    """
    chunk_files = sorted(input_dir.glob("*.chunks.json"))
    results = []
    
    for filepath in chunk_files:
        result = process_chunks_file(filepath)
        results.append(result)
    
    return results


def save_results_to_json(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Save extracted NER results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def print_results_summary(results: List[Dict[str, Any]]) -> None:
    """Print a formatted summary of extraction results."""
    print("\n" + "="*80)
    print("NER Extraction Summary")
    print("="*80)
    
    successful = [r for r in results if r.get("status") == "SUCCESS"]
    skipped = [r for r in results if r.get("status") == "SKIP"]
    errors = [r for r in results if r.get("status") == "ERROR"]
    
    print(f"\nTotal files processed: {len(results)}")
    print(f"Successful extractions: {len(successful)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Errors: {len(errors)}")
    
    if successful:
        print("\n" + "-"*80)
        print("Sample Extractions:")
        print("-"*80)
        for result in successful[:5]:  # Show first 5
            print(f"\nFile: {result['file']}")
            entities = result.get("entities", {})
            for key, value in entities.items():
                if value:
                    print(f"  {key}: {value}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract medical NER entities from .chunks.json files"
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        default="data/2-remove_pre_pendahuluan",
        help="Directory containing .chunks.json files",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/3-extract_medical_ner/extracted_ner.json",
        help="Output JSON file for extracted NER",
    )
    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help="Print summary to console",
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return
    
    chunk_files = list(input_dir.glob("*.chunks.json"))
    if not chunk_files:
        print(f"No .chunks.json files found in {input_dir}")
        return
    
    print(f"Processing {len(chunk_files)} files from {input_dir}...")
    results = process_directory(input_dir)
    
    # Save results
    output_path = Path(args.output)
    save_results_to_json(results, output_path)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary if requested
    if args.summary:
        print_results_summary(results)


if __name__ == "__main__":
    main()


# # Basic run (processes data/2-remove_pre_pendahuluan/*.chunks.json, outputs to data/extracted_ner.json)
# python src/pipelines/3-extract_medical_ner.py

# # With custom input/output directories
# python src/pipelines/extract_medical_ner.py -i data/2-remove_pre_pendahuluan -o data/extracted_ner.json
# # With summary report printed to console
# python src/pipelines/extract_medical_ner.py --summary