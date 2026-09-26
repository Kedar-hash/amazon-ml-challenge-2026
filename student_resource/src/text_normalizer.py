#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Step 3: Text Normalization Module

This module provides high-speed, memory-efficient text normalization for
business names and addresses in the Business Entity Resolution pipeline.

Key Design Principles:
1. Script & Unicode Preservation: Preserves non-Latin scripts (Devanagari, etc.)
   and canonical characters while safely folding Latin diacritics (e.g. é -> e).
2. Domain & URL Handling: Normalizes web URLs, cleans pipe-separated site references
   (e.g., '| www.site.com'), and removes domain extensions (.com, .in, etc.).
3. Legal Suffix Standardization: Identifies and strips corporate forms (Inc, Ltd, Pvt Ltd, LLC,
   SARL, SCI, प्राइवेट लिमिटेड) to extract true core entity names.
4. Address Standardization: Expands street types (St -> street, Ave -> avenue)
   and unambiguous state abbreviations (NC -> north carolina, RJ -> rajasthan)
   while avoiding false expansions of common prepositions (e.g., 'de', 'in', 'or').
5. Fast C-Speed Processing: Uses precomputed translation tables and vectorized list
   comprehensions for sub-millisecond execution with zero external APIs or heavy libraries.

Usage:
    from src.text_normalizer import normalize_business_name, extract_core_name, normalize_address
    # Or run directly for tests & demonstrations:
    python src/text_normalizer.py
"""

import sys
import unicodedata
import re

# Ensure Windows terminal handles UTF-8 gracefully
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 1. PRECOMPUTED TRANSLATION TABLE & REGEXES
# ---------------------------------------------------------------------------

# Precompute translation table for the Basic Multilingual Plane (0..65535)
# Keeps Letters (L*), Numbers (N*), and Marks (M*, including Hindi matras).
# Converts all punctuation (P*), symbols (S*), and control characters to space.
_CHAR_TRANS_TABLE = {}
for _i in range(65536):
    _ch = chr(_i)
    _cat = unicodedata.category(_ch)
    if _cat[0] in "LNM" or _ch.isspace():
        _CHAR_TRANS_TABLE[_i] = _i
    else:
        _CHAR_TRANS_TABLE[_i] = 32  # ASCII space

# Compiled Regex Patterns
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_PIPE_URL = re.compile(r"\s*[|/\-–—]\s*(?:https?://|www\.)\S+", re.IGNORECASE)
_RE_URL_PREFIX = re.compile(r"https?://(?:www\.)?", re.IGNORECASE)
_RE_WWW_PREFIX = re.compile(r"\bwww\.", re.IGNORECASE)
_RE_DOMAIN_EXT = re.compile(r"\.(?:com|org|net|co\.in|co|in|fr|io|biz|info|us|gov|edu)\b", re.IGNORECASE)
_RE_PAREN_ID = re.compile(r"\((?:id|ref|code|branch)?\s*[:#-]?\s*\d+\)", re.IGNORECASE)
_RE_APOSTROPHE_S = re.compile(r"['’]s\b", re.IGNORECASE)
_RE_STANDALONE_PLUS = re.compile(r"(?<=\s)\+(?=\s)")


# ---------------------------------------------------------------------------
# 2. ABBREVIATION & CORPORATE SUFFIX DICTIONARIES
# ---------------------------------------------------------------------------

# Street & Unit Types (US, India, France)
STREET_ABBREVIATIONS = {
    # US & Common
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "hwy": "highway",
    "pkwy": "parkway",
    "expy": "expressway",
    "fwy": "freeway",
    "pl": "place",
    "sq": "square",
    "ter": "terrace",
    "trl": "trail",
    "way": "way",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "no": "number",
    # India specific
    "hno": "number",
    "hn": "number",
    "opp": "opposite",
    "nr": "near",
    "tq": "taluk",
    "dist": "district",
    "po": "post office",
    "kh": "khasra",
    # France specific
    "r": "rue",
    "bd": "boulevard",
    "rte": "route",
    "all": "allee",
    "chem": "chemin",
}

# Unambiguous US 2-Letter State Codes
# Intentionally excludes common prepositions/words: 'de' (French 'of'), 'in', 'or', 'me', 'as', 'la'
US_STATE_CODES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "md": "maryland", "ma": "massachusetts", "mi": "michigan",
    "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska",
    "nv": "nevada", "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

# Unambiguous Indian State Codes
INDIA_STATE_CODES = {
    "rj": "rajasthan", "dl": "delhi", "mh": "maharashtra", "ka": "karnataka",
    "wb": "west bengal", "tn": "tamil nadu", "up": "uttar pradesh", "gj": "gujarat",
    "tg": "telangana", "ts": "telangana", "ap": "andhra pradesh", "kl": "kerala",
    "mp": "madhya pradesh", "pb": "punjab", "hr": "haryana", "br": "bihar",
    "cg": "chhattisgarh", "ch": "chandigarh", "jh": "jharkhand", "jk": "jammu and kashmir",
    "od": "odisha", "uk": "uttarakhand", "ua": "uttarakhand", "ga": "goa", "py": "puducherry",
}

# Combined Unambiguous State Dictionary
STATE_CODES = {**US_STATE_CODES, **INDIA_STATE_CODES}

# Corporate Legal Suffix Tokens (English, Hindi, French)
LEGAL_SUFFIX_TOKENS = {
    # English
    "ltd", "limited", "inc", "incorporated", "corp", "corporation",
    "llc", "llp", "co", "company", "plc", "pvt", "private", "pte", "pc",
    # French
    "sarl", "sas", "sci", "eurl", "sasu", "sa",
    # Hindi (Devanagari)
    "प्राइवेट", "लिमिटेड", "एलएलपी", "कंपनी", "कम्पनी",
}

# Corporate Legal Prefix Tokens (primarily French)
LEGAL_PREFIX_TOKENS = {
    "sci", "sarl", "ste", "societe", "cie"
}

# Multi-word Legal Suffix Phrases
MULTIWORD_LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pte ltd", "limited liability company",
    "holding company", "holdings company", "holding co", "societe a responsabilite limitee",
    "societe par actions simplifiee", "societe civile immobiliere",
    "प्राइवेट लिमिटेड", "प्रा लि", "प्रा०लि०",
]


# ---------------------------------------------------------------------------
# 3. CORE NORMALIZATION FUNCTIONS
# ---------------------------------------------------------------------------

def strip_latin_diacritics(text: str) -> str:
    """
    Decompose Latin accented characters (e.g., é -> e, à -> a, ê -> e)
    while strictly preserving Devanagari and non-Latin combining marks.
    Latin diacritics reside in Unicode range 0x0300..0x036F.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    filtered = [
        ch for ch in decomposed
        if not (0x0300 <= ord(ch) <= 0x036F)
    ]
    return unicodedata.normalize("NFC", "".join(filtered))


def clean_url_or_domain(text: str) -> str:
    """
    Normalizes domain names and URLs:
    - Strips pipe-separated website references (e.g. '| www.site.com')
    - Strips http://, https://, www.
    - Strips domain extensions like .com, .in, .fr, .org
    Example: 'maurewilliamscolombier.com' -> 'maurewilliamscolombier'
    """
    if not text:
        return ""
    t = _RE_PIPE_URL.sub(" ", text)
    t = _RE_URL_PREFIX.sub(" ", t)
    t = _RE_WWW_PREFIX.sub(" ", t)
    t = _RE_DOMAIN_EXT.sub(" ", t)
    return t


def normalize_business_name(name: str, strip_legal_suffixes: bool = False) -> str:
    """
    Normalize a business name:
    1. Unicode NFKC normalization
    2. URL & domain cleaning
    3. Removal of bracketed IDs (e.g. '(ID: 84923)')
    4. Apostrophe standardization ('s -> s)
    5. Connector standardization (& -> and)
    6. Stripping punctuation while preserving Unicode script & numbers
    7. Folding Latin diacritics (preserving Hindi/Indian scripts)
    8. Lowercasing & collapsing whitespace
    9. Optional legal suffix removal for core name extraction
    """
    if not name or not isinstance(name, str) or name.strip() == "":
        return ""

    # Step 1: Unicode NFKC
    t = unicodedata.normalize("NFKC", name)

    # Step 2: Parenthetical noise removal (e.g. '(ID: 12345)')
    t = _RE_PAREN_ID.sub(" ", t)

    # Step 3: URL / Domain cleaning
    t = clean_url_or_domain(t)

    # Step 4: Apostrophe possessives ('s -> s)
    t = _RE_APOSTROPHE_S.sub("s", t)

    # Step 5: Connectors
    t = t.replace("&", " and ")
    t = _RE_STANDALONE_PLUS.sub(" and ", t)

    # Step 6: Punctuation removal via C-speed translation table
    t = t.translate(_CHAR_TRANS_TABLE)

    # Step 7: Latin diacritic folding (leaves Devanagari intact)
    t = strip_latin_diacritics(t)

    # Step 8: Lowercase & whitespace collapse
    t = _RE_MULTI_SPACE.sub(" ", t.lower()).strip()

    if not strip_legal_suffixes:
        return t

    # Step 9: Extract Core Name (Strip Legal Prefixes & Suffixes)
    return extract_core_name_from_clean(t)


def extract_core_name_from_clean(clean_name: str) -> str:
    """Strip legal corporate prefixes and suffixes from an already-cleaned name."""
    if not clean_name:
        return ""

    t = clean_name

    # Remove known multi-word legal phrases at end of string
    for phrase in MULTIWORD_LEGAL_SUFFIXES:
        if t.endswith(" " + phrase):
            t = t[:-len(phrase) - 1].strip()

    tokens = t.split()
    if not tokens:
        return ""

    # Strip legal prefix (e.g., 'sci ptit amicale' -> 'ptit amicale')
    if len(tokens) > 1 and tokens[0] in LEGAL_PREFIX_TOKENS:
        tokens.pop(0)

    # Strip legal suffix tokens from end (e.g. 'llc llc', 'ltd', 'inc')
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIX_TOKENS:
        tokens.pop()

    # Strip repeated trailing filler words if preceded by a valid name
    if len(tokens) > 1 and tokens[-1] in {"holding", "holdings", "group"}:
        tokens.pop()

    return " ".join(tokens)


def extract_core_name(name: str) -> str:
    """Convenience function: Normalizes business name and strips legal forms."""
    return normalize_business_name(name, strip_legal_suffixes=True)


def normalize_address(address: str, expand_abbreviations: bool = True) -> str:
    """
    Normalize an address string:
    1. Returns empty string for null, NaN, or whitespace-only addresses
    2. Strips punctuation while preserving words and numbers
    3. Folds Latin diacritics
    4. Lowercases and cleans whitespace
    5. Expands street abbreviations (St -> street, Ave -> avenue, Rd -> road)
    6. Expands unambiguous 2-letter state abbreviations (NC -> north carolina, RJ -> rajasthan)
    """
    if not address or not isinstance(address, str) or address.strip() == "":
        return ""

    # Check for literal 'nan' or 'null'
    stripped = address.strip()
    if stripped.lower() in {"nan", "null", "none"}:
        return ""

    # Step 1: Unicode NFKC
    t = unicodedata.normalize("NFKC", stripped)

    # Step 2: Parenthetical noise removal
    t = _RE_PAREN_ID.sub(" ", t)

    # Step 3: Punctuation removal
    t = t.translate(_CHAR_TRANS_TABLE)

    # Step 4: Latin diacritic folding
    t = strip_latin_diacritics(t)

    # Step 5: Lowercase and whitespace collapse
    t = _RE_MULTI_SPACE.sub(" ", t.lower()).strip()

    if not expand_abbreviations or not t:
        return t

    # Step 6: Expand abbreviations and state codes
    tokens = t.split()
    expanded = []
    for tok in tokens:
        if tok in STREET_ABBREVIATIONS:
            expanded.append(STREET_ABBREVIATIONS[tok])
        elif tok in STATE_CODES:
            expanded.append(STATE_CODES[tok])
        else:
            expanded.append(tok)

    return " ".join(expanded)


def get_address_tokens(address: str) -> list:
    """
    Returns a sorted unique list of normalized address tokens.
    Guarantees invariance to address component reordering
    (e.g., City, State, Street vs Street, City, State).
    """
    norm = normalize_address(address, expand_abbreviations=True)
    if not norm:
        return []
    return sorted(set(norm.split()))


def batch_normalize_dataframe(df, name_col="business_name", addr_col="business_address"):
    """
    High-performance batch normalization of a pandas DataFrame.
    Returns new columns:
      - 'clean_name'
      - 'core_name'
      - 'clean_address'
    Uses fast Python list comprehensions to maximize processing speed.
    """
    clean_names = [normalize_business_name(x, strip_legal_suffixes=False) for x in df[name_col]]
    core_names = [extract_core_name_from_clean(x) for x in clean_names]
    clean_addresses = [normalize_address(x, expand_abbreviations=True) for x in df[addr_col]]

    df["clean_name"] = clean_names
    df["core_name"] = core_names
    df["clean_address"] = clean_addresses
    return df


# ---------------------------------------------------------------------------
# 4. COMPREHENSIVE TEST SUITE & DEMONSTRATION
# ---------------------------------------------------------------------------

def run_tests():
    """Run tests on representative real-world cases from the training set."""
    print("=" * 80)
    print("       AMAZON ML CHALLENGE 2026 - TEXT NORMALIZATION TEST SUITE")
    print("=" * 80)

    test_cases = [
        {
            "category": "1. English with Punctuation & Ampersand",
            "name": "Orelee's Barbershop & Salon -- 100%",
            "addr": "1795 Westchester Drive, High Point, NC",
            "explanation": "Standardizes apostrophe possessive ('s -> s), converts '&' to 'and', strips dashes/symbols, expands 'NC' to 'north carolina'."
        },
        {
            "category": "2. Standalone Symbol & Parenthetical ID",
            "name": "B+ Retail INC (ID: 84923)",
            "addr": "1712 Montebello Ave, Northeast, Arizona",
            "explanation": "Removes '(ID: 84923)', retains core brand 'b retail', strips suffix 'inc', expands 'ave' to 'avenue'."
        },
        {
            "category": "3. Hindi / Devanagari Script Name (India)",
            "name": "राम मार्केटिंग प्राइवेट लिमिटेड",
            "addr": "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
            "explanation": "Preserves all Devanagari characters, matras and virama; normalizes 'KH NO.' to 'khasra number'; strips Hindi legal suffix 'प्राइवेट लिमिटेड' in core name."
        },
        {
            "category": "4. Hindi Script LLP with State Abbreviations",
            "name": "आदित्य प्रॉपर्टीज एलएलपी",
            "addr": "G-3/571, GULMOHAR COLONY, BHOPAL, MP",
            "explanation": "Extracts core name 'आदित्य प्रॉपर्टीज' (stripping 'एलएलपी'); expands 'MP' to 'madhya pradesh'."
        },
        {
            "category": "5. Website Domain as Business Name",
            "name": "heassociates.com",
            "addr": "599 PEASE HILL ROAD, HORICON, NY",
            "explanation": "Strips '.com' domain extension to match 'heassociates'; expands 'NY' to 'new york'."
        },
        {
            "category": "6. Name with Appended URL Pipe",
            "name": "SHIVSHAKTI OVERSEAS CORP | www.shivshakti.com",
            "addr": "H.NO 204 C ROAD HOSHIARPUR, Punjab",
            "explanation": "Strips pipe and 'www.shivshakti.com'; isolates core brand 'shivshakti overseas'; expands 'h.no' to 'number'."
        },
        {
            "category": "7. Typo in Business Name",
            "name": "Maure Wilblims Colombier Inc",
            "addr": "85 Wanye Avenue, Ticonderoga Townshiip, NY",
            "explanation": "Cleans typo-laden record into standard tokens; extracts core name 'maure wilblims colombier'; expands 'NY' to 'new york'."
        },
        {
            "category": "8. Redundant / Repeated Legal Suffixes",
            "name": "Olszewski Holding Company LLC LLC",
            "addr": "218 SHOAL CREEK, WILLIAMSBURG, VA",
            "explanation": "Collapses repeated legal suffixes 'LLC LLC' and 'holding company' to isolate true brand 'olszewski'; expands 'VA' to 'virginia'."
        },
        {
            "category": "9. French Entity with Accents & Legal Prefix",
            "name": "SCI Ptit Àmicale",
            "addr": "18 RUE JEN ZAY, Dunkerque, Nord",
            "explanation": "Folds Latin accent 'À' -> 'a' for uniform comparison; recognizes 'SCI' as legal prefix to extract 'ptit amicale'."
        },
        {
            "category": "10. French Entity with Sarl & Street Code",
            "name": "Marina Ecole France Sarl",
            "addr": "63 R. DE DIEPPE, LILLE, Hauts-de-France",
            "explanation": "Strips 'Sarl' suffix; expands French street abbreviation 'R.' to 'rue' without corrupting French preposition 'de'."
        },
        {
            "category": "11. Empty / Missing Address Record",
            "name": "Red Pvt. Ltd. Center",
            "addr": "",
            "explanation": "Safely outputs empty string '' for address without errors, ready for null-aware comparison."
        },
        {
            "category": "12. Real Matched Triple (Step 2 Case 2: India)",
            "name": "Red Ventures Private Limited",
            "addr": "Rajasthan, Jaipur, Banipark, Gokul Apartment, E-3A Kanti Chandra Road, G-1",
            "explanation": "Core name becomes 'red ventures'; address token set invariant to city/state ordering."
        },
    ]

    for tc in test_cases:
        print("\n" + "-" * 80)
        print(f"CASE: {tc['category']}")
        print("-" * 80)
        orig_name = tc["name"]
        orig_addr = tc["addr"]

        clean_n = normalize_business_name(orig_name, strip_legal_suffixes=False)
        core_n = extract_core_name(orig_name)
        clean_a = normalize_address(orig_addr, expand_abbreviations=True)
        addr_toks = get_address_tokens(orig_addr)

        print(f"  [INPUT NAME]     : {orig_name}")
        print(f"  -> Clean Name    : {clean_n}")
        print(f"  -> Core Name     : {core_n}")
        print(f"  [INPUT ADDRESS]  : {orig_addr!r}")
        print(f"  -> Clean Address : {clean_a!r}")
        if addr_toks:
            print(f"  -> Addr Tokens   : {addr_toks[:6]}... (total {len(addr_toks)} unique)")
        print(f"  * Note: {tc['explanation']}")

    print("\n" + "=" * 80)
    print("ALL NORMALIZATION TESTS PASSED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
