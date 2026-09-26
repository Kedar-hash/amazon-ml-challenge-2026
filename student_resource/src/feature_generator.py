#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Step 5: Similarity Feature Generation Module

This module generates memory-efficient, multi-faceted similarity features for
candidate pairs (Source 1 <-> Source 2 / Source 3) identified during blocking.

Key Feature Categories:
1. Business Name Similarities:
   - Exact match indicators (clean name, core name, compact spaceless name)
   - Token-level metrics: Token Jaccard, Token Overlap, Token Containment
   - Character & Edit-distance metrics: Levenshtein similarity, SequenceMatcher ratio
   - Character 3-gram Cosine similarity (sublinear term frequency)
   - Length difference, length ratio, common prefix ratio
2. Business Address Similarities:
   - Explicit missing address indicator flags (s1 missing, cand missing, both present)
   - Address Token Jaccard, Token Overlap, and Token Containment (order-invariant)
   - Street / Building number exact match indicator (e.g. 1795 == 1795, 0017560 == 17560)
   - Address character SequenceMatcher ratio
3. Cross-Field & Script Metadata:
   - Country match indicator
   - Candidate source indicators (is_s2, is_s3)
   - Script indicators (is_cross_script flag for Devanagari vs Latin)

Usage:
    from src.feature_generator import compute_pair_features, batch_generate_features
    # Or run directly for demonstration and verification on real candidate pairs:
    python src/feature_generator.py
"""

import os
import sys
import math
import difflib
from collections import Counter
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure student_resource root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from src.text_normalizer import (
    normalize_business_name,
    extract_core_name,
    normalize_address,
    get_address_tokens,
)
from src.blocking import extract_addr_numbers


# ---------------------------------------------------------------------------
# 1. CORE SIMILARITY ALGORITHMS
# ---------------------------------------------------------------------------

def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein similarity: 1.0 - (edit_distance / max_len).
    Optimized 2-row dynamic programming algorithm in pure Python.
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)

    for i, c1 in enumerate(s1, 1):
        curr[0] = i
        for j, c2 in enumerate(s2, 1):
            cost = 0 if c1 == c2 else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev

    dist = prev[len2]
    max_len = max(len1, len2)
    return 1.0 - (dist / max_len) if max_len > 0 else 0.0


def char_ngram_cosine(s1: str, s2: str, n: int = 3) -> float:
    """
    Compute character n-gram cosine similarity directly in O(|s1| + |s2|).
    Requires zero pre-indexing and operates seamlessly across all Unicode scripts.
    """
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 < n or len2 < n:
        return 1.0 if s1 == s2 else 0.0

    c1 = Counter(s1[i:i + n] for i in range(len1 - n + 1))
    c2 = Counter(s2[i:i + n] for i in range(len2 - n + 1))

    dot = sum(v * c2.get(k, 0) for k, v in c1.items())
    if dot == 0:
        return 0.0

    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    return dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0


def token_jaccard_and_containment(t1_set: set, t2_set: set) -> tuple:
    """
    Compute (jaccard_similarity, overlap_count, containment_ratio) for two token sets.
    """
    if not t1_set or not t2_set:
        return 0.0, 0.0, 0.0

    inter = len(t1_set.intersection(t2_set))
    union = len(t1_set.union(t2_set))
    jaccard = inter / union if union > 0 else 0.0
    overlap = float(inter)
    min_len = min(len(t1_set), len(t2_set))
    containment = inter / min_len if min_len > 0 else 0.0

    return jaccard, overlap, containment


def is_devanagari(text: str) -> bool:
    """Check if a string contains Devanagari codepoints (U+0900 to U+097F)."""
    return any(0x0900 <= ord(ch) <= 0x097F for ch in text)


# ---------------------------------------------------------------------------
# 2. FEATURE EXTRACTION PIPELINE
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Business Name Features
    "name_exact_clean",
    "name_exact_core",
    "name_compact_match",
    "name_token_jaccard",
    "name_token_overlap",
    "name_token_containment",
    "name_levenshtein_sim",
    "name_seq_ratio",
    "name_char_cosine",
    "name_len_diff",
    "name_len_ratio",
    "name_prefix_ratio",
    # Business Address Features
    "addr_both_present",
    "addr_missing_cand",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_token_containment",
    "addr_seq_ratio",
    "addr_number_match",
    # Metadata & Cross-Field Features
    "country_match",
    "cand_is_s2",
    "cand_is_s3",
    "is_cross_script",
]


def compute_pair_features(s1_rec: dict, cand_rec: dict) -> dict:
    """
    Compute rich similarity feature dictionary for a single S1 <-> Candidate pair.
    Handles missing addresses and multilingual/Unicode text safely.
    """
    s1_name_clean = s1_rec.get("clean_name", "")
    s1_name_core = s1_rec.get("core_name", "")
    cand_name_clean = cand_rec.get("clean_name", "")
    cand_name_core = cand_rec.get("core_name", "")

    s1_addr_clean = s1_rec.get("clean_addr", "")
    cand_addr_clean = cand_rec.get("clean_addr", "")
    s1_addr_toks = set(s1_rec.get("addr_tokens", []))
    cand_addr_toks = set(cand_rec.get("addr_tokens", []))

    feats = {}

    # --- 1. Business Name Features ---
    feats["name_exact_clean"] = 1.0 if (s1_name_clean == cand_name_clean and s1_name_clean) else 0.0
    feats["name_exact_core"] = 1.0 if (s1_name_core == cand_name_core and s1_name_core) else 0.0

    s1_compact = s1_name_clean.replace(" ", "")
    cand_compact = cand_name_clean.replace(" ", "")
    feats["name_compact_match"] = 1.0 if (s1_compact == cand_compact and s1_compact) else 0.0

    t1_set = set(s1_name_core.split())
    t2_set = set(cand_name_core.split())
    jaccard, overlap, containment = token_jaccard_and_containment(t1_set, t2_set)
    feats["name_token_jaccard"] = jaccard
    feats["name_token_overlap"] = overlap
    feats["name_token_containment"] = containment

    feats["name_levenshtein_sim"] = levenshtein_similarity(s1_name_core, cand_name_core)
    feats["name_seq_ratio"] = (
        difflib.SequenceMatcher(None, s1_name_core, cand_name_core).ratio()
        if (s1_name_core and cand_name_core) else 0.0
    )
    feats["name_char_cosine"] = char_ngram_cosine(s1_name_clean, cand_name_clean, n=3)

    len1, len2 = len(s1_name_core), len(cand_name_core)
    feats["name_len_diff"] = float(abs(len1 - len2))
    feats["name_len_ratio"] = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0

    # Common Prefix Ratio
    p_len = 0
    for c1, c2 in zip(s1_name_core, cand_name_core):
        if c1 == c2:
            p_len += 1
        else:
            break
    feats["name_prefix_ratio"] = p_len / max(len1, len2) if max(len1, len2) > 0 else 0.0

    # --- 2. Business Address Features ---
    has_s1_addr = bool(s1_addr_clean)
    has_cand_addr = bool(cand_addr_clean)
    feats["addr_both_present"] = 1.0 if (has_s1_addr and has_cand_addr) else 0.0
    feats["addr_missing_cand"] = 1.0 if not has_cand_addr else 0.0

    if has_s1_addr and has_cand_addr:
        a_jaccard, a_overlap, a_containment = token_jaccard_and_containment(s1_addr_toks, cand_addr_toks)
        feats["addr_token_jaccard"] = a_jaccard
        feats["addr_token_overlap"] = a_overlap
        feats["addr_token_containment"] = a_containment
        feats["addr_seq_ratio"] = difflib.SequenceMatcher(None, s1_addr_clean, cand_addr_clean).ratio()

        # Street / Building number match
        n1 = set(extract_addr_numbers(s1_rec.get("raw_addr", "")))
        n2 = set(extract_addr_numbers(cand_rec.get("raw_addr", "")))
        feats["addr_number_match"] = 1.0 if (n1 and n2 and n1.intersection(n2)) else 0.0
    else:
        feats["addr_token_jaccard"] = 0.0
        feats["addr_token_overlap"] = 0.0
        feats["addr_token_containment"] = 0.0
        feats["addr_seq_ratio"] = 0.0
        feats["addr_number_match"] = 0.0

    # --- 3. Cross-Field & Script Metadata ---
    feats["country_match"] = 1.0 if s1_rec.get("country", "").upper() == cand_rec.get("country", "").upper() else 0.0
    cand_id = cand_rec.get("id", "")
    feats["cand_is_s2"] = 1.0 if cand_id.startswith("S2-") else 0.0
    feats["cand_is_s3"] = 1.0 if cand_id.startswith("S3-") else 0.0

    # Cross-script indicator (e.g. English S1 vs Hindi S2/S3)
    s1_is_dev = is_devanagari(s1_rec.get("name", ""))
    cand_is_dev = is_devanagari(cand_rec.get("name", ""))
    feats["is_cross_script"] = 1.0 if (s1_is_dev != cand_is_dev) else 0.0

    return feats


def batch_generate_features(pairs_list: list, s1_dict: dict, cand_dict: dict, as_dataframe: bool = True):
    """
    Generate feature matrix for a list of (s1_id, candidate_id) tuples.
    Returns:
      If as_dataframe=True: pandas DataFrame with FEATURE_COLUMNS + ['s1_id', 'cand_id']
      If as_dataframe=False: (numpy.ndarray of shape (N, num_features), FEATURE_COLUMNS)
    """
    rows = []
    for s1_id, cand_id in pairs_list:
        s1_rec = s1_dict.get(s1_id)
        cand_rec = cand_dict.get(cand_id)
        if not s1_rec or not cand_rec:
            continue
        feat_dict = compute_pair_features(s1_rec, cand_rec)
        if as_dataframe:
            feat_dict["s1_id"] = s1_id
            feat_dict["cand_id"] = cand_id
        rows.append(feat_dict)

    if as_dataframe:
        cols = ["s1_id", "cand_id"] + FEATURE_COLUMNS
        return pd.DataFrame(rows, columns=cols)
    else:
        matrix = np.array([[r[col] for col in FEATURE_COLUMNS] for r in rows], dtype=np.float32)
        return matrix, FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# 3. VERIFICATION & REAL CANDIDATE PAIR DEMONSTRATION
# ---------------------------------------------------------------------------

def resolve_base_dir():
    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, "dataset", "train")):
        return cwd
    if os.path.isdir(os.path.join(cwd, "student_resource", "dataset", "train")):
        return os.path.join(cwd, "student_resource")
    return cwd


def load_real_demo_cases(base_dir: str):
    """
    Dynamically loads real records from train_source1.tsv, train_source2.tsv,
    and train_source3.tsv. No hardcoded records or synthetic IDs.
    """
    s1_path = os.path.join(base_dir, "dataset/train/train_source1.tsv")
    s2_path = os.path.join(base_dir, "dataset/train/train_source2.tsv")
    s3_path = os.path.join(base_dir, "dataset/train/train_source3.tsv")

    # 5 distinct real Source 1 entities across US and India
    target_s1_ids = [
        "S1-377745466",  # B+ Retail Inc (US)
        "S1-773889195",  # Prime Money (US)
        "S1-925783039",  # Orelee's Barbershop (US)
        "S1-755362802",  # Prabhav Business Center (India)
        "S1-851869949",  # Custom Wealth Services LLC (US)
    ]

    s1_records = {}
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[0] in target_s1_ids:
                s1_records[parts[0]] = {
                    "id": parts[0],
                    "name": parts[1],
                    "raw_addr": parts[2],
                    "country": parts[3],
                    "clean_name": normalize_business_name(parts[1]),
                    "core_name": extract_core_name(parts[1]),
                    "clean_addr": normalize_address(parts[2]),
                    "addr_tokens": get_address_tokens(parts[2]),
                }
            if len(s1_records) == len(target_s1_ids):
                break

    # Real candidate IDs from S2 and S3 (including 4 true matches and 1 real distractor negative)
    target_cand_ids = {
        "S3-402918963",  # True match for S1-377745466 (Accent & Legal suffix)
        "S2-970528089",  # True match for S1-773889195 (Handle format & zero-padded address)
        "S2-517291332",  # True match for S1-925783039 (Typo/accent in name)
        "S2-587230276",  # True match for S1-755362802 (India entity variation)
        "S2-764573417",  # Real candidate from S2 for S1-851869949 (Distractor Negative Pair)
    }

    cand_records = {}
    for path in [s2_path, s3_path]:
        rem = {
            eid for eid in target_cand_ids
            if (eid.startswith("S2-") if "source2" in path else eid.startswith("S3-"))
        }
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if parts and parts[0] in rem:
                    cand_records[parts[0]] = {
                        "id": parts[0],
                        "name": parts[1],
                        "raw_addr": parts[2],
                        "country": parts[3],
                        "clean_name": normalize_business_name(parts[1]),
                        "core_name": extract_core_name(parts[1]),
                        "clean_addr": normalize_address(parts[2]),
                        "addr_tokens": get_address_tokens(parts[2]),
                    }
                    rem.remove(parts[0])
                    if not rem:
                        break

    cases = [
        {
            "title": "CASE 1: Real Positive Pair (Accent & Corporation Legal Form)",
            "s1": s1_records["S1-377745466"],
            "cand": cand_records["S3-402918963"],
            "label": "TRUE MATCH",
            "notes": "Real S3 record has accent 'Rétail' and 'Incorporated' vs 'Inc'. Clean address matches perfectly.",
        },
        {
            "title": "CASE 2: Real Positive Pair (Handle Format & Zero-Padded Address)",
            "s1": s1_records["S1-773889195"],
            "cand": cand_records["S2-970528089"],
            "label": "TRUE MATCH",
            "notes": "Real S2 record has handle '@primemoney' and zero-padded address number '0017560' vs '17560'.",
        },
        {
            "title": "CASE 3: Real Positive Pair (Typo / Accent Variation in Name)",
            "s1": s1_records["S1-925783039"],
            "cand": cand_records["S2-517291332"],
            "label": "TRUE MATCH",
            "notes": "Real S2 record has accent variation 'Bârbershop' and uppercase city formatting.",
        },
        {
            "title": "CASE 4: Real Positive Pair (India Entity Address Reorganization)",
            "s1": s1_records["S1-755362802"],
            "cand": cand_records["S2-587230276"],
            "label": "TRUE MATCH",
            "notes": "Real S2 Indian record with reordered address tokens and house number formatting.",
        },
        {
            "title": "CASE 5: Real Negative Candidate Pair (Distractor from Source 2)",
            "s1": s1_records["S1-851869949"],
            "cand": cand_records["S2-764573417"],
            "label": "NEGATIVE (FALSE CANDIDATE)",
            "notes": "Real candidate records from dataset. Different business entities in the same country.",
        },
    ]

    return cases


def run_demonstration():
    print("=" * 85)
    print("      AMAZON ML CHALLENGE 2026 - STEP 5: FEATURE GENERATION DEMO")
    print("=" * 85)

    base_dir = resolve_base_dir()
    print(f"Loading real candidate pairs dynamically from: {base_dir}")
    cases = load_real_demo_cases(base_dir)

    for c in cases:
        print("\n" + "#" * 85)
        print(f" {c['title']} [{c['label']}]")
        print("#" * 85)
        s1 = c["s1"]
        cand = c["cand"]
        feats = compute_pair_features(s1, cand)

        print(f"  Source 1 : [{s1['id']}] {s1['name']} | {s1['raw_addr']} ({s1['country']})")
        print(f"  Candidate: [{cand['id']}] {cand['name']} | {cand['raw_addr']} ({cand['country']})")
        print(f"  Notes    : {c['notes']}")
        print("\n  Computed Feature Vector (23 dimensions):")
        print("  " + "-" * 75)

        # Print in clean 2-column format
        items = list(feats.items())
        half = (len(items) + 1) // 2
        for i in range(half):
            k1, v1 = items[i]
            col1 = f"{k1:<24}: {v1:>7.3f}"
            if i + half < len(items):
                k2, v2 = items[i + half]
                col2 = f"{k2:<24}: {v2:>7.3f}"
                print(f"    {col1}    |    {col2}")
            else:
                print(f"    {col1}")

    print("\n" + "=" * 85)
    print("ALL FEATURE GENERATION DEMONSTRATIONS COMPLETED SUCCESSFULLY.")
    print("=" * 85)


if __name__ == "__main__":
    run_demonstration()
