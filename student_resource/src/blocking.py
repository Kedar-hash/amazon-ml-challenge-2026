#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Step 4: Candidate Generation & Blocking Pipeline

This module implements a memory-efficient, high-recall blocking engine for
matching Source 1 entities against Source 2 and Source 3 candidate records.

Key Innovations in Blocking Design:
1. Multi-Index Disjunctive Keys (Union of complementary keys):
   - Exact Core Name Key: Captures normalized business names
   - Compact Merged Name Key: Solves domain/handle variations (e.g. '@primemoney' vs 'Prime Money',
     'moorebitwise.com' vs 'Moore Bitwise')
   - Informative Name Token Key: Captures names with typos or dropped non-essential words
   - Address Number + Landmark Token Key: Strips leading zeros (e.g. '0017560' -> '17560',
     '0337' -> '337') and pairs with street/city to match even when names differ drastically
   - Distinctive Address Bigram Key: Bridges script differences (e.g. English vs Devanagari)
     when addresses share geographic landmark words ('south delhi', 'osarambagh hyderabad')
2. Country Hard Partition: Partitions by country code (US, India, France) to eliminate cross-country search
3. Scalable Inverted Index: Uses light-weight dictionary postings with configurable block-size caps
   to avoid combinatorial explosion on frequent generic terms.

Usage:
    from src.blocking import InvertedIndexBlocking, generate_blocking_keys
    # Or run directly for evaluation on a representative dataset sample:
    python src/blocking.py
"""

import os
import sys
import time
import re
from collections import defaultdict

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

# Generic corporate/industry terms that produce high-cardinality, uninformative blocks
COMMON_BLOCKING_STOPWORDS = {
    "services", "consultants", "enterprises", "solutions", "international",
    "associates", "technologies", "technology", "management", "industries",
    "global", "retail", "group", "india", "national", "holdings", "trading",
    "marketing", "products", "store", "shop", "center", "centre", "corp",
    "company", "commercial", "enterprise", "system", "systems", "general",
    "road", "street", "avenue", "drive", "lane", "court", "boulevard",
    "building", "floor", "suite", "apartment", "number", "near", "opposite",
}

_RE_DIGITS = re.compile(r"\b0*([1-9]\d{1,5})\b")


def extract_addr_numbers(raw_address: str) -> list:
    """
    Extract house/building/unit numbers and strip leading zeros
    (e.g., '0017560' -> '17560', '0337' -> '337', '4024B' -> '4024').
    """
    if not raw_address:
        return []
    nums = set()
    for m in _RE_DIGITS.finditer(raw_address):
        nums.add(m.group(1))
    return sorted(nums)


def generate_blocking_keys(record: dict) -> list:
    """
    Generate a diverse set of blocking keys for an entity record.
    record expected keys: 'country', 'core_name', 'raw_addr', 'addr_tokens'
    """
    c = record.get("country", "").strip().upper()
    core = record.get("core_name", "").strip()
    raw_addr = record.get("raw_addr", "")
    addr_toks = record.get("addr_tokens", [])

    keys = []

    # 1. Exact Core Name Key
    if len(core) >= 3:
        keys.append(f"{c}|core|{core}")

        # 2. Compact Merged Name Key (removes spaces, e.g. 'moorebitwise', 'primemoney')
        compact = core.replace(" ", "")
        if len(compact) >= 5 and compact != core:
            keys.append(f"{c}|compact|{compact}")

    # 3. Informative Individual Name Words (len >= 3, non-stopwords)
    words = [w for w in core.split() if len(w) >= 3 and w not in COMMON_BLOCKING_STOPWORDS]
    for w in words:
        keys.append(f"{c}|name_word|{w}")

    # 4. Normalized Address Number + Text Token Key
    nums = extract_addr_numbers(raw_addr)
    text_toks = [
        t for t in addr_toks
        if not t.isdigit() and len(t) >= 4 and t not in COMMON_BLOCKING_STOPWORDS
    ]

    if nums and text_toks:
        for n in nums[:2]:
            for t in text_toks[:3]:
                keys.append(f"{c}|num_tok|{n}_{t}")

    # 5. Distinctive Address Bigram Key (e.g. 'south_delhi', 'osarambagh_hyderabad')
    # Connects entities where the business name is transliterated across scripts
    if len(text_toks) >= 2:
        for i in range(min(2, len(text_toks) - 1)):
            keys.append(f"{c}|addr_bi|{text_toks[i]}_{text_toks[i+1]}")

    return keys


class InvertedIndexBlocking:
    """
    Memory-efficient Inverted Index for candidate generation.
    Indexes target records (Source 2 & Source 3) by blocking keys
    and retrieves candidate sets for Source 1 entities in milliseconds.
    """

    def __init__(self, max_block_size: int = 250):
        self.max_block_size = max_block_size
        self.index = defaultdict(list)
        self.total_indexed_records = 0

    def add_record(self, entity_id: str, record: dict):
        """Index a single candidate record."""
        keys = generate_blocking_keys(record)
        for k in keys:
            self.index[k].append(entity_id)
        self.total_indexed_records += 1

    def build_from_dict(self, records_dict: dict):
        """Bulk index records from a dictionary of {entity_id: record}."""
        t0 = time.time()
        for eid, rec in records_dict.items():
            self.add_record(eid, rec)
        elapsed = time.time() - t0
        print(f"Indexed {len(records_dict):,} candidate records into {len(self.index):,} keys ({elapsed:.2f}s).")

    def get_candidates(self, s1_record: dict) -> set:
        """Retrieve candidate S2/S3 entity IDs for a Source 1 record."""
        keys = generate_blocking_keys(s1_record)
        candidates = set()
        for k in keys:
            postings = self.index.get(k, [])
            # Prune oversized blocks to prevent low-precision explosions
            if len(postings) <= self.max_block_size:
                candidates.update(postings)
        return candidates

    def evaluate(self, s1_records: dict, ground_truth: dict) -> dict:
        """
        Evaluate blocking recall and candidate set metrics against ground truth.
        """
        total_true_pairs = 0
        captured_true_pairs = 0
        candidate_counts = []

        for s1_id, s1_rec in s1_records.items():
            true_matches = ground_truth.get(s1_id, set())
            total_true_pairs += len(true_matches)

            cands = self.get_candidates(s1_rec)
            candidate_counts.append(len(cands))

            if true_matches:
                captured_true_pairs += len(true_matches.intersection(cands))

        recall = (captured_true_pairs / total_true_pairs * 100) if total_true_pairs > 0 else 0.0
        avg_candidates = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0
        sorted_counts = sorted(candidate_counts)
        median_candidates = sorted_counts[len(sorted_counts) // 2] if sorted_counts else 0

        total_candidate_pairs = sum(candidate_counts)
        total_possible_pairs = len(s1_records) * self.total_indexed_records
        reduction_ratio = (
            (1.0 - (total_candidate_pairs / total_possible_pairs)) * 100
            if total_possible_pairs > 0 else 0.0
        )

        return {
            "total_s1_evaluated": len(s1_records),
            "total_indexed_pool": self.total_indexed_records,
            "total_true_pairs": total_true_pairs,
            "captured_true_pairs": captured_true_pairs,
            "blocking_recall": recall,
            "avg_candidates_per_s1": avg_candidates,
            "median_candidates_per_s1": median_candidates,
            "min_candidates": min(candidate_counts) if candidate_counts else 0,
            "max_candidates": max(candidate_counts) if candidate_counts else 0,
            "total_candidate_pairs": total_candidate_pairs,
            "total_possible_pairs": total_possible_pairs,
            "reduction_ratio": reduction_ratio,
        }


# ---------------------------------------------------------------------------
# EVALUATION & DEMONSTRATION HARNESS
# ---------------------------------------------------------------------------

def resolve_base_dir():
    cwd = os.getcwd()
    candidates = [
        cwd,
        os.path.join(cwd, "student_resource"),
        os.path.dirname(os.path.abspath(cwd)),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "dataset", "train")) or \
           os.path.isdir(os.path.join(cand, "dataset", "dataset", "train")):
            return cand
    if os.path.isdir(os.path.join(cwd, "dataset", "train")):
        return cwd
    if os.path.isdir(os.path.join(cwd, "student_resource", "dataset", "train")):
        return os.path.join(cwd, "student_resource")
    return cwd


def resolve_dataset_file(base_dir: str, rel_path: str) -> str:
    """Resolve file path checking both standard dataset/ and nested dataset/dataset/ locations."""
    candidates = [
        os.path.join(base_dir, rel_path),
        os.path.join(base_dir, "dataset", rel_path),
        os.path.join(os.path.dirname(os.path.abspath(base_dir)), rel_path),
        os.path.join(os.path.dirname(os.path.abspath(base_dir)), "dataset", rel_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.join(base_dir, rel_path)


def load_evaluation_data(base_dir, num_s1=1000, num_distractors=50000):
    """
    Load a statistically representative sample for evaluating blocking:
    - 1,000 Source 1 entities
    - All true S2/S3 matches from ground truth
    - 50,000 real negative/distractor records from Source 2 and Source 3
    """
    print(f"Loading {num_s1:,} Source 1 records and associated ground-truth matches...")

    s1_path = resolve_dataset_file(base_dir, "dataset/train/train_source1.tsv")
    gt_path = resolve_dataset_file(base_dir, "dataset/train/train_ground_truth.tsv")
    s2_path = resolve_dataset_file(base_dir, "dataset/train/train_source2.tsv")
    s3_path = resolve_dataset_file(base_dir, "dataset/train/train_source3.tsv")

    # 1. Load S1 Sample
    s1_records = {}
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for _ in range(num_s1):
            line = f.readline()
            if not line:
                break
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                eid = parts[0]
                s1_records[eid] = {
                    "id": eid,
                    "name": parts[1],
                    "raw_addr": parts[2],
                    "country": parts[3],
                    "core_name": extract_core_name(parts[1]),
                    "addr_tokens": get_address_tokens(parts[2]),
                }

    # 2. Load Ground Truth for S1 sample
    gt_matches = {}
    target_matches = set()
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in s1_records:
                if len(parts) > 1 and parts[1].strip():
                    ms = [x.strip() for x in parts[1].split(",") if x.strip()]
                    gt_matches[parts[0]] = set(ms)
                    target_matches.update(ms)
                else:
                    gt_matches[parts[0]] = set()

    # 3. Load Candidate Pool (True matches + 50,000 Distractors)
    s2_s3_pool = {}
    distractors_per_file = num_distractors // 2

    for path in [s2_path, s3_path]:
        distractor_count = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4:
                    eid = parts[0]
                    is_true_target = eid in target_matches
                    if is_true_target or distractor_count < distractors_per_file:
                        s2_s3_pool[eid] = {
                            "id": eid,
                            "name": parts[1],
                            "raw_addr": parts[2],
                            "country": parts[3],
                            "core_name": extract_core_name(parts[1]),
                            "addr_tokens": get_address_tokens(parts[2]),
                        }
                        if not is_true_target:
                            distractor_count += 1

    return s1_records, s2_s3_pool, gt_matches


def run_pipeline_demo():
    print("=" * 80)
    print("        AMAZON ML CHALLENGE 2026 - STEP 4: BLOCKING EVALUATION")
    print("=" * 80)

    base_dir = resolve_base_dir()
    s1_sample, candidate_pool, gt_matches = load_evaluation_data(
        base_dir, num_s1=1000, num_distractors=50000
    )

    indexer = InvertedIndexBlocking(max_block_size=200)
    indexer.build_from_dict(candidate_pool)

    print("\nRunning blocking evaluation against Ground Truth...")
    t0 = time.time()
    metrics = indexer.evaluate(s1_sample, gt_matches)
    elapsed = time.time() - t0

    print(f"Evaluation completed in {elapsed:.2f} seconds.")
    print("-" * 80)
    print(f"  Source 1 Entities Evaluated : {metrics['total_s1_evaluated']:,}")
    print(f"  Candidate Pool (S2+S3) Size  : {metrics['total_indexed_pool']:,}")
    print(f"  Total True Match Pairs       : {metrics['total_true_pairs']:,}")
    print(f"  Captured True Match Pairs    : {metrics['captured_true_pairs']:,}")
    print(f"  --> BLOCKING RECALL CEILING  : {metrics['blocking_recall']:.2f}%")
    print("-" * 80)
    print(f"  Average Candidates per S1    : {metrics['avg_candidates_per_s1']:.2f}")
    print(f"  Median Candidates per S1     : {metrics['median_candidates_per_s1']}")
    print(f"  Min / Max Candidates per S1  : {metrics['min_candidates']} / {metrics['max_candidates']}")
    print(f"  Total Candidate Pairs        : {metrics['total_candidate_pairs']:,}")
    print(f"  Total Brute-Force Pairs      : {metrics['total_possible_pairs']:,}")
    print(f"  --> SEARCH SPACE REDUCTION   : {metrics['reduction_ratio']:.4f}%")
    print("=" * 80)

    # Display 3 qualitative inspection examples
    print("\nSAMPLE QUALITATIVE BLOCKING RESULTS:")
    print("=" * 80)
    sample_ids = list(s1_sample.keys())[:3]
    for s1_id in sample_ids:
        rec = s1_sample[s1_id]
        true_ms = gt_matches.get(s1_id, set())
        cands = indexer.get_candidates(rec)
        captured = true_ms.intersection(cands)
        print(f"\nS1 Entity: {s1_id}")
        print(f"  Name    : {rec['name']}")
        print(f"  Address : {rec['raw_addr']} ({rec['country']})")
        print(f"  Total Candidates Generated: {len(cands)}")
        print(f"  True Matches in GT        : {len(true_ms)} -> {true_ms}")
        print(f"  Matches Captured          : {len(captured)} / {len(true_ms)} {'[RECALL 100%]' if len(captured) == len(true_ms) else '[MISSED]'}")


if __name__ == "__main__":
    run_pipeline_demo()
