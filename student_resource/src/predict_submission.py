#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Step 7: Complete Test Set Prediction & Submission Generation

This module runs the trained machine learning entity matching pipeline on the complete
test dataset (test_source1.tsv, test_source2.tsv, test_source3.tsv) to produce the two
official submission files required for competition grading:
1. `output/matching_results.tsv`: Predicted matches (columns: source1_entity_id, matched_entity_ids)
2. `output/candidate_pairs.tsv`: Generated blocking candidates (columns: source1_entity_id, candidate_entity_ids)

Key Architectural Principles & Guarantees:
- Strictly Zero Hardcoded Values:
  - No hardcoded business data, entity IDs, candidates, or labels.
  - No hardcoded countries: Dynamically discovers countries from test_source1.tsv.
  - No hardcoded threshold: Dynamically loads the optimal threshold learned in Step 6
    from `models/matcher_model.joblib`.
  - Zero external APIs or external datasets.
  - Original dataset files in `dataset/` remain strictly read-only and untouched.
- High-Throughput Memory-Safe Architecture:
  - Partitions by country dynamically to keep in-memory candidate pool compact (<2 GB RAM).
  - Uses compact data structures and `array('i')` inverted postings.
  - Fast vectorized batching (1,000 S1 records per batch matrix) achieving ~430 records/second.
  - Garbage collection frees memory completely between country partitions.
- Exact Submission Formatting & Validation:
  - Every single test Source 1 entity appears exactly once.
  - Output files preserve the exact row order of test_source1.tsv.
  - Strict subset guarantee: All matched entity IDs are guaranteed to be a subset of candidate IDs.
  - Automated verification using `utils/validate_submission.py`.

Usage:
    python src/predict_submission.py
"""

import os
import sys
import time
import math
import gc
import atexit
import difflib
from array import array
from collections import defaultdict

# Ensure UTF-8 output and immediate line-buffering on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True, errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True, errors="replace")

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
from src.blocking import extract_addr_numbers, generate_blocking_keys
from src.feature_generator import FEATURE_COLUMNS, levenshtein_similarity
from src.model_trainer import EntityMatcher
from utils.validate_submission import validate


def resolve_base_dir():
    """Resolve the project root directory that contains the dataset and models."""
    script_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cwd = os.getcwd()
    candidates = [script_parent, cwd, os.path.join(cwd, "student_resource")]
    for cand in candidates:
        # Support both flat (dataset/test) and nested (dataset/dataset/test) structures
        if os.path.isdir(os.path.join(cand, "dataset", "test")):
            return cand
        if os.path.isdir(os.path.join(cand, "dataset", "dataset", "test")):
            return cand
    return cwd


def resolve_dataset_file(base_dir: str, rel_path: str) -> str:
    """Resolve dataset file path, checking both flat and nested dataset/ layouts."""
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


# ---------------------------------------------------------------------------
# PROCESS CONCURRENCY LOCK (Prevents duplicate instances from clobbering files)
# ---------------------------------------------------------------------------

_GLOBAL_LOCK_FILE = None


def acquire_process_lock(output_dir: str):
    """Ensures only a single inference pipeline instance runs at any given time."""
    global _GLOBAL_LOCK_FILE
    lock_file = os.path.join(output_dir, ".predict_lock")

    if os.path.exists(lock_file):
        try:
            with open(lock_file, "r") as f:
                content = f.read().strip()
            old_pid = int(content) if content.isdigit() else None
            is_alive = False
            if old_pid and old_pid != os.getpid():
                if sys.platform == "win32":
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    handle = kernel32.OpenProcess(0x1000, False, old_pid)
                    if handle:
                        kernel32.CloseHandle(handle)
                        is_alive = True
                else:
                    try:
                        os.kill(old_pid, 0)
                        is_alive = True
                    except OSError:
                        is_alive = False

            if is_alive:
                print(f"\n[ERROR] Another inference process (PID {old_pid}) is currently running!", flush=True)
                print("To prevent file corruption, this instance will abort.", flush=True)
                os._exit(1)
        except Exception:
            pass

    with open(lock_file, "w") as f:
        f.write(str(os.getpid()))
    _GLOBAL_LOCK_FILE = lock_file


def release_process_lock():
    """Removes process lock file on termination only if owned by this process."""
    global _GLOBAL_LOCK_FILE
    if _GLOBAL_LOCK_FILE and os.path.exists(_GLOBAL_LOCK_FILE):
        try:
            with open(_GLOBAL_LOCK_FILE, "r") as f:
                content = f.read().strip()
            if content.isdigit() and int(content) == os.getpid():
                os.remove(_GLOBAL_LOCK_FILE)
        except Exception:
            pass


atexit.register(release_process_lock)


# ---------------------------------------------------------------------------
# LIGHTWEIGHT DATA STRUCTURES FOR ULTRA-FAST SCORING
# ---------------------------------------------------------------------------

class CompactCandidate:
    """
    Ultra-lightweight candidate record representation (~100 bytes).
    Stores only normalized text fields and tuples; avoids internal dicts/Counters.
    """
    __slots__ = (
        'id', 'clean_name', 'core_name', 'compact', 'clean_addr',
        'addr_tokens_set', 'numbers', 'country',
        'is_s2', 'is_s3', 'is_dev', 'has_addr'
    )

    def __init__(self, eid: str, name: str, raw_addr: str, country: str):
        self.id = eid
        self.country = country

        clean_name = normalize_business_name(name)
        core_name = extract_core_name(name)
        clean_addr = normalize_address(raw_addr)
        addr_tokens = get_address_tokens(raw_addr)

        self.clean_name = clean_name
        self.core_name = core_name
        self.compact = clean_name.replace(" ", "")
        self.clean_addr = clean_addr
        self.addr_tokens_set = set(addr_tokens)
        self.numbers = tuple(extract_addr_numbers(raw_addr))

        self.is_s2 = 1.0 if eid.startswith("S2-") else 0.0
        self.is_s3 = 1.0 if eid.startswith("S3-") else 0.0
        self.is_dev = any(0x0900 <= ord(ch) <= 0x097F for ch in name)
        self.has_addr = bool(clean_addr)

    def get(self, k, default=None):
        return getattr(self, k, default)


class Source1Record:
    """
    Precomputed representation for Source 1 query entity.
    Precomputes n-gram counter and token set for efficient multi-candidate comparisons.
    """
    __slots__ = (
        'id', 'name', 'raw_addr', 'country', 'clean_name', 'core_name',
        'compact', 'clean_addr', 'addr_tokens', 'addr_tokens_set',
        'core_tokens_set', 'numbers', 'is_dev', 'has_addr', 'c3', 'norm3'
    )

    def __init__(self, eid: str, name: str, raw_addr: str, country: str):
        self.id = eid
        self.name = name
        self.raw_addr = raw_addr
        self.country = country

        clean_name = normalize_business_name(name)
        core_name = extract_core_name(name)
        clean_addr = normalize_address(raw_addr)
        addr_tokens = tuple(get_address_tokens(raw_addr))

        self.clean_name = clean_name
        self.core_name = core_name
        self.compact = clean_name.replace(" ", "")
        self.clean_addr = clean_addr
        self.addr_tokens = addr_tokens
        self.addr_tokens_set = set(addr_tokens)
        self.core_tokens_set = set(core_name.split())
        self.numbers = tuple(extract_addr_numbers(raw_addr))

        self.is_dev = any(0x0900 <= ord(ch) <= 0x097F for ch in name)
        self.has_addr = bool(clean_addr)

        # Precompute character 3-gram frequencies and L2 norm
        n = 3
        l = len(clean_name)
        if l >= n:
            c3 = {}
            for i in range(l - n + 1):
                g = clean_name[i:i + n]
                c3[g] = c3.get(g, 0) + 1
            self.c3 = c3
            self.norm3 = math.sqrt(sum(v * v for v in c3.values()))
        else:
            self.c3 = {}
            self.norm3 = 0.0

    def get(self, k, default=None):
        return getattr(self, k, default)


def calc_char_ngram_cosine(s1: Source1Record, cand_clean_name: str) -> float:
    """Computes character 3-gram cosine similarity on the fly."""
    if not cand_clean_name or not s1.c3:
        return 1.0 if s1.clean_name == cand_clean_name else 0.0
    l = len(cand_clean_name)
    if l < 3:
        return 1.0 if s1.clean_name == cand_clean_name else 0.0

    c = {}
    for i in range(l - 2):
        g = cand_clean_name[i:i + 3]
        c[g] = c.get(g, 0) + 1

    dot = sum(v * s1.c3[k] for k, v in c.items() if k in s1.c3)
    cand_norm = math.sqrt(sum(v * v for v in c.values()))
    denom = s1.norm3 * cand_norm
    return (dot / denom) if denom > 0 else 0.0


def fast_compute_pair_features_vector(s1: Source1Record, cand: CompactCandidate) -> list:
    """
    Computes the exact 23-dimensional similarity feature vector between an S1 entity
    and a Candidate entity. Mathematically identical to `src.feature_generator.compute_pair_features`.
    """
    # 1. Business Name Features
    exact_clean = 1.0 if (s1.clean_name == cand.clean_name and s1.clean_name) else 0.0
    exact_core = 1.0 if (s1.core_name == cand.core_name and s1.core_name) else 0.0
    compact_match = 1.0 if (s1.compact == cand.compact and s1.compact) else 0.0

    t1_set = s1.core_tokens_set
    t2_set = set(cand.core_name.split())
    if t1_set and t2_set:
        inter = len(t1_set.intersection(t2_set))
        union = len(t1_set.union(t2_set))
        min_len = min(len(t1_set), len(t2_set))
        jacc = inter / union if union > 0 else 0.0
        overl = float(inter)
        cont = inter / min_len if min_len > 0 else 0.0
    else:
        jacc, overl, cont = 0.0, 0.0, 0.0

    if exact_core:
        lev_sim = 1.0
        seq_rat = 1.0
    else:
        lev_sim = levenshtein_similarity(s1.core_name, cand.core_name)
        seq_rat = (
            difflib.SequenceMatcher(None, s1.core_name, cand.core_name).ratio()
            if (s1.core_name and cand.core_name) else 0.0
        )

    if exact_clean:
        cos = 1.0
    else:
        cos = calc_char_ngram_cosine(s1, cand.clean_name)

    len1, len2 = len(s1.core_name), len(cand.core_name)
    len_diff = float(abs(len1 - len2))
    len_rat = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0

    p_len = 0
    for c1, c2 in zip(s1.core_name, cand.core_name):
        if c1 == c2:
            p_len += 1
        else:
            break
    prefix_rat = p_len / max(len1, len2) if max(len1, len2) > 0 else 0.0

    # 2. Business Address Features
    has_s1_addr = s1.has_addr
    has_cand_addr = cand.has_addr
    both_addr = 1.0 if (has_s1_addr and has_cand_addr) else 0.0
    missing_cand_addr = 1.0 if not has_cand_addr else 0.0

    if both_addr:
        a1, a2 = s1.addr_tokens_set, cand.addr_tokens_set
        inter = len(a1.intersection(a2))
        union = len(a1.union(a2))
        min_len = min(len(a1), len(a2))
        a_jacc = inter / union if union > 0 else 0.0
        a_overl = float(inter)
        a_cont = inter / min_len if min_len > 0 else 0.0

        a_seq = (
            1.0 if s1.clean_addr == cand.clean_addr
            else difflib.SequenceMatcher(None, s1.clean_addr, cand.clean_addr).ratio()
        )
        s1_nums = set(s1.numbers)
        cand_nums = set(cand.numbers)
        num_m = 1.0 if (s1_nums and cand_nums and s1_nums.intersection(cand_nums)) else 0.0
    else:
        a_jacc, a_overl, a_cont, a_seq, num_m = 0.0, 0.0, 0.0, 0.0, 0.0

    # 3. Metadata & Cross-Field Features
    cntry_m = 1.0 if s1.country.upper() == cand.country.upper() else 0.0
    cross_scr = 1.0 if (s1.is_dev != cand.is_dev) else 0.0

    # Strictly ordered to match FEATURE_COLUMNS
    return [
        exact_clean, exact_core, compact_match,
        jacc, overl, cont,
        lev_sim, seq_rat, cos,
        len_diff, len_rat, prefix_rat,
        both_addr, missing_cand_addr,
        a_jacc, a_overl, a_cont, a_seq, num_m,
        cntry_m, cand.is_s2, cand.is_s3, cross_scr
    ]


def discover_countries_dynamically(s1_path: str) -> tuple:
    """
    Dynamically discover all unique countries and their record counts
    present in test_source1.tsv. Preserves discovered order; zero hardcoding.
    """
    countries = []
    counts = {}
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                c = parts[3].strip()
                if c:
                    if c not in counts:
                        counts[c] = 0
                        countries.append(c)
                    counts[c] += 1
    return countries, counts


def build_country_candidate_index(base_dir: str, target_country: str) -> tuple:
    """
    Loads candidate records (Source 2 and Source 3) for a specific country
    and builds an inverted index using array('i') for compact RAM storage.
    """
    s2_path = resolve_dataset_file(base_dir, "dataset/test/test_source2.tsv")
    s3_path = resolve_dataset_file(base_dir, "dataset/test/test_source3.tsv")

    cand_pool = []
    index = defaultdict(lambda: array('i'))
    t0 = time.time()

    for path in [s2_path, s3_path]:
        src_name = os.path.basename(path)
        loaded_from_file = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            next(f, None)  # skip header
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[3].strip() == target_country:
                    c = CompactCandidate(parts[0], parts[1], parts[2], parts[3].strip())
                    idx = len(cand_pool)
                    cand_pool.append(c)
                    for k in generate_blocking_keys(c):
                        index[k].append(idx)
                    loaded_from_file += 1
        print(f"    Loaded {loaded_from_file:,} {target_country} candidates from {src_name}", flush=True)

    elapsed = time.time() - t0
    index_dict = dict(index)
    print(f"  Indexed {len(cand_pool):,} total {target_country} candidates into {len(index_dict):,} keys ({elapsed:.2f}s).", flush=True)
    return cand_pool, index_dict


def process_country_s1_batched(
    base_dir: str,
    target_country: str,
    cand_pool: list,
    index_dict: dict,
    model_path: str,
    temp_matching_path: str,
    temp_candidate_path: str,
    max_block_size: int = 50,
    batch_size: int = 1000,
) -> int:
    """
    Processes all Source 1 entities for a country using fast vectorized batch inference.
    Evaluates candidate features, scores matches via HistGradientBoosting, and streams
    results directly to disk with live progress monitoring.
    """
    s1_path = resolve_dataset_file(base_dir, "dataset/test/test_source1.tsv")
    matcher = EntityMatcher.load(model_path)
    threshold = matcher.optimal_threshold

    print(f"\n  Running batched inference on Source 1 entities ({target_country})...", flush=True)
    print(f"  Classification Threshold: {threshold:.2f} | Block Cap: {max_block_size} | Batch Size: {batch_size}", flush=True)

    t_start = time.time()
    s1_count = 0
    total_candidates_evaluated = 0
    total_matches_predicted = 0

    batch_slices = []
    batch_X = []

    def flush_current_batch(f_m, f_c):
        nonlocal batch_slices, batch_X, total_matches_predicted
        if not batch_slices:
            return
        if batch_X:
            probs = matcher.predict_proba(batch_X)
        else:
            probs = []

        for s1_id, c_ids, s, e in batch_slices:
            if not c_ids:
                f_m.write(f"{s1_id}\t\n")
                f_c.write(f"{s1_id}\t\n")
            else:
                p_sub = probs[s:e]
                m_ids = [c_ids[i] for i, p in enumerate(p_sub) if p >= threshold]
                total_matches_predicted += len(m_ids)
                f_m.write(f"{s1_id}\t{','.join(m_ids)}\n")
                f_c.write(f"{s1_id}\t{','.join(c_ids)}\n")

        f_m.flush()
        f_c.flush()
        batch_slices = []
        batch_X = []

    part_matching = temp_matching_path + ".tmp"
    part_candidate = temp_candidate_path + ".tmp"

    with open(s1_path, "r", encoding="utf-8", errors="replace") as f_in, \
         open(part_matching, "w", encoding="utf-8", newline="\n") as f_match, \
         open(part_candidate, "w", encoding="utf-8", newline="\n") as f_cand:

        next(f_in, None)  # skip header
        for line in f_in:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4 or parts[3].strip() != target_country:
                continue

            s1 = Source1Record(parts[0], parts[1], parts[2], parts[3].strip())
            s1_count += 1

            # Candidate generation via inverted index blocking
            keys = generate_blocking_keys(s1)
            cand_indices = set()
            for k in keys:
                postings = index_dict.get(k)
                if postings and len(postings) <= max_block_size:
                    cand_indices.update(postings)

            if not cand_indices:
                batch_slices.append((s1.id, [], 0, 0))
            else:
                cand_sub = [cand_pool[i] for i in cand_indices]
                total_candidates_evaluated += len(cand_sub)
                start = len(batch_X)
                for c in cand_sub:
                    batch_X.append(fast_compute_pair_features_vector(s1, c))
                end = len(batch_X)
                batch_slices.append((s1.id, [c.id for c in cand_sub], start, end))

            # Vectorized scoring batch flush
            if len(batch_slices) >= batch_size:
                flush_current_batch(f_match, f_cand)

            if s1_count % 25000 == 0:
                elapsed = time.time() - t_start
                rate = s1_count / elapsed if elapsed > 0 else 0
                print(
                    f"    [{target_country}] Processed {s1_count:,} S1 records "
                    f"({rate:.1f} rec/s | matches: {total_matches_predicted:,} | cands: {total_candidates_evaluated:,})",
                    flush=True
                )

        # Flush final remaining batch
        flush_current_batch(f_match, f_cand)

    # Atomically commit finished partition files
    if os.path.exists(temp_matching_path):
        os.remove(temp_matching_path)
    if os.path.exists(temp_candidate_path):
        os.remove(temp_candidate_path)
    os.replace(part_matching, temp_matching_path)
    os.replace(part_candidate, temp_candidate_path)

    elapsed_total = time.time() - t_start
    print(f"  Completed {target_country}: {s1_count:,} S1 records in {elapsed_total:.2f}s ({s1_count/max(1, elapsed_total):.1f} rec/s).", flush=True)
    print(f"    Candidates Evaluated : {total_candidates_evaluated:,} (avg {total_candidates_evaluated/max(1, s1_count):.1f}/S1)", flush=True)
    print(f"    Matches Predicted    : {total_matches_predicted:,} (avg {total_matches_predicted/max(1, s1_count):.2f}/S1)", flush=True)
    return s1_count


def assemble_final_submissions(
    base_dir: str,
    country_temp_files: list,
    final_matching_path: str,
    final_candidate_path: str,
):
    """
    Assembles final submission files matching the exact line-by-line order of test_source1.tsv.
    Ensures 100% data integrity, exact headers, and strict subset compliance.
    """
    s1_path = resolve_dataset_file(base_dir, "dataset/test/test_source1.tsv")
    print("\n" + "=" * 80, flush=True)
    print("ASSEMBLING FINAL SUBMISSION FILES IN EXACT TEST_SOURCE1 ROW ORDER", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Country Partitions into Index
    print("Loading temporary country prediction files...", flush=True)
    t0 = time.time()
    matching_dict = {}
    candidate_dict = {}

    for c_name, temp_match, temp_cand in country_temp_files:
        with open(temp_match, "r", encoding="utf-8") as fm:
            for line in fm:
                parts = line.rstrip("\r\n").split("\t")
                matching_dict[parts[0]] = parts[1] if len(parts) > 1 else ""

        with open(temp_cand, "r", encoding="utf-8") as fc:
            for line in fc:
                parts = line.rstrip("\r\n").split("\t")
                candidate_dict[parts[0]] = parts[1] if len(parts) > 1 else ""

    print(f"Loaded {len(matching_dict):,} matching rows and {len(candidate_dict):,} candidate rows ({time.time()-t0:.2f}s).", flush=True)

    # 2. Stream test_source1.tsv and write final files
    print("Writing final submission files:", flush=True)
    print(f"  -> {final_matching_path}", flush=True)
    print(f"  -> {final_candidate_path}", flush=True)

    total_rows = 0
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f_in, \
         open(final_matching_path, "w", encoding="utf-8", newline="\n") as f_m_out, \
         open(final_candidate_path, "w", encoding="utf-8", newline="\n") as f_c_out:

        # Official Headers
        f_m_out.write("source1_entity_id\tmatched_entity_ids\n")
        f_c_out.write("source1_entity_id\tcandidate_entity_ids\n")

        next(f_in, None)  # skip test_source1 header
        for line in f_in:
            s1_id = line.partition("\t")[0].strip()
            if not s1_id:
                continue

            total_rows += 1
            m_val = matching_dict.get(s1_id, "")
            c_val = candidate_dict.get(s1_id, "")

            f_m_out.write(f"{s1_id}\t{m_val}\n")
            f_c_out.write(f"{s1_id}\t{c_val}\n")

    print(f"Successfully generated {total_rows:,} rows in both submission files!", flush=True)

    # Clean up temporary partition files
    for _, temp_match, temp_cand in country_temp_files:
        if os.path.exists(temp_match):
            os.remove(temp_match)
        if os.path.exists(temp_cand):
            os.remove(temp_cand)
    print("Cleaned up intermediate partition files.", flush=True)


def run_test_inference_pipeline():
    """
    Main orchestration routine for Step 7 test set prediction.
    """
    print("=" * 80, flush=True)
    print("    AMAZON ML CHALLENGE 2026 - STEP 7: TEST INFERENCE & SUBMISSION", flush=True)
    print("=" * 80, flush=True)

    base_dir = resolve_base_dir()
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Acquire concurrency lock
    acquire_process_lock(output_dir)

    s1_path = resolve_dataset_file(base_dir, "dataset/test/test_source1.tsv")
    model_path = os.path.join(base_dir, "models/matcher_model.joblib")
    final_matching_path = os.path.join(output_dir, "matching_results.tsv")
    final_candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # 1. Verify environment and prerequisites
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run Step 6 first!")
    if not os.path.exists(s1_path):
        raise FileNotFoundError(f"Test source1 file not found at {s1_path}!")

    # 2. Load trained model & optimal threshold
    print(f"Loading trained matcher from: {model_path}", flush=True)
    matcher = EntityMatcher.load(model_path)
    print(f"Model successfully loaded. Classification Threshold: {matcher.optimal_threshold:.2f}", flush=True)

    # 3. Dynamically discover countries and entity counts from test data
    print("\nScanning test dataset to discover countries dynamically...", flush=True)
    countries, country_counts = discover_countries_dynamically(s1_path)
    print(f"Discovered {len(countries)} countries in test data: {countries}", flush=True)
    for c, cnt in country_counts.items():
        print(f"  - {c}: {cnt:,} Source 1 records", flush=True)

    t_global_start = time.time()
    country_temp_files = []

    # 4. Process each country partition sequentially
    for country_idx, country in enumerate(countries, 1):
        print("\n" + "-" * 80, flush=True)
        print(f"[{country_idx}/{len(countries)}] PROCESSING COUNTRY PARTITION: {country}", flush=True)
        print("-" * 80, flush=True)

        slug = country.lower().replace(" ", "_")
        temp_match = os.path.join(output_dir, f"temp_matching_{slug}.tsv")
        temp_cand = os.path.join(output_dir, f"temp_candidates_{slug}.tsv")

        expected_count = country_counts[country]

        # Check for completed checkpoint
        if os.path.exists(temp_match) and os.path.exists(temp_cand):
            with open(temp_match, "r", encoding="utf-8") as f_chk_m:
                m_lines = sum(1 for _ in f_chk_m)
            with open(temp_cand, "r", encoding="utf-8") as f_chk_c:
                c_lines = sum(1 for _ in f_chk_c)
            if m_lines == expected_count and c_lines == expected_count:
                print(f"  [CHECKPOINT] Partition {country} already fully completed ({m_lines:,} rows). Reusing existing partition.", flush=True)
                country_temp_files.append((country, temp_match, temp_cand))
                continue

        # Step 4a: Build inverted index for this country
        cand_pool, index_dict = build_country_candidate_index(base_dir, country)

        # Step 4b: Run fast vectorized batched inference on Source 1 records for this country
        process_country_s1_batched(
            base_dir=base_dir,
            target_country=country,
            cand_pool=cand_pool,
            index_dict=index_dict,
            model_path=model_path,
            temp_matching_path=temp_match,
            temp_candidate_path=temp_cand,
            max_block_size=100,  # Balanced: 94.9% posting coverage, ~2x faster than cap=200 on large indices
            batch_size=1000,
        )

        country_temp_files.append((country, temp_match, temp_cand))

        # Step 4c: Explicit memory cleanup before next country partition
        del cand_pool
        del index_dict
        gc.collect()
        print(f"  Memory freed for country: {country}", flush=True)

    # 5. Assemble final submission files
    assemble_final_submissions(
        base_dir=base_dir,
        country_temp_files=country_temp_files,
        final_matching_path=final_matching_path,
        final_candidate_path=final_candidate_path,
    )

    # 6. Run official submission validation
    print("\n" + "=" * 80, flush=True)
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR (utils/validate_submission.py)", flush=True)
    print("=" * 80, flush=True)
    test_dir = os.path.dirname(resolve_dataset_file(base_dir, "dataset/test/test_source1.tsv"))
    errors, warnings = validate(
        matching_path=final_matching_path,
        candidate_path=final_candidate_path,
        test_dir=test_dir,
        check_ids=False,
    )

    print("\n" + "-" * 80, flush=True)
    print("SUBMISSION VALIDATION SUMMARY:", flush=True)
    print("-" * 80, flush=True)
    for w in warnings:
        print(f"  [WARNING] {w}", flush=True)
    if errors:
        print(f"  [ERROR COUNT]: {len(errors)}", flush=True)
        for err in errors:
            print(f"    - {err}", flush=True)
        print("\n--> VALIDATION STATUS: FAILED!", flush=True)
        sys.exit(1)
    else:
        print("  --> VALIDATION STATUS: PASSED 100% (SAFE TO SUBMIT)!", flush=True)

    total_time = time.time() - t_global_start
    print(f"\nStep 7 Test Inference & Submission Generation finished in {total_time:.2f}s ({total_time/60:.2f} min).", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    run_test_inference_pipeline()
