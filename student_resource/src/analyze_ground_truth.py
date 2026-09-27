#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Step 2: Ground Truth & Match Analysis

This script analyzes dataset/train/train_ground_truth.tsv in a memory-efficient,
chunked manner. It computes:
- Total Source 1 entities
- Match count distribution: 0 (singletons), 1, 2, 3+ matches
- Total S2 and S3 matches
- Key match statistics (mean, median, min, max, total pairs)
- Samples real matching pairs joined with train_source1.tsv, train_source2.tsv,
  and train_source3.tsv to examine real-world noise patterns.

Usage:
    python src/analyze_ground_truth.py
"""

import os
import sys
import time
from collections import Counter
import pandas as pd

# Reconfigure stdout/stderr for proper UTF-8 handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def resolve_base_dir():
    """Resolve base directory whether run from workspace root or student_resource/."""
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(script_dir)
    if os.path.isdir(os.path.join(parent, "dataset", "train")):
        return parent
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


def analyze_ground_truth(gt_path, chunk_size=250_000):
    """
    Chunked analysis of train_ground_truth.tsv.
    Memory footprint stays minimal (< 50MB) even on large files.
    """
    print(f"Reading and analyzing: {gt_path} (chunk_size={chunk_size:,})...")
    t0 = time.time()

    total_s1 = 0
    match_counts = Counter()
    total_s2 = 0
    total_s3 = 0
    source_combo_counts = Counter()  # 'none', 's2_only', 's3_only', 'both'

    # Track candidate S1 IDs for sampling diverse matches
    sample_candidate_ids = {
        "singleton_0": [],
        "single_s2": [],
        "single_s3": [],
        "one_each_s2_s3": [],
        "multi_matches": [],
    }

    for chunk in pd.read_csv(
        gt_path,
        sep="\t",
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
    ):
        total_s1 += len(chunk)

        for s1_id, m_str in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            m_str = m_str.strip()
            if not m_str:
                match_counts[0] += 1
                source_combo_counts["0 matches (singletons)"] += 1
                if len(sample_candidate_ids["singleton_0"]) < 3:
                    sample_candidate_ids["singleton_0"].append(s1_id)
                continue

            ids = [x.strip() for x in m_str.split(",") if x.strip()]
            num_matches = len(ids)
            match_counts[num_matches] += 1

            s2_sub = [x for x in ids if x.startswith("S2-")]
            s3_sub = [x for x in ids if x.startswith("S3-")]

            num_s2 = len(s2_sub)
            num_s3 = len(s3_sub)
            total_s2 += num_s2
            total_s3 += num_s3

            if num_s2 > 0 and num_s3 > 0:
                source_combo_counts["Both S2 and S3"] += 1
                if num_s2 == 1 and num_s3 == 1 and len(sample_candidate_ids["one_each_s2_s3"]) < 3:
                    sample_candidate_ids["one_each_s2_s3"].append((s1_id, ids))
                elif num_matches >= 4 and len(sample_candidate_ids["multi_matches"]) < 3:
                    sample_candidate_ids["multi_matches"].append((s1_id, ids))
            elif num_s2 > 0 and num_s3 == 0:
                source_combo_counts["S2 only"] += 1
                if num_matches == 1 and len(sample_candidate_ids["single_s2"]) < 3:
                    sample_candidate_ids["single_s2"].append((s1_id, ids))
            elif num_s3 > 0 and num_s2 == 0:
                source_combo_counts["S3 only"] += 1
                if num_matches == 1 and len(sample_candidate_ids["single_s3"]) < 3:
                    sample_candidate_ids["single_s3"].append((s1_id, ids))

    elapsed = time.time() - t0
    print(f"Ground truth scan completed in {elapsed:.2f} seconds.\n")

    return {
        "total_s1": total_s1,
        "match_counts": match_counts,
        "total_s2": total_s2,
        "total_s3": total_s3,
        "source_combo_counts": source_combo_counts,
        "sample_candidate_ids": sample_candidate_ids,
    }


def stream_fetch_records(file_path, target_ids):
    """
    Stream a TSV file line-by-line to retrieve only specific IDs without
    loading the entire multi-hundred MB file into RAM.
    """
    results = {}
    target_set = set(target_ids)
    if not target_set:
        return results

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().rstrip("\r\n")
        headers = header_line.split("\t")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts and parts[0] in target_set:
                results[parts[0]] = dict(zip(headers, parts))
                target_set.remove(parts[0])
                if not target_set:
                    break
    return results


def print_statistics(stats):
    total_s1 = stats["total_s1"]
    match_counts = stats["match_counts"]
    total_s2 = stats["total_s2"]
    total_s3 = stats["total_s3"]
    total_matches = total_s2 + total_s3

    count_0 = match_counts[0]
    count_1 = match_counts[1]
    count_2 = match_counts[2]
    count_3_plus = sum(c for k, c in match_counts.items() if k >= 3)

    # Compute median
    cumulative = 0
    median_val = None
    halfway = total_s1 / 2.0
    for k in sorted(match_counts.keys()):
        cumulative += match_counts[k]
        if cumulative >= halfway and median_val is None:
            median_val = k

    mean_val = total_matches / total_s1 if total_s1 > 0 else 0
    min_val = min(match_counts.keys())
    max_val = max(match_counts.keys())

    print("=" * 75)
    print("               STEP 2: GROUND TRUTH ANALYSIS RESULTS")
    print("=" * 75)

    print("\n1. ENTITY & MATCH TOTALS:")
    print("-" * 50)
    print(f"  Total Source 1 Entities : {total_s1:>12,}")
    print(f"  Total S2 Matches        : {total_s2:>12,}")
    print(f"  Total S3 Matches        : {total_s3:>12,}")
    print(f"  Total Match Pairs (S2+S3): {total_matches:>11,}")

    print("\n2. MATCH COUNT DISTRIBUTION (Source 1 Entities):")
    print("-" * 65)
    print(f"  {'Category':<22} | {'Count':>12} | {'Percentage':>10}")
    print("-" * 65)
    print(f"  {'0 matches (Singletons)':<22} | {count_0:>12,} | {count_0 / total_s1 * 100:>9.2f}%")
    print(f"  {'1 match':<22} | {count_1:>12,} | {count_1 / total_s1 * 100:>9.2f}%")
    print(f"  {'2 matches':<22} | {count_2:>12,} | {count_2 / total_s1 * 100:>9.2f}%")
    print(f"  {'3+ matches':<22} | {count_3_plus:>12,} | {count_3_plus / total_s1 * 100:>9.2f}%")
    print("-" * 65)

    print("\n3. DETAILED MATCH BREAKDOWN (Exact distribution 0..max):")
    print("-" * 50)
    for k in sorted(match_counts.keys()):
        c = match_counts[k]
        pct = c / total_s1 * 100
        bar = "#" * int(pct / 2)
        print(f"  {k:>2} matches : {c:>10,} ({pct:>5.2f}%) {bar}")

    print("\n4. SUMMARY STATISTICS:")
    print("-" * 50)
    print(f"  Min matches per entity    : {min_val}")
    print(f"  Max matches per entity    : {max_val}")
    print(f"  Mean matches per entity   : {mean_val:.2f}")
    print(f"  Median matches per entity : {median_val}")

    print("\n5. SOURCE COMBINATION BREAKDOWN:")
    print("-" * 50)
    for combo, c in stats["source_combo_counts"].most_common():
        pct = c / total_s1 * 100
        print(f"  {combo:<25} : {c:>10,} ({pct:>5.2f}%)")
    print("=" * 75)


def display_sample_pairs(base_dir, stats):
    """Fetch sample records across S1, S2, S3 and display side-by-side."""
    print("\n" + "=" * 75)
    print("         SAMPLE ACTUAL MATCHING PAIRS ACROSS SOURCES")
    print("=" * 75)

    samples = stats["sample_candidate_ids"]

    # Gather selected IDs to fetch in one pass per source file
    selected_cases = []

    # Case A: 1 match (S3)
    if samples["single_s3"]:
        s1_id, match_ids = samples["single_s3"][0]
        selected_cases.append({
            "title": "CASE 1: Single Match (S1 -> S3 only)",
            "s1_id": s1_id,
            "match_ids": match_ids,
            "category": "1 match"
        })

    # Case B: 2 matches (1 S2, 1 S3)
    if samples["one_each_s2_s3"]:
        s1_id, match_ids = samples["one_each_s2_s3"][0]
        selected_cases.append({
            "title": "CASE 2: Dual Source Match (S1 -> 1 in S2, 1 in S3)",
            "s1_id": s1_id,
            "match_ids": match_ids,
            "category": "2 matches"
        })

    # Case C: Multi matches (multiple S2 + S3)
    if samples["multi_matches"]:
        s1_id, match_ids = samples["multi_matches"][0]
        selected_cases.append({
            "title": "CASE 3: High-Multiplicity Match (S1 -> 2 in S2, 3 in S3)",
            "s1_id": s1_id,
            "match_ids": match_ids,
            "category": "multi-matches"
        })

    # Case D: Singleton (0 matches)
    if samples["singleton_0"]:
        s1_id = samples["singleton_0"][0]
        selected_cases.append({
            "title": "CASE 4: Singleton Entity (0 matches)",
            "s1_id": s1_id,
            "match_ids": [],
            "category": "singleton"
        })

    all_s1_ids = [c["s1_id"] for c in selected_cases]
    all_s2_ids = [m for c in selected_cases for m in c["match_ids"] if m.startswith("S2-")]
    all_s3_ids = [m for c in selected_cases for m in c["match_ids"] if m.startswith("S3-")]

    s1_path = resolve_dataset_file(base_dir, "dataset/train/train_source1.tsv")
    s2_path = resolve_dataset_file(base_dir, "dataset/train/train_source2.tsv")
    s3_path = resolve_dataset_file(base_dir, "dataset/train/train_source3.tsv")

    print(f"Fetching records from source files for {len(all_s1_ids)} S1, {len(all_s2_ids)} S2, {len(all_s3_ids)} S3 entities...")
    rec_s1 = stream_fetch_records(s1_path, all_s1_ids)
    rec_s2 = stream_fetch_records(s2_path, all_s2_ids)
    rec_s3 = stream_fetch_records(s3_path, all_s3_ids)

    for case in selected_cases:
        print("\n" + "#" * 75)
        print(f" {case['title']}")
        print("#" * 75)

        s1_rec = rec_s1.get(case["s1_id"], {})
        print(f"  [SOURCE 1 REFERENCE] ({case['s1_id']})")
        print(f"    Business Name   : {s1_rec.get('business_name', 'N/A')}")
        print(f"    Business Address: {s1_rec.get('business_address', 'N/A')}")
        print(f"    Country         : {s1_rec.get('country', 'N/A')}")

        if not case["match_ids"]:
            print("    -> No matching records exist in S2 or S3 (Singleton)")
            continue

        print("\n  [MATCHED RECORDS IN S2 & S3]:")
        for mid in case["match_ids"]:
            if mid.startswith("S2-"):
                m_rec = rec_s2.get(mid, {})
                src_label = "Source 2"
            else:
                m_rec = rec_s3.get(mid, {})
                src_label = "Source 3"

            print(f"  * [{src_label}] {mid}:")
            print(f"      Business Name   : {m_rec.get('business_name', 'N/A')}")
            print(f"      Business Address: {m_rec.get('business_address', 'N/A')}")
            print(f"      Country         : {m_rec.get('country', 'N/A')}")

    print("\n" + "=" * 75)


def main():
    base_dir = resolve_base_dir()
    gt_file = resolve_dataset_file(base_dir, "dataset/train/train_ground_truth.tsv")

    if not os.path.exists(gt_file):
        print(f"[ERROR] Ground truth file not found at: {gt_file}")
        sys.exit(1)

    # 1. Analyze ground truth distribution
    stats = analyze_ground_truth(gt_file, chunk_size=250_000)

    # 2. Print metrics and statistics
    print_statistics(stats)

    # 3. Sample and join matching pairs across source files
    display_sample_pairs(base_dir, stats)


if __name__ == "__main__":
    main()
