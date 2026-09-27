#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Dataset Sanity & Exploration Checker

This script checks the dataset files and project directory structure
for the Business Entity Resolution challenge.

Usage:
    python check_data.py
    python check_data.py --preview 3
    python check_data.py --count-rows
"""

import os
import sys
import argparse
import time

# Ensure Windows terminal handles UTF-8 characters gracefully
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SOURCE_EXPECTED_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_EXPECTED_COLS = ["source1_entity_id", "matched_entity_ids"]

EXPECTED_FILES = {
    "Train Source 1": ("dataset/train/train_source1.tsv", SOURCE_EXPECTED_COLS, "S1-"),
    "Train Source 2": ("dataset/train/train_source2.tsv", SOURCE_EXPECTED_COLS, "S2-"),
    "Train Source 3": ("dataset/train/train_source3.tsv", SOURCE_EXPECTED_COLS, "S3-"),
    "Train Ground Truth": ("dataset/train/train_ground_truth.tsv", GT_EXPECTED_COLS, None),
    "Test Source 1": ("dataset/test/test_source1.tsv", SOURCE_EXPECTED_COLS, "S1-"),
    "Test Source 2": ("dataset/test/test_source2.tsv", SOURCE_EXPECTED_COLS, "S2-"),
    "Test Source 3": ("dataset/test/test_source3.tsv", SOURCE_EXPECTED_COLS, "S3-"),
}

EXPECTED_DIRS = ["dataset/train", "dataset/test", "src", "output", "utils"]


def format_size(bytes_val):
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def count_file_lines(file_path):
    """Fast line count using raw binary chunks to avoid large memory footprint."""
    count = 0
    buffer_size = 1024 * 1024 * 8  # 8 MB buffer
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(buffer_size)
            if not chunk:
                break
            count += chunk.count(b"\n")
    return count


def resolve_base_dir():
    """Resolve directory so script runs from inside student_resource or workspace root."""
    cwd = os.getcwd()
    candidates = [
        cwd,
        os.path.join(cwd, "student_resource"),
        os.path.dirname(os.path.abspath(cwd)),
        os.path.dirname(os.path.abspath(__file__)),
    ]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "dataset", "train")) or \
           os.path.isdir(os.path.join(cand, "dataset", "dataset", "train")):
            return cand
    if os.path.isdir(os.path.join(cwd, "dataset")):
        return cwd
    if os.path.isdir(os.path.join(cwd, "student_resource", "dataset")):
        return os.path.join(cwd, "student_resource")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir(os.path.join(script_dir, "dataset")):
        return script_dir
    return cwd


def resolve_dataset_file(base_dir, rel_path):
    """Resolve file path across standard dataset/ and nested dataset/dataset/ structures."""
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


def check_directories(base_dir):
    print("=" * 70)
    print("1. DIRECTORY STRUCTURE CHECK")
    print("=" * 70)
    all_ok = True
    for rel_dir in EXPECTED_DIRS:
        candidates = [
            os.path.join(base_dir, rel_dir),
            os.path.join(base_dir, "dataset", rel_dir),
            os.path.join(base_dir, "student_resource", rel_dir),
            os.path.join(os.path.dirname(os.path.abspath(base_dir)), rel_dir),
            os.path.join(os.path.dirname(os.path.abspath(base_dir)), "student_resource", rel_dir),
        ]
        exists = any(os.path.isdir(c) for c in candidates)
        status = "OK" if exists else "MISSING"
        if not exists:
            all_ok = False
        print(f"  [{status:<7}] {rel_dir}/")
    print()
    return all_ok


def inspect_files(base_dir, preview_n=3, count_rows=False):
    print("=" * 70)
    print("2. DATASET FILES VALIDATION & PREVIEWS")
    print("=" * 70)

    summary = []

    for name, (rel_path, expected_cols, expected_prefix) in EXPECTED_FILES.items():
        full_path = resolve_dataset_file(base_dir, rel_path)
        print(f"\n--- Checking: {name} ({rel_path}) ---")

        if not os.path.exists(full_path):
            print(f"  [ERROR] File not found: {full_path}")
            summary.append({"name": name, "file": rel_path, "status": "NOT FOUND", "size": "-", "rows": "-"})
            continue

        file_size = os.path.getsize(full_path)
        size_str = format_size(file_size)
        print(f"  File size: {size_str} ({file_size:,} bytes)")

        # Read header and first preview_n lines
        rows = []
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            header_line = f.readline().rstrip("\r\n")
            header_cols = header_line.split("\t")
            for _ in range(preview_n):
                line = f.readline()
                if not line:
                    break
                rows.append(line.rstrip("\r\n").split("\t"))

        # Check delimiter & header
        if len(header_cols) == 1 and "\t" not in header_line:
            print("  [WARNING] Delimiter issue: Could not find tab delimiter in header!")
        else:
            print(f"  Columns ({len(header_cols)}): {header_cols}")

        if header_cols == expected_cols:
            print("  [OK] Header matches expected schema.")
        else:
            print(f"  [WARNING] Header mismatch! Expected: {expected_cols}, Got: {header_cols}")

        # Check entity prefix if applicable
        if expected_prefix and rows:
            first_id = rows[0][0] if rows[0] else ""
            if first_id.startswith(expected_prefix):
                print(f"  [OK] Entity ID prefix verified (e.g., '{first_id}')")
            else:
                print(f"  [WARNING] Expected ID prefix '{expected_prefix}', got '{first_id}'")

        # Preview sample records
        print(f"  Preview (first {len(rows)} data rows):")
        for idx, row in enumerate(rows, 1):
            if len(row) >= 4:
                print(f"    Row {idx}: ID={row[0]} | Country={row[3]} | Name={row[1][:40]!r} | Addr={row[2][:50]!r}...")
            else:
                print(f"    Row {idx}: {row}")

        total_lines = "-"
        if count_rows:
            t0 = time.time()
            total_lines = count_file_lines(full_path) - 1  # exclude header
            t1 = time.time()
            print(f"  Total records: {total_lines:,} (counted in {t1 - t0:.2f}s)")

        summary.append({
            "name": name,
            "file": rel_path,
            "status": "VALID",
            "size": size_str,
            "rows": f"{total_lines:,}" if isinstance(total_lines, int) else total_lines,
        })

    print("\n" + "=" * 70)
    print("3. DATASET SUMMARY")
    print("=" * 70)
    print(f"{'Dataset File':<22} | {'Relative Path':<35} | {'Size':<10} | {'Records':<12}")
    print("-" * 87)
    for item in summary:
        print(f"{item['name']:<22} | {item['file']:<35} | {item['size']:<10} | {item['rows']:<12}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Check Amazon ML Challenge 2026 dataset files and directories.")
    parser.add_argument("--preview", type=int, default=3, help="Number of rows to preview per file (default: 3)")
    parser.add_argument("--count-rows", action="store_true", help="Count total lines for each file (fast binary scan)")
    args = parser.parse_args()

    base_dir = resolve_base_dir()
    print(f"Base Directory: {base_dir}\n")

    check_directories(base_dir)
    inspect_files(base_dir, preview_n=args.preview, count_rows=args.count_rows)
    print("\nCheck completed successfully.")


if __name__ == "__main__":
    main()
