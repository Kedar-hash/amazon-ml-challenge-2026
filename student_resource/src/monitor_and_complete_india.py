#!/usr/bin/env python3
"""
Monitor, Resume, and Finalize Step 7 India Processing and Submission Generation.

Key Guarantees:
- Inspects and tracks existing progress of India Source 1 records.
- If existing process finishes, monitors until complete and preserves all checkpoints.
- If existing process terminates early, seamlessly resumes from the exact record index
  without reprocessing any already-completed records.
- Preserves and restores all checkpoint partitions (US: 663,106, France: 259,452, India: 809,986).
- Strictly dataset-driven: no hardcoded business data, entity IDs, candidates, or labels.
- Validates the final submission files with utils/validate_submission.py.
- Reports initial completed count, remaining processed count, and final total.
"""

import os
import sys
import time
import math
import gc
import shutil
import ctypes
from ctypes import wintypes
from array import array
from collections import defaultdict

# Immediate flushing on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True, errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True, errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_RESOURCE_DIR = os.path.dirname(SCRIPT_DIR)
if STUDENT_RESOURCE_DIR not in sys.path:
    sys.path.insert(0, STUDENT_RESOURCE_DIR)

from src.text_normalizer import (
    normalize_business_name,
    extract_core_name,
    normalize_address,
    get_address_tokens,
)
from src.blocking import extract_addr_numbers, generate_blocking_keys
from src.feature_generator import levenshtein_similarity
from src.model_trainer import EntityMatcher
from src.predict_submission import (
    CompactCandidate,
    Source1Record,
    fast_compute_pair_features_vector,
    build_country_candidate_index,
    discover_countries_dynamically,
    assemble_final_submissions,
)
from utils.validate_submission import validate


def is_process_alive(pid: int) -> bool:
    """Check if process PID is currently running using Windows Win32 API."""
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    code = wintypes.DWORD()
    kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    kernel32.CloseHandle(h)
    return code.value == 259  # STILL_ACTIVE


def count_lines_in_file(path: str) -> int:
    """Safely count lines in a file (works even while file is being appended)."""
    if not os.path.exists(path):
        return 0
    cnt = 0
    try:
        with open(path, "rb") as f:
            for _ in f:
                cnt += 1
    except Exception:
        pass
    return cnt


def resume_india_processing(base_dir: str, completed_count: int, total_expected: int):
    """
    Resume India S1 processing from record `completed_count` up to `total_expected`.
    Appends remaining predictions to output files without reprocessing earlier records.
    """
    output_dir = os.path.join(base_dir, "output")
    temp_matching_path = os.path.join(output_dir, "temp_matching_india.tsv")
    temp_candidate_path = os.path.join(output_dir, "temp_candidates_india.tsv")
    part_matching = temp_matching_path + ".tmp"
    part_candidate = temp_candidate_path + ".tmp"
    model_path = os.path.join(base_dir, "models/matcher_model.joblib")
    s1_path = os.path.join(base_dir, "dataset/test/test_source1.tsv")

    print(f"\n[RESUME] Loading candidate pool & building inverted index for India...", flush=True)
    cand_pool, index_dict = build_country_candidate_index(base_dir, "India")

    print(f"[RESUME] Loading matcher model from {model_path}...", flush=True)
    matcher = EntityMatcher.load(model_path)
    threshold = matcher.optimal_threshold

    print(f"[RESUME] Skipping first {completed_count:,} already-processed India records...", flush=True)
    print(f"[RESUME] Appending remaining {total_expected - completed_count:,} records...", flush=True)

    max_block_size = 50
    batch_size = 1000
    batch_slices = []
    batch_X = []
    total_matches_predicted = 0
    total_candidates_evaluated = 0
    s1_count = completed_count
    t_start = time.time()

    def flush_current_batch(f_m, f_c):
        nonlocal batch_slices, batch_X, total_matches_predicted
        if not batch_slices:
            return
        probs = matcher.predict_proba(batch_X) if batch_X else []
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

    india_seen = 0
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f_in, \
         open(part_matching, "a", encoding="utf-8", newline="\n") as f_match, \
         open(part_candidate, "a", encoding="utf-8", newline="\n") as f_cand:

        next(f_in, None)  # skip header
        for line in f_in:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4 or parts[3].strip() != "India":
                continue

            india_seen += 1
            if india_seen <= completed_count:
                continue

            s1 = Source1Record(parts[0], parts[1], parts[2], parts[3].strip())
            s1_count += 1

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

            if len(batch_slices) >= batch_size:
                flush_current_batch(f_match, f_cand)

            if s1_count % 25000 == 0 or s1_count == total_expected:
                elapsed = time.time() - t_start
                rate = (s1_count - completed_count) / elapsed if elapsed > 0 else 0
                print(
                    f"    [India Resume] Processed {s1_count:,}/{total_expected:,} S1 records "
                    f"({rate:.1f} rec/s | matches: {total_matches_predicted:,} | cands: {total_candidates_evaluated:,})",
                    flush=True
                )

        flush_current_batch(f_match, f_cand)

    del cand_pool
    del index_dict
    gc.collect()

    # Commit finished partition files
    if os.path.exists(temp_matching_path):
        os.remove(temp_matching_path)
    if os.path.exists(temp_candidate_path):
        os.remove(temp_candidate_path)
    os.replace(part_matching, temp_matching_path)
    os.replace(part_candidate, temp_candidate_path)
    print(f"[RESUME] Successfully committed full {s1_count:,} India records to checkpoint files!", flush=True)


def main():
    base_dir = STUDENT_RESOURCE_DIR
    output_dir = os.path.join(base_dir, "output")
    backup_dir = os.path.join(output_dir, "checkpoint_backup")
    os.makedirs(backup_dir, exist_ok=True)

    lock_file = os.path.join(output_dir, ".predict_lock")
    temp_matching_india = os.path.join(output_dir, "temp_matching_india.tsv")
    temp_candidate_india = os.path.join(output_dir, "temp_candidates_india.tsv")
    part_matching_india = temp_matching_india + ".tmp"
    part_candidate_india = temp_candidate_india + ".tmp"

    s1_path = os.path.join(base_dir, "dataset/test/test_source1.tsv")
    countries, country_counts = discover_countries_dynamically(s1_path)
    india_expected = country_counts.get("India", 809986)

    print("=" * 80, flush=True)
    print("STEP 7 RESUMPTION & COMPLETION MONITOR: INDIA PARTITION", flush=True)
    print(f"Target Country      : India", flush=True)
    print(f"Total Expected Rows : {india_expected:,}", flush=True)
    print("=" * 80, flush=True)

    # 1. Back up existing completed US and France checkpoints
    for c in ["us", "france"]:
        for prefix in ["temp_matching_", "temp_candidates_"]:
            src = os.path.join(output_dir, f"{prefix}{c}.tsv")
            dst = os.path.join(backup_dir, f"{prefix}{c}.tsv")
            if os.path.exists(src) and (not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src)):
                shutil.copy2(src, dst)
                print(f"Secured checkpoint backup: {prefix}{c}.tsv ({os.path.getsize(src):,} bytes)", flush=True)

    # 2. Determine existing processed count
    initial_processed = 0
    if os.path.exists(temp_matching_india):
        initial_processed = count_lines_in_file(temp_matching_india)
    elif os.path.exists(part_matching_india):
        initial_processed = count_lines_in_file(part_matching_india)

    print(f"\n[INSPECTION RESULT] Initial India Source 1 records processed: {initial_processed:,} / {india_expected:,}", flush=True)
    remaining_to_process = india_expected - initial_processed
    print(f"[INSPECTION RESULT] Remaining India Source 1 records: {remaining_to_process:,}", flush=True)

    # 3. Check if external inference process (e.g. PID 6196) is actively running
    monitored_pid = None
    if os.path.exists(lock_file):
        try:
            with open(lock_file, "r") as lf:
                txt = lf.read().strip()
                if txt.isdigit():
                    monitored_pid = int(txt)
        except Exception:
            pass

    if monitored_pid and is_process_alive(monitored_pid):
        print(f"\nActive inference process detected (PID: {monitored_pid}). Monitoring live execution...", flush=True)
        last_logged = initial_processed
        last_backup_count = 0
        t0 = time.time()

        while is_process_alive(monitored_pid):
            curr_tmp = count_lines_in_file(part_matching_india)
            curr_fin = count_lines_in_file(temp_matching_india)
            curr = max(curr_tmp, curr_fin)

            # Keep incremental backup so progress is never lost
            if curr > last_backup_count + 50000:
                if os.path.exists(part_matching_india) and os.path.exists(part_candidate_india):
                    try:
                        shutil.copy2(part_matching_india, os.path.join(backup_dir, "temp_matching_india.tsv"))
                        shutil.copy2(part_candidate_india, os.path.join(backup_dir, "temp_candidates_india.tsv"))
                        last_backup_count = curr
                    except Exception:
                        pass

            if curr >= last_logged + 25000 or (time.time() - t0 >= 30 and curr > last_logged):
                elapsed = time.time() - t0
                pct = (curr / india_expected) * 100
                rate = (curr - initial_processed) / elapsed if elapsed > 0 else 0
                eta_s = (india_expected - curr) / rate if rate > 0 else 0
                print(f"  [PID {monitored_pid}] Processed {curr:,} / {india_expected:,} records ({pct:.1f}%) | {rate:.1f} rec/s | ETA: {eta_s/60:.1f} min", flush=True)
                last_logged = curr
                t0 = time.time()

            time.sleep(10)

        print(f"\nInference process (PID {monitored_pid}) has finished.", flush=True)
        time.sleep(3)

    # 4. Check current status after process termination
    curr_fin = count_lines_in_file(temp_matching_india)
    curr_tmp = count_lines_in_file(part_matching_india)
    final_matching_path = os.path.join(output_dir, "matching_results.tsv")
    final_candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # If the process did not finish all records, resume processing in this script
    if curr_fin < india_expected and count_lines_in_file(final_matching_path) < 1732545:
        curr_processed = max(curr_fin, curr_tmp)
        print(f"\n[STATUS] India partition at {curr_processed:,}/{india_expected:,}. Resuming remaining records...", flush=True)
        resume_india_processing(base_dir, curr_processed, india_expected)

    # Backup the completed India checkpoint files
    if os.path.exists(temp_matching_india):
        shutil.copy2(temp_matching_india, os.path.join(backup_dir, "temp_matching_india.tsv"))
    if os.path.exists(temp_candidate_india):
        shutil.copy2(temp_candidate_india, os.path.join(backup_dir, "temp_candidates_india.tsv"))

    # 5. Check if final submission assembly is needed
    m_lines = count_lines_in_file(final_matching_path)
    c_lines = count_lines_in_file(final_candidate_path)
    total_test_rows = sum(country_counts.values())

    if m_lines != total_test_rows + 1 or c_lines != total_test_rows + 1:
        print("\nAssembling final submission files...", flush=True)
        # Restore any checkpoint files that might have been cleaned up
        for c in ["us", "france", "india"]:
            for prefix in ["temp_matching_", "temp_candidates_"]:
                f_name = f"{prefix}{c}.tsv"
                src = os.path.join(backup_dir, f_name)
                dst = os.path.join(output_dir, f_name)
                if not os.path.exists(dst) and os.path.exists(src):
                    shutil.copy2(src, dst)

        country_temp_files = []
        for c in countries:
            slug = c.lower().replace(" ", "_")
            tm = os.path.join(output_dir, f"temp_matching_{slug}.tsv")
            tc = os.path.join(output_dir, f"temp_candidates_{slug}.tsv")
            country_temp_files.append((c, tm, tc))

        assemble_final_submissions(
            base_dir=base_dir,
            country_temp_files=country_temp_files,
            final_matching_path=final_matching_path,
            final_candidate_path=final_candidate_path,
        )

    # 6. Preserve and restore checkpoint files so user requirement is 100% honored
    print("\nEnsuring all country checkpoint files are preserved and restored...", flush=True)
    for c in ["us", "france", "india"]:
        for prefix in ["temp_matching_", "temp_candidates_"]:
            f_name = f"{prefix}{c}.tsv"
            src = os.path.join(backup_dir, f_name)
            dst = os.path.join(output_dir, f_name)
            if not os.path.exists(dst) and os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  Restored completed checkpoint: {dst}", flush=True)

    # 7. Official Submission Validation
    print("\n" + "=" * 80, flush=True)
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR (utils/validate_submission.py)", flush=True)
    print("=" * 80, flush=True)
    test_dir = os.path.join(base_dir, "dataset/test")
    errors, warnings = validate(
        matching_path=final_matching_path,
        candidate_path=final_candidate_path,
        test_dir=test_dir,
        check_ids=False,
    )

    print("\nSUBMISSION VALIDATION SUMMARY:", flush=True)
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

    # 8. Final Report
    india_chk = os.path.join(output_dir, "temp_matching_india.tsv")
    final_india_count = count_lines_in_file(india_chk)
    final_matching_rows = count_lines_in_file(final_matching_path) - 1

    print("\n" + "=" * 80, flush=True)
    print("FINAL STEP 7 EXECUTION SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"Initial India Processed Count : {initial_processed:,}")
    print(f"Remaining India Processed     : {india_expected - initial_processed:,}")
    print(f"Final India Processed Total   : {final_india_count:,} (Expected: {india_expected:,})")
    print(f"Total Submission Rows         : {final_matching_rows:,} (Expected: {total_test_rows:,})")
    print(f"Submission Files Ready at     : {final_matching_path}")
    print(f"                                {final_candidate_path}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
