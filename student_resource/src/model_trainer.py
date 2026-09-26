#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Step 6: ML Entity Matcher Training & Validation

This module trains a precision-focused gradient boosted decision tree model
(HistGradientBoostingClassifier) to classify candidate pairs as true matches or non-matches.

Key Pipeline Steps:
1. Dynamic Data Loading: Loads representative Source 1 entities and corresponding
   candidate pools dynamically from the training TSVs. No hard-coded business data or IDs.
2. Entity-Level Split: Splits data strictly by Source 1 entities (e.g. 80% train / 20% val)
   to ensure the validation set tests model generalization on completely unseen businesses.
3. Hard Negative Mining: Generates natural hard negatives using the blocking index
   (candidates sharing blocking keys that are not true matches).
4. Feature Extraction: Computes the 23-dimensional similarity feature matrix using
   `src.feature_generator.compute_pair_features`.
5. Model Training: Fits a fast histogram gradient boosted classifier.
6. F0.5-Driven Threshold Optimization: Evaluates candidate probabilities across
   thresholds to optimize the official competition metric:
       Macro F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
7. Permutation Feature Importance & Model Serialization: Saves model artifact
   and configuration for inference.

Usage:
    from src.model_trainer import EntityMatcher, train_and_evaluate
    # Or run directly to train and evaluate:
    python src/model_trainer.py
"""

import os
import sys
import time
import random
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

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
from src.blocking import InvertedIndexBlocking
from src.feature_generator import compute_pair_features, FEATURE_COLUMNS


def resolve_base_dir():
    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, "dataset", "train")):
        return cwd
    if os.path.isdir(os.path.join(cwd, "student_resource", "dataset", "train")):
        return os.path.join(cwd, "student_resource")
    return cwd


def calculate_entity_f05(true_matches: set, pred_matches: set) -> tuple:
    """
    Computes per-entity Precision, Recall, and F0.5 according to the official formula:
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singletons (0 true matches) score:
        - 1.0 if predicted empty (correct singleton)
        - 0.0 if any false match is predicted
    """
    n_true = len(true_matches)
    n_pred = len(pred_matches)
    n_inter = len(pred_matches.intersection(true_matches))

    if n_true == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0  # Correct singleton
    elif n_true == 0 and n_pred > 0:
        return 0.0, 0.0, 1.0  # False merge on singleton
    elif n_true > 0 and n_pred == 0:
        return 0.0, 1.0, 0.0  # Missed all matches
    else:
        prec = n_inter / n_pred
        rec = n_inter / n_true
        denom = (0.25 * prec) + rec
        f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0
        return f05, prec, rec


class EntityMatcher:
    """Wrapper around trained gradient boosting matcher with optimal threshold."""

    def __init__(self, optimal_threshold: float = 0.5):
        self.model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            min_samples_leaf=20,
            random_state=42,
        )
        self.optimal_threshold = optimal_threshold
        self.feature_names = list(FEATURE_COLUMNS)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        t = self.optimal_threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return (probs >= t).astype(int)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "optimal_threshold": self.optimal_threshold,
            "feature_names": self.feature_names,
        }, filepath)
        print(f"Model successfully saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str):
        data = joblib.load(filepath)
        matcher = cls(optimal_threshold=data["optimal_threshold"])
        matcher.model = data["model"]
        matcher.feature_names = data["feature_names"]
        return matcher


def load_training_dataset(base_dir: str, num_s1: int = 1000, num_distractors: int = 15000):
    """
    Dynamically loads Source 1 records, ground truth links, and candidate pool.
    No hardcoded business data or IDs.
    """
    s1_path = os.path.join(base_dir, "dataset/train/train_source1.tsv")
    gt_path = os.path.join(base_dir, "dataset/train/train_ground_truth.tsv")
    s2_path = os.path.join(base_dir, "dataset/train/train_source2.tsv")
    s3_path = os.path.join(base_dir, "dataset/train/train_source3.tsv")

    print(f"Loading {num_s1:,} Source 1 records from {s1_path}...")
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
                    "clean_name": normalize_business_name(parts[1]),
                    "core_name": extract_core_name(parts[1]),
                    "clean_addr": normalize_address(parts[2]),
                    "addr_tokens": get_address_tokens(parts[2]),
                }

    print("Loading corresponding ground-truth matching links...")
    gt_matches = {}
    all_matched_ids = set()
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in s1_records:
                ms = [x.strip() for x in parts[1].split(",") if x.strip()] if len(parts) > 1 and parts[1].strip() else []
                gt_matches[parts[0]] = set(ms)
                all_matched_ids.update(ms)

    print(f"Loading candidate pool (target matches + {num_distractors:,} distractors)...")
    s2_s3_pool = {}
    distractors_per_file = num_distractors // 2

    for path in [s2_path, s3_path]:
        dist_count = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4:
                    eid = parts[0]
                    is_target = eid in all_matched_ids
                    if is_target or dist_count < distractors_per_file:
                        s2_s3_pool[eid] = {
                            "id": eid,
                            "name": parts[1],
                            "raw_addr": parts[2],
                            "country": parts[3],
                            "clean_name": normalize_business_name(parts[1]),
                            "core_name": extract_core_name(parts[1]),
                            "clean_addr": normalize_address(parts[2]),
                            "addr_tokens": get_address_tokens(parts[2]),
                        }
                        if not is_target:
                            dist_count += 1

    return s1_records, gt_matches, s2_s3_pool


def train_and_evaluate():
    print("=" * 80)
    print("       AMAZON ML CHALLENGE 2026 - STEP 6: MODEL TRAINING & VALIDATION")
    print("=" * 80)

    base_dir = resolve_base_dir()
    t_start = time.time()

    # 1. Load data
    s1_records, gt_matches, s2_s3_pool = load_training_dataset(
        base_dir, num_s1=1000, num_distractors=15000
    )

    # 2. Build blocking index
    indexer = InvertedIndexBlocking(max_block_size=200)
    indexer.build_from_dict(s2_s3_pool)

    # 3. Entity-level train/validation split (80% train, 20% validation)
    random.seed(42)
    s1_keys = list(s1_records.keys())
    random.shuffle(s1_keys)
    split_idx = int(len(s1_keys) * 0.8)
    train_s1_keys = set(s1_keys[:split_idx])
    val_s1_keys = set(s1_keys[split_idx:])

    print(f"\nEntity-Level Split: {len(train_s1_keys)} Train S1 Entities | {len(val_s1_keys)} Validation S1 Entities")

    # 4. Generate Training Pairs & Hard Negatives
    print("Extracting feature vectors for training pairs (positives + hard negatives)...")
    X_train = []
    y_train = []

    for s1_id in train_s1_keys:
        s1_rec = s1_records[s1_id]
        true_set = gt_matches.get(s1_id, set())
        cand_set = indexer.get_candidates(s1_rec)

        # True Positives
        for cid in true_set:
            if cid in s2_s3_pool:
                f_vec = compute_pair_features(s1_rec, s2_s3_pool[cid])
                X_train.append([f_vec[col] for col in FEATURE_COLUMNS])
                y_train.append(1)

        # Hard Negatives (sampled from blocking distractors)
        neg_cands = [cid for cid in cand_set if cid not in true_set]
        sampled_negs = random.sample(neg_cands, min(len(neg_cands), 8)) if neg_cands else []
        for cid in sampled_negs:
            f_vec = compute_pair_features(s1_rec, s2_s3_pool[cid])
            X_train.append([f_vec[col] for col in FEATURE_COLUMNS])
            y_train.append(0)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"Training Matrix: {X_train.shape[0]:,} pairs x {X_train.shape[1]} features")
    print(f"  Positives: {np.sum(y_train == 1):,} | Negatives: {np.sum(y_train == 0):,}")

    # 5. Train Model
    print("\nTraining HistGradientBoostingClassifier...")
    t_train = time.time()
    matcher = EntityMatcher()
    matcher.fit(X_train, y_train)
    print(f"Training completed in {time.time() - t_train:.2f} seconds.")

    # 6. Extract Validation Candidate Sets
    print("\nGenerating candidate features for validation entities...")
    val_data = []
    for s1_id in val_s1_keys:
        s1_rec = s1_records[s1_id]
        cand_list = list(indexer.get_candidates(s1_rec))
        if not cand_list:
            val_data.append((s1_id, [], np.empty((0, len(FEATURE_COLUMNS)))))
            continue
        X_ent = []
        for cid in cand_list:
            f_vec = compute_pair_features(s1_rec, s2_s3_pool[cid])
            X_ent.append([f_vec[col] for col in FEATURE_COLUMNS])
        val_data.append((s1_id, cand_list, np.array(X_ent, dtype=np.float32)))

    # 7. Optimize Threshold for Macro F0.5
    print("Optimizing classification threshold for Macro F0.5...")
    best_t = 0.50
    best_f05 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    threshold_results = []

    for t_val in np.arange(0.20, 0.92, 0.05):
        f05_scores = []
        prec_scores = []
        rec_scores = []

        for s1_id, cand_list, X_ent in val_data:
            true_set = gt_matches.get(s1_id, set())
            if len(cand_list) == 0:
                pred_set = set()
            else:
                probs = matcher.predict_proba(X_ent)
                pred_set = {cand_list[i] for i, p in enumerate(probs) if p >= t_val}

            f05_i, prec_i, rec_i = calculate_entity_f05(true_set, pred_set)
            f05_scores.append(f05_i)
            prec_scores.append(prec_i)
            rec_scores.append(rec_i)

        macro_f05 = float(np.mean(f05_scores))
        macro_prec = float(np.mean(prec_scores))
        macro_rec = float(np.mean(rec_scores))

        threshold_results.append((t_val, macro_f05, macro_prec, macro_rec))
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_t = float(t_val)
            best_prec = macro_prec
            best_rec = macro_rec

    matcher.optimal_threshold = best_t

    # 8. Report Validation Metrics
    print("\n" + "=" * 80)
    print("                      VALIDATION RESULTS (Macro F0.5)")
    print("=" * 80)
    print(f"  Validation Entities Evaluated  : {len(val_s1_keys):,}")
    print(f"  Optimal Probability Threshold  : {best_t:.2f}")
    print(f"  --> VALIDATION MACRO F0.5      : {best_f05:.4f}  ({best_f05*100:.2f}%)")
    print(f"  --> VALIDATION MACRO PRECISION : {best_prec:.4f}  ({best_prec*100:.2f}%)")
    print(f"  --> VALIDATION MACRO RECALL    : {best_rec:.4f}  ({best_rec*100:.2f}%)")
    print("-" * 80)

    print("Threshold Sweep Analysis:")
    print(f"  {'Threshold':<11} | {'Macro F0.5':<12} | {'Precision':<12} | {'Recall':<12}")
    print("  " + "-" * 55)
    for t_val, f05, p, r in threshold_results:
        star = " <-- OPTIMAL" if abs(t_val - best_t) < 1e-4 else ""
        print(f"  {t_val:<11.2f} | {f05:<12.4f} | {p:<12.4f} | {r:<12.4f}{star}")

    # 9. Top Permutation Feature Importances
    print("\n" + "-" * 80)
    print("Top Feature Importances (Permutation Importance on Training Subset):")
    print("-" * 80)
    perm_sample_idx = np.random.choice(len(X_train), min(1000, len(X_train)), replace=False)
    perm_res = permutation_importance(
        matcher.model,
        X_train[perm_sample_idx],
        y_train[perm_sample_idx],
        n_repeats=5,
        random_state=42,
    )
    sorted_importances = sorted(
        zip(FEATURE_COLUMNS, perm_res.importances_mean),
        key=lambda x: x[1],
        reverse=True,
    )
    for rank, (feat, score) in enumerate(sorted_importances[:8], 1):
        bar = "#" * max(1, int(score * 100))
        print(f"  {rank}. {feat:<24}: {score:>7.4f}  {bar}")

    # 10. Save Model
    model_save_path = os.path.join(base_dir, "models/matcher_model.joblib")
    matcher.save(model_save_path)

    total_time = time.time() - t_start
    print(f"\nStep 6 end-to-end pipeline completed in {total_time:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    train_and_evaluate()
