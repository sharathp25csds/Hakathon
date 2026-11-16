"""ISL Alphabet Model Training and Evaluation Pipeline for SignBridge AI v2.
Enforces strict dataset validation, prevents synthetic-only and mixed synthetic/real
datasets from creating production models, prioritizes signer-independent evaluation,
computes dataset SHA256 provenance, per-class metrics, confusion matrix,
and exports comprehensive metadata JSON sidecars.
Existing production models are strictly protected if validation fails.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ai.sign_recognition_v2.config import (
    DATASET_PATH,
    FEATURE_VERSION,
    ISL_ALPHABET,
    ISL_ALPHABET_METADATA,
    ISL_CLASSIFIER_MODEL_PATH,
    ISL_DATASET_PATH,
    TOTAL_FEATURE_DIM,
)


def compute_file_sha256(path: Path) -> str:
    """Computes the SHA256 checksum of a file for immutable provenance tracking."""
    if not path.exists() or path.stat().st_size == 0:
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_and_validate_dataset(
    csv_path: Path,
    allow_synthetic_only: bool = False,
    enforce_isl_vocab: bool = True,
    require_signer_independent: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str], Dict[str, int]]:
    """Loads and strictly validates the dataset.
    
    Validation requirements enforced:
    1. Dataset file must exist and not be empty.
    2. Enforces dataset separation: rejects training ISL models from legacy gestures.csv.
    3. Header must be well-formed with feature columns.
    4. Feature dimensions must strictly match TOTAL_FEATURE_DIM (162).
    5. Rejects any non-finite (NaN, Inf) values.
    6. Detects duplicate samples.
    7. Rejects unsupported/invalid ISL alphabet labels.
    8. Separates synthetic and real data.
    9. Production training safety:
       - Rejects synthetic-only datasets.
       - Rejects mixed synthetic + real datasets.
       - Only 100% real human data is allowed for production models,
         unless allow_synthetic_only is explicitly True (benchmark mode).
    10. Rejects datasets lacking sufficient independent signers when required.
    
    Returns:
        Tuple: (X, y, sources, session_ids, signer_ids, label_counts)
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Dataset not found or empty at {csv_path}.\n"
            "Run 'python -m ai.sign_recognition_v2.collect_data' to record real samples."
        )

    # Dataset separation safeguard: Reject training ISL from legacy gesture dataset
    if enforce_isl_vocab and DATASET_PATH.exists():
        try:
            if csv_path.resolve() == DATASET_PATH.resolve():
                raise ValueError(
                    f"DATASET SEPARATION SAFEGUARD: Refusing to train ISL Alphabet model from "
                    f"legacy gesture dataset '{csv_path}'. Use ISL_DATASET_PATH ('{ISL_DATASET_PATH}') instead."
                )
        except Exception as e:
            if "DATASET SEPARATION SAFEGUARD" in str(e):
                raise

    labels = []
    sources = []
    session_ids = []
    signer_ids = []
    features = []
    seen_hashes = set()
    duplicate_count = 0

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Empty header in {csv_path}")

        # Map column positions from header
        col_map = {name.strip(): idx for idx, name in enumerate(header)}
        label_col = col_map.get("label", 0)
        source_col = col_map.get("source_type", None)
        session_col = col_map.get("session_id", None)
        signer_col = col_map.get("signer_id", None)
        version_col = col_map.get("feature_version", None)

        # Locate feature start column
        feat_indices = [idx for idx, name in enumerate(header) if name.startswith("feat_")]
        if not feat_indices:
            feat_start = len(header) - TOTAL_FEATURE_DIM
            if feat_start < 1:
                feat_start = 1
            feat_indices = list(range(feat_start, len(header)))

        expected_dim = len(feat_indices)
        if expected_dim != TOTAL_FEATURE_DIM:
            raise ValueError(
                f"Dataset header indicates {expected_dim} features, but expected "
                f"TOTAL_FEATURE_DIM = {TOTAL_FEATURE_DIM}."
            )

        for row_idx, row in enumerate(reader, start=2):
            if not row or not any(cell.strip() for cell in row):
                continue

            row_feat_vals = [row[i] for i in feat_indices if i < len(row)]
            if len(row_feat_vals) != TOTAL_FEATURE_DIM:
                raise ValueError(
                    f"Row {row_idx}: Invalid feature dimension {len(row_feat_vals)}. "
                    f"Expected exactly {TOTAL_FEATURE_DIM}."
                )

            lbl = row[label_col].strip().upper()
            src = row[source_col].strip() if source_col is not None and source_col < len(row) else "unknown"
            sess = row[session_col].strip() if session_col is not None and session_col < len(row) else f"sess_{row_idx}"
            sgnr = row[signer_col].strip() if signer_col is not None and signer_col < len(row) else f"signer_{row_idx}"

            # Check feature version if present in row
            if version_col is not None and version_col < len(row):
                row_ver = row[version_col].strip()
                if row_ver and row_ver != FEATURE_VERSION:
                    raise ValueError(
                        f"Row {row_idx}: Feature version mismatch ('{row_ver}' vs current '{FEATURE_VERSION}')."
                    )

            # Validate label against ISL alphabet specification
            if enforce_isl_vocab:
                if lbl not in ISL_ALPHABET:
                    raise ValueError(
                        f"Row {row_idx}: Label '{lbl}' is not a recognized letter in the ISL Alphabet (A-Z)."
                    )
                meta = ISL_ALPHABET_METADATA.get(lbl, {})
                if meta.get("status") == "unsupported":
                    raise ValueError(
                        f"Row {row_idx}: Letter '{lbl}' is marked as 'unsupported' in ISLRTC configuration "
                        "due to high camera occlusion or anatomical tracking limitations. Cannot train on unsupported letters."
                    )

            # Parse float features & check for non-finite values (NaN / Inf)
            try:
                numeric_feats = [float(v) for v in row_feat_vals]
            except ValueError:
                raise ValueError(f"Row {row_idx}: Contains non-numeric feature values.")

            arr = np.array(numeric_feats, dtype=np.float32)
            if np.isnan(arr).any() or np.isinf(arr).any():
                raise ValueError(f"Row {row_idx}: Contains non-finite values (NaN or Inf).")

            # Duplicate sample detection
            feat_tuple = tuple(np.round(arr, 4))
            if feat_tuple in seen_hashes:
                duplicate_count += 1
            else:
                seen_hashes.add(feat_tuple)

            labels.append(lbl)
            sources.append(src)
            session_ids.append(sess)
            signer_ids.append(sgnr)
            features.append(arr)

    if not labels:
        raise ValueError(f"No valid records found in {csv_path}.")

    label_counts = Counter(labels)
    source_counts = Counter(sources)

    print("\n--- Dataset Distribution ---")
    for lbl, count in sorted(label_counts.items()):
        meta = ISL_ALPHABET_METADATA.get(lbl, {})
        status = meta.get("status", "unknown")
        print(f"  Letter '{lbl}' [{status:<16}]: {count} samples")
    print(f"Total samples: {len(labels)}")
    print(f"Duplicate samples detected: {duplicate_count}")

    print("\n--- Dataset Provenance ---")
    for src, count in sorted(source_counts.items()):
        pct = (count / len(labels)) * 100
        print(f"  {src:<22}: {count} samples ({pct:.1f}%)")

    # Strict Production Training Guard: Rejection of Synthetic-Only and Mixed Datasets
    real_sample_count = source_counts.get("live_camera", 0) + source_counts.get("real_human", 0)
    synthetic_sample_count = source_counts.get("synthetic_benchmark", 0) + source_counts.get("synthetic", 0)

    if not allow_synthetic_only:
        # 1. Pure synthetic rejection
        if real_sample_count == 0:
            raise ValueError(
                "SYNTHETIC-ONLY REJECTION: The dataset contains exclusively synthetic benchmark samples "
                f"({synthetic_sample_count} samples). Synthetic samples must NEVER be used to train production models.\n"
                "Please collect real human webcam samples using 'collect_data.py' or explicitly pass '--benchmark-mode' "
                "for internal testing/CI evaluation."
            )
        # 2. Mixed dataset rejection
        if synthetic_sample_count > 0 and real_sample_count > 0:
            raise ValueError(
                "MIXED-DATASET REJECTION: The dataset contains a mixture of synthetic benchmark data "
                f"({synthetic_sample_count} samples) and real human data ({real_sample_count} samples). "
                "Production models must be trained exclusively on real human data to guarantee integrity. "
                "Purge synthetic data from the dataset before training production models."
            )

    # Independent Signer Requirement Validation
    unique_signers = set(s for s in signer_ids if s and not s.startswith("signer_"))
    if require_signer_independent and len(unique_signers) < 2:
        raise ValueError(
            f"INSUFFICIENT SIGNERS: Dataset contains {len(unique_signers)} independent signer(s). "
            "A minimum of 2 distinct human signers is required for independent-signer evaluation."
        )

    if len(label_counts) < 2:
        raise ValueError(
            f"Dataset contains only {len(label_counts)} distinct class ({list(label_counts.keys())}). "
            "Minimum 2 distinct classes required for classification."
        )

    min_samples = min(label_counts.values())
    if min_samples < 3:
        rare_class = min(label_counts, key=label_counts.get)
        raise ValueError(f"Class '{rare_class}' has fewer than 3 samples ({min_samples}). Cannot reliably train or evaluate.")

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)

    return X, y, sources, session_ids, signer_ids, dict(label_counts)


def train_and_evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    sources: List[str],
    session_ids: List[str],
    signer_ids: List[str],
    output_path: Path,
    dataset_path: Optional[Path] = None,
    is_benchmark_mode: bool = False,
) -> Tuple[RandomForestClassifier, float, Dict]:
    """Trains a Random Forest classifier, prioritizes signer-independent evaluation,
    computes dataset SHA256 hash, per-class metrics, confusion matrix, and exports metadata JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute dataset SHA256 hash if dataset_path is provided
    dataset_sha256 = ""
    if dataset_path is not None:
        dataset_sha256 = compute_file_sha256(dataset_path)

    source_counts = Counter(sources)
    unique_signers = len(set(signer_ids))
    unique_sessions = len(set(session_ids))
    signer_independent_available = False
    split_strategy = "stratified_random"

    # 1. Prefer signer-grouped evaluation if unique_signers >= 2
    if unique_signers >= 2:
        try:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
            train_idx, test_idx = next(gss.split(X, y, groups=signer_ids))
            if len(test_idx) > 0 and len(train_idx) > 0 and len(set(y[test_idx])) >= 2:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                signer_independent_available = True
                split_strategy = "signer_independent_split"
                train_signers = sorted(list(set(np.array(signer_ids)[train_idx])))
                test_signers = sorted(list(set(np.array(signer_ids)[test_idx])))
                print(f"\n[Splitting] SIGNER-INDEPENDENT EVALUATION ACTIVE ({unique_signers} signers).")
                print(f"  Train signers: {train_signers} | Test signers: {test_signers}")
        except Exception as e:
            print(f"\n[Splitting] Signer GroupShuffleSplit attempt failed: {e}")
            signer_independent_available = False

    # 2. Fallback when independent-signer evaluation is unavailable
    if not signer_independent_available:
        print("\n[WARNING] INDEPENDENT-SIGNER EVALUATION UNAVAILABLE:")
        print(f"  Dataset contains {unique_signers} unique signer(s). Minimum 2 independent signers required.")
        print("  Reported evaluation does NOT demonstrate generalization to unseen signers.")

        if unique_sessions >= 2:
            try:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
                train_idx, test_idx = next(gss.split(X, y, groups=session_ids))
                if len(test_idx) > 0 and len(train_idx) > 0 and len(set(y[test_idx])) >= 2:
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    split_strategy = "session_independent_split (insufficient signers)"
                    print(f"[Splitting] Fallback: Using GroupShuffleSplit on {unique_sessions} sessions.")
            except Exception:
                pass

        if split_strategy == "stratified_random":
            split_strategy = "stratified_random_split (single signer/session)"
            print("[Splitting] Fallback: Using stratified random frame split.")
            print("[NOTICE] Frame-level splitting does NOT measure cross-signer or cross-session generalization.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )

    print(f"Training set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")
    print(f"Feature vector dimensionality: {X.shape[1]}")

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    classes = sorted(list(set(y)))

    # Compute metrics
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    accuracy = float(report_dict.get("accuracy", 0.0))

    print("\n==========================================")
    if is_benchmark_mode:
        print(f"BENCHMARK TEST ACCURACY: {accuracy * 100:.2f}%")
        print("[CRITICAL NOTICE] This accuracy was computed on SYNTHETIC BENCHMARK data.")
        print("[CRITICAL NOTICE] Real human demonstration accuracy will differ.")
    elif not signer_independent_available:
        print(f"WITHIN-SIGNER TEST ACCURACY: {accuracy * 100:.2f}%")
        print("[CRITICAL NOTICE] Evaluated within the same signer(s). Independent-signer evaluation is UNAVAILABLE.")
        print("[CRITICAL NOTICE] This metric does NOT demonstrate real-world generalization to new signers.")
    else:
        print(f"SIGNER-INDEPENDENT TEST ACCURACY: {accuracy * 100:.2f}%")
        print("[NOTE] Evaluated on unseen signers.")
    print("==========================================")

    print("\nPer-Letter Classification Report (Precision, Recall, F1):")
    print(report_text)

    print("\nConfusion Matrix (Rows: Ground Truth, Cols: Predicted):")
    header_str = "     " + " ".join(f"{c:>4}" for c in classes)
    print(header_str)
    for row_lbl, row in zip(classes, cm):
        row_str = f"{row_lbl:>3}: " + " ".join(f"{val:>4}" for val in row)
        print(row_str)

    # Save trained model
    joblib.dump(clf, output_path)
    print(f"\n[✓] Successfully exported model to: {output_path}")

    # Save comprehensive model metadata sidecar
    meta_path = output_path.with_suffix(".json")
    meta_info = {
        "model_type": type(clf).__name__,
        "training_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_version": FEATURE_VERSION,
        "feature_dimension": int(X.shape[1]),
        "classes": classes,
        "trained_class_labels": classes,
        "dataset_path": str(dataset_path) if dataset_path else "",
        "dataset_sha256": dataset_sha256,
        "source_type_counts": dict(source_counts),
        "sample_count": int(X.shape[0]),
        "train_sample_count": int(X_train.shape[0]),
        "test_sample_count": int(X_test.shape[0]),
        "unique_signer_count": unique_signers,
        "unique_session_count": unique_sessions,
        "evaluation_split_strategy": split_strategy,
        "signer_independent_evaluation_available": signer_independent_available,
        "is_benchmark_mode": is_benchmark_mode,
        "test_accuracy": accuracy,
        "evaluation_metrics": report_dict,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)
    print(f"[✓] Saved model metadata to: {meta_path}")

    return clf, accuracy, meta_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignBridge AI v2 ISL Alphabet Model Trainer")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(ISL_DATASET_PATH),
        help=f"Path to dataset CSV (default: {ISL_DATASET_PATH})",
    )
    parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        help="Permit training on synthetic benchmark data for testing/CI purposes",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ISL_CLASSIFIER_MODEL_PATH),
        help=f"Path to export trained model (default: {ISL_CLASSIFIER_MODEL_PATH})",
    )
    parser.add_argument(
        "--allow-non-isl",
        action="store_true",
        help="Permit non-ISL alphabet labels (e.g. for legacy gesture testing)",
    )
    parser.add_argument(
        "--require-independent-signers",
        action="store_true",
        help="Strictly fail if dataset does not contain >= 2 independent signers",
    )
    args = parser.parse_args()

    print("[*] Starting SignBridge AI v2 ISL Alphabet Model Training Pipeline...")
    in_file = Path(args.dataset)
    out_file = Path(args.output)
    try:
        X, y, sources, session_ids, signer_ids, counts = load_and_validate_dataset(
            in_file,
            allow_synthetic_only=args.benchmark_mode,
            enforce_isl_vocab=not args.allow_non_isl,
            require_signer_independent=args.require_independent_signers,
        )
        train_and_evaluate_model(
            X,
            y,
            sources,
            session_ids,
            signer_ids,
            out_file,
            dataset_path=in_file,
            is_benchmark_mode=args.benchmark_mode,
        )
    except Exception as e:
        print(f"\n[!] Training Pipeline Aborted: {e}")
        print("[!] Note: Any existing production model was safely preserved and not overwritten.")
        sys.exit(1)
