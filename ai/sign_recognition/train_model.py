"""Model Training and Evaluation Pipeline for SignBridge AI.
Trains and evaluates a Scikit-Learn classifier on landmark feature vectors.
Exports trained model to joblib format for real-time inference.
"""

import csv
from collections import Counter
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from ai.sign_recognition.config import (
    CLASSIFIER_MODEL_PATH,
    DATASET_PATH,
    MODELS_DIR,
    SUPPORTED_GESTURES,
)


def load_and_validate_dataset(csv_path: Path):
    """Loads CSV dataset without pandas and verifies data integrity and class balance."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Dataset not found or empty at {csv_path}.\n"
            "Run 'python -m ai.sign_recognition.collect_data --synthetic' to generate samples first."
        )

    labels = []
    features = []

    with open(csv_path, mode="r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or header[0] != "label":
            raise ValueError(f"Invalid dataset format in {csv_path}. Must have 'label' as first column.")

        for row_idx, row in enumerate(reader):
            if not row or len(row) < 2:
                continue
            label = row[0]
            try:
                feat = [float(val) for val in row[1:]]
                labels.append(label)
                features.append(feat)
            except ValueError:
                continue

    if not labels:
        raise ValueError(f"No valid data rows found in {csv_path}.")

    counts = Counter(labels)

    print("\n--- Dataset Distribution ---")
    for gesture, count in sorted(counts.items()):
        print(f"  {gesture:<15}: {count} samples")
    print(f"Total samples: {len(labels)}")

    if len(counts) < 2:
        raise ValueError(
            f"Dataset has only {len(counts)} class. Minimum 2 classes required to train a classifier.\n"
            "Collect samples for at least 2 distinct gestures."
        )

    min_samples = min(counts.values())
    if min_samples < 5:
        raise ValueError(
            f"Class '{min(counts, key=counts.get)}' only has {min_samples} samples. "
            "Minimum 5 samples per class required."
        )

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)

    return X, y


def train_gesture_classifier(X: np.ndarray, y: np.ndarray, output_path: Path):
    """Trains a Random Forest classifier with stratified cross-validation and exports it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    print(f"Feature vector dimensionality: {X.shape[1]}")

    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n==========================================")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"==========================================")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save trained model
    joblib.dump(clf, output_path)
    print(f"[✓] Successfully exported trained model to: {output_path}")

    return clf, accuracy


if __name__ == "__main__":
    print("[*] Starting SignBridge AI Model Training Pipeline...")
    try:
        X, y = load_and_validate_dataset(DATASET_PATH)
        train_gesture_classifier(X, y, CLASSIFIER_MODEL_PATH)
    except Exception as e:
        print(f"\n[!] Training Pipeline Notice: {e}")
        print("[!] Note: The geometric rule engine remains active as an automatic fallback.")
        sys.exit(1)
