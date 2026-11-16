"""Data Collection Utility for SignBridge AI.
Captures normalized 73-dimensional landmark feature vectors from live video or
synthetic sample generators and appends them to CSV for ML model training.
"""

import argparse
import csv
from pathlib import Path
import sys
import time
from typing import List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from ai.sign_recognition.config import (
    DATASET_PATH,
    DEFAULT_CAMERA_INDEX,
    SUPPORTED_GESTURES,
    TOTAL_FEATURE_DIM,
)
from ai.sign_recognition.hand_detector import HandDetector
from ai.sign_recognition.landmark_extractor import LandmarkExtractor


def initialize_dataset_file(filepath: Path) -> None:
    """Creates dataset CSV file with header if it does not already exist."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists() or filepath.stat().st_size == 0:
        header = ["label"] + [f"feat_{i}" for i in range(TOTAL_FEATURE_DIM)]
        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def save_samples(filepath: Path, label: str, samples: List[np.ndarray]) -> int:
    """Appends collected feature vectors to the CSV dataset."""
    if not samples:
        return 0
    initialize_dataset_file(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)
        for sample in samples:
            row = [label] + [f"{val:.6f}" for val in sample]
            writer.writerow(row)
    return len(samples)


def generate_synthetic_samples(label: str, count: int = 50) -> List[np.ndarray]:
    """Generates canonical geometric samples with noise for training and testing.
    
    Ensures that offline, headless, or camera-less environments can bootstrap
    a working ML model for the supported vocabulary.
    """
    extractor = LandmarkExtractor()
    samples: List[np.ndarray] = []

    # Base canonical landmark configurations (21 landmarks)
    # Default open hand facing camera
    base_pts = np.zeros((21, 3), dtype=np.float32)
    # Wrist
    base_pts[0] = [0.5, 0.8, 0.0]

    # Fingers: thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
    for finger_idx in range(5):
        mcp_x = 0.35 + finger_idx * 0.075
        for joint in range(4):
            idx = 1 + finger_idx * 4 + joint
            base_pts[idx] = [mcp_x, 0.65 - joint * 0.12, 0.0]

    for _ in range(count):
        pts = base_pts.copy()
        noise = np.random.normal(0, 0.015, pts.shape).astype(np.float32)

        if label in ("YES", "HELP"):
            # Fold index, middle, ring, pinky
            for f in range(1, 5):
                for j in range(1, 4):
                    pts[1 + f * 4 + j, 1] += 0.25  # Curl downward
            # Extend thumb upward
            pts[4, 1] -= 0.10
        elif label == "PEACE":
            # Extend index (5-8) and middle (9-12), fold ring (13-16) and pinky (17-20)
            for f in [0, 3, 4]:
                for j in range(1, 4):
                    pts[1 + f * 4 + j, 1] += 0.25
            pts[8, 0] -= 0.03  # Spread index
            pts[12, 0] += 0.03  # Spread middle
        elif label == "OK":
            # Pinch thumb tip and index tip together
            pts[4] = [0.45, 0.45, 0.0]
            pts[8] = [0.46, 0.46, 0.0]
        elif label == "I_LOVE_YOU":
            # Extend thumb, index, pinky. Fold middle, ring.
            for f in [2, 3]:
                for j in range(1, 4):
                    pts[1 + f * 4 + j, 1] += 0.25
        elif label == "NO":
            # Pinch thumb, index, middle together, fold ring, pinky
            pts[4] = [0.45, 0.45, 0.0]
            pts[8] = [0.46, 0.45, 0.0]
            pts[12] = [0.47, 0.46, 0.0]
            for f in [3, 4]:
                for j in range(1, 4):
                    pts[1 + f * 4 + j, 1] += 0.25

        perturbed_pts = pts + noise
        features = extractor.extract_features(perturbed_pts)
        samples.append(features)

    return samples


def collect_live_samples(label: str, target_count: int = 50, camera_idx: int = DEFAULT_CAMERA_INDEX) -> int:
    """Collects samples from live webcam feed with landmark visualization."""
    detector = HandDetector()
    extractor = LandmarkExtractor()

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[-] Camera {camera_idx} could not be opened. Using synthetic sample generator.")
        samples = generate_synthetic_samples(label, target_count)
        return save_samples(DATASET_PATH, label, samples)

    collected: List[np.ndarray] = []
    print(f"\n[+] Collecting {target_count} samples for gesture: '{label}'")
    print("[+] Press 'c' to capture a sample when your hand is positioned correctly.")
    print("[+] Press 'q' to finish and save.\n")

    while len(collected) < target_count:
        ret, frame = cap.read()
        if not ret:
            break

        res = detector.detect(frame)
        annotated = detector.draw_landmarks(frame, res)

        status_msg = f"Gesture: {label} | Collected: {len(collected)}/{target_count}"
        cv2.putText(annotated, status_msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, "Press 'c' to capture, 'q' to quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("SignBridge AI - Data Collector", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            if res.detected and len(res.landmarks) == 21:
                features = extractor.extract_features(np.array(res.landmarks))
                collected.append(features)
                print(f"Captured sample {len(collected)}/{target_count}")
            else:
                print("No hand detected! Position hand clearly before capturing.")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return save_samples(DATASET_PATH, label, collected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignBridge AI Data Collector")
    parser.add_argument("--label", type=str, default="HELLO", choices=SUPPORTED_GESTURES, help="Gesture label to collect")
    parser.add_argument("--count", type=int, default=50, help="Number of samples to capture")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic samples (for headless / no camera environments)")
    args = parser.parse_args()

    if args.synthetic:
        print(f"[+] Generating {args.count} synthetic samples for '{args.label}'...")
        samples = generate_synthetic_samples(args.label, args.count)
        saved = save_samples(DATASET_PATH, args.label, samples)
        print(f"[✓] Successfully saved {saved} samples to {DATASET_PATH}")
    else:
        saved = collect_live_samples(args.label, args.count)
        print(f"[✓] Finished. Saved {saved} samples to {DATASET_PATH}")
