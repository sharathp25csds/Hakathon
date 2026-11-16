"""ISL Alphabet Multi-Hand Landmark Dataset Collection Utility for SignBridge AI v2.
Records 162-dimensional multi-hand feature vectors to ISL_DATASET_PATH (isl_alphabet.csv).
Preserves existing gestures.csv untouched.
Includes signer/session IDs, feature version metadata, and strict source provenance.
NEVER silently generates synthetic samples when camera access fails.
"""

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
import time
from typing import List, Optional
import uuid
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.sign_recognition_v2.config import (
    DEFAULT_CAMERA_INDEX,
    FEATURE_VERSION,
    ISL_ALPHABET,
    ISL_DATASET_PATH,
    TOTAL_FEATURE_DIM,
)
from ai.sign_recognition_v2.hand_detector import HandDetectionResultV2, HandDetectorV2
from ai.sign_recognition_v2.landmark_extractor import LandmarkExtractorV2


def initialize_dataset_file(filepath: Path) -> None:
    """Initializes CSV header if dataset file does not exist."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists() or filepath.stat().st_size == 0:
        header = [
            "label",
            "source_type",
            "feature_version",
            "signer_id",
            "session_id",
            "timestamp",
        ] + [f"feat_{i}" for i in range(TOTAL_FEATURE_DIM)]
        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def save_samples(
    filepath: Path,
    label: str,
    samples: List[np.ndarray],
    source_type: str = "live_camera",
    signer_id: str = "signer_1",
    session_id: Optional[str] = None,
) -> int:
    """Appends feature vectors to the ISL CSV dataset with full provenance metadata."""
    if not samples:
        return 0

    initialize_dataset_file(filepath)
    sess = session_id or str(uuid.uuid4())[:8]
    now_ts = datetime.utcnow().isoformat()

    valid_samples = []
    for sample in samples:
        if len(sample) == TOTAL_FEATURE_DIM and not np.isnan(sample).any():
            valid_samples.append(sample)

    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)
        for s in valid_samples:
            row = [
                label,
                source_type,
                FEATURE_VERSION,
                signer_id,
                sess,
                now_ts,
            ] + [f"{val:.6f}" for val in s]
            writer.writerow(row)

    return len(valid_samples)


def generate_synthetic_benchmark_samples(label: str, count: int = 50) -> List[np.ndarray]:
    """Generates synthetic benchmark feature vectors for CI / unit testing only.
    
    WARNING: These samples are strictly marked as 'synthetic_benchmark' and
    CANNOT be used to train production models or claim real-world human accuracy.
    """
    extractor = LandmarkExtractorV2()
    samples: List[np.ndarray] = []

    base_pts = np.zeros((21, 3), dtype=np.float32)
    base_pts[0] = [0.5, 0.8, 0.0]

    for finger_idx in range(5):
        mcp_x = 0.35 + finger_idx * 0.075
        for joint in range(4):
            idx = 1 + finger_idx * 4 + joint
            base_pts[idx] = [mcp_x, 0.65 - joint * 0.12, 0.0]

    for _ in range(count):
        pts = base_pts.copy()
        noise = np.random.normal(0, 0.012, pts.shape).astype(np.float32)

        if label in ("A", "S"):
            for f in range(1, 5):
                for j in range(1, 4):
                    pts[1 + f * 4 + j, 1] += 0.25
        elif label == "L":
            for f in [2, 3, 4]:
                for j in range(1, 4):
                    pts[1 + f * 4 + j, 1] += 0.25
            pts[4, 0] -= 0.15
        elif label == "V":
            for f in [0, 3, 4]:
                for j in range(1, 4):
                    pts[1 + f * 4 + j, 1] += 0.25
            pts[8, 0] -= 0.04
            pts[12, 0] += 0.04
        elif label == "Y":
            for f in [1, 2, 3]:
                for j in range(1, 4):
                    pts[1 + f * 4 + j, 1] += 0.25
            pts[4, 0] -= 0.12
            pts[20, 0] += 0.12

        perturbed = pts + noise
        features = extractor.extract_features(perturbed)
        samples.append(features)

    return samples


def collect_live_samples(
    label: str,
    target_count: int = 50,
    camera_idx: int = DEFAULT_CAMERA_INDEX,
    signer_id: str = "signer_1",
) -> int:
    """Captures real-world human hand landmarks from a live webcam feed.
    
    NEVER silently generates synthetic data when webcam access fails.
    """
    clean_label = label.strip().upper()
    if clean_label not in ISL_ALPHABET:
        print(f"[ERROR] '{clean_label}' is not a valid ISL alphabet letter (A-Z).")
        return 0

    detector = HandDetectorV2()
    extractor = LandmarkExtractorV2()
    session_id = f"sess_{str(uuid.uuid4())[:6]}"
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    print(f"\n[+] Probing webcam (index {camera_idx}) for live ISL collection...")
    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        print(f"\n[ERROR] Webcam index {camera_idx} could not be opened.")
        print("[ERROR] Please verify camera connectivity and user permissions.")
        print("[ERROR] In WSL Ubuntu, attach your webcam via Windows PowerShell (Admin):")
        print("          usbipd wsl attach --busid <busid>")
        print("[ERROR] Aborting live collection. No synthetic data will be fabricated.\n")
        return 0

    collected: List[np.ndarray] = []
    print(f"\n=======================================================")
    print(f" Live ISL Data Collection: Letter '{clean_label}'")
    print(f" Signer: {signer_id} | Session: {session_id}")
    print(f" Target: {target_count} real human samples")
    print(" Instructions:")
    print("   1. Form the official ISLRTC handshape in front of the camera.")
    print("   2. Press 'c' to record a sample when your hand(s) are positioned.")
    print("   3. Vary your hand angle and distance slightly for diversity.")
    print("   4. Press 'q' to stop and save.")
    print("=======================================================\n")

    try:
        while len(collected) < target_count:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from webcam.")
                break

            res: HandDetectionResultV2 = detector.detect(frame)
            annotated = detector.draw_landmarks(frame, res)

            status_msg = f"ISL Letter: {clean_label} | Recorded: {len(collected)}/{target_count}"
            cv2.putText(
                annotated,
                status_msg,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )

            if has_display:
                cv2.imshow(f"SignBridge AI v2 - Live ISL Collector ({clean_label})", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    if res.detected:
                        try:
                            feats = extractor.extract_features(res)
                            collected.append(feats)
                            print(f"[✓] Captured sample {len(collected)}/{target_count} (Hands: {res.num_hands})")
                        except ValueError as e:
                            print(f"[-] Sample rejected: {e}")
                    else:
                        print("[-] No hand detected! Position hand(s) clearly before capturing.")
                elif key == ord("q"):
                    print("[*] Collection stopped by user.")
                    break
            else:
                # Terminal interactive mode if DISPLAY is unavailable
                if res.detected:
                    feats = extractor.extract_features(res)
                    collected.append(feats)
                    print(f"[✓] Auto-captured sample {len(collected)}/{target_count} (Hands: {res.num_hands})")
                    time.sleep(0.3)
                else:
                    time.sleep(0.1)

    finally:
        cap.release()
        if has_display:
            cv2.destroyAllWindows()

    saved = save_samples(
        ISL_DATASET_PATH,
        clean_label,
        collected,
        source_type="live_camera",
        signer_id=signer_id,
        session_id=session_id,
    )
    print(f"\n[✓] Successfully saved {saved} genuine human samples to: {ISL_DATASET_PATH}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignBridge AI v2 ISL Alphabet Data Collector")
    parser.add_argument(
        "--label",
        type=str,
        default="A",
        help="ISL alphabet letter to collect (A - Z)",
    )
    parser.add_argument("--count", type=int, default=50, help="Target number of samples")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="EXPLICIT FLAG: Generate synthetic benchmark samples for CI/testing only",
    )
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX, help="Camera index")
    parser.add_argument("--signer", type=str, default="signer_1", help="Signer identifier")
    args = parser.parse_args()

    lbl = args.label.strip().upper()
    if args.synthetic:
        print(f"\n[!] Generating {args.count} synthetic benchmark samples for '{lbl}'...")
        print("[!] These are tagged as 'synthetic_benchmark' and will be rejected for production models.")
        samples = generate_synthetic_benchmark_samples(lbl, args.count)
        saved = save_samples(
            ISL_DATASET_PATH,
            lbl,
            samples,
            source_type="synthetic_benchmark",
            signer_id="synthetic_generator",
        )
        print(f"[✓] Saved {saved} synthetic samples to {ISL_DATASET_PATH}")
    else:
        collect_live_samples(lbl, args.count, camera_idx=args.camera, signer_id=args.signer)
