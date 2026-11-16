"""Backend API Contract and Integration Tests for SignBridge AI.
Tests /api/health, /api/supported-gestures, /api/predict, /api/sentence endpoints,
and FacialAnalysisService integration.
"""

import base64
import json
from pathlib import Path
import sys
import unittest
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import BuiltinAPIHandler, ai_service, decode_base64_image, facial_service
from ai.sign_recognition_v2.config import SUPPORTED_ALPHABET, ISL_ALPHABET_METADATA


class TestBackendAPI(unittest.TestCase):
    """Verifies that the backend logic strictly satisfies the SignBridge AI API contract."""

    def setUp(self):
        ai_service.clear_sentence()
        facial_service.set_active(True)

    def tearDown(self):
        ai_service.clear_sentence()
        facial_service.set_active(True)

    def test_decode_base64_image(self):
        """Verifies raw base64 and data URLs are decoded safely into OpenCV BGR numpy arrays."""
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", dummy)
        b64 = base64.b64encode(buf).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"

        frame_raw = decode_base64_image(b64)
        frame_url = decode_base64_image(data_url)

        self.assertIsNotNone(frame_raw)
        self.assertIsNotNone(frame_url)
        self.assertEqual(frame_raw.shape, (100, 100, 3))
        self.assertEqual(frame_url.shape, (100, 100, 3))

        # Test corrupt base64
        self.assertIsNone(decode_base64_image("not_a_valid_base64!"))
        self.assertIsNone(decode_base64_image(""))
        self.assertIsNone(decode_base64_image(None))

    def test_predict_api_contract_structure(self):
        """Validates that prediction response strictly matches the required contract."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        res = ai_service.process_frame(dummy)

        expected_keys = [
            "success",
            "recognized_text",
            "confidence",
            "hand_detected",
            "num_hands",
            "gesture_recognized",
            "status",
            "system_status",
            "current_letter",
            "current_word",
            "sentence",
            "latency_ms",
        ]
        for key in expected_keys:
            self.assertIn(key, res, f"Missing required contract key: {key}")

        self.assertIsInstance(res["success"], bool)
        self.assertIsInstance(res["confidence"], float)
        self.assertIsInstance(res["hand_detected"], bool)
        self.assertIsInstance(res["gesture_recognized"], bool)
        self.assertIsInstance(res["status"], str)
        self.assertIsInstance(res["current_word"], str)
        self.assertIsInstance(res["sentence"], str)
        self.assertIsInstance(res["latency_ms"], float)

    def test_sentence_flow_via_service(self):
        """Validates v2 sentence accumulation, space insertion, backspacing, and clearing."""
        # Active spelling
        ai_service.sentence_builder.add_letter("H")
        ai_service.sentence_builder.add_letter("I")
        self.assertEqual(ai_service.current_word, "HI")
        self.assertEqual(ai_service.completed_sentence, "HI")

        # Space insertion completes the word
        completed = ai_service.insert_space()
        self.assertEqual(completed, "HI")
        self.assertEqual(ai_service.current_word, "")
        self.assertEqual(ai_service.completed_sentence, "HI")

        # Start second word
        ai_service.sentence_builder.add_letter("A")
        self.assertEqual(ai_service.current_word, "A")
        self.assertEqual(ai_service.completed_sentence, "HI A")

        # Backspace removes the active letter
        removed = ai_service.backspace()
        self.assertEqual(removed, "A")
        self.assertEqual(ai_service.current_word, "")
        self.assertEqual(ai_service.completed_sentence, "HI")

        # Clear sentence resets everything
        ai_service.clear_sentence()
        self.assertEqual(ai_service.current_word, "")
        self.assertEqual(ai_service.completed_sentence, "")

    def test_facial_analysis_behavior(self):
        """Verifies FacialAnalysisService contract, missing face handling, and honest status."""
        self.assertEqual(facial_service.status, "ready")

        # Test invalid frame
        res_invalid = facial_service.analyze_face(None)
        self.assertFalse(res_invalid["success"])
        self.assertEqual(res_invalid["status"], "invalid_frame")
        self.assertEqual(res_invalid["confidence"], 0.0)
        self.assertIn("disclaimer", res_invalid)

        # Test blank frame (no face detected)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        res_blank = facial_service.analyze_face(blank)
        self.assertFalse(res_blank["success"])
        self.assertEqual(res_blank["status"], "no_face_detected")
        self.assertFalse(res_blank["face_detected"])
        self.assertEqual(res_blank["confidence"], 0.0)
        self.assertIn("disclaimer", res_blank)

        # Test disabled mode
        facial_service.set_active(False)
        self.assertEqual(facial_service.status, "module_disabled")
        res_disabled = facial_service.analyze_face(blank)
        self.assertFalse(res_disabled["success"])
        self.assertEqual(res_disabled["status"], "module_disabled")
        self.assertEqual(res_disabled["num_faces"], 0)
        self.assertEqual(res_disabled["faces"], [])
        self.assertEqual(len(res_disabled["supported_expressions"]), 9)

        # Restore active state
        facial_service.set_active(True)
        self.assertEqual(facial_service.status, "ready")

    def test_supported_alphabet_config(self):
        """Verifies supported ISL alphabet items and metadata integrity."""
        self.assertGreater(len(SUPPORTED_ALPHABET), 0)
        for letter in SUPPORTED_ALPHABET:
            self.assertIn(letter, ISL_ALPHABET_METADATA)
            meta = ISL_ALPHABET_METADATA[letter]
            self.assertIn("status", meta)
            self.assertIn("hands_required", meta)
            self.assertIn("is_dynamic", meta)


    def test_live_http_server_endpoints(self):
        """Verifies full HTTP server request/response cycle on a live socket."""
        import threading
        import urllib.request
        from http.server import ThreadingHTTPServer

        server = ThreadingHTTPServer(("127.0.0.1", 0), BuiltinAPIHandler)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            base_url = f"http://127.0.0.1:{port}"

            # 1. /api/health
            with urllib.request.urlopen(f"{base_url}/api/health") as resp:
                data = json.loads(resp.read().decode())
                self.assertEqual(resp.status, 200)
                self.assertEqual(data["status"], "online")
                self.assertEqual(data["detector_backend"], "mediapipe_tasks")

            # 2. /api/supported-gestures
            with urllib.request.urlopen(f"{base_url}/api/supported-gestures") as resp:
                data = json.loads(resp.read().decode())
                self.assertEqual(resp.status, 200)
                self.assertGreater(data["count"], 0)
                self.assertIn("disclaimer", data)

            # 3. /api/predict with blank frame
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", dummy)
            b64 = base64.b64encode(buf).decode()
            post_data = json.dumps({"image": b64, "annotate": False}).encode()
            req = urllib.request.Request(
                f"{base_url}/api/predict",
                data=post_data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                self.assertEqual(resp.status, 200)
                self.assertFalse(data["hand_detected"])
                self.assertEqual(data["status"], "no_hand_detected")

            # 4. /api/sentence endpoints
            # Add letter
            req = urllib.request.Request(
                f"{base_url}/api/sentence/add",
                data=json.dumps({"letter": "A"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                self.assertEqual(data["current_word"], "A")

            # Space
            req = urllib.request.Request(
                f"{base_url}/api/sentence/space",
                data=b"{}",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                self.assertEqual(data["sentence"], "A")
                self.assertEqual(data["current_word"], "")

            # Backspace
            req = urllib.request.Request(
                f"{base_url}/api/sentence/backspace",
                data=b"{}",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                self.assertTrue(data["success"])

            # Clear
            req = urllib.request.Request(
                f"{base_url}/api/sentence/clear",
                data=b"{}",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                self.assertEqual(data["sentence"], "")

            # 5. /api/analyze/face
            req = urllib.request.Request(
                f"{base_url}/api/analyze/face",
                data=json.dumps({"image": b64}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                self.assertEqual(resp.status, 200)
                self.assertEqual(data["status"], "no_face_detected")

            # 6. /api/language endpoints & bilingual mode
            with urllib.request.urlopen(f"{base_url}/api/language") as resp:
                lang_data = json.loads(resp.read().decode())
                self.assertEqual(resp.status, 200)
                self.assertIn("current_language", lang_data)
                self.assertIn("available_languages", lang_data)

            # Switch language to ASL
            req_lang = urllib.request.Request(
                f"{base_url}/api/language",
                data=json.dumps({"language": "ASL"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req_lang) as resp:
                lang_switch = json.loads(resp.read().decode())
                self.assertEqual(resp.status, 200)
                self.assertEqual(lang_switch["language"], "ASL")

            # Verify ASL gestures
            with urllib.request.urlopen(f"{base_url}/api/supported-gestures?lang=ASL") as resp:
                asl_data = json.loads(resp.read().decode())
                self.assertEqual(resp.status, 200)
                self.assertEqual(asl_data["language"], "ASL")
                self.assertEqual(asl_data["count"], 26)
                self.assertEqual(asl_data["counts_by_status"]["experimental"], 24)
                self.assertEqual(asl_data["counts_by_status"]["experimental_dynamic"], 2)

            # Switch back to ISL
            req_isl = urllib.request.Request(
                f"{base_url}/api/language",
                data=json.dumps({"language": "ISL"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req_isl) as resp:
                lang_switch_isl = json.loads(resp.read().decode())
                self.assertEqual(lang_switch_isl["language"], "ISL")

            # 7. Static index.html
            with urllib.request.urlopen(f"{base_url}/") as resp:
                self.assertEqual(resp.status, 200)
                content = resp.read().decode()
                self.assertIn("SIGN BRIDGE", content)

        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
