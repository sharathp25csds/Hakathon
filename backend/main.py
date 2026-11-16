"""SignBridge AI - Backend API Server.
Provides REST API endpoints conforming to the SignBridge AI API contract.
Integrates SignRecognitionServiceV2 and FacialAnalysisService.
Supports both FastAPI (production mode) and Python ThreadingHTTPServer (zero-dependency mode).
"""

import base64
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.sign_recognition_v2.config import (
    ASL_ALPHABET_METADATA,
    ASL_COMPLIANCE_DISCLAIMER,
    ISL_ALPHABET_METADATA,
    ISL_COMPLIANCE_DISCLAIMER,
    SignLanguage,
    SUPPORTED_ALPHABET,
    SUPPORTED_ASL_ALPHABET,
    SUPPORTED_ISL_ALPHABET,
)
from ai.sign_recognition_v2.prediction import SignRecognitionServiceV2
from ai.facial_analysis import facial_service

# Initialize global AI service (SignRecognitionServiceV2)
ai_service = SignRecognitionServiceV2()
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def get_supported_gestures_dict(lang_str: Optional[str] = None) -> Dict[str, Any]:
    """Builds the alphabet metadata and count dictionary for the requested or active language."""
    target_lang = (lang_str or ai_service.current_language).upper()
    if target_lang == "ASL":
        metadata = ASL_ALPHABET_METADATA
        supported = SUPPORTED_ASL_ALPHABET
        disclaimer = ASL_COMPLIANCE_DISCLAIMER
    else:
        target_lang = "ISL"
        metadata = ISL_ALPHABET_METADATA
        supported = SUPPORTED_ISL_ALPHABET
        disclaimer = ISL_COMPLIANCE_DISCLAIMER

    counts_by_status = {
        "supported": 0,
        "experimental": 0,
        "experimental_dynamic": 0,
        "unsupported": 0,
        "total": len(metadata),
    }
    for meta in metadata.values():
        st = meta.get("status", "unsupported")
        counts_by_status[st] = counts_by_status.get(st, 0) + 1

    return {
        "language": target_lang,
        "supported_alphabet": supported,
        "metadata": metadata,
        "count": len(supported),
        "disclaimer": disclaimer,
        "counts_by_status": counts_by_status,
        "gestures": supported,
        "descriptions": {
            k: f"{v.get('dominant_hand', '')} ({v.get('status', 'experimental')})"
            for k, v in metadata.items()
        },
    }


def get_language_status_dict() -> Dict[str, Any]:
    """Returns currently selected sign language and available language modes."""
    return {
        "current_language": ai_service.current_language,
        "available_languages": ["ISL", "ASL"],
        "supported_counts": {
            "ISL": len(SUPPORTED_ISL_ALPHABET),
            "ASL": len(SUPPORTED_ASL_ALPHABET),
        },
    }


def decode_base64_image(image_data: str) -> Optional[np.ndarray]:
    """Decodes raw base64 or data-URL string into OpenCV BGR numpy array."""
    if not image_data or not isinstance(image_data, str):
        return None
    try:
        # Strip data URL prefix if present (e.g. data:image/jpeg;base64,...)
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FastAPI Implementation (Used when FastAPI and Uvicorn are installed)
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    class PredictRequest(BaseModel):
        image: str
        annotate: bool = False
        language: Optional[str] = None

    class LanguageRequest(BaseModel):
        language: str

    class FaceAnalyzeRequest(BaseModel):
        image: str

    class SentenceAddRequest(BaseModel):
        letter: Optional[str] = None
        word: Optional[str] = None

    app = FastAPI(
        title="SignBridge AI API",
        description="Assistive Bilingual (ISL & ASL) Alphabet & Facial Analysis REST API",
        version="2.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health_check():
        return {
            "status": "online",
            "service": "SignBridge AI",
            "version": "2.1.0",
            "current_language": ai_service.current_language,
            "available_languages": ["ISL", "ASL"],
            "detector_backend": ai_service.detector.backend,
            "ml_model_loaded": ai_service.classifier.ml_classifier.is_loaded,
            "supported_alphabet_count": len(SUPPORTED_ALPHABET),
            "facial_module_status": facial_service.status,
        }

    @app.get("/api/language")
    def get_language_endpoint():
        return get_language_status_dict()

    @app.post("/api/language")
    def set_language_endpoint(req: LanguageRequest):
        ai_service.set_language(req.language)
        return {"success": True, "language": ai_service.current_language}

    @app.get("/api/supported-gestures")
    def get_supported_gestures(lang: Optional[str] = None):
        return get_supported_gestures_dict(lang)

    @app.post("/api/predict")
    def predict_sign(req: PredictRequest):
        if req.language:
            ai_service.set_language(req.language)

        frame = decode_base64_image(req.image)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

        res = ai_service.process_frame(frame, annotate=req.annotate)

        annotated_b64 = None
        if req.annotate and res.get("annotated_frame") is not None:
            _, buffer = cv2.imencode(".jpg", res["annotated_frame"])
            annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        return {
            "success": res["success"],
            "recognized_text": res.get("recognized_text"),
            "current_letter": res.get("current_letter"),
            "current_word": res.get("current_word", ""),
            "sentence": res.get("sentence", ""),
            "confidence": res.get("confidence", 0.0),
            "language": res.get("language", ai_service.current_language),
            "hand_detected": res.get("hand_detected", False),
            "num_hands": res.get("num_hands", 0),
            "gesture_recognized": res.get("gesture_recognized", False),
            "status": res.get("status", "unknown"),
            "system_status": res.get("system_status", ""),
            "handedness": res.get("handedness", "Unknown"),
            "bbox": res.get("bbox", [0, 0, 0, 0]),
            "landmarks": res.get("landmarks", []),
            "latency_ms": res.get("latency_ms", 0.0),
            "annotated_image": annotated_b64,
        }

    @app.post("/api/sentence/clear")
    def clear_sentence():
        ai_service.clear_sentence()
        return {
            "success": True,
            "sentence": "",
            "current_word": "",
        }

    @app.post("/api/sentence/backspace")
    def backspace_sentence():
        removed = ai_service.backspace()
        return {
            "success": True,
            "removed": removed,
            "sentence": ai_service.completed_sentence,
            "current_word": ai_service.current_word,
        }

    @app.post("/api/sentence/space")
    def space_sentence():
        completed = ai_service.insert_space()
        return {
            "success": True,
            "completed_word": completed,
            "sentence": ai_service.completed_sentence,
            "current_word": ai_service.current_word,
        }

    @app.post("/api/sentence/add")
    def add_to_sentence(req: SentenceAddRequest):
        if req.letter:
            ai_service.sentence_builder.add_letter(req.letter)
        elif req.word:
            for ch in req.word:
                ai_service.sentence_builder.add_letter(ch)
            ai_service.insert_space()
        return {
            "success": True,
            "sentence": ai_service.completed_sentence,
            "current_word": ai_service.current_word,
        }

    @app.post("/api/sentence/undo")
    def undo_sentence():
        removed = ai_service.backspace()
        return {
            "success": True,
            "removed": removed,
            "sentence": ai_service.completed_sentence,
            "current_word": ai_service.current_word,
        }

    @app.post("/api/analyze/face")
    def analyze_face_endpoint(req: FaceAnalyzeRequest):
        frame = decode_base64_image(req.image)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        return facial_service.analyze_face(frame)

    @app.get("/api/facial-analysis/status")
    def get_facial_status():
        return facial_service.analyze_face(None)

    # Serve static frontend files
    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

        @app.get("/")
        def serve_index():
            return FileResponse(str(FRONTEND_DIR / "index.html"))

    FASTAPI_AVAILABLE = True

except ImportError:
    FASTAPI_AVAILABLE = False
    app = None


# ---------------------------------------------------------------------------
# Zero-Dependency Builtin HTTP Server Fallback
# ---------------------------------------------------------------------------
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse


class BuiltinAPIHandler(BaseHTTPRequestHandler):
    """Standard-library HTTP server implementing the SignBridge AI API contract."""

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/health" or path == "/health":
            self._send_json({
                "status": "online",
                "service": "SignBridge AI (Built-in Server)",
                "version": "2.1.0",
                "current_language": ai_service.current_language,
                "available_languages": ["ISL", "ASL"],
                "detector_backend": ai_service.detector.backend,
                "ml_model_loaded": ai_service.classifier.ml_classifier.is_loaded,
                "supported_alphabet_count": len(SUPPORTED_ALPHABET),
                "facial_module_status": facial_service.status,
            })
        elif path == "/api/language":
            self._send_json(get_language_status_dict())
        elif path == "/api/supported-gestures":
            query_dict = urllib.parse.parse_qs(parsed.query)
            lang = query_dict.get("lang", [None])[0]
            self._send_json(get_supported_gestures_dict(lang))
        elif path == "/api/facial-analysis/status" or path == "/api/facial/status":
            self._send_json(facial_service.analyze_face(None))
        elif path == "/" or path == "/index.html":
            index_path = FRONTEND_DIR / "index.html"
            if index_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(index_path.read_bytes())
            else:
                self._send_json({"error": "frontend_not_found"}, 404)
        elif (FRONTEND_DIR / path.lstrip("/")).exists():
            file_path = FRONTEND_DIR / path.lstrip("/")
            mime_type = "text/plain"
            if file_path.suffix == ".css":
                mime_type = "text/css"
            elif file_path.suffix == ".js":
                mime_type = "application/javascript"
            elif file_path.suffix == ".svg":
                mime_type = "image/svg+xml"
            elif file_path.suffix == ".png":
                mime_type = "image/png"
            elif file_path.suffix in (".jpg", ".jpeg"):
                mime_type = "image/jpeg"
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.end_headers()
            self.wfile.write(file_path.read_bytes())
        else:
            self._send_json({"error": "not_found"}, 404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            body = {}

        if path == "/api/language":
            lang = body.get("language")
            if lang:
                ai_service.set_language(lang)
            self._send_json({"success": True, "language": ai_service.current_language})
        elif path == "/api/predict":
            lang = body.get("language")
            if lang:
                ai_service.set_language(lang)

            image_data = body.get("image", "")
            annotate = body.get("annotate", False)
            frame = decode_base64_image(image_data)

            if frame is None:
                self._send_json({"success": False, "error": "Invalid base64 image data"}, 400)
                return

            res = ai_service.process_frame(frame, annotate=annotate)
            annotated_b64 = None
            if annotate and res.get("annotated_frame") is not None:
                _, buffer = cv2.imencode(".jpg", res["annotated_frame"])
                annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

            self._send_json({
                "success": res["success"],
                "recognized_text": res.get("recognized_text"),
                "current_letter": res.get("current_letter"),
                "current_word": res.get("current_word", ""),
                "sentence": res.get("sentence", ""),
                "confidence": res.get("confidence", 0.0),
                "language": res.get("language", ai_service.current_language),
                "hand_detected": res.get("hand_detected", False),
                "num_hands": res.get("num_hands", 0),
                "gesture_recognized": res.get("gesture_recognized", False),
                "status": res.get("status", "unknown"),
                "system_status": res.get("system_status", ""),
                "handedness": res.get("handedness", "Unknown"),
                "bbox": res.get("bbox", [0, 0, 0, 0]),
                "landmarks": res.get("landmarks", []),
                "latency_ms": res.get("latency_ms", 0.0),
                "annotated_image": annotated_b64,
            })
        elif path == "/api/sentence/clear":
            ai_service.clear_sentence()
            self._send_json({
                "success": True,
                "sentence": "",
                "current_word": "",
            })
        elif path == "/api/sentence/backspace" or path == "/api/sentence/undo":
            removed = ai_service.backspace()
            self._send_json({
                "success": True,
                "removed": removed,
                "sentence": ai_service.completed_sentence,
                "current_word": ai_service.current_word,
            })
        elif path == "/api/sentence/space":
            completed = ai_service.insert_space()
            self._send_json({
                "success": True,
                "completed_word": completed,
                "sentence": ai_service.completed_sentence,
                "current_word": ai_service.current_word,
            })
        elif path == "/api/sentence/add":
            letter = body.get("letter")
            word = body.get("word")
            if letter:
                ai_service.sentence_builder.add_letter(letter)
            elif word:
                for ch in word:
                    ai_service.sentence_builder.add_letter(ch)
                ai_service.insert_space()
            self._send_json({
                "success": True,
                "sentence": ai_service.completed_sentence,
                "current_word": ai_service.current_word,
            })
        elif path == "/api/analyze/face":
            image_data = body.get("image", "")
            frame = decode_base64_image(image_data)
            if frame is None:
                self._send_json({"success": False, "error": "Invalid base64 image data"}, 400)
                return
            res = facial_service.analyze_face(frame)
            self._send_json(res)
        else:
            self._send_json({"error": "endpoint_not_found"}, 404)

    def log_message(self, format, *args):
        sys.stderr.write(f"[SignBridge API] {self.address_string()} - {format % args}\n")


def run_server(port: int = 8000, host: str = "0.0.0.0"):
    """Starts backend server on designated host and port."""
    if FASTAPI_AVAILABLE and "UVICORN_PREFER" in os.environ:
        import uvicorn
        print(f"[+] Starting FastAPI production server on http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
    else:
        print(f"[+] Starting SignBridge AI API server on http://localhost:{port}")
        server = ThreadingHTTPServer((host, port), BuiltinAPIHandler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Shutting down server gracefully.")
            server.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port=port)
