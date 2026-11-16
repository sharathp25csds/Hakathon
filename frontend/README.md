# SIGN BRIDGE AI — Static Frontend

Plain HTML5 + CSS3 + vanilla JavaScript. No frameworks, no build step, no backend.

## Files
- `index.html` — page structure and the five sections (Dashboard, Text, Voice, Face, Settings)
- `style.css`  — all styling
- `script.js`  — navigation, recording, camera, uploads and backend calls
- `assets/`    — logo, favicon, illustrations

## Run locally
Double-click `index.html`, or (recommended, so camera and microphone work everywhere) serve the folder:

    python -m http.server 5500

then open http://localhost:5500

## Connect the Python backend
1. Open **Settings**, enter the backend URL (e.g. `http://localhost:8000`) and save.
2. Default endpoints (all editable in Settings):
   - `GET  /health`
   - `POST /api/analyze/text`  — JSON `{ "text", "language", "mode" }`
   - `POST /api/analyze/voice` — multipart field `audio`
   - `POST /api/analyze/face`  — multipart field `image`
3. Enable CORS on the backend (FastAPI: `CORSMiddleware`, Flask: `flask-cors`).
4. Results are shown as raw JSON. Change `renderResult()` in `script.js` to draw them your way.

No AI results are generated in the browser; result areas stay empty until the backend replies.
