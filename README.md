# Speaker Diarization

An end-to-end **speaker diarization system** with a clean web interface.

Users can:
- Upload any speech audio file (`.wav`, `.mp3`, `.m4a`, etc.)
- Try built-in **sample audio files**
- View a waveform with **speaker-colored segments**
- Play audio with a **moving cursor synced to time**
- Inspect diarization results as structured JSON

The backend is built using **FastAPI**, and the frontend is a lightweight HTML/JavaScript app using **WaveSurfer.js**.

---

## Demo Screenshot 



![Diarization UI Screenshot](assets/screenshot.png)
---

## Project Structure 

```
Speaker Diarization/
│
├── main.py                # FastAPI server (API + static file hosting)
├── dia.py                 # Diarization pipeline (VAD, ECAPA, AHC, JSON)
├── requirements.txt
├── README.md
│
├── frontend/
│   └── index.html         # Web UI (waveform, legend, cursor)
│
├── uploads/               # Uploaded audio files (server-side)
├── outputs/               # Converted WAV files served to browser (/files/*)
├── samples/               # Sample audio files served to browser (/samples/*)
└── assets/
    └── screenshot.png     # UI screenshot for README
```

---

## Diarization Pipeline 

1. **Voice Activity Detection (VAD)**  
   Uses Silero VAD to detect speech regions.

2. **Chunking**  
   Speech regions are split into overlapping chunks.

3. **Speaker Embeddings**  
   ECAPA-TDNN embeddings via SpeechBrain.

4. **Clustering**  
   Agglomerative Hierarchical Clustering (cosine distance).

5. **Post-processing**  
   Merge adjacent segments and remove very short segments.

---

## Requirements 

### System Dependency

You must have **ffmpeg** installed (required for decoding `.m4a` and other formats).

**macOS**
```bash
brew install ffmpeg
```

**Ubuntu / Debian**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Verify:
```bash
ffmpeg -version
```

---

### Python Dependencies 

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Backend 

From the project root (the folder containing `main.py`):

```bash
uvicorn main:app --reload --port 8000
```

API docs:
```
http://127.0.0.1:8000/docs
```

Static routes:
- `/files/*` → `outputs/`
- `/samples/*` → `samples/`

---

## Running the Frontend 

From the `frontend/` directory:

```bash
python -m http.server 5173
```

Open:
```
http://127.0.0.1:5173/index.html
```

Ensure your frontend points to the backend:
```js
const API = "http://127.0.0.1:8000";
```

---

## API Endpoints 🔗

### POST `/diarize`

Upload an audio file and return diarization results.

Example response:
```json
{
  "audioUrl": "/files/abcd1234.wav",
  "duration": 28.755,
  "speakers": [
    { "start": 0.5, "end": 3.1, "speaker": 0 },
    { "start": 3.7, "end": 5.6, "speaker": 0 }
  ]
}
```

---

### POST `/diarize_url`

Run diarization on a sample audio file served from `/samples/*`.

Used by the **“Try Sample”** button in the UI.

---

## Notes 

- The first request may be slow due to model initialization.
- Speaker IDs are **cluster labels**, not real identities.
- Sample files must exist in the `samples/` folder to appear in the UI.
- Intended for **demo, research, and portfolio use**.
