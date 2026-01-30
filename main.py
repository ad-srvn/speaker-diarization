import os
import uuid
import shutil
import traceback
import subprocess
import shutil as sh
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Body
import requests

from dia import diarize_file, make_diarization_payload, to_wav_16k_mono_ffmpeg
BASE_DIR = Path(__file__).resolve().parent  
UPLOAD_DIR = BASE_DIR / "uploads"
OUT_DIR = BASE_DIR / "outputs"
SAMPLES_DIR = BASE_DIR / "samples"

UPLOAD_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)

UPLOAD_DIR = "uploads"
OUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/files", StaticFiles(directory=OUT_DIR), name="files")


app.mount("/samples", StaticFiles(directory="samples"), name="samples")


@app.post("/diarize_url")
def diarize_url(payload: dict = Body(...)):
    url = payload.get("url")
    if not url:
        return JSONResponse({"error": "Missing 'url' field"}, status_code=400)

    uid = uuid.uuid4().hex
    in_path = os.path.join(UPLOAD_DIR, f"{uid}_remote")
    out_audio_path = os.path.join(OUT_DIR, f"{uid}.wav")

    # 1) download
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(in_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    # 2) convert -> wav
    to_wav_16k_mono_ffmpeg(in_path, out_audio_path, target_sr=16000)

    # 3) diarize
    chunks, labels = diarize_file(out_audio_path)

    # 4) payload
    payload_out = make_diarization_payload(
        audio_url=f"/files/{uid}.wav",
        audio_path=out_audio_path,
        chunks=chunks,
        labels=labels,
    )
    return JSONResponse(payload_out)


@app.get("/")
def root():
    return {"status": "ok", "hint": "Go to /docs and POST an audio file to /diarize"}


@app.post("/diarize")
async def diarize(file: UploadFile = File(...)):
    
    try:
        uid = uuid.uuid4().hex
        in_path = os.path.join(UPLOAD_DIR, f"{uid}_{file.filename}")
        with open(in_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        if sh.which("ffmpeg") is None:
            return JSONResponse(
                {"error": "ffmpeg not found. Install it with: brew install ffmpeg"},
                status_code=500,
            )

        out_audio_path = os.path.join(OUT_DIR, f"{uid}.wav")
        try:
            to_wav_16k_mono_ffmpeg(in_path, out_audio_path, target_sr=16000)
        except subprocess.CalledProcessError:
            return JSONResponse(
                {"error": "ffmpeg failed to decode this file. Try exporting as wav/mp3 or re-downloading the audio."},
                status_code=500,
            )
        chunks, labels = diarize_file(out_audio_path)
        payload = make_diarization_payload(
            audio_url=f"/files/{uid}.wav",
            audio_path=out_audio_path,
            chunks=chunks,
            labels=labels,
        )

        return JSONResponse(payload)

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)
