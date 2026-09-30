from __future__ import annotations

import csv
import io
import logging
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from dia import (
    DEFAULT_CLUSTERING_THRESHOLD,
    chunks_to_segments,
    diarize_file,
    make_diarization_payload,
    to_wav_16k_mono_ffmpeg,
)
from voxconverse_utils import DiarizationScore, Segment, read_rttm, score_diarization


LOGGER = logging.getLogger("speaker-diarization")
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
SAMPLES_DIR = BASE_DIR / "samples"
SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".mp4", ".webm"}

for directory in (UPLOAD_DIR, OUTPUT_DIR, SAMPLES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Speaker Diarization Studio", version="1.0.0")
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")
app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required. On macOS, install it with: brew install ffmpeg")


def safe_name(filename: str | None, fallback: str) -> str:
    name = Path(filename or fallback).name
    return name if name not in {"", ".", ".."} else fallback


def save_upload(upload: UploadFile, destination: Path) -> None:
    with destination.open("wb") as target:
        shutil.copyfileobj(upload.file, target, length=1024 * 1024)


def process_audio(input_path: Path, original_name: str) -> dict:
    require_ffmpeg()
    uid = uuid.uuid4().hex
    output_path = OUTPUT_DIR / f"{uid}.wav"
    try:
        to_wav_16k_mono_ffmpeg(str(input_path), str(output_path), target_sr=16000)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"ffmpeg could not decode {original_name}") from error

    chunks, labels = diarize_file(str(output_path))
    payload = make_diarization_payload(
        audio_url=f"/files/{output_path.name}",
        audio_path=str(output_path),
        chunks=chunks,
        labels=labels,
    )
    payload.update(
        {
            "filename": original_name,
            "speakerCount": len({segment["speaker"] for segment in payload["speakers"]}),
            "clusteringThreshold": DEFAULT_CLUSTERING_THRESHOLD,
        }
    )
    return payload


def group_benchmark_uploads(files: list[UploadFile]) -> tuple[dict[str, UploadFile], dict[str, UploadFile]]:
    audio = {}
    annotations = {}
    for upload in files:
        name = safe_name(upload.filename, "upload")
        suffix = Path(name).suffix.lower()
        stem = Path(name).stem
        target = annotations if suffix == ".rttm" else audio if suffix in SUPPORTED_AUDIO else None
        if target is None:
            continue
        if stem in target:
            raise ValueError(f"Duplicate {suffix} file for recording '{stem}'")
        target[stem] = upload

    missing_rttm = sorted(audio.keys() - annotations.keys())
    missing_audio = sorted(annotations.keys() - audio.keys())
    if not audio and not annotations:
        raise ValueError("No supported audio or RTTM files were selected")
    if missing_rttm or missing_audio:
        details = []
        if missing_rttm:
            details.append("missing RTTM: " + ", ".join(missing_rttm[:8]))
        if missing_audio:
            details.append("missing audio: " + ", ".join(missing_audio[:8]))
        raise ValueError("Unmatched dataset files (" + "; ".join(details) + ")")
    return audio, annotations


def evaluation_csv(rows: list[dict], weighted_der_percent: float) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "file",
            "duration_seconds",
            "der_percent",
            "reference_speakers",
            "predicted_speakers",
            "missed_speech_seconds",
            "false_alarm_seconds",
            "speaker_confusion_seconds",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row["filename"],
                row["durationSeconds"],
                row["derPercent"],
                row["referenceSpeakers"],
                row["predictedSpeakers"],
                row["missedSpeechSeconds"],
                row["falseAlarmSeconds"],
                row["speakerConfusionSeconds"],
            ]
        )
    writer.writerow(["Duration-weighted DER", "", weighted_der_percent, "", "", "", "", ""])
    return output.getvalue()


def process_benchmark(files: list[UploadFile], collar: float, ignore_overlap: bool) -> dict:
    require_ffmpeg()
    if collar < 0:
        raise ValueError("Collar must be non-negative")
    audio_uploads, rttm_uploads = group_benchmark_uploads(files)

    rows = []
    aggregate = DiarizationScore()
    with tempfile.TemporaryDirectory(prefix="benchmark-", dir=str(UPLOAD_DIR)) as temporary:
        work_dir = Path(temporary)
        for recording_id in sorted(audio_uploads):
            audio_upload = audio_uploads[recording_id]
            audio_name = safe_name(audio_upload.filename, f"{recording_id}.wav")
            input_path = work_dir / f"{recording_id}{Path(audio_name).suffix.lower()}"
            rttm_path = work_dir / f"{recording_id}.rttm"
            wav_path = work_dir / f"{recording_id}_16k.wav"
            save_upload(audio_upload, input_path)
            save_upload(rttm_uploads[recording_id], rttm_path)
            try:
                to_wav_16k_mono_ffmpeg(str(input_path), str(wav_path), target_sr=16000)
            except subprocess.CalledProcessError as error:
                raise ValueError(f"ffmpeg could not decode {audio_name}") from error

            duration = sf.info(wav_path).duration
            reference = read_rttm(rttm_path)
            reference_ids = {segment.recording_id for segment in reference}
            if reference_ids != {recording_id}:
                raise ValueError(
                    f"{recording_id}.rttm must contain recording ID '{recording_id}', "
                    f"found: {', '.join(sorted(reference_ids)) or 'none'}"
                )

            chunks, labels = diarize_file(str(wav_path))
            hypothesis = [] if len(chunks) != len(labels) else [
                Segment(recording_id, item["start"], item["end"], f"speaker_{item['speaker']}")
                for item in chunks_to_segments(chunks, labels)
            ]
            score = score_diarization(
                reference,
                hypothesis,
                evaluation_end=duration,
                collar=collar,
                ignore_overlap=ignore_overlap,
            )
            aggregate.add(score)
            rows.append(
                {
                    "filename": audio_name,
                    "durationSeconds": round(duration, 3),
                    "derPercent": round(score.der * 100.0, 3),
                    "referenceSpeakers": len({segment.speaker for segment in reference}),
                    "predictedSpeakers": len({segment.speaker for segment in hypothesis}),
                    "missedSpeechSeconds": round(score.missed_speech, 3),
                    "falseAlarmSeconds": round(score.false_alarm, 3),
                    "speakerConfusionSeconds": round(score.speaker_confusion, 3),
                }
            )

    total_duration = sum(row["durationSeconds"] for row in rows)
    weighted_der = sum(row["durationSeconds"] * row["derPercent"] for row in rows) / total_duration
    result = {
        "files": len(rows),
        "totalDurationSeconds": round(total_duration, 3),
        "durationWeightedDerPercent": round(weighted_der, 3),
        "aggregateDerPercent": round(aggregate.der * 100.0, 3),
        "collarSeconds": collar,
        "overlapScored": not ignore_overlap,
        "clusteringThreshold": DEFAULT_CLUSTERING_THRESHOLD,
        "results": rows,
    }
    result["csv"] = evaluation_csv(rows, result["durationWeightedDerPercent"])
    return result


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "clusteringThreshold": DEFAULT_CLUSTERING_THRESHOLD,
    }


@app.post("/api/diarize")
@app.post("/diarize", include_in_schema=False)
async def diarize(file: UploadFile = File(...)) -> dict:
    name = safe_name(file.filename, "audio")
    if Path(name).suffix.lower() not in SUPPORTED_AUDIO:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type: {Path(name).suffix or 'none'}")
    input_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{name}"
    try:
        save_upload(file, input_path)
        return await run_in_threadpool(process_audio, input_path, name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("Diarization failed")
        raise HTTPException(status_code=500, detail="Diarization failed. Check the server log.") from error
    finally:
        input_path.unlink(missing_ok=True)


@app.post("/api/evaluate")
async def evaluate(
    files: list[UploadFile] = File(...),
    collar: float = Form(0.25),
    ignore_overlap: bool = Form(False),
) -> dict:
    if len(files) > 1000:
        raise HTTPException(status_code=400, detail="At most 500 audio/RTTM pairs are supported per run")
    try:
        return await run_in_threadpool(process_benchmark, files, collar, ignore_overlap)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("Benchmark evaluation failed")
        raise HTTPException(status_code=500, detail="Evaluation failed. Check the server log.") from error
