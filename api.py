"""FastAPI service for ASR pipeline."""

import asyncio
import glob
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from asr.pipeline import run_pipeline
from asr.engine import asr_transcribe
from asr.config import get_queue_size


# ── Request/Response models ──────────────────────────────────────

class TranscribeRequest(BaseModel):
    """Request for full pipeline transcription."""
    language: Optional[str] = None
    model_size: str = "1.7B"
    max_chars: int = 14
    fmt: str = "srt"
    ass_style: str = "default"
    fix_dir: Optional[str] = None


class TranscribeResponse(BaseModel):
    """Response with paths to generated subtitle files."""
    task_id: str
    srt_path: Optional[str] = None
    ass_path: Optional[str] = None
    status: str = "completed"


class TextOnlyRequest(BaseModel):
    """Request for text-only transcription."""
    language: Optional[str] = None
    model_size: str = "1.7B"


class TextOnlyResponse(BaseModel):
    """Response with transcribed text."""
    text: str
    language: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    backend: str


# ── Queue ───────────────────────────────────────────────────────

_queue_semaphore: Optional[asyncio.Semaphore] = None


def _get_queue_semaphore() -> asyncio.Semaphore:
    """Get or create the queue semaphore."""
    global _queue_semaphore
    if _queue_semaphore is None:
        _queue_semaphore = asyncio.Semaphore(get_queue_size())
    return _queue_semaphore


# ── Lifespan ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Initialize semaphore on startup
    _get_queue_semaphore()
    yield


# ── App ─────────────────────────────────────────────────────────

app = FastAPI(
    title="ASR Pipeline API",
    description="Multi-stage ASR subtitle generation with Qwen3-ASR",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Endpoints ───────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    from asr.platform import get_backend
    return HealthResponse(status="ok", backend=get_backend())


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(...),
    language: Optional[str] = None,
    model_size: str = "1.7B",
    max_chars: int = 14,
    fmt: str = "srt",
    ass_style: str = "default",
    fix_dir: Optional[str] = None,
):
    """Full pipeline transcription - audio to subtitles.

    Accepts audio file upload and returns SRT/ASS subtitle files.
    """
    # Validate format
    if fmt not in ("srt", "ass", "all"):
        raise HTTPException(400, f"Invalid format: {fmt}. Must be srt, ass, or all")

    # Acquire queue slot (non-blocking)
    semaphore = _get_queue_semaphore()
    if semaphore.locked():
        raise HTTPException(
            503,
            f"Queue is full. Maximum concurrent tasks: {get_queue_size()}. Please try again later.",
        )
    await semaphore.acquire()

    # Save uploaded file to temp location
    task_id = str(uuid.uuid4())
    filename = Path(audio.filename).name  # strip any path components
    allowed = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".wma", ".aac"}
    suffix = Path(filename).suffix.lower() if "." in filename else ""
    if suffix not in allowed:
        suffix = ".wav"  # safe default
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        audio_path = tmp.name

    # Create output directory
    output_dir = tempfile.mkdtemp(prefix=f"asr_{task_id}_")

    try:
        result = run_pipeline(
            audio_path=audio_path,
            output_dir=output_dir,
            fmt=fmt,
            ass_style=ass_style,
            fix_dir=fix_dir,
            language=language,
            model_size=model_size,
            max_chars=max_chars,
        )

        if isinstance(result, dict) and "check_errors" in result:
            raise HTTPException(422, f"Check errors: {result['check_errors']}")

        srt_path = result.get("srt")
        ass_path = result.get("ass")

        return TranscribeResponse(
            task_id=task_id,
            srt_path=srt_path,
            ass_path=ass_path,
            status="completed",
        )

    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        semaphore.release()
        # Cleanup temp audio file only (output_dir kept for /download)
        if os.path.exists(audio_path):
            os.unlink(audio_path)


@app.post("/transcribe/text", response_model=TextOnlyResponse)
async def transcribe_text(
    audio: UploadFile = File(...),
    language: Optional[str] = None,
    model_size: str = "1.7B",
):
    """Text-only transcription (no timestamps).

    Faster than full pipeline, returns only transcribed text.
    """
    # Acquire queue slot (non-blocking)
    semaphore = _get_queue_semaphore()
    if semaphore.locked():
        raise HTTPException(
            503,
            f"Queue is full. Maximum concurrent tasks: {get_queue_size()}. Please try again later.",
        )
    await semaphore.acquire()

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(suffix=Path(audio.filename).suffix, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        audio_path = tmp.name

    try:
        text = asr_transcribe(
            audio_path=audio_path,
            language=language,
            model_size=model_size,
        )

        from asr.platform import get_backend
        backend = get_backend()
        lang_code = language or backend

        return TextOnlyResponse(text=text, language=lang_code)

    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        semaphore.release()
        # Cleanup temp audio file
        if os.path.exists(audio_path):
            os.unlink(audio_path)


@app.get("/download/{task_id}/{fmt}")
async def download(task_id: str, fmt: str, background_tasks: BackgroundTasks):
    """Download generated subtitle file."""
    if fmt not in ("srt", "ass"):
        raise HTTPException(400, "Format must be srt or ass")

    # Find the output directory for this task
    temp_dir = tempfile.gettempdir()
    for entry in os.listdir(temp_dir):
        if entry.startswith(f"asr_{task_id}_"):
            output_dir = os.path.join(temp_dir, entry)
            file_path = os.path.join(output_dir, f"*.{fmt}")
            matches = glob.glob(file_path)
            if matches:
                background_tasks.add_task(shutil.rmtree, output_dir, ignore_errors=True)
                return FileResponse(
                    matches[0],
                    media_type="text/plain",
                    filename=f"subtitle.{fmt}",
                )

    raise HTTPException(404, "File not found or task expired")
