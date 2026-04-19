"""FastAPI service for ASR pipeline."""

import asyncio
import glob
import json
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from asr.pipeline import run_pipeline
from asr.engine import asr_transcribe
from asr.config import get_queue_size, get_intermediate_dir


# ── Request/Response models ──────────────────────────────────────

class TranscribeRequest(BaseModel):
    """Request for full pipeline transcription."""
    language: Optional[str] = None
    model_size: str = "1.7B"
    max_chars: int = 14
    fmt: str = "srt"
    ass_style: str = "default"
    fix_dir: Optional[str] = None


class AlignRequest(BaseModel):
    """Request for alignment-only mode (text provided by user)."""
    text: str
    language: Optional[str] = None
    max_chars: int = 14
    fmt: str = "srt"
    ass_style: str = "default"


class TranscribeResponse(BaseModel):
    """Response with paths to generated subtitle files."""
    task_id: str
    srt_path: Optional[str] = None
    ass_path: Optional[str] = None
    status: str = "completed"


class AlignResponse(BaseModel):
    """Response for alignment-only mode."""
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

# Mount static files for web UI
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Endpoints ───────────────────────────────────────────────────


@app.get("/")
async def index():
    """Serve the web UI."""
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))

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
            intermediate_dir=get_intermediate_dir(),
            task_id=task_id,
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


@app.post("/align", response_model=AlignResponse)
async def align(
    audio: UploadFile = File(...),
    text: str = File(...),
    language: Optional[str] = None,
    max_chars: int = 14,
    fmt: str = "srt",
    ass_style: str = "default",
):
    """Alignment-only mode: audio + pre-transcribed text → subtitles.

    User provides both audio and the correct transcript.
    Skips ASR recognition, only does forced alignment with provided text.
    """
    if fmt not in ("srt", "ass", "all"):
        raise HTTPException(400, f"Invalid format: {fmt}. Must be srt, ass, or all")

    if not text or not text.strip():
        raise HTTPException(400, "text cannot be empty")

    # Acquire queue slot
    semaphore = _get_queue_semaphore()
    if semaphore.locked():
        raise HTTPException(
            503,
            f"Queue is full. Maximum concurrent tasks: {get_queue_size()}. Please try again later.",
        )
    await semaphore.acquire()

    # Save uploaded audio
    task_id = str(uuid.uuid4())
    filename = Path(audio.filename).name
    allowed = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".wma", ".aac"}
    suffix = Path(filename).suffix.lower() if "." in filename else ""
    if suffix not in allowed:
        suffix = ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        audio_path = tmp.name

    output_dir = tempfile.mkdtemp(prefix=f"asr_{task_id}_")

    try:
        result = run_pipeline(
            audio_path=audio_path,
            output_dir=output_dir,
            fmt=fmt,
            ass_style=ass_style,
            language=language,
            max_chars=max_chars,
            align_text=text.strip(),
            intermediate_dir=get_intermediate_dir(),
            task_id=task_id,
        )

        if isinstance(result, dict) and "check_errors" in result:
            raise HTTPException(422, f"Check errors: {result['check_errors']}")

        srt_path = result.get("srt")
        ass_path = result.get("ass")

        return AlignResponse(
            task_id=task_id,
            srt_path=srt_path,
            ass_path=ass_path,
            status="completed",
        )

    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        semaphore.release()
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


class KaraokeResponse(BaseModel):
    """Response with karaoke subtitle data."""
    task_id: str
    lines: list
    audio_path: Optional[str] = None


@app.get("/karaoke/{task_id}", response_model=KaraokeResponse)
async def get_karaoke(task_id: str):
    """Get karaoke subtitle data for a completed task.

    Returns subtitle lines with word-level timing for karaoke display.
    """
    # Search in output_dir (temp) and intermediate_dir (if configured)
    search_dirs = [tempfile.gettempdir()]
    intermediate_dir = get_intermediate_dir()
    if intermediate_dir:
        search_dirs.append(intermediate_dir)

    for search_dir in search_dirs:
        for entry in os.listdir(search_dir):
            entry_path = os.path.join(search_dir, entry)

            # Subdirectory: asr_{task_id}_*
            if entry.startswith(f"asr_{task_id}_") and os.path.isdir(entry_path):
                audio_path = None
                lines_path = None
                raw_path = None
                for f in os.listdir(entry_path):
                    if f.endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma', '.aac')):
                        audio_path = os.path.join(entry_path, f)
                    if f.endswith('.lines.json'):
                        lines_path = os.path.join(entry_path, f)
                    if f.endswith('.raw.json'):
                        raw_path = os.path.join(entry_path, f)

                if lines_path:
                    with open(lines_path, "r", encoding="utf-8") as f:
                        lines_data = json.load(f)
                    return KaraokeResponse(task_id=task_id, lines=lines_data, audio_path=audio_path)
                elif raw_path:
                    with open(raw_path, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                    return KaraokeResponse(
                        task_id=task_id,
                        lines=[{"text": raw_data.get("text", ""), "start_time": 0,
                               "end_time": raw_data.get("duration", 0), "words": raw_data.get("words", [])}],
                        audio_path=audio_path,
                    )

            # Flat file: {task_id}*.lines.json or {task_id}*.raw.json in intermediate_dir
            if intermediate_dir and search_dir == intermediate_dir:
                if entry.startswith(task_id) and entry.endswith('.lines.json'):
                    with open(entry_path, "r", encoding="utf-8") as f:
                        lines_data = json.load(f)
                    # Try to find audio in temp output_dir
                    audio_path = None
                    temp_dir = tempfile.gettempdir()
                    for d in os.listdir(temp_dir):
                        if d.startswith(f"asr_{task_id}_") and os.path.isdir(os.path.join(temp_dir, d)):
                            ad = os.path.join(temp_dir, d)
                            for f in os.listdir(ad):
                                if f.endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma', '.aac')):
                                    audio_path = os.path.join(ad, f)
                                    break
                    return KaraokeResponse(task_id=task_id, lines=lines_data, audio_path=audio_path)
                if entry.startswith(task_id) and entry.endswith('.raw.json'):
                    with open(entry_path, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                    audio_path = None
                    temp_dir = tempfile.gettempdir()
                    for d in os.listdir(temp_dir):
                        if d.startswith(f"asr_{task_id}_") and os.path.isdir(os.path.join(temp_dir, d)):
                            ad = os.path.join(temp_dir, d)
                            for f in os.listdir(ad):
                                if f.endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma', '.aac')):
                                    audio_path = os.path.join(ad, f)
                                    break
                    return KaraokeResponse(
                        task_id=task_id,
                        lines=[{"text": raw_data.get("text", ""), "start_time": 0,
                               "end_time": raw_data.get("duration", 0), "words": raw_data.get("words", [])}],
                        audio_path=audio_path,
                    )

    raise HTTPException(404, "No subtitle data found for task")
