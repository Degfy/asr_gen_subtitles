# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ASR Pipeline** — Multi-stage subtitle generation pipeline using Qwen3-ASR with forced alignment. Converts audio → structured subtitles (SRT/ASS) with sentence breaking, text correction via CSV fixes, and karaoke-style highlighting. Provides both CLI and REST API interfaces.

## Architecture

### 4-Stage Pipeline

```
Stage 1: ASR + Forced Alignment → flat word-level JSON
Stage 2: Sentence Breaking      → paragraphs (。！？), then lines (max_chars)
Stage 3: CSV Fix                → apply multiple fix CSVs for text correction
Stage 4: Render                  → SRT and ASS output with style selection
```

Each stage saves intermediate JSON for debugging and resume support.

### Core Modules (`asr/` package)

| Module | Responsibility |
|--------|----------------|
| `asr/pipeline.py` | Pipeline orchestrator, stage coordination, resume support |
| `asr/engine.py` | Qwen3-ASR engine (CUDA/MLX backends), forced alignment |
| `asr/subtitle.py` | SRT/ASS rendering, karaoke effect tags (\kf) |
| `asr/text_utils.py` | CJK character handling, punctuation, character counting |
| `asr/platform.py` | Backend detection (cuda/mlx) |
| `asr/model_path.py` | Model path resolution/caching |
| `asr/download_models.py` | Model download script |

### Entry Points

| Module | Responsibility |
|--------|----------------|
| `main.py` | CLI entry point |
| `api.py` | FastAPI REST service |

### Backend Strategy

- **CUDA**: Linux + NVIDIA GPU — uses `qwen_asr` + PyTorch with `Qwen3ForcedAligner`
- **MLX**: macOS + Apple Silicon — uses `mlx-audio` with separate ASR + aligner models
- Auto-detected via `asr.platform.get_backend()`

### Data Structures

- `WordTimestamp` — single word with start/end time
- `ASRResult` — flat word list (no segment grouping at ASR level)
- `Paragraph` — sentence-level unit bounded by 。！？ punctuation
- `SubtitleLine` — single subtitle line with timing, word-level data, pause_after
- `ASSSubtitleStyle` — ASS style config (font, colors, karaoke mode)

### Character Counting

CJK characters = 1, non-CJK letters = 0.5, punctuation = 0. Used for max_chars line breaking.

## Commands

```bash
# Install dependencies
pip install -e ".[cuda]"  # for CUDA
pip install -e ".[mlx]"   # for Apple Silicon
pip install -e ".[dev]"   # dev dependencies

# CLI usage
python main.py transcribe audio.wav --output ./subs
python main.py transcribe audio.wav --fmt ass --style default
python main.py serve --port 8000

# API server
python main.py serve --host 0.0.0.0 --port 8000

# Python API
from asr.pipeline import run_pipeline
result = run_pipeline('audio.wav', output_dir='./subs')
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check with backend info |
| POST | `/transcribe` | Full pipeline (audio → SRT/ASS) |
| POST | `/transcribe/text` | Text-only transcription |
| GET | `/download/{task_id}/{fmt}` | Download generated subtitles |

## Key Implementation Notes

- Pipeline resumes from any stage via `resume_from` param (stage names: asr, break, fix, render)
- `split_line_after()` in pipeline splits a line at specific text, preserving word-level timing
- CSV fixes in `fix_dir` applied sequentially (fix_1.csv, fix_2.csv, ...)
- ASS karaoke uses `\kf` (fill) mode by default with golden highlight + white dim
- Audio > 5 min on CUDA is automatically chunked into 30s segments
