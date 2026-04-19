# ASR Pipeline - Multi-stage Subtitle Generation

Multi-stage subtitle generation pipeline using Qwen3-ASR with forced alignment. Converts audio to structured subtitles (SRT/ASS) with sentence breaking, text correction via CSV fixes, and karaoke-style highlighting. Provides both CLI and REST API interfaces.

## Architecture

### 4-Stage Pipeline

```
Stage 1: ASR + Forced Alignment → flat word-level JSON
Stage 2: Sentence Breaking      → paragraphs (。！？), then lines (max_chars)
Stage 3: CSV Fix                → apply multiple fix CSVs for text correction
Stage 4: Render                  → SRT and ASS output with style selection
```

Each stage saves intermediate JSON for debugging and resume support.

### Backend Strategy

- **CUDA**: Linux + NVIDIA GPU — uses `qwen-asr` + PyTorch with `Qwen3ForcedAligner`
- **MLX**: macOS + Apple Silicon — uses `mlx-audio` with separate ASR + aligner models
- Auto-detected via `scripts.shared.platform.get_backend()`

### Core Modules

| Module | Responsibility |
|--------|----------------|
| `asr_pipeline.py` | Pipeline orchestrator, stage coordination, resume support |
| `qwen_asr_engine.py` | Qwen3-ASR engine (CUDA/MLX backends), forced alignment |
| `subtitle_gen.py` | SRT/ASS rendering, karaoke effect tags (`\kf`) |
| `api.py` | FastAPI REST service |
| `main.py` | CLI entry point |
| `scripts/shared/platform.py` | Backend detection (cuda/mlx) |
| `scripts/shared/model_path.py` | Model path resolution/caching |

### Data Structures

- `WordTimestamp` — single word with start/end time
- `ASRResult` — flat word list (no segment grouping at ASR level)
- `Paragraph` — sentence-level unit bounded by 。！？ punctuation
- `SubtitleLine` — single subtitle line with timing, word-level data, pause_after
- `ASSSubtitleStyle` — ASS style config (font, colors, karaoke mode)

### Character Counting

CJK characters = 1, non-CJK letters = 0.5, punctuation = 0. Used for max_chars line breaking.

## Installation

```bash
# Install dependencies
pip install -e ".[cuda]"  # for CUDA (Linux + NVIDIA GPU)
pip install -e ".[mlx]"   # for Apple Silicon (macOS M1/M2/M3/M4)
pip install -e ".[dev]"   # dev dependencies
```

## CLI Usage

```bash
# Basic transcription (SRT output)
python main.py transcribe audio.wav --output ./subs

# ASS format with style
python main.py transcribe audio.wav --fmt ass --style default

# With language hint
python main.py transcribe audio.wav --language Chinese

# Smaller model (faster, lower accuracy)
python main.py transcribe audio.wav --model-size 0.6B

# Custom max characters per line
python main.py transcribe audio.wav --max-chars 20

# Apply text corrections from CSV files
python main.py transcribe audio.wav --fix-dir ./fixes

# Resume from a specific stage
python main.py transcribe audio.wav --resume-from render
```

## API Server

```bash
# Start API server
python main.py serve --host 0.0.0.0 --port 8000

# With auto-reload (development)
python main.py serve --reload
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check with backend info |
| POST | `/transcribe` | Full pipeline (audio → SRT/ASS) |
| POST | `/transcribe/text` | Text-only transcription |
| GET | `/download/{task_id}/{fmt}` | Download generated subtitles |

## Python API

```python
from asr_pipeline import run_pipeline

result = run_pipeline(
    audio_path='audio.wav',
    output_dir='./subs',
    fmt='srt',           # 'srt', 'ass', or 'all'
    ass_style='default', # ASS style name
    fix_dir='./fixes',   # optional CSV fix directory
    language='Chinese',   # optional language hint
    model_size='1.7B',   # '1.7B' or '0.6B'
    max_chars=14,        # max characters per subtitle line
    resume_from=None,   # 'asr', 'break', 'fix', 'render', or None
)
```

## CSV Fix Format

Fix CSV files should be named `fix_1.csv`, `fix_2.csv`, etc. and placed in a fix directory:

```csv
# Format: original_text,replacement_text
# Empty replacement deletes the line
错误,正确
要删除的文本,
```

## Key Implementation Notes

- Pipeline resumes from any stage via `resume_from` param
- `split_line_after()` in pipeline splits a line at specific text, preserving word-level timing
- CSV fixes in `fix_dir` applied sequentially (fix_1.csv, fix_2.csv, ...)
- ASS karaoke uses `\kf` (fill) mode by default with golden highlight + white dim
- Audio > 5 min on CUDA is automatically chunked into 30s segments
- Long segments are split using smart algorithm that finds largest time gaps
