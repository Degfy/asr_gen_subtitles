"""Configuration loading from .env file."""

import os
import sys
from pathlib import Path
from typing import Optional

# Default model cache directory
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "asr-models"

# Config file path
_CONFIG_FILE = Path(".env")


def _load_config():
    """Load config from .env file, set environment variables."""
    if not _CONFIG_FILE.exists():
        return

    for line in _CONFIG_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and not os.environ.get(key):
            os.environ[key] = value


def get_model_dir() -> Path:
    """Get model directory from config or default."""
    path = os.environ.get("ASR_MODEL_DIR")
    if path:
        return Path(path)
    return DEFAULT_MODEL_DIR


def get_model_size() -> str:
    """Get default model size from config."""
    return os.environ.get("ASR_MODEL_SIZE", "1.7B")


# Load config at module import
_load_config()