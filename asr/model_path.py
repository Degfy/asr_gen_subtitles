"""Model path resolution with caching and auto-download."""

import os
from pathlib import Path
from typing import Optional

from asr.config import get_model_dir, DEFAULT_MODEL_DIR

# Cache for resolved model paths
_model_cache: dict[str, str] = {}


def _find_model_in_cache(model_id: str, cache_dir: Optional[str] = None) -> Optional[Path]:
    """Search cache directory for a model, handling ModelScope path transformations.

    ModelScope may transform the model_id path (e.g., dots become triple underscores),
    so we search to find the actual model path.

    Returns:
        Path to the model directory if found, None otherwise.
    """
    cache_base = Path(cache_dir or str(get_model_dir()))

    if "/" in model_id:
        org, repo = model_id.split("/", 1)
        # ModelScope transforms dots to triple underscores in repo names
        # The repo name is like "Qwen3-ASR-1.7B-8bit" or "Qwen3-ForcedAligner-0.6B-8bit"
        # We match on the first two dash-separated components to distinguish ASR from ForcedAligner
        repo_parts = repo.split("-")
        repo_prefix = "-".join(repo_parts[:2])  # e.g., "Qwen3-ASR" or "Qwen3-ForcedAligner"
        search_dir = cache_base / org
        if search_dir.exists():
            for subdir in search_dir.iterdir():
                if subdir.is_dir():
                    # Check if subdir.name matches the repo prefix
                    # ModelScope may transform dots to triple underscores
                    normalized_name = subdir.name.replace("___", ".").replace("_", "-")
                    if normalized_name.startswith(repo_prefix):
                        if _has_model_files(subdir):
                            return subdir

    # Fallback: try direct path construction
    target_dir = cache_base / model_id.replace("/", "_")
    if _has_model_files(target_dir):
        return target_dir

    return None


def resolve_model_path(model_id: str, cache_dir: Optional[str] = None) -> str:
    """Resolve model ID to local path.

    If model_id is a local path, return it directly.
    Otherwise, return the cached download path or download on demand.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-ASR-1.7B")
        cache_dir: Optional custom cache directory

    Returns:
        Local path to the model
    """
    cache_key = f"{cache_dir or str(get_model_dir())}/{model_id}"

    if cache_key in _model_cache:
        cached_path = _model_cache[cache_key]
        if _has_model_files(Path(cached_path)):
            return cached_path
        del _model_cache[cache_key]

    # If it's already a local path, use it directly
    if os.path.exists(model_id):
        _model_cache[cache_key] = model_id
        return model_id

    # Search for model in cache (handles ModelScope path transformations)
    found = _find_model_in_cache(model_id, cache_dir)
    if found:
        _model_cache[cache_key] = str(found)
        return str(found)

    # Return expected path for download
    target_dir = Path(cache_dir or str(get_model_dir())) / model_id.replace("/", "_")
    _model_cache[cache_key] = str(target_dir)
    return str(target_dir)


def _has_model_files(model_dir: Path) -> bool:
    """Check if directory contains model files (config.json, safetensors, etc.)."""
    if not model_dir.exists():
        return False
    required = {"config.json"}
    if required.issubset({f.name for f in model_dir.iterdir()}):
        return True
    for subdir in model_dir.rglob("."):
        if subdir == model_dir:
            continue
        if required.issubset({f.name for f in subdir.iterdir()}):
            return True
    return False


def _find_model_root(model_dir: Path) -> Path:
    """Find the actual model root directory containing config.json."""
    if (model_dir / "config.json").exists():
        return model_dir
    for subdir in model_dir.rglob("."):
        if subdir == model_dir:
            continue
        if (subdir / "config.json").exists():
            return subdir
    return model_dir


def is_model_cached(model_id: str, cache_dir: Optional[str] = None) -> bool:
    """Check if a model is already cached locally."""
    return _find_model_in_cache(model_id, cache_dir) is not None


def ensure_model(model_id: str, cache_dir: Optional[str] = None) -> bool:
    """Ensure model exists locally, download if missing.

    Args:
        model_id: ModelScope/HuggingFace model ID
        cache_dir: Optional custom cache directory

    Returns:
        True if model was downloaded or already exists, False on failure
    """
    if is_model_cached(model_id, cache_dir):
        return True

    target_dir = Path(cache_dir or str(get_model_dir())) / model_id.replace("/", "_")

    try:
        from modelscope import snapshot_download
        print(f"Downloading {model_id} to {target_dir}...")
        actual_path = snapshot_download(model_id, cache_dir=str(target_dir.parent))
        return _has_model_files(Path(actual_path))
    except Exception as e:
        print(f"Failed to download {model_id}: {e}")
        return False


def clear_cache():
    """Clear the model path cache."""
    _model_cache.clear()
