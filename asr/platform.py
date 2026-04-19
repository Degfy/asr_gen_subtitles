"""Platform detection for ASR backend selection."""

import platform
import sys

_BACKEND = None


def get_backend() -> str:
    """Detect and return the appropriate backend: 'cuda' or 'mlx'.

    CUDA: Linux with NVIDIA GPU
    MLX:  macOS with Apple Silicon (M1/M2/M3/M4)
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    system = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine in ("arm64", "aarch64"):
        _BACKEND = "mlx"
        return _BACKEND

    if system == "Linux":
        try:
            import torch
            if torch.cuda.is_available():
                _BACKEND = "cuda"
                return _BACKEND
        except ImportError:
            pass

    raise RuntimeError(
        "No supported backend found. CUDA requires Linux+NVIDIA GPU; "
        "MLX requires macOS with Apple Silicon. Install torch for CUDA "
        "or mlx-audio for MLX."
    )


def is_mlx() -> bool:
    """Check if running on MLX (Apple Silicon)."""
    return get_backend() == "mlx"


def is_cuda() -> bool:
    """Check if running on CUDA (NVIDIA GPU)."""
    return get_backend() == "cuda"
