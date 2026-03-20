"""
Shared utility functions.
"""

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create a directory (and all parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path
