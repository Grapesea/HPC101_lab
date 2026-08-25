"""Allow the repository scripts to run without installing the local package first."""

from __future__ import annotations

import sys
from pathlib import Path


def add_local_src_to_path() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    src_text = str(src_dir)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
