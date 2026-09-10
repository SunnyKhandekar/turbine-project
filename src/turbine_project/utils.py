from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def portable_path(path: Path, root: Path) -> str:
    """Render a path relative to ``root`` using forward slashes.

    Output artifacts are persisted into JSON and then read back on a
    different machine/OS (e.g. a local Windows run deployed to Streamlit
    Cloud on Linux). Storing an absolute, OS-specific path breaks that
    round trip, so every path we write to a report must go through here.
    """
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return path.name
    return relative.as_posix()
