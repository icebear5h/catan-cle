"""Small durable receipts and installed Miles revision identity."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from sft.json_types import JsonLike

from .contracts import MILES_COMMIT, require


def receipt_directory() -> Path:
    value = os.environ.get("MILES_SFT_RECEIPT_DIR", "")
    require(bool(value) and Path(value).is_absolute(), "MILES_SFT_RECEIPT_DIR must be an absolute path")
    return Path(value)


def write_receipt(path: Path, payload: JsonLike) -> None:
    """Exclusive writes reject reused runs; interrupted JSON is visibly invalid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_miles_revision(module_file: str) -> str:
    """Verify the real checkout supplying imports, not an environment-version claim."""
    module = Path(module_file).resolve()
    result = subprocess.run(
        ["git", "-C", str(module.parent), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    revision = result.stdout.strip()
    require(revision == MILES_COMMIT, f"Miles revision mismatch: {revision}")
    return revision
