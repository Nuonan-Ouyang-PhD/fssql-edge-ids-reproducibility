#!/usr/bin/env python3
"""Build a deterministic SHA-256 manifest for the rebuild repository."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "provenance" / "PROVENANCE_MANIFEST.csv"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", "third_party"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


rows = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or EXCLUDED_PARTS.intersection(path.parts):
        continue
    if path == OUTPUT or path.suffix == ".part":
        continue
    rows.append((path.relative_to(ROOT).as_posix(), path.stat().st_size, digest(path)))

with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["relative_path", "size_bytes", "sha256"])
    writer.writerows(rows)

print(f"wrote {len(rows)} entries to {OUTPUT}")
