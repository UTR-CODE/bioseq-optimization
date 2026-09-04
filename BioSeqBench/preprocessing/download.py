"""Streaming, resumable downloader with SHA-256 recording."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path, expected_sha256: str | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (not expected_sha256 or sha256(destination) == expected_sha256):
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "BioSeqBench/0.1"})
    if start:
        request.add_header("Range", f"bytes={start}-")
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - manifest URLs are explicit
        mode = "ab" if start and response.status == 206 else "wb"
        with partial.open(mode) as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
    partial.replace(destination)
    actual = sha256(destination)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {destination}: {actual}")
    return destination


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())
