from __future__ import annotations

import argparse
import json
from pathlib import Path

from .download import download_file, load_manifest


def fetch_dataset(name: str, manifest_path: Path, raw_root: Path, include_large: bool = False) -> list[Path]:
    manifest = load_manifest(manifest_path)
    entry = manifest["datasets"][name]
    if entry.get("manual_only"):
        raise RuntimeError("Manual access required:\n" + "\n".join(entry["manual_steps"]))
    downloaded = []
    for item in entry.get("files", []):
        if item.get("large") and not include_large:
            continue
        destination = raw_root / name / item["filename"]
        downloaded.append(download_file(item["url"], destination, item.get("sha256")))
    receipt = raw_root / name / "download_receipt.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({"dataset": name, "files": [str(x) for x in downloaded], "source": entry}, indent=2) + "\n")
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public BioSeqBench source files")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", default="BioSeqBench/configs/sources.json")
    parser.add_argument("--raw-root", default="BioSeqBench/raw_data")
    parser.add_argument("--include-large", action="store_true", help="also fetch large optional raw archives")
    args = parser.parse_args()
    files = fetch_dataset(args.dataset, Path(args.manifest), Path(args.raw_root), args.include_large)
    print("\n".join(map(str, files)))


if __name__ == "__main__":
    main()
