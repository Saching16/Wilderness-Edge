#!/usr/bin/env python3
"""Download the manifested source PDFs into the corpus directory.

Only fetches sources already vetted and recorded in `sources.manifest.json`, saving each
under its manifest filename so `build_vector_db.py` can match it to its license and citation.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "WildernessEdge-OfflineTools/1.0 (corpus fetch; contact: repository maintainer)"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def download(url: str, destination: Path, timeout: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = (response.headers.get("Content-Type") or "").lower()
        payload = response.read()

    if not payload.startswith(b"%PDF"):
        raise ValueError(f"response was not a PDF (Content-Type: {content_type or 'unknown'})")

    destination.write_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch vetted source PDFs listed in the manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", default="sources.manifest.json")
    parser.add_argument("--output-dir", default="sources")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for entry in manifest.get("sources", []):
        filename = entry["filename"]
        destination = output_dir / filename

        if destination.exists() and not args.force:
            log(f"= {filename} (already present)")
            continue

        urls = [u for u in (entry.get("url"), entry.get("mirror_url")) if u]
        if not urls:
            log(f"! {filename}: no URL in manifest; download manually")
            failures.append(filename)
            continue

        for index, url in enumerate(urls):
            try:
                log(f"> {filename} <- {url}")
                download(url, destination, args.timeout)
                log(f"  saved {destination.stat().st_size / 1024 / 1024:.1f} MB")
                break
            except (urllib.error.URLError, ValueError, TimeoutError) as exc:
                log(f"  failed: {exc}")
                if index == len(urls) - 1:
                    failures.append(filename)

    if failures:
        log(
            "\nCould not fetch: "
            + ", ".join(failures)
            + "\nDownload these manually into "
            + str(output_dir)
            + " using the URLs in SOURCES.md, keeping the manifest filenames."
        )
        return 1

    log(f"\nAll manifested sources present in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
