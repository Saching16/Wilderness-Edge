#!/usr/bin/env python3
"""Build the flora & fauna hazard pack into an existing `protocols.db`.

Each entry in `species.manifest.json` becomes:

  1. One row in `chunks`, embedded with the *same* sentence-transformer model used by
     `build_vector_db.py`, so `VectorRAGManager.search(...)` retrieves hazard cards through
     the existing code path with no Swift search changes at all.
  2. One row in `species` carrying the structured field-guide detail.
  3. N rows in `chunk_images` holding licensed reference photographs as JPEG blobs, shown
     to the responder so a *human* confirms the identification.

Licensing is machine-enforced, exactly as `build_vector_db.py` enforces PDF provenance:
every candidate image's license is read from the Wikimedia Commons API and checked against
ALLOWED_LICENSE_PATTERNS. Anything not clearly PD/CC0/CC BY/CC BY-SA is dropped, and the
build fails if a species ends up with zero admissible images. Resolved files are pinned in
`species.images.lock.json` so a rebuild is reproducible and the licenses are reviewable in
a diff rather than trusted to a live API call.

Usage:
    python build_species_pack.py --dry-run          # resolve + report, write nothing
    python build_species_pack.py --refresh-images   # re-resolve categories, rewrite lockfile
    python build_species_pack.py                    # build into build/protocols.db
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "WildernessEdge-OfflineTools/1.0 (offline corpus build; contact: repository maintainer)"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIM = 384

# Only these licenses may enter the app bundle. NonCommercial and NoDerivatives are
# excluded deliberately: this is a redistributable compiled binary, and NC/ND terms are
# incompatible with shipping it. Unknown or unparseable licenses fail closed.
ALLOWED_LICENSE_PATTERNS = [
    re.compile(r"^public domain", re.IGNORECASE),
    re.compile(r"^cc0", re.IGNORECASE),
    re.compile(r"^cc[-\s]?pd", re.IGNORECASE),
    re.compile(r"^cc by(?![-\s]?n[cd])[-\s]?(sa[-\s]?)?\d", re.IGNORECASE),
]

# Files that are technically in-category but useless (or misleading) as a visual ID aid.
NAME_DENYLIST = re.compile(
    r"\b(map|range|distribution|chart|graph|diagram|logo|sign|stamp|label|"
    r"phylogen|cladogram|locator|icon|banner)\b",
    re.IGNORECASE,
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


class BuildError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------
# Commons resolution
# --------------------------------------------------------------------------------------


@dataclass
class ResolvedImage:
    title: str
    license: str
    attribution: str
    credit: str
    descriptor_url: str
    thumb_url: str
    width: int
    height: int


@dataclass
class SpeciesCard:
    slug: str
    common_name: str
    scientific_name: str
    hazard_class: str
    region: str
    identification: str
    lookalikes: str
    field_response: list[str]
    do_not: list[str]
    source_citation: str
    commons_categories: list[str]
    max_images: int
    images: list[ResolvedImage] = field(default_factory=list)


def clean_html(value: str) -> str:
    return WS_RE.sub(" ", TAG_RE.sub("", value or "")).strip()


# Commons throttles anonymous clients hard. Stay well under one request per second and
# back off generously on 429 rather than hammering a volunteer-funded API.
_last_request_at = 0.0
REQUEST_DELAY = 1.2


def _throttle() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - elapsed)
    _last_request_at = time.monotonic()


def api_get(params: dict[str, str], timeout: int, retries: int = 4) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    last_error: Exception | None = None
    for attempt in range(retries):
        _throttle()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            # 429/503 mean "slow down", not "broken" — wait substantially longer.
            time.sleep((8.0 if exc.code in (429, 503) else 1.5) * (attempt + 1))
        except Exception as exc:  # noqa: BLE001 - retried, then surfaced as BuildError
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise BuildError(f"Commons API request failed after {retries} attempts: {last_error}")


def license_is_allowed(license_name: str) -> bool:
    name = (license_name or "").strip()
    if not name:
        return False
    return any(pattern.match(name) for pattern in ALLOWED_LICENSE_PATTERNS)


def resolve_category(
    category: str, thumb_width: int, min_width: int, timeout: int
) -> list[ResolvedImage]:
    """Return every acceptably-licensed image file in a Commons category."""
    payload = api_get(
        {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": "60",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size",
            "iiurlwidth": str(thumb_width),
        },
        timeout=timeout,
    )

    pages = payload.get("query", {}).get("pages", [])
    resolved: list[ResolvedImage] = []

    for page in pages:
        title = page.get("title", "")
        if not title.lower().endswith(IMAGE_EXTENSIONS):
            continue
        if NAME_DENYLIST.search(title):
            continue

        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        meta = info.get("extmetadata", {})

        license_name = clean_html(meta.get("LicenseShortName", {}).get("value", ""))
        if not license_is_allowed(license_name):
            continue

        if int(info.get("width") or 0) < min_width:
            continue

        thumb_url = info.get("thumburl") or info.get("url")
        if not thumb_url:
            continue

        resolved.append(
            ResolvedImage(
                title=title,
                license=license_name,
                attribution=clean_html(meta.get("Artist", {}).get("value", "")) or "Unknown author",
                credit=clean_html(meta.get("Credit", {}).get("value", "")),
                descriptor_url=info.get("descriptionurl", ""),
                thumb_url=thumb_url,
                width=int(info.get("thumbwidth") or info.get("width") or 0),
                height=int(info.get("thumbheight") or info.get("height") or 0),
            )
        )

    # Deterministic ordering so repeat runs pick the same files.
    resolved.sort(key=lambda image: image.title)
    return resolved


def resolve_species_images(
    card: SpeciesCard, thumb_width: int, min_width: int, timeout: int
) -> list[ResolvedImage]:
    seen: set[str] = set()
    picked: list[ResolvedImage] = []

    for category in card.commons_categories:
        for image in resolve_category(category, thumb_width, min_width, timeout):
            if image.title in seen:
                continue
            seen.add(image.title)
            picked.append(image)
            if len(picked) >= card.max_images:
                return picked
    return picked


# --------------------------------------------------------------------------------------
# Card text
# --------------------------------------------------------------------------------------


def retrieval_text(card: SpeciesCard) -> str:
    """The text that gets embedded and spoken. Written so a plain-language voice query
    ('brown spider with a violin mark bit me') lands on the right card."""
    parts = [
        f"{card.common_name} ({card.scientific_name}). Hazard: {card.hazard_class}.",
        f"Where found: {card.region}.",
        f"Field identification: {card.identification}",
    ]
    if card.lookalikes:
        parts.append(f"Commonly confused with: {card.lookalikes}")
    parts.append(
        "Field response checklist: "
        + " ".join(f"{index}. {step}" for index, step in enumerate(card.field_response, start=1))
    )
    if card.do_not:
        parts.append("Do not: " + " ".join(card.do_not))
    parts.append(
        "Identification must be confirmed by the responder against the reference images; "
        "this assistant does not identify species."
    )
    return "\n".join(parts)


def build_citation(card: SpeciesCard, citation_prefix: str) -> str:
    # Plain hyphen, not an em dash: this string is both spoken by AVSpeechSynthesizer and
    # printed to a Windows console during the build.
    return (
        f"[Source: {citation_prefix} - {card.common_name} ({card.scientific_name}); "
        f"derived from {card.source_citation}]"
    )


# --------------------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------------------

SCHEMA_ADDITIONS = """
CREATE TABLE IF NOT EXISTS species (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id        INTEGER NOT NULL REFERENCES chunks(id),
    slug            TEXT NOT NULL UNIQUE,
    common_name     TEXT NOT NULL,
    scientific_name TEXT NOT NULL,
    hazard_class    TEXT NOT NULL,
    region          TEXT NOT NULL DEFAULT '',
    identification  TEXT NOT NULL,
    lookalikes      TEXT NOT NULL DEFAULT '',
    field_response  TEXT NOT NULL,
    do_not          TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS chunk_images (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id     INTEGER NOT NULL REFERENCES chunks(id),
    species_slug TEXT NOT NULL,
    ordinal      INTEGER NOT NULL,
    license      TEXT NOT NULL,
    attribution  TEXT NOT NULL,
    source_url   TEXT NOT NULL,
    mime         TEXT NOT NULL DEFAULT 'image/jpeg',
    width        INTEGER NOT NULL,
    height       INTEGER NOT NULL,
    bytes        BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_species_chunk ON species(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_images_chunk ON chunk_images(chunk_id);
"""


def encode_vector(vector: np.ndarray) -> bytes:
    """Little-endian float32, matching build_vector_db.py so Swift can memcpy into [Float]."""
    return np.ascontiguousarray(vector, dtype="<f4").tobytes()


def download_image(url: str, timeout: int, retries: int = 4) -> bytes:
    """Fetch one thumbnail, throttled and backed off exactly like the API calls.

    Commons renders thumbnails on demand, so a cold cache plus an impatient client is the
    fastest way to earn a 429. This shares the module-wide request clock with `api_get`.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    last_error: Exception | None = None
    for attempt in range(retries):
        _throttle()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            time.sleep((8.0 if exc.code in (429, 503) else 1.5) * (attempt + 1))
        except Exception as exc:  # noqa: BLE001 - retried, then surfaced as BuildError
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise BuildError(f"Image download failed after {retries} attempts ({url}): {last_error}")


def transcode(payload: bytes, max_edge: int, quality: int) -> tuple[bytes, int, int]:
    """Re-encode to a compact, metadata-stripped JPEG. Bundled assets should not carry
    EXIF GPS or camera serials into the app binary."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise BuildError("Pillow is required: pip install -r requirements.txt") from exc

    with Image.open(io.BytesIO(payload)) as image:
        image = image.convert("RGB")
        image.thumbnail((max_edge, max_edge), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue(), image.width, image.height


# --------------------------------------------------------------------------------------
# Manifest / lockfile
# --------------------------------------------------------------------------------------


def load_manifest(path: Path) -> tuple[dict, list[SpeciesCard]]:
    if not path.is_file():
        raise BuildError(f"Species manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))

    cards: list[SpeciesCard] = []
    for entry in payload.get("species", []):
        missing = [
            key
            for key in ("slug", "common_name", "scientific_name", "hazard_class",
                        "identification", "field_response", "source_citation",
                        "commons_categories")
            if not entry.get(key)
        ]
        if missing:
            raise BuildError(f"Species entry {entry.get('slug', '?')} missing keys: {missing}")

        cards.append(
            SpeciesCard(
                slug=entry["slug"],
                common_name=entry["common_name"],
                scientific_name=entry["scientific_name"],
                hazard_class=entry["hazard_class"],
                region=entry.get("region", ""),
                identification=entry["identification"],
                lookalikes=entry.get("lookalikes", ""),
                field_response=list(entry["field_response"]),
                do_not=list(entry.get("do_not", [])),
                source_citation=entry["source_citation"],
                commons_categories=list(entry["commons_categories"]),
                max_images=int(entry.get("max_images", 3)),
            )
        )

    if not cards:
        raise BuildError("Species manifest contains no entries.")
    return payload, cards


def load_lockfile(path: Path) -> dict[str, list[ResolvedImage]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    locked: dict[str, list[ResolvedImage]] = {}
    for slug, images in payload.get("species", {}).items():
        locked[slug] = [ResolvedImage(**image) for image in images]
    return locked


def write_lockfile(path: Path, cards: list[SpeciesCard]) -> None:
    payload = {
        "_comment": (
            "Pinned Wikimedia Commons files per species, with the license read from the "
            "Commons API at resolution time. Regenerate with --refresh-images. Review "
            "license changes in this diff before shipping; build_species_pack.py re-checks "
            "every license against ALLOWED_LICENSE_PATTERNS on each build."
        ),
        "species": {
            card.slug: [image.__dict__ for image in card.images] for card in cards
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------


def build(args: argparse.Namespace) -> int:
    global REQUEST_DELAY
    REQUEST_DELAY = args.request_delay

    manifest_path = Path(args.manifest).expanduser().resolve()
    lock_path = Path(args.lockfile).expanduser().resolve()
    payload, cards = load_manifest(manifest_path)

    if args.limit_species:
        cards = cards[: args.limit_species]

    citation_prefix = payload.get("citation_prefix", "Wilderness Edge Field Hazard Card")

    locked = {} if args.refresh_images else load_lockfile(lock_path)
    if locked:
        log(f"Using pinned image set from {lock_path.name}")

    # ---- Resolve images -------------------------------------------------------------
    for card in cards:
        if card.slug in locked and locked[card.slug]:
            card.images = locked[card.slug]
            # Re-verify pinned licenses; a Commons re-license must not slip through.
            bad = [image for image in card.images if not license_is_allowed(image.license)]
            if bad:
                raise BuildError(
                    f"{card.slug}: pinned image(s) carry a disallowed license: "
                    + ", ".join(f"{image.title} ({image.license})" for image in bad)
                )
        else:
            log(f"Resolving images for {card.slug} …")
            card.images = resolve_species_images(
                card, args.thumb_width, args.min_source_width, args.timeout
            )

        if not card.images:
            raise BuildError(
                f"{card.slug}: no acceptably-licensed images found in categories "
                f"{card.commons_categories}. Add another category or relax --min-source-width."
            )

    if args.refresh_images or not locked:
        write_lockfile(lock_path, cards)
        log(f"Wrote {lock_path}")

    # ---- Report ---------------------------------------------------------------------
    total_images = sum(len(card.images) for card in cards)
    log("")
    log(f"{len(cards)} hazard cards / {total_images} licensed images")
    for card in cards:
        licenses = ", ".join(sorted({image.license for image in card.images}))
        log(f"  {card.slug:<32} {len(card.images)} image(s)  [{licenses}]")

    if args.dry_run:
        log("")
        log("Sample card text:")
        log("-" * 78)
        log(build_citation(cards[0], citation_prefix))
        log(retrieval_text(cards[0]))
        log("-" * 78)
        log("Dry run: nothing downloaded, nothing written.")
        return 0

    # ---- Embed ----------------------------------------------------------------------
    database_path = Path(args.database).expanduser().resolve()
    if not database_path.is_file():
        raise BuildError(
            f"{database_path} not found. Run build_vector_db.py first — the hazard pack "
            "extends the protocol corpus, it does not replace it."
        )

    log("")
    log(f"Loading embedding model {args.model} …")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    texts = [retrieval_text(card) for card in cards]
    vectors = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    if vectors.shape[1] != args.expected_dim:
        raise BuildError(
            f"Embedding dimensionality {vectors.shape[1]} != expected {args.expected_dim}. "
            "The hazard pack must share protocols.db's embedding space exactly."
        )

    # ---- Write ----------------------------------------------------------------------
    connection = sqlite3.connect(database_path)
    connection.executescript(SCHEMA_ADDITIONS)

    if args.replace:
        connection.execute("DELETE FROM chunk_images")
        connection.execute("DELETE FROM species")
        connection.execute(
            "DELETE FROM chunks WHERE source_id IN (SELECT id FROM sources WHERE filename = ?)",
            ("species.manifest.json",),
        )
        connection.execute("DELETE FROM sources WHERE filename = ?", ("species.manifest.json",))

    cursor = connection.execute(
        "INSERT INTO sources (filename, title, publisher, license, url, citation_prefix)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            "species.manifest.json",
            "Wilderness Edge Field Hazard Cards (Flora & Fauna)",
            "Wilderness Edge, compiled from US federal public-domain sources",
            payload.get("license", "See species.manifest.json"),
            "",
            citation_prefix,
        ),
    )
    source_id = cursor.lastrowid

    image_bytes_total = 0
    for card, vector in zip(cards, vectors):
        text = retrieval_text(card)
        cursor = connection.execute(
            "INSERT INTO chunks (source_id, section, page_start, page_end, citation, text,"
            " token_count, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_id,
                f"{card.hazard_class} — {card.common_name}",
                0,
                0,
                build_citation(card, citation_prefix),
                text,
                len(text.split()),
                encode_vector(vector),
            ),
        )
        chunk_id = cursor.lastrowid

        connection.execute(
            "INSERT INTO species (chunk_id, slug, common_name, scientific_name, hazard_class,"
            " region, identification, lookalikes, field_response, do_not)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                chunk_id,
                card.slug,
                card.common_name,
                card.scientific_name,
                card.hazard_class,
                card.region,
                card.identification,
                card.lookalikes,
                "\n".join(card.field_response),
                "\n".join(card.do_not),
            ),
        )

        for ordinal, image in enumerate(card.images):
            log(f"  fetching {card.slug} [{ordinal + 1}/{len(card.images)}] {image.title}")
            raw = download_image(image.thumb_url, args.timeout)
            jpeg, width, height = transcode(raw, args.max_edge, args.jpeg_quality)
            image_bytes_total += len(jpeg)
            connection.execute(
                "INSERT INTO chunk_images (chunk_id, species_slug, ordinal, license,"
                " attribution, source_url, mime, width, height, bytes)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk_id,
                    card.slug,
                    ordinal,
                    image.license,
                    image.attribution,
                    image.descriptor_url or image.thumb_url,
                    "image/jpeg",
                    width,
                    height,
                    sqlite3.Binary(jpeg),
                ),
            )

    connection.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("species_pack_cards", str(len(cards))),
    )
    connection.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("species_pack_images", str(total_images)),
    )
    connection.commit()
    connection.execute("VACUUM")
    connection.close()

    size_mb = database_path.stat().st_size / (1024 * 1024)
    log("")
    log(f"Added {len(cards)} hazard cards and {total_images} images "
        f"({image_bytes_total / 1024:.0f} KB of JPEG) to {database_path.name}")
    log(f"{database_path.name} is now {size_mb:.1f} MB")
    log("Next: cp build/protocols.db ../WildernessEdge/Resources/")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the flora & fauna hazard pack into protocols.db.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", default="species.manifest.json")
    parser.add_argument("--lockfile", default="species.images.lock.json")
    parser.add_argument("--database", default="build/protocols.db")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Must match build_vector_db.py")
    parser.add_argument("--expected-dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--thumb-width", type=int, default=900, help="Width requested from Commons")
    parser.add_argument("--max-edge", type=int, default=640, help="Longest edge after transcode")
    parser.add_argument("--jpeg-quality", type=int, default=78)
    parser.add_argument("--min-source-width", type=int, default=800,
                        help="Reject Commons originals narrower than this")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--request-delay", type=float, default=REQUEST_DELAY,
                        help="Minimum seconds between Commons API requests")
    parser.add_argument("--refresh-images", action="store_true",
                        help="Re-resolve Commons categories and rewrite the lockfile")
    parser.add_argument("--replace", action="store_true", default=True,
                        help="Drop any previously-built hazard pack before inserting")
    parser.add_argument("--limit-species", type=int, default=0, help="Build only the first N (testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve and report licenses only; download and write nothing")
    return parser.parse_args(argv)


def main() -> int:
    try:
        return build(parse_args())
    except BuildError as error:
        log(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
