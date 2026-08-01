#!/usr/bin/env python3
"""Build `protocols.db`, the pre-indexed protocol vector database shipped inside Wilderness Edge.

Reads a directory of source PDFs plus a manifest describing each source's provenance
and license, chunks the extracted text, embeds every chunk, and writes a single
self-contained SQLite file.

The CoreML query embedder exported by `export_embedder_coreml.py` must use the same
model as this script. If the two embedding spaces diverge, on-device cosine similarity
is meaningless and retrieval silently returns garbage.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIM = 384

# Embedded on-device by TextEmbeddingManager's test suite to prove that the CoreML
# embedder reproduces this build's embedding space (PLAN.md Phase 2 verification).
PARITY_SENTENCES = [
    "patient has severe bleeding from the left thigh",
    "unresponsive adult not breathing normally",
    "suspected spinal injury after a fall from height",
    "hypothermia in a wet and windy environment",
    "snake bite to the lower leg two hours ago",
    "how do I splint a suspected forearm fracture",
]


@dataclass(frozen=True)
class SourceMeta:
    filename: str
    title: str
    publisher: str
    license: str
    url: str
    citation_prefix: str


@dataclass
class Chunk:
    text: str
    section: str | None
    page_start: int
    page_end: int
    token_count: int


class BuildError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, SourceMeta]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BuildError(f"Manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BuildError(f"Manifest is not valid JSON: {path} ({exc})") from exc

    entries = raw.get("sources")
    if not isinstance(entries, list) or not entries:
        raise BuildError(f"Manifest {path} must contain a non-empty 'sources' array.")

    required = ("filename", "title", "license", "citation_prefix")
    manifest: dict[str, SourceMeta] = {}
    for entry in entries:
        missing = [key for key in required if not entry.get(key)]
        if missing:
            raise BuildError(
                f"Manifest entry {entry.get('filename', '<unnamed>')} is missing: {', '.join(missing)}"
            )
        meta = SourceMeta(
            filename=entry["filename"],
            title=entry["title"],
            publisher=entry.get("publisher", ""),
            license=entry["license"],
            url=entry.get("url", ""),
            citation_prefix=entry["citation_prefix"],
        )
        manifest[meta.filename] = meta
    return manifest


# --------------------------------------------------------------------------------------
# PDF extraction & cleaning
# --------------------------------------------------------------------------------------

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_WHITESPACE = re.compile(r"[ \t\u00a0]+")
_PAGE_ARTIFACT = re.compile(r"^\s*(page\s+)?\d+\s*(of\s+\d+)?\s*$", re.IGNORECASE)
_ROMAN_PAGE = re.compile(r"^[ivxlc]+$", re.IGNORECASE)
# Dot leaders mark tables of contents and figure lists. They match query keywords strongly
# while carrying no actionable protocol text, so they crowd out real guidance in retrieval.
_DOT_LEADER = re.compile(r"\.{4,}|\u2026{2,}|_{4,}")

# Matches "4.2 Musculoskeletal Evaluation", "SECTION 3 - AIRWAY", "Chapter 12: Bleeding".
_NUMBERED_HEADING = re.compile(
    r"^\s*(?:(?:section|chapter|appendix)\s+)?(\d+(?:\.\d+)*)[.:\)\s-]+([A-Za-z][^\n]{2,80})$",
    re.IGNORECASE,
)
_CAPS_HEADING = re.compile(r"^[A-Z][A-Z &/,'()\-]{3,70}$")

# Cover pages, running heads and revision stamps are the main source of bogus headings.
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_MONTH = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_DOC_ID = re.compile(r"\b[A-Z]{2,4}\s?\d+[-.]\d+", re.IGNORECASE)
_FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "if", "in", "is",
    "not", "of", "on", "or", "the", "to", "with",
}
# Callout box labels, not section headings.
_ADMONITIONS = {"warning", "caution", "note", "notes", "danger", "important", "figure", "table"}


def extract_pages(pdf_path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise BuildError("pypdf is required. Install with: pip install -r requirements.txt") from exc

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise BuildError(f"Could not open {pdf_path.name}: {exc}") from exc

    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:
            log(f"  ! page {index} of {pdf_path.name} failed to extract ({exc}); skipping page")
            pages.append("")
    return pages


def clean_page(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = _WHITESPACE.sub(" ", raw_line).strip()
        if not line or _PAGE_ARTIFACT.match(line) or _ROMAN_PAGE.match(line):
            continue
        if _DOT_LEADER.search(line):
            continue
        lines.append(line)
    return lines


def find_boilerplate(pages: list[list[str]], min_ratio: float) -> set[str]:
    """Identify running headers, footers and distribution stamps repeated across pages.

    Left in place these both pollute chunk text and get mistaken for section headings.
    """
    populated = [page for page in pages if page]
    if len(populated) < 4:
        return set()

    counts: dict[str, int] = {}
    for page in populated:
        for line in set(page):
            if len(line) <= 120:
                counts[line] = counts.get(line, 0) + 1

    threshold = max(3, int(len(populated) * min_ratio))
    return {line for line, count in counts.items() if count >= threshold}


def title_case_ratio(text: str) -> float:
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*", text)
    if not words:
        return 0.0
    return sum(1 for word in words if word[0].isupper()) / len(words)


def detect_heading(line: str) -> str | None:
    """Return a section label for `line`, or None if it is body text.

    Protocol manuals number their checklist steps, so a leading "3." is far more often an
    enumerated instruction than a section heading. Misclassifying one produces a citation
    that gets read aloud verbatim, so this errs heavily toward returning None — a chunk
    with no section still cites its source and page correctly.
    """
    if len(line) > 90 or _YEAR.search(line) or _MONTH.search(line) or _DOC_ID.search(line):
        return None

    match = _NUMBERED_HEADING.match(line)
    if match:
        number, title = match.group(1), match.group(2).strip()
        keyword_prefixed = line.strip().lower().startswith(("section", "chapter", "appendix"))
        looks_like_heading = (
            int(number.split(".")[0]) <= 99
            and len(re.sub(r"[^A-Za-z]", "", title)) >= 3
            and len(title) <= 60
            and not title.endswith((".", ",", ";", ":"))
            and title_case_ratio(title) >= 0.6
            and (keyword_prefixed or "." in number)
        )
        return f"{number} {title}" if looks_like_heading else None

    if _CAPS_HEADING.match(line) and not line.endswith((".", ",", ";", ":")):
        words = re.findall(r"[A-Za-z]+", line.lower())
        if words and not all(word in _FUNCTION_WORDS | _ADMONITIONS for word in words):
            return line.title()
    return None


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------


def chunk_pages(
    pages: list[str],
    tokenizer,
    chunk_tokens: int,
    overlap_tokens: int,
    min_chunk_tokens: int,
    boilerplate_ratio: float,
    max_section_span: int,
) -> list[Chunk]:
    """Group cleaned lines into token-budgeted chunks, preserving section and page provenance.

    Chunking is line-oriented rather than sentence-oriented because protocol manuals are
    dominated by bulleted checklist steps that must not be split mid-item.
    """

    cleaned_pages = [clean_page(page) for page in pages]
    boilerplate = find_boilerplate(cleaned_pages, boilerplate_ratio)

    def count(line: str) -> int:
        return len(tokenizer.encode(line, add_special_tokens=False))

    chunks: list[Chunk] = []
    buffer: list[tuple[str, int, int]] = []  # (line, token_count, page_number)
    buffer_tokens = 0
    section: str | None = None
    section_page = 0
    section_at_buffer_start: str | None = None

    def flush() -> None:
        nonlocal buffer, buffer_tokens, section_at_buffer_start
        if not buffer:
            return
        total = sum(tokens for _, tokens, _ in buffer)
        if total >= min_chunk_tokens:
            chunks.append(
                Chunk(
                    text="\n".join(line for line, _, _ in buffer),
                    section=section_at_buffer_start,
                    page_start=min(page for _, _, page in buffer),
                    page_end=max(page for _, _, page in buffer),
                    token_count=total,
                )
            )
        carry: list[tuple[str, int, int]] = []
        carried = 0
        for item in reversed(buffer):
            if carried + item[1] > overlap_tokens:
                break
            carry.insert(0, item)
            carried += item[1]
        buffer = carry
        buffer_tokens = carried
        section_at_buffer_start = section

    for page_number, page_lines in enumerate(cleaned_pages, start=1):
        # A heading only labels the pages near where it was found. Without this, one detected
        # heading would keep attaching its name to content dozens of pages later.
        if section and page_number - section_page > max_section_span:
            section = None

        for line in page_lines:
            if line in boilerplate:
                continue

            heading = detect_heading(line)
            if heading:
                flush()
                section = heading
                section_page = page_number
                if not buffer:
                    section_at_buffer_start = section
                continue

            tokens = count(line)
            if tokens > chunk_tokens:
                # A single oversized line (dense table row, run-on paragraph) is kept whole
                # rather than truncated; the embedder will clip it, but the text stays citable.
                flush()
                chunks.append(
                    Chunk(
                        text=line,
                        section=section,
                        page_start=page_number,
                        page_end=page_number,
                        token_count=tokens,
                    )
                )
                continue

            if buffer_tokens + tokens > chunk_tokens:
                flush()
            if not buffer:
                section_at_buffer_start = section
            buffer.append((line, tokens, page_number))
            buffer_tokens += tokens

    if buffer and sum(tokens for _, tokens, _ in buffer) >= min_chunk_tokens:
        chunks.append(
            Chunk(
                text="\n".join(line for line, _, _ in buffer),
                section=section_at_buffer_start,
                page_start=min(page for _, _, page in buffer),
                page_end=max(page for _, _, page in buffer),
                token_count=sum(tokens for _, tokens, _ in buffer),
            )
        )
    return chunks


def format_citation(meta: SourceMeta, chunk: Chunk) -> str:
    parts = [meta.citation_prefix]
    if chunk.section:
        parts.append(f"Section {chunk.section}" if chunk.section[0].isdigit() else chunk.section)
    if chunk.page_start == chunk.page_end:
        parts.append(f"p. {chunk.page_start}")
    else:
        parts.append(f"pp. {chunk.page_start}-{chunk.page_end}")
    return f"[Source: {', '.join(parts)}]"


# --------------------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE sources (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    publisher       TEXT NOT NULL DEFAULT '',
    license         TEXT NOT NULL,
    url             TEXT NOT NULL DEFAULT '',
    citation_prefix TEXT NOT NULL
);

CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   INTEGER NOT NULL REFERENCES sources(id),
    section     TEXT,
    page_start  INTEGER NOT NULL,
    page_end    INTEGER NOT NULL,
    citation    TEXT NOT NULL,
    text        TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    embedding   BLOB NOT NULL
);

CREATE INDEX idx_chunks_source ON chunks(source_id);
"""


def create_database(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    for sidecar in (path.with_suffix(path.suffix + "-wal"), path.with_suffix(path.suffix + "-shm")):
        if sidecar.exists():
            sidecar.unlink()

    connection = sqlite3.connect(path)
    # The database ships read-only inside the app bundle, so it must be a single file with
    # no WAL sidecars that iOS would be unable to create at runtime.
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.executescript(SCHEMA)
    return connection


def write_meta(connection: sqlite3.Connection, values: dict[str, str]) -> None:
    connection.executemany(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        [(key, str(value)) for key, value in values.items()],
    )


def encode_vector(vector: np.ndarray) -> bytes:
    """Serialize as little-endian float32 so Swift can memcpy straight into [Float] for vDSP."""
    return np.ascontiguousarray(vector, dtype="<f4").tobytes()


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------


def build(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise BuildError(f"Input directory does not exist: {input_dir}")

    manifest = load_manifest(Path(args.manifest).expanduser().resolve())
    pdfs = sorted(p for p in input_dir.glob("*.pdf") if p.is_file())
    if not pdfs:
        raise BuildError(f"No PDFs found in {input_dir}")

    unmanifested = [p.name for p in pdfs if p.name not in manifest]
    if unmanifested and not args.allow_unmanifested:
        raise BuildError(
            "These PDFs have no manifest entry, so their license and citation are unknown:\n  "
            + "\n  ".join(unmanifested)
            + "\n\nEvery shipped chunk must carry a verified source citation. Add them to the "
            "manifest, or re-run with --allow-unmanifested to skip them."
        )
    pdfs = [p for p in pdfs if p.name in manifest]
    if not pdfs:
        raise BuildError("No manifested PDFs to process.")

    from transformers import AutoTokenizer

    log(f"Loading tokenizer for {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    all_chunks: list[tuple[SourceMeta, Chunk]] = []
    for pdf in pdfs:
        meta = manifest[pdf.name]
        log(f"Parsing {pdf.name} ({meta.license}) ...")
        pages = extract_pages(pdf)
        chunks = chunk_pages(
            pages,
            tokenizer,
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
            min_chunk_tokens=args.min_chunk_tokens,
            boilerplate_ratio=args.boilerplate_ratio,
            max_section_span=args.max_section_span,
        )
        if not chunks:
            log(f"  ! {pdf.name} produced no usable text (scanned image PDF?); skipping")
            continue
        log(f"  {len(pages)} pages -> {len(chunks)} chunks")
        all_chunks.extend((meta, chunk) for chunk in chunks)

    if not all_chunks:
        raise BuildError("No chunks were produced from any source. Are these scanned/image-only PDFs?")

    if args.dry_run:
        log(f"\nDry run: {len(all_chunks)} chunks from {len(pdfs)} sources. No database written.")
        sectioned = sum(1 for _, chunk in all_chunks if chunk.section)
        log(f"Chunks carrying a detected section heading: {sectioned}/{len(all_chunks)}")

        start = args.sample_from if args.sample_from >= 0 else max(0, len(all_chunks) // 2)
        for meta, chunk in all_chunks[start : start + args.sample]:
            log(f"\n--- {format_citation(meta, chunk)} ({chunk.token_count} tokens)")
            log(chunk.text[:400])
        return 0

    from sentence_transformers import SentenceTransformer

    log(f"\nLoading embedding model {args.model} ...")
    model = SentenceTransformer(args.model)
    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dim = get_dim()
    if dim != args.expected_dim:
        raise BuildError(
            f"{args.model} produces {dim}-dim embeddings but --expected-dim is {args.expected_dim}. "
            "The Swift VectorRAGManager and AGENTS.md assume a fixed dimensionality; update both "
            "deliberately or choose a different model."
        )

    log(f"Embedding {len(all_chunks)} chunks ...")
    vectors = model.encode(
        [chunk.text for _, chunk in all_chunks],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = create_database(output)
    try:
        source_ids: dict[str, int] = {}
        for meta in manifest.values():
            if not any(m.filename == meta.filename for m, _ in all_chunks):
                continue
            cursor = connection.execute(
                "INSERT INTO sources (filename, title, publisher, license, url, citation_prefix)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (meta.filename, meta.title, meta.publisher, meta.license, meta.url, meta.citation_prefix),
            )
            source_ids[meta.filename] = int(cursor.lastrowid)

        connection.executemany(
            "INSERT INTO chunks (source_id, section, page_start, page_end, citation, text, token_count, embedding)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    source_ids[meta.filename],
                    chunk.section,
                    chunk.page_start,
                    chunk.page_end,
                    format_citation(meta, chunk),
                    chunk.text,
                    chunk.token_count,
                    encode_vector(vector),
                )
                for (meta, chunk), vector in zip(all_chunks, vectors)
            ],
        )

        write_meta(
            connection,
            {
                "schema_version": SCHEMA_VERSION,
                "embedding_model": args.model,
                "embedding_dim": dim,
                "embeddings_normalized": "1",
                "embedding_dtype": "float32_le",
                "chunk_tokens": args.chunk_tokens,
                "overlap_tokens": args.overlap_tokens,
                "chunk_count": len(all_chunks),
                "source_count": len(source_ids),
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )
        connection.commit()
        connection.execute("VACUUM")
        connection.commit()
    finally:
        connection.close()

    parity_path = Path(args.parity_out).expanduser().resolve()
    parity_vectors = model.encode(PARITY_SENTENCES, normalize_embeddings=True, convert_to_numpy=True)
    parity_path.parent.mkdir(parents=True, exist_ok=True)
    parity_path.write_text(
        json.dumps(
            {
                "embedding_model": args.model,
                "embedding_dim": dim,
                "normalized": True,
                "fixtures": [
                    {"text": text, "embedding": [round(float(v), 6) for v in vector]}
                    for text, vector in zip(PARITY_SENTENCES, parity_vectors)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    size_mb = output.stat().st_size / (1024 * 1024)
    log(f"\nWrote {output} ({size_mb:.1f} MB)")
    log(f"  {len(all_chunks)} chunks from {len(source_ids)} sources, {dim}-dim normalized float32")
    log(f"Wrote parity fixtures to {parity_path}")
    log("\nNext: run export_embedder_coreml.py with the SAME --model value.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Wilderness Edge protocol vector database from source PDFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default="sources", help="Directory containing source PDFs")
    parser.add_argument("--manifest", default="sources.manifest.json", help="Source provenance/license manifest")
    parser.add_argument("--output", default="build/protocols.db", help="Output SQLite database path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model id")
    parser.add_argument("--expected-dim", type=int, default=DEFAULT_DIM, help="Required embedding dimensionality")
    parser.add_argument("--chunk-tokens", type=int, default=220, help="Target tokens per chunk")
    parser.add_argument("--overlap-tokens", type=int, default=48, help="Token overlap carried between chunks")
    parser.add_argument("--min-chunk-tokens", type=int, default=24, help="Discard chunks smaller than this")
    parser.add_argument(
        "--boilerplate-ratio",
        type=float,
        default=0.2,
        help="Drop lines repeated on at least this fraction of a document's pages",
    )
    parser.add_argument(
        "--max-section-span",
        type=int,
        default=10,
        help="Stop applying a detected heading after this many pages",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument(
        "--parity-out",
        default="build/embedding_parity_fixtures.json",
        help="Where to write reference embeddings used to verify the CoreML embedder",
    )
    parser.add_argument(
        "--allow-unmanifested",
        action="store_true",
        help="Skip (rather than reject) PDFs missing a manifest entry",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk only; do not embed or write")
    parser.add_argument("--sample", type=int, default=3, help="Chunks to print during --dry-run")
    parser.add_argument(
        "--sample-from",
        type=int,
        default=-1,
        help="Index to sample from during --dry-run; default samples from the middle of the corpus",
    )
    return parser.parse_args(argv)


def main() -> int:
    try:
        return build(parse_args())
    except BuildError as exc:
        log(f"\nERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        log("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
