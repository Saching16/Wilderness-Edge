#!/usr/bin/env python3
"""Generate the SQLite fixtures used by VectorRAGManagerTests.

These are committed binaries, so this script exists to make them reproducible and
reviewable. Run from the repository root:

    python OffLineTools/make_test_fixtures.py

The vectors are 4-dimensional on purpose. `VectorRAGManager` reads its dimensionality from
the stored blobs rather than assuming 384, so a toy dimension keeps the expected cosine
similarities hand-checkable (orthogonal unit vectors score exactly 1.0 or 0.0) while
exercising the same code path the real 384-dim corpus takes.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "WildernessEdgeTests" / "Fixtures"

# Mirrors the subset of build_vector_db.py's schema that VectorRAGManager reads.
SCHEMA = """
CREATE TABLE sources (
    id INTEGER PRIMARY KEY, filename TEXT, title TEXT, publisher TEXT,
    license TEXT, url TEXT, citation_prefix TEXT
);
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY, source_id INTEGER, section TEXT,
    page_start INTEGER, page_end INTEGER, citation TEXT, text TEXT,
    token_count INTEGER, embedding BLOB
);
"""

META_SCHEMA = "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"

ROWS = [
    (1, 1, "1.1 Bleeding", 10, 10, "[Source: Fixture Manual, Section 1.1, p. 10]",
     "Apply direct pressure to the wound.", 8, (1.0, 0.0, 0.0, 0.0)),
    (2, 1, "2.1 Fractures", 20, 20, "[Source: Fixture Manual, Section 2.1, p. 20]",
     "Splint the limb in the position found.", 8, (0.0, 1.0, 0.0, 0.0)),
    (3, 1, "3.1 Hypothermia", 30, 30, "[Source: Fixture Manual, Section 3.1, p. 30]",
     "Insulate and passively rewarm the patient.", 8, (0.0, 0.0, 1.0, 0.0)),
]


def encode(vector: tuple[float, ...]) -> bytes:
    """Little-endian float32, matching build_vector_db.py's encode_vector."""
    return struct.pack(f"<{len(vector)}f", *vector)


def build(path: Path, *, meta: dict[str, str] | None) -> None:
    path.unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.executescript(SCHEMA)
        connection.execute(
            "INSERT INTO sources VALUES (1, 'fixture.pdf', 'Fixture Manual', 'Test', 'test', '', 'Fixture Manual')"
        )
        connection.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [row[:-1] + (encode(row[-1]),) for row in ROWS],
        )
        if meta is not None:
            connection.executescript(META_SCHEMA)
            connection.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)", list(meta.items())
            )
        connection.commit()
        connection.execute("VACUUM")
        connection.commit()
    finally:
        connection.close()
    print(f"wrote {path.relative_to(Path.cwd())} ({path.stat().st_size} bytes)")


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    # No meta table at all: the real corpus has one, but VectorRAGManager must tolerate its
    # absence rather than refusing to open a database it can otherwise read.
    build(FIXTURES / "fixture-protocols.db", meta=None)

    # Declares its vectors un-normalized, which makes a dot product something other than
    # cosine similarity. VectorRAGManager must refuse this rather than rank silently wrongly.
    build(
        FIXTURES / "fixture-unnormalized.db",
        meta={"schema_version": "1", "embedding_dim": "4", "embeddings_normalized": "0"},
    )


if __name__ == "__main__":
    main()
