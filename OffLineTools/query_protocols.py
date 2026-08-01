#!/usr/bin/env python3
"""Query `protocols.db` from the command line, mirroring what VectorRAGManager will do on-device.

Useful for judging corpus quality and for choosing the similarity threshold below which the
app should report "no matching protocol found" instead of returning a weak chunk.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

# Deliberately includes an off-topic query so you can see what a non-match scores.
DEFAULT_PROBES = [
    "severe bleeding from the thigh, how do I stop it",
    "person collapsed and is not breathing",
    "suspected broken forearm after a fall",
    "hypothermic and shivering after falling in a river",
    "rattlesnake bit my leg an hour ago",
    "what is the best pizza topping",
]


def load_corpus(db_path: Path) -> tuple[np.ndarray, list[tuple[str, str]], dict[str, str]]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
        rows = connection.execute("SELECT citation, text, embedding FROM chunks").fetchall()
    finally:
        connection.close()

    if not rows:
        raise SystemExit(f"{db_path} contains no chunks.")

    dim = int(meta["embedding_dim"])
    matrix = np.vstack([np.frombuffer(row[2], dtype="<f4").reshape(1, dim) for row in rows])
    return matrix, [(row[0], row[1]) for row in rows], meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run similarity queries against protocols.db.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("queries", nargs="*", help="Queries to run (defaults to a built-in probe set)")
    parser.add_argument("--db", default="build/protocols.db")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--snippet", type=int, default=160, help="Characters of chunk text to show")
    args = parser.parse_args()

    matrix, chunks, meta = load_corpus(Path(args.db).expanduser().resolve())
    model_id = meta["embedding_model"]
    print(f"{len(chunks)} chunks | {meta['embedding_dim']}-dim | {model_id}\n", file=sys.stderr)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    queries = args.queries or DEFAULT_PROBES
    vectors = model.encode(queries, normalize_embeddings=True, convert_to_numpy=True)

    for query, vector in zip(queries, vectors):
        # Both sides are L2-normalized, so a dot product is cosine similarity. This is exactly
        # the vDSP_dotpr the Swift side will perform.
        scores = matrix @ vector.astype(np.float32)
        ranked = np.argsort(-scores)[: args.top_k]
        print(f"\n=== {query!r}  (best {scores[ranked[0]]:.3f})")
        for rank in ranked:
            citation, text = chunks[rank]
            snippet = " ".join(text.split())[: args.snippet]
            print(f"  {scores[rank]:.3f}  {citation}")
            print(f"         {snippet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
