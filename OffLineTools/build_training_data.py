#!/usr/bin/env python3
"""Generate the LoRA training dataset for Wilderness Edge from `protocols.db`.

The adapter's job is to enforce *output shape*, not to add medical knowledge: given retrieved
protocol context, respond with a cited, non-diagnostic checklist and never volunteer a
diagnosis or a drug recommendation. Clinical content always comes from the retrieved chunks
at inference time, never from the adapter's weights.

Every example is grounded in real text from the corpus, so the adapter cannot learn to invent
protocol content. Three example types are produced:

  grounded  — context supports the question; answer is a cited checklist drawn from context
  refusal   — context is unrelated; answer states no matching protocol was found
  deflect   — question demands a diagnosis or prescription; answer redirects to the checklist

Read the caveat in the module docstring of `main()` before treating this as a finished dataset.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

SYSTEM_PROMPT = (
    "You are Wilderness Edge, an offline field-protocol reference for emergency responders. "
    "You do not diagnose, and you do not prescribe or recommend drugs or dosages. "
    "You read from the provided protocol excerpts only. "
    "Always open with the source citation, then give short numbered action steps taken from "
    "the excerpts. If the excerpts do not cover the question, say so plainly instead of "
    "guessing. Keep responses under 120 words; they are read aloud to someone with their "
    "hands full."
)

# Each bait question is paired with a keyword used to retrieve a topically relevant chunk, so
# the model learns to decline the diagnosis *and* still surface the applicable checklist —
# rather than declining and then reciting whatever text happened to be retrieved.
DIAGNOSIS_BAIT = [
    ("does he have a broken leg or just a sprain", "splint"),
    ("is this a heart attack", "chest pain"),
    ("tell me if her arm is fractured", "fracture"),
    ("what's the diagnosis", "assessment"),
    ("is this bite venomous or not", "snake"),
    ("how many milligrams of ibuprofen should I give", "pain"),
    ("what antibiotic should he take", "wound"),
    ("should I give her epinephrine or not", "anaphylaxis"),
    ("diagnose what is wrong with this patient", "assessment"),
    ("prescribe something for the pain", "pain"),
]

DEFLECTION = (
    "I can't diagnose or recommend medication. I can show you the relevant assessment "
    "checklist so you can work through it and report findings to medical direction."
)

OFF_TOPIC = [
    "what's the weather forecast for tomorrow",
    "how do I fix a flat bike tire",
    "what time does the store close",
    "recommend a good campsite recipe",
    "how do I tie a bowline knot",
    "what's the score of the game",
]

NO_MATCH = (
    "I don't have a protocol covering that in my offline library. I can't answer it from the "
    "field manuals I carry."
)


@dataclass
class Chunk:
    citation: str
    section: str | None
    text: str


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_chunks(db_path: Path, min_tokens: int) -> list[Chunk]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT citation, section, text FROM chunks WHERE token_count >= ? ",
            (min_tokens,),
        ).fetchall()
    finally:
        connection.close()
    return [Chunk(citation=row[0], section=row[1], text=row[2]) for row in rows]


def extract_steps(text: str, limit: int) -> list[str]:
    """Pull the most checklist-like lines out of a chunk.

    Prefers lines that already read as instructions (bulleted, numbered, or imperative) so
    the target responses stay close to verbatim source text.
    """
    marked: list[str] = []
    unmarked: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        without_bullet = stripped.lstrip(" •-*\t")
        without_marker = re.sub(
            r"^(?:\d+[.):]|[a-z][.)]|[ivx]+\.)\s*", "", without_bullet, flags=re.IGNORECASE
        ).strip()
        if not 15 <= len(without_marker) <= 200:
            continue
        if without_marker.lower().startswith(
            ("figure", "table", "references", "revision date", "note:", "aliases")
        ):
            continue
        had_marker = without_marker != stripped
        (marked if had_marker else unmarked).append(without_marker)

    # Bulleted and numbered lines are already written as actions; fall back to plain prose
    # only when a chunk has too few of them.
    steps = marked[:limit]
    if len(steps) < limit:
        steps += [line for line in unmarked if line not in steps][: limit - len(steps)]
    return steps


def topic_of(chunk: Chunk) -> str | None:
    if chunk.section:
        cleaned = re.sub(r"^\d+(\.\d+)*\s*", "", chunk.section).strip()
        if len(cleaned) >= 4:
            return cleaned
    return None


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"{chunk.citation}\n{chunk.text}" for chunk in chunks)


def grounded_example(target: Chunk, distractors: list[Chunk], rng: random.Random) -> dict | None:
    steps = extract_steps(target.text, limit=5)
    topic = topic_of(target)
    if len(steps) < 2 or not topic:
        return None

    question = rng.choice(
        [
            f"what do I do for {topic.lower()}",
            f"walk me through {topic.lower()}",
            f"what's the protocol for {topic.lower()}",
            f"checklist for {topic.lower()}",
        ]
    )
    context_chunks = [target] + distractors
    rng.shuffle(context_chunks)

    numbered = "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))
    answer = f"{target.citation}\n{numbered}"
    return build_record(question, context_chunks, answer)


def refusal_example(question: str, distractors: list[Chunk]) -> dict:
    return build_record(question, distractors, NO_MATCH)


def deflection_example(question: str, relevant: Chunk | None, distractors: list[Chunk]) -> dict:
    if relevant is None:
        return build_record(question, distractors, DEFLECTION)

    steps = extract_steps(relevant.text, limit=3)
    context_chunks = [relevant] + distractors
    if not steps:
        return build_record(question, context_chunks, DEFLECTION)

    numbered = "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))
    answer = f"{DEFLECTION}\n\n{relevant.citation}\n{numbered}"
    return build_record(question, context_chunks, answer)


def build_record(question: str, context_chunks: list[Chunk], answer: str) -> dict:
    user_content = (
        f"Retrieved protocol excerpts:\n{format_context(context_chunks)}\n\nResponder asked: {question}"
        if context_chunks
        else f"No protocol excerpts were retrieved.\n\nResponder asked: {question}"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the Wilderness Edge LoRA training dataset from protocols.db.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", default="build/protocols.db")
    parser.add_argument("--output", default="build/wfr_lora_dataset.jsonl")
    parser.add_argument("--grounded", type=int, default=600, help="Grounded checklist examples")
    parser.add_argument("--refusals", type=int, default=90, help="Out-of-corpus refusal examples")
    parser.add_argument("--deflections", type=int, default=120, help="Diagnosis/prescription bait examples")
    parser.add_argument("--distractors", type=int, default=2, help="Extra retrieved chunks per example")
    parser.add_argument("--min-tokens", type=int, default=80, help="Ignore chunks shorter than this")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    chunks = load_chunks(Path(args.db).expanduser().resolve(), args.min_tokens)
    if len(chunks) < 50:
        raise SystemExit(f"Only {len(chunks)} usable chunks; build protocols.db first.")
    log(f"Loaded {len(chunks)} candidate chunks")

    records: list[dict] = []

    pool = chunks[:]
    rng.shuffle(pool)
    for target in pool:
        if len(records) >= args.grounded:
            break
        distractors = rng.sample(chunks, k=min(args.distractors, len(chunks) - 1))
        record = grounded_example(target, [c for c in distractors if c is not target], rng)
        if record:
            records.append(record)
    log(f"Grounded examples: {len(records)}")

    for index in range(args.refusals):
        question = OFF_TOPIC[index % len(OFF_TOPIC)]
        distractors = rng.sample(chunks, k=min(args.distractors + 1, len(chunks)))
        records.append(refusal_example(question, distractors))

    by_keyword: dict[str, list[Chunk]] = {}
    for _, keyword in DIAGNOSIS_BAIT:
        if keyword not in by_keyword:
            by_keyword[keyword] = [c for c in chunks if keyword in c.text.lower()]
            if not by_keyword[keyword]:
                log(f"  ! no chunk mentions {keyword!r}; those deflections will omit a checklist")

    for index in range(args.deflections):
        question, keyword = DIAGNOSIS_BAIT[index % len(DIAGNOSIS_BAIT)]
        matches = by_keyword[keyword]
        relevant = rng.choice(matches) if matches else None
        distractors = rng.sample(chunks, k=min(args.distractors, len(chunks)))
        records.append(deflection_example(question, relevant, distractors))

    rng.shuffle(records)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    log(f"\nWrote {len(records)} examples to {output}")
    log(
        "\nThis is a FORMAT-teaching seed set, not a finished clinical dataset. Questions are\n"
        "templated from section headings, so topical phrasing is far narrower than real radio\n"
        "traffic. Before shipping, have a WFR/EMS-qualified reviewer audit a sample, and add\n"
        "hand-written or teacher-model-generated queries in real responder language."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
