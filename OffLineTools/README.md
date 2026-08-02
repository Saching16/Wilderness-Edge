# OffLineTools

Pre-processing pipeline that produces the two data assets Wilderness Edge ships with:
`protocols.db` (the retrieval corpus) and the CoreML query embedder.

Everything in this directory runs on a laptop or Colab. None of it ships inside the app, and
none of it runs at app runtime — the air-gap guarantee applies to the iOS target only.

## Setup

```bash
cd OffLineTools
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11–3.13 all work. CoreML export requires macOS; building `protocols.db` does not.

Do not upgrade `transformers` past 4.x. Version 5 rewrote attention masking and emits fx nodes
that coremltools cannot convert, which breaks the embedder export with
`NotImplementedError: Unsupported fx node new_ones`.

## Pipeline

```bash
# 1. Fetch the vetted public-domain corpus (see SOURCES.md before adding anything)
python fetch_sources.py

# 2. Inspect chunking quality before spending time on embeddings
python build_vector_db.py --dry-run

# 3. Build the database + the parity fixtures used to validate the on-device embedder
python build_vector_db.py

# 4. Add the flora & fauna hazard cards + licensed reference imagery.
#    Run AFTER build_vector_db.py: it extends protocols.db rather than creating it.
python build_species_pack.py --dry-run   # resolve image licenses, download nothing
python build_species_pack.py

# 5. Export the CoreML embedder, which must use the same model
python export_embedder_coreml.py

# 6. Sanity-check retrieval quality and calibrate the similarity threshold
python query_protocols.py
```

### `build_species_pack.py`

The protocol corpus says what to *do*; it does not say what a responder is looking *at*.
This adds 20 hazard cards (see `species.manifest.json`) covering urushiol plants, toxic
plants and a mushroom, five snakes, two spiders, a tick, and four large mammals.

Each card becomes an ordinary row in `chunks`, embedded with the same model as everything
else, so `VectorRAGManager.search(...)` finds it through the existing code path — no Swift
search changes. Structured detail goes to a `species` table and licensed JPEGs to
`chunk_images`, which `VectorRAGManager.referenceImages(forChunkID:)` reads on demand.

Image licenses are **machine-enforced**: only Public domain / CC0 / CC BY / CC BY-SA are
admitted, NonCommercial and NoDerivatives are rejected, and an unparseable license string
fails closed. Resolved files are pinned in `species.images.lock.json` and re-verified on
every build, so a Commons re-license shows up in a diff rather than slipping into a binary.

```bash
python build_species_pack.py --refresh-images   # re-resolve categories, rewrite the lockfile
python build_species_pack.py --limit-species 3  # quick smoke test
```

Commons throttles anonymous clients, so both API calls and image downloads are rate-limited
to roughly one request per second with backoff on 429. A full 20-species run takes a few
minutes; that is the throttle, not a hang.

Attribution is a redistribution condition for the CC BY / CC BY-SA images. `SpeciesCardView`
renders `attribution — license` under every photograph — do not remove it.

`query_protocols.py` performs exactly the search `VectorRAGManager` will run on-device, so it
is the cheapest way to judge corpus quality and pick the cutoff below which the app should
say "no matching protocol found." On the current three-source corpus, genuine clinical
queries score roughly 0.53–0.67 and an off-topic control query scores 0.17, so a threshold
in the 0.30–0.40 range separates them cleanly.

Outputs land in `build/`:

| Artifact | Destination in Xcode |
| --- | --- |
| `protocols.db` | `WildernessEdge/Resources/` |
| `query-embedder.mlpackage` | `WildernessEdge/Resources/` |
| `query-embedder-vocab.txt` | `WildernessEdge/Resources/` |
| `query-embedder-tokenizer.json` | `WildernessEdge/Resources/` |
| `embedding_parity_fixtures.json` | `WildernessEdgeTests/` |

## The one invariant that matters

`build_vector_db.py` and `export_embedder_coreml.py` must be run with the same `--model`.
The database stores vectors from one model; the app generates query vectors from the other.
If they diverge, cosine similarity still returns confident-looking numbers — they are just
meaningless, and retrieval degrades silently rather than failing loudly.

`export_embedder_coreml.py` guards against this by checking its output against the parity
fixtures emitted during the database build, and exits non-zero on mismatch. Do not ship an
embedder that failed that check.

## Notes on the artifacts

**Vectors** are stored as L2-normalized little-endian `float32` BLOBs (384 floats = 1536
bytes per chunk). Because they are pre-normalized, `VectorRAGManager` only needs a dot
product via `vDSP_dotpr`, not a full cosine computation.

**The database** is written with `journal_mode=DELETE` and vacuumed, so it ships as a single
self-contained file. It is read-only inside the app bundle, so the Swift side must open it
with `SQLITE_OPEN_READONLY`.

**Tokenization** is not part of the CoreML model, because CoreML cannot accept strings. The
model takes `input_ids` and `attention_mask` at a fixed length of 128. The Swift side must
implement WordPiece tokenization against the exported vocabulary — `query-embedder-tokenizer.json`
records the lowercase/accent-stripping behavior, special token ids, and sequence length needed
to reproduce it exactly.

**Pooling and L2 normalization** are traced into the CoreML graph, so its output is directly
comparable to the stored vectors with no post-processing in Swift.

## Model training

Runs on a Colab GPU (free T4 is enough), not on your Mac:

```bash
python build_training_data.py          # locally: derives the dataset from protocols.db

# then in Colab, with the dataset uploaded:
!pip install -q unsloth trl datasets
!uv tool install litert-torch-nightly
!python train_lora_colab.py --hf-token $HF_TOKEN
```

`build_training_data.py` produces three example types, all grounded in real corpus text so the
adapter cannot learn to invent protocol content:

| Type | Teaches |
| --- | --- |
| grounded | Answer from retrieved excerpts, lead with the citation, give numbered steps |
| refusal | Say "no protocol covers that" instead of guessing when retrieval misses |
| deflect | Decline diagnosis/dosage requests, then offer the applicable checklist |

This is a **format-teaching seed set, not a finished clinical dataset.** Questions are
templated from section headings, so phrasing is much narrower than real radio traffic, and
chunk-boundary overlap means some extracted steps begin mid-sentence. Have a WFR/EMS-qualified
reviewer audit a sample and add real-language queries before shipping.

The adapter is defense in depth. It never replaces the on-device `SafetyFilter`.

### The model size tradeoff

Google's smallest published Gemma 4 E2B mobile build is ~1.1 GB, but it comes from a QAT
int2/int4 pipeline that `litert-torch export_hf` cannot currently consume. Exporting your own
fine-tune means post-training quantization instead: int4 (`dynamic_wi4c_hr_afp32`, the default
here) lands nearer ~2.9 GB, and the int8 default would be ~5.7 GB.

That is a bundle-size cost more than a runtime one — LiteRT-LM memory-maps weights with
XNNPACK weight caching, so resident memory runs well below file size. Still, measure actual
resident usage against `PLAN.md`'s 2.8 GB Phase 3 ceiling on a real device. If the fine-tune
is not clearly earning its keep, shipping the stock prebuilt
`litert-community/gemma-4-E2B-it-litert-lm` bundle is a legitimate fallback.

Pass `--no-vision` only if you are abandoning the camera feature; without the vision encoder
the exported bundle is text-only.
