# Wilderness Edge

**On-device, voice-first protocol checklists for off-grid emergency responders — powered by Gemma 4, with patient data that never leaves the phone.**

Native iOS · 100% offline · Decision-support only  
Built for [Build with Gemma NYC: On-Device AI for Healthcare](docs/hackathon-rules.md)  
**Track:** On-Device Private Health (primary) · Voice for Care (secondary)

**Team:** Pablo Beaus Iranzo · Sachin Ganpule · Daniel David · Vaibhav Chaudhari

---

## Problem

Wilderness and disaster responders need protocol guidance with **zero connectivity** — no cloud APIs, no EHR lookup, no hosted model. The phone may be the only computer on scene, and scene audio / injury context is sensitive.

Responders need **decision support**: fast, cited checklists from accredited field manuals — not a diagnosis, not a drug order.

## Solution

Wilderness Edge is a native iOS app that runs entirely offline:

1. Tap to talk (and capture an optional camera frame).
2. On-device speech recognition transcribes the question.
3. Local vector search retrieves protocol chunks from bundled `protocols.db`.
4. **Gemma 4 E4B** (Google AI Edge **LiteRT-LM**) turns transcript + RAG context (+ image) into a short checklist with a source citation.
5. A client-side **SafetyFilter** blocks diagnostic / prescribing language before display or speech.
6. Text-to-speech reads the checklist aloud so hands stay free.

**Demo condition:** Airplane Mode (Wi‑Fi, cellular, and Bluetooth off).

## How we use Gemma 4

Gemma 4 is the **core inference engine**, not a thin wrapper around search.

| Piece | Choice |
| --- | --- |
| Runtime | LiteRT-LM Swift API on device (GPU when available) |
| Model | Prebuilt multimodal `gemma-4-E4B-it` (`.litertlm` bundle) |
| Inputs | Transcript + retrieved protocol excerpts + optional camera snapshot |
| Output | Cited, numbered action checklist — never an independent diagnosis |

If retrieval has **no confident match**, the model is instructed to say so plainly instead of inventing a protocol.

## Architecture

```text
[ Tap-to-talk button ]
        │
        ├─► Camera snapshot (AVFoundation)
        └─► On-device STT (SFSpeechRecognizer)
                │
                ▼
        CoreML query embedder (384-d)
                │
                ▼
        Vector RAG over protocols.db (Accelerate cosine search)
                │
                ▼
        Gemma 4 E4B via LiteRT-LM  ◄── image + RAG context + transcript
                │
                ▼
        SafetyFilter (regex, fail closed)
                │
                ▼
        UI citation card + AVSpeechSynthesizer
```

Camera frames go straight into LiteRT-LM as multimodal image input. RAG runs only on the transcribed text.

## Offline resources (shipped on-device)

Everything needed at query time is bundled in the app — no downloads after install.

### Bundled in `WildernessEdge/Resources/`

| Asset | Role |
| --- | --- |
| `protocols.db` (~8.6 MB) | SQLite corpus: **2,192** embedded protocol chunks from **3** licensed sources (384-d float32 vectors + text + citations) |
| `query-embedder.mlpackage` | On-device CoreML sentence embedder (`all-MiniLM-L6-v2` space, mean-pool + L2 in-graph) |
| `query-embedder-vocab.txt` + `query-embedder-tokenizer.json` | WordPiece tokenizer assets for the Swift embedder path |
| `gemma-4-E4B-it.litertlm` | Multimodal Gemma 4 E4B LiteRT-LM bundle (**gitignored** — place locally before device runs) |

### Protocol corpus (what’s inside `protocols.db`)

Built offline from PDFs listed in [`OffLineTools/sources.manifest.json`](OffLineTools/sources.manifest.json). Vetting notes: [`OffLineTools/SOURCES.md`](OffLineTools/SOURCES.md).

| Source | License posture | Citation prefix |
| --- | --- | --- |
| US Army **ATP 4-02.11** — Casualty Response, TCCC, and First Aid | US federal public domain | `US Army ATP 4-02.11 First Aid & TCCC` |
| US Army **TCCC Handbook v5** (CALL 17-13) | US federal public domain | `US Army TCCC Handbook v5` |
| **NASEMSO** National Model EMS Clinical Guidelines v3.0 | Publisher invites harvest / adopt in whole or in part | `NASEMSO National Model EMS Clinical Guidelines v3.0` |

**Not included:** NOLS WFR/WFA course materials (copyrighted; not licensed for redistribution) and other Tier 3/4 sources in `SOURCES.md`.

Embedding model used to build the DB: `sentence-transformers/all-MiniLM-L6-v2` (must stay in lockstep with the CoreML query embedder).

## Privacy & safety

- **100% on-device at runtime** — no network for speech, embedding, retrieval, or generation. On-device STT unavailable → fail closed; never fall back to server recognition.
- **Decision-support only** — cited checklists from licensed public sources. No diagnosis. No drug doses.
- **SafetyFilter** — mandatory gate before UI and TTS.
- **Licensed corpus only** — see [`OffLineTools/SOURCES.md`](OffLineTools/SOURCES.md) and `sources.manifest.json`. NOLS materials are **not** licensed for ingestion.
- **No real patient data** — synthetic / public protocol text only.

> This app is **not** a medical device and does not provide clinical care. Follow only cited field-manual steps within your training and scope, and report findings to medical direction.

## Repo layout

```text
WildernessEdge/           # Native iOS app (SwiftUI)
  App/                    # Entry point, Info.plist, entitlements
  Core/                   # Speech, TTS, Camera, SafetyFilter, (RAG / LLM managers)
  Views/                  # Push-to-talk UI + state machine
  Resources/              # protocols.db, CoreML embedder, (.litertlm gitignored)
WildernessEdgeTests/      # XCTest (SafetyFilter, tokenizer/RAG when present)
OffLineTools/             # Offline corpus → vector DB → embedder export
docs/                     # Kaggle writeup draft, hackathon rules
plans/                    # Sprint plans per teammate
AGENTS.md                 # Agent rulebook & architecture
PLAN.md                   # Long-horizon implementation plan
```

## Quick start (Xcode)

**Requirements:** macOS, Xcode 15+, physical iPhone recommended for on-device speech + Airplane Mode demo.

```bash
brew install xcodegen          # if needed
cd Wilderness-Edge
xcodegen generate
open WildernessEdge.xcodeproj
```

1. Set your **Development Team** under Signing & Capabilities.
2. Place `gemma-4-E4B-it.litertlm` in `WildernessEdge/Resources/` (large bundle — gitignored; obtain the prebuilt LiteRT-LM community build).
3. Confirm `protocols.db` and `query-embedder.mlpackage` are already under `Resources/`.
4. Run on a device → enable Airplane Mode → tap to ask → tap to send.

Host-free unit tests (when Simulator is available):

```bash
xcodebuild test \
  -scheme WildernessEdgeTests \
  -destination 'platform=iOS Simulator,name=iPhone 17'
```

## Offline asset pipeline

Rebuild or extend the corpus with [`OffLineTools/`](OffLineTools/README.md) (laptop/Colab — not at app runtime):

```bash
cd OffLineTools
python fetch_sources.py              # download manifested PDFs
python build_vector_db.py            # → protocols.db (+ embedding parity fixtures)
python export_embedder_coreml.py     # → query-embedder.mlpackage + tokenizer assets
python query_protocols.py            # offline retrieval harness / threshold checks
```

Copy outputs into `WildernessEdge/Resources/` (and fixtures into the test target) before shipping.

## Hackathon notes

Built in a one-day sprint. Deliberate cuts so Gemma stays central and the offline story stays honest:

- Prebuilt **Gemma 4 E4B** (no custom LoRA this cycle)
- Grounding via **RAG + prompting** (not fine-tuned medical weights)
- Soft memory check on iPhone 16 Plus
- Demo bar: two reliable Airplane Mode runs

Longer roadmap: [`PLAN.md`](PLAN.md) · sprint design: [`plans/scope-design.md`](plans/scope-design.md)  
Kaggle narrative draft: [`docs/kaggle-writeup-draft.md`](docs/kaggle-writeup-draft.md)

## License & corpus

Application code: see repository license / team agreement.  
Protocol text: only sources recorded in `OffLineTools/sources.manifest.json` with verified redistribution rights. Do not add copyrighted manuals (e.g. NOLS WFR) without a written license.
