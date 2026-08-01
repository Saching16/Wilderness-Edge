# Wilderness Edge — AI Agent Context & Rulebook

## Mission & Purpose

Wilderness Edge is a 100% air-gapped, voice-first, and vision-assisted emergency protocol assistant built natively for iOS. Designed for first responders in zero-connectivity off-grid environments, it provides instant decision support by searching accredited field manuals (e.g., NOLS Wilderness First Aid, State EMS Protocols) and presenting structured assessment checklists.

This document serves as the primary context, system rulebook, and architectural guide for AI coding agents working on this codebase.

## Hackathon Sprint Scope (Build with Gemma NYC, 2026-08-01)

This build is being executed as an ~18-hour hackathon sprint. The scope trims below are deliberate engineering tradeoffs for the sprint, not a change to the project's long-term ambitions — see `docs/superpowers/specs/2026-08-01-hackathon-scope-design.md` for the full rationale and `docs/superpowers/plans/2026-08-01-hackathon-sprint.md` (or the per-person plan files in the same directory) for the task breakdown.

- **Model:** Gemma 4 **E4B** (not E2B), using the prebuilt `litert-community/gemma-4-E4B-it-litert-lm` bundle. No custom weights.
- **No LoRA fine-tune this sprint.** `train_lora_colab.py` and the clinical-review gate on `build_training_data.py` output are out of scope. Grounding comes from RAG context + prompting only; the LoRA adapter remains a documented future direction, not a component of this build.
- **Memory guardrails are a soft target, not a hard gate.** The team is sideloading to a physical iPhone 16 Plus, which has ample headroom. Keep the "Increased Memory Limit" entitlement, but do not block progress on formally measuring or optimizing resident memory footprint this sprint.
- **Corpus scope:** wilderness + general first aid, restricted to sources already verified in `OffLineTools/sources.manifest.json` (skip NOLS — confirmed unlicensed for ingestion, see `SOURCES.md`).
- **Demo condition:** a live Airplane Mode run on the sideloaded device is the actual proof of the air-gap claim, not just a written assertion.

## Non-Negotiable System Guardrails

All coding agents working on this project must strictly respect and enforce three core operational constraints:

### 1. Non-Diagnostic & Non-Prescriptive Guarantee

- The application must **NEVER** issue an independent medical diagnosis (e.g., "The patient has a grade-2 fracture") or prescribe drug treatments.
- All outputs must be explicitly framed as retrieved protocol checklists (e.g., "Displaying NOLS WFR Section 4.2: Musculoskeletal Evaluation Checklist").
- Output text must pass through a client-side safety verification layer before being spoken or displayed.

### 2. Strict Air-Gap (100% Offline) Operation

- Zero network requests are permitted at runtime.
- Speech recognition, vector retrieval, language model inference, and text-to-speech synthesis must run locally on the device.
- Speech recognition must explicitly enforce on-device processing flags.

### 3. Device Memory Safety (soft target this sprint)

- The app's long-term design target is a physical base-model iPhone with 6GB of unified memory; the hackathon sprint sideloads to a physical iPhone 16 Plus instead, which has substantially more headroom.
- Language model footprint is the prebuilt, quantized Gemma 4 **E4B** (`litert-community/gemma-4-E4B-it-litert-lm`) for this sprint — no custom fine-tune.
- High-memory system entitlements (`com.apple.developer.kernel.increased-memory-limit`) must remain enabled in the project build settings to prevent low-memory operating system terminations, but formal resident-memory measurement against a strict ceiling is deferred past this sprint.

## Technical Stack Overview

| Layer | Technology |
| --- | --- |
| **Target Platform** | Native iOS 17.0+ (Swift 5.9 / SwiftUI) |
| **LLM Runtime Engine** | Google AI Edge **LiteRT-LM Swift API** (native multimodal image + text inference, GPU/Metal accelerated) |
| **Base Language Model** | Quantized Gemma 4 **E4B** (natively multimodal: text + image + audio), prebuilt `litert-community/gemma-4-E4B-it-litert-lm` bundle, packaged as a `.litertlm` file. No custom LoRA fine-tune this sprint. |
| **Speech-to-Text (STT)** | Native Apple Framework (`SFSpeechRecognizer`) forced to on-device recognition |
| **Text-to-Speech (TTS)** | Native Apple Framework (`AVSpeechSynthesizer`) |
| **Camera Capture** | Native Apple Framework (`AVFoundation`) for single-frame snapshot capture |
| **On-Device Query Embedding** | Bundled CoreML sentence-embedding model producing 384-dim vectors, matching the embedding space used to build `protocols.db` |
| **Vector Search Engine** | Local SQLite database (`protocols.db`) accessed via the raw `SQLite3` C API (no third-party wrapper), queried via SIMD-accelerated linear algebra using Apple's Accelerate framework |
| **Safety Verification** | Regular expression pattern matcher evaluating LLM text outputs prior to speech playback |

> **Note on LLM runtime:** Google's MediaPipe LLM Inference API (`MediaPipeTasksGenAI`) is in maintenance-only mode and its from-source iOS build is currently broken upstream. LiteRT-LM is Google's actively developed, officially recommended successor and is required here for native Swift multimodal (image) input support.

## System Architecture & Data Flow

```text
                  [ PUSH-TO-TALK SINGLE BUTTON (SwiftUI) ]
                                      │
          ┌───────────────────────────┴───────────────────────────┐
          ▼                                                       ▼
[ AVFoundation Snapshot ]                             [ SFSpeechRecognizer ]
(Single Camera Frame)                                 (On-Device Offline Audio)
          │                                                       │
          │                                                       ▼
          │                                          Transcribed Query Text
          │                                                       │
          │                                                       ▼
          │                                        [ On-Device Text Embedder ]
          │                                       (Bundled CoreML sentence model)
          │                                                       │
          │                                                       ▼
          │                                        [ Local Vector RAG Manager ]
          │                               Accelerate SIMD Cosine Search over protocols.db
          │                                                       │
          └───────────────────────────┬───────────────────────────┘
                                      │
                                      ▼
                       [ LiteRT-LM Multimodal Engine ]
                          Gemma 4 E4B (prebuilt)
        (native image input + retrieved protocol text context)
                                      │
                                      ▼
                    [ Safety Guardrail Regex Layer ]
             Sanitizes output & enforces source citations
                                      │
                                      ▼
                     [ AVSpeechSynthesizer Output ]
            Streams spoken checklist items out loud to user
```

Note: the camera snapshot is passed directly to the LiteRT-LM engine as native multimodal image input — it is never vectorized or run through the text-based RAG search, which only ever operates on the transcribed query text.

## Repository Map & Architecture Blueprint

```text
WildernessEdge/
├── WildernessEdge.xcodeproj            # Main Xcode Project
├── AGENTS.md                           # This AI Agent Context & Rulebook
├── PLAN.md                             # Step-by-Step Implementation Plan & Checklists
├── OffLineTools/                       # Pre-processing scripts (Run on Laptop/Colab)
│   ├── SOURCES.md                      # Vetted corpus & redistribution license record
│   ├── sources.manifest.json           # Per-source provenance, license & citation prefix
│   ├── fetch_sources.py                # Downloads the manifested source PDFs
│   ├── build_vector_db.py              # Offline PDF parser & vector DB generator
│   ├── export_embedder_coreml.py       # CoreML query-embedder export & parity check
│   ├── query_protocols.py              # Offline retrieval harness for threshold calibration
│   ├── build_training_data.py          # Grounded/refusal/deflection LoRA dataset generator (not used this sprint)
│   └── train_lora_colab.py             # Unsloth LoRA fine-tune & .litertlm export (Colab GPU; out of scope this sprint)
└── WildernessEdge/
    ├── App/
    │   ├── WildernessEdgeApp.swift     # Application Entry Point & Audio Session setup
    │   └── Info.plist                  # Entitlements & Privacy Permission Strings
    ├── Resources/
    │   ├── protocols.db                # Pre-indexed SQLite database containing embeddings
    │   ├── query-embedder.mlpackage    # CoreML sentence-embedding model (matches protocols.db embedding space)
    │   ├── query-embedder-vocab.txt    # WordPiece vocabulary for the Swift tokenizer
    │   ├── query-embedder-tokenizer.json # Tokenizer settings (casing, special ids, seq length)
    │   └── gemma-4-E4B-it.litertlm     # Prebuilt LiteRT-LM Gemma 4 E4B bundle (multimodal, no custom fine-tune)
    ├── Core/
    │   ├── SpeechManager.swift         # On-device SFSpeechRecognizer wrapper
    │   ├── TTSManager.swift            # Native AVSpeechSynthesizer wrapper
    │   ├── CameraManager.swift         # AVFoundation single-frame snapshot capture
    │   ├── WordPieceTokenizer.swift    # BERT WordPiece tokenizer feeding the CoreML embedder
    │   ├── TextEmbeddingManager.swift  # CoreML wrapper generating query embeddings on-device
    │   ├── VectorRAGManager.swift      # SIMD-accelerated vector search engine (raw SQLite3 C API)
    │   ├── LLMInferenceManager.swift   # Google AI Edge LiteRT-LM Swift API wrapper
    │   └── SafetyFilter.swift          # Non-diagnostic regular expression filter
    └── Views/
        ├── ContentView.swift           # Primary container & state coordination view
        ├── EmergencyButtonView.swift   # Single circular push-to-talk button
        └── SubtitleCardView.swift      # High-contrast action checklist card
```

## Guidelines for AI Coding Agents

When generating, modifying, or refactoring code in this project, adhere to the following principles:

1. **Keep Managers Modular** — Each core capability (Speech, TTS, Camera, Text Embedding, Vector Search, LLM Inference) must remain isolated in its dedicated Manager class inside the `Core/` directory.
2. **Prioritize Swift Concurrency** — Use modern `async`/`await`, `@MainActor`, and Combine publishers for UI updates and state management. Avoid blocking the main UI thread during vector searches or model initialization.
3. **Handle Permissions Gracefully** — Ensure all iOS hardware access calls (Microphone, Speech Recognition, Camera) check authorization status before attempting capture.
4. **Enforce Hardcoded Safety Rules** — Never strip out or bypass the `SafetyFilter` before passing text to `TTSManager`.
5. **No Network Dependencies** — Do not introduce third-party Swift packages that rely on remote APIs or network requests.
6. **Keep Embedding Spaces in Sync** — `TextEmbeddingManager`'s bundled CoreML model must always match the embedding model used by `OffLineTools/build_vector_db.py` to generate `protocols.db`. Cosine similarity is meaningless if these diverge; never swap one without regenerating/re-validating the other.
7. **Fail Closed, Never Fall Back to Network** — If on-device speech recognition, embedding, or LLM inference is unavailable (e.g. `SFSpeechRecognizer.supportsOnDeviceRecognition == false`), surface a clear in-app error state. Never silently fall back to a server-based/networked alternative.
8. **Only Cite What We Are Licensed to Ship** — Every chunk in `protocols.db` must come from a source with a verified redistribution license recorded in `OffLineTools/sources.manifest.json`. See `OffLineTools/SOURCES.md`; note that NOLS course materials are copyrighted and are **not** currently licensed for ingestion despite being named in the mission statement above. For this sprint, the corpus is restricted to the sources already verified in the manifest (ATP 4-02.11, TCCC Handbook v5, NASEMSO Guidelines v3.0) — do not add new sources without recording their license first.
9. **Sprint Task Ownership** — See `docs/superpowers/plans/2026-08-01-hackathon-sprint.md` for the full task breakdown, or the per-person files in the same directory (`2026-08-01-pablo.md`, `2026-08-01-vaibhav.md`, `2026-08-01-daniel.md`, `2026-08-01-sachin.md`) for an individual's assigned tasks only.