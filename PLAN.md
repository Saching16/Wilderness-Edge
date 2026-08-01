# Wilderness Edge — Implementation Plan

## Executive Summary & Workflow Protocol

This document outlines the exact sequential roadmap for building Wilderness Edge.

To maintain high code quality and prevent system integration failures, AI coding agents must execute this plan phase by phase. **Agents must not proceed to a subsequent phase until all verification criteria for the current phase are fully met.**

> **Hackathon sprint note (2026-08-01):** For the Build with Gemma NYC hackathon (~4-hour window), this plan is trimmed and parallelized across a 4-person team. See `plans/scope-design.md` for what's cut and why, and `plans/hackathon-sprint.md` for the actual task-by-task sprint plan (or the per-person files in the same directory: `pablo.md`, `vaibhav.md`, `daniel.md`, `sachin.md`). The phase-by-phase roadmap below remains the project's long-term reference, but the sprint plan is authoritative for what to build *this cycle*. Key sprint deltas from the phases below: Gemma 4 **E4B** (prebuilt, not E2B), **no LoRA fine-tune** (Phase 0 §6 and its dependents are out of scope), and memory-footprint verification criteria are a soft target rather than a hard gate.

## Part 1: Prerequisites & Human Tasks (Outside Cursor)

Before AI agents begin writing application code inside Xcode, the human developer must complete the following manual setup steps:

### 1. Hardware & Environment Preparation

- [ ] **Apple Developer Mode:** Enable Developer Mode on the physical iPhone (Settings → Privacy & Security → Developer Mode) and restart the device.
- [ ] **Mac Trust:** Connect the physical iPhone to the Mac via USB and select "Trust This Computer".
- [ ] **Xcode Installation:** Ensure Xcode 15.0 or newer is installed on the Mac.

### 2. Offline Asset Generation (Colab / Laptop)

- [ ] **Assemble the Source Corpus:** Review `OffLineTools/SOURCES.md`, confirm each source's redistribution license, and run `python fetch_sources.py` to download the vetted public-domain corpus. Note that NOLS materials are **not** licensed for ingestion — see Tier 4 in `SOURCES.md`.
- [ ] **Generate Vector Database:** Run `python build_vector_db.py` to produce `protocols.db` plus `embedding_parity_fixtures.json`. Inspect output first with `--dry-run`.
- [ ] **Export the On-Device Query Embedder:** Run `python export_embedder_coreml.py` to produce `query-embedder.mlpackage` and the WordPiece tokenizer assets. The script's parity check must pass — it verifies that on-device embeddings reproduce the database's embedding space (cosine similarity ≈ 1.0).
- [ ] **Fetch the Gemma Model (hackathon sprint):** Download the prebuilt `litert-community/gemma-4-E4B-it-litert-lm` bundle directly (`gemma-4-E4B-it.litertlm`). No custom fine-tune this sprint — `OffLineTools/train_lora_colab.py` (LoRA adapter on Gemma 4 E2B via Unsloth) is deferred past the hackathon; see the sprint plan.
- [ ] **Bundle Assets into Xcode:** Drag `protocols.db`, `query-embedder.mlpackage`, `query-embedder-vocab.txt`, `query-embedder-tokenizer.json`, and `gemma-4-E4B-it.litertlm` into `WildernessEdge/Resources/`, and `embedding_parity_fixtures.json` into the test target. Xcode compiles the `.mlpackage` into `.mlmodelc` at build time — do not commit a prebuilt `.mlmodelc`.

### 3. Xcode Project Creation & Entitlements

- [x] **Create App Target:** `WildernessEdge` iOS 17+ SwiftUI app via XcodeGen (`project.yml` → `xcodegen generate`).
- [ ] **Add Swift Package Dependencies:** LiteRT-LM SPM deferred to Phase 3 (URL noted in `project.yml`). Do not add `MediaPipeTasksGenAI`.
- [x] **Configure Entitlements:** `WildernessEdge/App/WildernessEdge.entitlements` includes `com.apple.developer.kernel.increased-memory-limit`.
- [x] **Declare Privacy Keys in `Info.plist`:**
  - `NSMicrophoneUsageDescription` — Usage string for offline speech query recording.
  - `NSSpeechRecognitionUsageDescription` — Usage string for on-device speech transcription.
  - `NSCameraUsageDescription` — Usage string for visual context grounding snapshots.

## Part 2: Agent Execution Phases

### Phase 0: Offline Asset Tooling (No Xcode Required)

**Objective:** Build the Python pipeline that produces every data asset the app depends on. This phase gates Part 1 §2 — those human tasks cannot be performed until these scripts exist.

#### Action Items

1. ~~Create `OffLineTools/build_vector_db.py`~~ — PDF extraction, license-manifested chunking with section/page provenance, embedding, and SQLite output. **Done.**
2. ~~Create `OffLineTools/export_embedder_coreml.py`~~ — traces the same embedding model (with mean pooling and L2 normalization folded into the graph) to a CoreML package, exports the WordPiece vocabulary for the Swift tokenizer, and fails the build on embedding-space mismatch. **Done.**
3. ~~Create `OffLineTools/sources.manifest.json`, `fetch_sources.py`, and `SOURCES.md`~~ — vetted corpus with recorded licenses. **Done.**
4. ~~Create `OffLineTools/query_protocols.py`~~ — offline retrieval harness mirroring `VectorRAGManager`, used to calibrate the similarity threshold. **Done.**
5. ~~Create `OffLineTools/build_training_data.py`~~ — derives grounded / refusal / diagnosis-deflection training examples from `protocols.db`. **Done (seed quality; needs clinical review).**
6. ~~Create `OffLineTools/train_lora_colab.py`~~ — Unsloth LoRA fine-tune on Gemma 4 E2B with the vision tower frozen, 16-bit merge, and int4 `.litertlm` export including the vision encoder. **Written; unrun — requires a Colab GPU and gated Gemma weights. Out of scope for the 2026-08-01 hackathon sprint: use the prebuilt `litert-community/gemma-4-E4B-it-litert-lm` bundle instead (see sprint plan Task A3).**

#### Verification Criteria (Phase 0)

- [x] `python build_vector_db.py --dry-run` produces coherent, correctly-attributed chunks from the vetted corpus.
- [x] `python build_vector_db.py` writes a single self-contained `protocols.db` with populated `meta`, `sources`, and `chunks` tables. *(2192 chunks / 8.6 MB from 3 sources.)*
- [x] `python export_embedder_coreml.py` passes its parity check against the fixtures emitted by the database build. *(All 6 fixtures at cosine ≈ 1.000.)*
- [x] Every source in the corpus has a recorded, verified redistribution license in `sources.manifest.json`.
- [x] `python query_protocols.py` separates on-topic from off-topic queries by a clear margin. *(0.53–0.67 vs 0.17.)*
- [ ] `python build_training_data.py` output has been audited by a WFR/EMS-qualified reviewer. **Deferred past the hackathon sprint — no LoRA fine-tune this cycle.**
- [ ] `train_lora_colab.py` completes on a Colab GPU and exports a `.litertlm` bundle that `litert-lm run` can load. **Deferred past the hackathon sprint.**
- [x] ~~The exported bundle's on-device resident memory is measured and fits under the Phase 3 ceiling.~~ **Superseded for the hackathon sprint:** ship the prebuilt `litert-community/gemma-4-E4B-it-litert-lm` bundle directly (no custom fine-tune to measure); formal resident-memory measurement against a strict ceiling is a soft target this cycle, not a blocking gate (see AGENTS.md "Device Memory Safety").

### Phase 1: Native Audio, Speech & Safety Infrastructure

**Objective:** Establish local speech recognition, text-to-speech synthesis, single-frame camera capture, and output safety sanitization.

#### Action Items

1. ~~Create `Core/SpeechManager.swift` wrapping `SFSpeechRecognizer`.~~ **Done.** On-device only (`requiresOnDeviceRecognition = true`); publishes `.onDeviceUnavailable` and never falls back to network recognition.
2. ~~Create `Core/TTSManager.swift` wrapping `AVSpeechSynthesizer`.~~ **Done.**
3. ~~Create `Core/CameraManager.swift` using AVFoundation.~~ **Done.** Session pre-warmed via `prewarm()` at launch.
4. ~~Create `Core/SafetyFilter.swift`.~~ **Done.** Regex sanitizer replaces diagnostic/prescriptive language with citation framing.
5. ~~Create `WildernessEdgeTests/SafetyFilterTests.swift`.~~ **Done.** Host-free XCTest suite (scheme `WildernessEdgeTests`).

#### Verification Criteria (Phase 1)

- [ ] Speaking into the microphone populates the transcription property in real time with Airplane Mode enabled. *(Requires physical device — Phase 1 scaffold UI supports Start/Stop Listening.)*
- [ ] The TTS engine successfully speaks test strings out loud. *(Requires physical device / Simulator audio — "Speak Filtered" button wired.)*
- [x] Test phrases containing diagnostic terms (e.g., "The diagnosis is a fracture") are intercepted by `SafetyFilter` and replaced with standard protocol citation language.
- [x] `SafetyFilterTests` passes in CI/Simulator without requiring a physical device. *(5/5 tests via `xcodebuild test -scheme WildernessEdgeTests`.)*
- [x] Forcing `supportsOnDeviceRecognition` to `false` results in a visible error state, not a hang or silent no-op. *(Covered by `testOnDeviceUnavailableErrorSurfacesDistinctState` + `SpeechManager.simulateOnDeviceUnavailable()` / probe injection.)*

**Phase 1 gate for Phase 2:** Simulator/CI criteria above are green. Device mic + TTS Airplane Mode checks remain as human/device smoke tests and do not block Phase 2 start.

**Handoff notes for next agent:**
- Regenerate Xcode project with `xcodegen generate` (source of truth: `project.yml`).
- Run Phase 1 tests: `xcodebuild test -scheme WildernessEdgeTests -destination 'platform=iOS Simulator,name=iPhone 17'`.
- App target currently excludes `Assets.xcassets` because macOS `com.apple.provenance` xattrs make ad-hoc `codesign` treat `Assets.car` / AppIcon PNGs as nested unsigned code. Re-enable the catalog once a Development Team signs the app (or after finding a provenance-safe packaging path).
- LiteRT-LM SPM is deferred to Phase 3 (URL recorded in `project.yml` comments).
- Bundle offline assets (`protocols.db`, embedder, `.litertlm`) into `WildernessEdge/Resources/` before Phase 2/3.

### Phase 2: On-Device Vector RAG Search Engine

**Objective:** Build a low-latency, SIMD-accelerated vector search manager that queries the bundled SQLite database (`protocols.db`).

#### Action Items

1. Create `Core/WordPieceTokenizer.swift` implementing BERT-style WordPiece tokenization against the bundled `query-embedder-vocab.txt`, honoring the lowercase/accent-stripping behavior, special token ids, and 128-token sequence length recorded in `query-embedder-tokenizer.json`. CoreML cannot accept strings, so this must exist before the embedder can be called.
2. Create `Core/TextEmbeddingManager.swift` wrapping the bundled `query-embedder.mlpackage` CoreML model. Expose a function that takes transcribed query text and returns a 384-dimensional embedding vector, generated entirely on-device. The model already applies mean pooling and L2 normalization internally, so no post-processing is needed.
3. Create `Core/VectorRAGManager.swift`.
4. Implement SQLite connection handling using the raw `SQLite3` C API (no third-party wrapper), opened with `SQLITE_OPEN_READONLY` since the bundle is not writable, to read protocol text chunks and their embeddings from `protocols.db`. Embeddings are stored as little-endian `float32` BLOBs (384 floats, 1536 bytes) and can be copied straight into `[Float]`.
5. Implement vector similarity using Apple's Accelerate framework. Because stored vectors and the embedder's output are both pre-normalized, a `vDSP_dotpr` dot product is equivalent to cosine similarity — do not recompute magnitudes.
6. Expose a search function accepting query embeddings and returning the top-K matching protocol chunks with source citations. If the best match's similarity score falls below a defined confidence threshold, return a distinct "no confident match" result rather than a low-quality chunk.
7. Create `WildernessEdgeTests/VectorRAGManagerTests.swift` (XCTest, runs on Simulator/CI) validating the cosine similarity math against hand-computed reference vectors, validating the "no confident match" path, and asserting embedding-space parity against the bundled `embedding_parity_fixtures.json`.

#### Verification Criteria (Phase 2)

- [ ] `VectorRAGManager` successfully connects to `protocols.db` bundled in the app package.
- [ ] `TextEmbeddingManager` produces embeddings on-device whose cosine similarity to the offline-generated embedding of the same sentence is ≈ 1.0 (validating embedding-space parity with `build_vector_db.py`).
- [ ] Executing a sample query vector against 1,000+ stored chunks returns top matching results in under 10 milliseconds.
- [ ] Query results correctly include source manual titles and section numbers.
- [ ] A query with no confident match returns the distinct "no confident match" result instead of a misleading low-quality chunk.
- [ ] `VectorRAGManagerTests` passes in CI/Simulator without requiring a physical device.

### Phase 3: LiteRT-LM Multimodal Integration & Orchestration

**Objective:** Integrate the local, natively multimodal Gemma 4 **E4B** model (prebuilt `litert-community/gemma-4-E4B-it-litert-lm`, no custom fine-tune this sprint) via the Google AI Edge **LiteRT-LM Swift API** and construct the central orchestration pipeline, including direct image input.

#### Action Items

1. Create `Core/LLMInferenceManager.swift` wrapping LiteRT-LM's `Engine`/`Conversation` Swift API.
2. Configure `EngineConfig` pointing to `gemma-4-E4B-it.litertlm` inside the app bundle, enabling the GPU backend plus the vision backend (`visionBackend: .cpu()` or `.gpu()` as benchmarking dictates) so image input is supported.
3. Construct a multimodal `Message` per query combining: `Content.imageFile(...)` (or in-memory image data) from the camera snapshot, `Content.text(...)` carrying retrieved RAG context chunks, system instructions, and the user transcript. If `VectorRAGManager` returned "no confident match," omit fabricated context and explicitly instruct the model to state that no matching protocol was found rather than improvising.
4. Implement async token generation streaming via `conversation.sendMessage(...)`.
5. Handle model initialization failure (e.g. asset missing/corrupt, insufficient memory) by surfacing a blocking startup error state rather than allowing the app to proceed into a broken push-to-talk loop.

#### Verification Criteria (Phase 3)

- [ ] `LLMInferenceManager` initializes the model without memory crashes or low-memory warnings.
- [ ] Model processes a prompt combining a camera snapshot and RAG context and generates structured protocol text locally in Airplane Mode.
- [ ] A query engineered to have no RAG match produces an honest "no matching protocol found" response rather than a fabricated one.
- [ ] Total active memory usage during model inference is reasonable on the iPhone 16 Plus. **Soft target this sprint, not a hard gate** (was a strict 2.8 GB ceiling for the original 6GB-device design target — see AGENTS.md "Device Memory Safety").
- [ ] Corrupting or removing the bundled model asset in a debug build surfaces the startup error state instead of crashing or hanging.

### Phase 4: SwiftUI User Interface & State Machine

**Objective:** Build the single-button emergency user interface and wire all core managers into an interactive state machine.

#### Action Items

1. Create `Views/EmergencyButtonView.swift`: A large, high-contrast, circular push-to-talk button supporting press-and-hold gestures with visual state indicators (Idle, Listening, Processing, Speaking, Error).
2. Create `Views/SubtitleCardView.swift`: A high-contrast overlay card displaying the active source citation and spoken checklist text, plus a distinct visual treatment for the Error state.
3. Build `Views/ContentView.swift`: The main container view instantiating all managers, binding published properties to UI states, and coordinating the full push-to-talk pipeline:
   - **Button Down:** Trigger Camera Snapshot + Start Audio Recording.
   - **Button Release:** Stop Audio Recording → Embed Query Text → Run Vector RAG → Query Gemma via LiteRT-LM (image + context) → Apply Safety Filter → Trigger TTS Audio Playback.
   - **Error Transitions:** Route to the Error state (with a spoken/displayed explanation, no crash) on: empty/failed transcription, on-device speech recognition unavailable, and LLM initialization failure. A "no confident RAG match" is *not* an error — it flows through normally and is spoken as an honest "no matching protocol found" response.

#### Verification Criteria (Phase 4)

- [ ] Pressing and holding the central button transitions UI state to "Listening" and captures audio/camera frame.
- [ ] Releasing the button triggers the full local execution loop (STT → Embed → RAG → Gemma via LiteRT-LM → Safety Filter → TTS).
- [ ] Spoken audio playback begins within 200 milliseconds of text generation.
- [ ] The subtitle card clearly displays the source citation (e.g., `[Source: NASEMSO National Model EMS Clinical Guidelines v3.0, Extremity Trauma, p. 128]`) alongside the checklist text.
- [ ] Triggering each defined error condition (empty transcription, on-device speech unavailable, LLM init failure) transitions to the Error state with a clear, spoken/displayed message rather than a silent failure or crash.

### Phase 5: Final Air-Gap Validation & Stress Testing

**Objective:** Validate system stability, air-gap integrity, and compliance under simulated disaster conditions.

#### Action Items

1. Perform physical device testing in complete Airplane Mode (Wi-Fi OFF, Cellular OFF, Bluetooth OFF).
2. Perform rapid repeated push-to-talk queries to test memory retention and verify that no memory leaks occur.
3. Audit output responses against non-diagnostic safety rules using a suite of test prompts designed to bait diagnoses.
4. Run the full `SafetyFilterTests` and `VectorRAGManagerTests` XCTest suites one final time and confirm both pass on the CI/Simulator target.
5. Deliberately exercise all defined Error-state triggers (Airplane Mode + revoked speech permission, corrupted model asset, no-RAG-match queries) on the physical device to confirm graceful, non-crashing fallback behavior end-to-end.

#### Verification Criteria (Phase 5)

- [ ] Application operates seamlessly with zero active network interfaces.
- [ ] App completes 20 consecutive query loops without crashing or exceeding memory caps. **Reduced for the hackathon sprint:** demo script runs reliably twice in a row in Airplane Mode (see sprint plan Task E1); the full 20-loop stress pass is deferred past the hackathon.
- [ ] All responses cite accredited field manuals and strictly present action checklists without issuing medical diagnoses.
- [ ] `SafetyFilterTests` and `VectorRAGManagerTests` pass in CI/Simulator.
- [ ] All Error-state triggers are confirmed to fail closed (never silently succeed or fall back to network) on the physical device.