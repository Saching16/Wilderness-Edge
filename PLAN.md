# Wilderness Edge — Implementation Plan

## Executive Summary & Workflow Protocol

This document outlines the exact sequential roadmap for building Wilderness Edge.

To maintain high code quality and prevent system integration failures, AI coding agents must execute this plan phase by phase. **Agents must not proceed to a subsequent phase until all verification criteria for the current phase are fully met.**

## Part 1: Prerequisites & Human Tasks (Outside Cursor)

Before AI agents begin writing application code inside Xcode, the human developer must complete the following manual setup steps:

### 1. Hardware & Environment Preparation

- [ ] **Apple Developer Mode:** Enable Developer Mode on the physical iPhone (Settings → Privacy & Security → Developer Mode) and restart the device.
- [ ] **Mac Trust:** Connect the physical iPhone to the Mac via USB and select "Trust This Computer".
- [ ] **Xcode Installation:** Ensure Xcode 15.0 or newer is installed on the Mac.

### 2. Offline Asset Generation (Colab / Laptop)

- [ ] **Generate Vector Database:** Run the offline Python pre-processing script (`OffLineTools/build_vector_db.py`) on your laptop using public domain field manuals (NOLS, EMS protocols) to generate `protocols.db`.
- [ ] **Train & Export Gemma Model:** Run the Google Colab script (`OffLineTools/train_lora_colab.py`) to fine-tune the decision-tree LoRA adapter on Gemma 4 E2B using Unsloth, merge the weights, and export the `.task` or quantized model asset (`gemma-4-e2b-wfr.task`).
- [ ] **Bundle Assets into Xcode:** Drag `protocols.db` and `gemma-4-e2b-wfr.task` directly into the `WildernessEdge/Resources/` directory inside the Xcode project navigator.

### 3. Xcode Project Creation & Entitlements

- [ ] **Create App Target:** Create a new iOS App project named `WildernessEdge` using SwiftUI and Swift.
- [ ] **Add Swift Package Dependencies:** Add `GoogleMediaPipeTasksGenAI` via Swift Package Manager (SPM).
- [ ] **Configure Entitlements:** Select the project target → Signing & Capabilities → Add capability: **Increased Memory Limit** (`com.apple.developer.kernel.increased-memory-limit`).
- [ ] **Declare Privacy Keys in `Info.plist`:**
  - `NSMicrophoneUsageDescription` — Usage string for offline speech query recording.
  - `NSSpeechRecognitionUsageDescription` — Usage string for on-device speech transcription.
  - `NSCameraUsageDescription` — Usage string for visual context grounding snapshots.

## Part 2: Agent Execution Phases

### Phase 1: Native Audio, Speech & Safety Infrastructure

**Objective:** Establish local speech recognition, text-to-speech synthesis, single-frame camera capture, and output safety sanitization.

#### Action Items

1. Create `Core/SpeechManager.swift` wrapping `SFSpeechRecognizer`. Configure audio buffer taps, enable `requiresOnDeviceRecognition = true`, and expose published transcription properties.
2. Create `Core/TTSManager.swift` wrapping `AVSpeechSynthesizer`. Configure `AVAudioSession` categories to ensure audio plays clearly over system channels.
3. Create `Core/CameraManager.swift` using AVFoundation to handle single-frame snapshot capture when recording begins.
4. Create `Core/SafetyFilter.swift` implementing regular expression pattern matching to detect and replace diagnostic or prescriptive language before speech output.

#### Verification Criteria (Phase 1)

- [ ] Speaking into the microphone populates the transcription property in real time with Airplane Mode enabled.
- [ ] The TTS engine successfully speaks test strings out loud.
- [ ] Test phrases containing diagnostic terms (e.g., "The diagnosis is a fracture") are intercepted by `SafetyFilter` and replaced with standard protocol citation language.

### Phase 2: On-Device Vector RAG Search Engine

**Objective:** Build a low-latency, SIMD-accelerated vector search manager that queries the bundled SQLite database (`protocols.db`).

#### Action Items

1. Create `Core/VectorRAGManager.swift`.
2. Implement SQLite connection handling to read pre-stored protocol text chunks and 384-dimensional floating-point embeddings from `protocols.db`.
3. Implement cosine similarity vector distance calculation using Apple's Accelerate framework (vDSP/SIMD functions) for maximum performance.
4. Expose a search function accepting query embeddings and returning the top-K matching protocol chunks with source citations.

#### Verification Criteria (Phase 2)

- [ ] `VectorRAGManager` successfully connects to `protocols.db` bundled in the app package.
- [ ] Executing a sample query vector against 1,000+ stored chunks returns top matching results in under 10 milliseconds.
- [ ] Query results correctly include source manual titles and section numbers.

### Phase 3: MediaPipe LLM Integration & Orchestration

**Objective:** Integrate the local Gemma 4 E2B model via Google MediaPipe Tasks GenAI and construct the central orchestration pipeline.

#### Action Items

1. Create `Core/LLMInferenceManager.swift` wrapping `LlmInference` from `MediaPipeTasksGenAI`.
2. Configure model initialization pointing to `gemma-4-e2b-wfr.task` inside the app bundle.
3. Construct prompt formatting logic that combines retrieved RAG context chunks, system instructions, and user transcripts.
4. Implement async token generation streaming.

#### Verification Criteria (Phase 3)

- [ ] `LLMInferenceManager` initializes the model without memory crashes or low-memory warnings.
- [ ] Model processes a prompt with RAG context and generates structured protocol text locally in Airplane Mode.
- [ ] Total active memory usage during model inference stays safely below 2.8 GB on the target device.

### Phase 4: SwiftUI User Interface & State Machine

**Objective:** Build the single-button emergency user interface and wire all core managers into an interactive state machine.

#### Action Items

1. Create `Views/EmergencyButtonView.swift`: A large, high-contrast, circular push-to-talk button supporting press-and-hold gestures with visual state indicators (Idle, Listening, Processing, Speaking).
2. Create `Views/SubtitleCardView.swift`: A high-contrast overlay card displaying the active source citation and spoken checklist text.
3. Build `Views/ContentView.swift`: The main container view instantiating all managers, binding published properties to UI states, and coordinating the full push-to-talk pipeline:
   - **Button Down:** Trigger Camera Snapshot + Start Audio Recording.
   - **Button Release:** Stop Audio Recording → Run Vector RAG → Query Gemma via MediaPipe → Apply Safety Filter → Trigger TTS Audio Playback.

#### Verification Criteria (Phase 4)

- [ ] Pressing and holding the central button transitions UI state to "Listening" and captures audio/camera frame.
- [ ] Releasing the button triggers the full local execution loop (STT → RAG → Gemma → Safety Filter → TTS).
- [ ] Spoken audio playback begins within 200 milliseconds of text generation.
- [ ] The subtitle card clearly displays the source citation (e.g., `[Source: NOLS WFR Section 4.2]`) alongside the checklist text.

### Phase 5: Final Air-Gap Validation & Stress Testing

**Objective:** Validate system stability, air-gap integrity, and compliance under simulated disaster conditions.

#### Action Items

1. Perform physical device testing in complete Airplane Mode (Wi-Fi OFF, Cellular OFF, Bluetooth OFF).
2. Perform rapid repeated push-to-talk queries to test memory retention and verify that no memory leaks occur.
3. Audit output responses against non-diagnostic safety rules using a suite of test prompts designed to bait diagnoses.

#### Verification Criteria (Phase 5)

- [ ] Application operates seamlessly with zero active network interfaces.
- [ ] App completes 20 consecutive query loops without crashing or exceeding memory caps.
- [ ] All responses cite accredited field manuals and strictly present action checklists without issuing medical diagnoses.