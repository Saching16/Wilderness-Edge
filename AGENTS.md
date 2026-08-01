# Wilderness Edge — AI Agent Context & Rulebook

## Mission & Purpose

Wilderness Edge is a 100% air-gapped, voice-first, and vision-assisted emergency protocol assistant built natively for iOS. Designed for first responders in zero-connectivity off-grid environments, it provides instant decision support by searching accredited field manuals (e.g., NOLS Wilderness First Aid, State EMS Protocols) and presenting structured assessment checklists.

This document serves as the primary context, system rulebook, and architectural guide for AI coding agents working on this codebase.

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

### 3. Base Device RAM Safety (6GB Unified Memory Target)

- The app is designed to run on a physical base-model iPhone with 6GB of unified memory.
- Language model footprint must be constrained to quantized Gemma 4 E2B (~1.6 GB).
- High-memory system entitlements must be enabled in the project build settings to prevent low-memory operating system terminations.

## Technical Stack Overview

| Layer | Technology |
| --- | --- |
| **Target Platform** | Native iOS 17.0+ (Swift 5.9 / SwiftUI) |
| **LLM Runtime Engine** | Google MediaPipe Tasks GenAI SDK (`MediaPipeTasksGenAI`) |
| **Base Language Model** | Quantized Gemma 4 E2B (~1.6 GB download footprint) with a merged LoRA decision-tree adapter |
| **Speech-to-Text (STT)** | Native Apple Framework (`SFSpeechRecognizer`) forced to on-device recognition |
| **Text-to-Speech (TTS)** | Native Apple Framework (`AVSpeechSynthesizer`) |
| **Camera Capture** | Native Apple Framework (`AVFoundation`) for single-frame snapshot capture |
| **Vector Search Engine** | Local SQLite database (`protocols.db`) queried via SIMD-accelerated linear algebra using Apple's Accelerate framework |
| **Safety Verification** | Regular expression pattern matcher evaluating LLM text outputs prior to speech playback |

## System Architecture & Data Flow

```text
                  [ PUSH-TO-TALK SINGLE BUTTON (SwiftUI) ]
                                      │
          ┌───────────────────────────┴───────────────────────────┐
          ▼                                                       ▼
[ AVFoundation Snapshot ]                             [ SFSpeechRecognizer ]
(Single Camera Frame)                                 (On-Device Offline Audio)
          │                                                       │
          ▼                                                       ▼
Direct Visual Frame Context                              Transcribed Query Text
          │                                                       │
          └───────────────────────────┬───────────────────────────┘
                                      │
                                      ▼
                      [ Local Vector RAG Manager ]
             Accelerate SIMD Cosine Search over protocols.db
                                      │
                                      ▼
                     [ MediaPipe GenAI LLM Engine ]
                 Gemma 4 E2B + Decision LoRA Adapter
                                      │
                                      ▼
                    [ Safety Guardrail Regex Layer ]
             Sanitizes output & enforces source citations
                                      │
                                      ▼
                     [ AVSpeechSynthesizer Output ]
            Streams spoken checklist items out loud to user
```

## Repository Map & Architecture Blueprint

```text
WildernessEdge/
├── WildernessEdge.xcodeproj            # Main Xcode Project
├── AGENTS.md                           # This AI Agent Context & Rulebook
├── PLAN.md                             # Step-by-Step Implementation Plan & Checklists
├── OffLineTools/                       # Pre-processing scripts (Run on Laptop/Colab)
│   ├── build_vector_db.py              # Offline PDF parser & vector DB generator
│   └── train_lora_colab.py             # Unsloth QLoRA fine-tuning script
└── WildernessEdge/
    ├── App/
    │   ├── WildernessEdgeApp.swift     # Application Entry Point & Audio Session setup
    │   └── Info.plist                  # Entitlements & Privacy Permission Strings
    ├── Resources/
    │   ├── protocols.db                # Pre-indexed SQLite database containing embeddings
    │   └── gemma-4-e2b-wfr.task        # MediaPipe Gemma model file (~1.6 GB)
    ├── Core/
    │   ├── SpeechManager.swift         # On-device SFSpeechRecognizer wrapper
    │   ├── TTSManager.swift            # Native AVSpeechSynthesizer wrapper
    │   ├── CameraManager.swift         # AVFoundation single-frame snapshot capture
    │   ├── VectorRAGManager.swift      # SIMD-accelerated vector search engine
    │   ├── LLMInferenceManager.swift   # Google MediaPipe Tasks GenAI wrapper
    │   └── SafetyFilter.swift          # Non-diagnostic regular expression filter
    └── Views/
        ├── ContentView.swift           # Primary container & state coordination view
        ├── EmergencyButtonView.swift   # Single circular push-to-talk button
        └── SubtitleCardView.swift      # High-contrast action checklist card
```

## Guidelines for AI Coding Agents

When generating, modifying, or refactoring code in this project, adhere to the following principles:

1. **Keep Managers Modular** — Each core capability (Speech, TTS, Camera, Vector Search, LLM Inference) must remain isolated in its dedicated Manager class inside the `Core/` directory.
2. **Prioritize Swift Concurrency** — Use modern `async`/`await`, `@MainActor`, and Combine publishers for UI updates and state management. Avoid blocking the main UI thread during vector searches or model initialization.
3. **Handle Permissions Gracefully** — Ensure all iOS hardware access calls (Microphone, Speech Recognition, Camera) check authorization status before attempting capture.
4. **Enforce Hardcoded Safety Rules** — Never strip out or bypass the `SafetyFilter` before passing text to `TTSManager`.
5. **No Network Dependencies** — Do not introduce third-party Swift packages that rely on remote APIs or network requests.