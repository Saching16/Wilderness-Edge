# Wilderness Edge — Hackathon Sprint Plan: Sachin (Track D — LiteRT-LM Integration & Orchestration)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> This is your individual track from the full team plan at `plans/hackathon-sprint.md`. Read that file if you want the other 3 tracks' context — Pablo (offline assets), Vaibhav (RAG engine), and Daniel (native I/O/UI/safety) are each working their own file in parallel from hour zero.

**Goal:** Wire Gemma 4 E4B into the app via LiteRT-LM: add the SPM package, build `LLMInferenceManager` combining camera image + RAG context + transcript into a multimodal prompt, and — once Pablo's and Vaibhav's and Daniel's pieces land — wire the whole thing into `ContentView`'s real inference pipeline. You have the most cross-track dependencies; start on the parts you can build standalone first.

**Tech Stack:** Swift 5.9, LiteRT-LM Swift API (Google AI Edge).

## Global Constraints (apply to your track too)

- Zero network requests at runtime — never fall back to a networked model on any failure. Fail closed with a visible error state instead.
- Model is Gemma 4 **E4B** (not E2B) — use the prebuilt `litert-community/gemma-4-E4B-it-litert-lm` bundle. No custom LoRA fine-tune this sprint.
- "No confident RAG match" is a normal, honestly-spoken result, not an error — your prompt construction must explicitly instruct the model not to fabricate context when this happens.
- All LLM text output must pass through `SafetyFilter.sanitize(_:)` before display or TTS — this happens in `ContentView` (Daniel's code), not in your `LLMInferenceManager`, but make sure your `generate(...)` return value flows into that filter and isn't spoken/displayed before it.
- No hard memory-footprint gate this sprint.

## Dependencies — this is the track most other people's work feeds into

- **You need `google-ai-edge/LiteRT-LM` added to `project.yml` from hour zero** (Task D1) — don't wait for anyone.
- **Pablo** delivers `gemma-4-E4B-it.litertlm` (his Task A3) — this is **Checkpoint 2**, target hour 1–1.5. Until then, build and test `LLMInferenceManager`'s fail-closed path against the missing asset.
- **Vaibhav** delivers a functionally complete `VectorRAGManager.search(...)` (his Task B3) — this is **Checkpoint 3**, target hour 1.5–2. Message him if you haven't heard by then.
- **Daniel** delivers `ContentView`'s `runInferencePipeline` closure signature (his Task C4) — this is **Checkpoint 4**, target hour 2–2.5. You need the exact signature `(String, UIImage?) async -> (citation: String?, checklistText: String)` before wiring in.

---

## Task D1: Add the LiteRT-LM SPM package to project.yml

**Files:**
- Modify: `project.yml`

**Interfaces:**
- Produces: the `LiteRTLM` SPM product linked into the `WildernessEdge` target, required by Task D2.

- [ ] **Step 1: Edit project.yml to add the package**

Current `project.yml` has this comment block just above `targets:` (read it first to confirm line context):

```yaml
# LiteRT-LM SPM is added in Phase 3 (LLMInferenceManager). Declaring it here would
# force native LLM linkage during Phase 1 audio/safety verification builds.
# Package URL when wiring Phase 3: https://github.com/google-ai-edge/LiteRT-LM (product: LiteRTLM, from: 0.12.0)
targets:
```

Replace it with:

```yaml
packages:
  LiteRTLM:
    url: https://github.com/google-ai-edge/LiteRT-LM
    from: 0.12.0
targets:
```

Then, inside the `WildernessEdge` target definition, add a `dependencies:` key at the target level (sibling of `sources:`/`settings:`):

```yaml
  WildernessEdge:
    type: application
    platform: iOS
    dependencies:
      - package: LiteRTLM
        product: LiteRTLM
    sources:
      - path: WildernessEdge
        excludes:
          - Resources/**
```

(Insert `dependencies:` right after `platform: iOS` and before the existing `sources:` key — do not duplicate the `sources:` key.)

- [ ] **Step 2: Regenerate and verify the package resolves**

```bash
xcodegen generate
open WildernessEdge.xcodeproj
```

In Xcode, wait for Swift Package Manager to resolve `LiteRT-LM`. Expected: no red errors in the Package Dependencies section; `import LiteRTLM` becomes available.

- [ ] **Step 3: Commit**

```bash
git add project.yml
git commit -m "Add LiteRT-LM SPM package dependency"
```

## Task D2: LLMInferenceManager with a stubbed model path

**Files:**
- Create: `WildernessEdge/Core/LLMInferenceManager.swift`

**Interfaces:**
- Consumes (at Checkpoint 3): `VectorRAGManager.RAGResult` from Vaibhav's Task B3.
- Produces: `LLMInferenceManager.initialize() async throws`, `LLMInferenceManager.generate(transcript: String, ragResult: VectorRAGManager.RAGResult?, image: UIImage?) async throws -> String` — consumed by `ContentView.runInferencePipeline` (Daniel's Task C4, wired at Checkpoint 4).

**Before starting this task:** the exact type/method names below (`EngineConfig`, `Engine`, `Conversation`, `Content`, `Message`, `.imageData`, `.gpu()`) are a best-available reconstruction from PLAN.md's prose description of the LiteRT-LM Swift API, not verified against the real package. **First step of this task is to open the resolved `LiteRT-LM` package in Xcode (after Task D1) and read its actual public API** — jump to definition on `Engine`/`Conversation`/`Content`/`Message`, or check the package's README/examples on GitHub (`google-ai-edge/LiteRT-LM`). Adjust the code below to match whatever the real API surface is; the logic (build a system instruction + context block + transcript + optional image, stream tokens, concatenate) stays the same regardless of exact type names.

- [ ] **Step 1: Implement the manager against the LiteRT-LM Swift API**

```swift
// WildernessEdge/Core/LLMInferenceManager.swift
import Foundation
import LiteRTLM
import UIKit

/// Wraps LiteRT-LM's Engine/Conversation API for local multimodal Gemma 4 E4B inference.
/// Never falls back to a networked model on any failure.
@MainActor
final class LLMInferenceManager: ObservableObject {
    enum LLMError: LocalizedError {
        case modelAssetMissing
        case initializationFailed(String)
        case generationFailed(String)

        var errorDescription: String? {
            switch self {
            case .modelAssetMissing:
                return "The Gemma 4 E4B model bundle is missing from the app package."
            case .initializationFailed(let message):
                return "Model failed to initialize: \(message)"
            case .generationFailed(let message):
                return "Generation failed: \(message)"
            }
        }
    }

    @Published private(set) var isReady = false
    @Published private(set) var initializationError: LLMError?

    private var engine: Engine?
    private var conversation: Conversation?

    /// Must be called once at app startup, before the first query. Surfaces a blocking
    /// startup error rather than allowing the app into a broken push-to-talk loop.
    func initialize(bundle: Bundle = .main) async {
        guard let modelURL = bundle.url(forResource: "gemma-4-E4B-it", withExtension: "litertlm") else {
            initializationError = .modelAssetMissing
            return
        }

        do {
            let config = EngineConfig(
                modelPath: modelURL.path,
                visionBackend: .gpu()
            )
            let engine = try Engine(config: config)
            self.engine = engine
            self.conversation = try engine.createConversation()
            isReady = true
        } catch {
            initializationError = .initializationFailed(error.localizedDescription)
        }
    }

    /// Combines the camera snapshot, retrieved RAG context, and transcript into one
    /// multimodal prompt. When `ragResult` is `.noConfidentMatch` (or nil), the model is
    /// explicitly instructed not to fabricate protocol content.
    func generate(
        transcript: String,
        ragResult: VectorRAGManager.RAGResult?,
        image: UIImage?
    ) async throws -> String {
        guard let conversation else {
            throw LLMError.generationFailed("Model not initialized.")
        }

        let systemInstruction = """
        You are a non-diagnostic, non-prescriptive field-protocol assistant. Only present \
        retrieved checklist steps with their citation. Never state a diagnosis or a drug dose. \
        If no protocol context is provided below, say plainly that no matching protocol was \
        found instead of guessing.
        """

        let contextBlock: String
        switch ragResult {
        case .match(let chunks):
            contextBlock = chunks.map { "\($0.citation)\n\($0.text)" }.joined(separator: "\n\n")
        case .noConfidentMatch, .none:
            contextBlock = "NO_MATCHING_PROTOCOL_FOUND"
        }

        var contents: [Content] = []
        if let image, let imageData = image.jpegData(compressionQuality: 0.8) {
            contents.append(.imageData(imageData))
        }
        contents.append(.text("\(systemInstruction)\n\nRetrieved context:\n\(contextBlock)\n\nUser question: \(transcript)"))

        let message = Message(contents: contents)

        do {
            var response = ""
            for try await token in try conversation.sendMessage(message) {
                response += token
            }
            return response
        } catch {
            throw LLMError.generationFailed(error.localizedDescription)
        }
    }
}
```

- [ ] **Step 2: Manually verify initialization failure surfaces cleanly (before the real model bundle exists)**

In a Simulator run, call `await LLMInferenceManager().initialize()` from a temporary `#if DEBUG` hook and print `initializationError`. Expected (pre-Checkpoint 2): `.modelAssetMissing`, since `gemma-4-E4B-it.litertlm` isn't in `Resources/` yet — this confirms the fail-closed path works before wiring the real bundle.

- [ ] **Step 3: Commit**

```bash
git add WildernessEdge/Core/LLMInferenceManager.swift
git commit -m "Add LLMInferenceManager wrapping LiteRT-LM Engine/Conversation"
```

## Task D3: Swap in the real model bundle (Checkpoint 2)

**Files:** none (verification against Task D2's existing code)

- [ ] **Step 1: Confirm the model bundle is present**

```bash
ls -lh WildernessEdge/Resources/gemma-4-E4B-it.litertlm
```

Expected: file exists (from Pablo's Task A3).

- [ ] **Step 2: Run initialize() on the physical device (not Simulator — LiteRT-LM's GPU backend needs real Metal)**

Expected: `isReady == true`, `initializationError == nil`. If it fails, check the exact filename matches `gemma-4-E4B-it.litertlm` (case-sensitive) and that `visionBackend: .gpu()` doesn't crash on this device — fall back to `.cpu()` in `EngineConfig` if GPU backend init fails, and note the tradeoff for the writeup.

- [ ] **Step 3: Run a single manual multimodal generation call**

Temporarily call `generate(transcript: "How do I treat a bleeding wound?", ragResult: nil, image: cameraManager.latestSnapshot)` and print the result. Expected: coherent text response referencing the "NO_MATCHING_PROTOCOL_FOUND" instruction (since RAG isn't wired yet) — confirms the model and multimodal input path work before Checkpoint 3.

- [ ] **Step 4: Nothing to commit** (verification only, no code changed unless the GPU→CPU fallback was needed — if so, commit that change).

## Task D4: Wire real VectorRAGManager output into generate() (Checkpoint 3)

**Files:**
- Modify: `WildernessEdge/Core/LLMInferenceManager.swift` (no signature change — verification that Vaibhav's real output flows through Task D2's existing `ragResult` parameter)

- [ ] **Step 1: Confirm Vaibhav's VectorRAGManager is complete**

Check with Vaibhav that `WildernessEdgeTests/VectorRAGManagerTests.swift` passes and `WildernessEdge/Resources/protocols.db` is the real corpus.

- [ ] **Step 2: Manually call the real pipeline end-to-end (still outside ContentView)**

```swift
let embedder = try TextEmbeddingManager()
let rag = try VectorRAGManager(databasePath: Bundle.main.path(forResource: "protocols", ofType: "db")!)
let queryEmbedding = try embedder.embed("severe bleeding from the thigh")
let ragResult = rag.search(embedding: queryEmbedding, topK: 3, threshold: 0.35)
let response = try await llmManager.generate(transcript: "How do I treat severe bleeding?", ragResult: ragResult, image: nil)
print(response)
```

Expected: a response that cites one of the 3 real sources and gives checklist-style steps, not a fabricated diagnosis.

- [ ] **Step 3: No commit needed unless a bug surfaced** — if `VectorRAGManager`'s output type didn't match what `generate()` expects, fix the mismatch and commit.

## Task D5: Wire the full pipeline into ContentView (Checkpoint 4)

**Files:**
- Modify: `WildernessEdge/Views/ContentView.swift`

**Interfaces:**
- Replaces: `ContentView.runInferencePipeline`'s default stub closure (Daniel's Task C4) with a real implementation. Signature `(String, UIImage?) async -> (citation: String?, checklistText: String)` must not change.

- [ ] **Step 1: Confirm Daniel's ContentView state machine is ready**

Check with Daniel that his Task C4 is committed and the stub closure signature matches what's documented above.

- [ ] **Step 2: Instantiate the real managers in ContentView and replace the stub**

```swift
// In ContentView, add:
@StateObject private var llmManager = LLMInferenceManager()
private let embedder = try? TextEmbeddingManager()
private let ragManager = try? VectorRAGManager(
    databasePath: Bundle.main.path(forResource: "protocols", ofType: "db") ?? ""
)
```

Replace the default `runInferencePipeline` value with:

```swift
var runInferencePipeline: (String, UIImage?) async -> (citation: String?, checklistText: String) = { _, _ in
    (nil, "") // overwritten below in .task
}
```

And in `.task`, after `await cameraManager.prewarm()`, add:

```swift
await llmManager.initialize()
runInferencePipeline = { [ragManager, embedder, llmManager] transcript, image in
    guard let embedder, let ragManager else {
        return (nil, "Retrieval system unavailable.")
    }
    do {
        let queryEmbedding = try embedder.embed(transcript)
        let ragResult = ragManager.search(embedding: queryEmbedding, topK: 3, threshold: 0.35)
        let responseText = try await llmManager.generate(transcript: transcript, ragResult: ragResult, image: image)
        let citation: String? = {
            if case .match(let chunks) = ragResult { return chunks.first?.citation }
            return nil
        }()
        return (citation, responseText)
    } catch {
        return (nil, "Inference failed: \(error.localizedDescription)")
    }
}
```

Also handle `llmManager.initializationError` in `.task`: if non-nil after `initialize()`, set `appState = .error(...)` immediately so a broken model asset surfaces at launch, not on first query.

- [ ] **Step 3: Manually verify the fully-wired pipeline on the physical device in Airplane Mode**

Enable Airplane Mode. Press and hold the button, ask "How do I splint a broken arm?", release. Expected: Listening → Processing → Speaking states visible, the subtitle card shows a real citation (e.g. TCCC Handbook v5) and checklist text, and TTS speaks it aloud — end to end, no network.

- [ ] **Step 4: Commit**

```bash
git add WildernessEdge/Views/ContentView.swift
git commit -m "Wire full RAG + LiteRT-LM pipeline into ContentView push-to-talk flow"
```

---

## Final Validation (all 4 team members)

See `plans/hackathon-sprint.md` Task E1 (Airplane Mode demo run-through, twice) and Task E2 (Kaggle Writeup submission — you'll want to write up the LiteRT-LM/Gemma integration section since you own that code, even though Pablo owns the doc file itself).
