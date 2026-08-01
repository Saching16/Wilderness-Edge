# Wilderness Edge — Hackathon Sprint Plan: Daniel (Track C — Native I/O, SwiftUI Shell & Safety Enforcement)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> This is your individual track from the full team plan at `plans/hackathon-sprint.md`. Read that file if you want the other 3 tracks' context — Pablo (offline assets), Vaibhav (RAG engine), and Sachin (LiteRT-LM integration) are each working their own file in parallel from hour zero.

**Goal:** Verify the existing native managers on the physical iPhone 16 Plus, build the push-to-talk button and citation card views, and wire the full state machine in `ContentView` — initially against a stubbed inference pipeline that Sachin will swap for the real thing at Checkpoint 4.

**Tech Stack:** Swift 5.9 / SwiftUI (iOS 17+), AVFoundation, Speech framework.

## Global Constraints (apply to your track too)

- All LLM text output must pass through `SafetyFilter.sanitize(_:)` before display or TTS — never bypass it. Your `ContentView` wiring is the enforcement point for this rule.
- Never fall back to a networked alternative on any speech/inference failure — fail closed with a visible error state instead. This is exactly what your `AppState.error` case and its transitions exist for.
- "No confident RAG match" is a normal, honestly-spoken result, not an error — don't route it into your error state.
- No hard memory-footprint gate this sprint — don't be reckless on the iPhone 16 Plus, but don't spend time optimizing either.

## Who's waiting on you / who you're waiting on

- **Sachin** needs your `ContentView.runInferencePipeline` closure signature (`(String, UIImage?) async -> (citation: String?, checklistText: String)`) locked in before he can wire his `LLMInferenceManager` into it — this is **Checkpoint 4**, target hour 2–2.5. Message him the moment Task C4 is committed.
- **You + Pablo** co-own the Kaggle writeup — draft incrementally, not at the end.

---

## Task C1: Verify existing native managers on the physical device

**Files:** none (manual verification of existing `SpeechManager.swift`, `TTSManager.swift`, `CameraManager.swift` — these already exist from the first commit, you're validating them, not writing them)

- [ ] **Step 1: Build and sideload onto the iPhone 16 Plus**

```bash
xcodegen generate
open WildernessEdge.xcodeproj
```

In Xcode: select the physical device as the run destination, set a development team under Signing & Capabilities (Developer Mode must already be enabled on the phone and the Mac trusted), and Run.

- [ ] **Step 2: Verify on-device speech recognition with the network off**

Enable Airplane Mode on the phone. Tap "Start Listening" in the current `ContentView` scaffold and speak. Expected: `speechManager.transcript` updates live in the UI. If instead the error state shows `.onDeviceUnavailable`, check Settings → General → Keyboard → [your language] → Enable Dictation, since on-device speech models must be downloaded per-locale.

- [ ] **Step 3: Verify TTS audibility**

Tap "Speak Filtered" with an empty transcript (triggers the built-in bait-phrase demo string in `ContentView.speakFilteredDemo()`). Expected: audible speech plays, and `safetyNote` shows "SafetyFilter intercepted diagnostic/prescriptive language."

- [ ] **Step 4: Verify camera pre-warm and snapshot timing**

Confirm the camera permission prompt appears on first launch, then tap "Capture Snapshot" and confirm a near-instant preview image appears (the session should already be pre-warmed via `.task { await cameraManager.prewarm() }`).

- [ ] **Step 5: No code changes expected** — if all three pass, this task is verification-only. If something fails, note the specific failure (error state shown, exact steps) before touching `SpeechManager`/`TTSManager`/`CameraManager` code, and fix root cause rather than papering over it.

## Task C2: Build EmergencyButtonView

**Files:**
- Create: `WildernessEdge/Views/EmergencyButtonView.swift`

**Interfaces:**
- Consumes: an `AppState` enum (defined in Task C4) and press/release callbacks.
- Produces: `EmergencyButtonView(state: AppState, onPressDown: () -> Void, onPressUp: () -> Void)`, a SwiftUI `View` consumed by `ContentView` (Task C4).

- [ ] **Step 1: Implement the button**

```swift
// WildernessEdge/Views/EmergencyButtonView.swift
import SwiftUI

/// Large, high-contrast, circular push-to-talk button with per-state visual treatment.
struct EmergencyButtonView: View {
    let state: AppState
    let onPressDown: () -> Void
    let onPressUp: () -> Void

    @GestureState private var isPressing = false

    var body: some View {
        Circle()
            .fill(fillColor)
            .frame(width: 180, height: 180)
            .overlay(
                Text(label)
                    .font(.title2.bold())
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
                    .padding()
            )
            .scaleEffect(isPressing ? 0.95 : 1.0)
            .animation(.easeOut(duration: 0.15), value: isPressing)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .updating($isPressing) { _, pressing, _ in pressing = true }
                    .onChanged { _ in if !isPressing { onPressDown() } }
                    .onEnded { _ in onPressUp() }
            )
            .accessibilityLabel(label)
    }

    private var fillColor: Color {
        switch state {
        case .idle: return .blue
        case .listening: return .red
        case .processing: return .orange
        case .speaking: return .green
        case .error: return .gray
        }
    }

    private var label: String {
        switch state {
        case .idle: return "Hold to Ask"
        case .listening: return "Listening…"
        case .processing: return "Processing…"
        case .speaking: return "Speaking…"
        case .error: return "Error — Tap for Details"
        }
    }
}

#Preview {
    EmergencyButtonView(state: .idle, onPressDown: {}, onPressUp: {})
}
```

- [ ] **Step 2: Verify visually in Xcode Preview**

Open the Preview canvas for `EmergencyButtonView.swift`. Expected: a blue circular button reading "Hold to Ask". Temporarily change `state:` to `.listening`, `.processing`, `.speaking`, `.error("test")` in the `#Preview` to confirm each color/label renders correctly, then revert to `.idle`.

- [ ] **Step 3: Commit**

```bash
git add WildernessEdge/Views/EmergencyButtonView.swift
git commit -m "Add EmergencyButtonView push-to-talk button"
```

## Task C3: Build SubtitleCardView

**Files:**
- Create: `WildernessEdge/Views/SubtitleCardView.swift`

**Interfaces:**
- Produces: `SubtitleCardView(citation: String?, checklistText: String, isError: Bool)`, consumed by `ContentView` (Task C4).

- [ ] **Step 1: Implement the card**

```swift
// WildernessEdge/Views/SubtitleCardView.swift
import SwiftUI

/// High-contrast overlay card displaying the active source citation and spoken checklist text.
struct SubtitleCardView: View {
    let citation: String?
    let checklistText: String
    let isError: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let citation {
                Text(citation)
                    .font(.footnote.bold())
                    .foregroundStyle(isError ? .red : .secondary)
            }
            Text(checklistText)
                .font(.body)
                .foregroundStyle(isError ? .red : .primary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(isError ? Color.red.opacity(0.1) : Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(isError ? Color.red : Color.clear, lineWidth: 2)
        )
    }
}

#Preview {
    SubtitleCardView(
        citation: "[Source: NASEMSO National Model EMS Clinical Guidelines v3.0, Extremity Trauma, p. 128]",
        checklistText: "1. Expose and inspect the injured extremity.\n2. Check distal pulse, motor, and sensory function.",
        isError: false
    )
}
```

- [ ] **Step 2: Verify visually in Xcode Preview**

Confirm the citation renders in bold footnote style above the checklist body text, and that toggling `isError: true` switches to the red-bordered error treatment.

- [ ] **Step 3: Commit**

```bash
git add WildernessEdge/Views/SubtitleCardView.swift
git commit -m "Add SubtitleCardView citation/checklist card"
```

## Task C4: Build the ContentView state machine (stubbed RAG + LLM)

**Files:**
- Modify: `WildernessEdge/Views/ContentView.swift` (replace the Phase-1 scaffold entirely)

**Interfaces:**
- Defines: `enum AppState { case idle, listening, processing, speaking, error(String) }` — consumed by `EmergencyButtonView` (Task C2) and `SubtitleCardView` (Task C3).
- Defines: a closure `runInferencePipeline(transcript: String, snapshot: UIImage?) async -> (citation: String?, checklistText: String)` — **this exact signature is what Sachin swaps at Checkpoint 4. Do not change it without telling him.**

- [ ] **Step 1: Define AppState and rewrite ContentView with a stub pipeline**

```swift
// WildernessEdge/Views/ContentView.swift
import SwiftUI

enum AppState: Equatable {
    case idle
    case listening
    case processing
    case speaking
    case error(String)
}

/// Full push-to-talk pipeline: Button Down (snapshot + record) -> Button Up (STT -> Embed ->
/// RAG -> Gemma via LiteRT-LM -> Safety Filter -> TTS). The RAG/LLM step is stubbed here and
/// wired to the real LLMInferenceManager at Checkpoint 4.
struct ContentView: View {
    @StateObject private var speechManager = SpeechManager()
    @StateObject private var ttsManager = TTSManager()
    @StateObject private var cameraManager = CameraManager()

    @State private var appState: AppState = .idle
    @State private var citation: String?
    @State private var checklistText: String = ""

    /// Checkpoint 4 replaces this stub with a real call into LLMInferenceManager +
    /// VectorRAGManager. Signature must not change without updating both call sites.
    var runInferencePipeline: (String, UIImage?) async -> (citation: String?, checklistText: String) = { transcript, _ in
        (nil, "STUB: would retrieve protocol context for \"\(transcript)\" and query Gemma 4 E4B here.")
    }

    var body: some View {
        VStack(spacing: 24) {
            Text("Wilderness Edge")
                .font(.largeTitle.bold())

            SubtitleCardView(
                citation: citation,
                checklistText: displayText,
                isError: isErrorState
            )

            Spacer()

            EmergencyButtonView(
                state: appState,
                onPressDown: handlePressDown,
                onPressUp: handlePressUp
            )

            Spacer()
        }
        .padding()
        .task {
            await cameraManager.prewarm()
        }
        .onDisappear {
            cameraManager.shutdown()
        }
    }

    private var isErrorState: Bool {
        if case .error = appState { return true }
        return false
    }

    private var displayText: String {
        if case .error(let message) = appState { return message }
        return checklistText.isEmpty ? "Hold the button and ask a question." : checklistText
    }

    private func handlePressDown() {
        guard appState == .idle || isErrorState else { return }
        appState = .listening
        Task { await cameraManager.captureSnapshot() }
        speechManager.startListening()
    }

    private func handlePressUp() {
        guard appState == .listening else { return }
        speechManager.stopListening()

        Task {
            appState = .processing

            // Give the recognizer a brief moment to publish its final transcript after stop.
            try? await Task.sleep(nanoseconds: 300_000_000)

            if let speechError = speechManager.error {
                appState = .error(speechError.localizedDescription)
                return
            }

            let transcript = speechManager.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !transcript.isEmpty else {
                appState = .error("Didn't catch that — hold the button and try again.")
                return
            }

            let (resultCitation, resultText) = await runInferencePipeline(transcript, cameraManager.latestSnapshot)
            let filtered = SafetyFilter.sanitize(resultText)

            citation = resultCitation
            checklistText = filtered.text
            appState = .speaking
            ttsManager.speak(filtered.text)

            // Return to idle once speech finishes (polled since AVSpeechSynthesizerDelegate
            // already publishes isSpeaking on TTSManager).
            while ttsManager.isSpeaking {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            appState = .idle
        }
    }
}

#Preview {
    ContentView()
}
```

- [ ] **Step 2: Manually verify the stub pipeline end-to-end in Simulator**

Run in Simulator (or device). Press and hold the button, say something, release. Expected: state visibly transitions Idle → Listening → Processing → Speaking → Idle, the subtitle card shows the stub text, and TTS speaks it. Force an error by denying microphone permission once and confirm the button turns gray with the error message displayed (TTS itself won't speak the error unless you choose to wire that — displaying it is sufficient for this stub).

- [ ] **Step 3: Commit**

```bash
git add WildernessEdge/Views/ContentView.swift
git commit -m "Add full push-to-talk state machine with stubbed inference pipeline"
```

- [ ] **Step 4: Notify Sachin (Checkpoint 4 dependency satisfied from this side)**

Message Sachin that `ContentView.runInferencePipeline` has the exact signature `(String, UIImage?) async -> (citation: String?, checklistText: String)` he needs to implement against.

## Task C5: Kaggle writeup (co-owned with Pablo)

**Files:**
- Modify: `docs/kaggle-writeup-draft.md` (Pablo creates it in Task A4 — you contribute sections as Sachin's Checkpoint 4 lands)

- [ ] **Step 1: Contribute UI/UX and safety-enforcement sections**

Once the full pipeline is wired (Sachin's Checkpoint 4), add a section covering the push-to-talk UX and how `SafetyFilter` is enforced as a mandatory gate before any text reaches display/TTS — this maps directly to the Privacy & Safety rubric line (10%).

- [ ] **Step 2: Commit your contribution**

```bash
git add docs/kaggle-writeup-draft.md
git commit -m "Add UI/safety sections to Kaggle writeup draft"
```

---

## Final Validation (all 4 team members)

See `plans/hackathon-sprint.md` Task E1 (Airplane Mode demo run-through, twice — you'll likely be the one holding the phone since you own the UI) and Task E2 (Kaggle Writeup submission).
