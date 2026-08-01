import SwiftUI
import UIKit

/// Full push-to-talk pipeline: tap to listen (snapshot + STT) → tap to send
/// (stub RAG/LLM → SafetyFilter → TTS). Real LiteRT-LM + VectorRAG swap in at Checkpoint 4
/// via `runInferencePipeline` — keep that signature stable for Sachin.
struct ContentView: View {
    @StateObject private var speechManager = SpeechManager()
    @StateObject private var ttsManager = TTSManager()
    @StateObject private var cameraManager = CameraManager()

    @State private var appState: AppState = .idle
    @State private var citation: String?
    @State private var checklistText: String = ""

    /// Spoken / displayed copy for the three hard-fail paths.
    private enum SpokenError {
        static let emptyTranscript = "I didn't catch that. Try again."
        static let speechUnavailable = "On-device speech isn't available. I won't use the network."
        static let modelFailure = "I couldn't run the local model. Try again."
    }

    /// Checkpoint 4 replaces this stub with LLMInferenceManager + VectorRAGManager.
    /// Signature must not change without updating both call sites.
    /// Return an empty `checklistText` to signal model/pipeline failure.
    var runInferencePipeline: (String, UIImage?) async -> (citation: String?, checklistText: String) = { transcript, _ in
        let lowered = transcript.lowercased()
        if lowered.contains("force model failure") {
            return (nil, "")
        }
        if lowered.contains("weather") || lowered.contains("score of the game") {
            return (
                nil,
                "I don't have a protocol covering that in my offline library. I can't answer it from the field manuals I carry."
            )
        }
        return (
            "[Source: STUB — swap for retrieved protocol citation]",
            """
            Displaying retrieved protocol checklist for \"\(transcript)\".
            1. Scene safety and standard precautions.
            2. Primary assessment (airway, breathing, circulation).
            3. Follow the cited field-manual steps within your training and scope.
            """
        )
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

            if !speechManager.transcript.isEmpty, appState == .listening {
                Text(speechManager.transcript)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 4)
            }

            Spacer()

            EmergencyButtonView(
                state: appState,
                onPressDown: handleStartListening,
                onPressUp: handleStopAndProcess
            )

            Spacer()
        }
        .padding()
        .task {
            await cameraManager.prewarm()
            #if DEBUG
            // Task D2 Step 2: confirm fail-closed path before the real .litertlm lands.
            let llm = LLMInferenceManager()
            await llm.initialize()
            print(
                "LLMInferenceManager.initialize() → isReady=\(llm.isReady) error=\(String(describing: llm.initializationError))"
            )
            #endif
        }
        .onDisappear {
            cameraManager.shutdown()
        }
        .onChange(of: speechManager.error) { _, newError in
            guard appState == .listening, let newError else { return }
            failSpeech(newError)
        }
    }

    private var isErrorState: Bool {
        if case .error = appState { return true }
        return false
    }

    private var displayText: String {
        if case .error(let message) = appState { return message }
        if checklistText.isEmpty {
            return "Tap the button, ask a wilderness first-aid question, then tap again to send."
        }
        return checklistText
    }

    private func handleStartListening() {
        guard appState == .idle || isErrorState else { return }

        ttsManager.stop()
        citation = nil
        checklistText = ""
        appState = .listening

        Task {
            await cameraManager.captureSnapshot()
        }
        speechManager.startListening()
    }

    private func handleStopAndProcess() {
        guard appState == .listening else { return }
        speechManager.stopListening()

        Task {
            appState = .processing

            // Brief pause so partial results can settle after stop.
            try? await Task.sleep(nanoseconds: 300_000_000)

            if let speechError = speechManager.error {
                await presentError(spokenMessage(for: speechError))
                return
            }

            let transcript = speechManager.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !transcript.isEmpty else {
                await presentError(SpokenError.emptyTranscript)
                return
            }

            let (resultCitation, resultText) = await runInferencePipeline(
                transcript,
                cameraManager.latestSnapshot
            )

            let trimmedResult = resultText.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedResult.isEmpty else {
                await presentError(SpokenError.modelFailure)
                return
            }

            let filtered = SafetyFilter.sanitize(trimmedResult)
            citation = resultCitation
            checklistText = filtered.text
            appState = .speaking
            ttsManager.speak(filtered.text)

            while ttsManager.isSpeaking {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            if case .speaking = appState {
                appState = .idle
            }
        }
    }

    private func failSpeech(_ error: SpeechManager.SpeechError) {
        speechManager.stopListening()
        Task {
            await presentError(spokenMessage(for: error))
        }
    }

    private func spokenMessage(for error: SpeechManager.SpeechError) -> String {
        switch error {
        case .onDeviceUnavailable, .permissionDenied, .recognizerUnavailable:
            return SpokenError.speechUnavailable
        case .audioEngineFailure, .recognitionFailure:
            return SpokenError.emptyTranscript
        }
    }

    @MainActor
    private func presentError(_ message: String) async {
        let filtered = SafetyFilter.sanitize(message)
        citation = nil
        checklistText = filtered.text
        appState = .error(filtered.text)
        ttsManager.speak(filtered.text)

        while ttsManager.isSpeaking {
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
    }
}

#Preview {
    ContentView()
}
