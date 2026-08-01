import SwiftUI
import UIKit

/// Full push-to-talk pipeline: tap to listen (snapshot + STT) → tap to send
/// (Embed → RAG → Gemma via LiteRT-LM → SafetyFilter → TTS).
struct ContentView: View {
    @StateObject private var speechManager = SpeechManager()
    @StateObject private var ttsManager = TTSManager()
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var llmManager = LLMInferenceManager()

    @State private var appState: AppState = .idle
    @State private var citation: String?
    @State private var checklistText: String = ""

    private let embedder = try? TextEmbeddingManager()
    private let ragManager: VectorRAGManager? = {
        guard let path = Bundle.main.path(forResource: "protocols", ofType: "db") else { return nil }
        return try? VectorRAGManager(databasePath: path)
    }()

    /// Spoken / displayed copy for the three hard-fail paths.
    private enum SpokenError {
        static let emptyTranscript = "I didn't catch that. Try again."
        static let speechUnavailable = "On-device speech isn't available. I won't use the network."
        static let modelFailure = "I couldn't run the local model. Try again."
    }

    /// Real RAG + LiteRT-LM pipeline. Signature matches Daniel's Checkpoint 4 contract:
    /// `(String, UIImage?) async -> (citation: String?, checklistText: String)`.
    /// Empty `checklistText` signals model/pipeline failure to the state machine.
    @State private var runInferencePipeline: (String, UIImage?) async -> (citation: String?, checklistText: String) = { _, _ in
        (nil, "")
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
            await llmManager.initialize()

            if let initError = llmManager.initializationError {
                await presentError(initError.localizedDescription)
                return
            }

            let embedder = self.embedder
            let ragManager = self.ragManager
            let llmManager = self.llmManager

            runInferencePipeline = { transcript, image in
                guard let embedder, let ragManager else {
                    return (nil, "Retrieval system unavailable.")
                }
                guard llmManager.isReady else {
                    return (nil, "")
                }
                do {
                    let queryEmbedding = try embedder.embed(transcript)
                    let ragResult = ragManager.search(
                        embedding: queryEmbedding,
                        topK: 3,
                        threshold: 0.35
                    )
                    let responseText = try await llmManager.generate(
                        transcript: transcript,
                        ragResult: ragResult,
                        image: image
                    )
                    let citation: String? = {
                        if case .match(let chunks) = ragResult {
                            return chunks.first?.citation
                        }
                        return nil
                    }()
                    return (citation, responseText)
                } catch {
                    return (nil, "")
                }
            }
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
