import PhotosUI
import SwiftUI
import UIKit

/// Primary container and state coordination view.
///
/// Full pipeline: voice or typed question (plus an optional photo) → Embed → RAG → Gemma
/// via LiteRT-LM → SafetyFilter → TTS.
///
/// Three ways in, all converging on one `deliver(transcript:)` call so they cannot drift
/// apart. Every path, including error copy, goes through `SafetyFilter.sanitize` before
/// anything is displayed or spoken; there is deliberately no branch reaching `TTSManager`
/// without it.
struct ContentView: View {
    @StateObject private var speechManager = SpeechManager()
    @StateObject private var ttsManager = TTSManager()
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var llmManager = LLMInferenceManager()

    private let embedder = try? TextEmbeddingManager()
    private let ragManager: VectorRAGManager? = {
        guard let path = Bundle.main.path(forResource: "protocols", ofType: "db") else { return nil }
        return try? VectorRAGManager(databasePath: path)
    }()

    @State private var appState: AppState = .idle
    @State private var citation: String?
    @State private var checklistText: String = ""

    @State private var inputMode: QueryInputMode = .voice
    @State private var typedQuery: String = ""
    /// The frame that will be sent with the next question. Either an automatic snapshot
    /// taken when listening starts, or one the responder attached deliberately — an
    /// explicit choice is never overwritten by an automatic capture.
    @State private var attachedImage: UIImage?
    @State private var pickedItem: PhotosPickerItem?

    /// Spoken / displayed copy for the three hard-fail paths.
    private enum SpokenError {
        static let emptyTranscript = "I didn't catch that. Try again."
        static let speechUnavailable = "On-device speech isn't available. I won't use the network."
        static let modelFailure = "I couldn't run the local model. Try again."
    }

    /// Real RAG + LiteRT-LM pipeline, assembled in `startUp()` once the model has loaded.
    /// Signature is the Checkpoint 4 contract: `(String, UIImage?) async -> (citation:,
    /// checklistText:)`. An empty `checklistText` signals model/pipeline failure to the
    /// state machine.
    @State private var runInferencePipeline: (String, UIImage?) async -> (citation: String?, checklistText: String) = { _, _ in
        (nil, "")
    }

    // MARK: - Body

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()

            VStack(spacing: Theme.Metric.stackSpacing) {
                header
                answerArea
                Spacer(minLength: 8)
                inputDock
            }
            .padding(.horizontal, Theme.Metric.gutter)
            .padding(.bottom, 12)
        }
        // No forced colour scheme: the palette in Theme defines both appearances
        // deliberately, so the responder's own system setting wins.
        .task { await startUp() }
        .onDisappear { cameraManager.shutdown() }
        .onChange(of: speechManager.error) { _, newError in
            guard appState == .listening, let newError else { return }
            failSpeech(newError)
        }
        .onChange(of: pickedItem) { _, item in
            guard let item else { return }
            Task { @MainActor in
                if let data = try? await item.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    attachedImage = image
                }
                pickedItem = nil
            }
        }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Wilderness Edge")
                    .font(.title2.bold())
                    .foregroundStyle(Theme.textPrimary)
                Text("Offline protocol assistant")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
            Spacer()
            offlineBadge
        }
        .padding(.top, 4)
    }

    /// States the air-gap plainly. The whole product claim is that nothing leaves the
    /// device, so it should be visible rather than buried in a writeup.
    private var offlineBadge: some View {
        HStack(spacing: 6) {
            Image(systemName: "wifi.slash")
                .font(.caption2.weight(.bold))
            Text("On-device")
                .font(.caption2.weight(.semibold))
        }
        .foregroundStyle(Theme.speaking)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Theme.speaking.opacity(0.14))
        .clipShape(Capsule())
        .accessibilityLabel("Running entirely on this device")
    }

    private var answerArea: some View {
        VStack(spacing: 10) {
            SubtitleCardView(
                citation: citation,
                checklistText: displayText,
                isError: isErrorState
            )

            if appState == .listening {
                liveTranscript
            }
        }
    }

    private var liveTranscript: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "waveform")
                .font(.caption)
                .foregroundStyle(Theme.listening)
            Text(speechManager.transcript.isEmpty ? "Listening…" : speechManager.transcript)
                .font(.callout)
                .foregroundStyle(speechManager.transcript.isEmpty ? Theme.textSecondary : Theme.textPrimary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .animation(nil, value: speechManager.transcript)
        }
        .padding(12)
        .fieldPanel(raised: true)
        .transition(.opacity)
        .accessibilityLabel("Live transcript")
        .accessibilityValue(speechManager.transcript)
    }

    private var inputDock: some View {
        VStack(spacing: 12) {
            SnapshotStrip(
                image: attachedImage,
                isEnabled: acceptsInput,
                onCapture: captureSnapshot,
                onClear: { attachedImage = nil },
                pickedItem: $pickedItem
            )

            InputModeSwitch(mode: $inputMode, isEnabled: acceptsInput)

            switch inputMode {
            case .voice:
                EmergencyButtonView(
                    state: appState,
                    onPressDown: handleStartListening,
                    onPressUp: handleStopAndProcess
                )
                .padding(.top, 4)
            case .text:
                TextQueryField(
                    text: $typedQuery,
                    isEnabled: acceptsInput,
                    onSubmit: submitTypedQuery
                )
            }

            if appState == .speaking {
                Button {
                    ttsManager.stop()
                } label: {
                    Label("Stop reading", systemImage: "stop.fill")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .frame(height: Theme.Metric.minimumTouchTarget - 8)
                }
                .buttonStyle(.plain)
                .foregroundStyle(Theme.textPrimary)
                .background(Theme.surfaceRaised)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Metric.cornerRadiusSmall, style: .continuous))
            }
        }
    }

    // MARK: - Derived state

    private var isErrorState: Bool {
        if case .error = appState { return true }
        return false
    }

    /// Typing and attaching are allowed while idle or recovering from an error, but not
    /// while a query is already in flight.
    private var acceptsInput: Bool {
        switch appState {
        case .idle, .error, .listening: return true
        case .processing, .speaking: return false
        }
    }

    private var displayText: String {
        if case .error(let message) = appState { return message }
        if checklistText.isEmpty {
            return inputMode == .voice
                ? "Tap the button, ask a wilderness first-aid question, then tap again to send."
                : "Type your question, or attach a photo of the injury, then send."
        }
        return checklistText
    }

    // MARK: - Start-up

    /// Warms the camera, loads the model, then assembles the real pipeline. A model that
    /// fails to load surfaces a blocking error rather than letting the UI go on accepting
    /// questions it cannot answer.
    @MainActor
    private func startUp() async {
        await cameraManager.prewarm()
        await llmManager.initialize()

        if let initError = llmManager.initializationError {
            await presentError(initError.localizedDescription)
            return
        }

        // Captured locally so the escaping closure does not retain the view.
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

    // MARK: - Actions

    private func captureSnapshot() {
        Task { @MainActor in
            await cameraManager.captureSnapshot()
            if let snapshot = cameraManager.latestSnapshot {
                attachedImage = snapshot
            }
        }
    }

    private func handleStartListening() {
        guard appState == .idle || isErrorState else { return }

        ttsManager.stop()
        citation = nil
        checklistText = ""
        appState = .listening

        // Automatic capture only when nothing was attached deliberately, so an explicit
        // choice is never silently replaced.
        if attachedImage == nil {
            Task { @MainActor in
                await cameraManager.captureSnapshot()
                if attachedImage == nil { attachedImage = cameraManager.latestSnapshot }
            }
        }
        speechManager.startListening()
    }

    private func handleStopAndProcess() {
        guard appState == .listening else { return }

        Task { @MainActor in
            appState = .processing

            // End audio and wait for a final (or last partial) transcript — do not cancel.
            let transcript = await speechManager.finishListening()
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if let speechError = speechManager.error {
                await presentError(spokenMessage(for: speechError))
                return
            }

            guard !transcript.isEmpty else {
                await presentError(SpokenError.emptyTranscript)
                return
            }

            await deliver(transcript: transcript)
        }
    }

    private func submitTypedQuery() {
        let query = typedQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty, acceptsInput else { return }

        ttsManager.stop()
        speechManager.stopListening()
        citation = nil
        checklistText = ""
        typedQuery = ""

        Task { @MainActor in
            appState = .processing
            await deliver(transcript: query)
        }
    }

    /// The single path from a question to spoken output. Voice and text both land here so
    /// neither can bypass the safety filter or diverge in behaviour.
    @MainActor
    private func deliver(transcript: String) async {
        let (resultCitation, resultText) = await runInferencePipeline(transcript, attachedImage)

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

    private func failSpeech(_ error: SpeechManager.SpeechError) {
        speechManager.stopListening()
        Task { @MainActor in
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
