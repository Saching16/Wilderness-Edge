import PhotosUI
import SwiftUI
import UIKit

/// Primary container and state coordination view.
///
/// Three ways in — voice, typed text, and an attached photo — all converging on one
/// pipeline call so they cannot drift apart. In particular every path, including error
/// copy, goes through `SafetyFilter.sanitize` before anything is displayed or spoken;
/// there is deliberately no branch that reaches `TTSManager` without it.
///
/// `runInferencePipeline` keeps its exact signature from Checkpoint 4 — Sachin swaps the
/// stub for `LLMInferenceManager` + `VectorRAGManager` there and nothing else moves.
struct ContentView: View {
    @StateObject private var speechManager = SpeechManager()
    @StateObject private var ttsManager = TTSManager()
    @StateObject private var cameraManager = CameraManager()

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
        .task { await cameraManager.prewarm() }
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
        speechManager.stopListening()

        Task { @MainActor in
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

            await deliver(transcript: transcript)
        }
    }

    private func submitTypedQuery() {
        let query = typedQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty, acceptsInput else { return }

        ttsManager.stop()
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
