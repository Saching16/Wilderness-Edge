import SwiftUI

/// Phase 1 scaffold: wires Speech / TTS / Camera / SafetyFilter for infrastructure verification.
/// Full push-to-talk orchestration arrives in Phase 4.
struct ContentView: View {
    @StateObject private var speechManager = SpeechManager()
    @StateObject private var ttsManager = TTSManager()
    @StateObject private var cameraManager = CameraManager()

    @State private var lastSpokenPreview = ""
    @State private var safetyNote = ""

    var body: some View {
        VStack(spacing: 24) {
            Text("Wilderness Edge")
                .font(.largeTitle.bold())

            statusBlock

            ScrollView {
                Text(speechManager.transcript.isEmpty ? "Transcript will appear here…" : speechManager.transcript)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            .frame(maxHeight: 160)

            if !safetyNote.isEmpty {
                Text(safetyNote)
                    .font(.footnote)
                    .foregroundStyle(.orange)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack(spacing: 16) {
                Button(speechManager.isListening ? "Stop Listening" : "Start Listening") {
                    toggleListening()
                }
                .buttonStyle(.borderedProminent)

                Button("Speak Filtered") {
                    speakFilteredDemo()
                }
                .buttonStyle(.bordered)
                .disabled(speechManager.transcript.isEmpty && lastSpokenPreview.isEmpty)

                Button("Capture Snapshot") {
                    Task { await cameraManager.captureSnapshot() }
                }
                .buttonStyle(.bordered)
            }

            if let image = cameraManager.latestSnapshot {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 180)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }

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

    @ViewBuilder
    private var statusBlock: some View {
        VStack(alignment: .leading, spacing: 8) {
            labeled("Speech", speechStatusText)
            labeled("TTS", ttsManager.isSpeaking ? "Speaking…" : "Idle")
            labeled("Camera", cameraStatusText)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.tertiarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var speechStatusText: String {
        if let error = speechManager.error {
            return error.localizedDescription
        }
        if speechManager.isListening {
            return "Listening (on-device)"
        }
        return "Ready"
    }

    private var cameraStatusText: String {
        if let error = cameraManager.error {
            return error.localizedDescription
        }
        if cameraManager.isSessionRunning {
            return "Session warm"
        }
        return "Idle"
    }

    private func labeled(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title).fontWeight(.semibold)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
        }
    }

    private func toggleListening() {
        if speechManager.isListening {
            speechManager.stopListening()
        } else {
            speechManager.startListening()
        }
    }

    private func speakFilteredDemo() {
        let raw = speechManager.transcript.isEmpty
            ? "The diagnosis is a fracture. Take 400mg ibuprofen."
            : speechManager.transcript
        let filtered = SafetyFilter.sanitize(raw)
        safetyNote = filtered.wasModified
            ? "SafetyFilter intercepted diagnostic/prescriptive language."
            : "SafetyFilter passed checklist text unmodified."
        lastSpokenPreview = filtered.text
        ttsManager.speak(filtered.text)
    }
}

#Preview {
    ContentView()
}
