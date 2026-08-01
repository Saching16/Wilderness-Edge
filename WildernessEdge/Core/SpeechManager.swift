import AVFoundation
import Combine
import Foundation
import Speech

/// On-device speech recognition wrapper. Never falls back to server-based recognition.
@MainActor
final class SpeechManager: ObservableObject {
    enum SpeechError: LocalizedError, Equatable {
        case permissionDenied
        case recognizerUnavailable
        case onDeviceUnavailable
        case audioEngineFailure(String)
        case recognitionFailure(String)

        var errorDescription: String? {
            switch self {
            case .permissionDenied:
                return "Microphone or speech recognition permission was denied."
            case .recognizerUnavailable:
                return "Speech recognizer is unavailable for the current locale."
            case .onDeviceUnavailable:
                return "On-device speech recognition is unavailable. Wilderness Edge will not use network recognition."
            case .audioEngineFailure(let message):
                return "Audio engine failed: \(message)"
            case .recognitionFailure(let message):
                return "Speech recognition failed: \(message)"
            }
        }
    }

    @Published private(set) var transcript: String = ""
    @Published private(set) var isListening: Bool = false
    @Published private(set) var error: SpeechError?

    /// Injected for tests — production uses the real locale check.
    var onDeviceRecognitionProbe: () -> Bool = {
        SFSpeechRecognizer().map(\.supportsOnDeviceRecognition) ?? false
    }

    private let audioEngine = AVAudioEngine()
    private var recognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

    func startListening() {
        error = nil

        Task {
            let permitted = await requestPermissions()
            guard permitted else {
                error = .permissionDenied
                return
            }
            beginRecognition()
        }
    }

    func stopListening() {
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest?.endAudio()
        recognitionRequest = nil

        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        isListening = false
    }

    /// Test/debug helper: force the fail-closed on-device unavailable path without starting recognition.
    func simulateOnDeviceUnavailable() {
        stopListening()
        error = .onDeviceUnavailable
    }

    private func beginRecognition() {
        stopListening()
        error = nil
        transcript = ""

        let speechRecognizer = SFSpeechRecognizer()
        recognizer = speechRecognizer

        guard let speechRecognizer, speechRecognizer.isAvailable else {
            error = .recognizerUnavailable
            return
        }

        // Fail closed: never start recognition if on-device is unsupported.
        guard onDeviceRecognitionProbe() else {
            error = .onDeviceUnavailable
            return
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = true
        recognitionRequest = request

        do {
            let inputNode = audioEngine.inputNode
            let format = inputNode.outputFormat(forBus: 0)
            inputNode.removeTap(onBus: 0)
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
                self?.recognitionRequest?.append(buffer)
            }

            audioEngine.prepare()
            try audioEngine.start()
        } catch {
            self.error = .audioEngineFailure(error.localizedDescription)
            stopListening()
            return
        }

        isListening = true
        recognitionTask = speechRecognizer.recognitionTask(with: request) { [weak self] result, taskError in
            Task { @MainActor in
                guard let self else { return }
                if let taskError {
                    // Cancellation after stopListening is expected.
                    if (taskError as NSError).code == 216 || (taskError as NSError).code == 301 {
                        return
                    }
                    self.error = .recognitionFailure(taskError.localizedDescription)
                    self.stopListening()
                    return
                }

                if let result {
                    self.transcript = result.bestTranscription.formattedString
                    if result.isFinal {
                        self.stopListening()
                    }
                }
            }
        }
    }

    private func requestPermissions() async -> Bool {
        let micStatus = AVAudioApplication.shared.recordPermission
        let micGranted: Bool
        switch micStatus {
        case .granted:
            micGranted = true
        case .denied:
            micGranted = false
        case .undetermined:
            micGranted = await withCheckedContinuation { continuation in
                AVAudioApplication.requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            }
        @unknown default:
            micGranted = false
        }

        guard micGranted else { return false }

        let speechStatus = SFSpeechRecognizer.authorizationStatus()
        switch speechStatus {
        case .authorized:
            return true
        case .denied, .restricted:
            return false
        case .notDetermined:
            return await withCheckedContinuation { continuation in
                SFSpeechRecognizer.requestAuthorization { status in
                    continuation.resume(returning: status == .authorized)
                }
            }
        @unknown default:
            return false
        }
    }
}
