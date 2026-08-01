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
    private var finalContinuation: CheckedContinuation<Void, Never>?
    private var didReceiveFinalResult = false

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

    /// Graceful stop used when the user taps Send. Ends audio and waits briefly for a
    /// final result so partial transcripts are not discarded by `cancel()`.
    @discardableResult
    func finishListening() async -> String {
        guard isListening || recognitionRequest != nil else {
            return transcript
        }

        recognitionRequest?.endAudio()
        recognitionRequest = nil
        teardownAudioEngine()

        if !didReceiveFinalResult, recognitionTask != nil {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                self.finalContinuation = continuation
                self.scheduleFinalWaitTimeout()
            }
        }

        recognitionTask = nil
        isListening = false
        return transcript
    }

    /// Timeout companion for `finishListening()` — kept as a separate task so the
    /// continuation is only resumed once (final result or timeout wins).
    private func scheduleFinalWaitTimeout() {
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            resumeFinalWait()
        }
    }

    /// Hard cancel for errors / restart. Prefer `finishListening()` for user-initiated stop.
    func stopListening() {
        resumeFinalWait()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        teardownAudioEngine()
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
        didReceiveFinalResult = false

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
            try activateAudioSession()

            let inputNode = audioEngine.inputNode
            guard let format = usableInputFormat(for: inputNode) else {
                error = .audioEngineFailure("Microphone format is unavailable (sample rate 0).")
                stopListening()
                return
            }

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
                    let nsError = taskError as NSError
                    if nsError.domain == "kAFAssistantErrorDomain",
                       nsError.code == 216 || nsError.code == 301 {
                        self.resumeFinalWait()
                        return
                    }
                    self.error = .recognitionFailure(taskError.localizedDescription)
                    self.stopListening()
                    return
                }

                if let result {
                    self.transcript = result.bestTranscription.formattedString
                    if result.isFinal {
                        self.didReceiveFinalResult = true
                        self.isListening = false
                        self.resumeFinalWait()
                    }
                }
            }
        }
    }

    private func resumeFinalWait() {
        finalContinuation?.resume()
        finalContinuation = nil
    }

    private func teardownAudioEngine() {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
    }

    private func activateAudioSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(
            .playAndRecord,
            mode: .spokenAudio,
            options: [.defaultToSpeaker, .allowBluetooth, .duckOthers]
        )
        try session.setActive(true, options: [])
    }

    /// Simulator / cold-start mic nodes can report a 0 Hz format; fall back to the session rate.
    private func usableInputFormat(for inputNode: AVAudioInputNode) -> AVAudioFormat? {
        let hardwareFormat = inputNode.outputFormat(forBus: 0)
        if hardwareFormat.sampleRate > 0, hardwareFormat.channelCount > 0 {
            return hardwareFormat
        }

        let sampleRate = AVAudioSession.sharedInstance().sampleRate
        guard sampleRate > 0 else { return nil }
        return AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        )
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
