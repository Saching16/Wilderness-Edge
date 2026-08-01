import AVFoundation
import Combine
import UIKit

/// AVFoundation single-frame snapshot capture. Session is pre-warmed at launch for near-instant captures.
@MainActor
final class CameraManager: NSObject, ObservableObject {
    enum CameraError: LocalizedError, Equatable {
        case permissionDenied
        case cameraUnavailable
        case sessionConfigurationFailed(String)
        case captureFailed(String)

        var errorDescription: String? {
            switch self {
            case .permissionDenied:
                return "Camera permission was denied."
            case .cameraUnavailable:
                return "No rear camera is available on this device."
            case .sessionConfigurationFailed(let message):
                return "Camera session configuration failed: \(message)"
            case .captureFailed(let message):
                return "Snapshot capture failed: \(message)"
            }
        }
    }

    @Published private(set) var latestSnapshot: UIImage?
    @Published private(set) var isSessionRunning: Bool = false
    @Published private(set) var error: CameraError?

    private let session = AVCaptureSession()
    private let photoOutput = AVCapturePhotoOutput()
    private let sessionQueue = DispatchQueue(label: "com.wildernessedge.camera.session")
    private var captureContinuation: CheckedContinuation<UIImage, Error>?

    /// Call at app launch (not on button-press) so the first snapshot is near-instant.
    func prewarm() async {
        error = nil
        let permitted = await requestPermission()
        guard permitted else {
            error = .permissionDenied
            return
        }

        do {
            try await configureSessionIfNeeded()
            await startSession()
        } catch let cameraError as CameraError {
            error = cameraError
        } catch {
            self.error = .sessionConfigurationFailed(error.localizedDescription)
        }
    }

    func captureSnapshot() async {
        error = nil
        guard isSessionRunning else {
            // Attempt a late warm-up if prewarm was skipped / failed earlier.
            await prewarm()
            guard isSessionRunning else {
                error = error ?? .captureFailed("Camera session is not running.")
                return
            }
            return
        }

        do {
            let image = try await takePhoto()
            latestSnapshot = image
        } catch let cameraError as CameraError {
            error = cameraError
        } catch {
            self.error = .captureFailed(error.localizedDescription)
        }
    }

    func shutdown() {
        sessionQueue.async { [session] in
            if session.isRunning {
                session.stopRunning()
            }
        }
        isSessionRunning = false
    }

    private func requestPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }

    private func configureSessionIfNeeded() async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            sessionQueue.async { [weak self] in
                guard let self else {
                    continuation.resume(throwing: CameraError.sessionConfigurationFailed("Camera manager deallocated."))
                    return
                }

                if self.session.outputs.contains(self.photoOutput) {
                    continuation.resume()
                    return
                }

                self.session.beginConfiguration()
                self.session.sessionPreset = .photo

                guard
                    let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                    let input = try? AVCaptureDeviceInput(device: device),
                    self.session.canAddInput(input)
                else {
                    self.session.commitConfiguration()
                    continuation.resume(throwing: CameraError.cameraUnavailable)
                    return
                }

                self.session.addInput(input)

                guard self.session.canAddOutput(self.photoOutput) else {
                    self.session.commitConfiguration()
                    continuation.resume(throwing: CameraError.sessionConfigurationFailed("Unable to add photo output."))
                    return
                }

                self.session.addOutput(self.photoOutput)
                self.session.commitConfiguration()
                continuation.resume()
            }
        }
    }

    private func startSession() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            sessionQueue.async { [session] in
                if !session.isRunning {
                    session.startRunning()
                }
                continuation.resume()
            }
        }
        isSessionRunning = session.isRunning
        if !isSessionRunning {
            error = .sessionConfigurationFailed("Capture session failed to start.")
        }
    }

    private func takePhoto() async throws -> UIImage {
        try await withCheckedThrowingContinuation { continuation in
            // Only one in-flight capture at a time.
            if captureContinuation != nil {
                continuation.resume(throwing: CameraError.captureFailed("A capture is already in progress."))
                return
            }
            captureContinuation = continuation

            let settings = AVCapturePhotoSettings()
            settings.flashMode = .off
            photoOutput.capturePhoto(with: settings, delegate: self)
        }
    }
}

extension CameraManager: AVCapturePhotoCaptureDelegate {
    nonisolated func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        Task { @MainActor in
            guard let continuation = captureContinuation else { return }
            captureContinuation = nil

            if let error {
                continuation.resume(throwing: CameraError.captureFailed(error.localizedDescription))
                return
            }

            guard
                let data = photo.fileDataRepresentation(),
                let image = UIImage(data: data)
            else {
                continuation.resume(throwing: CameraError.captureFailed("Could not decode snapshot image."))
                return
            }

            continuation.resume(returning: image)
        }
    }
}
