import AVFoundation
import SwiftUI

@main
struct WildernessEdgeApp: App {
    init() {
        Self.configureAudioSession()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }

    /// Shared session setup so mic capture and TTS playback share a voice-friendly category.
    private static func configureAudioSession() {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(
                .playAndRecord,
                mode: .spokenAudio,
                options: [.defaultToSpeaker, .allowBluetooth, .duckOthers]
            )
            try session.setActive(true, options: [])
        } catch {
            // Fail closed for audio routing only — UI will surface mic/TTS errors when used.
            assertionFailure("AVAudioSession configuration failed: \(error)")
        }
    }
}
