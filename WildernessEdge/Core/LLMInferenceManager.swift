import Foundation
import UIKit

/// Wraps LiteRT-LM's Engine/Conversation API for local multimodal Gemma 4 E4B inference.
/// Never falls back to a networked model on any failure.
///
/// See plans/sachin.md Task D2 for the full implementation spec. Add the LiteRT-LM SPM
/// package (Task D1) before uncommenting `import LiteRTLM` and wiring the real API.
@MainActor
final class LLMInferenceManager: ObservableObject {
    enum LLMError: LocalizedError {
        case modelAssetMissing
        case initializationFailed(String)
        case generationFailed(String)

        var errorDescription: String? {
            switch self {
            case .modelAssetMissing:
                return "The Gemma 4 E4B model bundle is missing from the app package."
            case .initializationFailed(let message):
                return "Model failed to initialize: \(message)"
            case .generationFailed(let message):
                return "Generation failed: \(message)"
            }
        }
    }

    @Published private(set) var isReady = false
    @Published private(set) var initializationError: LLMError?

    // TODO(sachin): implement per plans/sachin.md Task D2.
}
