import CoreML
import Foundation

/// Wraps the bundled CoreML query embedder. Pooling and L2 normalization are already
/// folded into the model graph, so callers get a directly-comparable 384-dim vector.
///
/// See plans/vaibhav.md Task B2 for the full implementation spec.
final class TextEmbeddingManager {
    enum EmbeddingError: LocalizedError {
        case modelLoadFailed(String)
        case tokenizerAssetsMissing
        case predictionFailed(String)
        case unexpectedOutputShape

        var errorDescription: String? {
            switch self {
            case .modelLoadFailed(let message): return "Failed to load query embedder: \(message)"
            case .tokenizerAssetsMissing: return "Tokenizer vocabulary/config not found in bundle."
            case .predictionFailed(let message): return "Embedding prediction failed: \(message)"
            case .unexpectedOutputShape: return "Embedder returned an unexpected output shape."
            }
        }
    }

    // TODO(vaibhav): implement per plans/vaibhav.md Task B2.
}
