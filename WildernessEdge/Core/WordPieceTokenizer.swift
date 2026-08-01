import Foundation

/// BERT-style WordPiece tokenizer feeding the bundled CoreML query embedder.
/// CoreML cannot accept strings, so tokenization must happen here in Swift.
///
/// See plans/vaibhav.md Task B1 for the full implementation spec.
struct WordPieceTokenizer {
    enum TokenizerError: LocalizedError {
        case vocabLoadFailed(String)

        var errorDescription: String? {
            switch self {
            case .vocabLoadFailed(let message):
                return "Failed to load tokenizer vocabulary: \(message)"
            }
        }
    }

    // TODO(vaibhav): implement per plans/vaibhav.md Task B1.
}
