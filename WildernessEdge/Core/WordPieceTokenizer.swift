import Foundation

/// BERT-style WordPiece tokenizer feeding the bundled CoreML query embedder.
/// CoreML cannot accept strings, so tokenization must happen here in Swift.
///
/// This mirrors reference BERT's `BasicTokenizer` + `WordpieceTokenizer` pair, because the
/// CoreML embedder was traced from the HuggingFace tokenizer in
/// `OffLineTools/export_embedder_coreml.py`. Divergence between the two shows up as
/// silently degraded retrieval rather than a crash — the query lands in a slightly wrong
/// spot in the embedding space and the wrong protocol chunk comes back — so the behaviour
/// is pinned by `WordPieceTokenizerTests` rather than left to inspection.
///
/// See plans/vaibhav.md Task B1.
struct WordPieceTokenizer {
    enum TokenizerError: LocalizedError {
        case vocabLoadFailed(String)
        case invalidSequenceLength(Int)

        var errorDescription: String? {
            switch self {
            case .vocabLoadFailed(let message):
                return "Failed to load tokenizer vocabulary: \(message)"
            case .invalidSequenceLength(let length):
                return "maxSequenceLength must be at least 2 to fit [CLS] and [SEP], got \(length)."
            }
        }
    }

    private let vocab: [String: Int32]
    private let maxSequenceLength: Int
    private let doLowerCase: Bool
    private let stripAccents: Bool
    private let clsTokenId: Int32
    private let sepTokenId: Int32
    private let padTokenId: Int32
    private let unkTokenId: Int32

    private static let continuingSubwordPrefix = "##"
    /// Reference BERT sends anything longer straight to [UNK] rather than attempting the
    /// quadratic longest-match walk.
    private static let maxCharactersPerWord = 200

    init(
        vocabURL: URL,
        maxSequenceLength: Int,
        doLowerCase: Bool,
        stripAccents: Bool = true,
        clsTokenId: Int32,
        sepTokenId: Int32,
        padTokenId: Int32,
        unkTokenId: Int32
    ) throws {
        guard maxSequenceLength >= 2 else {
            throw TokenizerError.invalidSequenceLength(maxSequenceLength)
        }

        let contents: String
        do {
            contents = try String(contentsOf: vocabURL, encoding: .utf8)
        } catch {
            throw TokenizerError.vocabLoadFailed(error.localizedDescription)
        }

        // One token per line, id == line number. A trailing \r survives a CRLF checkout and
        // would corrupt every entry in the table, so it is stripped rather than trusted.
        var lines = contents
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.hasSuffix("\r") ? String($0.dropLast()) : String($0) }
        if lines.last?.isEmpty == true {
            lines.removeLast()  // the empty element produced by a trailing newline
        }

        var table: [String: Int32] = [:]
        table.reserveCapacity(lines.count)
        for (index, token) in lines.enumerated() where !token.isEmpty {
            table[token] = Int32(index)
        }

        self.vocab = table
        self.maxSequenceLength = maxSequenceLength
        self.doLowerCase = doLowerCase
        self.stripAccents = stripAccents
        self.clsTokenId = clsTokenId
        self.sepTokenId = sepTokenId
        self.padTokenId = padTokenId
        self.unkTokenId = unkTokenId
    }

    /// Returns fixed-length `input_ids` and `attention_mask` wrapped in [CLS] … [SEP],
    /// both padded or truncated to `maxSequenceLength`.
    func encode(_ text: String) -> (inputIds: [Int32], attentionMask: [Int32]) {
        var ids: [Int32] = [clsTokenId]
        ids.reserveCapacity(maxSequenceLength)

        // [CLS] and [SEP] each claim a slot, so the content budget is two short of the total.
        let contentBudget = maxSequenceLength - 2
        appendContent: for word in basicTokenize(text) {
            for id in wordpieceTokenize(word) {
                if ids.count - 1 == contentBudget { break appendContent }
                ids.append(id)
            }
        }
        ids.append(sepTokenId)

        let realTokenCount = ids.count
        let paddingCount = maxSequenceLength - realTokenCount
        ids.append(contentsOf: repeatElement(padTokenId, count: paddingCount))

        var mask = [Int32](repeating: 1, count: realTokenCount)
        mask.append(contentsOf: repeatElement(0, count: paddingCount))

        return (ids, mask)
    }

    // MARK: - Basic tokenization

    /// Whitespace and punctuation segmentation, matching BERT's `BasicTokenizer`.
    private func basicTokenize(_ text: String) -> [String] {
        var working = Self.cleaned(text)
        // BERT lowercases first and strips accents from the result, not the other way round.
        if doLowerCase { working = working.lowercased() }
        if stripAccents { working = Self.accentsStripped(working) }

        return working
            .split(whereSeparator: { $0.isWhitespace })
            .flatMap { Self.splitOnPunctuation(String($0)) }
    }

    /// Drops the null, replacement and control characters BERT's `_clean_text` removes, and
    /// folds every other whitespace form to a plain space. Whitespace is checked first
    /// because tab/newline/carriage return are themselves control characters and must
    /// become separators rather than vanish.
    private static func cleaned(_ text: String) -> String {
        var output = String.UnicodeScalarView()
        for scalar in text.unicodeScalars {
            if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                output.append(" ")
            } else if scalar == "\u{0}" || scalar == "\u{FFFD}" {
                continue
            } else if scalar.properties.generalCategory == .control {
                continue
            } else {
                output.append(scalar)
            }
        }
        return String(output)
    }

    /// Canonically decomposes and drops combining marks, matching BERT's `_run_strip_accents`.
    private static func accentsStripped(_ text: String) -> String {
        let scalars = text.decomposedStringWithCanonicalMapping.unicodeScalars
            .filter { $0.properties.generalCategory != .nonspacingMark }
        return String(String.UnicodeScalarView(scalars))
    }

    private static func splitOnPunctuation(_ token: String) -> [String] {
        var pieces: [String] = []
        var current = ""
        for character in token {
            if isPunctuation(character) {
                if !current.isEmpty {
                    pieces.append(current)
                    current = ""
                }
                pieces.append(String(character))
            } else {
                current.append(character)
            }
        }
        if !current.isEmpty { pieces.append(current) }
        return pieces
    }

    /// BERT counts the ASCII symbol ranges as punctuation on top of the Unicode P*
    /// categories, so `+` and `$` split off into their own tokens while `°` and `©` stay
    /// attached to the word they follow.
    private static func isPunctuation(_ character: Character) -> Bool {
        guard character.unicodeScalars.count == 1, let scalar = character.unicodeScalars.first else {
            return false
        }
        let value = scalar.value
        if (33...47).contains(value) || (58...64).contains(value)
            || (91...96).contains(value) || (123...126).contains(value) {
            return true
        }
        switch scalar.properties.generalCategory {
        case .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
             .initialPunctuation, .finalPunctuation, .otherPunctuation:
            return true
        default:
            return false
        }
    }

    // MARK: - WordPiece

    /// Greedy longest-match-first over the vocabulary. A word containing any unmatchable
    /// piece collapses to a single [UNK] rather than a partial decomposition, as in
    /// reference BERT.
    private func wordpieceTokenize(_ token: String) -> [Int32] {
        let characters = Array(token)
        guard !characters.isEmpty else { return [] }
        guard characters.count <= Self.maxCharactersPerWord else { return [unkTokenId] }

        var subTokens: [Int32] = []
        var start = 0

        while start < characters.count {
            var end = characters.count
            var matched: Int32?

            while end > start {
                let piece = String(characters[start..<end])
                let candidate = start == 0 ? piece : Self.continuingSubwordPrefix + piece
                if let id = vocab[candidate] {
                    matched = id
                    break
                }
                end -= 1
            }

            guard let id = matched else { return [unkTokenId] }
            subTokens.append(id)
            start = end
        }

        return subTokens
    }
}
