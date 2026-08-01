import XCTest

/// Every expected value in this file was generated from HuggingFace's `BertTokenizer`
/// loaded with `fixture-vocab.txt` (`do_lower_case=True, strip_accents=True,
/// max_length=16, padding="max_length", truncation=True`) — the same tokenizer family
/// `OffLineTools/export_embedder_coreml.py` traced the CoreML embedder from. They are
/// ground truth, not hand-derived guesses: if one of these fails, the Swift tokenizer has
/// drifted from the embedding space `protocols.db` was built in.
final class WordPieceTokenizerTests: XCTestCase {
    private static let sequenceLength = 16

    private func makeFixtureTokenizer(maxSequenceLength: Int = sequenceLength) throws -> WordPieceTokenizer {
        let vocabURL = try XCTUnwrap(
            Bundle(for: type(of: self)).url(forResource: "fixture-vocab", withExtension: "txt"),
            "fixture-vocab.txt is not in the test bundle — check the WildernessEdgeTests sources in project.yml."
        )
        return try WordPieceTokenizer(
            vocabURL: vocabURL,
            maxSequenceLength: maxSequenceLength,
            doLowerCase: true,
            clsTokenId: 2,
            sepTokenId: 3,
            padTokenId: 0,
            unkTokenId: 1
        )
    }

    /// Right-pads an expected content sequence to the fixed sequence length.
    private func padded(_ ids: [Int32]) -> [Int32] {
        ids + Array(repeating: 0, count: Self.sequenceLength - ids.count)
    }

    private func assertEncodes(
        _ text: String,
        to expectedIds: [Int32],
        realTokens expectedRealTokens: Int,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let tokenizer = try makeFixtureTokenizer()
        let (ids, mask) = tokenizer.encode(text)

        XCTAssertEqual(ids, padded(expectedIds), "input_ids mismatch", file: file, line: line)
        XCTAssertEqual(mask.count, Self.sequenceLength, "mask length", file: file, line: line)
        XCTAssertEqual(
            mask, padded(Array(repeating: 1, count: expectedRealTokens)),
            "attention_mask mismatch", file: file, line: line
        )
    }

    // MARK: - Core encoding

    func testEncodesKnownWordsWithClsAndSep() throws {
        try assertEncodes("patient has severe bleeding", to: [2, 4, 5, 6, 7, 3], realTokens: 6)
    }

    func testAttentionMaskMatchesRealTokenCount() throws {
        // [CLS] patient has [SEP] = 4 real tokens
        try assertEncodes("patient has", to: [2, 4, 5, 3], realTokens: 4)
    }

    func testUnknownWordMapsToUnkToken() throws {
        try assertEncodes("xyzzyunknownword", to: [2, 1, 3], realTokens: 3)
    }

    func testEmptyStringYieldsClsSepOnly() throws {
        try assertEncodes("", to: [2, 3], realTokens: 2)
    }

    // MARK: - Subword decomposition

    func testKnownSuffixDecomposesIntoContinuingSubword() throws {
        // "bleedings" -> ["bleeding", "##s"]
        try assertEncodes("bleedings", to: [2, 7, 13, 3], realTokens: 4)
    }

    /// Reference BERT emits a single [UNK] for a word it cannot fully decompose, rather
    /// than keeping the prefix it did match.
    func testPartiallyMatchableWordCollapsesToSingleUnk() throws {
        try assertEncodes("the\u{A9}", to: [2, 1, 3], realTokens: 3)
    }

    func testWordLongerThanTwoHundredCharactersBecomesUnk() throws {
        try assertEncodes(String(repeating: "a", count: 250), to: [2, 1, 3], realTokens: 3)
    }

    // MARK: - Normalization

    func testUppercaseIsLowercased() throws {
        try assertEncodes("PATIENT", to: [2, 4, 3], realTokens: 3)
    }

    func testAccentsAreStripped() throws {
        try assertEncodes("caf\u{E9}", to: [2, 16, 3], realTokens: 3)
    }

    func testTabsAndNewlinesSeparateWordsRatherThanVanishing() throws {
        try assertEncodes("patient\thas\nsevere", to: [2, 4, 5, 6, 3], realTokens: 5)
    }

    // MARK: - Punctuation

    func testPunctuationSplitsIntoItsOwnTokens() throws {
        try assertEncodes("patient, has.", to: [2, 4, 14, 5, 15, 3], realTokens: 6)
    }

    /// An em dash is Unicode category Pd, so it splits the word around it. The `©` case in
    /// `testPartiallyMatchableWordCollapsesToSingleUnk` is category So and does not — the
    /// distinction is BERT's, and getting it backwards silently changes retrieval.
    func testUnicodeDashSplitsButSymbolsDoNot() throws {
        try assertEncodes("left\u{2014}thigh", to: [2, 10, 1, 11, 3], realTokens: 5)
    }

    // MARK: - Truncation and padding

    func testTruncationFillsSequenceAndStillTerminatesWithSep() throws {
        let sixteenWords = "patient has severe bleeding from the left thigh "
            + "patient has severe bleeding from the left thigh"
        // 14 content tokens fit between [CLS] and [SEP]; the last two words are dropped.
        try assertEncodes(
            sixteenWords,
            to: [2, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 3],
            realTokens: 16
        )
    }

    // MARK: - Vocabulary loading

    /// The repo has no `.gitattributes`, so a vocab file checked out on Windows can carry
    /// CRLF endings. An unstripped `\r` would corrupt every entry in the table and send
    /// every query to [UNK].
    func testVocabularyWithWindowsLineEndingsLoadsCorrectly() throws {
        let crlfVocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "patient"].joined(separator: "\r\n") + "\r\n"
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("crlf-vocab-\(UUID().uuidString).txt")
        try crlfVocab.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }

        let tokenizer = try WordPieceTokenizer(
            vocabURL: url,
            maxSequenceLength: 8,
            doLowerCase: true,
            clsTokenId: 2,
            sepTokenId: 3,
            padTokenId: 0,
            unkTokenId: 1
        )

        let (ids, _) = tokenizer.encode("patient")
        XCTAssertEqual(ids, [2, 4, 3, 0, 0, 0, 0, 0])
    }

    func testMissingVocabularyThrows() {
        let missing = URL(fileURLWithPath: "/nonexistent/vocab.txt")
        XCTAssertThrowsError(
            try WordPieceTokenizer(
                vocabURL: missing,
                maxSequenceLength: 16,
                doLowerCase: true,
                clsTokenId: 2,
                sepTokenId: 3,
                padTokenId: 0,
                unkTokenId: 1
            )
        )
    }

    func testSequenceLengthTooShortForSpecialTokensThrows() throws {
        let vocabURL = try XCTUnwrap(
            Bundle(for: type(of: self)).url(forResource: "fixture-vocab", withExtension: "txt")
        )
        XCTAssertThrowsError(
            try WordPieceTokenizer(
                vocabURL: vocabURL,
                maxSequenceLength: 1,
                doLowerCase: true,
                clsTokenId: 2,
                sepTokenId: 3,
                padTokenId: 0,
                unkTokenId: 1
            )
        )
    }
}
