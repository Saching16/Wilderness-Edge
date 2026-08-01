import XCTest

/// Runs against `fixture-protocols.db`, three chunks carrying orthogonal 4-dim unit
/// vectors (see `OffLineTools/make_test_fixtures.py`). Orthogonality makes every expected
/// cosine similarity exactly 1.0 or 0.0, so these assert real numbers rather than
/// tolerances. `VectorRAGManager` reads its dimensionality from the stored blobs, so the
/// toy dimension exercises the same path the real 384-dim corpus takes.
///
/// Note: `RAGResult` declares `Equatable` itself, so unlike the sketch in plans/vaibhav.md
/// no retroactive conformance is needed here — adding one would be a duplicate.
final class VectorRAGManagerTests: XCTestCase {
    private func makeManager(fixture: String = "fixture-protocols") throws -> VectorRAGManager {
        let path = try XCTUnwrap(
            Bundle(for: type(of: self)).path(forResource: fixture, ofType: "db"),
            "\(fixture).db is not in the test bundle — check the WildernessEdgeTests sources in project.yml."
        )
        return try VectorRAGManager(databasePath: path)
    }

    private struct ExpectedMatch: Error {}

    /// Throws a plain error rather than `XCTSkip` on the unhappy path — an `XCTSkip` here
    /// would report a genuine failure as a skipped test.
    private func matchedChunks(
        _ result: VectorRAGManager.RAGResult,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws -> [VectorRAGManager.RetrievedChunk] {
        guard case .match(let chunks) = result else {
            XCTFail("Expected a match, got \(result)", file: file, line: line)
            throw ExpectedMatch()
        }
        return chunks
    }

    // MARK: - Ranking

    func testExactMatchReturnsHighestSimilarityFirst() throws {
        let manager = try makeManager()
        let result = manager.search(embedding: [1, 0, 0, 0], topK: 3, threshold: 0.35)
        let chunks = try matchedChunks(result)

        XCTAssertEqual(chunks.first?.text, "Apply direct pressure to the wound.")
        XCTAssertEqual(chunks.first?.citation, "[Source: Fixture Manual, Section 1.1, p. 10]")
        XCTAssertEqual(chunks.first?.similarity ?? 0, 1.0, accuracy: 0.0001)
    }

    func testResultsAreOrderedByDescendingSimilarity() throws {
        let manager = try makeManager()
        let result = manager.search(embedding: [0.9, 0.4, 0.1, 0], topK: 3, threshold: -1)
        let chunks = try matchedChunks(result)

        XCTAssertEqual(chunks.count, 3)
        for (chunk, expected) in zip(chunks, [Float(0.9), 0.4, 0.1]) {
            XCTAssertEqual(chunk.similarity, expected, accuracy: 0.0001)
        }
        XCTAssertEqual(
            chunks.map(\.text),
            [
                "Apply direct pressure to the wound.",
                "Splint the limb in the position found.",
                "Insulate and passively rewarm the patient.",
            ]
        )
    }

    func testTopKLimitsResultCount() throws {
        let manager = try makeManager()
        let result = manager.search(embedding: [0.5, 0.5, 0.5, 0], topK: 2, threshold: -1)
        XCTAssertEqual(try matchedChunks(result).count, 2)
    }

    func testTopKLargerThanCorpusReturnsEveryChunk() throws {
        let manager = try makeManager()
        let result = manager.search(embedding: [0.5, 0.5, 0.5, 0], topK: 10, threshold: -1)
        XCTAssertEqual(try matchedChunks(result).count, 3)
    }

    // MARK: - Confidence threshold

    func testBelowThresholdReturnsNoConfidentMatch() throws {
        let manager = try makeManager()
        // Orthogonal to all three stored vectors, so every similarity is 0.0.
        let result = manager.search(embedding: [0, 0, 0, 1], topK: 3, threshold: 0.35)
        XCTAssertEqual(result, .noConfidentMatch)
    }

    /// The threshold is inclusive: a score sitting exactly on it is a match, not a miss.
    func testScoreExactlyOnThresholdStillMatches() throws {
        let manager = try makeManager()
        let result = manager.search(embedding: [1, 0, 0, 0], topK: 1, threshold: 1.0)
        XCTAssertEqual(try matchedChunks(result).count, 1)
    }

    func testZeroTopKReturnsNoConfidentMatch() throws {
        let manager = try makeManager()
        XCTAssertEqual(
            manager.search(embedding: [1, 0, 0, 0], topK: 0, threshold: -1),
            .noConfidentMatch
        )
    }

    /// A query embedded by a different model would arrive at the wrong width. Returning
    /// `.noConfidentMatch` keeps that honest rather than comparing truncated vectors.
    func testDimensionMismatchReturnsNoConfidentMatch() throws {
        let manager = try makeManager()
        let wrongWidth = [Float](repeating: 0.1, count: 384)
        XCTAssertEqual(
            manager.search(embedding: wrongWidth, topK: 3, threshold: -1),
            .noConfidentMatch
        )
    }

    // MARK: - Corpus metadata

    func testChunkCountAndDimensionReflectTheDatabase() throws {
        let manager = try makeManager()
        XCTAssertEqual(manager.chunkCount, 3)
        XCTAssertEqual(manager.dimension, 4)
    }

    // MARK: - Failure modes

    func testOpeningMissingDatabaseThrows() {
        XCTAssertThrowsError(try VectorRAGManager(databasePath: "/nonexistent/path.db"))
    }

    /// `dot == cosine` only holds for unit vectors. A corpus that declares itself
    /// un-normalized must be refused, not ranked on a meaningless scale.
    func testUnnormalizedCorpusIsRejected() throws {
        let path = try XCTUnwrap(
            Bundle(for: type(of: self)).path(forResource: "fixture-unnormalized", ofType: "db")
        )
        XCTAssertThrowsError(try VectorRAGManager(databasePath: path)) { error in
            guard case VectorRAGManager.RAGError.corpusNotNormalized = error else {
                return XCTFail("Expected .corpusNotNormalized, got \(error)")
            }
        }
    }
}
