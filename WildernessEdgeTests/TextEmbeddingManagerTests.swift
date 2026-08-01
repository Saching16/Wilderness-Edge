import XCTest

/// Embedding-space parity against `embedding_parity_fixtures.json`, the reference vectors
/// `OffLineTools/build_vector_db.py` emitted while building `protocols.db`.
///
/// This is the check that PLAN.md Phase 2 hangs on. If it fails, on-device query vectors
/// live in a different space than the stored chunk vectors, and retrieval returns
/// confident-looking nonsense rather than failing loudly.
///
/// Skips until Pablo's Checkpoint 1 assets are present. See `assetsSkipReason` for exactly
/// what has to land.
final class TextEmbeddingManagerTests: XCTestCase {
    private struct FixtureFile: Decodable {
        struct Fixture: Decodable {
            let text: String
            let embedding: [Float]
        }

        let embeddingModel: String
        let embeddingDim: Int
        let normalized: Bool
        let fixtures: [Fixture]
    }

    private static let assetsSkipReason = """
        Query embedder assets are not in the test bundle. This test starts running once \
        Pablo's Checkpoint 1 delivery lands AND query-embedder.mlpackage, \
        query-embedder-vocab.txt and query-embedder-tokenizer.json are listed under the \
        WildernessEdgeTests target in project.yml (they are already there as optional \
        entries, so re-running `xcodegen generate` should be enough).
        """

    private func loadFixtures() throws -> FixtureFile {
        guard let url = Bundle(for: type(of: self)).url(
            forResource: "embedding_parity_fixtures", withExtension: "json"
        ) else {
            throw XCTSkip("embedding_parity_fixtures.json not yet present — run after Pablo's Checkpoint 1 delivery.")
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(FixtureFile.self, from: try Data(contentsOf: url))
    }

    private func makeManager() throws -> TextEmbeddingManager {
        do {
            return try TextEmbeddingManager(bundle: Bundle(for: type(of: self)))
        } catch let error as TextEmbeddingManager.EmbeddingError {
            switch error {
            case .tokenizerAssetsMissing, .modelAssetMissing:
                throw XCTSkip(Self.assetsSkipReason)
            default:
                throw error
            }
        }
    }

    func testEmbeddingMatchesParityFixtures() throws {
        let fixtureFile = try loadFixtures()
        let manager = try makeManager()

        for fixture in fixtureFile.fixtures {
            let produced = try manager.embed(fixture.text)

            XCTAssertEqual(
                produced.count, fixture.embedding.count,
                "Dimensionality mismatch for: \(fixture.text)"
            )
            guard produced.count == fixture.embedding.count else { continue }

            // Both sides are L2-normalized, so the dot product is cosine similarity.
            let similarity = zip(produced, fixture.embedding).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            XCTAssertGreaterThanOrEqual(
                similarity, 0.999,
                """
                Parity mismatch for "\(fixture.text)" (cosine \(similarity)). The CoreML \
                embedder and protocols.db came from different models — check that \
                build_vector_db.py and export_embedder_coreml.py ran with the same --model.
                """
            )
        }
    }

    /// `VectorRAGManager` treats a dot product as cosine similarity, which is only valid
    /// because the embedder folds L2 normalization into its graph. Pin that here rather
    /// than discovering it as silently wrong ranking.
    func testEmbeddingsAreL2Normalized() throws {
        _ = try loadFixtures()  // skip in lockstep with the parity test
        let manager = try makeManager()

        let vector = try manager.embed("severe bleeding from the thigh")
        let norm = sqrt(vector.reduce(Float(0)) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1.0, accuracy: 1e-3, "Embedder output is not unit length")
    }

    func testFixtureMetadataMatchesTheAgreedModel() throws {
        let fixtureFile = try loadFixtures()
        XCTAssertEqual(fixtureFile.embeddingModel, "sentence-transformers/all-MiniLM-L6-v2")
        XCTAssertEqual(fixtureFile.embeddingDim, 384)
        XCTAssertTrue(fixtureFile.normalized)
    }
}
