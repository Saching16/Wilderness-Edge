import Accelerate
import Foundation
import SQLite3

/// SIMD-accelerated vector search over the bundled, read-only `protocols.db`.
/// Uses the raw SQLite3 C API (no third-party wrapper) per AGENTS.md.
///
/// The corpus is read once at init and held resident: vectors in a single contiguous
/// row-major matrix, text and citations in parallel arrays. `search(...)` then touches no
/// SQLite at all, which is what lets it be a pure function over immutable state — safe to
/// call from any task without a lock, and with no database handle whose lifetime has to
/// outlive the caller.
///
/// That tradeoff is sized for this corpus: 2192 chunks is roughly 3.4 MB of vectors plus a
/// few MB of text. If the corpus grew by orders of magnitude, the right move is to keep the
/// matrix resident but fetch `text`/`citation` lazily for the top-K only.
///
/// See plans/vaibhav.md Task B3.
final class VectorRAGManager: Sendable {
    struct RetrievedChunk: Sendable, Equatable {
        let citation: String
        let text: String
        let similarity: Float
    }

    enum RAGResult: Sendable, Equatable {
        case match([RetrievedChunk])
        case noConfidentMatch
    }

    enum RAGError: LocalizedError {
        case openFailed(String)
        case queryFailed(String)
        case emptyCorpus
        case inconsistentVectorLength(expected: Int, found: Int)
        case corpusNotNormalized

        var errorDescription: String? {
            switch self {
            case .openFailed(let message):
                return "Could not open protocols.db: \(message)"
            case .queryFailed(let message):
                return "Query against protocols.db failed: \(message)"
            case .emptyCorpus:
                return "protocols.db contains no usable chunks."
            case .inconsistentVectorLength(let expected, let found):
                return "protocols.db mixes vector lengths (expected \(expected), found \(found))."
            case .corpusNotNormalized:
                return "protocols.db declares un-normalized embeddings, so a dot product is not cosine similarity."
            }
        }
    }

    private let citations: [String]
    private let texts: [String]
    /// Row-major, `chunkCount * dimension` floats.
    private let matrix: [Float]

    let chunkCount: Int
    let dimension: Int

    init(databasePath: String) throws {
        var handle: OpaquePointer?
        let status = sqlite3_open_v2(databasePath, &handle, SQLITE_OPEN_READONLY, nil)
        guard status == SQLITE_OK, let database = handle else {
            let message = handle.map { String(cString: sqlite3_errmsg($0)) }
                ?? "unable to open database (SQLite code \(status))"
            // sqlite3_open_v2 hands back a handle even on failure so the error can be read.
            if let handle { sqlite3_close(handle) }
            throw RAGError.openFailed(message)
        }
        // Everything is resident once init returns, so nothing needs the handle afterwards.
        defer { sqlite3_close(database) }

        // `dot == cosine` only holds for unit vectors. build_vector_db.py records whether it
        // normalized; if it says no, fail loudly instead of ranking on a meaningless scale.
        if Self.declaresNormalizedEmbeddings(in: database) == false {
            throw RAGError.corpusNotNormalized
        }

        var statement: OpaquePointer?
        let sql = "SELECT citation, text, embedding FROM chunks ORDER BY id"
        guard sqlite3_prepare_v2(database, sql, -1, &statement, nil) == SQLITE_OK, let statement else {
            throw RAGError.queryFailed(String(cString: sqlite3_errmsg(database)))
        }
        defer { sqlite3_finalize(statement) }

        var citations: [String] = []
        var texts: [String] = []
        var matrix: [Float] = []
        var dimension = 0

        while sqlite3_step(statement) == SQLITE_ROW {
            guard
                let citationBytes = sqlite3_column_text(statement, 0),
                let textBytes = sqlite3_column_text(statement, 1)
            else { continue }

            let byteCount = Int(sqlite3_column_bytes(statement, 2))
            guard
                byteCount > 0,
                byteCount % MemoryLayout<Float>.size == 0,
                let blob = sqlite3_column_blob(statement, 2)
            else { continue }

            let floatCount = byteCount / MemoryLayout<Float>.size
            if dimension == 0 {
                dimension = floatCount
            } else if floatCount != dimension {
                throw RAGError.inconsistentVectorLength(expected: dimension, found: floatCount)
            }

            citations.append(String(cString: citationBytes))
            texts.append(String(cString: textBytes))

            // SQLite makes no alignment promise about blob pointers, so these are unaligned
            // loads. The stored format is little-endian float32 and every Apple platform is
            // little-endian, so the bytes transfer straight across.
            let raw = UnsafeRawBufferPointer(start: blob, count: byteCount)
            matrix.reserveCapacity(matrix.count + floatCount)
            for offset in stride(from: 0, to: byteCount, by: MemoryLayout<Float>.size) {
                matrix.append(raw.loadUnaligned(fromByteOffset: offset, as: Float.self))
            }
        }

        guard dimension > 0, !citations.isEmpty else {
            throw RAGError.emptyCorpus
        }

        self.citations = citations
        self.texts = texts
        self.matrix = matrix
        self.chunkCount = citations.count
        self.dimension = dimension
    }

    /// Returns the top-K chunks by cosine similarity, or `.noConfidentMatch` when the best
    /// score falls below `threshold`.
    ///
    /// Both `embedding` and the stored vectors are L2-normalized, so a plain dot product is
    /// already the cosine similarity — magnitudes are never recomputed.
    func search(embedding: [Float], topK: Int, threshold: Float) -> RAGResult {
        guard topK > 0, embedding.count == dimension else { return .noConfidentMatch }

        var scores = [Float](repeating: 0, count: chunkCount)
        // One batched dot product rather than chunkCount separate vDSP_dotpr calls: the
        // matrix is (chunkCount x dimension), the query is (dimension x 1), so this is the
        // same arithmetic with far better use of a single pass over the vectors.
        vDSP_mmul(
            matrix, 1,
            embedding, 1,
            &scores, 1,
            vDSP_Length(chunkCount), 1, vDSP_Length(dimension)
        )

        let ranked = Self.topIndices(in: scores, count: min(topK, chunkCount))
        guard let best = ranked.first, scores[best] >= threshold else {
            return .noConfidentMatch
        }

        return .match(ranked.map { index in
            RetrievedChunk(citation: citations[index], text: texts[index], similarity: scores[index])
        })
    }

    /// Bounded insertion beats sorting all `chunkCount` scores when `count` is a handful,
    /// and it allocates nothing beyond the result.
    private static func topIndices(in scores: [Float], count: Int) -> [Int] {
        guard count > 0 else { return [] }

        var best: [Int] = []
        best.reserveCapacity(count)

        for index in scores.indices {
            let score = scores[index]
            if best.count == count, let weakest = best.last, score <= scores[weakest] {
                continue
            }
            var position = best.count
            while position > 0, score > scores[best[position - 1]] {
                position -= 1
            }
            best.insert(index, at: position)
            if best.count > count { best.removeLast() }
        }
        return best
    }

    /// Reads `meta.embeddings_normalized`, or nil when the row or the whole table is absent
    /// — fixture databases and older builds legitimately have no `meta`, and that is not an
    /// error, just an absence of evidence.
    private static func declaresNormalizedEmbeddings(in database: OpaquePointer) -> Bool? {
        var statement: OpaquePointer?
        let sql = "SELECT value FROM meta WHERE key = 'embeddings_normalized'"
        guard sqlite3_prepare_v2(database, sql, -1, &statement, nil) == SQLITE_OK, let statement else {
            return nil  // no meta table: prepare fails rather than returning zero rows
        }
        defer { sqlite3_finalize(statement) }

        guard sqlite3_step(statement) == SQLITE_ROW, let value = sqlite3_column_text(statement, 0) else {
            return nil
        }
        return String(cString: value) == "1"
    }
}
