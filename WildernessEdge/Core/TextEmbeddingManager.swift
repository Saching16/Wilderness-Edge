import CoreML
import Foundation

/// Wraps the bundled CoreML query embedder. Pooling and L2 normalization are already
/// folded into the model graph, so callers get a directly-comparable 384-dim vector and
/// `VectorRAGManager` can treat a plain dot product as cosine similarity.
///
/// `embed(_:)` is synchronous and runs a full transformer forward pass. Call it off the
/// main actor.
///
/// See plans/vaibhav.md Task B2.
final class TextEmbeddingManager {
    enum EmbeddingError: LocalizedError {
        case modelLoadFailed(String)
        case modelCompilationFailed(String)
        case tokenizerAssetsMissing
        case modelAssetMissing
        case predictionFailed(String)
        case unexpectedOutputShape

        var errorDescription: String? {
            switch self {
            case .modelLoadFailed(let message): return "Failed to load query embedder: \(message)"
            case .modelCompilationFailed(let message): return "Failed to compile query embedder: \(message)"
            case .tokenizerAssetsMissing: return "Tokenizer vocabulary/config not found in bundle."
            case .modelAssetMissing: return "Query embedder model not found in bundle."
            case .predictionFailed(let message): return "Embedding prediction failed: \(message)"
            case .unexpectedOutputShape: return "Embedder returned an unexpected output shape."
            }
        }
    }

    /// Mirrors `query-embedder-tokenizer.json` as written by
    /// `OffLineTools/export_embedder_coreml.py`. The special-token ids are read rather than
    /// assumed: for `all-MiniLM-L6-v2` they are [PAD]=0, [UNK]=100, [CLS]=101, [SEP]=102,
    /// which is not the compact layout the fixture vocabulary uses.
    private struct TokenizerConfig: Decodable {
        let maxSequenceLength: Int
        let doLowerCase: Bool
        let stripAccents: Bool?
        let clsTokenId: Int32
        let sepTokenId: Int32
        let padTokenId: Int32
        let unkTokenId: Int32
    }

    private let model: MLModel
    private let tokenizer: WordPieceTokenizer
    private let sequenceLength: Int

    private static let assetName = "query-embedder"
    private static let inputIdsFeature = "input_ids"
    private static let attentionMaskFeature = "attention_mask"
    private static let outputFeature = "embedding"

    init(bundle: Bundle = .main) throws {
        guard
            let vocabURL = bundle.url(forResource: "\(Self.assetName)-vocab", withExtension: "txt"),
            let configURL = bundle.url(forResource: "\(Self.assetName)-tokenizer", withExtension: "json")
        else {
            throw EmbeddingError.tokenizerAssetsMissing
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(TokenizerConfig.self, from: try Data(contentsOf: configURL))

        self.tokenizer = try WordPieceTokenizer(
            vocabURL: vocabURL,
            maxSequenceLength: config.maxSequenceLength,
            doLowerCase: config.doLowerCase,
            stripAccents: config.stripAccents ?? true,
            clsTokenId: config.clsTokenId,
            sepTokenId: config.sepTokenId,
            padTokenId: config.padTokenId,
            unkTokenId: config.unkTokenId
        )
        self.sequenceLength = config.maxSequenceLength

        let modelURL = try Self.resolveCompiledModel(in: bundle)
        do {
            self.model = try MLModel(contentsOf: modelURL)
        } catch {
            throw EmbeddingError.modelLoadFailed(error.localizedDescription)
        }
    }

    /// Generates a 384-dim, L2-normalized embedding for on-device retrieval.
    func embed(_ text: String) throws -> [Float] {
        let (ids, mask) = tokenizer.encode(text)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            Self.inputIdsFeature: makeInputArray(ids),
            Self.attentionMaskFeature: makeInputArray(mask),
        ])

        let output: MLFeatureProvider
        do {
            output = try model.prediction(from: input)
        } catch {
            throw EmbeddingError.predictionFailed(error.localizedDescription)
        }

        guard
            let embedding = output.featureValue(for: Self.outputFeature)?.multiArrayValue,
            embedding.count > 0
        else {
            throw EmbeddingError.unexpectedOutputShape
        }

        // MLShapedArray reads through the declared strides, so this stays correct whatever
        // layout CoreML hands back, and avoids unboxing 384 NSNumbers one at a time.
        return MLShapedArray<Float>(converting: embedding).scalars
    }

    /// Packs a token sequence into the `int32[1, sequenceLength]` shape the model declares.
    /// `WordPieceTokenizer` always returns exactly `sequenceLength` elements.
    private func makeInputArray(_ values: [Int32]) -> MLMultiArray {
        MLMultiArray(MLShapedArray<Int32>(scalars: values, shape: [1, sequenceLength]))
    }

    // MARK: - Model resolution

    /// Returns a URL CoreML can load directly.
    ///
    /// A `.mlpackage` reaching the bundle through a folder reference is copied verbatim and
    /// never sees Xcode's CoreML build rule, so there is no `.mlmodelc` to load. Compiling
    /// at first launch covers that case and costs nothing when the asset did get compiled
    /// at build time, which is what happens in the test target.
    private static func resolveCompiledModel(in bundle: Bundle) throws -> URL {
        if let compiled = bundle.url(forResource: assetName, withExtension: "mlmodelc") {
            return compiled
        }
        guard let package = bundle.url(forResource: assetName, withExtension: "mlpackage") else {
            throw EmbeddingError.modelAssetMissing
        }
        return try compiledModel(for: package)
    }

    private static func compiledModel(for package: URL) throws -> URL {
        let fileManager = FileManager.default
        let cacheDirectory = try fileManager
            .url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("CompiledModels", isDirectory: true)
        try fileManager.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)

        let destination = cacheDirectory.appendingPathComponent("\(assetName).mlmodelc")

        // Reuse the cached compile unless the bundled asset is newer, which is what a
        // reinstall with regenerated assets looks like. A wrong guess here only costs a
        // recompile, never a stale answer, because the timestamp moves with the bundle.
        if let cached = modificationDate(of: destination),
           let source = modificationDate(of: package),
           cached >= source {
            return destination
        }

        let temporary: URL
        do {
            temporary = try MLModel.compileModel(at: package)
        } catch {
            throw EmbeddingError.modelCompilationFailed(error.localizedDescription)
        }

        do {
            try? fileManager.removeItem(at: destination)
            try fileManager.moveItem(at: temporary, to: destination)
        } catch {
            // The compiled model is still usable where the system put it; it just will not
            // survive to the next launch. Better a slow relaunch than a failed one.
            return temporary
        }

        excludeFromBackup(destination)
        return destination
    }

    private static func modificationDate(of url: URL) -> Date? {
        try? FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date
    }

    /// The compiled model is regenerable from the bundle, so it should not inflate backups.
    private static func excludeFromBackup(_ url: URL) {
        var mutable = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? mutable.setResourceValues(values)
    }
}
