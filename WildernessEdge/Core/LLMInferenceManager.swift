import Foundation
import LiteRTLM
import UIKit

/// Wraps LiteRT-LM's Engine/Conversation API for local multimodal Gemma 4 E4B inference.
/// Never falls back to a networked model on any failure.
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

    private var engine: Engine?
    private var conversation: Conversation?

    private static let systemInstruction = """
    You are a non-diagnostic, non-prescriptive field-protocol assistant. Only present \
    retrieved checklist steps with their citation. Never state a diagnosis or a drug dose. \
    If no protocol context is provided below, say plainly that no matching protocol was \
    found instead of guessing.
    """

    /// Must be called once at app startup, before the first query. Surfaces a blocking
    /// startup error rather than allowing the app into a broken push-to-talk loop.
    func initialize(bundle: Bundle = .main) async {
        initializationError = nil
        isReady = false
        engine = nil
        conversation = nil

        guard let modelURL = bundle.url(forResource: "gemma-4-E4B-it", withExtension: "litertlm") else {
            initializationError = .modelAssetMissing
            return
        }

        let cacheDir = FileManager.default.temporaryDirectory.path

        do {
            // Prefer GPU/Metal; fall back to CPU if this device rejects the GPU backend.
            let engine = try await Self.makeEngine(
                modelPath: modelURL.path,
                cacheDir: cacheDir,
                preferGPU: true
            )
            let conversation = try await engine.createConversation(
                with: ConversationConfig(
                    systemMessage: Message(Self.systemInstruction, role: .system)
                )
            )
            self.engine = engine
            self.conversation = conversation
            isReady = true
        } catch {
            initializationError = .initializationFailed(error.localizedDescription)
        }
    }

    /// Combines the camera snapshot, retrieved RAG context, and transcript into one
    /// multimodal prompt. When `ragResult` is `.noConfidentMatch` (or nil), the model is
    /// explicitly instructed not to fabricate protocol content.
    func generate(
        transcript: String,
        ragResult: VectorRAGManager.RAGResult?,
        image: UIImage?
    ) async throws -> String {
        guard let conversation else {
            throw LLMError.generationFailed("Model not initialized.")
        }

        let contextBlock: String
        switch ragResult {
        case .match(let chunks):
            contextBlock = chunks.map { "\($0.citation)\n\($0.text)" }.joined(separator: "\n\n")
        case .noConfidentMatch, .none:
            contextBlock = "NO_MATCHING_PROTOCOL_FOUND"
        }

        var contents: [Content] = []
        if let image, let imageData = image.jpegData(compressionQuality: 0.8) {
            contents.append(.imageData(imageData))
        }
        contents.append(
            .text(
                """
                Retrieved context:
                \(contextBlock)

                User question: \(transcript)
                """
            )
        )

        let message = Message(contents: contents)

        do {
            var response = ""
            for try await chunk in conversation.sendMessageStream(message) {
                response += chunk.toString
            }
            return response
        } catch {
            throw LLMError.generationFailed(error.localizedDescription)
        }
    }

    /// Builds and initializes an `Engine`. Tries GPU first when requested, then CPU.
    private static func makeEngine(
        modelPath: String,
        cacheDir: String,
        preferGPU: Bool
    ) async throws -> Engine {
        let attempts: [(backend: Backend, vision: Backend)] = preferGPU
            ? [(.gpu, .gpu), (.cpu(), .cpu())]
            : [(.cpu(), .cpu())]

        var lastError: Error?

        for attempt in attempts {
            do {
                let config = try EngineConfig(
                    modelPath: modelPath,
                    backend: attempt.backend,
                    visionBackend: attempt.vision,
                    cacheDir: cacheDir
                )
                let engine = Engine(engineConfig: config)
                // Engine.initialize() is synchronous and heavy; awaiting the actor hop
                // keeps the MainActor responsive during model load.
                try await engine.initialize()
                return engine
            } catch {
                lastError = error
            }
        }

        throw lastError ?? LLMError.initializationFailed("Unknown engine initialization failure.")
    }
}
