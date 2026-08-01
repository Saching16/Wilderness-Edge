# Wilderness Edge — Hackathon Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a working, demoable Wilderness Edge for the Build with Gemma NYC hackathon: push-to-talk voice+vision query → on-device vector RAG over a licensed first-aid corpus → Gemma 4 E4B multimodal inference via LiteRT-LM → non-diagnostic safety filter → spoken response, running fully offline (Airplane Mode) on a sideloaded iPhone 16 Plus.

**Architecture:** Four people work in parallel on the four pipeline stages defined in `plans/scope-design.md`. Stages that consume another stage's output start immediately against stub data/fixtures and swap in the real artifact at the integration checkpoints called out below, so nobody blocks on anybody else for more than a couple of hours.

**Tech Stack:** Swift 5.9 / SwiftUI (iOS 17+), LiteRT-LM Swift API, CoreML, raw SQLite3 C API, Accelerate (vDSP), Python 3.11–3.13 (offline tooling only, never ships).

## Global Constraints

- Every LLM/embedding operation on-device must be reachable with zero network requests at runtime (Airplane Mode is the actual test condition, not a claim). Never fall back to a networked alternative on any failure — fail closed with a visible error state instead.
- Model is Gemma 4 **E4B** (not E2B) — use the prebuilt `litert-community/gemma-4-E4B-it-litert-lm` bundle. No custom LoRA fine-tune this sprint.
- `TextEmbeddingManager`'s CoreML model and `protocols.db`'s stored embeddings must come from the exact same `--model` value (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim). Never swap one without regenerating/re-validating the other (`AGENTS.md` guideline 6).
- All LLM text output must pass through `SafetyFilter.sanitize(_:)` before display or TTS — never bypass it (`AGENTS.md` guideline 4).
- Corpus sources are restricted to those already verified in `OffLineTools/sources.manifest.json` (ATP 4-02.11, TCCC Handbook v5, NASEMSO Guidelines v3.0). Do not add NOLS material — confirmed unlicensed in `SOURCES.md`.
- No hard memory-footprint gate this sprint (soft target only: don't be reckless on an iPhone 16 Plus). Keep the existing `com.apple.developer.kernel.increased-memory-limit` entitlement.
- "No confident RAG match" is a normal, honestly-spoken result, not an error — the model must be told explicitly not to fabricate context when this happens.
- Similarity threshold for "no confident match": use **0.35** (per `OffLineTools/README.md`'s calibration: genuine queries score 0.53–0.67, off-topic scores ~0.17 on the current corpus).

## File Structure

New files this sprint:
- `WildernessEdge/Core/WordPieceTokenizer.swift` — BERT WordPiece tokenizer (Vaibhav)
- `WildernessEdge/Core/TextEmbeddingManager.swift` — CoreML embedder wrapper (Vaibhav)
- `WildernessEdge/Core/VectorRAGManager.swift` — SQLite + Accelerate vector search (Vaibhav)
- `WildernessEdgeTests/VectorRAGManagerTests.swift` — RAG unit tests (Vaibhav)
- `WildernessEdgeTests/WordPieceTokenizerTests.swift` — tokenizer unit tests (Vaibhav)
- `WildernessEdge/Core/LLMInferenceManager.swift` — LiteRT-LM wrapper (Sachin)
- `WildernessEdge/Views/EmergencyButtonView.swift` — push-to-talk button (Daniel)
- `WildernessEdge/Views/SubtitleCardView.swift` — citation/checklist card (Daniel)

Modified files:
- `WildernessEdge/Views/ContentView.swift` — replace Phase-1 scaffold with the full state machine (Daniel, then Sachin wires in LLM calls)
- `project.yml` — add LiteRT-LM SPM package, add `VectorRAGManagerTests`/`WordPieceTokenizerTests` to the test target sources (Sachin for the package; Vaibhav for the test sources)
- `OffLineTools/sources.manifest.json` / `SOURCES.md` — no changes needed; already scoped to licensed-only sources
- `WildernessEdge/Resources/` — receives `protocols.db`, `query-embedder.mlpackage`, `query-embedder-vocab.txt`, `query-embedder-tokenizer.json`, `gemma-4-E4B-it.litertlm` (Pablo)

## Integration Checkpoints

- **Checkpoint 1 (target: ~30-45 min):** Pablo delivers `protocols.db` + CoreML embedder assets into `WildernessEdge/Resources/`. Vaibhav swaps `WordPieceTokenizer`/`TextEmbeddingManager` from fixture vocab to the real `query-embedder-vocab.txt`/`query-embedder-tokenizer.json`, re-runs the parity test against `embedding_parity_fixtures.json`.
- **Checkpoint 2 (target: ~1 hr):** Pablo delivers the `.litertlm` bundle. Sachin swaps `LLMInferenceManager`'s `EngineConfig` path from a not-yet-initialized stub to the real bundle and confirms model load succeeds on-device.
- **Checkpoint 3 (target: ~1.5 hr):** Vaibhav's `VectorRAGManager` is functionally complete. Sachin replaces his stubbed RAG input with real `VectorRAGManager.search(...)` output in the prompt construction.
- **Checkpoint 4 (target: ~2 hr):** Daniel's `ContentView` state machine and Sachin's `LLMInferenceManager` are both ready. Wire the full button-down → button-up pipeline together in `ContentView`.
- **Checkpoint 5 (final hours):** Full pipeline dry run in Airplane Mode on the physical iPhone 16 Plus, twice consecutively, per the Testing Bar below.

## Testing Bar For This Sprint

- `SafetyFilterTests` (existing) and new `VectorRAGManagerTests` / `WordPieceTokenizerTests` pass in Simulator/CI.
- Manual Airplane Mode run-through on the physical device: press button, ask a wilderness/first-aid question, get a spoken checklist + citation back — reliably twice in a row. This replaces PLAN.md Phase 5's 20-consecutive-query stress suite for this sprint.

---

## Track A — Offline Assets & Corpus (Owner: Pablo)

### Task A1: Generate the licensed corpus vector database

**Files:**
- Modify: none (uses existing `OffLineTools/fetch_sources.py`, `OffLineTools/build_vector_db.py`)
- Output: `OffLineTools/build/protocols.db`, `OffLineTools/build/embedding_parity_fixtures.json`

**Interfaces:**
- Produces: `protocols.db` (SQLite, schema per `OffLineTools/build_vector_db.py` — `meta`, `sources`, `chunks` tables; 384-dim L2-normalized float32 little-endian embeddings) consumed by Vaibhav's `VectorRAGManager` (Track B) and cross-checked by `embedding_parity_fixtures.json`.

- [ ] **Step 1: Install offline tooling dependencies**

```bash
cd OffLineTools
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Expected: no dependency errors. Do not let `transformers` resolve to `>=5`; `requirements.txt` already pins `<5`.

- [ ] **Step 2: Fetch the vetted corpus**

```bash
python fetch_sources.py
```

Expected output ends with `All manifested sources present in sources` (3 PDFs: `atp4-02-11-casualty-response-tccc-first-aid.pdf`, `tccc-handbook-v5.pdf`, `nasemso-national-model-ems-clinical-guidelines-v3.pdf`). If any fetch fails, download manually from the URL in `SOURCES.md` into `OffLineTools/sources/` under the exact manifest filename, then re-run.

- [ ] **Step 3: Dry-run the chunker and inspect output quality**

```bash
python build_vector_db.py --dry-run
```

Expected: a per-PDF `pages -> chunks` line for all 3 sources, then sampled chunk text with citations like `[Source: NASEMSO National Model EMS Clinical Guidelines v3.0, Section X, p. N]`. Read the samples — if a source produced garbled/empty text (scanned-image PDF with no text layer), flag it, don't silently ship it.

- [ ] **Step 4: Build the database and parity fixtures**

```bash
python build_vector_db.py
```

Expected: `Wrote build/protocols.db (N.N MB)` and `Wrote parity fixtures to build/embedding_parity_fixtures.json`. This step also fails loudly (`BuildError`) if any source PDF lacks a manifest entry — expected behavior, not a bug, since every shipped chunk must carry a verified license.

- [ ] **Step 5: Calibrate retrieval quality**

```bash
python query_protocols.py
```

Expected: on-topic probes score roughly 0.53–0.67, the off-topic control ("what is the best pizza topping") scores roughly 0.17. Confirm this margin holds — if it's degraded, note it for Vaibhav since it affects the 0.35 threshold assumption in Global Constraints.

- [ ] **Step 6: Commit the offline tooling run is reproducible (no code changes — nothing to commit here)**

`OffLineTools/build/` and `OffLineTools/sources/` are gitignored (regenerable). Nothing to `git add`/commit for this task — proceed directly to Task A2.

### Task A2: Export the CoreML query embedder and hand off Resources

**Files:**
- Output: `OffLineTools/build/query-embedder.mlpackage`, `OffLineTools/build/query-embedder-vocab.txt`, `OffLineTools/build/query-embedder-tokenizer.json`
- Modify: `WildernessEdge/Resources/` (copy destination)
- Modify: `WildernessEdgeTests/` (copy destination for parity fixtures)

**Interfaces:**
- Consumes: `OffLineTools/build/embedding_parity_fixtures.json` from Task A1 (must exist first — the export's parity check reads it).
- Produces: `query-embedder.mlpackage` (CoreML model, inputs `input_ids`/`attention_mask` as `int32[1,128]`, output `embedding` as `float32[384]`), `query-embedder-vocab.txt` (newline-delimited WordPiece vocab), `query-embedder-tokenizer.json` (config: `do_lower_case`, `strip_accents`, `continuing_subword_prefix: "##"`, `cls_token_id`, `sep_token_id`, `pad_token_id`, `unk_token_id`, `max_sequence_length: 128`) — all consumed by Vaibhav's `WordPieceTokenizer`/`TextEmbeddingManager` (Track B, Checkpoint 1).

- [ ] **Step 1: Export the embedder**

```bash
python export_embedder_coreml.py
```

Expected: `CoreML vs PyTorch parity` section with all rows flagged `ok` (similarity ≥ 0.999), then `CoreML vs protocols.db build parity` section also all `ok`, then a final `Next: drag the .mlpackage...` message and exit code 0. **Do not proceed if any row says `MISMATCH`** — that means the export model diverged from the database's embedding space and retrieval would silently return garbage on-device.

- [ ] **Step 2: Copy assets into the Xcode resource bundle**

```bash
cp OffLineTools/build/protocols.db WildernessEdge/Resources/
cp -R OffLineTools/build/query-embedder.mlpackage WildernessEdge/Resources/
cp OffLineTools/build/query-embedder-vocab.txt WildernessEdge/Resources/
cp OffLineTools/build/query-embedder-tokenizer.json WildernessEdge/Resources/
cp OffLineTools/build/embedding_parity_fixtures.json WildernessEdgeTests/
```

Expected: all 5 files present at their destinations. Xcode compiles the `.mlpackage` into `.mlmodelc` at build time — do not commit a prebuilt `.mlmodelc`.

- [ ] **Step 3: Notify the team assets are ready (Checkpoint 1)**

Message Vaibhav that `WildernessEdge/Resources/` now has real corpus/embedder assets and `WildernessEdgeTests/embedding_parity_fixtures.json` is in place, so he can swap off fixture data in `TextEmbeddingManager`.

- [ ] **Step 4: Commit**

```bash
git add WildernessEdge/Resources/protocols.db WildernessEdge/Resources/query-embedder.mlpackage \
  WildernessEdge/Resources/query-embedder-vocab.txt WildernessEdge/Resources/query-embedder-tokenizer.json \
  WildernessEdgeTests/embedding_parity_fixtures.json
git commit -m "Add protocols.db and CoreML query embedder assets"
```

### Task A3: Fetch the prebuilt Gemma 4 E4B LiteRT-LM bundle

**Files:**
- Output: `WildernessEdge/Resources/gemma-4-E4B-it.litertlm` (gitignored — `*.litertlm` is already excluded in `.gitignore`)

**Interfaces:**
- Produces: the `.litertlm` bundle path consumed by Sachin's `LLMInferenceManager` (Track D, Checkpoint 2).

- [ ] **Step 1: Download the prebuilt bundle**

Download `litert-community/gemma-4-E4B-it-litert-lm` (the multimodal, image+text-capable variant) from its Hugging Face / Kaggle model page onto the Mac that will build the Xcode project. This requires accepting Gemma's license terms if gated.

- [ ] **Step 2: Place it in Resources and verify size**

```bash
cp <downloaded-path>/gemma-4-E4B-it.litertlm WildernessEdge/Resources/
ls -lh WildernessEdge/Resources/gemma-4-E4B-it.litertlm
```

Expected: a multi-GB file present. No hard size ceiling this sprint (Global Constraints), but note the size for the team — an iPhone 16 Plus has 8GB RAM, so keep an eye out if the file is unexpectedly huge (>6GB) since LiteRT-LM memory-maps it.

- [ ] **Step 3: Notify the team the model bundle is ready (Checkpoint 2)**

Message Sachin that `WildernessEdge/Resources/gemma-4-E4B-it.litertlm` is in place so he can point `EngineConfig` at the real bundle and test model load on-device.

- [ ] **Step 4: Nothing to commit** (`.litertlm` is gitignored by design — large model bundles stay out of git). Confirm with `git status` that it shows as untracked, not staged.

### Task A4: Draft the Kaggle writeup (co-owned with Daniel)

**Files:**
- Create: `docs/kaggle-writeup-draft.md` (working draft; final copy goes into the Kaggle Writeup UI at submission time)

- [ ] **Step 1: Draft architecture + Gemma-usage sections as soon as Checkpoint 3 lands**

Cover: title/subtitle, problem statement (decision support for off-grid first responders, patient data never leaves the device), architecture diagram (reuse the data-flow diagram from `AGENTS.md`), specifically how Gemma 4 E4B is used (multimodal image+RAG-context+transcript prompt via LiteRT-LM), what was cut for the 1-day sprint and why (LoRA fine-tune dropped, prebuilt weights used instead — a deliberate engineering tradeoff, not a shortcut), track selection (On-Device Private Health, primary; Voice for Care, secondary). Keep total under 1,500 words.

- [ ] **Step 2: Attach links once the public repo and demo recording/notebook exist**

Add the public code repository link and live demo link (recording or clonable notebook) under "Project Links" in the Kaggle Writeup's Attachments section before submitting.

- [ ] **Step 3: Commit the working draft**

```bash
git add docs/kaggle-writeup-draft.md
git commit -m "Draft Kaggle writeup"
```

---

## Track B — On-Device Vector RAG Engine (Owner: Vaibhav)

### Task B1: WordPieceTokenizer with fixture vocab (TDD)

**Files:**
- Create: `WildernessEdge/Core/WordPieceTokenizer.swift`
- Create: `WildernessEdgeTests/WordPieceTokenizerTests.swift`
- Create (temporary fixture, deleted at Checkpoint 1): `WildernessEdgeTests/Fixtures/fixture-vocab.txt`

**Interfaces:**
- Produces: `WordPieceTokenizer.encode(_ text: String) -> (inputIds: [Int32], attentionMask: [Int32])`, both fixed-length 128, consumed by `TextEmbeddingManager` (Task B2).

- [ ] **Step 1: Write a minimal fixture vocabulary for TDD before real assets exist**

```bash
mkdir -p WildernessEdgeTests/Fixtures
```

Write `WildernessEdgeTests/Fixtures/fixture-vocab.txt`:
```
[PAD]
[UNK]
[CLS]
[SEP]
patient
has
severe
bleeding
from
the
left
thigh
##ing
##s
```
(Index 0 = `[PAD]`, 1 = `[UNK]`, 2 = `[CLS]`, 3 = `[SEP]`, matching BERT convention — this must match whatever `query-embedder-tokenizer.json` records at Checkpoint 1, i.e. `pad_token_id: 0, cls_token_id: 2, sep_token_id: 3, unk_token_id: 1`.)

- [ ] **Step 2: Write the failing test**

```swift
// WildernessEdgeTests/WordPieceTokenizerTests.swift
import XCTest

final class WordPieceTokenizerTests: XCTestCase {
    func makeFixtureTokenizer() throws -> WordPieceTokenizer {
        let vocabURL = Bundle(for: type(of: self)).url(forResource: "fixture-vocab", withExtension: "txt")!
        return try WordPieceTokenizer(
            vocabURL: vocabURL,
            maxSequenceLength: 16,
            doLowerCase: true,
            clsTokenId: 2,
            sepTokenId: 3,
            padTokenId: 0,
            unkTokenId: 1
        )
    }

    func testEncodesKnownWordsWithClsAndSep() throws {
        let tokenizer = try makeFixtureTokenizer()
        let (ids, mask) = tokenizer.encode("patient has severe bleeding")
        XCTAssertEqual(ids.count, 16)
        XCTAssertEqual(mask.count, 16)
        XCTAssertEqual(ids[0], 2) // [CLS]
        XCTAssertTrue(ids.contains(4)) // "patient"
        XCTAssertEqual(mask[0], 1)
        XCTAssertEqual(mask[15], 0) // padded tail
    }

    func testUnknownWordMapsToUnkToken() throws {
        let tokenizer = try makeFixtureTokenizer()
        let (ids, _) = tokenizer.encode("xyzzyunknownword")
        XCTAssertTrue(ids.contains(1)) // [UNK]
    }

    func testAttentionMaskMatchesRealTokenCount() throws {
        let tokenizer = try makeFixtureTokenizer()
        let (_, mask) = tokenizer.encode("patient has")
        // [CLS] patient has [SEP] = 4 real tokens
        XCTAssertEqual(mask.filter { $0 == 1 }.count, 4)
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Run the `WildernessEdgeTests` scheme (`xcodebuild test -scheme WildernessEdgeTests -destination 'platform=iOS Simulator,name=iPhone 15'` or via Xcode). Expected: FAIL — `WordPieceTokenizer` does not exist.

- [ ] **Step 4: Implement WordPieceTokenizer**

```swift
// WildernessEdge/Core/WordPieceTokenizer.swift
import Foundation

/// BERT-style WordPiece tokenizer feeding the bundled CoreML query embedder.
/// CoreML cannot accept strings, so tokenization must happen here in Swift.
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

    private let vocab: [String: Int32]
    private let maxSequenceLength: Int
    private let doLowerCase: Bool
    private let clsTokenId: Int32
    private let sepTokenId: Int32
    private let padTokenId: Int32
    private let unkTokenId: Int32
    private static let continuingSubwordPrefix = "##"

    init(
        vocabURL: URL,
        maxSequenceLength: Int,
        doLowerCase: Bool,
        clsTokenId: Int32,
        sepTokenId: Int32,
        padTokenId: Int32,
        unkTokenId: Int32
    ) throws {
        let contents: String
        do {
            contents = try String(contentsOf: vocabURL, encoding: .utf8)
        } catch {
            throw TokenizerError.vocabLoadFailed(error.localizedDescription)
        }

        var table: [String: Int32] = [:]
        for (index, line) in contents.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            let token = String(line)
            if !token.isEmpty || index == 0 {
                table[token] = Int32(index)
            }
        }
        self.vocab = table
        self.maxSequenceLength = maxSequenceLength
        self.doLowerCase = doLowerCase
        self.clsTokenId = clsTokenId
        self.sepTokenId = sepTokenId
        self.padTokenId = padTokenId
        self.unkTokenId = unkTokenId
    }

    /// Returns fixed-length `input_ids` and `attention_mask`, both padded/truncated to `maxSequenceLength`.
    func encode(_ text: String) -> (inputIds: [Int32], attentionMask: [Int32]) {
        let basicTokens = basicTokenize(text)
        var ids: [Int32] = [clsTokenId]

        for token in basicTokens {
            ids.append(contentsOf: wordpieceTokenize(token))
            if ids.count >= maxSequenceLength - 1 {
                break
            }
        }

        if ids.count > maxSequenceLength - 1 {
            ids = Array(ids.prefix(maxSequenceLength - 1))
        }
        ids.append(sepTokenId)

        let realCount = ids.count
        if ids.count < maxSequenceLength {
            ids.append(contentsOf: Array(repeating: padTokenId, count: maxSequenceLength - ids.count))
        }

        var mask = Array(repeating: Int32(1), count: realCount)
        mask.append(contentsOf: Array(repeating: Int32(0), count: maxSequenceLength - realCount))

        return (ids, mask)
    }

    private func basicTokenize(_ text: String) -> [String] {
        var working = doLowerCase ? text.lowercased() : text
        if doLowerCase {
            working = working.folding(options: .diacriticInsensitive, locale: nil)
        }
        return working
            .components(separatedBy: .whitespacesAndNewlines)
            .flatMap { splitPunctuation($0) }
            .filter { !$0.isEmpty }
    }

    private func splitPunctuation(_ token: String) -> [String] {
        var result: [String] = []
        var current = ""
        for char in token {
            if char.isLetter || char.isNumber {
                current.append(char)
            } else {
                if !current.isEmpty { result.append(current); current = "" }
                result.append(String(char))
            }
        }
        if !current.isEmpty { result.append(current) }
        return result
    }

    private func wordpieceTokenize(_ token: String) -> [Int32] {
        if let direct = vocab[token] {
            return [direct]
        }

        var subTokens: [Int32] = []
        var start = token.startIndex
        var isBad = false

        while start < token.endIndex {
            var end = token.endIndex
            var matched: Int32?

            while end > start {
                let substring = String(token[start..<end])
                let candidate = start == token.startIndex ? substring : Self.continuingSubwordPrefix + substring
                if let id = vocab[candidate] {
                    matched = id
                    break
                }
                end = token.index(before: end)
            }

            guard let id = matched else {
                isBad = true
                break
            }
            subTokens.append(id)
            start = end
        }

        return isBad ? [unkTokenId] : subTokens
    }
}
```

- [ ] **Step 5: Run to verify it passes**

Re-run the `WildernessEdgeTests` scheme. Expected: all 3 `WordPieceTokenizerTests` PASS.

- [ ] **Step 6: Commit**

```bash
git add WildernessEdge/Core/WordPieceTokenizer.swift WildernessEdgeTests/WordPieceTokenizerTests.swift \
  WildernessEdgeTests/Fixtures/fixture-vocab.txt
git commit -m "Add WordPieceTokenizer with fixture-based tests"
```

### Task B2: TextEmbeddingManager wrapping the CoreML embedder

**Files:**
- Create: `WildernessEdge/Core/TextEmbeddingManager.swift`
- Modify: `project.yml` (ensure `WildernessEdge/Resources` folder reference already covers the `.mlpackage` — it does per existing config, no change needed)

**Interfaces:**
- Consumes: `WordPieceTokenizer.encode(_:) -> (inputIds: [Int32], attentionMask: [Int32])` from Task B1.
- Produces: `TextEmbeddingManager.embed(_ text: String) throws -> [Float]` (384-dim), consumed by `VectorRAGManager` callers in `ContentView` (Track D/Checkpoint 4) and by this task's own parity test.

- [ ] **Step 1: Write the parity test against real assets (this test can only run after Checkpoint 1 — write it now, mark expected-to-fail until then)**

```swift
// WildernessEdgeTests/TextEmbeddingManagerTests.swift
import XCTest

final class TextEmbeddingManagerTests: XCTestCase {
    func testEmbeddingMatchesParityFixtures() throws {
        guard let fixturesURL = Bundle(for: type(of: self)).url(
            forResource: "embedding_parity_fixtures", withExtension: "json"
        ) else {
            throw XCTSkip("embedding_parity_fixtures.json not yet present — run after Checkpoint 1 (Task A2).")
        }

        let data = try Data(contentsOf: fixturesURL)
        struct Fixture: Decodable { let text: String; let embedding: [Float] }
        struct FixtureFile: Decodable { let fixtures: [Fixture] }
        let fixtureFile = try JSONDecoder().decode(FixtureFile.self, from: data)

        let manager = try TextEmbeddingManager()
        for fixture in fixtureFile.fixtures {
            let produced = try manager.embed(fixture.text)
            XCTAssertEqual(produced.count, fixture.embedding.count)
            var dot: Float = 0
            for i in 0..<produced.count { dot += produced[i] * fixture.embedding[i] }
            XCTAssertGreaterThanOrEqual(dot, 0.999, "Parity mismatch for: \(fixture.text)")
        }
    }
}
```

- [ ] **Step 2: Run to verify it currently skips (pre-Checkpoint 1) or fails (post-Checkpoint 1, no implementation yet)**

Expected before Checkpoint 1: `XCTSkip` — no fixtures file yet. This is fine; proceed to implementation.

- [ ] **Step 3: Implement TextEmbeddingManager**

```swift
// WildernessEdge/Core/TextEmbeddingManager.swift
import CoreML
import Foundation

/// Wraps the bundled CoreML query embedder. Pooling and L2 normalization are already
/// folded into the model graph, so callers get a directly-comparable 384-dim vector.
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

    private let model: MLModel
    private let tokenizer: WordPieceTokenizer
    private let sequenceLength: Int

    init(bundle: Bundle = .main) throws {
        guard
            let vocabURL = bundle.url(forResource: "query-embedder-vocab", withExtension: "txt"),
            let configURL = bundle.url(forResource: "query-embedder-tokenizer", withExtension: "json"),
            let modelURL = bundle.url(forResource: "query-embedder", withExtension: "mlmodelc")
        else {
            throw EmbeddingError.tokenizerAssetsMissing
        }

        let configData = try Data(contentsOf: configURL)
        struct TokenizerConfig: Decodable {
            let max_sequence_length: Int
            let do_lower_case: Bool
            let cls_token_id: Int32
            let sep_token_id: Int32
            let pad_token_id: Int32
            let unk_token_id: Int32
        }
        let config = try JSONDecoder().decode(TokenizerConfig.self, from: configData)

        self.tokenizer = try WordPieceTokenizer(
            vocabURL: vocabURL,
            maxSequenceLength: config.max_sequence_length,
            doLowerCase: config.do_lower_case,
            clsTokenId: config.cls_token_id,
            sepTokenId: config.sep_token_id,
            padTokenId: config.pad_token_id,
            unkTokenId: config.unk_token_id
        )
        self.sequenceLength = config.max_sequence_length

        do {
            self.model = try MLModel(contentsOf: modelURL)
        } catch {
            throw EmbeddingError.modelLoadFailed(error.localizedDescription)
        }
    }

    /// Generates a 384-dim, L2-normalized embedding for on-device retrieval.
    func embed(_ text: String) throws -> [Float] {
        let (ids, mask) = tokenizer.encode(text)

        guard
            let idsArray = try? MLMultiArray(shape: [1, NSNumber(value: sequenceLength)], dataType: .int32),
            let maskArray = try? MLMultiArray(shape: [1, NSNumber(value: sequenceLength)], dataType: .int32)
        else {
            throw EmbeddingError.predictionFailed("Could not allocate MLMultiArray inputs.")
        }

        for i in 0..<sequenceLength {
            idsArray[i] = NSNumber(value: ids[i])
            maskArray[i] = NSNumber(value: mask[i])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": idsArray,
            "attention_mask": maskArray,
        ])

        let output: MLFeatureProvider
        do {
            output = try model.prediction(from: input)
        } catch {
            throw EmbeddingError.predictionFailed(error.localizedDescription)
        }

        guard let embeddingValue = output.featureValue(for: "embedding")?.multiArrayValue else {
            throw EmbeddingError.unexpectedOutputShape
        }

        var vector = [Float](repeating: 0, count: embeddingValue.count)
        for i in 0..<embeddingValue.count {
            vector[i] = embeddingValue[i].floatValue
        }
        return vector
    }
}
```

- [ ] **Step 4: Run the parity test after Checkpoint 1 lands**

Once Pablo's assets are in `WildernessEdge/Resources/` and `embedding_parity_fixtures.json` is in `WildernessEdgeTests/`, re-run `TextEmbeddingManagerTests`. Expected: PASS with dot product ≥ 0.999 for every fixture. If it fails, stop and check with Pablo whether `build_vector_db.py` and `export_embedder_coreml.py` were run with matching `--model` values (Global Constraints).

- [ ] **Step 5: Commit**

```bash
git add WildernessEdge/Core/TextEmbeddingManager.swift WildernessEdgeTests/TextEmbeddingManagerTests.swift
git commit -m "Add TextEmbeddingManager wrapping bundled CoreML embedder"
```

### Task B3: VectorRAGManager over protocols.db (TDD)

**Files:**
- Create: `WildernessEdge/Core/VectorRAGManager.swift`
- Create: `WildernessEdgeTests/VectorRAGManagerTests.swift`

**Interfaces:**
- Consumes: a `[Float]` query embedding (from `TextEmbeddingManager.embed(_:)`, Task B2).
- Produces: `VectorRAGManager.search(embedding: [Float], topK: Int, threshold: Float) -> RAGResult`, `VectorRAGManager.RetrievedChunk { citation: String, text: String, similarity: Float }`, `VectorRAGManager.RAGResult { case match([RetrievedChunk]), noConfidentMatch }` — consumed by Sachin's `LLMInferenceManager` prompt construction (Track D, Checkpoint 3).

- [ ] **Step 1: Build a tiny fixture SQLite database for TDD (independent of Pablo's real corpus)**

```bash
mkdir -p WildernessEdgeTests/Fixtures
python3 - <<'EOF'
import sqlite3, struct

conn = sqlite3.connect("WildernessEdgeTests/Fixtures/fixture-protocols.db")
conn.executescript("""
CREATE TABLE sources (id INTEGER PRIMARY KEY, filename TEXT, title TEXT, publisher TEXT, license TEXT, url TEXT, citation_prefix TEXT);
CREATE TABLE chunks (id INTEGER PRIMARY KEY, source_id INTEGER, section TEXT, page_start INTEGER, page_end INTEGER, citation TEXT, text TEXT, token_count INTEGER, embedding BLOB);
""")
conn.execute("INSERT INTO sources VALUES (1, 'fixture.pdf', 'Fixture Manual', 'Test', 'test', '', 'Fixture Manual')")

def vec(*floats):
    return struct.pack(f"<{len(floats)}f", *floats)

# 4-dim toy vectors for a fast, hand-computable test (real model is 384-dim; dimension is opaque to VectorRAGManager).
rows = [
    (1, 1, "1.1 Bleeding", 10, 10, "[Source: Fixture Manual, Section 1.1, p. 10]", "Apply direct pressure to the wound.", 8, vec(1.0, 0.0, 0.0, 0.0)),
    (2, 1, "2.1 Fractures", 20, 20, "[Source: Fixture Manual, Section 2.1, p. 20]", "Splint the limb in the position found.", 8, vec(0.0, 1.0, 0.0, 0.0)),
    (3, 1, "3.1 Hypothermia", 30, 30, "[Source: Fixture Manual, Section 3.1, p. 30]", "Insulate and passively rewarm the patient.", 8, vec(0.0, 0.0, 1.0, 0.0)),
]
conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
conn.commit()
conn.close()
EOF
```

Expected: `WildernessEdgeTests/Fixtures/fixture-protocols.db` created with 3 chunks and orthogonal 4-dim unit vectors (so cosine similarity is trivially 1.0 for an exact match, 0.0 for an orthogonal query).

- [ ] **Step 2: Write the failing tests**

```swift
// WildernessEdgeTests/VectorRAGManagerTests.swift
import XCTest

final class VectorRAGManagerTests: XCTestCase {
    func fixtureDBPath() -> String {
        Bundle(for: type(of: self)).path(forResource: "fixture-protocols", ofType: "db")!
    }

    func testExactMatchReturnsHighestSimilarityFirst() throws {
        let manager = try VectorRAGManager(databasePath: fixtureDBPath())
        let result = manager.search(embedding: [1.0, 0.0, 0.0, 0.0], topK: 3, threshold: 0.35)

        guard case .match(let chunks) = result else {
            return XCTFail("Expected a match")
        }
        XCTAssertEqual(chunks.first?.text, "Apply direct pressure to the wound.")
        XCTAssertEqual(chunks.first?.similarity ?? 0, 1.0, accuracy: 0.0001)
        XCTAssertEqual(chunks.first?.citation, "[Source: Fixture Manual, Section 1.1, p. 10]")
    }

    func testTopKLimitsResultCount() throws {
        let manager = try VectorRAGManager(databasePath: fixtureDBPath())
        let result = manager.search(embedding: [0.5, 0.5, 0.5, 0.0], topK: 2, threshold: -1.0)
        guard case .match(let chunks) = result else {
            return XCTFail("Expected a match")
        }
        XCTAssertEqual(chunks.count, 2)
    }

    func testBelowThresholdReturnsNoConfidentMatch() throws {
        let manager = try VectorRAGManager(databasePath: fixtureDBPath())
        // Orthogonal to all 3 stored vectors -> similarity 0.0 for all, below any positive threshold.
        let result = manager.search(embedding: [0.0, 0.0, 0.0, 1.0], topK: 3, threshold: 0.35)
        XCTAssertEqual(result, .noConfidentMatch)
    }

    func testOpeningMissingDatabaseThrows() {
        XCTAssertThrowsError(try VectorRAGManager(databasePath: "/nonexistent/path.db"))
    }
}

extension VectorRAGManager.RAGResult: Equatable {
    static func == (lhs: VectorRAGManager.RAGResult, rhs: VectorRAGManager.RAGResult) -> Bool {
        switch (lhs, rhs) {
        case (.noConfidentMatch, .noConfidentMatch): return true
        case (.match(let a), .match(let b)): return a.map(\.text) == b.map(\.text)
        default: return false
        }
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Expected: FAIL — `VectorRAGManager` does not exist.

- [ ] **Step 4: Implement VectorRAGManager**

```swift
// WildernessEdge/Core/VectorRAGManager.swift
import Accelerate
import Foundation
import SQLite3

/// SIMD-accelerated vector search over the bundled, read-only `protocols.db`.
/// Uses the raw SQLite3 C API (no third-party wrapper) per AGENTS.md.
final class VectorRAGManager {
    struct RetrievedChunk {
        let citation: String
        let text: String
        let similarity: Float
    }

    enum RAGResult {
        case match([RetrievedChunk])
        case noConfidentMatch
    }

    enum RAGError: LocalizedError {
        case openFailed(String)
        case queryFailed(String)

        var errorDescription: String? {
            switch self {
            case .openFailed(let message): return "Could not open protocols.db: \(message)"
            case .queryFailed(let message): return "Query against protocols.db failed: \(message)"
            }
        }
    }

    private var database: OpaquePointer?

    init(databasePath: String) throws {
        var handle: OpaquePointer?
        let result = sqlite3_open_v2(databasePath, &handle, SQLITE_OPEN_READONLY, nil)
        guard result == SQLITE_OK, let handle else {
            let message = handle.map { String(cString: sqlite3_errmsg($0)) } ?? "unknown error"
            if let handle { sqlite3_close(handle) }
            throw RAGError.openFailed(message)
        }
        self.database = handle
    }

    deinit {
        if let database { sqlite3_close(database) }
    }

    /// Returns the top-K chunks by cosine similarity, or `.noConfidentMatch` if the best
    /// score is below `threshold`. Both `embedding` and stored vectors must already be
    /// L2-normalized so a plain dot product equals cosine similarity.
    func search(embedding: [Float], topK: Int, threshold: Float) -> RAGResult {
        guard let database else { return .noConfidentMatch }

        var statement: OpaquePointer?
        let sql = "SELECT citation, text, embedding FROM chunks"
        guard sqlite3_prepare_v2(database, sql, -1, &statement, nil) == SQLITE_OK, let statement else {
            return .noConfidentMatch
        }
        defer { sqlite3_finalize(statement) }

        var scored: [RetrievedChunk] = []

        while sqlite3_step(statement) == SQLITE_ROW {
            guard
                let citationCString = sqlite3_column_text(statement, 0),
                let textCString = sqlite3_column_text(statement, 1),
                let blobPointer = sqlite3_column_blob(statement, 2)
            else { continue }

            let citation = String(cString: citationCString)
            let text = String(cString: textCString)
            let byteCount = Int(sqlite3_column_bytes(statement, 2))
            let floatCount = byteCount / MemoryLayout<Float>.size

            guard floatCount == embedding.count else { continue }

            var storedVector = [Float](repeating: 0, count: floatCount)
            storedVector.withUnsafeMutableBytes { destination in
                destination.copyMemory(from: UnsafeRawBufferPointer(start: blobPointer, count: byteCount))
            }

            var similarity: Float = 0
            vDSP_dotpr(embedding, 1, storedVector, 1, &similarity, vDSP_Length(embedding.count))

            scored.append(RetrievedChunk(citation: citation, text: text, similarity: similarity))
        }

        scored.sort { $0.similarity > $1.similarity }
        let topResults = Array(scored.prefix(topK))

        guard let best = topResults.first, best.similarity >= threshold else {
            return .noConfidentMatch
        }
        return .match(topResults)
    }
}
```

- [ ] **Step 5: Add the fixture DB and new test file to the test target sources in `project.yml`**

`project.yml`'s `WildernessEdgeTests` target already includes `path: WildernessEdgeTests` as a source, which covers the new test file and `Fixtures/` directory automatically (no `project.yml` edit needed — verify by confirming `WildernessEdgeTests/Fixtures/fixture-protocols.db` and `fixture-vocab.txt` are plain files under that existing source path). Regenerate the Xcode project:

```bash
xcodegen generate
```

Expected: no errors; new files appear in the `WildernessEdgeTests` target in Xcode.

- [ ] **Step 6: Run to verify it passes**

Expected: all 4 `VectorRAGManagerTests` PASS.

- [ ] **Step 7: Commit**

```bash
git add WildernessEdge/Core/VectorRAGManager.swift WildernessEdgeTests/VectorRAGManagerTests.swift \
  WildernessEdgeTests/Fixtures/fixture-protocols.db
git commit -m "Add VectorRAGManager with SQLite3+Accelerate cosine search"
```

### Task B4: Swap fixtures for real assets (Checkpoint 1)

**Files:**
- Modify: `WildernessEdgeTests/WordPieceTokenizerTests.swift` (optional: leave fixture tests as-is, they're still valid unit tests)
- Verify: `WildernessEdgeTests/TextEmbeddingManagerTests.swift`

- [ ] **Step 1: Confirm Pablo's real assets are present**

```bash
ls WildernessEdge/Resources/protocols.db WildernessEdge/Resources/query-embedder.mlpackage \
   WildernessEdge/Resources/query-embedder-vocab.txt WildernessEdge/Resources/query-embedder-tokenizer.json \
   WildernessEdgeTests/embedding_parity_fixtures.json
```

Expected: all 5 files exist (from Task A2).

- [ ] **Step 2: Re-run TextEmbeddingManagerTests against real assets**

Expected: `testEmbeddingMatchesParityFixtures` now runs (no longer `XCTSkip`) and PASSes with similarity ≥ 0.999 for every fixture sentence.

- [ ] **Step 3: Manually exercise VectorRAGManager against the real protocols.db**

Write a quick scratch call (in a SwiftUI Preview or a throwaway `#if DEBUG` block in `ContentView`) instantiating `VectorRAGManager(databasePath:)` pointed at the real bundled `protocols.db` path, embed a real query like "severe bleeding from the thigh" via `TextEmbeddingManager`, and confirm `search(embedding:topK:threshold:)` returns a `.match` whose top citation is one of the 3 real sources (ATP 4-02.11, TCCC Handbook v5, or NASEMSO Guidelines).

- [ ] **Step 4: Commit any adjustments**

```bash
git add -A
git commit -m "Validate RAG pipeline against real corpus assets"
```

(Only commit if something actually changed — this task may be pure verification.)

---

## Track C — Native I/O, SwiftUI Shell & Safety Enforcement (Owner: Daniel)

### Task C1: Verify existing native managers on the physical device

**Files:** none (manual verification of existing `SpeechManager.swift`, `TTSManager.swift`, `CameraManager.swift`)

- [ ] **Step 1: Build and sideload onto the iPhone 16 Plus**

```bash
xcodegen generate
open WildernessEdge.xcodeproj
```

In Xcode: select the physical device as the run destination, set a development team under Signing & Capabilities (per `PLAN.md` Part 1 §3 — Developer Mode must already be enabled on the phone and the Mac trusted), and Run.

- [ ] **Step 2: Verify on-device speech recognition with the network off**

Enable Airplane Mode on the phone. Tap "Start Listening" in the current `ContentView` scaffold and speak. Expected: `speechManager.transcript` updates live in the UI. If instead the error state shows `.onDeviceUnavailable`, check Settings → General → Keyboard → [your language] → Enable Dictation, since on-device speech models must be downloaded per-locale.

- [ ] **Step 3: Verify TTS audibility**

Tap "Speak Filtered" with an empty transcript (triggers the built-in bait-phrase demo string in `ContentView.speakFilteredDemo()`). Expected: audible speech plays, and `safetyNote` shows "SafetyFilter intercepted diagnostic/prescriptive language."

- [ ] **Step 4: Verify camera pre-warm and snapshot timing**

Confirm the camera permission prompt appears on first launch, then tap "Capture Snapshot" and confirm a near-instant preview image appears (the session should already be pre-warmed via `.task { await cameraManager.prewarm() }`).

- [ ] **Step 5: No code changes expected** — if all three pass, this task is verification-only. If something fails, file the specific failure (error state shown, exact steps) before touching `SpeechManager`/`TTSManager`/`CameraManager` code, and fix root cause rather than papering over it.

### Task C2: Build EmergencyButtonView

**Files:**
- Create: `WildernessEdge/Views/EmergencyButtonView.swift`

**Interfaces:**
- Consumes: an `AppState` enum (defined in Task C4) and press/release callbacks.
- Produces: `EmergencyButtonView(state: AppState, onPressDown: () -> Void, onPressUp: () -> Void)`, a SwiftUI `View` consumed by `ContentView` (Task C4).

- [ ] **Step 1: Implement the button**

```swift
// WildernessEdge/Views/EmergencyButtonView.swift
import SwiftUI

/// Large, high-contrast, circular push-to-talk button with per-state visual treatment.
struct EmergencyButtonView: View {
    let state: AppState
    let onPressDown: () -> Void
    let onPressUp: () -> Void

    @GestureState private var isPressing = false

    var body: some View {
        Circle()
            .fill(fillColor)
            .frame(width: 180, height: 180)
            .overlay(
                Text(label)
                    .font(.title2.bold())
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
                    .padding()
            )
            .scaleEffect(isPressing ? 0.95 : 1.0)
            .animation(.easeOut(duration: 0.15), value: isPressing)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .updating($isPressing) { _, pressing, _ in pressing = true }
                    .onChanged { _ in if !isPressing { onPressDown() } }
                    .onEnded { _ in onPressUp() }
            )
            .accessibilityLabel(label)
    }

    private var fillColor: Color {
        switch state {
        case .idle: return .blue
        case .listening: return .red
        case .processing: return .orange
        case .speaking: return .green
        case .error: return .gray
        }
    }

    private var label: String {
        switch state {
        case .idle: return "Hold to Ask"
        case .listening: return "Listening…"
        case .processing: return "Processing…"
        case .speaking: return "Speaking…"
        case .error: return "Error — Tap for Details"
        }
    }
}

#Preview {
    EmergencyButtonView(state: .idle, onPressDown: {}, onPressUp: {})
}
```

- [ ] **Step 2: Verify visually in Xcode Preview**

Open the Preview canvas for `EmergencyButtonView.swift`. Expected: a blue circular button reading "Hold to Ask". Temporarily change `state:` to `.listening`, `.processing`, `.speaking`, `.error("test")` in the `#Preview` to confirm each color/label renders correctly, then revert to `.idle`.

- [ ] **Step 3: Commit**

```bash
git add WildernessEdge/Views/EmergencyButtonView.swift
git commit -m "Add EmergencyButtonView push-to-talk button"
```

### Task C3: Build SubtitleCardView

**Files:**
- Create: `WildernessEdge/Views/SubtitleCardView.swift`

**Interfaces:**
- Produces: `SubtitleCardView(citation: String?, checklistText: String, isError: Bool)`, consumed by `ContentView` (Task C4).

- [ ] **Step 1: Implement the card**

```swift
// WildernessEdge/Views/SubtitleCardView.swift
import SwiftUI

/// High-contrast overlay card displaying the active source citation and spoken checklist text.
struct SubtitleCardView: View {
    let citation: String?
    let checklistText: String
    let isError: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let citation {
                Text(citation)
                    .font(.footnote.bold())
                    .foregroundStyle(isError ? .red : .secondary)
            }
            Text(checklistText)
                .font(.body)
                .foregroundStyle(isError ? .red : .primary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(isError ? Color.red.opacity(0.1) : Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(isError ? Color.red : Color.clear, lineWidth: 2)
        )
    }
}

#Preview {
    SubtitleCardView(
        citation: "[Source: NASEMSO National Model EMS Clinical Guidelines v3.0, Extremity Trauma, p. 128]",
        checklistText: "1. Expose and inspect the injured extremity.\n2. Check distal pulse, motor, and sensory function.",
        isError: false
    )
}
```

- [ ] **Step 2: Verify visually in Xcode Preview**

Confirm the citation renders in bold footnote style above the checklist body text, and that toggling `isError: true` switches to the red-bordered error treatment.

- [ ] **Step 3: Commit**

```bash
git add WildernessEdge/Views/SubtitleCardView.swift
git commit -m "Add SubtitleCardView citation/checklist card"
```

### Task C4: Build the ContentView state machine (stubbed RAG + LLM)

**Files:**
- Modify: `WildernessEdge/Views/ContentView.swift` (replace the Phase-1 scaffold entirely)

**Interfaces:**
- Defines: `enum AppState { case idle, listening, processing, speaking, error(String) }` — consumed by `EmergencyButtonView` (Task C2) and `SubtitleCardView` (Task C3).
- Consumes (stubbed until Checkpoint 4): a closure `runInferencePipeline(transcript: String, snapshot: UIImage?) async -> (citation: String?, checklistText: String)` — Sachin swaps this stub for a real call into `LLMInferenceManager` at Checkpoint 4.

- [ ] **Step 1: Define AppState and rewrite ContentView with a stub pipeline**

```swift
// WildernessEdge/Views/ContentView.swift
import SwiftUI

enum AppState: Equatable {
    case idle
    case listening
    case processing
    case speaking
    case error(String)
}

/// Full push-to-talk pipeline: Button Down (snapshot + record) -> Button Up (STT -> Embed ->
/// RAG -> Gemma via LiteRT-LM -> Safety Filter -> TTS). The RAG/LLM step is stubbed here and
/// wired to the real LLMInferenceManager at Checkpoint 4.
struct ContentView: View {
    @StateObject private var speechManager = SpeechManager()
    @StateObject private var ttsManager = TTSManager()
    @StateObject private var cameraManager = CameraManager()

    @State private var appState: AppState = .idle
    @State private var citation: String?
    @State private var checklistText: String = ""

    /// Checkpoint 4 replaces this stub with a real call into LLMInferenceManager +
    /// VectorRAGManager. Signature must not change without updating both call sites.
    var runInferencePipeline: (String, UIImage?) async -> (citation: String?, checklistText: String) = { transcript, _ in
        (nil, "STUB: would retrieve protocol context for \"\(transcript)\" and query Gemma 4 E4B here.")
    }

    var body: some View {
        VStack(spacing: 24) {
            Text("Wilderness Edge")
                .font(.largeTitle.bold())

            SubtitleCardView(
                citation: citation,
                checklistText: displayText,
                isError: isErrorState
            )

            Spacer()

            EmergencyButtonView(
                state: appState,
                onPressDown: handlePressDown,
                onPressUp: handlePressUp
            )

            Spacer()
        }
        .padding()
        .task {
            await cameraManager.prewarm()
        }
        .onDisappear {
            cameraManager.shutdown()
        }
    }

    private var isErrorState: Bool {
        if case .error = appState { return true }
        return false
    }

    private var displayText: String {
        if case .error(let message) = appState { return message }
        return checklistText.isEmpty ? "Hold the button and ask a question." : checklistText
    }

    private func handlePressDown() {
        guard appState == .idle || isErrorState else { return }
        appState = .listening
        Task { await cameraManager.captureSnapshot() }
        speechManager.startListening()
    }

    private func handlePressUp() {
        guard appState == .listening else { return }
        speechManager.stopListening()

        Task {
            appState = .processing

            // Give the recognizer a brief moment to publish its final transcript after stop.
            try? await Task.sleep(nanoseconds: 300_000_000)

            if let speechError = speechManager.error {
                appState = .error(speechError.localizedDescription)
                return
            }

            let transcript = speechManager.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !transcript.isEmpty else {
                appState = .error("Didn't catch that — hold the button and try again.")
                return
            }

            let (resultCitation, resultText) = await runInferencePipeline(transcript, cameraManager.latestSnapshot)
            let filtered = SafetyFilter.sanitize(resultText)

            citation = resultCitation
            checklistText = filtered.text
            appState = .speaking
            ttsManager.speak(filtered.text)

            // Return to idle once speech finishes (polled since AVSpeechSynthesizerDelegate
            // already publishes isSpeaking on TTSManager).
            while ttsManager.isSpeaking {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            appState = .idle
        }
    }
}

#Preview {
    ContentView()
}
```

- [ ] **Step 2: Manually verify the stub pipeline end-to-end in Simulator**

Run in Simulator (or device). Press and hold the button, say something, release. Expected: state visibly transitions Idle → Listening → Processing → Speaking → Idle, the subtitle card shows the stub text, and TTS speaks it. Force an error by denying microphone permission once and confirm the button turns gray with the error message displayed and spoken-equivalent text shown (TTS itself won't speak the error unless you choose to wire that — displaying it is sufficient for this stub).

- [ ] **Step 3: Commit**

```bash
git add WildernessEdge/Views/ContentView.swift
git commit -m "Add full push-to-talk state machine with stubbed inference pipeline"
```

- [ ] **Step 4: Notify Sachin the stub closure is ready to swap (Checkpoint 4 dependency satisfied from this side)**

Message Sachin that `ContentView.runInferencePipeline` has the exact signature `(String, UIImage?) async -> (citation: String?, checklistText: String)` he needs to implement against.

---

## Track D — LiteRT-LM Integration & Orchestration (Owner: Sachin)

### Task D1: Add the LiteRT-LM SPM package to project.yml

**Files:**
- Modify: `project.yml`

**Interfaces:**
- Produces: the `LiteRTLM` SPM product linked into the `WildernessEdge` target, required by Task D2.

- [ ] **Step 1: Edit project.yml to add the package**

Current `project.yml` has this comment block just above `targets:` (read it first to confirm line context):

```yaml
# LiteRT-LM SPM is added in Phase 3 (LLMInferenceManager). Declaring it here would
# force native LLM linkage during Phase 1 audio/safety verification builds.
# Package URL when wiring Phase 3: https://github.com/google-ai-edge/LiteRT-LM (product: LiteRTLM, from: 0.12.0)
targets:
```

Replace it with:

```yaml
packages:
  LiteRTLM:
    url: https://github.com/google-ai-edge/LiteRT-LM
    from: 0.12.0
targets:
```

Then, inside the `WildernessEdge` target's `settings:` block (after `configs:`), add a `dependencies:` key at the target level (sibling of `sources:`/`settings:`):

```yaml
  WildernessEdge:
    type: application
    platform: iOS
    dependencies:
      - package: LiteRTLM
        product: LiteRTLM
    sources:
      - path: WildernessEdge
        excludes:
          - Resources/**
```

(Insert `dependencies:` right after `platform: iOS` and before the existing `sources:` key — do not duplicate the `sources:` key.)

- [ ] **Step 2: Regenerate and verify the package resolves**

```bash
xcodegen generate
open WildernessEdge.xcodeproj
```

In Xcode, wait for Swift Package Manager to resolve `LiteRT-LM`. Expected: no red errors in the Package Dependencies section; `import LiteRTLM` becomes available.

- [ ] **Step 3: Commit**

```bash
git add project.yml
git commit -m "Add LiteRT-LM SPM package dependency"
```

### Task D2: LLMInferenceManager with a stubbed model path

**Files:**
- Create: `WildernessEdge/Core/LLMInferenceManager.swift`

**Interfaces:**
- Consumes (at Checkpoint 3): `VectorRAGManager.RAGResult` from Task B3.
- Produces: `LLMInferenceManager.initialize() async throws`, `LLMInferenceManager.generate(transcript: String, ragResult: VectorRAGManager.RAGResult?, image: UIImage?) async throws -> String` — consumed by `ContentView.runInferencePipeline` (Task C4, wired at Checkpoint 4).

**Before starting this task:** the exact type/method names below (`EngineConfig`, `Engine`, `Conversation`, `Content`, `Message`, `.imageData`, `.gpu()`) are Sachin's best-available reconstruction from PLAN.md's prose description of the LiteRT-LM Swift API, not verified against the real package (this plan was written without SPM package access). **First step of this task is to open the resolved `LiteRT-LM` package in Xcode (after Task D1) and read its actual public API** — jump to definition on `Engine`/`Conversation`/`Content`/`Message`, or check the package's README/examples on GitHub. Adjust the code below to match whatever the real API surface is; the logic (build a system instruction + context block + transcript + optional image, stream tokens, concatenate) stays the same regardless of exact type names.

- [ ] **Step 1: Implement the manager against the LiteRT-LM Swift API**

```swift
// WildernessEdge/Core/LLMInferenceManager.swift
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

    /// Must be called once at app startup, before the first query. Surfaces a blocking
    /// startup error rather than allowing the app into a broken push-to-talk loop.
    func initialize(bundle: Bundle = .main) async {
        guard let modelURL = bundle.url(forResource: "gemma-4-E4B-it", withExtension: "litertlm") else {
            initializationError = .modelAssetMissing
            return
        }

        do {
            let config = EngineConfig(
                modelPath: modelURL.path,
                visionBackend: .gpu()
            )
            let engine = try Engine(config: config)
            self.engine = engine
            self.conversation = try engine.createConversation()
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

        let systemInstruction = """
        You are a non-diagnostic, non-prescriptive field-protocol assistant. Only present \
        retrieved checklist steps with their citation. Never state a diagnosis or a drug dose. \
        If no protocol context is provided below, say plainly that no matching protocol was \
        found instead of guessing.
        """

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
        contents.append(.text("\(systemInstruction)\n\nRetrieved context:\n\(contextBlock)\n\nUser question: \(transcript)"))

        let message = Message(contents: contents)

        do {
            var response = ""
            for try await token in try conversation.sendMessage(message) {
                response += token
            }
            return response
        } catch {
            throw LLMError.generationFailed(error.localizedDescription)
        }
    }
}
```

- [ ] **Step 2: Manually verify initialization failure surfaces cleanly (before the real model bundle exists)**

In a Simulator run, call `await LLMInferenceManager().initialize()` from a temporary `#if DEBUG` hook in `ContentView.task` and print `initializationError`. Expected (pre-Checkpoint 2): `.modelAssetMissing`, since `gemma-4-E4B-it.litertlm` isn't in `Resources/` yet — this confirms the fail-closed path works before wiring the real bundle.

- [ ] **Step 3: Commit**

```bash
git add WildernessEdge/Core/LLMInferenceManager.swift
git commit -m "Add LLMInferenceManager wrapping LiteRT-LM Engine/Conversation"
```

### Task D3: Swap in the real model bundle (Checkpoint 2)

**Files:** none (verification against Task D2's existing code)

- [ ] **Step 1: Confirm the model bundle is present**

```bash
ls -lh WildernessEdge/Resources/gemma-4-E4B-it.litertlm
```

Expected: file exists (from Task A3).

- [ ] **Step 2: Run initialize() on the physical device (not Simulator — LiteRT-LM's GPU backend needs real Metal)**

Expected: `isReady == true`, `initializationError == nil`. If it fails, check the exact filename matches `gemma-4-E4B-it.litertlm` (case-sensitive) and that `visionBackend: .gpu()` doesn't crash on this device — fall back to `.cpu()` in `EngineConfig` if GPU backend init fails, and note the tradeoff for the writeup.

- [ ] **Step 3: Run a single manual multimodal generation call**

Temporarily call `generate(transcript: "How do I treat a bleeding wound?", ragResult: nil, image: cameraManager.latestSnapshot)` and print the result. Expected: coherent text response referencing the "NO_MATCHING_PROTOCOL_FOUND" instruction (since RAG isn't wired yet) — confirms the model and multimodal input path work before Checkpoint 3.

- [ ] **Step 4: Nothing to commit** (verification only, no code changed unless the GPU→CPU fallback was needed — if so, commit that change).

### Task D4: Wire real VectorRAGManager output into generate() (Checkpoint 3)

**Files:**
- Modify: `WildernessEdge/Core/LLMInferenceManager.swift` (no signature change — verification that Task B3's real output flows through Task D2's existing `ragResult` parameter)

- [ ] **Step 1: Confirm Vaibhav's VectorRAGManager is complete**

Check that `WildernessEdgeTests/VectorRAGManagerTests.swift` passes and `WildernessEdge/Resources/protocols.db` is the real corpus (from Checkpoint 1/Task B4).

- [ ] **Step 2: Manually call the real pipeline end-to-end (still outside ContentView)**

```swift
let embedder = try TextEmbeddingManager()
let rag = try VectorRAGManager(databasePath: Bundle.main.path(forResource: "protocols", ofType: "db")!)
let queryEmbedding = try embedder.embed("severe bleeding from the thigh")
let ragResult = rag.search(embedding: queryEmbedding, topK: 3, threshold: 0.35)
let response = try await llmManager.generate(transcript: "How do I treat severe bleeding?", ragResult: ragResult, image: nil)
print(response)
```

Expected: a response that cites one of the 3 real sources and gives checklist-style steps, not a fabricated diagnosis.

- [ ] **Step 3: No commit needed unless a bug surfaced** — if `VectorRAGManager`'s output type didn't match what `generate()` expects, fix the mismatch and commit.

### Task D5: Wire the full pipeline into ContentView (Checkpoint 4)

**Files:**
- Modify: `WildernessEdge/Views/ContentView.swift`

**Interfaces:**
- Replaces: `ContentView.runInferencePipeline`'s default stub closure with a real implementation. Signature `(String, UIImage?) async -> (citation: String?, checklistText: String)` must not change.

- [ ] **Step 1: Instantiate the real managers in ContentView and replace the stub**

```swift
// In ContentView, add:
@StateObject private var llmManager = LLMInferenceManager()
private let embedder = try? TextEmbeddingManager()
private let ragManager = try? VectorRAGManager(
    databasePath: Bundle.main.path(forResource: "protocols", ofType: "db") ?? ""
)
```

Replace the default `runInferencePipeline` value with:

```swift
var runInferencePipeline: (String, UIImage?) async -> (citation: String?, checklistText: String) = { _, _ in
    (nil, "") // overwritten below in .task
}
```

And in `.task`, after `await cameraManager.prewarm()`, add:

```swift
await llmManager.initialize()
runInferencePipeline = { [ragManager, embedder, llmManager] transcript, image in
    guard let embedder, let ragManager else {
        return (nil, "Retrieval system unavailable.")
    }
    do {
        let queryEmbedding = try embedder.embed(transcript)
        let ragResult = ragManager.search(embedding: queryEmbedding, topK: 3, threshold: 0.35)
        let responseText = try await llmManager.generate(transcript: transcript, ragResult: ragResult, image: image)
        let citation: String? = {
            if case .match(let chunks) = ragResult { return chunks.first?.citation }
            return nil
        }()
        return (citation, responseText)
    } catch {
        return (nil, "Inference failed: \(error.localizedDescription)")
    }
}
```

Also handle `llmManager.initializationError` in `.task`: if non-nil after `initialize()`, set `appState = .error(...)` immediately so a broken model asset surfaces at launch, not on first query (per `AGENTS.md` guideline 7 / PLAN.md Phase 3 verification criteria).

- [ ] **Step 2: Manually verify the fully-wired pipeline on the physical device in Airplane Mode**

Enable Airplane Mode. Press and hold the button, ask "How do I splint a broken arm?", release. Expected: Listening → Processing → Speaking states visible, the subtitle card shows a real citation (e.g. TCCC Handbook v5) and checklist text, and TTS speaks it aloud — end to end, no network.

- [ ] **Step 3: Commit**

```bash
git add WildernessEdge/Views/ContentView.swift
git commit -m "Wire full RAG + LiteRT-LM pipeline into ContentView push-to-talk flow"
```

---

## Final Validation (All 4, before demo)

### Task E1: Airplane Mode demo run-through, twice

**Files:** none

- [ ] **Step 1: Full airplane-mode isolation**

On the physical iPhone 16 Plus: Settings → Airplane Mode ON, then manually re-verify Wi-Fi and Bluetooth are also off (Airplane Mode can leave Wi-Fi toggled on in some iOS versions).

- [ ] **Step 2: Run the demo script once**

Press-and-hold, ask a wilderness/first-aid question (optionally point the camera at something relevant, e.g. a bandage or splint material), release, and confirm: transcript captured → citation + checklist shown → spoken aloud → SafetyFilter did not need to intercept anything (benign checklist text passes unmodified per `SafetyFilterTests`).

- [ ] **Step 3: Run it again with a bait question**

Ask something engineered to bait a diagnosis, e.g. "What's wrong with my arm, is it broken?" Confirm the response either cites a checklist (assessment steps) or explicitly says no diagnosis is given — and if the raw LLM output ever contained diagnostic language, confirm `SafetyFilter.sanitize(_:)` caught it (check `wasModified`/`matchedPatterns` via a debug log).

- [ ] **Step 4: Run the full automated test suite one final time**

```bash
xcodebuild test -scheme WildernessEdgeTests -destination 'platform=iOS Simulator,name=iPhone 15'
```

Expected: `SafetyFilterTests`, `WordPieceTokenizerTests`, `TextEmbeddingManagerTests`, `VectorRAGManagerTests` all PASS.

- [ ] **Step 5: No commit** — this is a manual/CI verification task, not a code change.

### Task E2: Finalize and submit the Kaggle Writeup

**Files:**
- Modify: `docs/kaggle-writeup-draft.md`

- [ ] **Step 1: Finalize writeup content**

Confirm under 1,500 words, Track selected (On-Device Private Health), architecture + Gemma-usage sections accurate to what actually shipped (not the original 5-phase plan — reflect the LoRA cut and E4B swap honestly as a sprint engineering decision).

- [ ] **Step 2: Attach the public repo link and live demo**

Push the repo to a public GitHub remote if not already public. Record a screen capture of the Airplane Mode demo (Task E1) as the live-demo artifact, or link a clonable notebook if applicable.

- [ ] **Step 3: Submit on Kaggle**

Create the Writeup in the competition's Writeups tab, save, attach links under Project Links in Attachments, then click Submit before the deadline.

- [ ] **Step 4: Commit the final writeup draft**

```bash
git add docs/kaggle-writeup-draft.md
git commit -m "Finalize Kaggle writeup for submission"
```
