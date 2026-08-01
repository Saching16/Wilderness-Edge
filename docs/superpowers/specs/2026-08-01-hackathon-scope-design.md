# Wilderness Edge — Hackathon Scope Design

**Event:** Build with Gemma NYC (On-Device AI for Healthcare)
**Deadline:** ~18 hours from design time (2026-08-01)
**Track:** On-Device Private Health (primary), Voice for Care (secondary framing)
**Team:** Pablo Beaus Iranzo, Sachin Ganpule, Daniel David, Vaibhav Chaudhari

## Purpose

PLAN.md describes a 5-phase, multi-day native iOS build for Wilderness Edge, a
100% air-gapped, voice-first, vision-assisted emergency protocol assistant.
That plan assumes a custom LoRA fine-tune, strict device memory
verification, and extensive stress testing — none of which fit an 18-hour
sprint. This document trims PLAN.md to what the team can actually build,
test, and demo before the deadline, and assigns the trimmed work across the
4-person team by background fit.

## What changes from PLAN.md / AGENTS.md

- **Model swap:** Every reference to Gemma 4 **E2B** becomes Gemma 4 **E4B**.
  Use the prebuilt `litert-community/gemma-4-E4B-it-litert-lm` bundle —
  no custom weights, no gating friction.
- **Drop the LoRA fine-tune entirely.** `train_lora_colab.py` and the
  clinical-review gate on `build_training_data.py` output are cut from
  scope. Grounding comes from RAG context + prompting only. This is a
  legitimate simplification, not a corner cut: Gemma 4 is still the core
  inference engine, multimodal and RAG-grounded.
- **Relax memory guardrails to a soft check.** AGENTS.md's 6GB
  device-target / 2.8GB inference-ceiling verification criteria are
  dropped as hard gates. The team is sideloading to a physical
  **iPhone 16 Plus**, which has ample headroom — keep the "Increased
  Memory Limit" entitlement (it's free) but do not block progress on
  measuring or optimizing footprint. Avoid reckless bloat, but no formal
  measurement step this sprint.
- **Reduce Phase 5 stress testing** from "20 consecutive query loops, full
  error-trigger matrix on device" to "demo script runs reliably twice in
  Airplane Mode."
- **Keep:** full on-device vector RAG (CoreML embedder + SQLite
  `protocols.db`), camera/vision multimodal input, `SafetyFilter`
  fail-closed behavior, Airplane Mode as the literal demo condition (not
  just a claim in the writeup).
- **Corpus scope:** wilderness first aid + general first aid, restricted to
  sources already verified in `sources.manifest.json` (skip NOLS —
  confirmed unlicensed for ingestion per `SOURCES.md`).
- **`project.yml` change:** add the `google-ai-edge/LiteRT-LM` SPM package
  now, rather than deferring it to "Phase 3" as PLAN.md currently states —
  the LLM integration owner needs it from hour zero to work in parallel.

## Architecture & data flow (unchanged from AGENTS.md)

```
Button-down  → camera snapshot + audio recording start
Button-up    → STT transcript
             → on-device CoreML embed
             → vector RAG search over protocols.db
             → multimodal Gemma 4 E4B call via LiteRT-LM
               (image + RAG context + transcript)
             → SafetyFilter regex pass
             → TTS playback
```

"No confident RAG match" is a normal, honestly-spoken result — the model is
told explicitly not to fabricate context — not an error state. Defined
error states (empty/failed transcription, on-device STT unavailable, LLM
init failure) remain fail-closed with no network fallback, per AGENTS.md's
non-negotiable guardrails.

## Task breakdown (pipeline-order, one owner per stage)

Each person can start immediately — later stages consume earlier stages'
outputs as they land, but begin against stub/fixture data rather than
blocking.

### Pablo — Offline assets & corpus curation
- Trim `sources.manifest.json` / `SOURCES.md` to already-licensed wilderness
  + general first-aid sources.
- Run `fetch_sources.py` → `build_vector_db.py` → `export_embedder_coreml.py`.
- Fetch the prebuilt `gemma-4-E4B-it-litert-lm` bundle.
- Deliverables into `WildernessEdge/Resources/`: `protocols.db`,
  `query-embedder.mlpackage` + tokenizer assets, `.litertlm` file.
- Co-owns the Kaggle writeup with Daniel (drafted incrementally, not at the
  end).
- **Why this fit:** healthcare ML pipeline background (breast cancer
  detection, EU AI regulatory/interpretability work), research writing
  experience (ICASSP 2024 co-author).

### Vaibhav — On-device vector RAG engine
- Build `Core/WordPieceTokenizer.swift`, `Core/TextEmbeddingManager.swift`,
  `Core/VectorRAGManager.swift` per PLAN.md Phase 2 spec (raw SQLite3 C API,
  Accelerate `vDSP_dotpr` cosine similarity, "no confident match" path).
- Unit test with `VectorRAGManagerTests.swift`.
- Starts against stub/fixture embeddings; swaps in Pablo's real
  `protocols.db` as it lands.
- **Why this fit:** built VecFast, a concurrent HNSW vector database
  (SIFT1M benchmarks, recall/throughput tuning) — the closest real-world
  analog to this exact component.

### Daniel — Native I/O, SwiftUI shell, SafetyFilter enforcement
- Verify/finish `SpeechManager`, `TTSManager`, `CameraManager` (already
  exist from commit `439e0a3`) on the real sideloaded iPhone 16 Plus.
- Build the push-to-talk state machine in `Views/ContentView.swift` per
  PLAN.md Phase 4.
- Enforce SafetyFilter fail-closed behavior and the defined error-state
  transitions.
- Co-owns the Kaggle writeup with Pablo.
- **Why this fit:** healthcare compliance background (HIPAA/PHI, FHIR at
  Rhino), guardrail/audit-trail experience (Clawdstrike), full-stack rapid
  prototyping experience for fast SwiftUI ramp-up, documentation authorship
  experience.

### Sachin — LiteRT-LM integration & orchestration
- Add `google-ai-edge/LiteRT-LM` SPM package to `project.yml` now.
- Build `Core/LLMInferenceManager.swift`: `EngineConfig` pointing at the
  bundled `.litertlm`, multimodal `Message` construction (image + RAG
  context + transcript), async streaming via `conversation.sendMessage`.
- Implement the "no confident match → honest deflection" prompt path.
- Starts immediately against stubbed RAG output + placeholder text; swaps
  in Vaibhav's and Daniel's real outputs as they land.
- **Why this fit:** built a production hybrid RAG pipeline (Azure AI
  Search, 95% recall@1) and LLM/embedding infrastructure work (Hawl) —
  most direct match for wiring retrieval + multimodal context into a model
  call.

## Testing bar for this sprint

- `SafetyFilterTests` (existing) and `VectorRAGManagerTests` (new) pass in
  Simulator/CI.
- Manual Airplane Mode run-through on the physical iPhone 16 Plus stands in
  for PLAN.md Phase 5's full stress suite — demo script runs reliably
  twice, not 20 consecutive loops.

## Demo & writeup

- **Demo:** live Airplane Mode run on the sideloaded iPhone. Press button,
  ask a wilderness/first-aid question (optionally with camera pointed at
  something relevant), get a spoken checklist + source citation card back.
  Full offline operation on stage is the single most convincing proof for
  the Privacy & Safety and On-Device Private Health rubric lines.
- **Writeup:** Kaggle Writeup, track = On-Device Private Health, secondary
  framing toward Voice for Care. Drafted incrementally by Pablo + Daniel as
  soon as the RAG-to-model path works end-to-end, not held until the final
  hours. Must include architecture, how Gemma 4 is used, challenges
  overcome in the 1-day sprint, and rationale for technical choices — under
  1,500 words.

## Out of scope for this sprint (explicitly cut)

- Custom LoRA decision-tree adapter fine-tune.
- Clinical review audit of generated training data.
- Formal on-device memory footprint measurement.
- Full 20-consecutive-query stress test and full error-trigger matrix on
  device (replaced by a lighter manual pass).
- Expanding the corpus beyond already-licensed wilderness + general
  first-aid sources.
