# Phase 1 Handoff

## Status

Phase 1 Core managers + SafetyFilterTests are implemented. Simulator/CI verification criteria are green. Device mic/TTS Airplane Mode smoke tests still need a human on hardware.

Do **not** start Phase 2 until you confirm `SafetyFilterTests` still pass in your environment.

## Quick verify

```bash
brew install xcodegen   # if needed
xcodegen generate
xcodebuild test \
  -scheme WildernessEdgeTests \
  -destination 'platform=iOS Simulator,name=iPhone 17'
```

Expected: `** TEST SUCCEEDED **` (5 tests).

## Layout added in Phase 1

- `WildernessEdge/App/` — entry point, Info.plist, increased-memory-limit entitlements
- `WildernessEdge/Core/` — `SpeechManager`, `TTSManager`, `CameraManager`, `SafetyFilter`
- `WildernessEdge/Views/ContentView.swift` — Phase 1 scaffold UI (listen / speak filtered / snapshot)
- `WildernessEdgeTests/SafetyFilterTests.swift` — host-free XCTest suite
- `project.yml` — XcodeGen source of truth (`WildernessEdge` + `WildernessEdgeTests` schemes)

## Known local issues (this Mac)

1. Ad-hoc simulator **app install** can fail with "Missing bundle ID" / unbound Info.plist when macOS stamps `com.apple.provenance` on bundle files. Host-free `WildernessEdgeTests` still runs.
2. `Assets.xcassets` is kept in the repo but **excluded from the app target** for the same provenance/codesign reason. Re-enable after setting a real `DEVELOPMENT_TEAM`, or when packaging for device.
3. Set `DEVELOPMENT_TEAM` in `project.yml` (or Xcode Signing) before physical-device runs.

## Next phase

Phase 2: `WordPieceTokenizer`, `TextEmbeddingManager`, `VectorRAGManager` + tests. Bundle `protocols.db` + embedder assets into `WildernessEdge/Resources/` first (Part 1 §2).
