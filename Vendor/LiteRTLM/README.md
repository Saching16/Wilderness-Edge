# Vendored LiteRT-LM Swift package

Upstream `google-ai-edge/LiteRT-LM` is currently inconsistent for SPM consumers:

1. Tag **v0.14.0** `Package.swift` checksums no longer match the release XCFramework zips
   (GitHub re-uploaded the assets).
2. **main** has corrected checksums, but its Swift wrappers already call C APIs
   (`thinking_config`, constrained decoding, etc.) that are **not** present in the
   published `v0.14.0` `CLiteRTLM.xcframework`.

This folder vendors the **v0.14.0 Swift sources** (`80f301ff`) with `Package.swift`
checksums patched to the live release hashes:

- `CLiteRTLM.xcframework.zip` → `dddac2f6713ed65eaf01c18e115d9fec22184adf575cc7856a21387e8ba937e1`
- `CLiteRTLM_mac.xcframework.zip` → `450615483509aaa6d34b321fdc6862e41a224b674468ab10aff64ebe113d21b7`

Binaries are still downloaded from the official GitHub release URL at resolve/build time
(not checked into git). Revisit when Google publishes a coherent tag/binary pair.
