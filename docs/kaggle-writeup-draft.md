# Wilderness Edge

**On-device, voice-first protocol checklists for off-grid emergency responders — powered by Gemma 4, with patient data that never leaves the phone.**

**Track:** On-Device Private Health (primary) · Voice for Care (secondary framing)  
**Team:** Pablo Beaus Iranzo, Sachin Ganpule, Daniel David, Vaibhav Chaudhari

---

### Inspiration
What local problem are you solving today?

Wilderness and disaster responders often need field-protocol guidance with **zero connectivity** — no cloud APIs, no EHR, no hosted model. The phone may be the only computer on scene, and audio/visual context about an injury is sensitive. Existing apps either require the network or act like an unsupervised doctor. We built **Wilderness Edge**: an on-device, voice-first assistant that returns **cited action checklists** from licensed first-aid / EMS manuals — **decision support only**, never a diagnosis or drug order, with patient data that never leaves the phone.

---

### How we built it
Which Gemma model did you use? Did you use RAG, prompt engineering, or fine-tuning? What frameworks (Transformers, Keras, etc.) did you use?

We used **Gemma 4 E4B** (prebuilt multimodal `gemma-4-E4B-it`) running fully on-device via Google AI Edge **LiteRT-LM** (Swift), with GPU when available.

**Grounding:** **RAG + prompt engineering** — not fine-tuning this sprint. A bundled SQLite DB (`protocols.db`, ~2,192 chunks from 3 licensed sources: US Army ATP 4-02.11, TCCC Handbook v5, NASEMSO EMS Guidelines v3.0) is searched with an on-device **CoreML** embedder (`all-MiniLM-L6-v2` space) + Accelerate cosine search. Gemma receives transcript + retrieved protocol text (+ optional camera frame) and must output a short cited checklist; if there’s no confident match, it must say so instead of inventing protocol.

**Stack:** native **iOS / SwiftUI**; on-device **SFSpeechRecognizer** + **AVSpeechSynthesizer** + **AVFoundation** camera; client-side **SafetyFilter** before any display/TTS. Offline corpus tooling is Python (`sentence-transformers`, SQLite). We deliberately **skipped LoRA fine-tuning** in the one-day window so Gemma stays central and grounding stays retrieval-based.

---

### Offline resources

Everything runs from assets bundled in the app — nothing is fetched after install.

**On-device bundles**
- `protocols.db` (~8.6 MB): **2,192** embedded protocol chunks with citations  
- CoreML query embedder (`all-MiniLM-L6-v2` space) + WordPiece tokenizer  
- Gemma 4 E4B LiteRT-LM multimodal model (local generation)

**Knowledge base (PDFs ingested into `protocols.db`)**
1. **US Army ATP 4-02.11** — *Casualty Response, Tactical Combat Casualty Care, and First Aid* (US federal public domain)  
2. **US Army TCCC Handbook v5** (CALL Handbook 17-13) — tactical casualty care / MARCH-oriented field care (US federal public domain)  
3. **NASEMSO National Model EMS Clinical Guidelines v3.0** (March 2022) — publisher invites harvest and adoption in whole or in part  

Provenance and license text live in `OffLineTools/sources.manifest.json` (see also `SOURCES.md`). **NOLS** wilderness course materials were **not** included — copyrighted and not licensed for app redistribution. The offline build pipeline (`fetch_sources.py` → `build_vector_db.py` → `export_embedder_coreml.py`) runs on a laptop before shipping; the phone only reads the prebuilt bundles.

---

### The Prototype

Demo video: https://youtu.be/r5BoZbtaVkk  

GitHub repo: https://github.com/Saching16/Wilderness-Edge  

---

### Challenges we ran into
What was the hardest part of building this in one day?

Shipping a **real offline multimodal stack** under a ~5-hour clock: coordinating four parallel tracks (corpus/assets, vector RAG, LiteRT-LM/Gemma, native voice UI + safety), keeping embedding spaces in sync between the offline DB build and the on-device CoreML embedder, and dealing with **device/Simulator tooling friction** (large iOS runtime downloads, signing, install quirks) while still aiming for an Airplane Mode demo. We also had to be strict about **licensing** — only bundling sources we can redistribute (and excluding NOLS despite the wilderness use case) — and enforce **non-diagnostic** behavior with prompt rules plus a hard SafetyFilter gate before speech.
