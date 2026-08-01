# Wilderness Edge

**Subtitle:** On-device, voice-first protocol checklists for off-grid emergency responders — powered by Gemma 4, with patient data that never leaves the phone.

**Track:** On-Device Private Health (primary) · Voice for Care (secondary framing)

**Team:** Pablo Beaus Iranzo, Sachin Ganpule, Daniel David, Vaibhav Chaudhari  
**Repo:** https://github.com/Saching16/Wilderness-Edge  
**Live demo:** _[attach screen recording or device demo link before submit]_

---

## Problem

Wilderness and disaster responders often need protocol guidance with **zero connectivity** — no cloud APIs, no EHR lookup, no model hosted elsewhere. Phones may be the only computer on scene, and the information that matters (injury description, scene audio, a quick visual of the limb or wound) is sensitive.

Existing apps either require the network or turn the model into an unsupervised “doctor.” Neither is acceptable. Responders need **decision support**: fast, cited checklists from accredited field manuals — not a diagnosis, not a drug order.

## Solution

**Wilderness Edge** is a native iOS app that runs entirely offline:

1. Responder taps once to talk (and optionally capture a camera frame).
2. On-device speech recognition transcribes the question.
3. A local vector search retrieves the most relevant protocol chunks from a bundled SQLite database (`protocols.db`).
4. **Gemma 4 E4B** (via Google AI Edge LiteRT-LM) turns transcript + retrieved context (+ image) into a short, spoken checklist with a source citation.
5. A client-side **SafetyFilter** blocks diagnostic / prescribing language before anything is shown or spoken.
6. Text-to-speech reads the checklist aloud so hands stay free.

Demo condition: **Airplane Mode** — Wi‑Fi, cellular, and Bluetooth off.

## How we use Gemma 4

Gemma 4 is the **core inference engine**, not a thin wrapper around search results.

- **Runtime:** LiteRT-LM Swift API on device (GPU-accelerated where available).
- **Model:** Prebuilt multimodal `gemma-4-E4B-it` LiteRT-LM bundle (hackathon sprint choice; see tradeoffs below).
- **Inputs per query:**
  - User transcript (from on-device `SFSpeechRecognizer`)
  - Top retrieved protocol excerpts + citation metadata (RAG)
  - Optional single camera snapshot as native multimodal image input
- **Output shape:** Short numbered action checklist, opened with a source citation, framed as retrieved protocol guidance — never an independent diagnosis.

When retrieval has **no confident match**, we do not invent a protocol: the model is instructed to say clearly that no matching manual section was found.

## Architecture

```text
[ Tap-to-talk button ]
        │
        ├─► Camera snapshot (AVFoundation)
        └─► On-device STT (SFSpeechRecognizer)
                │
                ▼
        CoreML query embedder (384-d)
                │
                ▼
        Vector RAG over protocols.db (Accelerate cosine search)
                │
                ▼
        Gemma 4 E4B via LiteRT-LM  ◄── image + RAG context + transcript
                │
                ▼
        SafetyFilter (regex, fail closed)
                │
                ▼
        UI citation card + AVSpeechSynthesizer
```

**Team split (4-hour sprint):** offline corpus/assets (Pablo) · on-device RAG (Vaibhav) · LiteRT-LM / Gemma wiring (Sachin) · native I/O, SwiftUI state machine, SafetyFilter enforcement, writeup (Daniel).

## Privacy & safety

- **100% on-device at runtime** — no network calls for speech, embedding, retrieval, or generation. If on-device speech is unavailable, we fail closed (visible + spoken error); we never fall back to server recognition.
- **Decision-support only** — outputs are cited checklists from licensed public protocol sources. The app does not diagnose conditions or recommend drug doses.
- **SafetyFilter** — mandatory gate before display and TTS; intercepts diagnostic / prescriptive phrasing.
- **Corpus licensing** — only sources with recorded redistribution rights in `sources.manifest.json` (NOLS course materials explicitly excluded).
- **Synthetic / public protocol text only** — no real patient data.

## Sprint challenges & technical choices

| Challenge | Choice | Why |
| --- | --- | --- |
| ~4-hour build window | Parallel tracks + stubbed `runInferencePipeline` until Gemma wiring landed | Unblocked UI and RAG from waiting on model I/O |
| Custom LoRA fine-tune | **Dropped for this sprint** | Training + clinical review do not fit the clock; grounding comes from RAG + prompting |
| Model size / access | Ship **prebuilt Gemma 4 E4B** LiteRT-LM bundle | Avoid gated fine-tune export friction; keep multimodal path |
| Memory gates | Soft check on iPhone 16 Plus | Entitlement kept; no multi-hour profiling in a one-day sprint |
| Stress testing | Two reliable Airplane Mode demo runs | Replaces a 20-loop matrix for submission day |

These are deliberate scope cuts so Gemma stays central and the offline story stays honest — not a pivot away from on-device inference.

## Healthcare impact

Wilderness Edge targets **care teams in austere environments**: wilderness first responders, ski patrol, SAR, and EMS staging where connectivity fails. It shortens time-to-protocol (“what does the manual say for extremity trauma?”) while keeping judgment with the trained human and medical direction. That maps directly to private, on-device health tooling and hands-busy voice UX.

## What’s next (post-hackathon)

- Reintroduce a small decision-tree LoRA once training data is clinically reviewed  
- Formal memory footprint measurement on base iPhones  
- Broader licensed protocol corpus and threshold calibration  

## Attachments (Kaggle UI)

- **Public code repository:** https://github.com/Saching16/Wilderness-Edge  
- **Live demo:** _TODO — screen recording of Airplane Mode run_  

---

_Word count target: keep final Kaggle paste under 1,500 words. Trim the “What’s next” section first if over limit._
