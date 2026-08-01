# Daniel — Hackathon Checklist

**Role:** Glue + SafetyFilter + writeup/pitch  
**Phone / device install:** Sachin’s scope — verify after install, not blocking now  
**As of:** Sat Aug 1, **12:05** (build started 11:00; submissions lock **3:45**)

Venue clock: lunch 1:00 · mentor check-ins 3:00 · submit 3:45 · pitch 4:00

---

## Done / skipped

- [x] Scope locked (phone deferred to Sachin)
- [x] Git push access to `Saching16/Wilderness-Edge`
- [x] Working branch: `daniel/native-ui` (dd01 planning docs ported here)
- [ ] ~~Hour-0 phone install / Airplane Mode smoke~~ → **Sachin**; Daniel verifies later

---

## Now → 1:00 — State machine + stubs

- [x] Agree shared contracts with Vaibhav + Sachin (5–10 min):
  - [x] RAG: hit vs no-match (+ citation fields) — inside `runInferencePipeline` (Sachin swaps later)
  - [x] LLM: `(String, UIImage?) async -> (citation: String?, checklistText: String)` locked
  - [x] Errors: empty transcript / STT unavailable / LLM fail (empty checklistText)
- [x] Replace Phase 1 scaffold with push-to-talk state machine in `ContentView`
  - [x] States: Idle → Listening → Processing → Speaking → Error
  - [x] Tap to start STT + camera snapshot; tap again to send
  - [x] Stop STT → stub RAG/LLM → `SafetyFilter` → TTS
- [x] SafetyFilter on **every** spoken path (fail closed)
- [x] Error transitions show + speak fixed lines (no crash / silent fail)
- [ ] Smoke in Simulator: full stub loop works end-to-end

---

## 1:00 → 1:30 — Lunch + writeup draft

- [x] Writeup draft v1 scaffolded at `docs/kaggle-writeup-draft.md` (review/edit with Pablo)
  - [x] Problem (off-grid responders, zero connectivity)
  - [x] Architecture (STT → embed → RAG → Gemma 4 → SafetyFilter → TTS)
  - [x] How Gemma 4 is used (core inference, multimodal if ready)
  - [x] Privacy (100% on-device / Airplane Mode; decision-support only)
  - [x] Sprint challenges + why we cut LoRA / used prebuilt bundle
- [x] Confirm public GitHub link for writeup attachments (`Saching16/Wilderness-Edge`)
- [ ] Draft 60–90s pitch spine (problem → offline Gemma → checklist → privacy)
- [ ] Paste into Kaggle Writeup UI + attach demo recording before 3:45

---

## 1:30 → 2:30 — Integration

- [ ] Wire Vaibhav’s real RAG manager (drop stub)
- [ ] Wire Sachin’s LiteRT / Gemma path (drop stub)
- [ ] “No confident match” → honest spoken deflection (not Error state)
- [ ] First full loop in Simulator
- [ ] Hand off to Sachin for device install when ready
- [ ] **Verify on phone after install** (when Sachin has it sideloaded):
  - [ ] Mic + TTS
  - [ ] Full query loop
  - [ ] Airplane Mode run

---

## 2:30 → 3:00 — Freeze + demo prep

- [ ] **Feature freeze** — no new capabilities
- [ ] Rehearse demo script twice (or coach Sachin through it on device)
- [ ] Capture **backup screen recording** of a good run (valid Live Demo attachment)
- [ ] Mentor check-in: show working loop + one-line story

---

## 3:00 → 3:45 — Submit

- [ ] Finalize Kaggle Writeup with Pablo
- [ ] Attach: public code repo
- [ ] Attach: Live Demo (recording and/or device demo notes)
- [ ] Hit **Submit** before 3:45 (drafts don’t count)
- [ ] Repo clearly shows Gemma 4 / LiteRT usage in README or obvious path

---

## 4:00 — Pitch

- [ ] Live presentation: problem → offline Gemma → spoken checklist → privacy punchline
- [ ] Stay decision-support only (no diagnosis/treatment claims)

---

## Stretch (only if ahead)

- [ ] Subtitle card UI (citation + checklist text)
- [ ] Camera snapshot in live multimodal path
- [ ] UI polish

---

## Explicit non-goals (today)

- Device provisioning / signing (Sachin)
- `VectorRAGManager` / CoreML embedder internals (Vaibhav)
- LiteRT `EngineConfig` / model bundling (Sachin / Pablo)
- LoRA fine-tune, corpus rebuild, formal memory benchmarking
