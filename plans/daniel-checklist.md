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

- [ ] Agree shared contracts with Vaibhav + Sachin (5–10 min):
  - [ ] RAG: hit vs no-match (+ citation fields)
  - [ ] LLM: transcript + context (+ optional image later) → text
  - [ ] Errors: empty transcript / STT unavailable / LLM init fail
- [ ] Replace Phase 1 scaffold with push-to-talk state machine in `ContentView`
  - [ ] States: Idle → Listening → Processing → Speaking → Error
  - [ ] Button down: start STT (camera stretch — skip unless free)
  - [ ] Button up: stop STT → stub RAG → stub LLM → `SafetyFilter` → TTS
- [ ] SafetyFilter on **every** spoken path (fail closed)
- [ ] Error transitions show + speak a clear message (no crash / silent fail)
- [ ] Smoke in Simulator: full stub loop works end-to-end

---

## 1:00 → 1:30 — Lunch + writeup draft

- [ ] Writeup draft v1 with Pablo (Kaggle, ≤1,500 words, track: **On-Device Private Health**)
  - [ ] Problem (off-grid responders, zero connectivity)
  - [ ] Architecture (STT → embed → RAG → Gemma 4 → SafetyFilter → TTS)
  - [ ] How Gemma 4 is used (core inference, multimodal if ready)
  - [ ] Privacy (100% on-device / Airplane Mode; decision-support only)
  - [ ] Sprint challenges + why we cut LoRA / used prebuilt bundle
- [ ] Confirm public GitHub link for writeup attachments
- [ ] Draft 60–90s pitch spine (problem → offline Gemma → checklist → privacy)

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
