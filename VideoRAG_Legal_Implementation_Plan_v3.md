# VideoRAG for Courtroom Video — Full Implementation Document (v3)
*Base architecture + three novelty layers, with automatic cross-session speaker identification*

---

## 1. One-line pitch

Standard Video-RAG answers *"what's in this video."* This system answers *"what's in this video, who said it, does it match what that same person said in earlier sessions of this case, and can you prove it."*

---

## 2. Base architecture (recap)

Video → split into **Audio** + **Sampled Frames** →
- Audio → **ASR** (Whisper) → ASR Database
- Frames → **OCR** → OCR Database
- Frames → **CLIP embedding → key frames → Visual Grounding → Object Detection → Scene Graph** → DET Database

User query → **Query Decouple** (LLM breaks query into a structured retrieval request) → **Retriever** pulls matching entries from ASR/OCR/DET databases above a similarity threshold → **Integration & Generation** merges everything → **Final Answer**.

*(Bug fixes required before any of the below: correct per-frame timestamp calculation, remove the fixed 48-frame sampling cap in favor of adaptive sampling, default captioning to BLIP not CLIP-labels, fix the caption-ready race condition, add segment/clip merging to retrieval. These were covered in the earlier bug-fix pass and are prerequisites, not optional.)*

---

## 3. Novelty 1 — Speaker/Role Tagging + Cross-Session Voice Identification

**Why:** the base ASR Database stores *what* was said but not *who* said it, and has no way of knowing that "Speaker 2" in Monday's video is the same person as "Speaker 1" in Friday's video. Both gaps need closing before contradiction detection (Novelty 2) is possible at all.

### Step by step

1. **Transcribe** the audio track with Whisper (existing pipeline) → timestamped text segments.
2. **Diarize** the same audio with `pyannote-audio` → speaker turns labeled generically ("Speaker 1," "Speaker 2"...), local to this video only.
3. **Align** ASR segments to diarization turns by timestamp overlap → each transcript line now has a local `speaker_label`.
4. **Extract a voiceprint** for each local speaker cluster using a speaker-embedding model (`SpeechBrain` ECAPA-TDNN or `Resemblyzer`) — a fixed-length vector representing that person's voice, computed from their audio segments in this video.
5. **Match against stored voiceprints for this `case_id`:**
   - If this is the *first* video for the case: no prior voiceprints exist — store each new voiceprint under this case, prompt the user once to assign a name/role (Judge, Witness, Counsel).
   - If prior videos exist for this case: compare each new local voiceprint against all stored voiceprints for this `case_id` using cosine similarity.
     - Similarity above threshold → same person → automatically carry over their name/role from before.
     - Similarity below threshold → treat as a new speaker → prompt for a name/role once.
   - **Always show the auto-match to the user for confirmation** rather than trusting it silently — a wrong auto-match would corrupt the contradiction detection downstream, so a one-click human confirm is worth the small friction.
6. **Store** the final `{speaker_name, role, voiceprint}` in a persistent `speaker_registry` table keyed by `case_id`, and tag every ASR segment with `speaker_name` + `role` before it's written to the ASR Database.

### Tech stack
- `pyannote-audio` — diarization, local, no API key
- `SpeechBrain` (ECAPA-TDNN) or `Resemblyzer` — voiceprint embeddings, local, no API key
- Cosine similarity — plain NumPy, no extra library needed
- Existing Whisper setup — unchanged

---

## 4. Novelty 2 — Cross-Session Testimony Memory

**Why:** every box in the base architecture operates on one video at a time. A trial spans many separate recordings across days/weeks; nothing in the base design connects a statement in one session to a statement in another. This is the headline contribution — it's what catches a witness changing their story.

### Step by step

1. Every video upload is assigned a `case_id` (user selects "add to existing case" or "new case").
2. After Novelty 1 tags a new ASR segment with `{speaker_name, role, case_id, timestamp, text}`, embed the statement text (reuse a sentence embedder, e.g. `all-MiniLM-L6-v2`).
3. **Query the persistent `testimony_memory` store**: retrieve the top-k most similar *prior* statements where `speaker_name` and `case_id` match — not the whole database, just this person's prior statements in this case.
4. If no prior statements exist yet for this speaker (first time they've spoken in this case), just store the new one — nothing to compare against yet.
5. If prior statements are found, run a **contradiction check** on each candidate pair: ask a model (start with zero-shot via your existing Groq LLM: "do these two statements agree, disagree, or are they unrelated?"; upgrade later to a local NLI model like `roberta-large-mnli` if you want it faster/cheaper at scale) and get a label + confidence score.
6. If flagged as a contradiction, store `{statement_a, timestamp_a, video_a, statement_b, timestamp_b, video_b, confidence}` in a `contradiction_flags` table.
7. Store the new statement's embedding into `testimony_memory` regardless of outcome, so it becomes part of what *future* sessions get compared against — the memory keeps growing with every upload.
8. Expose this as a callable agent tool, e.g. `check_contradictions(speaker_name, case_id)`, so a user can directly ask "did this witness contradict themselves" and get the flagged pairs with citations.

### Tech stack
- Persistent SQLite table (or a dedicated ChromaDB collection) for `testimony_memory` — must persist across video uploads, unlike the per-video ASR/OCR/DET stores
- `sentence-transformers` (`all-MiniLM-L6-v2`) for statement embeddings
- Groq LLM (existing) for zero-shot contradiction scoring, or `roberta-large-mnli` locally as an upgrade
- No new API key required

---

## 5. Novelty 3 — Citation Verification

**Why:** the base architecture generates a final answer and shows it with no check that it's actually correct. A courtroom-adjacent tool needs its claims to be provable, not just plausible — especially claims coming out of the contradiction check in Novelty 2.

### Step by step

1. After the final LLM drafts an answer with a timestamp citation (including any contradiction flags surfaced from Novelty 2), hold it back before showing it to the user.
2. Run a second, narrower LLM call: feed it *only* the cited transcript line/frame and the specific claim, and ask "does this evidence actually support this claim — yes/no, with confidence."
3. Compute a SHA-256 hash of the cited clip (the specific frame range or audio segment) so there's a tamper-evident record of exactly what was cited.
4. Log `{claim, cited_timestamp, cited_video, verification_result, confidence, hash}` to a `citation_log` table.
5. If verification passes with high confidence: release the answer as-is, with the citation.
6. If verification fails or confidence is low: either widen the retrieval window and retry once, or return the answer explicitly flagged as "unverified — review manually" rather than stating it as settled fact.

### Tech stack
- Same Groq LLM already in use — one extra call per answer, no new model
- Python's built-in `hashlib` for SHA-256 hashing
- A small SQLite table for `citation_log`

---

## 6. Databases — full picture

| Store | Scope | Contents |
|---|---|---|
| `image_index`, `caption_index` (ChromaDB) | Per video | CLIP frame embeddings, BLIP captions |
| `speech_index` / ASR DB (ChromaDB) | Per video | Transcript segments **+ speaker_name + role** |
| `ocr_index` (ChromaDB) | Per video | OCR text per frame |
| `det_index`, `scene_graph_index` (SQLite) | Per video | Object detections, relation triples |
| **`speaker_registry` (SQLite) — NEW** | **Per case** | `{case_id, speaker_name, role, voiceprint_vector}` — persists across all videos in a case |
| **`testimony_memory` (SQLite/Chroma) — NEW** | **Per case** | `{case_id, speaker_name, timestamp, video_id, statement_text, embedding}` — persists and grows with every upload |
| **`contradiction_flags` (SQLite) — NEW** | Per case | Flagged contradictory statement pairs with both citations |
| **`citation_log` (SQLite) — NEW** | Global | Every answer's verification result + clip hash, for audit purposes |

---

## 7. Build order

1. Bug fixes to base architecture (timestamp, sampling, caption backend, caption-ready flag, clip merging)
2. Base architecture completion: OCR, DET, Scene Graph (if not already done)
3. **Novelty 1**: diarization → voiceprint extraction → cross-video matching with user confirmation → role tagging
4. **Novelty 2**: persistent testimony memory → similarity retrieval → contradiction scoring → flag storage → `check_contradictions` tool
5. **Novelty 3**: post-answer verification call → hashing → `citation_log`
6. Query Decouple update: route "did X contradict themselves" style questions to `check_contradictions`
7. Testing & evaluation (below)

---

## 8. Evaluation plan

- **Voiceprint matching accuracy**: manually verify a sample of auto-matches across sessions are actually the same person (precision of same-speaker linking)
- **Contradiction detection**: precision/recall on a small manually labeled set of consistent vs. contradictory statement pairs, supplemented with synthetic contradictions (paraphrase + negate real statements)
- **Citation faithfulness**: percentage of answers where the cited timestamp actually supports the claim, with vs. without the verification layer
- **Timestamp accuracy**: confirm retrieved timestamps against ground truth (validates the base bug fixes)

---

## 9. Where to get test videos

- **U.S. Courts Case Video Archive** (uscourts.gov) — free, official, real federal proceedings; look for cases with multiple recorded sessions to test cross-session matching genuinely
- **Real-Life Trial Deception Detection Dataset** (Pérez-Rosas et al.) — short labeled clips with transcripts, good for fast pipeline testing
- **Law & Crime Network / Court TV** (YouTube) — full multi-day trial coverage for stress-testing at scale
- Extraction note: many court video pages use JS-based players with no visible download button — try `yt-dlp <page_url>` first, then browser DevTools → Network tab for the underlying `.mp4`/`.m3u8` stream, with screen recording as a reliable fallback

---

## 10. Summary checklist

- [ ] Fix base architecture bugs (timestamp, sampling cap, caption backend, race condition, clip merging)
- [ ] Diarize + extract voiceprints per video (Novelty 1)
- [ ] Match voiceprints across sessions within a case, with human confirmation (Novelty 1)
- [ ] Tag ASR segments with speaker name + role (Novelty 1)
- [ ] Build persistent, case-scoped `testimony_memory` (Novelty 2)
- [ ] Run contradiction scoring on new statements against prior ones by the same speaker (Novelty 2)
- [ ] Expose `check_contradictions` as a queryable tool (Novelty 2)
- [ ] Add post-answer citation verification + clip hashing (Novelty 3)
- [ ] Only required API key throughout: `GROQ_API_KEY` — everything else runs locally, open-source
- [ ] Source test videos from U.S. Courts archive, prioritizing multi-session cases
