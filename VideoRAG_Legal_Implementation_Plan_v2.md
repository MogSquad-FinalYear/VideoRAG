# VideoRAG for Courtroom Video — Updated Implementation Plan (v2)
*Incorporates: bug fixes, base architecture corrections, and the novel testimony-consistency layer*

---

## 1. What this version adds over the base plan

The earlier plan fixed bugs and filled in your original architecture (ASR/OCR/DET/Scene Graph). This version adds the **novelty layer** that makes the project stand out for a legal/courtroom use case, built as three additions on top of that base:

| # | Addition | Inserted where |
|---|---|---|
| 3 | **Speaker/role tagging** — tags every ASR line with who said it (judge/witness/counsel) | Inside ② Auxiliary Text Generation, right after ASR |
| 1 | **Cross-session testimony memory** — remembers what each speaker said across *all* videos in a case and flags contradictions | New block between ② Auxiliary Retrieval and ③ Integration |
| 2 | **Citation verification** — re-checks that a cited timestamp actually supports the claim before answering, and attaches a hash for tamper-evidence | New block inside ③, right before Final Answer |

Build order: **3 → 1 → 2** (3 is a small prerequisite, 1 is the headline contribution, 2 finishes it into something legally defensible).

---

## 2. Bug fixes (do first, unchanged from before)

1. **Fix the timestamp bug**: `extract_frames()` widens sampling interval past 48 seconds but `index_frames()` still assumes `timestamp = i / FRAME_SAMPLE_FPS`. Compute and pass through the *true* per-frame timestamp instead.
2. **Remove the 48-frame cap**; replace fixed-interval sampling with scene-change-adaptive sampling.
3. **Default `CAPTION_BACKEND` to `blip`**, not `clip`.
4. **Add a `captions_ready` flag** to avoid querying before background captioning finishes.
5. **Add clip/segment merging** so retrieval returns a time range, not a single frame.

---

## 3. Models needed (updated)

| Component | Model | Notes |
|---|---|---|
| ASR | `openai-whisper` (`small`/`medium`), optionally **LoRA fine-tuned on legal audio** | Fine-tuning here is worthwhile and cheap — legal vocabulary + courtroom acoustics |
| Visual embeddings | CLIP `ViT-B-32` / `ViT-L-14` | Unchanged |
| Captioning | BLIP (`blip-image-captioning-base`) | Real captions, not CLIP-label captions |
| OCR | PaddleOCR or EasyOCR | For exhibit labels, screen text |
| Object detection | YOLOv8 | Feeds counting, location, scene graph |
| Scene graph | Rule-based spatial heuristics over YOLO boxes | No new model required |
| **Speaker diarization + role tagging (Idea 3)** | `pyannote-audio` (diarization) + a small rule/classifier layer mapping speaker turns to roles (judge/witness/counsel) using cues like seating order, phrasing ("objection," "the court finds"), or manual per-session speaker labeling at ingestion | Diarization model is open-source, local, no API key |
| **Contradiction/entailment scorer (Idea 1)** | A frozen NLI-style model (e.g. `roberta-large-mnli` or similar) or a zero-shot check via your existing Groq LLM, optionally **fine-tuned on a small legal statement-pair set** | Fine-tuning optional; zero-shot with the LLM is a reasonable first version |
| **Citation verifier (Idea 2)** | Same frozen LLM (Groq) doing a targeted "does this timestamp support this claim" check, plus a simple hash function (SHA-256) over the cited clip bytes | No new model, just an extra call + hashing logic |
| Final answer generation | Groq-hosted LLM (unchanged) | Stays frozen — no fine-tuning here |

**On fine-tuning overall**: only ASR and (optionally) the contradiction scorer are worth fine-tuning; everything else stays frozen, consistent with how every published Video-RAG paper in this space (Video-RAG, VideoRAG-KDD, Vgent, iRAG, Graph-to-Frame RAG, etc.) keeps the underlying model frozen and adds retrieval/reasoning logic around it instead.

---

## 4. Databases — updated

| Store | Contents | Notes |
|---|---|---|
| `image_index` (ChromaDB) | CLIP frame embeddings | Unchanged |
| `caption_index` (ChromaDB) | BLIP captions | Unchanged |
| `speech_index` / ASR DB (ChromaDB) | Whisper transcript segments **+ speaker/role tag** | Add `speaker_id`, `role` fields to metadata |
| `ocr_index` (ChromaDB) | OCR text per frame | As planned before |
| `det_index` (SQLite) | Object detections per frame | As planned before |
| `scene_graph_index` (SQLite) | Relation triples per frame | As planned before |
| **`testimony_memory` (NEW — SQLite or a small graph store)** | One row per statement: `{session_id, speaker_id, role, timestamp, text, embedding_id}`, spanning **all sessions in a case**, not just one video | This is what makes cross-session contradiction checking possible — it must persist across video uploads, not reset per video |
| **`citation_log` (NEW — SQLite)** | `{answer_id, cited_timestamp, cited_video, claim_text, verification_score, clip_hash}` | Produced by the Idea 2 verifier; gives you an audit trail per answer |

---

## 5. API keys — unchanged conclusion

Only `GROQ_API_KEY` is required. Diarization (`pyannote-audio`), NLI/entailment models, and hashing all run locally with open-source weights — no new paid API key needed even with the full novelty layer added.

---

## 6. Implementation phases (updated)

**Phase 0–2**: environment setup + bug fixes (as before).

**Phase 3**: Add OCR pipeline (as before).

**Phase 4**: Add YOLOv8 detection (as before).

**Phase 5**: Add rule-based scene graph (as before).

**Phase 6 (NEW) — Speaker/role tagging (Idea 3)**
1. Add `pyannote-audio` diarization after Whisper transcription to segment speech by speaker turn.
2. Map each diarized speaker to a role — simplest approach for a first version: let the user manually label each speaker once per session (e.g. "Speaker 1 = Judge, Speaker 2 = Witness"); a fully automatic classifier is a stretch goal.
3. Store `speaker_id` and `role` alongside each ASR segment in `speech_index`.

**Phase 7 (NEW) — Cross-session testimony memory (Idea 1)**
1. Create the persistent `testimony_memory` store, keyed by `case_id` (not `video_id`) so it spans every session uploaded for that case.
2. On each new ASR segment (with speaker/role attached), retrieve prior statements by the *same speaker* from `testimony_memory` using embedding similarity.
3. Run a contradiction check between the new statement and retrieved prior statements (NLI model or LLM zero-shot classification: entail / contradict / unrelated).
4. If contradiction is flagged, store it with both source timestamps + a confidence score; surface it through a new agent tool, e.g. `check_contradictions(speaker_id, case_id)`.

**Phase 8 (NEW) — Citation verification (Idea 2)**
1. After the final LLM drafts an answer with a timestamp citation, run a second, narrower LLM call: "Does the transcript/frame at this timestamp actually support this specific claim?"
2. Compute a SHA-256 hash of the cited clip (frame range or audio segment) and store it with the answer in `citation_log`.
3. If verification fails or confidence is low, either widen the search window and retry once, or explicitly tell the user the citation is uncertain rather than presenting it as fact.

**Phase 9 — Query decouple + merge (as before)**, now also considering: does the query ask about contradictions/consistency? If so, route to `check_contradictions` in addition to ASR/OCR/DET.

**Phase 10 — Testing & validation (updated, see §7 and §8 below)**.

---

## 7. Evaluation plan (updated)

- **Timestamp accuracy**: manually verify retrieved timestamps against ground truth on your test videos (fixes to check: Bug #1).
- **ASR word error rate**: before/after any Whisper fine-tuning on legal audio, if you attempt it.
- **Contradiction detection**: precision/recall on a small manually labeled set of consistent vs. contradictory statement pairs (see below for where to get statements) plus synthetic contradictions (paraphrase + negate real statements to scale up the test set).
- **Citation faithfulness**: for a sample of generated answers, manually check whether the cited timestamp actually supports the claim; report the percentage verified correctly by your Idea 2 layer vs. without it.

---

## 8. Where to get courtroom video to test with

| Source | What it gives you | Access | Best use in your project |
|---|---|---|---|
| **U.S. Courts Case Video Archive** ([uscourts.gov](https://www.uscourts.gov/court-records/access-court-proceedings/remote-public-access-proceedings/cameras-courts/case-video-archive)) | Real federal jury trials, bench trials, motion hearings, and evidentiary hearings from district courts (e.g. Southern District of Iowa, District of Kansas, Northern District of Illinois), filterable by subject matter and procedural posture | Free, official, public — no request needed | Best primary source: real courtroom footage with real testimony, multiple procedural stages per case (e.g. a hearing *and* a trial for the same case) — good for testing cross-session contradiction detection on genuinely separate proceedings for the same matter |
| **Real-Life Trial Deception Detection Dataset** (Pérez-Rosas et al., 2015 — hosted at [web.eecs.umich.edu/~mihalcea/downloads.html](https://web.eecs.umich.edu/~mihalcea/downloads.html)) | 121 short clips (~28 sec each) from real public court trials (including well-known cases), each labeled truthful/deceptive, with transcripts | Public dataset, requires filling a short access/agreement form per the site's policy | Good for short clips with transcripts already prepared — useful for quick pipeline testing (ASR accuracy, retrieval) without needing to process long videos; the truthful/deceptive labels are also a natural fit for testing your contradiction/consistency scorer, since "deceptive" testimony often correlates with inconsistency |
| **Law & Crime Network / Court TV (YouTube channels)** | Full, long-form trial livestream recordings and replays, many spanning multiple days per case | Free to view/download for research/testing (check each video's terms before any redistribution) | Best for realistic **long, multi-day** courtroom footage — ideal for stress-testing your adaptive frame sampling and multi-session testimony memory across several full days of the same trial |
| **Old Bailey Proceedings Online** | Not video — digitized historical trial *text* transcripts (1674–1913), but useful for text-only testing of your contradiction/entailment scorer at scale before wiring up full video | Free, public | Use only for bootstrapping/testing the NLI contradiction-detection component on large volumes of realistic testimony-style text, independent of video |

**Practical recommendation**: start with 2–3 cases from the **U.S. Courts Case Video Archive** that have multiple recorded proceedings (e.g. a motion hearing plus a later trial for the same case) — this gives you genuine multi-session data for testing Idea 1 without relying on synthetic contradictions. Use the **Real-Life Trial dataset** for fast, short-clip pipeline testing early on. Use **Law & Crime Network/Court TV** footage once your pipeline is stable, to stress-test on realistic full-length, multi-day trial coverage.

---

## 9. Summary checklist (updated)

- [ ] Fix timestamp bug, remove frame cap, fix caption backend, fix caption-ready race
- [ ] Add OCR, DET, scene graph (base architecture)
- [ ] Add speaker diarization + role tagging (Idea 3)
- [ ] Add persistent cross-session `testimony_memory` + contradiction scorer (Idea 1)
- [ ] Add citation verification + clip hashing (Idea 2)
- [ ] Only required API key throughout: `GROQ_API_KEY`
- [ ] Pull initial test videos from U.S. Courts Case Video Archive + Real-Life Trial dataset
- [ ] Evaluate: timestamp accuracy, contradiction-detection precision/recall, citation faithfulness rate
