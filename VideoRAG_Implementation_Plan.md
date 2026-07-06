# VideoRAG — Full Correction & Implementation Plan
*(Aligned with the Query-Decouple / ASR-OCR-DET / Scene-Graph architecture)*

---

## 1. Where you are today vs. where the architecture needs you to be

Your current repo (`MogSquad-FinalYear/VideoRAG`) implements roughly **35%** of the architecture diagram:

| Diagram block | Status in current code | Gap |
|---|---|---|
| ① Query Decouple | ❌ Not implemented | Agent picks ONE tool per query, no JSON decoupling |
| Frame Sampler | ⚠️ Implemented but buggy | Uniform sampling + broken timestamp math (see §2) |
| ASR (Whisper) + ASR DB | ✅ Implemented | Works, keep it |
| OCR + OCR DB | ❌ Not implemented | Missing entirely |
| Object Detection (DET) + DET DB | ❌ Not implemented | Missing entirely |
| Scene Graph | ❌ Not implemented | Missing entirely |
| CLIP similarity / Visual Grounding | ✅ Partial | CLIP embeddings exist, but "visual grounding" (localizing an object inside a frame) does not |
| ③ Merge & Rearrange + Final Answer | ⚠️ Partial | LLM just gets raw tool outputs, no structured merge step |

So Phase 1 is **bug fixes to what exists**, Phase 2 is **filling the architecture gaps**, Phase 3 is **the merge/decouple orchestration layer that ties it all together**.

---

## 2. CRITICAL BUG FIXES (do these first — nothing else matters until this is right)

### Bug #1 — Wrong timestamps for any video > 48 seconds (root cause of "wrong clip" issue)
**File:** `backend/services/video_processor.py` + `backend/routers/execute.py` + `backend/services/indexing_service.py`

- `extract_frames()` silently widens the sampling interval once a video would exceed `MAX_FRAMES_PER_VIDEO` (48), but `index_frames()`/`index_captions()` still compute `timestamp = i / FRAME_SAMPLE_FPS` — a **hardcoded 1-second-per-frame assumption**. This is wrong for every video longer than 48 seconds.

**Fix:**
1. Change `extract_frames()` to return `(frame_paths, true_fps_used, frame_interval, video_fps)` instead of just paths.
2. Compute real per-frame timestamp at extraction time: `timestamp = frame_idx / video_fps` (using the *actual* OpenCV frame index at capture time, not the sequential saved-frame counter).
3. Save this timestamp list alongside frame paths and pass it directly into `index_frames()` / `index_captions()` — stop recomputing it downstream from a static constant.

### Bug #2 — Frame cap far too small for real videos
- `MAX_FRAMES_PER_VIDEO = 48` means a 10-minute video gets one frame every ~12.5 seconds; a 30-minute video, one every ~37 seconds. Any short/fast event between samples is invisible to retrieval.
**Fix:** Remove the hard frame cap. Replace uniform interval sampling with **adaptive scene-change sampling** (§4.2). Cap by *compute budget* (e.g. max 400–600 frames per video) instead of a flat 48.

### Bug #3 — Caption backend silently defaults away from BLIP
- `CAPTION_BACKEND` defaults to `"clip"` even though the README advertises BLIP captions. Zero-shot CLIP-label captions are far less descriptive than actual generated BLIP captions.
**Fix:** Set `CAPTION_BACKEND=blip` as the real default, and only fall back to CLIP-label captions if BLIP fails to load (e.g., low-RAM environment).

### Bug #4 — Async caption race condition
- Video processing is marked `"completed"` while captions are still generating in a background thread. A query fired immediately after upload silently gets no caption hits.
**Fix:** Add a `captions_ready: bool` flag to task/video status; surface it in the UI ("Captions still indexing…") and have the retrieval layer skip/deprioritize the caption index until it flips true.

### Bug #5 — No clip/segment merging
- `get_video_clip_info` only returns frames inside a manually given time range; nothing merges multiple adjacent hits into one coherent event.
**Fix:** Add a segment-merge step (§4.5) before the final answer.

---

## 3. Do CLIP and BLIP do the same thing? (direct answer)

**No — they are different models solving different problems, and you actually need both.**

| | CLIP | BLIP |
|---|---|---|
| What it does | Embeds images and text into the **same vector space** so you can compute similarity between them | Actually **generates** natural-language captions/answers from an image (and can also do retrieval, but that's not its main strength) |
| Output | A fixed-size vector (embedding) | Free-form text ("a man is loading boxes into a white van") |
| Use case here | **Semantic search** — "find frames similar to this text query" via cosine similarity | **Description generation** — produce a human-readable caption per frame, which then gets embedded (with CLIP or a text embedder) and indexed for text search |
| Training objective | Contrastive (image-text pairs pulled together, mismatched pairs pushed apart) | Encoder-decoder captioning objective (image → generated sentence), fine-tuned with contrastive + generation losses |

**In your pipeline, both are needed for different jobs:**
- **CLIP** → the `image_index` collection (raw visual similarity search — "find the frame that looks like X").
- **BLIP** → generates the caption text, which is then embedded (by CLIP's text encoder or a separate text embedder) and stored in the `caption_index` collection (semantic/natural-language search over what's *described* to be happening).

Using CLIP to *generate* captions (as your current default does) is a workaround — it just returns the closest label/keyword from a small fixed vocabulary, not a real sentence. BLIP gives genuinely richer captions, which is what your architecture diagram implies by having ASR Text and OCR Text as parallel structured knowledge sources.

---

## 4. Models to use (full list, with why)

| Component | Model | Why |
|---|---|---|
| **ASR (speech-to-text)** | `openai-whisper` (`small` or `medium` if GPU available, `tiny`/`base` for CPU-only) | Already integrated; upgrading from `tiny` → `small`/`base` meaningfully improves transcript accuracy for the query decouple step |
| **Visual embeddings (retrieval)** | CLIP `ViT-B-32` (already used) or `ViT-L-14` if you have GPU headroom | Larger CLIP = better semantic frame retrieval, at higher compute cost |
| **Captioning** | `Salesforce/blip-image-captioning-base` (already in requirements, just not defaulted to) | Real natural-language frame descriptions |
| **OCR (text in frames)** | `PaddleOCR` (recommended) or `EasyOCR` | Both run CPU-only reasonably; PaddleOCR is generally more accurate for varied fonts/angles |
| **Object Detection** | `YOLOv8` (`ultralytics` package, `yolov8n` for CPU, `yolov8m/l` for GPU) | Fast, well-supported, gives bounding boxes + class + confidence per frame — feeds counting, location, and scene graph |
| **Object Counting** | Derived directly from YOLO detections (count instances per class per frame) | No separate model needed |
| **Object Location** | Derived from YOLO bounding box center + normalized image coordinates | No separate model needed |
| **Object Relations / Scene Graph** | Rule-based first pass (spatial heuristics: "left of", "on top of", "near") using bounding box geometry; optional upgrade to a learned scene-graph model (`RelTR`) later if time permits | Rule-based is realistic for a final-year project timeline and is defensible in your report |
| **Query Decouple + Final Answer generation ("Any VLM" in diagram)** | Groq-hosted `meta-llama/llama-4-scout-17b-16e-instruct` (already integrated) — text-only LLM is fine for decouple/merge steps since it only needs to reason over structured text, not raw pixels | Keep your existing Groq integration; no need for a true VLM unless you want direct frame-to-answer verification (see §4.6) |
| **(Optional) Frame-level VLM verification** | A vision-capable model (e.g. via Groq's vision-capable Llama models, or a locally-run `LLaVA`/`Qwen2-VL` if GPU available) | Used only in the final verification/boundary-tightening step (§4.5), not for every frame — keeps cost low |

---

## 5. Databases — what to store where

Keep **ChromaDB** as your vector store (already integrated, works fine at this project's scale — no need to migrate to Milvus/Qdrant/Weaviate unless you expect very large multi-video corpora or need production-grade horizontal scaling). Just restructure the collections to match the architecture:

| Collection | Contents | Embedding used | Metadata to store |
|---|---|---|---|
| `image_index` (existing) | Raw CLIP frame embeddings | CLIP image encoder | `video_id`, `frame_number`, **true timestamp (fixed per Bug #1)**, `frame_path` |
| `caption_index` (existing, fix backend) | BLIP-generated captions | CLIP text encoder (or a dedicated text embedder like `all-MiniLM-L6-v2`) | same as above + `caption_text` |
| `speech_index` / ASR DB (existing) | Whisper transcript segments | CLIP text encoder or `all-MiniLM-L6-v2` | `video_id`, `start_time`, `end_time`, `text` |
| **`ocr_index` (NEW)** | Extracted on-screen text per frame | Same text embedder as above | `video_id`, `frame_number`, `timestamp`, `ocr_text`, `bbox` |
| **`det_index` (NEW)** | Per-frame object detections | Not embedded as vectors — store as **structured metadata** (class, bbox, confidence) since it's queried by filter/count, not semantic similarity. Chroma can hold these as a metadata-only collection, or simpler: a lightweight **SQLite table** for structured queries | `video_id`, `frame_number`, `timestamp`, `object_class`, `bbox`, `confidence` |
| **`scene_graph_index` (NEW)** | Per-frame relation triples (object A – relation – object B) | Not vector-searched; structured storage (SQLite or a JSON blob per frame) | `video_id`, `frame_number`, `timestamp`, `triples: [(subject, relation, object)]` |

**Practical recommendation:** Use ChromaDB for anything that needs *semantic similarity search* (image, caption, speech, OCR-text), and a simple **SQLite database** (or even structured JSON files, given your scale) for anything that's *structured/counted/filtered* rather than similarity-searched (DET, scene graph). This mirrors your diagram's separation — the DET/Scene-Graph box in your architecture is doing structured counting/location lookups, not vector similarity, so forcing it into a vector DB adds complexity without benefit.

---

## 6. API keys — what you actually need

| Component | Needs an API key? | Notes |
|---|---|---|
| Whisper (ASR) | **No** | Runs fully locally via `openai-whisper` package (not the paid OpenAI API) |
| CLIP | **No** | Local via `open_clip` / `transformers`, weights download once from HuggingFace (no key needed, just internet access on first run) |
| BLIP | **No** | Same — HuggingFace weights, local inference |
| YOLOv8 | **No** | `ultralytics` package, local inference, weights auto-download from GitHub releases (no key) |
| PaddleOCR / EasyOCR | **No** | Fully local |
| **Groq (LLM for decouple + final answer)** | **Yes** — `GROQ_API_KEY` | Already required by your current code, free tier available at console.groq.com |
| (Optional) Vision-capable verification model | **Yes, if you use a hosted VLM** | Only needed if you add the optional frame-verification step (§4.6) via a hosted model rather than a local one; skip this entirely and you need zero extra keys |

**Bottom line: you only strictly need the one Groq API key you already have.** Everything else (ASR, CLIP, BLIP, YOLO, OCR) can run 100% locally/offline with open-source weights — no additional paid API key is required to fully match the architecture diagram.

---

## 7. Full implementation plan (phased)

### Phase 0 — Environment setup
- Add to `requirements.txt`: `ultralytics` (YOLOv8), `paddleocr` (or `easyocr`), keep existing `openai-whisper`, `open_clip_torch`, `transformers` (for BLIP), `chromadb`.
- Confirm GPU availability; if CPU-only, use `yolov8n`, `whisper-base`, `ViT-B-32` (lighter variants) to keep processing time reasonable.

### Phase 1 — Bug fixes (§2 above)
1. Fix timestamp computation (Bug #1) — **highest priority, blocks everything else**.
2. Remove/relax the 48-frame cap (Bug #2).
3. Fix `CAPTION_BACKEND` default to `blip` (Bug #3).
4. Add `captions_ready` status flag (Bug #4).
5. Validate: re-upload a known test video, manually confirm 5–6 retrieved timestamps against the actual video by eye.

### Phase 2 — Adaptive frame sampling
Replace fixed-interval sampling with scene-change detection:
```
for each consecutive frame pair:
    compute histogram difference (or SSIM)
    if difference > threshold: mark as keyframe
enforce a minimum spacing (e.g. no closer than 0.5s) and a maximum spacing (e.g. no farther than 3s) to bound total frame count
```
This gives you dense sampling exactly where the video is changing (action, cuts) and sparse sampling during static scenes — directly improving "show me the clip where X happens" accuracy.

### Phase 3 — Add OCR pipeline
1. New `backend/services/ocr_service.py`: run PaddleOCR on each sampled frame, extract text + bounding boxes.
2. New `ocr_index` collection: embed extracted text (reuse your text embedder), store per architecture table in §5.
3. New agent tool: `search_ocr(query, video_id)`.

### Phase 4 — Add Object Detection + counting/location
1. New `backend/services/detection_service.py`: run YOLOv8 on each sampled frame, get `[{class, bbox, confidence}]`.
2. Store in `det_index` (SQLite table: `video_id, frame_number, timestamp, object_class, x, y, w, h, confidence`).
3. New agent tools: `count_objects(class, video_id, time_range)`, `locate_object(class, video_id, time_range)`.

### Phase 5 — Scene graph (relations)
1. New `backend/services/scene_graph_service.py`: given YOLO boxes for a frame, apply spatial heuristics (overlap → "on/in", relative position → "left of/right of/above/below", distance thresholds → "near").
2. Store triples per frame in `scene_graph_index` (SQLite or JSON).
3. New agent tool: `get_relations(video_id, time_range)`.

### Phase 6 — Query Decouple layer (the actual "①" box in your diagram)
1. Before calling the existing multi-tool agent, add a **decouple prompt** step: one Groq call that takes the raw user query and outputs strict JSON, e.g.:
```json
{
  "asr_query": "Mike is ready to go to the library",
  "det_classes": ["book", "table"],
  "ocr_needed": false,
  "answer_type": "count"
}
```
2. Use this structured output to decide *which* of your tools/collections to call (instead of letting the LLM freely pick one tool per turn as it does now) — this directly closes the gap between your current flat tool-calling agent and the diagram's structured decouple-and-route design.

### Phase 7 — Merge & Rearrange + Final Answer
1. Collect the structured outputs from each sub-system (ASR text, OCR text, object counts, object locations, relations) for the relevant time window.
2. Merge into a single structured context block (this is the "Merge and Rearrange" box).
3. Send that merged context + original query to the final LLM call for the answer — same Groq model, but now grounded in structured multi-modal evidence rather than one tool's raw output.

### Phase 8 — Clip/segment boundary tightening (novelty layer, optional but recommended)
1. Once a candidate timestamp is found, cluster temporally-adjacent high-similarity frames into a candidate segment `[start, end]`.
2. Binary-search inward from both ends using a lightweight recheck (CLIP similarity or a VLM yes/no check: "is this event still happening here?") to tighten the true start/end boundary.
3. Return the tightened segment to the frontend for accurate clip playback — this is what actually fixes "show me the clip where this happens" at a *segment* level, not just a single frame.

### Phase 9 — Testing & validation
- Build a small labeled test set: 5–10 videos with manually annotated ground-truth timestamps for a set of test questions.
- Measure: timestamp error (seconds off from ground truth), retrieval precision@k, caption/OCR/DET accuracy spot checks.
- Load-test with a longer video (15–30 min) to confirm adaptive sampling and frame-cap changes hold up.

---

## 8. Summary checklist

- [ ] Fix timestamp bug (Bug #1) — **do this before anything else**
- [ ] Remove/raise frame cap, add adaptive scene-change sampling
- [ ] Default `CAPTION_BACKEND=blip`, keep CLIP for embeddings (they're complementary, not redundant)
- [ ] Fix caption-ready race condition
- [ ] Add OCR service + `ocr_index`
- [ ] Add YOLOv8 detection service + `det_index` (SQLite)
- [ ] Add rule-based scene graph service
- [ ] Add query-decouple step before tool routing
- [ ] Add merge-and-rearrange context assembly before final LLM call
- [ ] Add segment boundary tightening for accurate "show clip" behavior
- [ ] Only API key required throughout: `GROQ_API_KEY` (everything else runs locally, open-source)
