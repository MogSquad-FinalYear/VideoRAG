"""
VideoRAG — Groq-Powered Agent (Full Architecture)
Implements the 3-stage pipeline from the architecture diagram:
  ① Query Decouple — decompose user query into structured retrieval requests
  ② Auxiliary Text Generation & Retrieval — multi-modal search (ASR, OCR, DET, CLIP, Caption)
  ③ Merge & Rearrange + Final Answer — assemble evidence and generate answer

Strategy A: Text-to-Video — returns top matches with detailed descriptions
Strategy B: Image-to-Video — finds visually similar frames and describes them
"""
import json
import logging
from pathlib import Path

from backend.config import GROQ_API_KEY, GROQ_MODEL, FRAMES_DIR
from backend.agent.tools import TOOL_DEFINITIONS
from backend.services import indexing_service, embedding_service

logger = logging.getLogger(__name__)


def _expand_visual_query(query: str) -> str:
    """Expand user query with retrieval-friendly visual synonyms for CLIP text search."""
    q = query.lower()
    expansions = []
    if any(t in q for t in ["bike", "bicycle", "motorbike", "motorcycle"]):
        expansions.append("bike bicycle motorcycle scooter two wheeler")
    if any(t in q for t in ["car", "vehicle", "auto"]):
        expansions.append("car vehicle sedan hatchback")
    if any(t in q for t in ["hit", "hits", "hitting", "crash", "collision", "accident"]):
        expansions.append("collision crash accident impact hitting")
    if not expansions:
        return query
    return f"{query}. Related visual concepts: {'; '.join(expansions)}"


# ── Singleton Groq client ────────────────────────────────────────────────────
_groq_client = None


def _get_groq_client():
    """Return a singleton Groq client instance."""
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Please add it to your .env file. "
                "Get a free key at https://console.groq.com"
            )
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized.")
    return _groq_client


# ── Frame path resolution ────────────────────────────────────────────────────

def _url_to_fs_path(frame_url: str) -> str | None:
    """Convert a browser URL like /frames/vid123/frame_000007.jpg to filesystem path."""
    if not frame_url:
        return None
    try:
        # /frames/{video_id}/{frame_name} → FRAMES_DIR / video_id / frame_name
        parts = frame_url.strip("/").split("/")
        if len(parts) >= 3 and parts[0] == "frames":
            video_id = parts[1]
            frame_name = parts[2]
            fs_path = FRAMES_DIR / video_id / frame_name
            if fs_path.exists():
                return str(fs_path)
    except Exception:
        pass
    return None


def _enrich_with_descriptions(sources: list[dict], max_describe: int = 2) -> list[dict]:
    """Enrich the top N search results with detailed descriptions.

    Strategy (fast-first, memory-safe):
    1. If source already has a meaningful 'content' (from caption index), use it.
    2. Otherwise, look up stored caption from the caption index for that frame.
    3. Try CLIP-based label description (fast, always available since CLIP is loaded).
    4. Try BLIP on-demand (last resort, memory-intensive).
    5. Fallback: generate from metadata.
    """
    described = 0
    for src in sources:
        if described >= max_describe:
            break

        video_id = src.get("video_id", "")
        frame_number = src.get("frame_number")
        frame_path = src.get("frame_path", "")
        existing_content = src.get("content", "")

        # ── Step 1: If this came from caption index, it already has a caption ──
        is_generic = (
            not existing_content
            or "Visual similarity match" in existing_content
            or "Visual match for" in existing_content
        )

        if not is_generic and existing_content:
            src["description"] = existing_content
            described += 1
            continue

        # ── Step 2: Look up stored caption from caption index (fast) ──────────
        if video_id and frame_number is not None:
            stored_caption = indexing_service.get_caption_for_frame(video_id, frame_number)
            if stored_caption:
                src["description"] = stored_caption
                src["content"] = stored_caption
                described += 1
                continue

        # ── Step 3: CLIP-based label description (fast, reliable) ─────────────
        fs_path = _url_to_fs_path(frame_path)
        if fs_path:
            try:
                from backend.services.captioning_service import describe_frame_clip
                clip_desc = describe_frame_clip(fs_path)
                if clip_desc and len(clip_desc) > 20:
                    src["description"] = clip_desc
                    src["content"] = clip_desc
                    described += 1
                    continue
            except Exception as e:
                logger.warning(f"CLIP description failed for {fs_path}: {e}")

        # ── Step 4: Try BLIP on-demand (memory-intensive) ─────────────────────
        if fs_path:
            try:
                from backend.services.captioning_service import describe_frame_detailed
                detailed = describe_frame_detailed(fs_path)
                if detailed:
                    src["description"] = detailed
                    src["content"] = detailed
                    described += 1
                    continue
            except (MemoryError, OSError) as e:
                logger.warning(f"BLIP skipped (memory): {e}")
            except Exception as e:
                logger.warning(f"BLIP description failed for {fs_path}: {e}")

        # ── Step 5: Fallback — generate from metadata ─────────────────────────
        ts = src.get("timestamp")
        if ts is not None:
            minutes = int(ts) // 60
            seconds = int(ts) % 60
            src["description"] = f"Frame captured at [{minutes:02d}:{seconds:02d}] in video {video_id}. Visual content could not be described automatically."
        elif frame_number is not None:
            src["description"] = f"Frame #{frame_number} in video {video_id}. Visual content could not be described automatically."
        else:
            src["description"] = f"Matched frame from video {video_id}."
        described += 1

    return sources


# ── Phase 6: Query Decouple ──────────────────────────────────────────────────

DECOUPLE_PROMPT = """You are a query decomposition engine for a courtroom video analysis system.
Given a user's natural language question about a video, decompose it into a structured JSON retrieval request.

Output ONLY valid JSON with these fields:
{
  "asr_query": "text to search in speech transcripts, or null if speech is not relevant",
  "visual_query": "text description for visual/CLIP search, or null if not needed",
  "caption_query": "text for caption search, or null if not needed",
  "ocr_query": "text to search in on-screen text, or null if not needed",
  "det_classes": ["list", "of", "object", "classes", "to", "detect"],
  "need_relations": false,
  "check_contradictions": false,
  "answer_type": "one of: count, location, description, timestamp, relation, contradiction, general"
}

Rules:
- For counting questions ("how many X"), set answer_type="count" and det_classes=["X"]
- For location questions ("where is X"), set answer_type="location" and det_classes=["X"]
- For relation questions ("what is next to X"), set answer_type="relation" and need_relations=true
- For speech/dialog questions, set asr_query and answer_type="description"
- For visual search ("show me X"), set visual_query and caption_query
- For text/reading questions, set ocr_query
- For consistency/contradiction questions ("did the witness contradict", "testimony consistency", "changed their story"), set check_contradictions=true and answer_type="contradiction"
- Always include visual_query and caption_query for general questions
- Use COCO class names for det_classes when possible (person, car, dog, cat, book, chair, etc.)
"""


def _decouple_query(query: str) -> dict:
    """Phase 6: Decompose user query into structured retrieval request using LLM."""
    default = {
        "asr_query": None,
        "visual_query": query,
        "caption_query": query,
        "ocr_query": None,
        "det_classes": [],
        "need_relations": False,
        "check_contradictions": False,
        "answer_type": "general",
    }

    if not GROQ_API_KEY:
        return default

    try:
        client = _get_groq_client()
        result = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": DECOUPLE_PROMPT},
                {"role": "user", "content": f"User query: \"{query}\""},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        text = result.choices[0].message.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        parsed = json.loads(text)
        # Validate and merge with defaults
        for key in default:
            if key not in parsed:
                parsed[key] = default[key]
        logger.info(f"Query decoupled: answer_type={parsed['answer_type']}, det_classes={parsed['det_classes']}")
        return parsed
    except Exception as e:
        logger.warning(f"Query decouple failed, using defaults: {e}")
        return default


# ── Tool Execution ───────────────────────────────────────────────────────────

def _execute_tool(tool_name: str, args: dict) -> str:
    """Execute a tool call and return the result as a string."""
    try:
        if tool_name == "search_by_caption":
            semantic_hits = indexing_service.search_captions(
                query_text=args["query"],
                n=args.get("n", 10),
                video_id=args.get("video_id"),
            )
            keyword_hits = indexing_service.search_captions_by_keyword(
                keyword=args["query"],
                n=args.get("n", 10),
                video_id=args.get("video_id"),
            )
            merged = semantic_hits + keyword_hits
            merged.sort(key=lambda x: x.get("score", 0), reverse=True)
            dedup = []
            seen = set()
            for hit in merged:
                key = (hit.get("video_id"), hit.get("frame_number"), hit.get("timestamp"), hit.get("source_index"))
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(hit)
            return json.dumps(dedup[:args.get("n", 10)], indent=2)

        elif tool_name == "search_by_visual_similarity":
            expanded_query = _expand_visual_query(args["query"])
            text_embedding = embedding_service.embed_text(expanded_query)
            results = indexing_service.search_images(
                query_embedding=text_embedding,
                n=args.get("n", 10),
                video_id=args.get("video_id"),
            )
            for r in results:
                r["content"] = f"Visual similarity match for: '{args['query']}'"
            return json.dumps(results, indent=2)

        elif tool_name == "search_transcripts":
            results = indexing_service.search_transcripts(
                query_text=args["query"],
                n=args.get("n", 10),
                video_id=args.get("video_id"),
            )
            return json.dumps(results, indent=2)

        elif tool_name == "search_by_image":
            image_embedding = embedding_service.embed_image(args["image_path"])
            results = indexing_service.search_images(
                query_embedding=image_embedding,
                n=args.get("n", 10),
                video_id=args.get("video_id"),
                min_score=args.get("min_score", 0.35),
            )
            for r in results:
                r["content"] = f"Visual match for uploaded reference image (similarity: {r.get('score', 0):.1%})."
            return json.dumps(results, indent=2)

        elif tool_name == "get_video_clip_info":
            video_id = args["video_id"]
            start_time = args["start_time"]
            end_time = args["end_time"]

            # Bug #5 fix: use real timestamps from ChromaDB metadata
            try:
                image_col, _, _ = indexing_service.get_collections()
                results = image_col.get(
                    where={"video_id": video_id},
                    include=["metadatas"],
                )
                clip_frames = []
                if results and results["metadatas"]:
                    for meta in results["metadatas"]:
                        ts = meta.get("timestamp", 0)
                        if start_time <= ts <= end_time:
                            clip_frames.append({
                                "frame_path": meta.get("frame_path", ""),
                                "timestamp": ts,
                                "frame_number": meta.get("frame_number"),
                            })
                clip_frames.sort(key=lambda x: x["timestamp"])
            except Exception as e:
                logger.warning(f"Failed to get clip info from index: {e}")
                frames_dir = FRAMES_DIR / video_id
                clip_frames = []
                if frames_dir.exists():
                    for f in sorted(frames_dir.glob("frame_*.jpg")):
                        num = int(f.stem.split("_")[1])
                        clip_frames.append({
                            "frame_path": f"/frames/{video_id}/{f.name}",
                            "timestamp": num,
                            "frame_number": num,
                        })
                    clip_frames = [cf for cf in clip_frames if start_time <= cf["timestamp"] <= end_time]

            return json.dumps({
                "video_id": video_id,
                "start_time": start_time,
                "end_time": end_time,
                "frames": clip_frames,
            }, indent=2)

        # ── Phase 3: OCR search ──────────────────────────────────────────────
        elif tool_name == "search_ocr":
            results = indexing_service.search_ocr(
                query_text=args["query"],
                n=args.get("n", 10),
                video_id=args.get("video_id"),
            )
            return json.dumps(results, indent=2)

        # ── Phase 4: Object counting/location ────────────────────────────────
        elif tool_name == "count_objects":
            from backend.services import detection_db
            result = detection_db.count_objects(
                video_id=args["video_id"],
                class_name=args.get("class_name"),
                start_time=args.get("start_time"),
                end_time=args.get("end_time"),
            )
            return json.dumps(result, indent=2)

        elif tool_name == "locate_object":
            from backend.services import detection_db
            results = detection_db.locate_objects(
                video_id=args["video_id"],
                class_name=args["class_name"],
                start_time=args.get("start_time"),
                end_time=args.get("end_time"),
            )
            return json.dumps(results, indent=2)

        # ── Phase 5: Scene graph relations ───────────────────────────────────
        elif tool_name == "get_relations":
            from backend.services import detection_db
            results = detection_db.get_relations(
                video_id=args["video_id"],
                start_time=args.get("start_time"),
                end_time=args.get("end_time"),
            )
            return json.dumps(results, indent=2)

        # ── Phase 7: Contradiction checking ────────────────────────────────
        elif tool_name == "check_contradictions":
            from backend.services import testimony_db
            contradictions = testimony_db.get_contradictions(
                case_id=args["case_id"],
                speaker_id=args.get("speaker_id"),
            )
            return json.dumps(contradictions, indent=2)

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        logger.error(f"Tool execution error ({tool_name}): {e}")
        return json.dumps({"error": str(e)})


# ── Phase 6: Structured Source Collection (Query Decouple → Route) ───────────

def _collect_sources_decoupled(query: str, decouple: dict, video_id: str = None,
                                image_path: str = None,
                                case_id: str = None) -> tuple[list[dict], list[str], dict]:
    """
    Phase 6: Collect sources using structured decouple output to decide which
    tools/collections to query instead of flat heuristic routing.

    Returns:
        (sources, strategies, structured_evidence)
    """
    sources: list[dict] = []
    used_strategies: list[str] = []
    structured_evidence = {
        "asr_results": [],
        "ocr_results": [],
        "detection_results": {},
        "location_results": [],
        "relation_results": [],
        "contradiction_results": [],
    }

    # ── Image-to-Video search (Strategy B) ───────────────────────────────
    if image_path:
        used_strategies.append("image-to-video")
        image_hits = json.loads(_execute_tool("search_by_image", {
            "image_path": image_path,
            "video_id": video_id,
            "n": 10,
            "min_score": 0.35,
        }))
        if isinstance(image_hits, list):
            sources.extend(image_hits)

        if query and query.strip().lower() not in ("find frames similar to this image", ""):
            used_strategies.append("text-to-video (cross-ref)")
            text_visual_hits = json.loads(_execute_tool("search_by_visual_similarity", {
                "query": query,
                "video_id": video_id,
                "n": 10,
            }))
            if isinstance(text_visual_hits, list):
                sources.extend(text_visual_hits)
        return sources, used_strategies, structured_evidence

    # ── ASR search ───────────────────────────────────────────────────────
    if decouple.get("asr_query"):
        used_strategies.append("speech/transcript")
        speech_hits = json.loads(_execute_tool("search_transcripts", {
            "query": decouple["asr_query"],
            "video_id": video_id,
            "n": 12,
        }))
        if isinstance(speech_hits, list):
            sources.extend(speech_hits)
            structured_evidence["asr_results"] = speech_hits

    # ── Visual / Caption search ──────────────────────────────────────────
    if decouple.get("visual_query"):
        used_strategies.append("visual-similarity")
        visual_hits = json.loads(_execute_tool("search_by_visual_similarity", {
            "query": decouple["visual_query"],
            "video_id": video_id,
            "n": 12,
        }))
        if isinstance(visual_hits, list):
            sources.extend(visual_hits)

    if decouple.get("caption_query"):
        used_strategies.append("caption-search")
        caption_hits = json.loads(_execute_tool("search_by_caption", {
            "query": decouple["caption_query"],
            "video_id": video_id,
            "n": 12,
        }))
        if isinstance(caption_hits, list):
            sources.extend(caption_hits)

    # ── OCR search ───────────────────────────────────────────────────────
    if decouple.get("ocr_query"):
        used_strategies.append("ocr-text")
        ocr_hits = json.loads(_execute_tool("search_ocr", {
            "query": decouple["ocr_query"],
            "video_id": video_id,
            "n": 10,
        }))
        if isinstance(ocr_hits, list):
            sources.extend(ocr_hits)
            structured_evidence["ocr_results"] = ocr_hits

    # ── Object detection (counting / location) ───────────────────────────
    if decouple.get("det_classes") and video_id:
        for cls in decouple["det_classes"]:
            if decouple.get("answer_type") == "count":
                used_strategies.append(f"object-count({cls})")
                count_result = json.loads(_execute_tool("count_objects", {
                    "class_name": cls,
                    "video_id": video_id,
                }))
                structured_evidence["detection_results"][cls] = count_result

            if decouple.get("answer_type") in ("location", "relation"):
                used_strategies.append(f"object-locate({cls})")
                loc_results = json.loads(_execute_tool("locate_object", {
                    "class_name": cls,
                    "video_id": video_id,
                }))
                if isinstance(loc_results, list):
                    structured_evidence["location_results"].extend(loc_results)

    # ── Scene graph relations ────────────────────────────────────────────
    if decouple.get("need_relations") and video_id:
        used_strategies.append("scene-graph")
        relations = json.loads(_execute_tool("get_relations", {
            "video_id": video_id,
        }))
        if isinstance(relations, list):
            structured_evidence["relation_results"] = relations

    # ── Contradiction checking (Phase 7) ───────────────────────────────
    if decouple.get("check_contradictions") and case_id:
        used_strategies.append("contradiction-check")
        contradiction_results = json.loads(_execute_tool("check_contradictions", {
            "case_id": case_id,
        }))
        if isinstance(contradiction_results, list):
            structured_evidence["contradiction_results"] = contradiction_results

    # Deduplicate sources: keep best score per unique (video_id, frame_number/timestamp)
    best_by_key = {}
    for src in sources:
        key = (src.get("video_id"), src.get("frame_number"), src.get("timestamp"),
               src.get("start_time"), src.get("end_time"))
        existing = best_by_key.get(key)
        if existing is None or src.get("score", 0) > existing.get("score", 0):
            best_by_key[key] = src

    deduped = list(best_by_key.values())

    # Boost image index results for image-to-video queries
    source_boost = {"image": 0.12, "caption": 0.06, "speech": 0.03, "ocr": 0.04}
    deduped.sort(
        key=lambda item: item.get("score", 0) + (source_boost.get(item.get("source_index", ""), 0.0) if image_path else 0.0),
        reverse=True,
    )
    return deduped, used_strategies, structured_evidence


# ── Phase 7: Merge & Rearrange ───────────────────────────────────────────────

def _merge_and_rearrange(query: str, sources: list[dict], structured_evidence: dict,
                          decouple: dict, strategies: list[str],
                          video_id: str = None, is_image_search: bool = False) -> str:
    """
    Phase 7: Merge all structured outputs into a single context block,
    then send to LLM for the final answer.

    This is the '③ Integration and Generation' box in the architecture diagram.
    """
    # Build structured context from all sub-systems
    context_parts = []

    # Frame-level evidence (captions, visual descriptions)
    if sources:
        frame_evidence = []
        for i, src in enumerate(sources[:8]):  # Limit to top 8
            ts = src.get("timestamp")
            ts_str = f"[{int(ts)//60:02d}:{int(ts)%60:02d}]" if ts is not None else "[?]"
            desc = src.get("description", src.get("content", ""))
            source_type = src.get("source_index", "unknown")
            score = src.get("score", 0)
            frame_evidence.append(f"  - {ts_str} ({source_type}, score={score:.2f}): {desc}")
        if frame_evidence:
            context_parts.append("## Frame Evidence\n" + "\n".join(frame_evidence))

    # ASR evidence
    asr = structured_evidence.get("asr_results", [])
    if asr:
        asr_lines = []
        for seg in asr[:5]:
            st = seg.get("start_time", 0)
            et = seg.get("end_time", 0)
            text = seg.get("content", "")
            asr_lines.append(f"  - [{int(st)//60:02d}:{int(st)%60:02d} – {int(et)//60:02d}:{int(et)%60:02d}] \"{text}\"")
        context_parts.append("## Speech/Transcript Evidence\n" + "\n".join(asr_lines))

    # OCR evidence
    ocr = structured_evidence.get("ocr_results", [])
    if ocr:
        ocr_lines = []
        for item in ocr[:5]:
            ts = item.get("timestamp")
            ts_str = f"[{int(ts)//60:02d}:{int(ts)%60:02d}]" if ts is not None else "[?]"
            text = item.get("content", "")
            ocr_lines.append(f"  - {ts_str}: {text}")
        context_parts.append("## On-Screen Text (OCR) Evidence\n" + "\n".join(ocr_lines))

    # Detection counts
    det = structured_evidence.get("detection_results", {})
    if det:
        det_lines = []
        for cls, result in det.items():
            if isinstance(result, dict):
                counts = result.get("counts", {})
                for obj_cls, cnt in counts.items():
                    det_lines.append(f"  - {obj_cls}: max {cnt} instances in a single frame")
        if det_lines:
            context_parts.append("## Object Detection Counts\n" + "\n".join(det_lines))

    # Object locations
    locs = structured_evidence.get("location_results", [])
    if locs:
        loc_lines = []
        for loc in locs[:5]:
            ts = loc.get("timestamp", 0)
            ts_str = f"[{int(ts)//60:02d}:{int(ts)%60:02d}]"
            cls = loc.get("class_name", "")
            bbox = loc.get("bbox", [])
            loc_lines.append(f"  - {ts_str}: {cls} at bbox {bbox} (confidence: {loc.get('confidence', 0):.2f})")
        context_parts.append("## Object Locations\n" + "\n".join(loc_lines))

    # Scene graph relations
    rels = structured_evidence.get("relation_results", [])
    if rels:
        rel_lines = []
        for rel in rels[:10]:
            ts = rel.get("timestamp", 0)
            ts_str = f"[{int(ts)//60:02d}:{int(ts)%60:02d}]"
            triple = rel.get("triple", "")
            rel_lines.append(f"  - {ts_str}: {triple}")
        context_parts.append("## Object Relations (Scene Graph)\n" + "\n".join(rel_lines))

    # Contradiction evidence (Phase 7)
    contradictions = structured_evidence.get("contradiction_results", [])
    if contradictions:
        contra_lines = []
        for c in contradictions[:5]:
            speaker = c.get("speaker_id", "unknown")
            conf = c.get("confidence", 0)
            stmt_a = c.get("stmt_a_text", "")
            stmt_b = c.get("stmt_b_text", "")
            ts_a = c.get("stmt_a_timestamp", 0)
            ts_b = c.get("stmt_b_timestamp", 0)
            vid_a = c.get("stmt_a_video_id", "")
            vid_b = c.get("stmt_b_video_id", "")
            contra_lines.append(
                f"  - **{speaker}** (confidence: {conf:.0%}):\n"
                f"    Session 1 [{int(ts_a)//60:02d}:{int(ts_a)%60:02d}] (video {vid_a}): \"{stmt_a}\"\n"
                f"    Session 2 [{int(ts_b)//60:02d}:{int(ts_b)%60:02d}] (video {vid_b}): \"{stmt_b}\"\n"
                f"    Explanation: {c.get('explanation', 'N/A')}"
            )
        context_parts.append("## Testimony Contradictions\n" + "\n".join(contra_lines))

    merged_context = "\n\n".join(context_parts) if context_parts else "No evidence found."

    # Build the final LLM prompt
    answer_type = decouple.get("answer_type", "general")

    system_prompt = """You are **Kubrick**, a forensic video analysis AI assistant for courtroom proceedings.

You analyze multimodal evidence from a video retrieval system. You receive structured evidence from:
1. **Caption/Visual Index** — AI-generated scene descriptions and CLIP similarity matches
2. **Speech/ASR Index** — Transcribed speech and dialogue with speaker/role tags
3. **OCR Index** — On-screen text extracted from frames
4. **Object Detection** — YOLO-detected objects with counts and locations
5. **Scene Graph** — Spatial relationships between detected objects
6. **Testimony Memory** — Cross-session contradiction detection results

## CRITICAL RULES
1. Base your answer ONLY on the provided evidence — do NOT fabricate details.
2. Report timestamps in [MM:SS] format.
3. Group consecutive timestamps into ranges.
4. Be specific about what the evidence shows.
5. State confidence level when relevant.
6. For counting questions, use the detection count data.
7. For location questions, describe spatial positions from detection data.
8. For relation questions, reference the scene graph triples.
9. When speaker/role information is available, attribute quotes to the speaker (e.g., "The witness stated...").
10. For contradiction questions, present both conflicting statements with their timestamps and sessions."""

    user_prompt = f"""User query: "{query}"
Answer type: {answer_type}
Search strategies used: {', '.join(strategies)}

{merged_context}

Based on the above evidence, provide a clear, professional answer to the user's query."""

    if not GROQ_API_KEY or not context_parts:
        return _render_answer_fallback(query, sources, strategies, video_id, is_image_search,
                                        structured_evidence)

    try:
        client = _get_groq_client()
        result = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        text = result.choices[0].message.content
        if text:
            return text
    except Exception as e:
        logger.warning(f"Final answer generation failed: {e}")

    return _render_answer_fallback(query, sources, strategies, video_id, is_image_search,
                                    structured_evidence)


# ── Fallback answer rendering ────────────────────────────────────────────────

def _format_timestamp(hit: dict) -> str:
    """Format a hit's timestamp into [MM:SS] string."""
    ts = hit.get("timestamp")
    if ts is not None:
        minutes = int(ts) // 60
        seconds = int(ts) % 60
        return f"[{minutes:02d}:{seconds:02d}]"
    st = hit.get("start_time")
    et = hit.get("end_time")
    if st is not None and et is not None:
        return f"[{int(st)//60:02d}:{int(st)%60:02d} – {int(et)//60:02d}:{int(et)%60:02d}]"
    return "[unknown]"


def _render_answer_fallback(query: str, sources: list[dict], strategies: list[str],
                             video_id: str = None, is_image_search: bool = False,
                             structured_evidence: dict = None) -> str:
    """Build a readable text answer when LLM is unavailable."""
    if not sources and not structured_evidence:
        scope = f" in video {video_id}" if video_id else ""
        return f"I could not find strong evidence matches for this query{scope}. Try a more specific object, person, or spoken phrase."

    lines = [f"### Analysis for: \"{query}\""]
    lines.append(f"Search strategies: {', '.join(sorted(set(strategies)))}")
    lines.append("")

    # Show detection counts if available
    if structured_evidence:
        det = structured_evidence.get("detection_results", {})
        if det:
            lines.append("### Object Counts")
            for cls, result in det.items():
                if isinstance(result, dict):
                    counts = result.get("counts", {})
                    for obj_cls, cnt in counts.items():
                        lines.append(f"- **{obj_cls}**: {cnt} instances (max in any frame)")
            lines.append("")

        rels = structured_evidence.get("relation_results", [])
        if rels:
            lines.append("### Object Relations")
            for rel in rels[:5]:
                lines.append(f"- {rel.get('triple', '')}")
            lines.append("")

    # Show top matches
    display_sources = sources[:5] if is_image_search else sources[:3]
    for i, hit in enumerate(display_sources):
        vid = hit.get("video_id", "")
        time_str = _format_timestamp(hit)
        desc = hit.get("description", hit.get("content", ""))
        score = hit.get("score", 0)
        source_idx = hit.get("source_index", "")

        lines.append(f"### Match {i+1}")
        lines.append(f"- **Timestamp**: {time_str} in video `{vid}`")
        lines.append(f"- **Confidence**: {score:.1%} ({source_idx} index)")
        if desc:
            lines.append(f"- **Description**: {desc}")
        lines.append("")

    return "\n".join(lines)


# ── Main Agent Entry Point ───────────────────────────────────────────────────

def run_agent(query: str, video_id: str = None, image_path: str = None,
              conversation_history: list = None, case_id: str = None) -> dict:
    """
    Run the full 3-stage VideoRAG pipeline:
    ① Query Decouple — decompose user query
    ② Multimodal Retrieval — search across all indexes
    ③ Merge & Rearrange — assemble evidence and generate answer
    """
    is_image_search = image_path is not None

    # ── Resolve case_id if not provided but video_id is known ────────────
    if not case_id and video_id:
        try:
            from backend.services import testimony_db
            case_id = testimony_db.get_case_id_for_video(video_id)
        except Exception:
            pass

    # ── Stage ①: Query Decouple ──────────────────────────────────────────
    if is_image_search:
        decouple = {
            "asr_query": None,
            "visual_query": query,
            "caption_query": query if query else None,
            "ocr_query": None,
            "det_classes": [],
            "need_relations": False,
            "check_contradictions": False,
            "answer_type": "description",
        }
    else:
        decouple = _decouple_query(query)

        # Heuristic fallback: detect contradiction queries by keyword
        q_lower = query.lower()
        contradiction_keywords = [
            "contradict", "contradiction", "inconsisten", "consistent",
            "changed", "lied", "lying", "conflicting", "discrepan",
            "testimony", "recant",
        ]
        if any(kw in q_lower for kw in contradiction_keywords):
            decouple["check_contradictions"] = True
            if decouple.get("answer_type") == "general":
                decouple["answer_type"] = "contradiction"

    # ── Stage ②: Multimodal Retrieval ────────────────────────────────────
    sources, strategies, structured_evidence = _collect_sources_decoupled(
        query=query, decouple=decouple, video_id=video_id,
        image_path=image_path, case_id=case_id,
    )

    # ── Hard confidence gate for image search ──
    if is_image_search:
        best_score = max((s.get("score", 0) for s in sources), default=0) if sources else 0
        if best_score < 0.30:
            logger.info(f"Image search: best score {best_score:.3f} < 0.30 — no confident match.")
            return {
                "answer": (
                    "### No Match Found\n\n"
                    "The uploaded reference image does **not** match any frames in the video.\n\n"
                    "This means the person or object in your image is **not present** in this footage.\n\n"
                    "**Suggestions:**\n"
                    "- Try a different reference image with better lighting or angle\n"
                    "- Try a text description instead (e.g., \"person in red shirt\")\n"
                    "- Upload a different video that may contain the subject"
                ),
                "clips": [],
                "sources": [],
            }

    # Determine how many results to describe
    if is_image_search:
        max_results = 5
        max_describe = 3
    else:
        max_results = 5
        max_describe = 3

    sources = sources[:max_results]

    # Enrich top results with detailed descriptions
    if sources:
        logger.info(f"Enriching top {min(max_describe, len(sources))} results with descriptions...")
        sources = _enrich_with_descriptions(sources, max_describe=max_describe)

    # ── Stage ③: Merge & Rearrange + Final Answer ────────────────────────
    answer = _merge_and_rearrange(
        query=query, sources=sources, structured_evidence=structured_evidence,
        decouple=decouple, strategies=strategies,
        video_id=video_id, is_image_search=is_image_search,
    )

    # ── Stage ④: Citation Verification (Phase 8) ─────────────────────────
    citations = []
    if video_id and answer and not is_image_search:
        try:
            from backend.services import citation_service
            citations = citation_service.verify_and_store_citations(
                answer=answer, video_id=video_id,
            )
        except Exception as e:
            logger.warning("Citation verification failed (non-fatal): %s", e)

    return {
        "answer": answer,
        "clips": _build_clips(sources, max_clips=max_results),
        "sources": sources,
        "citations": citations,
        "contradictions": [
            {
                "contradiction_id": c.get("id", ""),
                "speaker_id": c.get("speaker_id", ""),
                "statement_a": c.get("stmt_a_text", ""),
                "statement_b": c.get("stmt_b_text", ""),
                "confidence": c.get("confidence", 0),
                "explanation": c.get("explanation", ""),
                "video_a": c.get("stmt_a_video_id"),
                "video_b": c.get("stmt_b_video_id"),
                "timestamp_a": c.get("stmt_a_timestamp"),
                "timestamp_b": c.get("stmt_b_timestamp"),
            }
            for c in structured_evidence.get("contradiction_results", [])
        ],
    }


def _build_clips(all_sources: list, max_clips: int = 5) -> list:
    """Build deduplicated clip list from search sources with descriptions."""
    clips = []

    def _frame_exists(video_id: str, frame_path: str) -> bool:
        if not frame_path:
            return False
        frame_name = frame_path.split("/")[-1]
        if not frame_name:
            return False
        return (FRAMES_DIR / video_id / frame_name).exists()

    for src in all_sources:
        vid = src.get("video_id", "")
        ts = float(src.get("timestamp") or src.get("start_time") or 0.0)
        frame_path = src.get("frame_path", "")
        description = src.get("description", src.get("content", ""))

        if vid and _frame_exists(vid, frame_path):
            merged = False
            for clip in clips:
                # Merge clips that are close to each other (within 4 seconds)
                if clip["video_id"] == vid and abs(clip["start_time"] - ts) <= 4.0:
                    clip["start_time"] = min(clip["start_time"], ts)
                    clip["end_time"] = max(clip["end_time"], ts + 2.0)
                    if frame_path not in clip["frame_paths"]:
                        clip["frame_paths"].append(frame_path)
                    # Keep the best description
                    if description and len(description) > len(clip.get("description", "")):
                        clip["description"] = description
                    merged = True
                    break

            if not merged:
                clips.append({
                    "video_id": vid,
                    "start_time": ts,
                    "end_time": float(src.get("end_time", ts + 2.0)),
                    "frame_paths": [frame_path] if frame_path else [],
                    "description": description,
                })

    return clips[:max_clips]
