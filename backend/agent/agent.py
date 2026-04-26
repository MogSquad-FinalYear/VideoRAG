"""
VideoRAG — Groq-Powered Agent
Agentic loop: receives query → retrieval → enrich with descriptions → LLM answers.

Strategy A: Text-to-Video — returns top 2 matches with detailed BLIP descriptions
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


# ── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are **Kubrick**, a forensic video analysis AI assistant. You help investigators search and analyze evidence videos.

You have access to a multimodal video retrieval system with 3 indexes:
1. **Caption Index** — Short scene descriptions of each frame
2. **Image Index** — CLIP visual embeddings of each frame (best for finding specific people, objects, appearances)
3. **Speech Index** — Transcribed speech/dialogue from the video

## CRITICAL RULES
1. You are given pre-retrieved search results. Analyze them and provide a clear, professional answer.
2. Frame descriptions are AI-generated — report them as observations, not certainties.
3. Report timestamps in [MM:SS] format. Each frame_number N = timestamp N seconds.
4. Group consecutive matching timestamps into ranges (e.g., [00:11] to [00:15]).
5. Be specific about WHAT you see in each matched frame based on the provided descriptions.
6. Always describe the visual content — colors, objects, people, actions visible in the frame.
7. Keep responses focused and professional."""


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
                min_score=args.get("min_score", 0.25),
            )
            for r in results:
                r["content"] = f"Visual match for uploaded reference image (similarity: {r.get('score', 0):.1%})."
            return json.dumps(results, indent=2)

        elif tool_name == "get_video_clip_info":
            video_id = args["video_id"]
            start_time = args["start_time"]
            end_time = args["end_time"]
            frames_dir = FRAMES_DIR / video_id
            if not frames_dir.exists():
                return json.dumps({"error": f"No frames found for video {video_id}"})

            frame_files = sorted(frames_dir.glob("frame_*.jpg"))
            clip_frames = []
            for f in frame_files:
                num = int(f.stem.split("_")[1])
                timestamp = num  # 1 FPS assumed
                if start_time <= timestamp <= end_time:
                    clip_frames.append({
                        "frame_path": f"/frames/{video_id}/{f.name}",
                        "timestamp": timestamp,
                    })
            return json.dumps({
                "video_id": video_id,
                "start_time": start_time,
                "end_time": end_time,
                "frames": clip_frames,
            }, indent=2)

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        logger.error(f"Tool execution error ({tool_name}): {e}")
        return json.dumps({"error": str(e)})


def _is_speech_query(query: str) -> bool:
    q = query.lower()
    speech_terms = [
        "said", "say", "spoken", "speech", "audio", "voice", "transcript",
        "when did", "quote", "mention", "witness", "told", "heard",
    ]
    return any(term in q for term in speech_terms)


def _collect_sources(query: str, video_id: str = None, image_path: str = None) -> tuple[list[dict], list[str]]:
    """Deterministic multimodal routing to keep retrieval reliable under all conditions."""
    sources: list[dict] = []
    used_strategies: list[str] = []

    if image_path:
        # ── Strategy B: Image-to-Video Search ────────────────────────────
        used_strategies.append("image-to-video")
        image_hits = json.loads(_execute_tool("search_by_image", {
            "image_path": image_path,
            "video_id": video_id,
            "n": 10,
            "min_score": 0.25,
        }))
        if isinstance(image_hits, list):
            sources.extend(image_hits)

        # Cross-reference with text if user provided a meaningful query
        if query and query.strip().lower() not in ("find frames similar to this image", ""):
            used_strategies.append("text-to-video (cross-ref)")
            text_visual_hits = json.loads(_execute_tool("search_by_visual_similarity", {
                "query": query,
                "video_id": video_id,
                "n": 10,
            }))
            if isinstance(text_visual_hits, list):
                sources.extend(text_visual_hits)

    elif _is_speech_query(query):
        # ── Audio/Transcript Search ──────────────────────────────────────
        used_strategies.append("audio/transcript")
        speech_hits = json.loads(_execute_tool("search_transcripts", {
            "query": query,
            "video_id": video_id,
            "n": 12,
        }))
        if isinstance(speech_hits, list):
            sources.extend(speech_hits)

        caption_hits = json.loads(_execute_tool("search_by_caption", {
            "query": query,
            "video_id": video_id,
            "n": 8,
        }))
        if isinstance(caption_hits, list):
            sources.extend(caption_hits)
    else:
        # ── Strategy A: Text-to-Video Search ─────────────────────────────
        used_strategies.append("text-to-video")
        caption_hits = json.loads(_execute_tool("search_by_caption", {
            "query": query,
            "video_id": video_id,
            "n": 12,
        }))
        if isinstance(caption_hits, list):
            sources.extend(caption_hits)

        visual_hits = json.loads(_execute_tool("search_by_visual_similarity", {
            "query": query,
            "video_id": video_id,
            "n": 12,
        }))
        if isinstance(visual_hits, list):
            sources.extend(visual_hits)

    # Deduplicate: keep best score per unique (video_id, frame_number/timestamp)
    best_by_key = {}
    for src in sources:
        key = (src.get("video_id"), src.get("frame_number"), src.get("timestamp"), src.get("start_time"), src.get("end_time"))
        existing = best_by_key.get(key)
        if existing is None or src.get("score", 0) > existing.get("score", 0):
            best_by_key[key] = src

    deduped = list(best_by_key.values())

    # Boost image index results for image-to-video queries
    source_boost = {"image": 0.12, "caption": 0.06, "speech": 0.03}
    deduped.sort(
        key=lambda item: item.get("score", 0) + (source_boost.get(item.get("source_index", ""), 0.0) if image_path else 0.0),
        reverse=True,
    )
    return deduped, used_strategies


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


def _render_answer(query: str, sources: list[dict], strategies: list[str], video_id: str = None, is_image_search: bool = False) -> str:
    """Build a readable text answer from search results.

    This is the fallback when the LLM is unavailable. It produces a
    structured response with detailed descriptions for each match.
    """
    if not sources:
        scope = f" in video {video_id}" if video_id else ""
        return f"I could not find strong evidence matches for this query{scope}. Try a more specific object, person, or spoken phrase."

    max_display = 5 if is_image_search else 2
    display_sources = sources[:max_display]

    if is_image_search:
        # Check if any result has a strong confidence
        best_score = max((s.get("score", 0) for s in display_sources), default=0)

        if best_score < 0.25:
            return ("### No Confident Match Found\n\n"
                    "The uploaded reference image does **not** appear to match any frames in the video with sufficient confidence. "
                    "This could mean:\n"
                    "- The person/object in the reference image is not present in this video\n"
                    "- The video angle or lighting is too different for visual matching\n\n"
                    "Try uploading a clearer reference image or searching with a text description instead.")

        lines = ["### Image Search Results"]
        if best_score < 0.35:
            lines.append("⚠️ **Low confidence** — These matches are weak. The person/object may NOT actually be in the video.")
        lines.append(f"Found **{len(display_sources)}** visually similar frames using: {', '.join(sorted(set(strategies)))}")
        lines.append("")

        for i, hit in enumerate(display_sources):
            vid = hit.get("video_id", "")
            time_str = _format_timestamp(hit)
            desc = hit.get("description", hit.get("content", ""))
            score = hit.get("score", 0)

            confidence = "🟢 High" if score >= 0.50 else ("🟡 Medium" if score >= 0.35 else "🔴 Low")
            lines.append(f"**Match {i+1}** — Video `{vid}` at {time_str} (similarity: {score:.1%}, {confidence})")
            if desc:
                lines.append(f"- **Description**: {desc}")
            lines.append("")

        lines.append("### Summary")
        if best_score >= 0.50:
            lines.append("The uploaded reference image has **strong** visual matches at the timestamps above.")
        elif best_score >= 0.35:
            lines.append("The uploaded reference image has **possible** visual matches. Review the matched frames carefully to confirm identity.")
        else:
            lines.append("⚠️ The matches are **weak**. The uploaded image may not correspond to anyone/anything in this video. Manual verification is strongly recommended.")
    else:
        lines = [f"### Detailed Scene Analysis for: \"{query}\""]
        lines.append(f"Search strategy: {', '.join(sorted(set(strategies)))}")
        lines.append("")

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
            else:
                lines.append(f"- **Description**: Frame matched by visual similarity but no detailed description available.")
            lines.append("")

        lines.append("### Summary")
        lines.append(f"Found **{len(display_sources)}** relevant matches for \"{query}\". Review the matched frames above for forensic analysis.")

    return "\n".join(lines)


def run_agent(query: str, video_id: str = None, image_path: str = None, conversation_history: list = None) -> dict:
    """
    Run the agentic retrieval pipeline:
    1. Collect sources via multimodal search
    2. Enrich top results with detailed BLIP descriptions
    3. Send to LLM for professional answer
    4. Return answer + clips with descriptions

    Strategy A (text search): Returns top 2 matches with detailed descriptions
    Strategy B (image search): Returns all matches with descriptions
    """
    is_image_search = image_path is not None

    # Step 1: Collect raw search results
    sources, strategies = _collect_sources(query=query, video_id=video_id, image_path=image_path)

    # Step 2: Determine how many results to describe and return
    if is_image_search:
        max_results = 5  # Show more for image search
        max_describe = 3
    else:
        max_results = 2  # Strategy A: top 2 only
        max_describe = 2

    # Limit sources to max_results
    sources = sources[:max_results]

    # Step 3: Enrich top results with detailed BLIP descriptions
    if sources:
        logger.info(f"Enriching top {min(max_describe, len(sources))} results with BLIP descriptions...")
        sources = _enrich_with_descriptions(sources, max_describe=max_describe)

    # Step 4: Try LLM for polished answer
    answer = None
    if GROQ_API_KEY and sources:
        try:
            client = _get_groq_client()

            compact_sources = [
                {
                    "video_id": s.get("video_id"),
                    "timestamp": s.get("timestamp"),
                    "start_time": s.get("start_time"),
                    "end_time": s.get("end_time"),
                    "source_index": s.get("source_index"),
                    "description": s.get("description", s.get("content", "")),
                    "score": round(s.get("score", 0), 3),
                    "frame_number": s.get("frame_number"),
                }
                for s in sources
            ]

            if is_image_search:
                task_prompt = (
                    "The user uploaded a reference image and provided this query: "
                    f"'{query}'.\n\n"
                    "Based on the visual match search results below, provide a professional forensic analysis.\n"
                    "Your response MUST be structured exactly like this:\n\n"
                    "### Findings\n"
                    "- State clearly whether the person/object from the reference image appears in the video.\n"
                    "- For EACH matched frame: state the timestamp [MM:SS], describe what is visible (people, objects, colors, actions).\n"
                    "- If the frame description mentions specific items (car, person, weapon), reference them explicitly.\n\n"
                    "### Confirmed Timestamps\n"
                    "- List ALL timestamps where the match appears in [MM:SS] format.\n"
                    "- Group consecutive timestamps into ranges (e.g., [00:05] to [00:08]).\n\n"
                    "### Summary\n"
                    "- State the confidence level (high/medium/low) based on similarity scores.\n"
                    "- Recommend next steps if needed.\n\n"
                    "Be specific and professional. Base your descriptions on the frame descriptions provided, not assumptions.\n"
                    f"Search results: {json.dumps(compact_sources)}"
                )
            else:
                task_prompt = (
                    f"User asked: \"{query}\"\n\n"
                    "Based on the TOP 2 video search results below, provide a detailed forensic analysis.\n"
                    "Your response MUST be structured exactly like this:\n\n"
                    "### Match 1\n"
                    "- **Timestamp**: [MM:SS]\n"
                    "- **Description**: Describe IN DETAIL what is visible in this frame. "
                    "Use the frame description provided to mention specific items: colors of objects, types of vehicles, "
                    "what people are doing, whether the scene is indoors/outdoors, lighting conditions, and any forensic-relevant details.\n\n"
                    "### Match 2\n"
                    "- **Timestamp**: [MM:SS]\n"
                    "- **Description**: Same level of detail for the second match. "
                    "Highlight what makes this frame DIFFERENT from Match 1.\n\n"
                    "### Summary\n"
                    "- Directly answer the user's question based on the evidence from both matches.\n"
                    "- State confidence level (high/medium/low).\n\n"
                    "IMPORTANT: Use the frame descriptions provided — do NOT fabricate details not mentioned in the data.\n"
                    f"Search results: {json.dumps(compact_sources)}"
                )

            polished = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task_prompt}
                ],
                temperature=0.1,
                max_tokens=800,
            )
            text = polished.choices[0].message.content
            if text:
                answer = text
        except Exception as e:
            logger.warning(f"Answer generation error: {e}")

    if not answer:
        answer = _render_answer(query, sources, strategies, video_id, is_image_search)

    return {
        "answer": answer,
        "clips": _build_clips(sources, max_clips=max_results),
        "sources": sources,
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
