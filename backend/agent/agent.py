"""
VideoRAG — Groq-Powered Agent
Agentic loop: receives query → LLM decides tools → execute → LLM answers.
"""
import json
import logging
from groq import Groq

from backend.config import GROQ_API_KEY, GROQ_MODEL, FRAMES_DIR, ENABLE_LLM_POLISH
from backend.agent.tools import TOOL_DEFINITIONS
from backend.services import indexing_service, embedding_service

logger = logging.getLogger(__name__)

# ── Singleton Groq client (Fix 7) ────────────────────────────────────────────
_groq_client = None


def _get_groq_client() -> Groq:
    """Return a singleton Groq client instance."""
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Please add it to your .env file. "
                "Get a free key at https://console.groq.com"
            )
        _groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized.")
    return _groq_client


# ── System Prompt (Fix 16) ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are **Kubrick**, a forensic video analysis AI assistant. You help investigators search and analyze evidence videos.

You have access to a multimodal video retrieval system with 3 indexes:
1. **Caption Index** — Short scene descriptions of each frame (can be generic/inaccurate)
2. **Image Index** — CLIP visual embeddings of each frame (best for finding specific people, objects, appearances)
3. **Speech Index** — Transcribed speech/dialogue from the video

## Query Strategy — FOLLOW STRICTLY
- For questions about **specific people, characters, or visual appearances**: Use `search_by_visual_similarity` as PRIMARY. The caption index has generic descriptions that often miss specific identities.
- For questions about **general scenes or activities**: Use `search_by_caption`.
- For questions about **spoken words or dialogue**: Use `search_transcripts`.
- **ALWAYS call at least 2 different search tools** for any question about video content. Cross-reference results from visual similarity AND captions for the most accurate answer.

## CRITICAL RULES
1. You MUST call search tools before answering. NEVER answer from assumptions.
2. ALWAYS use `search_by_visual_similarity` when the question is about a person, character, or specific object.
3. For every search, set n=10 or higher to find ALL occurrences.
4. **Only report frames where you are confident about the match.** If the caption or visual match seems generic or unrelated, EXCLUDE it. Quality over quantity.
5. Report ALL matching timestamps as [MM:SS] format.
6. Each frame is sampled at 1fps, so frame_number N = timestamp N seconds.
7. When results from different tools disagree, prefer the visual similarity results for appearance questions.
8. Group consecutive matching timestamps into ranges (e.g., [00:11] to [00:15]) instead of listing each second."""


def _execute_tool(tool_name: str, args: dict) -> str:
    """Execute a tool call and return the result as a string."""
    try:
        if tool_name == "search_by_caption":
            results = indexing_service.search_captions(
                query_text=args["query"],
                n=args.get("n", 10),
                video_id=args.get("video_id"),
            )
            return json.dumps(results, indent=2)

        elif tool_name == "search_by_visual_similarity":
            text_embedding = embedding_service.embed_text(args["query"])
            results = indexing_service.search_images(
                query_embedding=text_embedding,
                n=args.get("n", 10),
                video_id=args.get("video_id"),
            )
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
            )
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
                # Extract frame number from filename
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


def _build_local_answer(query: str, video_id: str = None, image_path: str = None) -> dict:
    """Fallback retrieval path when the LLM is unavailable."""
    query_lower = query.lower()
    sources = []

    if image_path:
        sources.extend(json.loads(_execute_tool("search_by_image", {
            "image_path": image_path,
            "video_id": video_id,
            "n": 10,
        })))

    if any(keyword in query_lower for keyword in ["said", "say", "spoken", "tell", "audio", "voice", "witness", "mention", "mentioned"]):
        sources.extend(json.loads(_execute_tool("search_transcripts", {
            "query": query,
            "video_id": video_id,
            "n": 10,
        })))
    elif any(keyword in query_lower for keyword in ["person", "man", "woman", "car", "vehicle", "object", "bag", "weapon", "blue", "red", "green", "shirt", "jacket", "face"]):
        sources.extend(json.loads(_execute_tool("search_by_visual_similarity", {
            "query": query,
            "video_id": video_id,
            "n": 10,
        })))
        sources.extend(json.loads(_execute_tool("search_by_caption", {
            "query": query,
            "video_id": video_id,
            "n": 10,
        })))
    else:
        sources.extend(json.loads(_execute_tool("search_by_caption", {
            "query": query,
            "video_id": video_id,
            "n": 10,
        })))
        sources.extend(json.loads(_execute_tool("search_by_visual_similarity", {
            "query": query,
            "video_id": video_id,
            "n": 10,
        })))

    deduped = []
    seen = set()
    for src in sources:
        key = (
            src.get("video_id"),
            src.get("frame_number"),
            src.get("timestamp"),
            src.get("start_time"),
            src.get("end_time"),
            src.get("content"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(src)

    top_hits = deduped[:5]
    if not top_hits:
        return {
            "answer": "I could not find matching evidence in the current indexes.",
            "clips": [],
            "sources": [],
        }

    summary_lines = ["I found the following relevant evidence:"]
    for hit in top_hits:
        timestamp = hit.get("timestamp")
        start_time = hit.get("start_time")
        end_time = hit.get("end_time")
        if timestamp is not None:
            summary_lines.append(f"- {hit.get('video_id')} at {timestamp:.0f}s")
        elif start_time is not None and end_time is not None:
            summary_lines.append(f"- {hit.get('video_id')} from {start_time:.0f}s to {end_time:.0f}s")
        else:
            summary_lines.append(f"- {hit.get('video_id')}")

    return {
        "answer": "\n".join(summary_lines),
        "clips": _build_clips(deduped),
        "sources": deduped[:20],
    }


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
        used_strategies.append("image-to-video")
        image_hits = json.loads(_execute_tool("search_by_image", {
            "image_path": image_path,
            "video_id": video_id,
            "n": 12,
        }))
        if isinstance(image_hits, list):
            sources.extend(image_hits)

    if _is_speech_query(query):
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

    deduped: list[dict] = []
    seen = set()
    for src in sources:
        key = (
            src.get("video_id"),
            src.get("frame_number"),
            src.get("timestamp"),
            src.get("start_time"),
            src.get("end_time"),
            src.get("content"),
            src.get("source_index"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(src)

    deduped.sort(key=lambda item: item.get("score", 0), reverse=True)
    return deduped, used_strategies


def _render_answer(query: str, sources: list[dict], strategies: list[str], video_id: str = None) -> str:
    if not sources:
        scope = f" in video {video_id}" if video_id else ""
        return f"I could not find strong evidence matches for this query{scope}. Try a more specific object, person, or spoken phrase."

    lines = [f"Found evidence using: {', '.join(sorted(set(strategies)))}"]

    return "\n".join(lines)


def run_agent(query: str, video_id: str = None, image_path: str = None, conversation_history: list = None) -> dict:
    """
    Run the agentic loop:
    1. Send query + tools to Groq
    2. If tool_calls → execute → send results back
    3. Return final answer with sources
    """
    sources, strategies = _collect_sources(query=query, video_id=video_id, image_path=image_path)
    answer = _render_answer(query=query, sources=sources, strategies=strategies, video_id=video_id)

    # Optional concise LLM polish (no tool-calling) if explicitly enabled.
    if ENABLE_LLM_POLISH and GROQ_API_KEY and sources:
        try:
            client = _get_groq_client()
            compact_sources = [
                {
                    "video_id": s.get("video_id"),
                    "timestamp": s.get("timestamp"),
                    "start_time": s.get("start_time"),
                    "end_time": s.get("end_time"),
                    "source_index": s.get("source_index"),
                    "content": s.get("content", "")[:120],
                }
                for s in sources[:8]
            ]
            prompt = (
                "Summarize forensic video retrieval results in 4-6 bullet points. "
                "Do not ask follow-up questions. Mention timestamps and source indexes.\n"
                f"Query: {query}\n"
                f"Results: {json.dumps(compact_sources)}"
            )
            polished = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
            )
            text = polished.choices[0].message.content
            if text:
                answer = text
        except Exception as e:
            logger.warning(f"Answer polishing skipped due to Groq error: {e}")

    return {
        "answer": answer,
        "clips": _build_clips(sources),
        "sources": sources[:20],
    }


def _build_clips(all_sources: list) -> list:
    """Build deduplicated clip list from search sources with existing frame images only."""
    clips = []
    seen_clips = set()

    def _frame_exists(video_id: str, frame_path: str) -> bool:
        if not frame_path:
            return False
        frame_name = frame_path.split("/")[-1]
        if not frame_name:
            return False
        return (FRAMES_DIR / video_id / frame_name).exists()

    for src in all_sources:
        vid = src.get("video_id", "")
        ts = src.get("timestamp") or src.get("start_time")
        frame_path = src.get("frame_path", "")
        if vid and ts is not None and _frame_exists(vid, frame_path):
            clip_key = f"{vid}_{ts}"
            if clip_key not in seen_clips:
                seen_clips.add(clip_key)
                clips.append({
                    "video_id": vid,
                    "start_time": float(ts),
                    "end_time": float(src.get("end_time", ts + 2)),
                    "frame_paths": [frame_path],
                    "description": src.get("content", ""),
                })
    return clips[:20]  # increased from 10 to 20
