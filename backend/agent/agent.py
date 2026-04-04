"""
VideoRAG — Groq-Powered Agent
Agentic loop: receives query → LLM decides tools → execute → LLM answers.
"""
import json
import logging
from groq import Groq

from backend.config import GROQ_API_KEY, GROQ_MODEL, FRAMES_DIR
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


def run_agent(query: str, video_id: str = None, image_path: str = None, conversation_history: list = None) -> dict:
    """
    Run the agentic loop:
    1. Send query + tools to Groq
    2. If tool_calls → execute → send results back
    3. Return final answer with sources
    """
    client = _get_groq_client()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    # Build user message
    user_msg = query
    if video_id:
        user_msg += f"\n\n[Context: The user is asking about video ID: {video_id}]"
    if image_path:
        user_msg += f"\n\n[Context: The user provided a reference image at: {image_path}. Use search_by_image tool with this path.]"

    messages.append({"role": "user", "content": user_msg})

    all_sources = []
    max_iterations = 5  # prevent infinite loops

    for iteration in range(max_iterations):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=4096,
            )
        except Exception as e:
            error_str = str(e)
            logger.error(f"Groq API error: {error_str}")

            # Handle tool_use_failed — extract the failed_generation text
            if "tool_use_failed" in error_str or "failed_generation" in error_str:
                # Try to extract the actual generated text from the error
                try:
                    # Parse the error to find the failed_generation
                    import re
                    match = re.search(r"'failed_generation':\s*'(.+?)'}", error_str)
                    if match:
                        generated_text = match.group(1)
                        logger.info(f"Recovered failed_generation: {generated_text[:100]}...")
                        return {
                            "answer": generated_text,
                            "clips": _build_clips(all_sources),
                            "sources": all_sources[:20],
                        }
                except Exception:
                    pass

                # If we can't parse it, retry without tools
                try:
                    logger.info("Retrying without tools after tool_use_failed...")
                    fallback_response = client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=4096,
                    )
                    answer = fallback_response.choices[0].message.content or "Could not generate a response."
                    return {
                        "answer": answer,
                        "clips": _build_clips(all_sources),
                        "sources": all_sources[:20],
                    }
                except Exception as fallback_err:
                    logger.error(f"Fallback also failed: {fallback_err}")

            return {
                "answer": f"I encountered an error communicating with the AI service: {error_str}",
                "clips": [],
                "sources": [],
            }

        choice = response.choices[0]

        # Fix 2: Check for tool_calls by looking at the message object directly,
        # not just finish_reason (Groq can return tool_calls with finish_reason="stop")
        has_tool_calls = (
            choice.message.tool_calls is not None
            and len(choice.message.tool_calls) > 0
        )

        if has_tool_calls:
            # Add assistant message with tool calls
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse tool args: {tool_call.function.arguments}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"error": "Invalid tool arguments"}),
                    })
                    continue

                # Override video_id if user scoped to a specific video
                if video_id and "video_id" not in fn_args:
                    fn_args["video_id"] = video_id

                logger.info(f"Executing tool: {fn_name}({fn_args})")
                result = _execute_tool(fn_name, fn_args)

                # Track sources
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, list):
                        all_sources.extend(parsed)
                except Exception:
                    pass

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            continue  # loop back for next LLM response

        # Model is done — return final answer
        answer = choice.message.content or "I couldn't generate a response."

        return {
            "answer": answer,
            "clips": _build_clips(all_sources),
            "sources": all_sources[:20],
        }

    # If we exhausted iterations
    return {
        "answer": "I performed multiple search operations but couldn't fully resolve the query. Please try rephrasing.",
        "clips": [],
        "sources": all_sources,
    }


def _build_clips(all_sources: list) -> list:
    """Build deduplicated clip list from search sources."""
    clips = []
    seen_clips = set()
    for src in all_sources:
        vid = src.get("video_id", "")
        ts = src.get("timestamp") or src.get("start_time")
        if vid and ts is not None:
            clip_key = f"{vid}_{ts}"
            if clip_key not in seen_clips:
                seen_clips.add(clip_key)
                clips.append({
                    "video_id": vid,
                    "start_time": float(ts),
                    "end_time": float(src.get("end_time", ts + 2)),
                    "frame_paths": [src.get("frame_path", "")] if src.get("frame_path") else [],
                    "description": src.get("content", ""),
                })
    return clips[:20]  # increased from 10 to 20
