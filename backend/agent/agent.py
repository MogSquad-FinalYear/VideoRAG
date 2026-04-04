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

SYSTEM_PROMPT = """You are **Kubrick**, a forensic video analysis AI assistant. You help investigators search and analyze evidence videos.

You have access to a multimodal video retrieval system with 3 indexes:
1. **Caption Index** — Scene descriptions generated from video frames (BLIP captions)
2. **Image Index** — Visual embeddings of frames (CLIP vectors) for visual similarity search
3. **Speech Index** — Transcribed audio/speech from videos (Whisper)

When answering queries:
- Use `search_by_caption` for questions about what is SEEN in the video (objects, people, scenes, actions)
- Use `search_by_visual_similarity` for precise visual matching (colors, specific objects, appearances)
- Use `search_transcripts` for questions about what was SAID or SPOKEN in the video
- Use `search_by_image` when the user provides a reference image to find similar frames
- Use `get_video_clip_info` to get details about a specific time range
- You can call MULTIPLE tools to cross-reference visual and audio evidence
- Always mention timestamps when reporting findings
- Be precise and factual — this is forensic evidence analysis
- If results are inconclusive, say so honestly

Format your responses clearly with timestamps like [00:15 - 00:23] when referencing specific moments."""


def _execute_tool(tool_name: str, args: dict) -> str:
    """Execute a tool call and return the result as a string."""
    try:
        if tool_name == "search_by_caption":
            results = indexing_service.search_captions(
                query_text=args["query"],
                n=args.get("n", 5),
                video_id=args.get("video_id"),
            )
            return json.dumps(results, indent=2)

        elif tool_name == "search_by_visual_similarity":
            text_embedding = embedding_service.embed_text(args["query"])
            results = indexing_service.search_images(
                query_embedding=text_embedding,
                n=args.get("n", 5),
                video_id=args.get("video_id"),
            )
            return json.dumps(results, indent=2)

        elif tool_name == "search_transcripts":
            results = indexing_service.search_transcripts(
                query_text=args["query"],
                n=args.get("n", 5),
                video_id=args.get("video_id"),
            )
            return json.dumps(results, indent=2)

        elif tool_name == "search_by_image":
            image_embedding = embedding_service.embed_image(args["image_path"])
            results = indexing_service.search_images(
                query_embedding=image_embedding,
                n=args.get("n", 5),
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
    client = Groq(api_key=GROQ_API_KEY)

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
                max_tokens=2048,
            )
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return {
                "answer": f"I encountered an error communicating with the AI service: {str(e)}",
                "clips": [],
                "sources": [],
            }

        choice = response.choices[0]

        # If the model wants to call tools
        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            # Add assistant message with tool calls
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

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
                except:
                    pass

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            continue  # loop back for next LLM response

        # Model is done — return final answer
        answer = choice.message.content or "I couldn't generate a response."

        # Build clips from sources
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

        return {
            "answer": answer,
            "clips": clips[:10],  # limit to 10 clips
            "sources": all_sources[:20],  # limit sources
        }

    # If we exhausted iterations
    return {
        "answer": "I performed multiple search operations but couldn't fully resolve the query. Please try rephrasing.",
        "clips": [],
        "sources": all_sources,
    }
