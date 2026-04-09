"""
VideoRAG — Groq-Powered Agent
Agentic loop: receives query → deterministic retrieval → structured answer.
"""
import json
import logging
from groq import Groq
from PIL import Image, ImageStat
import cv2
import numpy as np
import re
from pathlib import Path

from backend.config import GROQ_API_KEY, GROQ_MODEL, FRAMES_DIR, ENABLE_LLM_POLISH
from backend.agent.tools import TOOL_DEFINITIONS
from backend.services import indexing_service, embedding_service, captioning_service

logger = logging.getLogger(__name__)

# ── Singleton Groq client ────────────────────────────────────────────────────
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


# ── System Prompt ────────────────────────────────────────────────────────────
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


# ── Helper utilities ─────────────────────────────────────────────────────────

def _is_speech_query(query: str) -> bool:
    q = query.lower()
    speech_terms = [
        "said", "say", "spoken", "speech", "audio", "voice", "transcript",
        "when did", "quote", "mention", "witness", "told", "heard",
    ]
    return any(term in q for term in speech_terms)


def _has_any(text: str, words: list[str]) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)


def _query_tokens(query: str) -> set[str]:
    raw = re.findall(r"[a-zA-Z]+", (query or "").lower())
    stop = {
        "show", "me", "where", "the", "a", "an", "in", "this", "video", "clip",
        "give", "is", "are", "to", "of", "and", "with", "at", "find", "search",
        "look", "for", "any", "all", "that", "it", "does", "do",
    }
    return {t for t in raw if t not in stop and len(t) > 2}


def _frame_key(hit: dict) -> tuple:
    return (
        hit.get("video_id", ""),
        hit.get("frame_number"),
        hit.get("timestamp"),
        hit.get("frame_path", ""),
    )


def _description_similarity(a: str, b: str) -> float:
    sa = set(re.findall(r"[a-zA-Z]+", (a or "").lower()))
    sb = set(re.findall(r"[a-zA-Z]+", (b or "").lower()))
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / max(1, union)


def _clean_caption_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    prefixes = [
        "a detailed description of this scene showing",
        "detailed description of this scene showing",
        "this scene shows",
        "scene showing",
    ]
    low = t.lower()
    for p in prefixes:
        if low.startswith(p):
            t = t[len(p):].strip(" .:-")
            break
    return t.strip()


def _frame_file_for_hit(hit: dict) -> str:
    vid = hit.get("video_id", "")
    frame_path = hit.get("frame_path", "")
    frame_name = frame_path.split("/")[-1] if frame_path else ""
    if not vid or not frame_name:
        return ""
    fs_path = FRAMES_DIR / vid / frame_name
    return str(fs_path) if fs_path.exists() else ""


def _format_timestamp(seconds) -> str:
    """Convert seconds to MM:SS format."""
    if seconds is None:
        return "??:??"
    s = int(float(seconds))
    return f"{s // 60}:{s % 60:02d}"


# ── Source collection (shared by all strategies) ─────────────────────────────

def _collect_sources(query: str, video_id: str = None, image_path: str = None) -> tuple[list[dict], list[str]]:
    """Deterministic multimodal routing to keep retrieval reliable under all conditions."""
    sources: list[dict] = []
    used_strategies: list[str] = []

    # Strategy B: image-to-video — always search by uploaded image embedding
    if image_path:
        used_strategies.append("image-to-video")
        image_hits = json.loads(_execute_tool("search_by_image", {
            "image_path": image_path,
            "video_id": video_id,
            "n": 12,
        }))
        if isinstance(image_hits, list):
            sources.extend(image_hits)

    # Speech queries
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
        # Strategy A: text-to-video — search captions + visual similarity
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

    # Deduplicate by frame identity
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


# ── Caption retrieval for a hit ──────────────────────────────────────────────

def _caption_for_hit(hit: dict, all_sources: list[dict]) -> str:
    """Pick the best caption text for a hit (same frame/timestamp/video when available)."""
    vid = hit.get("video_id")
    frame_no = hit.get("frame_number")
    ts = hit.get("timestamp")

    # Try to find a matching caption source
    for src in all_sources:
        if src.get("source_index") != "caption":
            continue
        if src.get("video_id") != vid:
            continue
        if frame_no is not None and src.get("frame_number") == frame_no and src.get("content"):
            raw = src.get("content", "").strip()
            return _clean_caption_text(raw)
        if ts is not None and src.get("timestamp") is not None:
            if abs(float(src.get("timestamp")) - float(ts)) <= 1.0 and src.get("content"):
                raw = src.get("content", "").strip()
                return _clean_caption_text(raw)

    # Use inline content
    text = (hit.get("content") or "").strip()
    if text:
        return _clean_caption_text(text)

    # On-demand captioning via BLIP for frames with no indexed caption
    vid = hit.get("video_id", "")
    frame_path = hit.get("frame_path", "")
    frame_name = frame_path.split("/")[-1] if frame_path else ""
    if vid and frame_name:
        fs_path = FRAMES_DIR / vid / frame_name
        if fs_path.exists():
            generated = (captioning_service.caption_frame(str(fs_path)) or "").strip()
            if generated:
                return _clean_caption_text(generated)

    return ""


# ── Visual analysis of frame pixels ──────────────────────────────────────────

def _analyze_frame_visuals(hit: dict) -> dict:
    """Extract visual properties from the actual frame image."""
    frame_file = _frame_file_for_hit(hit)
    if not frame_file:
        return {}

    try:
        with Image.open(frame_file).convert("RGB") as pil_img:
            stat = ImageStat.Stat(pil_img)
            r, g, b = stat.mean[:3]
            brightness = (r + g + b) / 3.0
            contrast = float(np.mean(stat.stddev[:3]))

            rgb = np.array(pil_img)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            edges = cv2.Canny(gray, 80, 160)
            edge_density = float(np.count_nonzero(edges)) / float(edges.size)

            # Lighting
            if brightness >= 165:
                lighting = "bright, well-lit"
            elif brightness >= 110:
                lighting = "moderately lit"
            else:
                lighting = "dimly lit or dark"

            # Color
            spread = max(r, g, b) - min(r, g, b)
            if spread < 18:
                color_tone = "neutral/grayscale tones"
            elif r >= g and r >= b:
                color_tone = "warm tones (reds/oranges)"
            elif b >= r and b >= g:
                color_tone = "cool tones (blues)"
            else:
                color_tone = "natural/green tones"

            # Scene complexity
            if edge_density >= 0.13:
                complexity = "complex scene with many elements"
            elif edge_density >= 0.08:
                complexity = "moderately detailed scene"
            else:
                complexity = "simple/open scene"

            # People detection
            people_count = 0
            try:
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                rects, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
                people_count = len(rects)
            except Exception:
                pass

            return {
                "lighting": lighting,
                "color_tone": color_tone,
                "complexity": complexity,
                "people_count": people_count,
                "sharpness": sharpness,
            }
    except Exception:
        return {}


def _motion_label_for_hit(hit: dict) -> str:
    """Use adjacent frame change as a human-readable action cue."""
    vid = hit.get("video_id", "")
    frame_no = hit.get("frame_number")
    if not vid or frame_no is None:
        return ""

    base = FRAMES_DIR / vid
    cur = base / f"frame_{int(frame_no):06d}.jpg"
    nxt = base / f"frame_{int(frame_no)+1:06d}.jpg"
    prv = base / f"frame_{max(0, int(frame_no)-1):06d}.jpg"

    compare = nxt if nxt.exists() else (prv if prv.exists() else None)
    if not cur.exists() or compare is None:
        return ""

    try:
        cur_g = cv2.cvtColor(np.array(Image.open(cur).convert("RGB")), cv2.COLOR_RGB2GRAY)
        cmp_g = cv2.cvtColor(np.array(Image.open(compare).convert("RGB")), cv2.COLOR_RGB2GRAY)
        delta = float(np.mean(cv2.absdiff(cur_g, cmp_g)))
        if delta >= 26:
            return "High motion/scene change detected between adjacent frames."
        if delta >= 15:
            return "Noticeable movement detected around this timestamp."
        return "Relatively steady scene around this timestamp."
    except Exception:
        return ""


# ── Strategy A: Text-to-Video ────────────────────────────────────────────────

def _rerank_strategy_a_sources(query: str, sources: list[dict]) -> list[dict]:
    """Rerank candidates for text-to-video so the top 2 are the most relevant."""
    if not sources:
        return []

    q_tokens = _query_tokens(query)
    pool = sources[:12]
    scored = []

    for hit in pool:
        desc = _caption_for_hit(hit, pool).lower()
        desc_tokens = set(re.findall(r"[a-zA-Z]+", desc))
        token_overlap = len(q_tokens & desc_tokens) / max(1, len(q_tokens))

        src_score = float(hit.get("score", 0.0))
        # Blend retrieval score with token overlap for relevance
        rel = 0.55 * src_score + 0.45 * token_overlap

        # Bonus for caption-index sources (they have text descriptions)
        if hit.get("source_index") == "caption":
            rel += 0.05

        scored.append((rel, hit, desc))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return sources

    # Diversity-aware top-2 selection to avoid near-duplicate descriptions
    chosen = [scored[0]]
    if len(scored) > 1:
        best_second = None
        best_second_val = -10**9
        _, _, first_desc = chosen[0]
        first_ts = chosen[0][1].get("timestamp")

        for cand in scored[1:]:
            c_rel, c_hit, c_desc = cand
            sim_penalty = 0.25 * _description_similarity(first_desc, c_desc)
            time_penalty = 0.0
            c_ts = c_hit.get("timestamp")
            if first_ts is not None and c_ts is not None and abs(float(first_ts) - float(c_ts)) <= 1.0:
                time_penalty = 0.10

            val = c_rel - sim_penalty - time_penalty
            if val > best_second_val:
                best_second_val = val
                best_second = cand

        if best_second is not None:
            chosen.append(best_second)

    selected_hits = [c[1] for c in chosen]

    # Keep remaining items after selected for data availability
    selected_keys = {_frame_key(h) for h in selected_hits}
    rest = [h for h in sources if _frame_key(h) not in selected_keys]
    return selected_hits + rest


def _build_detailed_description(hit: dict, all_sources: list[dict], query: str, match_index: int) -> str:
    """Build a rich, detailed description of a matched frame — used by both Strategy A and B."""
    caption = _caption_for_hit(hit, all_sources)
    visuals = _analyze_frame_visuals(hit)
    motion = _motion_label_for_hit(hit)
    timestamp = _format_timestamp(hit.get("timestamp"))
    video_id = hit.get("video_id", "unknown")
    score = hit.get("score", 0)
    source_idx = hit.get("source_index", "")

    lines = [f"**Match {match_index}** — Video: `{video_id}` at [{timestamp}] (confidence: {score:.2f})"]

    # Scene description from caption
    if caption:
        lines.append(f"  📝 **Scene**: {caption}")
    else:
        lines.append(f"  📝 **Scene**: Visual content matched to your query")

    # Visual analysis from actual frame pixels
    if visuals:
        visual_parts = []
        if visuals.get("lighting"):
            visual_parts.append(visuals["lighting"])
        if visuals.get("color_tone"):
            visual_parts.append(visuals["color_tone"])
        if visuals.get("complexity"):
            visual_parts.append(visuals["complexity"])
        if visual_parts:
            lines.append(f"  🎨 **Visual**: {', '.join(visual_parts)}")

        people = visuals.get("people_count", 0)
        if people >= 4:
            lines.append(f"  👥 **People**: Multiple people/group detected ({people} regions)")
        elif people >= 1:
            lines.append(f"  👤 **People**: Person(s) detected in frame ({people} region{'s' if people > 1 else ''})")

    # Motion context
    if motion:
        lines.append(f"  🎬 **Motion**: {motion}")

    # Source index info
    source_label = {"image": "Visual Index (CLIP)", "caption": "Caption Index", "speech": "Speech Index"}.get(source_idx, source_idx)
    if source_label:
        lines.append(f"  📊 **Source**: {source_label}")

    return "\n".join(lines)


def _render_strategy_a_answer(query: str, selected_sources: list[dict], all_sources: list[dict], video_id: str = None) -> str:
    """Render Strategy A answer: top 2 text-to-video matches with detailed descriptions."""
    if not selected_sources:
        scope = f" in video {video_id}" if video_id else ""
        return f"I could not find matches for '{query}'{scope}. Try rephrasing your query or using different keywords."

    lines = [f"🔍 **Top {len(selected_sources)} matches for: \"{query}\"**\n"]

    for idx, hit in enumerate(selected_sources, start=1):
        description = _build_detailed_description(hit, all_sources, query, idx)
        lines.append(description)
        lines.append("")  # blank line between matches

    return "\n".join(lines)


# ── Strategy B: Image-to-Video ───────────────────────────────────────────────

def _render_strategy_b_answer(query: str, sources: list[dict], video_id: str = None) -> str:
    """Render Strategy B answer: image-to-video search results with detailed descriptions."""
    # Separate image-index hits (from the uploaded image) from other hits
    image_hits = [s for s in sources if s.get("source_index") == "image"]
    other_hits = [s for s in sources if s.get("source_index") != "image"]

    if not image_hits and not other_hits:
        scope = f" in video {video_id}" if video_id else ""
        return (
            f"🔍 No visual matches found{scope}. "
            "The uploaded image did not closely match any indexed video frames. "
            "Try uploading a clearer image or a different reference photo."
        )

    # Use top image hits as primary results
    top_hits = image_hits[:5] if image_hits else other_hits[:5]
    best_score = max((float(h.get("score", 0)) for h in top_hits), default=0.0)

    if best_score < 0.15:
        return (
            "🔍 The uploaded image has very low similarity to any indexed video frames. "
            "This person/object may not appear in the available footage."
        )

    lines = [f"🖼️ **Image-to-Video Search Results**\n"]

    if best_score >= 0.30:
        lines.append(f"✅ **Strong visual matches found** (best similarity: {best_score:.2f})\n")
    elif best_score >= 0.20:
        lines.append(f"⚠️ **Moderate visual matches found** (best similarity: {best_score:.2f}) — review frames carefully\n")
    else:
        lines.append(f"ℹ️ **Weak visual matches found** (best similarity: {best_score:.2f}) — matches may not be accurate\n")

    for idx, hit in enumerate(top_hits, start=1):
        description = _build_detailed_description(hit, sources, query, idx)
        lines.append(description)
        lines.append("")

    # Group matched timestamps into summary
    matched_videos = {}
    for hit in top_hits:
        vid = hit.get("video_id", "")
        ts = hit.get("timestamp")
        if vid and ts is not None:
            matched_videos.setdefault(vid, []).append(float(ts))

    if matched_videos:
        lines.append("📋 **Summary of appearances:**")
        for vid, timestamps in matched_videos.items():
            timestamps.sort()
            ts_strs = [_format_timestamp(t) for t in timestamps]
            lines.append(f"  - Video `{vid}`: at timestamps [{', '.join(ts_strs)}]")

    return "\n".join(lines)


# ── Main agent entry point ───────────────────────────────────────────────────

def run_agent(query: str, video_id: str = None, image_path: str = None, conversation_history: list = None) -> dict:
    """
    Run the agentic retrieval loop:
    1. Collect sources from relevant indexes
    2. Rerank and select top results
    3. Build detailed answer with frame descriptions
    """
    sources, strategies = _collect_sources(query=query, video_id=video_id, image_path=image_path)

    # Strategy B: Image-to-Video search
    if image_path:
        top_sources = sources[:5]
        return {
            "answer": _render_strategy_b_answer(query=query, sources=sources, video_id=video_id),
            "clips": _build_clips(top_sources),
            "sources": top_sources,
        }

    # Strategy A: Text-to-Video search (top 2 detailed matches)
    if not _is_speech_query(query):
        sources = _rerank_strategy_a_sources(query=query, sources=sources)
        selected = sources[:2]
        return {
            "answer": _render_strategy_a_answer(query=query, selected_sources=selected, all_sources=sources, video_id=video_id),
            "clips": _build_clips(selected)[:2],
            "sources": selected,
        }

    # Speech/transcript queries — render a simple summary
    answer = _render_speech_answer(query=query, sources=sources, video_id=video_id)

    # Optional LLM polish for speech queries
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


def _render_speech_answer(query: str, sources: list[dict], video_id: str = None) -> str:
    """Render answer for speech/transcript queries."""
    if not sources:
        scope = f" in video {video_id}" if video_id else ""
        return f"I could not find matching speech/transcript evidence for this query{scope}."

    lines = [f"🎤 **Transcript search results for: \"{query}\"**\n"]

    speech_hits = [s for s in sources if s.get("source_index") == "speech"][:5]
    other_hits = [s for s in sources if s.get("source_index") != "speech"][:3]

    for idx, hit in enumerate(speech_hits, start=1):
        content = hit.get("content", "").strip()
        st = hit.get("start_time")
        et = hit.get("end_time")
        vid = hit.get("video_id", "")
        score = hit.get("score", 0)

        time_range = ""
        if st is not None and et is not None:
            time_range = f"[{_format_timestamp(st)} – {_format_timestamp(et)}]"

        lines.append(f"**{idx}.** Video `{vid}` {time_range} (score: {score:.2f})")
        if content:
            lines.append(f"   > \"{content}\"")
        lines.append("")

    if other_hits:
        lines.append("**Related visual context:**")
        for hit in other_hits:
            ts = _format_timestamp(hit.get("timestamp"))
            vid = hit.get("video_id", "")
            caption = _caption_for_hit(hit, sources)
            if caption:
                lines.append(f"  - `{vid}` at [{ts}]: {caption}")

    return "\n".join(lines)


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
                    "end_time": float(src.get("end_time", float(ts) + 2)),
                    "frame_paths": [frame_path],
                    "description": src.get("content", ""),
                })
    return clips[:20]
