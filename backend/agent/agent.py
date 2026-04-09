"""
VideoRAG — Groq-Powered Agent
Agentic loop: receives query → LLM decides tools → execute → LLM answers.
"""
import json
import logging
import concurrent.futures
from groq import Groq
from PIL import Image, ImageStat
import cv2
import numpy as np
import re
from pathlib import Path

from backend.config import GROQ_API_KEY, GROQ_MODEL, FRAMES_DIR, ENABLE_LLM_POLISH, ENABLE_QUERY_TIME_CAPTIONING
from backend.agent.tools import TOOL_DEFINITIONS
from backend.services import indexing_service, embedding_service, captioning_service

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


# ── Query expansion ──────────────────────────────────────────────────────────

_EVENT_EXPANSIONS: dict[str, list[str]] = {
    # Keys are trigger words; values are extra CLIP-friendly phrasings to search
    "hit":        ["vehicle collision road", "car accident impact", "crash scene street"],
    "hits":       ["vehicle collision road", "car accident impact", "crash scene street"],
    "crash":      ["car crash road", "vehicle collision accident", "smashed car street"],
    "collision":  ["cars colliding road", "vehicle accident street", "crash impact scene"],
    "collide":    ["cars colliding road", "vehicle accident street"],
    "knock":      ["car knocking person", "vehicle hitting pedestrian"],
    "running":    ["person running fast street", "people sprinting outdoors"],
    "chasing":    ["person being chased street", "pursuit running outdoors"],
    "fight":      ["people fighting street", "physical altercation outdoors"],
    "shoot":      ["gunfire street", "shooting scene outdoor"],
    "fire":       ["fire burning outdoors", "flames street scene"],
    "explode":    ["explosion street", "fire blast outdoors"],
    "fall":       ["person falling down", "someone tripping street"],
    "theft":      ["person stealing bag", "pickpocket street scene"],
    "robbery":    ["armed robbery store", "person holding weapon"],
}


def _expand_query(query: str) -> list[str]:
    """Return the original query + semantically expanded CLIP-friendly variants.
    Expansion only triggers for known action/event keywords to avoid noise.
    """
    q = (query or "").lower()
    extra: list[str] = []
    seen: set[str] = set()
    for trigger, expansions in _EVENT_EXPANSIONS.items():
        if trigger in q:
            for exp in expansions:
                if exp not in seen:
                    seen.add(exp)
                    extra.append(exp)
    # Return original first, then up to 2 best expansions
    return [query] + extra[:2]


# ── Parallel search helpers ───────────────────────────────────────────────────

def _safe_search_captions(query: str, video_id, n: int) -> list[dict]:
    try:
        result = json.loads(_execute_tool("search_by_caption", {"query": query, "video_id": video_id, "n": n}))
        return result if isinstance(result, list) else []
    except Exception as e:
        logger.warning(f"Caption search failed for '{query}': {e}")
        return []


def _safe_search_visual(query: str, video_id, n: int) -> list[dict]:
    try:
        result = json.loads(_execute_tool("search_by_visual_similarity", {"query": query, "video_id": video_id, "n": n}))
        return result if isinstance(result, list) else []
    except Exception as e:
        logger.warning(f"Visual search failed for '{query}': {e}")
        return []


def _safe_search_transcripts(query: str, video_id, n: int) -> list[dict]:
    try:
        result = json.loads(_execute_tool("search_transcripts", {"query": query, "video_id": video_id, "n": n}))
        return result if isinstance(result, list) else []
    except Exception as e:
        logger.warning(f"Transcript search failed for '{query}': {e}")
        return []


def _merge_and_dedup(sources: list[dict]) -> list[dict]:
    """Deduplicate by frame identity, keeping the highest-scoring version of each frame."""
    best: dict[tuple, dict] = {}
    for src in sources:
        key = (
            src.get("video_id"),
            src.get("frame_number"),
            src.get("timestamp"),
            src.get("start_time"),
            src.get("end_time"),
            src.get("source_index"),
        )
        if key not in best or float(src.get("score", 0)) > float(best[key].get("score", 0)):
            best[key] = src
    result = list(best.values())
    result.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
    return result


def _collect_sources(query: str, video_id: str = None, image_path: str = None) -> tuple[list[dict], list[str]]:
    """Parallel multimodal retrieval with query expansion for event queries."""
    used_strategies: list[str] = []

    # ── Strategy B: image-to-video (single CLIP image embedding search) ───────
    if image_path:
        used_strategies.append("image-to-video")
        image_hits = json.loads(_execute_tool("search_by_image", {
            "image_path": image_path,
            "video_id": video_id,
            "n": 12,
        }))
        hits = image_hits if isinstance(image_hits, list) else []
        return _merge_and_dedup(hits), used_strategies

    # ── Speech queries: transcript search + supporting caption search ─────────
    if _is_speech_query(query):
        used_strategies.append("audio/transcript")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_speech  = pool.submit(_safe_search_transcripts, query, video_id, 12)
            f_caption = pool.submit(_safe_search_captions,    query, video_id, 8)
        sources = f_speech.result() + f_caption.result()
        return _merge_and_dedup(sources), used_strategies

    # ── Strategy A: text-to-video — parallel caption + visual, with expansion ─
    used_strategies.append("text-to-video")
    expanded_queries = _expand_query(query)  # [original, expansion1, expansion2]

    all_hits: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures: list[concurrent.futures.Future] = []

        # Original query: both caption and visual
        futures.append(pool.submit(_safe_search_captions, query, video_id, 12))
        futures.append(pool.submit(_safe_search_visual,   query, video_id, 12))

        # Expansion queries: visual only (captions are BLIP-generated, less useful for expansions)
        for exp_query in expanded_queries[1:]:
            futures.append(pool.submit(_safe_search_visual, exp_query, video_id, 8))

        for f in concurrent.futures.as_completed(futures):
            try:
                all_hits.extend(f.result())
            except Exception as e:
                logger.warning(f"Search future failed: {e}")

    return _merge_and_dedup(all_hits), used_strategies



def _render_answer(query: str, sources: list[dict], strategies: list[str], video_id: str = None) -> str:
    if not sources:
        scope = f" in video {video_id}" if video_id else ""
        return f"I could not find strong evidence matches for this query{scope}. Try a more specific object, person, or spoken phrase."

    lines = [f"Found evidence using: {', '.join(sorted(set(strategies)))}"]

    return "\n".join(lines)


def _format_time_label(hit: dict) -> str:
    ts = hit.get("timestamp")
    st = hit.get("start_time")
    et = hit.get("end_time")
    if ts is not None:
        return f"{float(ts):.0f}s"
    if st is not None and et is not None:
        return f"{float(st):.0f}s to {float(et):.0f}s"
    return "unknown time"


def _caption_for_hit(hit: dict, all_sources: list[dict]) -> str:
    """Pick the best caption text for a hit (same frame/timestamp/video when available)."""
    vid = hit.get("video_id")
    frame_no = hit.get("frame_number")
    ts = hit.get("timestamp")

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

    text = (hit.get("content") or "").strip()
    if text:
        return _clean_caption_text(text)

    # Query-time BLIP captioning is expensive; keep it optional for low-latency chat.
    if ENABLE_QUERY_TIME_CAPTIONING:
        vid = hit.get("video_id", "")
        frame_path = hit.get("frame_path", "")
        frame_name = frame_path.split("/")[-1] if frame_path else ""
        if vid and frame_name:
            fs_path = FRAMES_DIR / vid / frame_name
            if fs_path.exists():
                generated = (captioning_service.caption_frame(str(fs_path)) or "").strip()
                if generated:
                    return _clean_caption_text(generated)

    return "No caption text available; visual match based on image similarity."


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


def _image_visual_facts(hit: dict, desc: str) -> list[str]:
    """Produce image-grounded descriptive lines from frame pixels and caption text."""
    facts = []
    frame_file = _frame_file_for_hit(hit)

    if frame_file:
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

                if brightness >= 165:
                    light = "bright daylight-style lighting"
                elif brightness >= 110:
                    light = "medium balanced lighting"
                else:
                    light = "darker cinematic lighting"

                if contrast >= 70:
                    contrast_label = "strong contrast"
                elif contrast >= 45:
                    contrast_label = "moderate contrast"
                else:
                    contrast_label = "soft low contrast"

                spread = max(r, g, b) - min(r, g, b)
                if spread < 18:
                    tone = "neutral color balance"
                elif r >= g and r >= b:
                    tone = "warm orange-red dominant palette"
                elif b >= r and b >= g:
                    tone = "cool blue dominant palette"
                else:
                    tone = "green-earth dominant palette"

                if sharpness >= 140:
                    detail = "high fine-detail clarity"
                elif sharpness >= 70:
                    detail = "medium detail clarity"
                else:
                    detail = "slightly soft or motion-blurred detail"

                if edge_density >= 0.13:
                    structure = "dense scene structure with many contours"
                elif edge_density >= 0.08:
                    structure = "moderate scene structure"
                else:
                    structure = "open scene with fewer structural edges"

                people_regions = 0
                try:
                    hog = cv2.HOGDescriptor()
                    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                    rects, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
                    people_regions = len(rects)
                except Exception:
                    people_regions = 0

                facts.append(f"Lighting is {light} with {contrast_label} and a {tone}.")
                facts.append(f"Image quality shows {detail} and {structure}.")

                if people_regions >= 4:
                    facts.append("Multiple human-sized regions are visible, indicating a crowd or group action frame.")
                elif people_regions >= 1:
                    facts.append("At least one human-sized region is detected, suggesting person-focused framing.")
                else:
                    facts.append("No strong HOG person boxes were detected, but scene composition still suggests human activity.")
        except Exception:
            pass

    desc_lower = desc.lower()
    if any(word in desc_lower for word in ["running", "chasing", "moving", "action", "fight"]):
        facts.append("Caption cues indicate active movement or action in this frame.")
    elif any(word in desc_lower for word in ["standing", "sitting", "closeup", "portrait"]):
        facts.append("Caption cues indicate a more static composition focused on subject posture.")
    else:
        facts.append("Caption cues indicate a mixed scene with both subjects and environmental context.")

    return facts


def _has_any(text: str, words: list[str]) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)


def _query_tokens(query: str) -> set[str]:
    raw = re.findall(r"[a-zA-Z]+", (query or "").lower())
    stop = {
        "show", "me", "where", "the", "a", "an", "in", "this", "video", "clip", "give", "is", "are", "to", "of", "and", "with", "at",
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


def _query_constraint_terms(query: str) -> dict[str, set[str]]:
    q = (query or "").lower()
    terms = set(re.findall(r"[a-zA-Z]+", q))
    colors = {"blue", "red", "green", "yellow", "black", "white", "orange", "gray", "grey", "brown"}
    objects = {
        "car", "vehicle", "truck", "bus", "bike", "bicycle", "motorcycle", "person", "people", "man", "woman",
        "bag", "backpack", "helmet", "phone", "weapon", "gun", "knife",
    }
    return {
        "colors": terms & colors,
        "objects": terms & objects,
    }


def _hit_query_overlap(query: str, hit: dict, all_sources: list[dict]) -> tuple[float, str]:
    desc = _caption_for_hit(hit, all_sources).lower()
    q_tokens = _query_tokens(query)
    d_tokens = set(re.findall(r"[a-zA-Z]+", desc))
    overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
    return overlap, desc


def _is_hit_relevant_to_query(query: str, hit: dict, all_sources: list[dict]) -> bool:
    """Gate: keep a hit if it has at least one anchor term from the query.
    We do NOT enforce lexical overlap because BLIP captions are too generic
    (e.g. 'scene with car and street') and would filter out valid results.
    CLIP cosine score is the primary signal; anchors are a light sanity check.
    """
    desc = _caption_for_hit(hit, all_sources).lower()
    constraints = _query_constraint_terms(query)
    color_terms  = constraints["colors"]
    object_terms = constraints["objects"]

    # If no specific anchors were requested, accept all CLIP hits
    if not color_terms and not object_terms:
        return True

    # Accept if at least one requested object/color appears anywhere in the caption
    has_color_anchor  = not color_terms  or any(c in desc for c in color_terms)
    has_object_anchor = not object_terms or any(o in desc for o in object_terms)

    # Relaxed: require only one of color/object anchor (not both)
    return has_color_anchor or has_object_anchor


def _motion_line_for_hit(hit: dict) -> str:
    """Use adjacent frame change as a human-readable action cue."""
    vid = hit.get("video_id", "")
    frame_no = hit.get("frame_number")
    if not vid or frame_no is None:
        return "Motion appears moderate around this moment in the clip."

    base = FRAMES_DIR / vid
    cur = base / f"frame_{int(frame_no):06d}.jpg"
    nxt = base / f"frame_{int(frame_no)+1:06d}.jpg"
    prv = base / f"frame_{max(0, int(frame_no)-1):06d}.jpg"

    compare = nxt if nxt.exists() else (prv if prv.exists() else None)
    if not cur.exists() or compare is None:
        return "Motion appears moderate around this moment in the clip."

    try:
        cur_g = cv2.cvtColor(np.array(Image.open(cur).convert("RGB")), cv2.COLOR_RGB2GRAY)
        cmp_g = cv2.cvtColor(np.array(Image.open(compare).convert("RGB")), cv2.COLOR_RGB2GRAY)
        delta = float(np.mean(cv2.absdiff(cur_g, cmp_g)))
        if delta >= 26:
            return "There is a strong scene change between adjacent frames, suggesting an impact/action peak."
        if delta >= 15:
            return "Adjacent frames show noticeable movement, indicating active motion in this moment."
        return "Adjacent frames change only slightly, suggesting a steadier moment around this frame."
    except Exception:
        return "Motion appears moderate around this moment in the clip."


def _rerank_strategy_a_sources(query: str, sources: list[dict]) -> list[dict]:
    """Rerank candidates for text-to-video so event queries get the most relevant frames.
    Scoring model:
      - 70% CLIP cosine score (semantic understanding, query expansion benefits this)
      - 30% token overlap with caption (light lexical anchor)
      - Bonuses for matching object/action keywords in captions
    """
    if not sources:
        return []

    q = (query or "").lower()
    q_tokens = _query_tokens(query)
    collision_intent = _has_any(q, ["hit", "hits", "crash", "collision", "collide", "knock"])
    running_intent   = _has_any(q, ["running", "run", "chasing", "sprint"])

    pool = sources[:24]  # expanded pool for better diversity
    scored = []

    for hit in pool:
        desc = _caption_for_hit(hit, pool).lower()
        desc_tokens = set(re.findall(r"[a-zA-Z]+", desc))
        token_overlap = len(q_tokens & desc_tokens) / max(1, len(q_tokens))

        src_score = float(hit.get("score", 0.0))
        # Weight CLIP score heavier than token overlap — CLIP understands semantics
        rel = 0.70 * src_score + 0.30 * token_overlap

        has_car    = _has_any(desc, ["car", "vehicle", "truck", "sedan"])
        has_bike   = _has_any(desc, ["bike", "bicycle", "motorcycle", "motorbike"])
        has_person = _has_any(desc, ["person", "people", "man", "woman", "pedestrian"])
        has_run    = _has_any(desc, ["running", "run", "chasing", "sprint"])
        has_road   = _has_any(desc, ["road", "street", "highway", "sidewalk"])

        if collision_intent:
            if has_car and has_bike:    rel += 0.40  # exact match: both objects
            elif has_car and has_person: rel += 0.28  # car + person
            elif has_car or has_bike:   rel += 0.12  # partial
            if has_road:                rel += 0.08  # road context bonus

        if running_intent and (has_run or has_person):
            rel += 0.18

        # Caption sources have real text — slight bonus
        if hit.get("source_index") == "caption":
            rel += 0.04

        scored.append((rel, hit, desc))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return sources

    # Relevance filter: accept all if filter is too strict (fallback to full list)
    filtered = [x for x in scored if _is_hit_relevant_to_query(query, x[1], pool)]
    ranked_pool = filtered if len(filtered) >= 2 else scored

    # Diversity-aware top-2 selection
    chosen = [ranked_pool[0]]
    if len(ranked_pool) > 1:
        best_second     = None
        best_second_val = -10**9
        _, _, first_desc = chosen[0]
        first_ts         = chosen[0][1].get("timestamp")
        first_key        = _frame_key(chosen[0][1])

        for cand in ranked_pool[1:]:
            c_rel, c_hit, c_desc = cand
            if _frame_key(c_hit) == first_key:
                continue
            sim_penalty  = 0.20 * _description_similarity(first_desc, c_desc)
            time_penalty = 0.0
            c_ts = c_hit.get("timestamp")
            if first_ts is not None and c_ts is not None and abs(float(first_ts) - float(c_ts)) <= 1.0:
                time_penalty = 0.08
            val = c_rel - sim_penalty - time_penalty
            if val > best_second_val:
                best_second_val = val
                best_second = cand

        if best_second is not None:
            chosen.append(best_second)

    selected_hits = [c[1] for c in chosen]
    selected_keys = {_frame_key(h) for h in selected_hits}
    rest = [h for h in sources if _frame_key(h) not in selected_keys]
    return selected_hits + rest



def _human_scene_lines(hit: dict, all_sources: list[dict], query: str) -> list[str]:
    """Generate human-style, image-specific description lines for a matched frame."""
    desc = _caption_for_hit(hit, all_sources).strip().rstrip(".")
    desc_lower = desc.lower()
    query_lower = (query or "").lower()

    if not desc:
        desc = "A scene with visible activity and moving subjects"

    # Ground truth object cues should come from image description, not only user query.
    has_car_desc = _has_any(desc_lower, ["car", "vehicle", "sedan", "truck"])
    has_bike_desc = _has_any(desc_lower, ["bike", "bicycle", "motorcycle", "motorbike"])
    has_person_desc = _has_any(desc_lower, ["person", "people", "man", "woman", "pedestrian"])

    collision_query = _has_any(query_lower, ["hit", "hits", "crash", "collide", "collision", "knock"]) 
    running_query = _has_any(query_lower, ["running", "run", "chasing", "sprint"])

    line_1 = f"Image summary: {desc}."

    if collision_query and has_car_desc and has_bike_desc:
        line_2 = "This frame appears to capture a possible car-and-bike impact moment, or the seconds immediately around it."
        line_3 = "The road interaction between the car and bike is the key event visible in this image."
    elif collision_query and has_car_desc and has_person_desc:
        line_2 = "This frame appears to show a possible car-to-person impact moment, or the immediate lead-up/aftermath."
        line_3 = "The primary event is the close interaction between the vehicle and the pedestrian."
    elif collision_query:
        line_2 = "This frame does not clearly show a confirmed collision between a car and a bike/person."
        line_3 = "It appears to depict a different moment, so use nearby frames to locate the exact impact point."
    elif running_query and has_person_desc:
        line_2 = "People in this frame appear to be in active motion, consistent with a running/action moment."
        line_3 = "The scene focuses on movement and urgency rather than a static pose."
    elif has_car_desc and has_bike_desc:
        line_2 = "A car and a bike are both visible in this frame, sharing the same road space."
        line_3 = "This image is relevant because it captures their proximity and interaction context."
    elif has_person_desc and has_car_desc:
        line_2 = "A person and a vehicle are both visible in this frame."
        line_3 = "This image is relevant because it shows their spatial interaction in the scene."
    else:
        line_2 = "This image shows the main subjects clearly and captures the event context of the scene."
        line_3 = "It is one of the best visual matches for the requested incident."

    # Add a final concise line tied to clip evidence, without technical jargon.
    line_4 = _motion_line_for_hit(hit)

    return [line_1, line_2, line_3, line_4]


def _render_strategy_a_answer(query: str, selected_sources: list[dict], all_sources: list[dict], video_id: str = None) -> str:
    if not selected_sources:
        scope = f" in video {video_id}" if video_id else ""
        return f"I could not find matches for '{query}'{scope}."

    lines = [f"Detailed scene descriptions for '{query}':"]
    for idx, hit in enumerate(selected_sources, start=1):
        human_lines = _human_scene_lines(hit, all_sources, query)

        lines.append(
            f"Match {idx}:\n"
            f"- {human_lines[0]}\n"
            f"- {human_lines[1]}\n"
            f"- {human_lines[2]}\n"
            f"- {human_lines[3]}"
        )

    return "\n".join(lines)


def _render_strategy_b_answer(query: str, selected_sources: list[dict], all_sources: list[dict], video_id: str = None) -> str:
    """Render image-to-video search answer focused on visual evidence."""
    if not selected_sources:
        scope = f" in video {video_id}" if video_id else ""
        return (
            f"No reliable visual matches were found for the uploaded reference image{scope}. "
            "Try a clearer reference photo (front-facing, less blur) or narrow to a specific video."
        )

    top_score = float(selected_sources[0].get("score", 0.0))
    if top_score >= 0.35:
        verdict = "Strong visual matches found for the uploaded reference image."
    elif top_score >= 0.25:
        verdict = "Possible visual matches found; manual review recommended."
    else:
        verdict = "Only weak visual similarity matches found; treat these as low confidence."

    lines = [
        verdict,
        f"Query context: {query}",
    ]

    for idx, hit in enumerate(selected_sources, start=1):
        score = float(hit.get("score", 0.0))
        time_label = _format_time_label(hit)
        human_lines = _human_scene_lines(hit, all_sources, query)
        lines.append(
            f"Match {idx} (similarity: {score:.3f}, time: {time_label}, video: {hit.get('video_id', 'unknown')}):\n"
            f"- {human_lines[0]}\n"
            f"- {human_lines[1]}\n"
            f"- {human_lines[2]}"
        )

    return "\n".join(lines)


def _select_strategy_b_sources(sources: list[dict]) -> list[dict]:
    """Keep image-index hits first for image-to-video flow and avoid duplicate frame picks."""
    if not sources:
        return []

    image_hits = [s for s in sources if s.get("source_index") == "image"]
    other_hits = [s for s in sources if s.get("source_index") != "image"]

    image_hits.sort(key=lambda s: float(s.get("score", 0.0)), reverse=True)
    other_hits.sort(key=lambda s: float(s.get("score", 0.0)), reverse=True)

    ranked = image_hits + other_hits
    selected = []
    seen = set()
    for hit in ranked:
        key = _frame_key(hit)
        if key in seen:
            continue
        seen.add(key)
        selected.append(hit)
        if len(selected) >= 5:
            break

    return selected


def run_agent(query: str, video_id: str = None, image_path: str = None, conversation_history: list = None) -> dict:
    """
    Run the agentic loop:
    1. Send query + tools to Groq
    2. If tool_calls → execute → send results back
    3. Return final answer with sources
    """
    sources, strategies = _collect_sources(query=query, video_id=video_id, image_path=image_path)

    # Strategy B: image-to-video search should prioritize visual index evidence.
    if image_path:
        selected = _select_strategy_b_sources(sources)
        return {
            "answer": _render_strategy_b_answer(query=query, selected_sources=selected[:3], all_sources=sources, video_id=video_id),
            "clips": _build_clips(selected)[:5],
            "sources": selected,
        }

    # Strategy A: text-to-video search should return only top 2 detailed matches.
    if not image_path and not _is_speech_query(query):
        sources = _rerank_strategy_a_sources(query=query, sources=sources)
        selected = sources[:2]
        return {
            "answer": _render_strategy_a_answer(query=query, selected_sources=selected, all_sources=sources, video_id=video_id),
            "clips": _build_clips(selected)[:2],
            "sources": selected,
        }

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
    """Build deduplicated clip list with neighboring frames for playback context."""
    clips = []
    seen_clips = set()

    def _frame_exists(video_id: str, frame_path: str) -> bool:
        if not frame_path:
            return False
        frame_name = frame_path.split("/")[-1]
        if not frame_name:
            return False
        return (FRAMES_DIR / video_id / frame_name).exists()

    def _collect_neighbor_frames(video_id: str, frame_number: int | None, fallback_path: str) -> list[str]:
        if frame_number is None:
            return [fallback_path] if fallback_path else []

        result = []
        for offset in (-1, 0, 1):
            n = max(0, int(frame_number) + offset)
            fname = f"frame_{n:06d}.jpg"
            fpath = FRAMES_DIR / video_id / fname
            if fpath.exists():
                result.append(f"/frames/{video_id}/{fname}")

        if not result and fallback_path:
            result.append(fallback_path)
        return result

    for src in all_sources:
        vid = src.get("video_id", "")
        ts = src.get("timestamp") or src.get("start_time")
        frame_path = src.get("frame_path", "")
        frame_number = src.get("frame_number")
        if vid and ts is not None and _frame_exists(vid, frame_path):
            clip_key = f"{vid}_{ts}"
            if clip_key not in seen_clips:
                seen_clips.add(clip_key)
                frame_paths = _collect_neighbor_frames(vid, frame_number, frame_path)
                start_time = float(ts) - (1.0 if len(frame_paths) > 1 else 0.0)
                end_time = float(ts) + (1.0 if len(frame_paths) > 1 else 2.0)
                clips.append({
                    "video_id": vid,
                    "start_time": max(0.0, start_time),
                    "end_time": float(src.get("end_time", end_time)),
                    "frame_paths": frame_paths,
                    "description": src.get("content", ""),
                })
    return clips[:20]  # increased from 10 to 20
