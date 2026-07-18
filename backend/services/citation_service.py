"""
VideoRAG — Citation Verification Service (Phase 8)
Verifies that cited timestamps actually support the claims made in generated answers.
Computes SHA-256 hashes of cited clips for tamper-evidence.
"""
import hashlib
import json
import re
import uuid
import logging
from pathlib import Path

from backend.config import (
    GROQ_API_KEY, GROQ_MODEL, FRAMES_DIR, AUDIO_DIR
)
from backend.services import citation_db, indexing_service

logger = logging.getLogger(__name__)

# ── Singleton Groq client ────────────────────────────────────────────────────

_groq_client = None


def _get_groq_client():
    """Return a singleton Groq client instance."""
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set.")
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ── Timestamp Extraction ─────────────────────────────────────────────────────

def extract_citations_from_answer(answer: str) -> list[dict]:
    """
    Parse timestamps from a generated answer text.

    Looks for patterns like [MM:SS], [HH:MM:SS], or [MM:SS – MM:SS].
    Returns list of {"timestamp_text": str, "timestamp_sec": float, "context": str}.
    """
    citations = []

    # Match [MM:SS] or [HH:MM:SS] patterns
    pattern = r'\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]'
    for match in re.finditer(pattern, answer):
        full_match = match.group(0)
        parts = [int(x) for x in match.groups() if x is not None]

        if len(parts) == 3:  # HH:MM:SS
            timestamp_sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
        else:  # MM:SS
            timestamp_sec = parts[0] * 60 + parts[1]

        # Extract surrounding context (the claim near this timestamp)
        start = max(0, match.start() - 100)
        end = min(len(answer), match.end() + 100)
        context = answer[start:end].strip()

        citations.append({
            "timestamp_text": full_match,
            "timestamp_sec": float(timestamp_sec),
            "context": context,
        })

    return citations


# ── Clip Hashing ─────────────────────────────────────────────────────────────

def compute_clip_hash(video_id: str, timestamp: float,
                      window_sec: float = 2.0) -> str:
    """
    Compute a SHA-256 hash of frame files around the cited timestamp
    for tamper-evidence.

    Hashes the raw bytes of all frames within [timestamp - window, timestamp + window].
    """
    frames_dir = FRAMES_DIR / video_id
    if not frames_dir.exists():
        return ""

    hasher = hashlib.sha256()
    hashed_count = 0

    # Get all frames and their timestamps from the index
    try:
        image_col, _, _ = indexing_service.get_collections()
        results = image_col.get(
            where={"video_id": video_id},
            include=["metadatas"],
        )

        if results and results["metadatas"]:
            target_frames = []
            for meta in results["metadatas"]:
                ts = meta.get("timestamp", 0)
                if abs(ts - timestamp) <= window_sec:
                    frame_path = meta.get("frame_path", "")
                    if frame_path:
                        # Convert URL to filesystem path
                        frame_name = Path(frame_path).name
                        fs_path = frames_dir / frame_name
                        if fs_path.exists():
                            target_frames.append(str(fs_path))

            for frame_path in sorted(target_frames):
                with open(frame_path, "rb") as f:
                    hasher.update(f.read())
                    hashed_count += 1

    except Exception as e:
        logger.warning("Failed to hash frames for clip: %s", e)

    # Also hash the audio segment if available
    audio_path = AUDIO_DIR / f"{video_id}.wav"
    if audio_path.exists():
        try:
            import wave
            with wave.open(str(audio_path), 'rb') as wf:
                framerate = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()

                start_frame = max(0, int((timestamp - window_sec) * framerate))
                end_frame = int((timestamp + window_sec) * framerate)

                wf.setpos(start_frame)
                n_frames = end_frame - start_frame
                audio_data = wf.readframes(n_frames)
                hasher.update(audio_data)
                hashed_count += 1
        except Exception as e:
            logger.debug("Audio hashing skipped: %s", e)

    if hashed_count == 0:
        return ""

    clip_hash = hasher.hexdigest()
    logger.debug("Computed clip hash for %s@%.1fs: %s (%d files hashed)",
                 video_id, timestamp, clip_hash[:16], hashed_count)
    return clip_hash


# ── Citation Verification ────────────────────────────────────────────────────

VERIFY_PROMPT = """You are a citation verification system for legal video analysis.

Given a CLAIM from an AI-generated answer and the EVIDENCE from the cited timestamp,
determine if the evidence actually supports the claim.

Respond with ONLY valid JSON:
{
    "supported": true | false,
    "confidence": 0.0 to 1.0,
    "explanation": "Brief explanation"
}

Rules:
- "supported" = true means the evidence at the cited timestamp genuinely supports the claim.
- Be strict: if the evidence doesn't clearly support the specific claim, mark it as unsupported.
- Partial support with low confidence is better than false verification."""


def verify_citation(claim_text: str, cited_timestamp: float,
                    video_id: str) -> dict:
    """
    Verify that the evidence at a cited timestamp supports the claim.

    Retrieves transcript/caption evidence at the timestamp and asks the LLM
    to verify the citation is accurate.

    Returns:
        {"supported": bool, "confidence": float, "explanation": str}
    """
    default = {"supported": False, "confidence": 0.0,
               "explanation": "Verification unavailable."}

    if not GROQ_API_KEY:
        return default

    # Gather evidence at the cited timestamp
    evidence_parts = []

    # Get transcript segments near the timestamp
    try:
        speech_col = indexing_service.get_collections()[2]  # speech collection
        count = speech_col.count()
        if count > 0:
            results = speech_col.get(
                where={"video_id": video_id},
                include=["metadatas", "documents"],
            )
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"]):
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    st = meta.get("start_time", 0)
                    et = meta.get("end_time", 0)
                    # Within ±5 seconds of cited timestamp
                    if abs(st - cited_timestamp) <= 5.0 or abs(et - cited_timestamp) <= 5.0:
                        speaker = meta.get("speaker_id", "")
                        role = meta.get("role", "")
                        prefix = f"[{speaker}/{role}] " if speaker else ""
                        evidence_parts.append(f"Transcript [{st:.0f}s-{et:.0f}s]: {prefix}{doc}")
    except Exception as e:
        logger.debug("Failed to get transcript evidence: %s", e)

    # Get captions near the timestamp
    try:
        caption_col = indexing_service.get_collections()[1]  # caption collection
        count = caption_col.count()
        if count > 0:
            results = caption_col.get(
                where={"video_id": video_id},
                include=["metadatas", "documents"],
            )
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"]):
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    ts = meta.get("timestamp", 0)
                    if abs(ts - cited_timestamp) <= 3.0:
                        evidence_parts.append(f"Caption [{ts:.0f}s]: {doc}")
    except Exception as e:
        logger.debug("Failed to get caption evidence: %s", e)

    if not evidence_parts:
        return {"supported": False, "confidence": 0.0,
                "explanation": "No evidence found at cited timestamp."}

    evidence_text = "\n".join(evidence_parts[:5])  # Limit evidence

    try:
        client = _get_groq_client()
        user_msg = f"""CLAIM: "{claim_text}"

CITED TIMESTAMP: {cited_timestamp:.0f} seconds in video {video_id}

EVIDENCE AT TIMESTAMP:
{evidence_text}

Does this evidence support the claim?"""

        result = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": VERIFY_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        text = result.choices[0].message.content.strip()

        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        parsed = json.loads(text)
        return {
            "supported": bool(parsed.get("supported", False)),
            "confidence": min(1.0, max(0.0, float(parsed.get("confidence", 0.5)))),
            "explanation": parsed.get("explanation", ""),
        }
    except Exception as e:
        logger.warning("Citation verification failed: %s", e)
        return default


def verify_and_store_citations(answer: str, video_id: str,
                               answer_id: str = None) -> list[dict]:
    """
    Full citation verification pipeline:
    1. Extract timestamps from the answer
    2. Verify each citation against evidence at that timestamp
    3. Compute clip hashes for tamper-evidence
    4. Store results in citation_log

    Returns list of citation verification results.
    """
    if not answer_id:
        answer_id = str(uuid.uuid4())[:12]

    citations = extract_citations_from_answer(answer)
    if not citations:
        return []

    results = []
    for cite in citations:
        timestamp = cite["timestamp_sec"]
        context = cite["context"]

        # Verify the citation
        verification = verify_citation(context, timestamp, video_id)

        # Compute tamper-evident hash
        clip_hash = compute_clip_hash(video_id, timestamp)

        # Determine result category
        if verification["supported"] and verification["confidence"] >= 0.7:
            result_label = "verified"
        elif verification["confidence"] >= 0.4:
            result_label = "uncertain"
        else:
            result_label = "unsupported"

        # Store in citation log
        citation_id = citation_db.store_citation(
            answer_id=answer_id,
            cited_timestamp=timestamp,
            cited_video=video_id,
            claim_text=context,
            verification_score=verification["confidence"],
            verification_result=result_label,
            clip_hash=clip_hash,
        )

        results.append({
            "citation_id": citation_id,
            "timestamp": timestamp,
            "timestamp_text": cite["timestamp_text"],
            "verification_score": verification["confidence"],
            "verification_result": result_label,
            "explanation": verification["explanation"],
            "clip_hash": clip_hash[:16] + "..." if clip_hash else "",
        })

    verified_count = sum(1 for r in results if r["verification_result"] == "verified")
    logger.info(
        "Citation verification: %d/%d citations verified for answer %s",
        verified_count, len(results), answer_id
    )

    return results
