"""
VideoRAG — Contradiction Detection Service (Phase 7)
Uses Groq LLM zero-shot classification to detect contradictions between
testimony statements from the same speaker across different sessions.
"""
import json
import logging

from backend.config import GROQ_API_KEY, GROQ_MODEL
from backend.services import testimony_db

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


# ── Contradiction Checking ───────────────────────────────────────────────────

CONTRADICTION_PROMPT = """You are a legal testimony consistency analyzer.

Given two statements from the SAME speaker (possibly from different court sessions),
determine if they are:
1. **CONTRADICT** — The statements make incompatible factual claims (e.g., different times, different actions, different people involved).
2. **ENTAIL** — The statements are consistent and one supports the other.
3. **UNRELATED** — The statements discuss different topics and cannot be compared.

Respond with ONLY valid JSON:
{
    "label": "CONTRADICT" | "ENTAIL" | "UNRELATED",
    "confidence": 0.0 to 1.0,
    "explanation": "Brief explanation of why"
}

Focus on FACTUAL contradictions, not stylistic differences. Minor wording
differences are NOT contradictions. Only flag clear factual inconsistencies."""


def check_contradiction(text_a: str, text_b: str) -> dict:
    """
    Use Groq LLM zero-shot classification to determine if two statements
    contradict each other.

    Returns:
        {"label": "CONTRADICT"|"ENTAIL"|"UNRELATED",
         "confidence": float, "explanation": str}
    """
    default = {"label": "UNRELATED", "confidence": 0.0, "explanation": "Unable to check."}

    if not GROQ_API_KEY:
        return default

    if not text_a.strip() or not text_b.strip():
        return default

    # Skip very short statements — they're unlikely to contain contradictions
    if len(text_a.strip()) < 10 or len(text_b.strip()) < 10:
        return default

    try:
        client = _get_groq_client()
        user_msg = f"""Statement A: "{text_a.strip()}"

Statement B: "{text_b.strip()}"

Are these statements contradictory, entailing, or unrelated?"""

        result = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": CONTRADICTION_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        text = result.choices[0].message.content.strip()

        # Extract JSON from response
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        parsed = json.loads(text)

        label = parsed.get("label", "UNRELATED").upper()
        if label not in ("CONTRADICT", "ENTAIL", "UNRELATED"):
            label = "UNRELATED"

        return {
            "label": label,
            "confidence": min(1.0, max(0.0, float(parsed.get("confidence", 0.5)))),
            "explanation": parsed.get("explanation", ""),
        }
    except Exception as e:
        logger.warning("Contradiction check failed: %s", e)
        return default


def scan_for_contradictions(case_id: str, video_id: str,
                            segments: list[dict],
                            min_confidence: float = 0.6) -> list[dict]:
    """
    Scan new ASR segments against prior testimony for the same speaker
    in the same case. Flag any contradictions found.

    Args:
        case_id: The case identifier.
        video_id: The current video being processed.
        segments: New ASR segments with speaker_id, role, text, start_time.
        min_confidence: Minimum confidence to flag a contradiction.

    Returns:
        List of detected contradictions.
    """
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — skipping contradiction scanning.")
        return []

    detected = []

    # Group new segments by speaker
    speaker_segments: dict[str, list[dict]] = {}
    for seg in segments:
        sid = seg.get("speaker_id", "SPEAKER_0")
        speaker_segments.setdefault(sid, []).append(seg)

    for speaker_id, new_segs in speaker_segments.items():
        # Get prior statements by this speaker from earlier sessions
        prior_statements = testimony_db.get_statements_by_speaker(
            case_id=case_id,
            speaker_id=speaker_id,
            exclude_video_id=video_id,  # Don't compare against same video
            limit=30,
        )

        if not prior_statements:
            continue  # No prior testimony to compare against

        logger.info(
            "Checking %d new segments vs %d prior statements for %s in case %s",
            len(new_segs), len(prior_statements), speaker_id, case_id
        )

        # Check each new segment against relevant prior statements
        # (only check substantive segments, skip very short ones)
        substantive_new = [s for s in new_segs if len(s.get("text", "")) > 15]

        for new_seg in substantive_new[:10]:  # Limit to avoid too many API calls
            new_text = new_seg.get("text", "")

            for prior in prior_statements[:10]:  # Compare against top 10 priors
                prior_text = prior.get("text", "")

                result = check_contradiction(new_text, prior_text)

                if result["label"] == "CONTRADICT" and result["confidence"] >= min_confidence:
                    # Store the contradiction
                    new_stmt_id = new_seg.get("stmt_id", "unknown")
                    prior_stmt_id = prior.get("id", "unknown")

                    contradiction_id = testimony_db.store_contradiction(
                        case_id=case_id,
                        speaker_id=speaker_id,
                        stmt_a_id=prior_stmt_id,
                        stmt_a_text=prior_text,
                        stmt_a_timestamp=prior.get("timestamp", 0.0),
                        stmt_a_video_id=prior.get("video_id", ""),
                        stmt_b_id=new_stmt_id,
                        stmt_b_text=new_text,
                        stmt_b_timestamp=new_seg.get("start_time", 0.0),
                        stmt_b_video_id=video_id,
                        confidence=result["confidence"],
                        explanation=result["explanation"],
                    )

                    detected.append({
                        "contradiction_id": contradiction_id,
                        "speaker_id": speaker_id,
                        "statement_a": prior_text,
                        "statement_b": new_text,
                        "confidence": result["confidence"],
                        "explanation": result["explanation"],
                    })

                    logger.warning(
                        "CONTRADICTION DETECTED for %s: %.0f%% confidence. "
                        "A: '%s' vs B: '%s'",
                        speaker_id, result["confidence"] * 100,
                        prior_text[:60], new_text[:60]
                    )

    if detected:
        logger.info("Detected %d contradictions in case %s", len(detected), case_id)
    else:
        logger.info("No contradictions detected in case %s for video %s",
                     case_id, video_id)

    return detected
