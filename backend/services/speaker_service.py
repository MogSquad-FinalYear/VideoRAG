"""
VideoRAG — Speaker Service (Phase 6)
Speaker turn detection and role mapping for courtroom video analysis.

Uses gap-based segmentation from Whisper ASR segments to separate speaker turns,
then allows manual or rule-based role assignment (judge/witness/counsel).
"""
import sqlite3
import logging
from pathlib import Path

from backend.config import TESTIMONY_DB_PATH

logger = logging.getLogger(__name__)

_db_initialized = False

# ── Role Detection Keywords ──────────────────────────────────────────────────

_JUDGE_CUES = [
    "the court finds", "the court orders", "sustained", "overruled",
    "i will allow", "the court will", "order in the court",
    "you may proceed", "the witness may", "let the record show",
    "this court", "i'm going to sustain", "i'm going to overrule",
    "members of the jury", "the jury is instructed",
]

_COUNSEL_CUES = [
    "objection", "your honor", "i object", "move to strike",
    "i'd like to enter", "exhibit", "permission to approach",
    "no further questions", "the prosecution rests",
    "the defense rests", "redirect", "cross-examination",
    "direct examination", "ladies and gentlemen of the jury",
    "may it please the court",
]

_WITNESS_CUES = [
    "i saw", "i was", "i remember", "i don't recall",
    "i witnessed", "i noticed", "from where i was standing",
    "at that time", "i believe", "to the best of my recollection",
    "i swear", "i affirm", "so help me god",
]


def _get_conn() -> sqlite3.Connection:
    """Get SQLite connection for speaker role mappings."""
    global _db_initialized
    conn = sqlite3.connect(str(TESTIMONY_DB_PATH))
    conn.row_factory = sqlite3.Row

    if not _db_initialized:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS speaker_roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                speaker_id TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'unknown',
                label TEXT DEFAULT '',
                UNIQUE(video_id, speaker_id)
            );
            CREATE INDEX IF NOT EXISTS idx_speaker_video
                ON speaker_roles(video_id);
        """)
        _db_initialized = True
        logger.info("Speaker roles table initialized.")

    return conn


# ── Speaker Turn Detection ───────────────────────────────────────────────────

def assign_speaker_ids(segments: list[dict],
                       gap_threshold: float = 2.0) -> list[dict]:
    """
    Assign speaker IDs to ASR segments using gap-based turn detection.

    When there is a silence gap longer than `gap_threshold` seconds between
    consecutive segments, we assume a different speaker is talking.

    This is a practical first-version heuristic. For production-grade
    diarization, swap in pyannote-audio here.

    Args:
        segments: List of {"text", "start_time", "end_time"} dicts from Whisper.
        gap_threshold: Minimum silence gap (seconds) to signal a speaker change.

    Returns:
        Same segments with added "speaker_id" field (e.g. "SPEAKER_0").
    """
    if not segments:
        return segments

    current_speaker = 0
    prev_end = segments[0].get("end_time", 0.0)

    for i, seg in enumerate(segments):
        start = seg.get("start_time", 0.0)
        end = seg.get("end_time", 0.0)

        if i == 0:
            seg["speaker_id"] = f"SPEAKER_{current_speaker}"
            prev_end = end
            continue

        # If gap between end of previous and start of current > threshold,
        # assume a new speaker
        gap = start - prev_end
        if gap > gap_threshold:
            current_speaker += 1

        seg["speaker_id"] = f"SPEAKER_{current_speaker}"
        prev_end = end

    n_speakers = current_speaker + 1
    logger.info(
        "Assigned %d speaker IDs to %d segments (gap_threshold=%.1fs)",
        n_speakers, len(segments), gap_threshold
    )
    return segments


# ── Rule-Based Role Detection ────────────────────────────────────────────────

def auto_detect_role_from_text(text: str) -> str:
    """
    Attempt to detect speaker role from transcript text using keyword cues.

    Returns one of: "judge", "counsel", "witness", "unknown".
    """
    text_lower = text.lower()

    # Score each role by number of matching cue phrases
    judge_score = sum(1 for cue in _JUDGE_CUES if cue in text_lower)
    counsel_score = sum(1 for cue in _COUNSEL_CUES if cue in text_lower)
    witness_score = sum(1 for cue in _WITNESS_CUES if cue in text_lower)

    if judge_score == 0 and counsel_score == 0 and witness_score == 0:
        return "unknown"

    best = max(judge_score, counsel_score, witness_score)
    if best == judge_score:
        return "judge"
    elif best == counsel_score:
        return "counsel"
    else:
        return "witness"


def auto_detect_roles_for_segments(segments: list[dict]) -> list[dict]:
    """
    Apply rule-based role detection to segments that have speaker_id assigned.

    Groups all text per speaker, detects role for each speaker, then assigns
    the role back to all segments for that speaker.
    """
    if not segments:
        return segments

    # Collect all text per speaker
    speaker_texts: dict[str, list[str]] = {}
    for seg in segments:
        sid = seg.get("speaker_id", "SPEAKER_0")
        speaker_texts.setdefault(sid, []).append(seg.get("text", ""))

    # Detect role per speaker from combined text
    speaker_roles = {}
    for sid, texts in speaker_texts.items():
        combined = " ".join(texts)
        role = auto_detect_role_from_text(combined)
        speaker_roles[sid] = role

    # Assign roles back to segments
    for seg in segments:
        sid = seg.get("speaker_id", "SPEAKER_0")
        seg["role"] = speaker_roles.get(sid, "unknown")

    logger.info("Auto-detected roles: %s", speaker_roles)
    return segments


# ── Manual Speaker Role Mapping ──────────────────────────────────────────────

def set_speaker_role(video_id: str, speaker_id: str, role: str,
                     label: str = "") -> dict:
    """
    Store a manual speaker-to-role mapping.

    Args:
        video_id: The video identifier.
        speaker_id: e.g. "SPEAKER_0".
        role: One of "judge", "witness", "counsel", "unknown".
        label: Optional human-readable name (e.g. "John Doe").
    """
    valid_roles = {"judge", "witness", "counsel", "unknown"}
    if role.lower() not in valid_roles:
        role = "unknown"

    conn = _get_conn()
    conn.execute(
        """INSERT INTO speaker_roles (video_id, speaker_id, role, label)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(video_id, speaker_id)
           DO UPDATE SET role = excluded.role, label = excluded.label""",
        (video_id, speaker_id, role.lower(), label)
    )
    conn.commit()
    conn.close()

    logger.info("Set role for %s in video %s: %s (%s)",
                speaker_id, video_id, role, label or "no label")
    return {"video_id": video_id, "speaker_id": speaker_id,
            "role": role, "label": label}


def get_speaker_roles(video_id: str) -> list[dict]:
    """Get all speaker-to-role mappings for a video."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT speaker_id, role, label FROM speaker_roles WHERE video_id = ?",
        (video_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_speaker_role(video_id: str, speaker_id: str) -> str:
    """Get the role for a specific speaker in a video. Returns 'unknown' if not set."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT role FROM speaker_roles WHERE video_id = ? AND speaker_id = ?",
        (video_id, speaker_id)
    ).fetchone()
    conn.close()
    return row["role"] if row else "unknown"
