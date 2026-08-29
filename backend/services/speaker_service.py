"""
VideoRAG — Speaker Service (Phase 6)
Speaker diarization, role detection, and cross-session voiceprint matching.

Primary: pyannote-audio for speaker diarization + Resemblyzer for voiceprint embedding.
Fallback: gap-based heuristic when pyannote is unavailable.
"""
import sqlite3
import json
import logging
import struct
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
    """Get SQLite connection for speaker role mappings and voiceprint registry."""
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

            CREATE TABLE IF NOT EXISTS speaker_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'unknown',
                voiceprint_blob BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(case_id, canonical_name)
            );
            CREATE INDEX IF NOT EXISTS idx_registry_case
                ON speaker_registry(case_id);

            CREATE TABLE IF NOT EXISTS speaker_video_map (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                local_speaker_id TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                confirmed_by_user INTEGER DEFAULT 0,
                voiceprint_blob BLOB,
                UNIQUE(video_id, local_speaker_id)
            );
            CREATE INDEX IF NOT EXISTS idx_svm_video
                ON speaker_video_map(video_id);
            CREATE INDEX IF NOT EXISTS idx_svm_case
                ON speaker_video_map(case_id);
        """)
        # Defensive migration: speaker_video_map may already exist from a
        # prior run without this column (CREATE TABLE IF NOT EXISTS won't
        # add it to an existing table).
        try:
            conn.execute("ALTER TABLE speaker_video_map ADD COLUMN voiceprint_blob BLOB")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
        _db_initialized = True
        logger.info("Speaker roles + registry tables initialized.")

    return conn


# ── Voiceprint Utilities ─────────────────────────────────────────────────────

def _blob_to_vector(blob: bytes) -> list[float]:
    """Deserialize voiceprint vector from blob."""
    if not blob:
        return []
    n = len(blob) // 4  # float32 = 4 bytes
    return list(struct.unpack(f'{n}f', blob))


def _vector_to_blob(vector: list[float]) -> bytes:
    """Serialize voiceprint vector to blob."""
    return struct.pack(f'{len(vector)}f', *vector)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Pyannote Diarization ─────────────────────────────────────────────────────

_diarization_pipeline = None
_pyannote_available = None


def _is_pyannote_available() -> bool:
    """Check if pyannote.audio is installed and usable."""
    global _pyannote_available
    if _pyannote_available is not None:
        return _pyannote_available
    try:
        from pyannote.audio import Pipeline
        _pyannote_available = True
    except ImportError:
        _pyannote_available = False
        logger.info("pyannote.audio not installed — will use gap-based speaker detection.")
    return _pyannote_available


def _get_diarization_pipeline():
    """Lazy-load the pyannote diarization pipeline."""
    global _diarization_pipeline
    if _diarization_pipeline is not None:
        return _diarization_pipeline

    try:
        from pyannote.audio import Pipeline
        import inspect
        import os
        hf_token = os.environ.get("HF_TOKEN", "")
        logger.info("Loading pyannote diarization pipeline...")
        # pyannote.audio 4.x renamed from_pretrained's auth kwarg from
        # use_auth_token to token; support both installed API generations.
        kwarg_name = "token" if "token" in inspect.signature(Pipeline.from_pretrained).parameters else "use_auth_token"
        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            **{kwarg_name: hf_token if hf_token else None},
        )
        logger.info("Pyannote diarization pipeline loaded successfully.")
        return _diarization_pipeline
    except Exception as e:
        logger.warning("Failed to load pyannote pipeline: %s — falling back to gap-based detection.", e)
        return None


def diarize_audio(audio_path: str) -> list[dict]:
    """
    Run pyannote speaker diarization on an audio file.

    Returns list of speaker turns:
        [{"start_time": float, "end_time": float, "speaker_id": str}, ...]
    """
    if not _is_pyannote_available():
        return []

    pipeline = _get_diarization_pipeline()
    if pipeline is None:
        return []

    try:
        diarization = pipeline(audio_path)
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({
                "start_time": turn.start,
                "end_time": turn.end,
                "speaker_id": speaker,
            })
        logger.info("Pyannote diarization found %d turns from %d speakers.",
                     len(turns), len(set(t["speaker_id"] for t in turns)))
        return turns
    except Exception as e:
        logger.warning("Pyannote diarization failed: %s", e)
        return []


def align_segments_with_diarization(
    segments: list[dict],
    diarization_turns: list[dict],
) -> list[dict]:
    """
    Align Whisper ASR segments with pyannote diarization turns.

    For each ASR segment, find the diarization turn with maximum temporal overlap
    and assign that turn's speaker_id.

    Args:
        segments: Whisper output [{"text", "start_time", "end_time"}, ...]
        diarization_turns: Pyannote output [{"start_time", "end_time", "speaker_id"}, ...]

    Returns:
        segments with speaker_id assigned from diarization.
    """
    if not diarization_turns:
        return segments

    for seg in segments:
        seg_start = seg.get("start_time", 0.0)
        seg_end = seg.get("end_time", 0.0)
        seg_duration = max(seg_end - seg_start, 0.001)

        best_speaker = None
        best_overlap = 0.0

        for turn in diarization_turns:
            turn_start = turn["start_time"]
            turn_end = turn["end_time"]

            # Calculate overlap
            overlap_start = max(seg_start, turn_start)
            overlap_end = min(seg_end, turn_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker_id"]

        if best_speaker and best_overlap > seg_duration * 0.3:
            seg["speaker_id"] = best_speaker
        else:
            # If no good overlap, assign "UNKNOWN"
            seg["speaker_id"] = seg.get("speaker_id", "SPEAKER_UNKNOWN")

    return segments


# ── Gap-Based Speaker Detection (Fallback) ───────────────────────────────────

def assign_speaker_ids(segments: list[dict],
                       gap_threshold: float = 2.0) -> list[dict]:
    """
    Assign speaker IDs to ASR segments using gap-based turn detection.

    When there is a silence gap longer than `gap_threshold` seconds between
    consecutive segments, we assume a different speaker is talking.

    This is used as a fallback when pyannote-audio is not available.

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
        "Gap-based: assigned %d speaker IDs to %d segments (gap_threshold=%.1fs)",
        n_speakers, len(segments), gap_threshold
    )
    return segments


# ── Combined Diarization Entry Point ─────────────────────────────────────────

def diarize_and_assign(segments: list[dict], audio_path: str = None) -> list[dict]:
    """
    Main entry point for speaker diarization.

    1. If audio_path is provided and pyannote is available, use pyannote diarization.
    2. Otherwise, fall back to gap-based heuristic.

    Args:
        segments: Whisper ASR segments.
        audio_path: Path to the audio file for pyannote diarization.

    Returns:
        segments with speaker_id assigned.
    """
    if audio_path and _is_pyannote_available():
        turns = diarize_audio(audio_path)
        if turns:
            logger.info("Using pyannote diarization for speaker assignment.")
            return align_segments_with_diarization(segments, turns)
        logger.warning("Pyannote returned no turns, falling back to gap-based detection.")

    return assign_speaker_ids(segments)


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


# ── Voiceprint Extraction ────────────────────────────────────────────────────

_voice_encoder = None
_resemblyzer_available = None


def _is_resemblyzer_available() -> bool:
    """Check if Resemblyzer is installed."""
    global _resemblyzer_available
    if _resemblyzer_available is not None:
        return _resemblyzer_available
    try:
        from resemblyzer import VoiceEncoder
        _resemblyzer_available = True
    except ImportError:
        _resemblyzer_available = False
        logger.info("Resemblyzer not installed — voiceprint matching disabled.")
    return _resemblyzer_available


def _get_voice_encoder():
    """Lazy-load the Resemblyzer voice encoder."""
    global _voice_encoder
    if _voice_encoder is not None:
        return _voice_encoder

    try:
        from resemblyzer import VoiceEncoder
        logger.info("Loading Resemblyzer voice encoder...")
        _voice_encoder = VoiceEncoder()
        logger.info("Resemblyzer voice encoder loaded.")
        return _voice_encoder
    except Exception as e:
        logger.warning("Failed to load Resemblyzer: %s", e)
        return None


def extract_voiceprint(audio_path: str, start_time: float = 0.0,
                       end_time: float = None) -> list[float]:
    """
    Extract a voiceprint embedding vector for a speaker from an audio segment.

    Uses Resemblyzer (GE2E / ECAPA-TDNN equivalent) to produce a 256-dim
    voice embedding vector from a segment of audio.

    Args:
        audio_path: Path to the full audio file.
        start_time: Start time in seconds.
        end_time: End time in seconds (None = to end).

    Returns:
        256-dimensional voiceprint vector, or empty list on failure.
    """
    if not _is_resemblyzer_available():
        return []

    encoder = _get_voice_encoder()
    if encoder is None:
        return []

    try:
        from resemblyzer import preprocess_wav
        import numpy as np

        wav = preprocess_wav(audio_path)
        sr = 16000  # Resemblyzer expects 16kHz

        # Slice to the speaker's segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr) if end_time else len(wav)
        segment_wav = wav[start_sample:end_sample]

        if len(segment_wav) < sr * 0.5:  # Need at least 0.5 seconds
            return []

        embedding = encoder.embed_utterance(segment_wav)
        return embedding.tolist()
    except Exception as e:
        logger.warning("Voiceprint extraction failed: %s", e)
        return []


def extract_voiceprints_for_speakers(
    audio_path: str,
    segments: list[dict],
) -> dict[str, list[float]]:
    """
    Extract voiceprint for each unique speaker by aggregating their speech segments.

    For each speaker_id, concatenate all their speech segments and extract
    a single voiceprint embedding.

    Returns:
        Dict of {speaker_id: voiceprint_vector}.
    """
    if not _is_resemblyzer_available():
        return {}

    encoder = _get_voice_encoder()
    if encoder is None:
        return {}

    try:
        from resemblyzer import preprocess_wav
        import numpy as np

        wav = preprocess_wav(audio_path)
        sr = 16000

        # Group segments by speaker
        speaker_segments: dict[str, list[tuple[float, float]]] = {}
        for seg in segments:
            sid = seg.get("speaker_id")
            if not sid:
                continue
            start = seg.get("start_time", 0.0)
            end = seg.get("end_time", 0.0)
            speaker_segments.setdefault(sid, []).append((start, end))

        voiceprints = {}
        for sid, time_ranges in speaker_segments.items():
            # Concatenate all audio for this speaker
            audio_chunks = []
            for start, end in time_ranges:
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                chunk = wav[start_sample:end_sample]
                if len(chunk) > 0:
                    audio_chunks.append(chunk)

            if not audio_chunks:
                continue

            combined = np.concatenate(audio_chunks)

            # Cap total audio fed to the encoder in one call. A long
            # courtroom session can give one speaker 10+ minutes of
            # concatenated audio, which both OOMs embed_utterance() on
            # a modest GPU and buys no real accuracy — GE2E voice
            # embeddings are trained on short utterances, so ~45s of
            # clean speech is already enough for a robust voiceprint.
            max_samples = 45 * sr
            if len(combined) > max_samples:
                combined = combined[:max_samples]

            if len(combined) < sr * 0.5:
                continue

            # Use embed_utterance for the combined audio
            embedding = encoder.embed_utterance(combined)
            voiceprints[sid] = embedding.tolist()

        logger.info("Extracted voiceprints for %d speakers.", len(voiceprints))
        return voiceprints
    except Exception as e:
        logger.warning("Voiceprint batch extraction failed: %s", e)
        return {}


# ── Cross-Session Speaker Matching ───────────────────────────────────────────

def match_voiceprint_to_registry(
    case_id: str,
    voiceprint: list[float],
    threshold: float = 0.75,
) -> tuple[str | None, float]:
    """
    Compare a voiceprint against the speaker registry for a case.

    Returns:
        (canonical_name, similarity_score) if a match is found above threshold,
        or (None, 0.0) if no match.
    """
    if not voiceprint:
        return None, 0.0

    conn = _get_conn()
    rows = conn.execute(
        "SELECT canonical_name, voiceprint_blob FROM speaker_registry WHERE case_id = ?",
        (case_id,)
    ).fetchall()
    conn.close()

    best_name = None
    best_score = 0.0

    for row in rows:
        stored_vp = _blob_to_vector(row["voiceprint_blob"])
        if not stored_vp:
            continue
        score = _cosine_similarity(voiceprint, stored_vp)
        if score > best_score:
            best_score = score
            best_name = row["canonical_name"]

    if best_score >= threshold:
        return best_name, best_score
    return None, best_score


def register_speaker(case_id: str, canonical_name: str, role: str = "unknown",
                     voiceprint: list[float] = None) -> dict:
    """
    Register a new speaker in the case's speaker registry.

    If a speaker with the same canonical_name already exists in this case,
    update their voiceprint and role.
    """
    conn = _get_conn()
    vp_blob = _vector_to_blob(voiceprint) if voiceprint else None

    conn.execute(
        """INSERT INTO speaker_registry (case_id, canonical_name, role, voiceprint_blob)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(case_id, canonical_name)
           DO UPDATE SET role = excluded.role,
                         voiceprint_blob = COALESCE(excluded.voiceprint_blob, voiceprint_blob)""",
        (case_id, canonical_name, role.lower(), vp_blob)
    )
    conn.commit()
    conn.close()

    logger.info("Registered speaker '%s' (role=%s) in case %s", canonical_name, role, case_id)
    return {"case_id": case_id, "canonical_name": canonical_name, "role": role}


def map_local_speaker_to_canonical(
    case_id: str,
    video_id: str,
    local_speaker_id: str,
    canonical_name: str,
    confidence: float = 0.0,
    confirmed: bool = False,
    voiceprint: list[float] = None,
) -> dict:
    """
    Map a video-local speaker_id to a case-level canonical name.

    This creates the link between e.g. video A's SPEAKER_0 and the
    case-level canonical name "witness_john_doe". When provided, the local
    speaker's own voiceprint is stored alongside the mapping so a later
    human correction (see confirm_speaker_match) can re-register it under a
    different canonical identity without re-running diarization.
    """
    conn = _get_conn()
    vp_blob = _vector_to_blob(voiceprint) if voiceprint else None
    conn.execute(
        """INSERT INTO speaker_video_map
           (case_id, video_id, local_speaker_id, canonical_name, confidence,
            confirmed_by_user, voiceprint_blob)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(video_id, local_speaker_id)
           DO UPDATE SET canonical_name = excluded.canonical_name,
                         confidence = excluded.confidence,
                         confirmed_by_user = excluded.confirmed_by_user,
                         voiceprint_blob = COALESCE(excluded.voiceprint_blob, voiceprint_blob)""",
        (case_id, video_id, local_speaker_id, canonical_name,
         confidence, 1 if confirmed else 0, vp_blob)
    )
    conn.commit()
    conn.close()

    return {"video_id": video_id, "local_speaker_id": local_speaker_id,
            "canonical_name": canonical_name}


def get_canonical_name(video_id: str, local_speaker_id: str) -> str | None:
    """Get the canonical name for a local speaker_id in a video."""
    conn = _get_conn()
    row = conn.execute(
        """SELECT canonical_name FROM speaker_video_map
           WHERE video_id = ? AND local_speaker_id = ?""",
        (video_id, local_speaker_id)
    ).fetchone()
    conn.close()
    return row["canonical_name"] if row else None


def match_and_register_speakers(
    case_id: str,
    video_id: str,
    audio_path: str,
    segments: list[dict],
    threshold: float = 0.75,
) -> dict[str, str]:
    """
    Full cross-session speaker matching workflow:

    1. Extract voiceprints for each speaker in this video.
    2. Compare against the case's speaker registry.
    3. If match found (above threshold): map local speaker → canonical name.
    4. If no match: register as a new speaker in the registry.

    Returns:
        Dict of {local_speaker_id: canonical_name} mappings.
    """
    mappings = {}

    # Step 1: Extract voiceprints
    voiceprints = extract_voiceprints_for_speakers(audio_path, segments)

    # Collect unique speakers and their roles
    speaker_roles = {}
    for seg in segments:
        sid = seg.get("speaker_id")
        if sid and sid not in speaker_roles:
            speaker_roles[sid] = seg.get("role", "unknown")

    # Step 2-4: Match or register each speaker
    for local_sid, vp in voiceprints.items():
        role = speaker_roles.get(local_sid, "unknown")

        # Try to match against existing registry
        canonical, score = match_voiceprint_to_registry(case_id, vp, threshold)

        if canonical:
            # Match found — map this video's speaker to the existing canonical name
            logger.info(
                "Voiceprint match: %s in video %s → '%s' (score=%.3f)",
                local_sid, video_id, canonical, score
            )
            map_local_speaker_to_canonical(
                case_id, video_id, local_sid, canonical,
                confidence=score, confirmed=False, voiceprint=vp,
            )
            mappings[local_sid] = canonical
        else:
            # No match — register as new speaker
            new_name = f"{role}_{local_sid}_{video_id[:6]}"
            register_speaker(case_id, new_name, role=role, voiceprint=vp)
            map_local_speaker_to_canonical(
                case_id, video_id, local_sid, new_name,
                confidence=1.0, confirmed=False, voiceprint=vp,
            )
            mappings[local_sid] = new_name
            logger.info(
                "New speaker registered: %s → '%s' (role=%s)",
                local_sid, new_name, role
            )

    # Handle speakers without voiceprints
    for local_sid in speaker_roles:
        if local_sid not in mappings:
            role = speaker_roles[local_sid]
            new_name = f"{role}_{local_sid}_{video_id[:6]}"
            map_local_speaker_to_canonical(
                case_id, video_id, local_sid, new_name,
                confidence=0.0, confirmed=False
            )
            mappings[local_sid] = new_name

    return mappings


# ── Speaker Match Confirmation (Novelty 1, Step 5) ───────────────────────────
#
# match_and_register_speakers() auto-matches voiceprints across sessions, but
# the plan requires a human-in-the-loop confirmation before that match is
# trusted downstream by contradiction detection. These functions expose the
# pending auto-matches for a case and let a user confirm or correct them.

def get_pending_matches(case_id: str) -> list[dict]:
    """Get all speaker_video_map rows for a case not yet confirmed by a user."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT video_id, local_speaker_id, canonical_name, confidence, confirmed_by_user
           FROM speaker_video_map
           WHERE case_id = ? AND confirmed_by_user = 0
           ORDER BY video_id, local_speaker_id""",
        (case_id,)
    ).fetchall()
    conn.close()

    results = []
    conn = _get_conn()
    for row in rows:
        reg_row = conn.execute(
            "SELECT role FROM speaker_registry WHERE case_id = ? AND canonical_name = ?",
            (case_id, row["canonical_name"])
        ).fetchone()
        results.append({
            "video_id": row["video_id"],
            "local_speaker_id": row["local_speaker_id"],
            "canonical_name": row["canonical_name"],
            "confidence": row["confidence"],
            "role": reg_row["role"] if reg_row else "unknown",
            "confirmed_by_user": bool(row["confirmed_by_user"]),
        })
    conn.close()
    return results


def list_case_speakers(case_id: str) -> list[dict]:
    """List all canonical speakers registered for a case."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT canonical_name, role FROM speaker_registry WHERE case_id = ? ORDER BY canonical_name",
        (case_id,)
    ).fetchall()
    conn.close()
    return [{"canonical_name": r["canonical_name"], "role": r["role"]} for r in rows]


def confirm_speaker_match(
    case_id: str,
    video_id: str,
    local_speaker_id: str,
    action: str,
    corrected_name: str = None,
    role: str = None,
) -> dict:
    """
    Human-in-the-loop confirmation/correction of a cross-session voiceprint match.

    action="confirm": accept the existing auto-match as-is, mark confirmed.
    action="rename": the auto-match was wrong — reassign this local speaker to
        `corrected_name`. If that name already exists in the case's registry,
        relink to it (merging in this local speaker's stored voiceprint to
        improve future matching). Otherwise register it as a brand-new speaker
        using the voiceprint captured at match time.
    """
    conn = _get_conn()
    row = conn.execute(
        """SELECT canonical_name, confidence, voiceprint_blob FROM speaker_video_map
           WHERE video_id = ? AND local_speaker_id = ?""",
        (video_id, local_speaker_id)
    ).fetchone()
    conn.close()

    if row is None:
        raise ValueError(f"No speaker mapping found for video={video_id}, speaker={local_speaker_id}")

    stored_vp = _blob_to_vector(row["voiceprint_blob"]) if row["voiceprint_blob"] else None

    if action == "confirm":
        return map_local_speaker_to_canonical(
            case_id, video_id, local_speaker_id, row["canonical_name"],
            confidence=row["confidence"], confirmed=True, voiceprint=stored_vp,
        )

    if action == "rename":
        if not corrected_name:
            raise ValueError("corrected_name is required for action='rename'")

        conn = _get_conn()
        existing = conn.execute(
            "SELECT canonical_name, role FROM speaker_registry WHERE case_id = ? AND canonical_name = ?",
            (case_id, corrected_name)
        ).fetchone()
        conn.close()

        final_role = role or (existing["role"] if existing else "unknown")
        register_speaker(case_id, corrected_name, role=final_role, voiceprint=stored_vp)

        return map_local_speaker_to_canonical(
            case_id, video_id, local_speaker_id, corrected_name,
            confidence=1.0, confirmed=True, voiceprint=stored_vp,
        )

    raise ValueError(f"Unknown action: {action!r} (expected 'confirm' or 'rename')")


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


def delete_video_speakers(video_id: str):
    """Remove a video's speaker footprint: its role labels and its
    local-speaker-to-canonical-name mappings.

    Deliberately does NOT touch speaker_registry — a canonical identity may
    still be legitimately referenced by that speaker's OTHER videos in the
    same case, and their voiceprint should stay available for future
    cross-session matching even after this one video is deleted.
    """
    conn = _get_conn()
    conn.execute("DELETE FROM speaker_roles WHERE video_id = ?", (video_id,))
    conn.execute("DELETE FROM speaker_video_map WHERE video_id = ?", (video_id,))
    conn.commit()
    conn.close()
    logger.info("Deleted speaker roles/mappings for video %s", video_id)


def get_speaker_role(video_id: str, speaker_id: str) -> str:
    """Get the role for a specific speaker in a video. Returns 'unknown' if not set."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT role FROM speaker_roles WHERE video_id = ? AND speaker_id = ?",
        (video_id, speaker_id)
    ).fetchone()
    conn.close()
    return row["role"] if row else "unknown"
