"""
VideoRAG — Testimony Memory Database (Phase 7)
Persistent cross-session testimony storage keyed by case_id (not video_id),
enabling contradiction detection across multiple video sessions in the same case.
"""
import sqlite3
import uuid
import logging
from datetime import datetime

from backend.config import TESTIMONY_DB_PATH

logger = logging.getLogger(__name__)

_db_initialized = False


def _get_conn() -> sqlite3.Connection:
    """Get SQLite connection with testimony tables."""
    global _db_initialized
    conn = sqlite3.connect(str(TESTIMONY_DB_PATH))
    conn.row_factory = sqlite3.Row

    if not _db_initialized:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS testimony_memory (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                speaker_id TEXT NOT NULL,
                role TEXT DEFAULT 'unknown',
                timestamp REAL NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_testimony_case
                ON testimony_memory(case_id);
            CREATE INDEX IF NOT EXISTS idx_testimony_speaker
                ON testimony_memory(case_id, speaker_id);
            CREATE INDEX IF NOT EXISTS idx_testimony_video
                ON testimony_memory(video_id);

            CREATE TABLE IF NOT EXISTS contradictions (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                speaker_id TEXT NOT NULL,
                stmt_a_id TEXT NOT NULL,
                stmt_a_text TEXT NOT NULL,
                stmt_a_timestamp REAL,
                stmt_a_video_id TEXT,
                stmt_b_id TEXT NOT NULL,
                stmt_b_text TEXT NOT NULL,
                stmt_b_timestamp REAL,
                stmt_b_video_id TEXT,
                confidence REAL NOT NULL,
                explanation TEXT DEFAULT '',
                detected_at TEXT NOT NULL,
                FOREIGN KEY (stmt_a_id) REFERENCES testimony_memory(id),
                FOREIGN KEY (stmt_b_id) REFERENCES testimony_memory(id)
            );
            CREATE INDEX IF NOT EXISTS idx_contradiction_case
                ON contradictions(case_id);
            CREATE INDEX IF NOT EXISTS idx_contradiction_speaker
                ON contradictions(case_id, speaker_id);
        """)
        _db_initialized = True
        logger.info("Testimony memory tables initialized at %s", TESTIMONY_DB_PATH)

    return conn


# ── Testimony Storage ────────────────────────────────────────────────────────

def store_statement(case_id: str, session_id: str, video_id: str,
                    speaker_id: str, role: str, timestamp: float,
                    text: str) -> str:
    """
    Store a single testimony statement in the cross-session memory.

    Returns the generated statement ID.
    """
    stmt_id = str(uuid.uuid4())[:12]
    conn = _get_conn()
    conn.execute(
        """INSERT INTO testimony_memory
           (id, case_id, session_id, video_id, speaker_id, role, timestamp, text, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (stmt_id, case_id, session_id, video_id, speaker_id, role,
         round(timestamp, 2), text.strip(), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return stmt_id


def store_statements_batch(case_id: str, session_id: str, video_id: str,
                           segments: list[dict]) -> list[str]:
    """
    Store a batch of testimony statements from transcription segments.

    Each segment should have: text, start_time, speaker_id, role.
    Returns list of generated statement IDs.
    """
    if not segments:
        return []

    conn = _get_conn()
    stmt_ids = []
    rows = []

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text or len(text) < 3:
            continue

        stmt_id = str(uuid.uuid4())[:12]
        stmt_ids.append(stmt_id)
        rows.append((
            stmt_id, case_id, session_id, video_id,
            seg.get("speaker_id", "SPEAKER_0"),
            seg.get("role", "unknown"),
            round(seg.get("start_time", 0.0), 2),
            text,
            datetime.now().isoformat()
        ))

    if rows:
        conn.executemany(
            """INSERT INTO testimony_memory
               (id, case_id, session_id, video_id, speaker_id, role, timestamp, text, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows
        )
        conn.commit()
        logger.info("Stored %d testimony statements for case %s, video %s",
                     len(rows), case_id, video_id)

    conn.close()
    return stmt_ids


# ── Testimony Retrieval ──────────────────────────────────────────────────────

def get_statements_by_speaker(case_id: str, speaker_id: str,
                              exclude_video_id: str = None,
                              limit: int = 50) -> list[dict]:
    """
    Retrieve all prior statements by a speaker across all sessions in a case.

    Args:
        case_id: The case identifier.
        speaker_id: The speaker identifier.
        exclude_video_id: Optional — exclude statements from this video
            (useful when checking a new video's statements against prior ones).
        limit: Maximum number of statements to return.
    """
    conn = _get_conn()
    query = """SELECT id, session_id, video_id, speaker_id, role,
                      timestamp, text, created_at
               FROM testimony_memory
               WHERE case_id = ? AND speaker_id = ?"""
    params: list = [case_id, speaker_id]

    if exclude_video_id:
        query += " AND video_id != ?"
        params.append(exclude_video_id)

    query += " ORDER BY timestamp LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_statements(case_id: str, limit: int = 200) -> list[dict]:
    """Retrieve all testimony statements for a case."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, session_id, video_id, speaker_id, role,
                  timestamp, text, created_at
           FROM testimony_memory
           WHERE case_id = ?
           ORDER BY timestamp LIMIT ?""",
        (case_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_case_id_for_video(video_id: str) -> str | None:
    """Look up which case a video belongs to."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT DISTINCT case_id FROM testimony_memory WHERE video_id = ?",
        (video_id,)
    ).fetchone()
    conn.close()
    return row["case_id"] if row else None


# ── Contradiction Storage ────────────────────────────────────────────────────

def store_contradiction(case_id: str, speaker_id: str,
                        stmt_a_id: str, stmt_a_text: str,
                        stmt_a_timestamp: float, stmt_a_video_id: str,
                        stmt_b_id: str, stmt_b_text: str,
                        stmt_b_timestamp: float, stmt_b_video_id: str,
                        confidence: float, explanation: str = "") -> str:
    """Store a detected contradiction between two statements."""
    contradiction_id = str(uuid.uuid4())[:12]
    conn = _get_conn()
    conn.execute(
        """INSERT INTO contradictions
           (id, case_id, speaker_id,
            stmt_a_id, stmt_a_text, stmt_a_timestamp, stmt_a_video_id,
            stmt_b_id, stmt_b_text, stmt_b_timestamp, stmt_b_video_id,
            confidence, explanation, detected_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (contradiction_id, case_id, speaker_id,
         stmt_a_id, stmt_a_text, stmt_a_timestamp, stmt_a_video_id,
         stmt_b_id, stmt_b_text, stmt_b_timestamp, stmt_b_video_id,
         round(confidence, 3), explanation, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    logger.info(
        "Stored contradiction %s for speaker %s in case %s (confidence=%.2f)",
        contradiction_id, speaker_id, case_id, confidence
    )
    return contradiction_id


def get_contradictions(case_id: str, speaker_id: str = None,
                       min_confidence: float = 0.0,
                       limit: int = 50) -> list[dict]:
    """Retrieve flagged contradictions for a case."""
    conn = _get_conn()
    query = "SELECT * FROM contradictions WHERE case_id = ?"
    params: list = [case_id]

    if speaker_id:
        query += " AND speaker_id = ?"
        params.append(speaker_id)

    if min_confidence > 0:
        query += " AND confidence >= ?"
        params.append(min_confidence)

    query += " ORDER BY confidence DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_case_data(case_id: str):
    """Delete all testimony and contradiction data for a case."""
    conn = _get_conn()
    conn.execute("DELETE FROM testimony_memory WHERE case_id = ?", (case_id,))
    conn.execute("DELETE FROM contradictions WHERE case_id = ?", (case_id,))
    conn.commit()
    conn.close()
    logger.info("Deleted all testimony data for case %s", case_id)


def delete_video_testimony(video_id: str):
    """Delete testimony data for a specific video."""
    conn = _get_conn()
    conn.execute("DELETE FROM testimony_memory WHERE video_id = ?", (video_id,))
    conn.commit()
    conn.close()
    logger.info("Deleted testimony data for video %s", video_id)
