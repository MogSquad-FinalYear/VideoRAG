"""
VideoRAG — Citation Log Database (Phase 8)
SQLite storage for citation verification results and tamper-evident clip hashes.
Provides an audit trail for every citation made in generated answers.
"""
import sqlite3
import uuid
import logging
from datetime import datetime

from backend.config import CITATION_DB_PATH

logger = logging.getLogger(__name__)

_db_initialized = False


def _get_conn() -> sqlite3.Connection:
    """Get SQLite connection with citation log tables."""
    global _db_initialized
    conn = sqlite3.connect(str(CITATION_DB_PATH))
    conn.row_factory = sqlite3.Row

    if not _db_initialized:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS citation_log (
                id TEXT PRIMARY KEY,
                answer_id TEXT NOT NULL,
                cited_timestamp REAL,
                cited_video TEXT NOT NULL,
                claim_text TEXT NOT NULL,
                verification_score REAL DEFAULT 0.0,
                verification_result TEXT DEFAULT 'unverified',
                clip_hash TEXT DEFAULT '',
                verified_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_citation_answer
                ON citation_log(answer_id);
            CREATE INDEX IF NOT EXISTS idx_citation_video
                ON citation_log(cited_video);
        """)
        _db_initialized = True
        logger.info("Citation log table initialized at %s", CITATION_DB_PATH)

    return conn


def store_citation(answer_id: str, cited_timestamp: float, cited_video: str,
                   claim_text: str, verification_score: float,
                   verification_result: str, clip_hash: str = "") -> str:
    """
    Store a citation verification result.

    Args:
        answer_id: Unique identifier for the generated answer.
        cited_timestamp: The timestamp cited in the answer.
        cited_video: The video ID cited.
        claim_text: The specific claim being cited.
        verification_score: 0.0-1.0 confidence that citation supports claim.
        verification_result: "verified", "uncertain", "unsupported".
        clip_hash: SHA-256 hash of the cited clip bytes for tamper-evidence.

    Returns:
        The generated citation ID.
    """
    citation_id = str(uuid.uuid4())[:12]
    conn = _get_conn()
    conn.execute(
        """INSERT INTO citation_log
           (id, answer_id, cited_timestamp, cited_video, claim_text,
            verification_score, verification_result, clip_hash, verified_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (citation_id, answer_id, round(cited_timestamp, 2), cited_video,
         claim_text, round(verification_score, 3), verification_result,
         clip_hash, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    logger.info("Stored citation %s (score=%.2f, result=%s)",
                citation_id, verification_score, verification_result)
    return citation_id


def get_citations_for_answer(answer_id: str) -> list[dict]:
    """Retrieve all citations for a specific answer."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM citation_log WHERE answer_id = ? ORDER BY cited_timestamp",
        (answer_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_citations_for_video(video_id: str, limit: int = 100) -> list[dict]:
    """Retrieve all citations for a specific video."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM citation_log WHERE cited_video = ? ORDER BY verified_at DESC LIMIT ?",
        (video_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
