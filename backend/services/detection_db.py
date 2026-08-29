"""
VideoRAG — Detection & Scene Graph Database (Phase 4-5)
SQLite storage for structured detection data and scene graph relations.
Not vector-searched — queried by filter/count/join.
"""
import sqlite3
import json
import logging
from pathlib import Path

from backend.config import DETECTION_DB_PATH

logger = logging.getLogger(__name__)

_db_initialized = False


def _get_conn() -> sqlite3.Connection:
    """Get a SQLite connection, creating tables if needed."""
    global _db_initialized
    conn = sqlite3.connect(str(DETECTION_DB_PATH))
    conn.row_factory = sqlite3.Row

    if not _db_initialized:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                object_class TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                w REAL NOT NULL,
                h REAL NOT NULL,
                confidence REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_det_video ON detections(video_id);
            CREATE INDEX IF NOT EXISTS idx_det_class ON detections(object_class);
            CREATE INDEX IF NOT EXISTS idx_det_ts ON detections(video_id, timestamp);

            CREATE TABLE IF NOT EXISTS scene_graph (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                subject TEXT NOT NULL,
                relation TEXT NOT NULL,
                object TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sg_video ON scene_graph(video_id);
            CREATE INDEX IF NOT EXISTS idx_sg_ts ON scene_graph(video_id, timestamp);
        """)
        _db_initialized = True
        logger.info(f"Detection DB initialized at {DETECTION_DB_PATH}")

    return conn


def store_detections(video_id: str, frame_number: int, timestamp: float,
                     detections: list[dict]):
    """Store YOLO detections for a single frame."""
    if not detections:
        return
    conn = _get_conn()
    rows = []
    for det in detections:
        x, y, w, h = det["bbox"]
        rows.append((
            video_id, frame_number, timestamp,
            det["class_name"], x, y, w, h, det["confidence"]
        ))
    conn.executemany(
        "INSERT INTO detections (video_id, frame_number, timestamp, object_class, x, y, w, h, confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows
    )
    conn.commit()
    conn.close()


def store_detections_batch(video_id: str, frame_paths: list[str],
                           det_results: dict, timestamps: list[float] = None):
    """Store detections for a batch of frames."""
    from backend.services.indexing_service import _extract_frame_number
    conn = _get_conn()
    rows = []
    for i, path in enumerate(frame_paths):
        dets = det_results.get(path, [])
        if not dets:
            continue
        frame_number = _extract_frame_number(path)
        if frame_number is None:
            frame_number = i
        ts = timestamps[i] if timestamps and i < len(timestamps) else float(frame_number)
        for det in dets:
            x, y, w, h = det["bbox"]
            rows.append((
                video_id, frame_number, round(ts, 2),
                det["class_name"], x, y, w, h, det["confidence"]
            ))
    if rows:
        conn.executemany(
            "INSERT INTO detections (video_id, frame_number, timestamp, object_class, x, y, w, h, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows
        )
        conn.commit()
        logger.info(f"Stored {len(rows)} detections for video {video_id}")
    conn.close()


def count_objects(video_id: str, class_name: str = None,
                  start_time: float = None, end_time: float = None) -> dict:
    """
    Count objects per class across frames.

    Returns:
        {"counts": {class_name: max_count_in_any_frame}, "frames_with_object": N}
    """
    conn = _get_conn()
    query = "SELECT frame_number, timestamp, object_class, COUNT(*) as cnt FROM detections WHERE video_id = ?"
    params = [video_id]

    if class_name:
        query += " AND LOWER(object_class) = LOWER(?)"
        params.append(class_name)
    if start_time is not None:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time is not None:
        query += " AND timestamp <= ?"
        params.append(end_time)

    query += " GROUP BY frame_number, object_class ORDER BY timestamp"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    # Find max count per class across all frames
    class_max = {}
    frames_with = set()
    for row in rows:
        cls = row["object_class"]
        cnt = row["cnt"]
        ts = row["timestamp"]
        class_max[cls] = max(class_max.get(cls, 0), cnt)
        frames_with.add(row["frame_number"])

    return {
        "video_id": video_id,
        "counts": class_max,
        "frames_with_object": len(frames_with),
        "time_range": f"{start_time or 'start'} - {end_time or 'end'}",
    }


def locate_objects(video_id: str, class_name: str,
                   start_time: float = None, end_time: float = None,
                   limit: int = 10) -> list[dict]:
    """
    Find locations of a specific object class across frames.

    Returns list of frames with object locations and spatial positions.
    """
    conn = _get_conn()
    query = """SELECT frame_number, timestamp, object_class, x, y, w, h, confidence
               FROM detections WHERE video_id = ? AND LOWER(object_class) = LOWER(?)"""
    params = [video_id, class_name]

    if start_time is not None:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time is not None:
        query += " AND timestamp <= ?"
        params.append(end_time)

    query += " ORDER BY confidence DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    results = []
    for row in rows:
        cx = row["x"] + row["w"] / 2
        cy = row["y"] + row["h"] / 2
        results.append({
            "frame_number": row["frame_number"],
            "timestamp": row["timestamp"],
            "class_name": row["object_class"],
            "bbox": [row["x"], row["y"], row["w"], row["h"]],
            "center": [round(cx, 1), round(cy, 1)],
            "confidence": row["confidence"],
        })
    return results


def get_detections_for_frame(video_id: str, frame_number: int) -> list[dict]:
    """Get all detections for a specific frame."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM detections WHERE video_id = ? AND frame_number = ?",
        (video_id, frame_number)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Scene Graph (Phase 5) ───────────────────────────────────────────────────

def store_relations(video_id: str, frame_number: int, timestamp: float,
                    triples: list[tuple]):
    """Store scene graph relation triples for a frame.

    Args:
        triples: List of (subject, relation, object) tuples.
    """
    if not triples:
        return
    conn = _get_conn()
    rows = [(video_id, frame_number, timestamp, s, r, o) for s, r, o in triples]
    conn.executemany(
        "INSERT INTO scene_graph (video_id, frame_number, timestamp, subject, relation, object) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows
    )
    conn.commit()
    conn.close()


def get_relations(video_id: str, start_time: float = None,
                  end_time: float = None, limit: int = 50) -> list[dict]:
    """Query scene graph relations for a video within a time range."""
    conn = _get_conn()
    query = "SELECT * FROM scene_graph WHERE video_id = ?"
    params = [video_id]

    if start_time is not None:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time is not None:
        query += " AND timestamp <= ?"
        params.append(end_time)

    query += " ORDER BY timestamp LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "frame_number": row["frame_number"],
            "timestamp": row["timestamp"],
            "subject": row["subject"],
            "relation": row["relation"],
            "object": row["object"],
            "triple": f"{row['subject']} {row['relation']} {row['object']}",
        })
    return results


def delete_video_data(video_id: str):
    """Delete all detection and scene graph data for a video."""
    conn = _get_conn()
    conn.execute("DELETE FROM detections WHERE video_id = ?", (video_id,))
    conn.execute("DELETE FROM scene_graph WHERE video_id = ?", (video_id,))
    conn.commit()
    conn.close()
    logger.info(f"Deleted detection/scene-graph data for video {video_id}")
