"""
VideoRAG — ChromaDB Indexing Service
Manages 3 vector collections: image (CLIP), caption (BLIP), speech (Whisper).
"""
import logging
from pathlib import Path
import chromadb

from backend.config import CHROMADB_DIR, IMAGE_COLLECTION, CAPTION_COLLECTION, SPEECH_COLLECTION, FRAMES_DIR
from backend.services import embedding_service

logger = logging.getLogger(__name__)

_chroma_client = None
_image_col = None
_caption_col = None
_speech_col = None


def _extract_frame_number(frame_path: str) -> int | None:
    """Extract numeric frame index from frame filename like frame_000123.jpg."""
    try:
        stem = Path(frame_path).stem
        if not stem.startswith("frame_"):
            return None
        return int(stem.split("_")[1])
    except Exception:
        return None


def _to_url_path(absolute_path: str, video_id: str) -> str:
    """Convert an absolute frame path to a relative URL path for the browser.
    e.g. 'C:\\...\\data\\frames\\abc123\\frame_000007.jpg' -> '/frames/abc123/frame_000007.jpg'
    """
    try:
        p = Path(absolute_path)
        filename = p.name
        return f"/frames/{video_id}/{filename}"
    except Exception:
        return absolute_path


def _normalize_frame_path(frame_path: str, video_id: str) -> str:
    """Ensure frame_path is a browser-friendly relative URL, not an absolute filesystem path."""
    if not frame_path:
        return ""
    # Already a relative URL
    if frame_path.startswith("/frames/"):
        return frame_path
    # Absolute path — convert to URL
    return _to_url_path(frame_path, video_id)


def _frame_url_exists(frame_path: str, video_id: str) -> bool:
    """Return True when the frame URL maps to an existing frame file on disk."""
    if not frame_path or not video_id:
        return False
    frame_name = Path(frame_path).name
    if not frame_name:
        return False
    return (FRAMES_DIR / video_id / frame_name).exists()


def get_chroma_client():
    """Initialize ChromaDB persistent client."""
    global _chroma_client, _image_col, _caption_col, _speech_col
    if _chroma_client is not None:
        return _chroma_client

    logger.info(f"Initializing ChromaDB at {CHROMADB_DIR}")
    _chroma_client = chromadb.PersistentClient(path=str(CHROMADB_DIR))

    _image_col = _chroma_client.get_or_create_collection(
        name=IMAGE_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    _caption_col = _chroma_client.get_or_create_collection(
        name=CAPTION_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    _speech_col = _chroma_client.get_or_create_collection(
        name=SPEECH_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    logger.info("ChromaDB collections ready.")
    return _chroma_client


def get_collections():
    """Return the 3 collections, initializing if needed."""
    get_chroma_client()
    return _image_col, _caption_col, _speech_col


def index_frames(video_id: str, frame_paths: list[str], embeddings: list[list[float]], frame_interval_sec: float = 1.0):
    """Index frame CLIP embeddings into the image collection.
    
    Args:
        video_id: The video identifier.
        frame_paths: List of frame file paths.
        embeddings: CLIP embeddings for each frame.
        frame_interval_sec: Actual seconds between consecutive extracted frames.
            E.g., if 48 frames were extracted from a 79s video, this is ~1.65.
    """
    get_chroma_client()
    if len(frame_paths) != len(embeddings):
        raise ValueError(
            f"frame_paths and embeddings must have the same length: {len(frame_paths)} != {len(embeddings)}"
        )
    ids = []
    metadatas = []
    for i, path in enumerate(frame_paths):
        frame_number = _extract_frame_number(path)
        if frame_number is None:
            frame_number = i
        frame_id = f"{video_id}_frame_{frame_number}"
        ids.append(frame_id)
        metadatas.append({
            "video_id": video_id,
            "frame_number": frame_number,
            "timestamp": round(frame_number * frame_interval_sec, 2),
            "frame_path": _to_url_path(path, video_id),  # Store as relative URL
        })

    # ChromaDB has a batch limit, chunk if needed
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        _image_col.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
    logger.info(f"Indexed {len(ids)} frames for video {video_id}")


def index_captions(video_id: str, frame_paths: list[str], captions: list[str], frame_interval_sec: float = 1.0):
    """Index captions into the caption collection using local CLIP text embeddings."""
    get_chroma_client()
    ids = []
    documents = []
    metadatas = []
    for i, (path, caption) in enumerate(zip(frame_paths, captions)):
        if not caption:
            continue
        frame_number = _extract_frame_number(path)
        if frame_number is None:
            frame_number = i
        ids.append(f"{video_id}_caption_{frame_number}")
        documents.append(caption)
        metadatas.append({
            "video_id": video_id,
            "frame_number": frame_number,
            "timestamp": round(frame_number * frame_interval_sec, 2),
            "frame_path": _to_url_path(path, video_id),
        })

    if not ids:
        return

    embeddings = [embedding_service.embed_text(doc) for doc in documents]

    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        _caption_col.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
    logger.info(f"Indexed {len(ids)} captions for video {video_id}")


def index_transcripts(video_id: str, segments: list[dict]):
    """Index transcript segments using local CLIP text embeddings."""
    get_chroma_client()
    ids = []
    documents = []
    metadatas = []
    for i, seg in enumerate(segments):
        if not seg.get("text"):
            continue
        ids.append(f"{video_id}_speech_{i}")
        documents.append(seg["text"])
        metadatas.append({
            "video_id": video_id,
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
        })

    if not ids:
        return

    embeddings = [embedding_service.embed_text(doc) for doc in documents]

    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        _speech_col.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
    logger.info(f"Indexed {len(ids)} transcript segments for video {video_id}")


# Minimum similarity thresholds — results below these are noise, not signal
_IMAGE_MIN_SCORE = 0.12
_CAPTION_MIN_SCORE = 0.15
_SPEECH_MIN_SCORE = 0.10


def search_images(query_embedding: list[float], n: int = 10, video_id: str = None, min_score: float = _IMAGE_MIN_SCORE) -> list[dict]:
    """Search image index by CLIP embedding similarity."""
    get_chroma_client()

    count = _image_col.count()
    if count == 0:
        logger.warning("Image index is empty, nothing to search.")
        return []

    where = {"video_id": video_id} if video_id else None
    requested = max(1, n)
    probe_n = min(max(requested * 8, requested), count)
    try:
        results = _image_col.query(
            query_embeddings=[query_embedding],
            n_results=probe_n,
            where=where,
            include=["metadatas", "distances"]
        )
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        return []

    hits = []
    if results and results["ids"] and results["ids"][0]:
        for i, id_ in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results["distances"] else 0
            score = round(1 - dist, 4)  # cosine distance to similarity
            if score < min_score:
                continue  # Skip noise results
            vid = meta.get("video_id", "")
            frame_path = _normalize_frame_path(meta.get("frame_path", ""), vid)
            if not _frame_url_exists(frame_path, vid):
                continue
            hits.append({
                "id": id_,
                "score": score,
                "video_id": vid,
                "frame_number": meta.get("frame_number"),
                "timestamp": meta.get("timestamp"),
                "frame_path": frame_path,
                "source_index": "image",
            })
            if len(hits) >= requested:
                break
    return hits


def search_captions(query_text: str, n: int = 10, video_id: str = None) -> list[dict]:
    """Search caption index by CLIP text-embedding similarity."""
    get_chroma_client()

    count = _caption_col.count()
    if count == 0:
        logger.warning("Caption index is empty, nothing to search.")
        return []

    where = {"video_id": video_id} if video_id else None
    requested = max(1, n)
    probe_n = min(max(requested * 8, requested), count)
    query_embedding = embedding_service.embed_text(query_text)
    try:
        results = _caption_col.query(
            query_embeddings=[query_embedding],
            n_results=probe_n,
            where=where,
            include=["metadatas", "documents", "distances"]
        )
    except Exception as e:
        logger.error(f"Caption search failed: {e}")
        return []

    hits = []
    if results and results["ids"] and results["ids"][0]:
        for i, id_ in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            doc = results["documents"][0][i] if results["documents"] else ""
            dist = results["distances"][0][i] if results["distances"] else 0
            score = round(1 - dist, 4)  # cosine distance → similarity
            if score < _CAPTION_MIN_SCORE:
                continue  # Skip noise results
            vid = meta.get("video_id", "")
            frame_path = _normalize_frame_path(meta.get("frame_path", ""), vid)
            if not _frame_url_exists(frame_path, vid):
                continue
            hits.append({
                "id": id_,
                "score": score,
                "video_id": vid,
                "frame_number": meta.get("frame_number"),
                "timestamp": meta.get("timestamp"),
                "frame_path": frame_path,
                "content": doc,
                "source_index": "caption",
            })
            if len(hits) >= requested:
                break
    return hits


def search_captions_by_keyword(keyword: str, n: int = 10, video_id: str = None) -> list[dict]:
    """Search caption index by keyword substring match in document text.

    This supplements embedding search — when a user asks about 'car' we also
    find captions that literally contain the word 'car'.
    """
    get_chroma_client()

    count = _caption_col.count()
    if count == 0:
        return []

    where = {"video_id": video_id} if video_id else None
    try:
        results = _caption_col.get(
            where=where,
            include=["metadatas", "documents"],
        )
    except Exception as e:
        logger.error(f"Caption keyword search failed: {e}")
        return []

    if not results or not results["ids"]:
        return []

    keyword_lower = keyword.lower()
    stop_words = {
        "show", "find", "where", "when", "what", "which", "that", "this",
        "with", "from", "into", "onto", "about", "there", "their", "have",
        "give", "clip", "video", "scene", "frames", "frame", "please"
    }
    search_terms = [
        t.strip()
        for t in keyword_lower.replace(",", " ").replace(".", " ").split()
        if len(t.strip()) > 2 and t.strip() not in stop_words
    ]
    if not search_terms:
        return []

    hits = []
    for i, id_ in enumerate(results["ids"]):
        doc = results["documents"][i] if results["documents"] else ""
        meta = results["metadatas"][i] if results["metadatas"] else {}
        doc_lower = doc.lower()
        # Check if any search term appears in the caption.
        matching_terms = [t for t in search_terms if t in doc_lower]
        if not matching_terms:
            continue
        vid = meta.get("video_id", "")
        frame_path = _normalize_frame_path(meta.get("frame_path", ""), vid)
        if not _frame_url_exists(frame_path, vid):
            continue
        # Score based on term coverage, with bonus for event/action words.
        event_terms = {"hit", "hits", "hitting", "crash", "collision", "fall", "fighting", "chasing"}
        event_bonus = 0.1 if any(t in event_terms for t in matching_terms) else 0.0
        match_score = round(0.45 + 0.14 * len(matching_terms) + event_bonus, 4)
        hits.append({
            "id": id_,
            "score": min(match_score, 0.95),
            "video_id": vid,
            "frame_number": meta.get("frame_number"),
            "timestamp": meta.get("timestamp"),
            "frame_path": frame_path,
            "content": doc,
            "source_index": "caption",
        })

    # Sort by score descending and limit
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:n]


def search_transcripts(query_text: str, n: int = 10, video_id: str = None) -> list[dict]:
    """Search speech index by CLIP text-embedding similarity."""
    get_chroma_client()

    count = _speech_col.count()
    if count == 0:
        logger.warning("Speech index is empty, nothing to search.")
        return []

    where = {"video_id": video_id} if video_id else None
    query_embedding = embedding_service.embed_text(query_text)
    try:
        results = _speech_col.query(
            query_embeddings=[query_embedding],
            n_results=min(n, count),
            where=where,
            include=["metadatas", "documents", "distances"]
        )
    except Exception as e:
        logger.error(f"Transcript search failed: {e}")
        return []

    hits = []
    if results and results["ids"] and results["ids"][0]:
        for i, id_ in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            doc = results["documents"][0][i] if results["documents"] else ""
            dist = results["distances"][0][i] if results["distances"] else 0
            score = round(1 - dist, 4)  # cosine distance → similarity
            if score < _SPEECH_MIN_SCORE:
                continue  # Skip noise results
            hits.append({
                "id": id_,
                "score": score,
                "video_id": meta.get("video_id", ""),
                "start_time": meta.get("start_time"),
                "end_time": meta.get("end_time"),
                "content": doc,
                "source_index": "speech",
            })
    return hits


def get_index_stats() -> dict:
    """Return count of documents in each collection."""
    get_chroma_client()
    return {
        "image_index": _image_col.count(),
        "caption_index": _caption_col.count(),
        "speech_index": _speech_col.count(),
    }


def delete_video_from_indexes(video_id: str):
    """Remove all indexed data for a specific video."""
    get_chroma_client()
    for col in [_image_col, _caption_col, _speech_col]:
        try:
            existing = col.get(where={"video_id": video_id})
            if existing and existing["ids"]:
                col.delete(ids=existing["ids"])
        except Exception as e:
            logger.warning(f"Error deleting video {video_id} from {col.name}: {e}")


def get_caption_for_frame(video_id: str, frame_number: int) -> str:
    """Retrieve stored caption for a specific frame from the caption index.

    Used to enrich image search results (which don't have text) with
    the caption that was generated during video processing.
    """
    get_chroma_client()
    doc_id = f"{video_id}_caption_{frame_number}"
    try:
        result = _caption_col.get(ids=[doc_id], include=["documents"])
        if result and result["documents"] and result["documents"][0]:
            return result["documents"][0]
    except Exception:
        pass
    return ""

