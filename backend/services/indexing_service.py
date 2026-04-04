"""
VideoRAG — ChromaDB Indexing Service
Manages 3 vector collections: image (CLIP), caption (BLIP), speech (Whisper).
"""
import logging
import os
from pathlib import Path
import chromadb
from chromadb.config import Settings

from backend.config import CHROMADB_DIR, IMAGE_COLLECTION, CAPTION_COLLECTION, SPEECH_COLLECTION, FRAMES_DIR

logger = logging.getLogger(__name__)

_chroma_client = None
_image_col = None
_caption_col = None
_speech_col = None


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
    )
    _speech_col = _chroma_client.get_or_create_collection(
        name=SPEECH_COLLECTION,
    )

    logger.info("ChromaDB collections ready.")
    return _chroma_client


def get_collections():
    """Return the 3 collections, initializing if needed."""
    get_chroma_client()
    return _image_col, _caption_col, _speech_col


def index_frames(video_id: str, frame_paths: list[str], embeddings: list[list[float]], fps: float = 1.0):
    """Index frame CLIP embeddings into the image collection."""
    get_chroma_client()
    ids = []
    metadatas = []
    for i, path in enumerate(frame_paths):
        frame_id = f"{video_id}_frame_{i}"
        ids.append(frame_id)
        metadatas.append({
            "video_id": video_id,
            "frame_number": i,
            "timestamp": round(i / fps, 2),
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


def index_captions(video_id: str, frame_paths: list[str], captions: list[str], fps: float = 1.0):
    """Index BLIP captions into the caption collection."""
    get_chroma_client()
    ids = []
    documents = []
    metadatas = []
    for i, (path, caption) in enumerate(zip(frame_paths, captions)):
        if not caption:
            continue
        ids.append(f"{video_id}_caption_{i}")
        documents.append(caption)
        metadatas.append({
            "video_id": video_id,
            "frame_number": i,
            "timestamp": round(i / fps, 2),
            "frame_path": _to_url_path(path, video_id),  # Store as relative URL
        })

    if not ids:
        return

    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        _caption_col.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
    logger.info(f"Indexed {len(ids)} captions for video {video_id}")


def index_transcripts(video_id: str, segments: list[dict]):
    """Index Whisper transcript segments into the speech collection."""
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

    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        _speech_col.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
    logger.info(f"Indexed {len(ids)} transcript segments for video {video_id}")


def search_images(query_embedding: list[float], n: int = 10, video_id: str = None) -> list[dict]:
    """Search image index by CLIP embedding similarity."""
    get_chroma_client()

    # Fix 5: Don't query empty collections
    count = _image_col.count()
    if count == 0:
        logger.warning("Image index is empty, nothing to search.")
        return []

    where = {"video_id": video_id} if video_id else None
    try:
        results = _image_col.query(
            query_embeddings=[query_embedding],
            n_results=min(n, count),
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
            vid = meta.get("video_id", "")
            hits.append({
                "id": id_,
                "score": round(1 - dist, 4),  # cosine distance to similarity
                "video_id": vid,
                "frame_number": meta.get("frame_number"),
                "timestamp": meta.get("timestamp"),
                "frame_path": _normalize_frame_path(meta.get("frame_path", ""), vid),
                "source_index": "image",
            })
    return hits


def search_captions(query_text: str, n: int = 10, video_id: str = None) -> list[dict]:
    """Search caption index by text similarity."""
    get_chroma_client()

    # Fix 5: Don't query empty collections
    count = _caption_col.count()
    if count == 0:
        logger.warning("Caption index is empty, nothing to search.")
        return []

    where = {"video_id": video_id} if video_id else None
    try:
        results = _caption_col.query(
            query_texts=[query_text],
            n_results=min(n, count),
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
            vid = meta.get("video_id", "")
            hits.append({
                "id": id_,
                "score": round(1 - dist, 4) if dist < 2 else round(dist, 4),
                "video_id": vid,
                "frame_number": meta.get("frame_number"),
                "timestamp": meta.get("timestamp"),
                "frame_path": _normalize_frame_path(meta.get("frame_path", ""), vid),
                "content": doc,
                "source_index": "caption",
            })
    return hits


def search_transcripts(query_text: str, n: int = 10, video_id: str = None) -> list[dict]:
    """Search speech index by text similarity."""
    get_chroma_client()

    # Fix 5: Don't query empty collections
    count = _speech_col.count()
    if count == 0:
        logger.warning("Speech index is empty, nothing to search.")
        return []

    where = {"video_id": video_id} if video_id else None
    try:
        results = _speech_col.query(
            query_texts=[query_text],
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
            hits.append({
                "id": id_,
                "score": round(1 - dist, 4) if dist < 2 else round(dist, 4),
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
