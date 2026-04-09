"""
Re-index all captions for existing processed video frames.
Run this once to fix sparse caption indexes.
"""
import sys
sys.path.insert(0, r'd:\VideoRAG')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

from pathlib import Path
from backend.config import FRAMES_DIR, FRAME_SAMPLE_FPS, CAPTION_COLLECTION
from backend.services import captioning_service, indexing_service, embedding_service
import chromadb
from backend.config import CHROMADB_DIR

logger = logging.getLogger("reindex_captions")

def main():
    # Find all video frame directories
    if not FRAMES_DIR.exists():
        logger.error(f"FRAMES_DIR {FRAMES_DIR} does not exist")
        return

    video_dirs = [d for d in FRAMES_DIR.iterdir() if d.is_dir()]
    logger.info(f"Found {len(video_dirs)} video directories in {FRAMES_DIR}")

    # Delete old caption collection and recreate with cosine space
    client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
    try:
        client.delete_collection(CAPTION_COLLECTION)
        logger.info(f"Deleted old caption collection: {CAPTION_COLLECTION}")
    except Exception:
        logger.info(f"No existing collection to delete: {CAPTION_COLLECTION}")

    # Force re-init of indexing service
    import backend.services.indexing_service as idx_svc
    idx_svc._chroma_client = None
    idx_svc._caption_col = None
    idx_svc.get_chroma_client()

    for video_dir in video_dirs:
        video_id = video_dir.name
        frame_files = sorted(video_dir.glob("frame_*.jpg"))

        if not frame_files:
            logger.warning(f"  No frames found for video {video_id}, skipping")
            continue

        frame_paths = [str(f) for f in frame_files]
        logger.info(f"  Video {video_id}: {len(frame_paths)} frames, captioning ALL...")

        # Generate captions for ALL frames (not just stride-sampled)
        captions = captioning_service.caption_batch(frame_paths)
        valid_count = sum(1 for c in captions if c)
        logger.info(f"  Generated {valid_count}/{len(frame_paths)} captions")

        # Index captions
        indexing_service.index_captions(video_id, frame_paths, captions, fps=FRAME_SAMPLE_FPS)
        logger.info(f"  Indexed captions for video {video_id}")

    # Verify
    stats = indexing_service.get_index_stats()
    logger.info(f"Final index stats: {stats}")
    print(f"\n✅ Caption re-indexing complete. Stats: {stats}")


if __name__ == "__main__":
    main()
