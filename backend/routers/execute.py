"""
VideoRAG — Execute Router
POST /execute — Upload and process a video file.
DELETE /videos/{video_id} — Delete a video and its indexed data.
"""
import uuid
import shutil
import threading
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.config import (
    VIDEOS_DIR,
    FRAMES_DIR,
    AUDIO_DIR,
    METADATA_DIR,
    FRAME_SAMPLE_FPS,
    MAX_FRAMES_PER_VIDEO,
    CAPTION_STRIDE,
    CAPTION_MAX_FRAMES,
    CAPTION_MODE,
)
from backend.models import ExecuteResponse
from backend.services import video_processor, embedding_service, captioning_service, transcription_service, indexing_service

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory task tracking
tasks: dict[str, dict] = {}

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def _update_task(task_id: str, **kwargs):
    """Thread-safe task state update."""
    if task_id in tasks:
        tasks[task_id].update(kwargs)


def _select_caption_frames(frame_paths: list[str]) -> list[str]:
    """Select a sparse subset of frames for caption indexing."""
    caption_step = max(1, CAPTION_STRIDE)
    selected = frame_paths[::caption_step]
    if CAPTION_MAX_FRAMES > 0 and len(selected) > CAPTION_MAX_FRAMES:
        downsample_step = max(1, len(selected) // CAPTION_MAX_FRAMES)
        selected = selected[::downsample_step]
    return selected


def _generate_captions(video_id: str, frame_paths: list[str], valid_frame_paths: list[str], frame_interval_sec: float = 1.0) -> int:
    """Generate and index captions for selected key frames only."""
    try:
        caption_frame_paths = _select_caption_frames(frame_paths)
        if not caption_frame_paths:
            return 0

        logger.info(
            "Generating captions for %s key frames (from %s total) for video %s",
            len(caption_frame_paths),
            len(frame_paths),
            video_id,
        )
        captions = captioning_service.caption_batch(caption_frame_paths)
        caption_map = {
            path: caption
            for path, caption in zip(caption_frame_paths, captions)
            if caption
        }
        aligned_captions = [caption_map.get(path, "") for path in valid_frame_paths]
        indexing_service.index_captions(video_id, valid_frame_paths, aligned_captions, frame_interval_sec=frame_interval_sec)
        return sum(1 for c in aligned_captions if c)
    except Exception as e:
        logger.exception("Caption generation failed for video %s: %s", video_id, e)
        return 0


def _process_video(task_id: str, video_path: str, video_id: str):
    """Full video processing pipeline — runs in background thread."""
    try:
        total_steps = 5 if CAPTION_MODE == "sync" else 4
        current = 0

        # Step 1: Extract metadata
        _update_task(task_id, message="Extracting video metadata...", progress=current / total_steps)
        metadata = video_processor.get_video_metadata(video_path, video_id)
        current += 1

        # Step 2: Extract frames
        _update_task(task_id, message="Extracting frames from video...", progress=current / total_steps)
        frame_paths = video_processor.extract_frames(
            video_path,
            video_id,
            fps=FRAME_SAMPLE_FPS,
            max_frames=MAX_FRAMES_PER_VIDEO,
        )
        if not frame_paths:
            _update_task(task_id, status="failed", message="No frames extracted from video.", progress=1.0)
            return
        current += 1

        # Step 3: Extract audio
        _update_task(task_id, message="Extracting audio track...", progress=current / total_steps)
        audio_path = video_processor.extract_audio(video_path, video_id)
        current += 1

        # Step 4: Generate CLIP embeddings
        # Fix 8: embed_batch now returns (embeddings, valid_paths) — skip failed images
        _update_task(task_id, message=f"Generating visual embeddings for {len(frame_paths)} frames...", progress=current / total_steps)
        embeddings, valid_frame_paths = embedding_service.embed_batch(frame_paths)

        # Compute real time interval between consecutive frames
        video_duration = metadata.get("duration", len(valid_frame_paths))
        frame_interval_sec = video_duration / max(len(valid_frame_paths), 1)

        indexing_service.index_frames(video_id, valid_frame_paths, embeddings, frame_interval_sec=frame_interval_sec)
        current += 1

        caption_count = 0
        if CAPTION_MODE == "sync":
            caption_count = _generate_captions(video_id, frame_paths, valid_frame_paths, frame_interval_sec)
            current += 1
        elif CAPTION_MODE == "async":
            threading.Thread(
                target=_generate_captions,
                args=(video_id, frame_paths, valid_frame_paths, frame_interval_sec),
                daemon=True,
            ).start()

        # Step 5 (or 4): Transcribe audio
        if audio_path:
            _update_task(task_id, message="Transcribing audio...", progress=current / total_steps)
            segments = transcription_service.transcribe_audio(audio_path)
            indexing_service.index_transcripts(video_id, segments)
        else:
            _update_task(task_id, message="No audio track found, skipping transcription.", progress=current / total_steps)
        current += 1

        if CAPTION_MODE == "off":
            caption_state = "captions disabled"
        elif CAPTION_MODE == "async":
            caption_state = "captions indexing in background"
        else:
            caption_state = f"{caption_count} captions indexed"

        _update_task(
            task_id,
            status="completed",
            message=f"Processing complete. {len(valid_frame_paths)} frames, {caption_state}.",
            progress=1.0,
            video_id=video_id,
        )
        logger.info(f"Video {video_id} processing completed successfully.")

    except Exception as e:
        logger.exception(f"Processing failed for video {video_id}")
        _update_task(task_id, status="failed", message=f"Processing failed: {str(e)}", progress=1.0)


@router.post("/execute", response_model=ExecuteResponse)
async def execute_upload(file: UploadFile = File(...)):
    """Upload a video file and start background processing."""
    # Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

    # Generate IDs
    video_id = str(uuid.uuid4())[:12]
    task_id = str(uuid.uuid4())[:12]

    # Fix 9: Stream file to disk in chunks instead of loading entirely into RAM
    video_path = str(VIDEOS_DIR / f"{video_id}{ext}")
    try:
        chunk_size = 1024 * 1024  # 1MB chunks
        with open(video_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        # Clean up partial file
        Path(video_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    # Initialize task state
    tasks[task_id] = {
        "task_id": task_id,
        "video_id": video_id,
        "status": "processing",
        "progress": 0.0,
        "message": "Video uploaded. Starting processing pipeline...",
    }

    # Start background processing
    thread = threading.Thread(target=_process_video, args=(task_id, video_path, video_id), daemon=True)
    thread.start()

    return ExecuteResponse(task_id=task_id, status="processing", message="Video upload received. Processing started.")


# Fix 14: Delete video endpoint
@router.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a video and all its indexed data."""
    deleted_items = []

    # Delete from ChromaDB indexes
    try:
        indexing_service.delete_video_from_indexes(video_id)
        deleted_items.append("indexes")
    except Exception as e:
        logger.error(f"Error deleting indexes for {video_id}: {e}")

    # Delete video file (try all extensions)
    for ext in ALLOWED_EXTENSIONS:
        vpath = VIDEOS_DIR / f"{video_id}{ext}"
        if vpath.exists():
            vpath.unlink()
            deleted_items.append("video_file")
            break

    # Delete frames directory
    frames_dir = FRAMES_DIR / video_id
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
        deleted_items.append("frames")

    # Delete audio file
    audio_file = AUDIO_DIR / f"{video_id}.wav"
    if audio_file.exists():
        audio_file.unlink()
        deleted_items.append("audio")

    # Delete metadata
    meta_file = METADATA_DIR / f"{video_id}.json"
    if meta_file.exists():
        meta_file.unlink()
        deleted_items.append("metadata")

    if not deleted_items:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found.")

    logger.info(f"Deleted video {video_id}: {deleted_items}")
    return {"status": "deleted", "video_id": video_id, "deleted": deleted_items}
