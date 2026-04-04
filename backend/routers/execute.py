"""
VideoRAG — Execute Router
POST /execute — Upload and process a video file.
"""
import uuid
import threading
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.config import VIDEOS_DIR, FRAME_SAMPLE_FPS
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


def _process_video(task_id: str, video_path: str, video_id: str):
    """Full video processing pipeline — runs in background thread."""
    try:
        total_steps = 6
        current = 0

        # Step 1: Extract metadata
        _update_task(task_id, message="Extracting video metadata...", progress=current / total_steps)
        metadata = video_processor.get_video_metadata(video_path, video_id)
        current += 1

        # Step 2: Extract frames
        _update_task(task_id, message="Extracting frames from video...", progress=current / total_steps)
        frame_paths = video_processor.extract_frames(video_path, video_id, fps=FRAME_SAMPLE_FPS)
        if not frame_paths:
            _update_task(task_id, status="failed", message="No frames extracted from video.", progress=1.0)
            return
        current += 1

        # Step 3: Extract audio
        _update_task(task_id, message="Extracting audio track...", progress=current / total_steps)
        audio_path = video_processor.extract_audio(video_path, video_id)
        current += 1

        # Step 4: Generate CLIP embeddings
        _update_task(task_id, message=f"Generating visual embeddings for {len(frame_paths)} frames...", progress=current / total_steps)
        embeddings = embedding_service.embed_batch(frame_paths)
        indexing_service.index_frames(video_id, frame_paths, embeddings, fps=FRAME_SAMPLE_FPS)
        current += 1

        # Step 5: Generate BLIP captions
        _update_task(task_id, message=f"Generating captions for {len(frame_paths)} frames...", progress=current / total_steps)
        captions = captioning_service.caption_batch(frame_paths)
        indexing_service.index_captions(video_id, frame_paths, captions, fps=FRAME_SAMPLE_FPS)
        current += 1

        # Step 6: Transcribe audio
        if audio_path:
            _update_task(task_id, message="Transcribing audio...", progress=current / total_steps)
            segments = transcription_service.transcribe_audio(audio_path)
            indexing_service.index_transcripts(video_id, segments)
        else:
            _update_task(task_id, message="No audio track found, skipping transcription.", progress=current / total_steps)
        current += 1

        _update_task(
            task_id,
            status="completed",
            message=f"Processing complete. {len(frame_paths)} frames, {len(captions)} captions indexed.",
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

    # Save video to disk
    video_path = str(VIDEOS_DIR / f"{video_id}{ext}")
    try:
        content = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)
    except Exception as e:
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
