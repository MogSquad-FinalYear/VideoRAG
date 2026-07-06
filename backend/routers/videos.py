"""
VideoRAG — Videos Router
GET /videos — List all uploaded videos with metadata and status.
"""
import json
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from backend.config import METADATA_DIR, VIDEOS_DIR, FRAMES_DIR
from backend.models import VideoInfo


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/videos", response_model=list[VideoInfo])
async def list_videos():
    """List all uploaded and processed videos."""
    videos = []

    for meta_file in sorted(METADATA_DIR.glob("*.json")):
        try:
            with open(meta_file) as f:
                meta = json.load(f)

            video_id = meta.get("video_id", meta_file.stem)

            # Check if video file exists
            video_exists = any(VIDEOS_DIR.glob(f"{video_id}.*"))

            # Get thumbnail (first frame)
            frames_dir = FRAMES_DIR / video_id
            thumbnail = None
            if frames_dir.exists():
                first_frame = sorted(frames_dir.glob("frame_*.jpg"))
                if first_frame:
                    thumbnail = f"/frames/{video_id}/{first_frame[0].name}"

            videos.append(VideoInfo(
                video_id=video_id,
                filename=meta.get("filename", "unknown"),
                duration=meta.get("duration"),
                resolution=meta.get("resolution"),
                fps=meta.get("fps"),
                frame_count=meta.get("frame_count"),
                status="completed" if video_exists else "missing",
                uploaded_at=meta.get("uploaded_at"),
                thumbnail=thumbnail,
                captions_ready=meta.get("captions_ready", False),
            ))
        except Exception as e:
            logger.warning(f"Error reading metadata {meta_file}: {e}")
            continue

    return videos

@router.get("/videos/{video_id}/play")
async def play_video(video_id: str):
    """Redirect to the static files mount to natively support byte-range requests for video players."""
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
    for ext in extensions:
        if (VIDEOS_DIR / f"{video_id}{ext}").exists():
            return RedirectResponse(url=f"/video-files/{video_id}{ext}")
            
    raise HTTPException(status_code=404, detail=f"Video file for {video_id} not found.")
