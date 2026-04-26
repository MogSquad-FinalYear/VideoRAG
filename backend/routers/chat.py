"""
VideoRAG — Chat Router
POST /chat — Query the AI agent about video content.
POST /chat/image — Upload a reference image for visual search.
"""
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.config import DATA_DIR, GROQ_API_KEY
from backend.models import ChatRequest, ChatResponse, VideoClip, SearchResult
from backend.agent.agent import run_agent

logger = logging.getLogger(__name__)
router = APIRouter()


def _build_response(result: dict) -> ChatResponse:
    """Convert agent result dict to ChatResponse model."""
    clips = [VideoClip(**c) for c in result.get("clips", [])]
    sources = []
    for s in result.get("sources", []):
        sources.append(SearchResult(
            video_id=s.get("video_id", ""),
            frame_number=s.get("frame_number"),
            timestamp=s.get("timestamp"),
            start_time=s.get("start_time"),
            end_time=s.get("end_time"),
            score=s.get("score", 0),
            content=s.get("content", ""),
            frame_path=s.get("frame_path", ""),
            source_index=s.get("source_index", ""),
        ))
    return ChatResponse(
        answer=result["answer"],
        clips=clips,
        sources=sources,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a natural language query to the AI agent."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"Chat query: '{request.query}' (video_id={request.video_id})")

    result = run_agent(
        query=request.query,
        video_id=request.video_id,
        image_path=request.image_path,
    )
    return _build_response(result)


@router.post("/chat/image", response_model=ChatResponse)
async def chat_with_image(
    query: str = Form(default="Find frames similar to this image"),
    video_id: str = Form(default=None),
    image: UploadFile = File(...),
):
    """Upload a reference image and search for visually similar frames.

    Uses CLIP semantic embeddings to find frames where the same person, object,
    or scene appears — even if the uploaded image is NOT a frame from the video.
    This is the key difference from pixel-level hash matching.
    """
    # Save uploaded image to disk
    upload_dir = DATA_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    image_id = str(uuid.uuid4())[:8]
    ext = Path(image.filename).suffix or ".jpg"
    image_path = str(upload_dir / f"{image_id}{ext}")

    content = await image.read()
    with open(image_path, "wb") as f:
        f.write(content)

    logger.info(f"Image search: '{query}' with image {image_path} (video_id={video_id})")

    # Use CLIP-based semantic search via the agent pipeline
    # This finds visually similar frames (same person, same object, etc.)
    # even when the uploaded image is NOT an exact frame from the video
    result = run_agent(
        query=query,
        video_id=video_id,
        image_path=image_path,
    )
    return _build_response(result)
