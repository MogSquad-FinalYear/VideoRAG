"""
VideoRAG — Chat Router
POST /chat — Query the AI agent about video content.
POST /chat/image — Upload a reference image for visual search.
"""
import os
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.config import DATA_DIR
from backend.models import ChatRequest, ChatResponse, VideoClip, SearchResult
from backend.agent.agent import run_agent

logger = logging.getLogger(__name__)
router = APIRouter()

# Simple conversation history per session (in-memory)
_conversations: dict[str, list] = {}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a natural language query to the AI agent."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Fix 6: Validate API key before calling agent
    from backend.config import GROQ_API_KEY
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured. Please add it to your .env file."
        )

    logger.info(f"Chat query: '{request.query}' (video_id={request.video_id})")

    result = run_agent(
        query=request.query,
        video_id=request.video_id,
        image_path=request.image_path,
    )

    # Convert to response model
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


@router.post("/chat/image", response_model=ChatResponse)
async def chat_with_image(
    query: str = Form(default="Find frames similar to this image"),
    video_id: str = Form(default=None),
    image: UploadFile = File(...),
):
    """Upload a reference image and search for visually similar frames."""
    # Save uploaded image
    upload_dir = DATA_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    image_id = str(uuid.uuid4())[:8]
    ext = Path(image.filename).suffix or ".jpg"
    image_path = str(upload_dir / f"{image_id}{ext}")

    content = await image.read()
    with open(image_path, "wb") as f:
        f.write(content)

    logger.info(f"Image-to-video search: '{query}' with image {image_path}")

    result = run_agent(
        query=query,
        video_id=video_id,
        image_path=image_path,
    )

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
