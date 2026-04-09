"""
VideoRAG — Chat Router
POST /chat — Query the AI agent about video content (Strategy A: text-to-video).
POST /chat/image — Upload a reference image for visual search (Strategy B: image-to-video).
"""
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.config import DATA_DIR
from backend.models import ChatRequest, ChatResponse, VideoClip, SearchResult
from backend.agent.agent import run_agent

logger = logging.getLogger(__name__)
router = APIRouter()


def _build_response(result: dict) -> ChatResponse:
    """Convert raw agent result dict to a ChatResponse model."""
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
    """Strategy A: Text-to-Video search.
    Takes a natural language query, searches the caption & image indexes,
    and returns top 2 matches with detailed descriptions.
    """
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
    """Strategy B: Image-to-Video search.
    Upload a reference image (e.g. mugshot, photo of a suspect/object) and
    search for visually similar frames across indexed videos using CLIP
    semantic similarity — NOT pixel-level hash matching.

    This works because CLIP understands visual concepts: a mugshot of a person
    will match CCTV frames showing the same person, even in different poses,
    lighting, or camera angles.
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

    logger.info(f"Image-to-video search: '{query}' with image {image_path}")

    try:
        # Use CLIP-based semantic search directly via the agent.
        # run_agent() will call search_by_image (CLIP embedding of the uploaded
        # photo → cosine similarity against all indexed frame embeddings).
        result = run_agent(
            query=query,
            video_id=video_id,
            image_path=image_path,
        )

        return _build_response(result)

    except Exception as e:
        logger.error(f"Image-to-video search failed: {e}", exc_info=True)
        return ChatResponse(
            answer="⚠️ Image search failed. Please try again.",
            clips=[],
            sources=[],
        )
