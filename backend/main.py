"""
VideoRAG — FastAPI Application Entry Point
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import FRAMES_DIR, DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="VideoRAG",
    description="Multimodal Video Retrieval-Augmented Generation for forensic evidence analysis",
    version="1.0.0",
)

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving extracted frames
app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")

# Also serve uploaded videos
videos_dir = DATA_DIR / "videos"
app.mount("/video-files", StaticFiles(directory=str(videos_dir)), name="video-files")

# Register routers
from backend.routers import execute, status, chat, videos, speakers, testimony

app.include_router(execute.router, tags=["Execute"])
app.include_router(status.router, tags=["Status"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(videos.router, tags=["Videos"])
app.include_router(speakers.router, tags=["Speakers"])
app.include_router(testimony.router, tags=["Testimony"])


@app.get("/")
async def root():
    return {"message": "VideoRAG API is running", "docs": "/docs"}


@app.get("/health")
async def health():
    from backend.services.indexing_service import get_index_stats
    stats = get_index_stats()
    # Check detection DB
    try:
        from backend.config import DETECTION_DB_PATH
        stats["detection_db"] = "available" if DETECTION_DB_PATH.parent.exists() else "missing"
    except Exception:
        stats["detection_db"] = "unavailable"
    # Check testimony DB
    try:
        from backend.config import TESTIMONY_DB_PATH
        stats["testimony_db"] = "available" if TESTIMONY_DB_PATH.parent.exists() else "missing"
    except Exception:
        stats["testimony_db"] = "unavailable"
    # Check citation DB
    try:
        from backend.config import CITATION_DB_PATH
        stats["citation_db"] = "available" if CITATION_DB_PATH.parent.exists() else "missing"
    except Exception:
        stats["citation_db"] = "unavailable"
    return {"status": "healthy", "indexes": stats}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
