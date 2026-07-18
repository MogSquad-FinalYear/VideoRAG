"""
VideoRAG Configuration — loads .env, defines paths and model settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_environment() -> None:
    """Load environment variables from the most likely .env locations."""
    for env_path in (
        ROOT_DIR / ".env",
        ROOT_DIR / "backend" / ".env",
        Path.cwd() / ".env",
    ):
        if env_path.exists():
            load_dotenv(env_path, override=True, encoding="utf-8-sig")


_load_environment()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Data paths
DATA_DIR = ROOT_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
FRAMES_DIR = DATA_DIR / "frames"
AUDIO_DIR = DATA_DIR / "audio"
METADATA_DIR = DATA_DIR / "metadata"
CHROMADB_DIR = DATA_DIR / "chromadb"

for d in [VIDEOS_DIR, FRAMES_DIR, AUDIO_DIR, METADATA_DIR, CHROMADB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ML models (CPU-optimized)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")
BLIP_MODEL = os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-base")

# Processing
FRAME_SAMPLE_FPS = float(os.getenv("FRAME_SAMPLE_FPS", "1"))
MAX_FRAMES_PER_VIDEO = int(os.getenv("MAX_FRAMES_PER_VIDEO", "300"))
CAPTION_STRIDE = int(os.getenv("CAPTION_STRIDE", "3"))
CAPTION_MAX_FRAMES = int(os.getenv("CAPTION_MAX_FRAMES", "24"))
CAPTION_MODE = os.getenv("CAPTION_MODE", "sync").strip().lower()
if CAPTION_MODE not in {"sync", "async", "off"}:
    CAPTION_MODE = "async"
CAPTION_BACKEND = os.getenv("CAPTION_BACKEND", "blip").strip().lower()
if CAPTION_BACKEND not in {"clip", "blip"}:
    CAPTION_BACKEND = "blip"

# Groq
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
ENABLE_LLM_POLISH = os.getenv("ENABLE_LLM_POLISH", "false").strip().lower() == "true"

# Server
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "5173"))

# ChromaDB collections
IMAGE_COLLECTION = os.getenv("IMAGE_COLLECTION", "image_index")
CAPTION_COLLECTION = os.getenv("CAPTION_COLLECTION", "caption_index_clip")
SPEECH_COLLECTION = os.getenv("SPEECH_COLLECTION", "speech_index_clip")
OCR_COLLECTION = os.getenv("OCR_COLLECTION", "ocr_index")

# Phase 4-5: Object Detection & Scene Graph
DETECTION_DB_PATH = DATA_DIR / "detections.db"
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n")

# Phase 6-8: Legal / Courtroom Features
TESTIMONY_DB_PATH = DATA_DIR / "testimony.db"
CITATION_DB_PATH = DATA_DIR / "citations.db"
