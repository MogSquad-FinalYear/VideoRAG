"""
VideoRAG Configuration — loads .env, defines paths and model settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

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

# Groq
GROQ_MODEL = "llama-4-scout-17b-16e-instruct"

# Server
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "5173"))

# ChromaDB collections
IMAGE_COLLECTION = "image_index"
CAPTION_COLLECTION = "caption_index"
SPEECH_COLLECTION = "speech_index"
