"""
VideoRAG — CLIP Embedding Service
Generates visual embeddings for frames and text queries using OpenCLIP.
"""
import logging
import threading
import torch
import open_clip
import numpy as np
from PIL import Image
from functools import lru_cache

from backend.config import CLIP_MODEL, CLIP_PRETRAINED

logger = logging.getLogger(__name__)

# Global model singletons (loaded once)
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_device = "cpu"
_model_lock = threading.Lock()


def load_clip_model():
    """Load the CLIP model once into memory (thread-safe)."""
    global _clip_model, _clip_preprocess, _clip_tokenizer, _device
    if _clip_model is not None:
        return
    with _model_lock:
        if _clip_model is not None:
            return
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model {CLIP_MODEL} on {_device}...")
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=_device
        )
        _clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        _clip_model.eval()
        logger.info("CLIP model loaded successfully.")


def embed_image(image_path: str) -> list[float]:
    """Generate CLIP embedding for a single image."""
    load_clip_model()
    image = Image.open(image_path).convert("RGB")
    image_tensor = _clip_preprocess(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        embedding = _clip_model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy().flatten().tolist()


@lru_cache(maxsize=256)
def _embed_text_cached(text: str) -> tuple:
    """Internal cached CLIP text embedding — returns a tuple (hashable for lru_cache).
    Same query string only runs one CLIP forward pass per process lifetime.
    """
    load_clip_model()
    tokens = _clip_tokenizer([text]).to(_device)
    with torch.no_grad():
        embedding = _clip_model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return tuple(embedding.cpu().numpy().flatten().tolist())


def embed_text(text: str) -> list[float]:
    """Generate CLIP embedding for a text query.

    Cached via _embed_text_cached — same query string is free after first call.
    Returns list[float] so all existing callers work unchanged.
    """
    return list(_embed_text_cached(text))


def embed_batch(image_paths: list[str], batch_size: int = 16) -> tuple[list[list[float]], list[str]]:
    """Generate CLIP embeddings for a batch of images.
    Returns (embeddings, valid_paths) — only successfully processed images.
    """
    load_clip_model()
    all_embeddings = []
    valid_paths = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        batch_valid_paths = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(_clip_preprocess(img))
                batch_valid_paths.append(p)
            except Exception as e:
                logger.warning(f"Failed to load image {p}, SKIPPING: {e}")

        if not images:
            continue

        batch_tensor = torch.stack(images).to(_device)

        with torch.no_grad():
            embeddings = _clip_model.encode_image(batch_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        all_embeddings.extend(embeddings.cpu().numpy().tolist())
        valid_paths.extend(batch_valid_paths)
        logger.info(f"Embedded batch {i // batch_size + 1}, total: {len(all_embeddings)}/{len(image_paths)}")

    return all_embeddings, valid_paths
