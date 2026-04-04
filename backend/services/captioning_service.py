"""
VideoRAG — BLIP Captioning Service
Generates natural language captions for video frames.
"""
import logging
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from backend.config import BLIP_MODEL

logger = logging.getLogger(__name__)

_blip_model = None
_blip_processor = None
_device = "cpu"


def load_blip_model():
    """Load the BLIP captioning model once."""
    global _blip_model, _blip_processor, _device
    if _blip_model is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading BLIP model {BLIP_MODEL} on {_device}...")

    _blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    _blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(_device)
    _blip_model.eval()
    logger.info("BLIP model loaded successfully.")


def caption_frame(image_path: str) -> str:
    """Generate a text caption for a single frame."""
    load_blip_model()
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = _blip_processor(image, return_tensors="pt").to(_device)

        with torch.no_grad():
            output = _blip_model.generate(**inputs, max_new_tokens=50)

        caption = _blip_processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        logger.warning(f"Captioning failed for {image_path}: {e}")
        return ""


def caption_batch(image_paths: list[str], batch_size: int = 8) -> list[str]:
    """Generate captions for a batch of frames."""
    load_blip_model()
    all_captions = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        captions = []
        for p in batch_paths:
            cap = caption_frame(p)
            captions.append(cap)
        all_captions.extend(captions)
        logger.info(f"Captioned batch {i // batch_size + 1}, total: {len(all_captions)}/{len(image_paths)}")

    return all_captions
