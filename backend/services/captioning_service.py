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
    """Generate a text caption for a single frame using conditional captioning."""
    load_blip_model()
    try:
        image = Image.open(image_path).convert("RGB")
        # Use conditional captioning with a descriptive prefix for better results
        inputs = _blip_processor(
            image,
            text="a detailed description of this scene showing",
            return_tensors="pt"
        ).to(_device)

        with torch.no_grad():
            output = _blip_model.generate(
                **inputs,
                max_new_tokens=75,
                num_beams=3,         # Beam search for better quality
                repetition_penalty=1.5,  # Avoid repetitive captions
            )

        caption = _blip_processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        logger.warning(f"Captioning failed for {image_path}: {e}")
        return ""


def caption_batch(image_paths: list[str], batch_size: int = 8) -> list[str]:
    """Generate captions for a batch of frames using TRUE batching.

    Fix 3: Process images in proper batches through the model,
    instead of calling caption_frame() one at a time.
    """
    load_blip_model()
    all_captions = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]

        # Load and validate images for this batch
        images = []
        valid_indices = []
        for j, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_indices.append(j)
            except Exception as e:
                logger.warning(f"Could not load {p}: {e}")

        if not images:
            all_captions.extend([""] * len(batch_paths))
            continue

        try:
            # TRUE batch processing — process all images at once
            prefix = "a detailed description of this scene showing"
            inputs = _blip_processor(
                images,
                text=[prefix] * len(images),
                return_tensors="pt",
                padding=True,
            ).to(_device)

            with torch.no_grad():
                outputs = _blip_model.generate(
                    **inputs,
                    max_new_tokens=75,
                    num_beams=3,
                    repetition_penalty=1.5,
                )

            captions = _blip_processor.batch_decode(outputs, skip_special_tokens=True)

            # Rebuild results maintaining original order (with "" for failed images)
            batch_captions = [""] * len(batch_paths)
            for idx, valid_idx in enumerate(valid_indices):
                batch_captions[valid_idx] = captions[idx].strip() if idx < len(captions) else ""

            all_captions.extend(batch_captions)

        except Exception as e:
            # Fallback to individual captioning if batch fails
            logger.warning(f"Batch captioning failed, falling back to individual: {e}")
            for p in batch_paths:
                all_captions.append(caption_frame(p))

        logger.info(f"Captioned batch {i // batch_size + 1}, total: {len(all_captions)}/{len(image_paths)}")

    return all_captions
