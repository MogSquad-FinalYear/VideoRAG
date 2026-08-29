"""
VideoRAG — BLIP Captioning Service
Generates natural language captions for video frames.
"""
import logging
import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from backend.config import BLIP_MODEL, CAPTION_BACKEND
from backend.services import embedding_service

logger = logging.getLogger(__name__)

_blip_model = None
_blip_processor = None
_device = "cpu"
_label_embeddings = None

_LABELS = [
    # People & identity
    "person", "man", "woman", "child", "group of people", "face closeup",
    "crowd", "pedestrian", "suspect", "bystander",
    # Vehicles
    "car", "truck", "motorcycle", "bicycle", "bike", "bus", "van",
    "auto rickshaw", "scooter", "ambulance", "police car",
    # Vehicle actions & events
    "car accident", "vehicle collision", "car crash", "car hitting bike",
    "car hitting motorcycle", "vehicle speeding", "traffic",
    "vehicle parked", "vehicle driving on road",
    # Locations & scenes
    "street", "road", "highway", "intersection", "parking lot",
    "building", "office", "room", "corridor", "door", "window",
    "sidewalk", "alley", "bridge", "tunnel",
    # Objects
    "table", "chair", "bag", "backpack", "phone", "laptop",
    "weapon", "knife", "gun", "blood", "evidence bag",
    "helmet", "license plate", "sign", "traffic light",
    # Sports & venues
    "sports field", "basketball court", "stadium", "scoreboard",
    # Actions & events
    "walking", "running", "fighting", "arguing", "falling",
    "hitting", "kicking", "punching", "chasing", "fleeing",
    "talking", "holding object", "sitting", "standing",
    "stealing", "breaking", "climbing", "jumping",
    # Conditions
    "outdoor daylight", "outdoor night", "indoor scene",
    "rain", "fog", "camera close shot", "surveillance footage",
    "cctv view", "dashcam view",
]


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
                max_new_tokens=40,
                num_beams=1,
                repetition_penalty=1.2,
            )

        caption = _blip_processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        logger.warning(f"Captioning failed for {image_path}: {e}")
        return ""


def _load_label_embeddings() -> np.ndarray:
    """Build CLIP text embeddings for a fixed, lightweight label vocabulary."""
    global _label_embeddings
    if _label_embeddings is None:
        _label_embeddings = np.array(
            [embedding_service.embed_text(label) for label in _LABELS],
            dtype=np.float32,
        )
    return _label_embeddings


def _caption_batch_clip(image_paths: list[str]) -> list[str]:
    """Generate fast pseudo-captions by matching frame embeddings to label prompts.

    Uses top-5 matching labels with a lower threshold to produce richer,
    more descriptive captions for better downstream retrieval.
    """
    if not image_paths:
        return []

    label_emb = _load_label_embeddings()
    img_emb, valid_paths = embedding_service.embed_batch(image_paths, batch_size=24)
    if not img_emb:
        return ["" for _ in image_paths]

    # Categorize labels for structured captions
    _ACTION_LABELS = {
        "walking", "running", "fighting", "arguing", "falling",
        "hitting", "kicking", "punching", "chasing", "fleeing",
        "talking", "holding object", "sitting", "standing",
        "stealing", "breaking", "climbing", "jumping",
    }
    _EVENT_LABELS = {
        "car accident", "vehicle collision", "car crash", "car hitting bike",
        "car hitting motorcycle", "vehicle speeding",
    }

    path_to_caption = {}
    for emb, path in zip(img_emb, valid_paths):
        vec = np.array(emb, dtype=np.float32)
        sims = label_emb @ vec
        top_idx = np.argsort(sims)[-5:][::-1]  # Top-5 labels
        top_labels = [_LABELS[i] for i in top_idx if sims[i] > 0.14]

        if not top_labels:
            path_to_caption[path] = "unidentified scene in video"
            continue

        # Separate into actions/events vs objects/scenes
        actions = [l for l in top_labels if l in _ACTION_LABELS or l in _EVENT_LABELS]
        objects = [l for l in top_labels if l not in _ACTION_LABELS and l not in _EVENT_LABELS]

        parts = []
        if actions and objects:
            parts.append(f"{', '.join(objects)} with {', '.join(actions)}")
        elif objects:
            parts.append(', '.join(objects))
        elif actions:
            parts.append(', '.join(actions))

        caption = f"a scene showing {' and '.join(parts)}" if parts else "unidentified scene"

        # Add all detected labels as keywords for better keyword-based retrieval
        all_keywords = ' '.join(top_labels)
        caption = f"{caption}. Keywords: {all_keywords}"

        path_to_caption[path] = caption

    captions = [path_to_caption.get(p, "") for p in image_paths]
    return captions


def describe_frame_clip(image_path: str) -> str:
    """Generate a rich text description of a single frame using CLIP label matching.

    This is faster and more reliable than BLIP — uses the already-loaded CLIP model
    to match the frame against our label vocabulary. Produces richer descriptions
    than the batch captioner by using more labels and confidence scoring.
    """
    try:
        label_emb = _load_label_embeddings()
        emb_list, valid = embedding_service.embed_batch([image_path], batch_size=1)
        if not emb_list or not valid:
            return ""

        vec = np.array(emb_list[0], dtype=np.float32)
        sims = label_emb @ vec
        top_idx = np.argsort(sims)[-5:][::-1]  # Top-5 labels for richer detail

        # Classify labels by confidence
        high_conf = [_LABELS[i] for i in top_idx if sims[i] > 0.24]
        med_conf = [_LABELS[i] for i in top_idx if 0.18 < sims[i] <= 0.24]

        parts = []
        if high_conf:
            parts.append(f"The frame clearly shows: {', '.join(high_conf)}")
        if med_conf:
            parts.append(f"Also possibly visible: {', '.join(med_conf)}")

        if not parts:
            low_conf = [_LABELS[i] for i in top_idx[:2] if sims[i] > 0.14]
            if low_conf:
                parts.append(f"The frame appears to contain: {', '.join(low_conf)}")
            else:
                return "A frame from the video with indeterminate content."

        # Add scene context based on detected labels
        all_detected = high_conf + med_conf
        all_lower = " ".join(all_detected).lower()

        if any(t in all_lower for t in ["person", "man", "woman", "crowd", "pedestrian", "suspect"]):
            parts.append("Human subjects are present in the scene.")
        if any(t in all_lower for t in ["car", "truck", "vehicle", "motorcycle", "bus", "bicycle"]):
            parts.append("Vehicles or transport are visible.")
        if any(t in all_lower for t in ["accident", "collision", "crash", "hitting"]):
            parts.append("An impact or collision event appears to be occurring.")
        if any(t in all_lower for t in ["outdoor", "street", "road", "highway", "parking"]):
            parts.append("The scene is outdoors.")
        elif any(t in all_lower for t in ["indoor", "room", "office", "corridor"]):
            parts.append("The scene is indoors.")

        return ". ".join(parts) + "."
    except Exception as e:
        logger.warning(f"CLIP frame description failed for {image_path}: {e}")
        return ""


def caption_batch(image_paths: list[str], batch_size: int = 8) -> list[str]:
    """Generate captions for a batch of frames using TRUE batching.

    Fix 3: Process images in proper batches through the model,
    instead of calling caption_frame() one at a time.
    """
    if CAPTION_BACKEND == "clip":
        return _caption_batch_clip(image_paths)

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
                    max_new_tokens=40,
                    num_beams=1,
                    repetition_penalty=1.2,
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


def describe_frame_detailed(image_path: str) -> str:
    """Generate a detailed description of a specific frame using BLIP.

    This is called on-demand for top search results (not during indexing),
    so we can afford to use more beam search and multiple prompts for quality.
    """
    load_blip_model()
    try:
        image = Image.open(image_path).convert("RGB")
        prompts = [
            "a detailed description of this image:",
            "this scene shows",
        ]
        descriptions = []
        for prompt in prompts:
            inputs = _blip_processor(image, text=prompt, return_tensors="pt").to(_device)
            with torch.no_grad():
                output = _blip_model.generate(
                    **inputs,
                    max_new_tokens=60,
                    num_beams=3,
                    repetition_penalty=1.5,
                )
            caption = _blip_processor.decode(output[0], skip_special_tokens=True).strip()
            if caption and caption not in descriptions:
                descriptions.append(caption)

        if descriptions:
            combined = descriptions[0]
            if len(descriptions) > 1:
                second = descriptions[1]
                if second.lower() not in combined.lower():
                    combined = f"{combined}. Additionally: {second}"
            return combined
        return "Scene from video"
    except Exception as e:
        logger.warning(f"Detailed captioning failed for {image_path}: {e}")
        return ""


def describe_frames_batch(image_paths: list[str]) -> list[str]:
    """Generate detailed descriptions for multiple frames efficiently.

    Processes frames through BLIP one at a time but with richer prompts.
    Designed for small batches (2-5 frames) from search results.
    """
    return [describe_frame_detailed(p) for p in image_paths]
