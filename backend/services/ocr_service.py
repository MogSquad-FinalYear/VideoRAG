"""
VideoRAG — OCR Service (Phase 3)
Extracts on-screen text from video frames using EasyOCR.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_reader = None


def _get_reader():
    """Lazy-load EasyOCR reader (downloads models on first use)."""
    global _reader
    if _reader is None:
        try:
            import easyocr
            logger.info("Loading EasyOCR reader...")
            _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR reader loaded successfully.")
        except ImportError:
            logger.error("EasyOCR not installed. Run: pip install easyocr")
            raise
    return _reader


def extract_text_from_frame(image_path: str) -> list[dict]:
    """
    Extract text and bounding boxes from a single frame.

    Returns list of dicts:
        [{"text": str, "bbox": [x1, y1, x2, y2], "confidence": float}]
    """
    reader = _get_reader()
    try:
        results = reader.readtext(image_path)
        extracted = []
        for bbox, text, confidence in results:
            if confidence < 0.3 or len(text.strip()) < 2:
                continue
            # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] — simplify to [x1,y1,x3,y3]
            x1 = int(min(p[0] for p in bbox))
            y1 = int(min(p[1] for p in bbox))
            x2 = int(max(p[0] for p in bbox))
            y2 = int(max(p[1] for p in bbox))
            extracted.append({
                "text": text.strip(),
                "bbox": [x1, y1, x2, y2],
                "confidence": round(confidence, 3),
            })
        return extracted
    except Exception as e:
        logger.warning(f"OCR failed for {image_path}: {e}")
        return []


def extract_text_batch(image_paths: list[str], stride: int = 3) -> dict[str, list[dict]]:
    """
    Extract text from a batch of frames with optional stride (skip frames).

    Args:
        image_paths: List of frame file paths.
        stride: Process every Nth frame to save time (default: 3).

    Returns:
        Dict mapping frame_path -> list of OCR results.
    """
    results = {}
    for i, path in enumerate(image_paths):
        if i % stride != 0:
            results[path] = []
            continue
        results[path] = extract_text_from_frame(path)
        if results[path]:
            texts = [r["text"] for r in results[path]]
            logger.debug(f"OCR frame {i}: {texts}")

    total_with_text = sum(1 for v in results.values() if v)
    logger.info(f"OCR extracted text from {total_with_text}/{len(image_paths)} frames (stride={stride})")
    return results
