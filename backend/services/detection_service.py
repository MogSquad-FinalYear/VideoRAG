"""
VideoRAG — Object Detection Service (Phase 4)
Runs YOLOv8 on video frames for object detection, counting, and location.
"""
import logging

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazy-load YOLOv8 model."""
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
            from backend.config import YOLO_MODEL
            logger.info(f"Loading YOLO model: {YOLO_MODEL}...")
            _model = YOLO(f"{YOLO_MODEL}.pt")
            logger.info("YOLO model loaded successfully.")
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise
    return _model


def detect_objects(image_path: str, conf_threshold: float = 0.3) -> list[dict]:
    """
    Run object detection on a single frame.

    Returns list of detections:
        [{"class_name": str, "confidence": float, "bbox": [x, y, w, h]}]
    where bbox is [center_x, center_y, width, height] normalized to image size.
    """
    model = _get_model()
    try:
        results = model(image_path, verbose=False, conf=conf_threshold)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                # Get bbox as [x1, y1, x2, y2] then convert to [x, y, w, h]
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                w = x2 - x1
                h = y2 - y1
                detections.append({
                    "class_name": result.names[cls_id],
                    "confidence": round(conf, 3),
                    "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                })
        return detections
    except Exception as e:
        logger.warning(f"Detection failed for {image_path}: {e}")
        return []


def detect_batch(image_paths: list[str], stride: int = 2,
                 conf_threshold: float = 0.3) -> dict[str, list[dict]]:
    """
    Run object detection on a batch of frames with optional stride.

    Args:
        image_paths: List of frame file paths.
        stride: Process every Nth frame (default: 2).
        conf_threshold: Minimum detection confidence.

    Returns:
        Dict mapping frame_path -> list of detections.
    """
    results = {}
    for i, path in enumerate(image_paths):
        if i % stride != 0:
            results[path] = []
            continue
        results[path] = detect_objects(path, conf_threshold)

    total_with_det = sum(1 for v in results.values() if v)
    logger.info(f"Detection: found objects in {total_with_det}/{len(image_paths)} frames (stride={stride})")
    return results


def count_objects_in_detections(detections: list[dict], class_name: str = None) -> dict[str, int]:
    """
    Count object instances by class from a list of detections.

    Args:
        detections: List of detection dicts from detect_objects().
        class_name: Optional filter — count only this class.

    Returns:
        Dict of {class_name: count}.
    """
    counts = {}
    for det in detections:
        cls = det["class_name"]
        if class_name and cls.lower() != class_name.lower():
            continue
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def get_object_locations(detections: list[dict], class_name: str,
                         image_width: int = None, image_height: int = None) -> list[dict]:
    """
    Get locations of a specific object class from detections.

    Returns list of location dicts with bounding box and spatial position description.
    """
    locations = []
    for det in detections:
        if det["class_name"].lower() != class_name.lower():
            continue
        x, y, w, h = det["bbox"]
        # Compute center
        cx = x + w / 2
        cy = y + h / 2

        # Spatial description
        if image_width and image_height:
            rx = cx / image_width
            ry = cy / image_height
            h_pos = "left" if rx < 0.33 else ("right" if rx > 0.66 else "center")
            v_pos = "top" if ry < 0.33 else ("bottom" if ry > 0.66 else "middle")
            spatial = f"{v_pos}-{h_pos}"
        else:
            spatial = "unknown"

        locations.append({
            "class_name": det["class_name"],
            "confidence": det["confidence"],
            "bbox": det["bbox"],
            "center": [round(cx, 1), round(cy, 1)],
            "spatial_position": spatial,
        })
    return locations
