"""
VideoRAG — Scene Graph Service (Phase 5)
Rule-based spatial relationship extraction from YOLO bounding boxes.
Generates relation triples like (book, on top of, table).
"""
import logging

logger = logging.getLogger(__name__)


def _compute_iou(box_a: list, box_b: list) -> float:
    """Compute Intersection over Union between two [x, y, w, h] boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    # Convert to [x1, y1, x2, y2]
    a_x1, a_y1, a_x2, a_y2 = ax, ay, ax + aw, ay + ah
    b_x1, b_y1, b_x2, b_y2 = bx, by, bx + bw, by + bh

    # Intersection
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    a_area = aw * ah
    b_area = bw * bh
    union_area = a_area + b_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _center(box: list) -> tuple:
    """Get center point of [x, y, w, h] box."""
    return (box[0] + box[2] / 2, box[1] + box[3] / 2)


def _distance(box_a: list, box_b: list) -> float:
    """Euclidean distance between centers of two boxes."""
    ca = _center(box_a)
    cb = _center(box_b)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


def extract_relations(detections: list[dict], distance_threshold: float = 200.0) -> list[tuple]:
    """
    Extract spatial relation triples from a list of detections using rule-based heuristics.

    Args:
        detections: List of {"class_name": str, "bbox": [x, y, w, h], "confidence": float}
        distance_threshold: Maximum pixel distance between objects to consider "near"

    Returns:
        List of (subject, relation, object) string triples.
    """
    if len(detections) < 2:
        return []

    triples = []
    seen = set()

    for i, det_a in enumerate(detections):
        for j, det_b in enumerate(detections):
            if i >= j:
                continue

            name_a = det_a["class_name"]
            name_b = det_b["class_name"]
            box_a = det_a["bbox"]
            box_b = det_b["bbox"]

            ca = _center(box_a)
            cb = _center(box_b)
            dist = _distance(box_a, box_b)
            iou = _compute_iou(box_a, box_b)

            relations = []

            # ── Containment / Overlap ──
            if iou > 0.5:
                # One is largely inside the other
                area_a = box_a[2] * box_a[3]
                area_b = box_b[2] * box_b[3]
                if area_a < area_b:
                    relations.append((name_a, "inside", name_b))
                else:
                    relations.append((name_b, "inside", name_a))
            elif iou > 0.15:
                relations.append((name_a, "overlapping with", name_b))

            # ── Vertical spatial relationships ──
            # "on top of": A's bottom edge is near B's top edge (A rests on B)
            a_bottom = box_a[1] + box_a[3]
            b_top = box_b[1]
            b_bottom = box_b[1] + box_b[3]
            a_top = box_a[1]

            vertical_gap = 30  # pixels tolerance

            if abs(a_bottom - b_top) < vertical_gap and abs(ca[0] - cb[0]) < max(box_a[2], box_b[2]):
                relations.append((name_a, "on top of", name_b))
            elif abs(b_bottom - a_top) < vertical_gap and abs(ca[0] - cb[0]) < max(box_a[2], box_b[2]):
                relations.append((name_b, "on top of", name_a))

            # ── Horizontal spatial relationships ──
            if dist < distance_threshold:
                if ca[0] < cb[0] - box_b[2] * 0.3:
                    relations.append((name_a, "left of", name_b))
                elif ca[0] > cb[0] + box_b[2] * 0.3:
                    relations.append((name_a, "right of", name_b))

                if ca[1] < cb[1] - box_b[3] * 0.3:
                    relations.append((name_a, "above", name_b))
                elif ca[1] > cb[1] + box_b[3] * 0.3:
                    relations.append((name_a, "below", name_b))

            # ── Proximity ──
            if 0 < dist < distance_threshold and iou < 0.1:
                relations.append((name_a, "near", name_b))

            # Deduplicate
            for triple in relations:
                key = (triple[0], triple[1], triple[2])
                if key not in seen:
                    seen.add(key)
                    triples.append(triple)

    return triples


def process_frame_relations(detections: list[dict]) -> list[tuple]:
    """
    Process a single frame's detections and return scene graph triples.
    Wrapper for extract_relations with sensible defaults.
    """
    if len(detections) < 2:
        return []
    return extract_relations(detections)
