"""
Eval 3 — Image-to-video person search (Strategy B): precision/recall/ROC
as a function of the CLIP cosine-similarity threshold.

Ground truth: a face crop taken directly from the "obama" video is the query.
Every stored frame across ALL uploaded videos is scored against it; frames
from the obama video are the positive class (the person genuinely appears
there), frames from every other video are the negative class (a different
person/scene entirely) — a real discriminative task, not a toy.
"""
import numpy as np
from eval_common import REPO_ROOT, VIDEOS, apply_paper_style, save_metrics, savefig, maybe_title, OI

import sys
sys.path.insert(0, str(REPO_ROOT))
from backend.services import embedding_service  # noqa: E402

REFERENCE_IMAGE = REPO_ROOT.parent / "tmp"  # placeholder, overridden below
SCRATCH = None  # set in main from CLI/env


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main(reference_image_path: str):
    import chromadb
    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    plt = apply_paper_style()

    query_emb = embedding_service.embed_image(reference_image_path)

    client = chromadb.PersistentClient(path=str(REPO_ROOT / "data" / "chromadb"))
    col = client.get_or_create_collection("image_index")
    res = col.get(include=["embeddings", "metadatas"])

    scores, labels = [], []
    for emb, meta in zip(res["embeddings"], res["metadatas"]):
        sim = cosine(query_emb, emb)
        scores.append(sim)
        labels.append(1 if meta.get("video_id") == VIDEOS["obama"] else 0)

    scores = np.array(scores)
    labels = np.array(labels)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())

    fpr, tpr, roc_thresh = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    prec, rec, pr_thresh = precision_recall_curve(labels, scores)

    # EER-style operating point + the system's actual configured thresholds
    fnr = 1 - tpr
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_threshold = float(roc_thresh[eer_idx])

    configured_thresholds = {"chat_min_score (0.35)": 0.35, "hard_gate (0.30)": 0.30}
    op_points = {}
    for name, t in configured_thresholds.items():
        pred = scores >= t
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        op_points[name] = {"threshold": t, "precision": round(p, 3), "recall": round(r, 3),
                            "tp": tp, "fp": fp, "fn": fn}

    results = {
        "n_positive_frames": n_pos,
        "n_negative_frames": n_neg,
        "roc_auc": round(float(roc_auc), 3),
        "eer": round(eer, 3),
        "eer_threshold": round(eer_threshold, 3),
        "operating_points": op_points,
    }

    # ── Figure: ROC curve ──
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, color=OI["blue"], lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=OI["gray"], lw=1, ls="--", label="Chance")
    ax.scatter([fpr[eer_idx]], [tpr[eer_idx]], color=OI["vermillion"], zorder=5,
               label=f"EER = {eer:.3f} @ t={eer_threshold:.2f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    maybe_title(ax, "Eval 3 — Image-to-video person search ROC")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    savefig(fig, "eval03_image_search_roc")

    # ── Figure: Precision-Recall curve with configured operating points ──
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(rec, prec, color=OI["blue"], lw=2)
    colors_op = [OI["vermillion"], OI["green"]]
    for (name, pt), c in zip(op_points.items(), colors_op):
        ax2.scatter([pt["recall"]], [pt["precision"]], color=c, s=60, zorder=5, label=f"{name}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlim(-0.05, 1.05)
    maybe_title(ax2, "Eval 3 — Precision-Recall vs. similarity threshold")
    ax2.legend(loc="lower left", fontsize=9)
    fig2.tight_layout()
    savefig(fig2, "eval03_image_search_pr")

    # ── Figure: score distribution by class ──
    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    ax3.hist(scores[labels == 1], bins=20, alpha=0.75, color=OI["blue"], label=f"Positive (same person, n={n_pos})")
    ax3.hist(scores[labels == 0], bins=20, alpha=0.75, color=OI["vermillion"], label=f"Negative (different scene, n={n_neg})")
    ax3.axvline(0.35, color=OI["gray"], ls="--", lw=1)
    ax3.set_xlabel("CLIP cosine similarity to reference image")
    ax3.set_ylabel("Frame count")
    maybe_title(ax3, "Eval 3 — Similarity score distribution by class")
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    savefig(fig3, "eval03_image_search_score_dist")

    save_metrics("eval03_image_search", results)
    import json
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    ref = sys.argv[1] if len(sys.argv) > 1 else None
    if not ref:
        print("Usage: python eval_03_image_search.py <reference_image_path>")
        sys.exit(1)
    main(ref)
