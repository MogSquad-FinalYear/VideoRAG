"""
Eval 6 — Citation verification faithfulness (Novelty 3): does the
post-answer verification layer actually catch a hallucinated/mismatched
citation, and correctly pass a genuinely supported one?

Ground truth: original claims I authored myself, each paired with a
(video_id, timestamp) — some are accurate summaries of what's really being
discussed at that point in the real "obama" video (should verify as
supported), some are deliberately mismatched to an unrelated topic at that
same timestamp (should verify as unsupported). This scores the verifier as a
binary classifier against known-correct labels, then compares that to the
naive "trust every citation" baseline the system would fall back to without
this layer — the delta is the layer's measurable contribution.
"""
import sys
from eval_common import REPO_ROOT, VIDEOS, apply_paper_style, save_metrics, savefig, maybe_title, OI
sys.path.insert(0, str(REPO_ROOT))
from backend.services import citation_service  # noqa: E402

OBAMA = VIDEOS["obama"]

# (claim, timestamp_s, ground_truth_supported) — claims are my own original
# wording; "supported" ones accurately summarize what's discussed near that
# timestamp (per earlier real-transcript verification in this project), the
# rest are deliberately swapped to an unrelated topic at that same timestamp.
TEST_CASES = [
    ("The speaker discusses not knowing who funds certain political advertisements.", 18, True),
    ("The speaker references a Supreme Court decision related to campaign spending.", 23, True),
    ("The speaker says the ad sponsors don't have to disclose who is paying for them.", 39, True),
    ("The speaker describes blocking a vote on an issue in the legislature.", 106, True),
    ("The speaker argues the issue shouldn't be split along party lines.", 195, True),
    ("The speaker gives detailed instructions on how to plant a vegetable garden.", 18, False),
    ("The speaker announces a new international trade agreement with Canada.", 23, False),
    ("The speaker discusses the results of a recent basketball championship.", 39, False),
    ("The speaker explains how to file annual tax returns.", 106, False),
    ("The speaker talks about the history of the national highway system.", 195, False),
]


def main():
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    plt = apply_paper_style()

    records = []
    for claim, ts, gt_supported in TEST_CASES:
        result = citation_service.verify_citation(claim, float(ts), OBAMA)
        pred_supported = bool(result.get("supported", False))
        records.append({
            "claim": claim, "timestamp": ts, "ground_truth_supported": gt_supported,
            "predicted_supported": pred_supported, "confidence": result.get("confidence"),
            "explanation": result.get("explanation"), "correct": pred_supported == gt_supported,
        })

    y_true = [r["ground_truth_supported"] for r in records]
    y_pred = [r["predicted_supported"] for r in records]
    naive_pred = [True] * len(records)  # "trust every citation" baseline (no verification layer)

    verifier_acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(records)
    naive_acc = sum(t == p for t, p in zip(y_true, naive_pred)) / len(records)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[True, False], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])

    results = {
        "n_cases": len(records),
        "verifier_accuracy": round(verifier_acc, 3),
        "naive_always_trust_accuracy": round(naive_acc, 3),
        "accuracy_gain_from_verification_layer": round(verifier_acc - naive_acc, 3),
        "supported_class": {"precision": round(float(precision[0]), 3), "recall": round(float(recall[0]), 3), "f1": round(float(f1[0]), 3)},
        "unsupported_class": {"precision": round(float(precision[1]), 3), "recall": round(float(recall[1]), 3), "f1": round(float(f1[1]), 3)},
        "confusion_matrix_labels": ["supported", "unsupported"],
        "confusion_matrix": cm.tolist(),
        "records": records,
    }

    # ── Figure: with vs without verification layer accuracy ──
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(["Without verification\n(trust every citation)", "With verification\n(Novelty 3 layer)"],
                   [naive_acc, verifier_acc], color=[OI["gray"], OI["blue"]], width=0.5)
    for b, v in zip(bars, [naive_acc, verifier_acc]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0%}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Citation-faithfulness accuracy")
    ax.set_ylim(0, 1.15)
    maybe_title(ax, "Eval 6 — Citation faithfulness, with vs. without verification")
    fig.tight_layout()
    savefig(fig, "eval06_citation_with_vs_without")

    # ── Figure: confusion matrix ──
    fig2, ax2 = plt.subplots(figsize=(5, 4.5))
    im = ax2.imshow(cm, cmap="Blues")
    labels = ["supported", "unsupported"]
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(labels)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(labels)
    ax2.set_xlabel("Verifier predicted"); ax2.set_ylabel("Ground truth")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14, fontweight="bold")
    maybe_title(ax2, f"Eval 6 — Verifier confusion matrix (n={len(records)})")
    fig2.tight_layout()
    savefig(fig2, "eval06_citation_confusion_matrix")

    save_metrics("eval06_citation_verification", results)
    import json
    print(json.dumps({k: v for k, v in results.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
