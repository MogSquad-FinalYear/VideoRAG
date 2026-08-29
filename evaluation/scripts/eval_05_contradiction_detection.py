"""
Eval 5 — Contradiction detection: 3-way classification (CONTRADICT / ENTAIL /
UNRELATED) confusion matrix, precision/recall/F1 per class.

Ground truth = 18 originally-authored, generic courtroom-style statement
pairs (6 per class, written for this evaluation — not derived from any real
transcript, so labels are unambiguous and copyright-clean) + the 3 pairs
already produced by the running system's own synthetic cross-session test
(session1/2/3), which gives 2 more real CONTRADICT examples and 1 real
ENTAIL example end-to-end from actual uploaded audio.
"""
import sys
from eval_common import REPO_ROOT, apply_paper_style, save_metrics, savefig, maybe_title, OI
sys.path.insert(0, str(REPO_ROOT))
from backend.services import contradiction_service  # noqa: E402

# ── Self-authored ground truth (6 pairs per class) ──
HAND_LABELED = [
    # CONTRADICT
    ("The defendant was wearing a red jacket that night.",
     "The defendant was not wearing a jacket at all that night.", "CONTRADICT"),
    ("I saw her leave the building before 9 PM.",
     "She was still inside the building well after 9 PM.", "CONTRADICT"),
    ("The gun was found on the kitchen table.",
     "There was no gun anywhere in the kitchen.", "CONTRADICT"),
    ("He told me he had never met the victim.",
     "He admitted to me that he had known the victim for years.", "CONTRADICT"),
    ("The car was parked outside the entire evening.",
     "The car was gone for at least two hours that evening.", "CONTRADICT"),
    ("I was alone in the office when it happened.",
     "There were three other people in the office with me at the time.", "CONTRADICT"),
    # ENTAIL (consistent / paraphrase)
    ("The defendant was wearing a red jacket that night.",
     "Yes, I remember his jacket was red.", "ENTAIL"),
    ("I saw her leave the building before 9 PM.",
     "She definitely left before nine, I'm certain of that.", "ENTAIL"),
    ("The gun was found on the kitchen table.",
     "It was sitting right there on the table in the kitchen.", "ENTAIL"),
    ("He told me he had never met the victim.",
     "He was clear that they were strangers to each other.", "ENTAIL"),
    ("The car was parked outside the entire evening.",
     "It didn't move from that spot all night.", "ENTAIL"),
    ("I was alone in the office when it happened.",
     "No one else was around at that moment.", "ENTAIL"),
    # UNRELATED
    ("The defendant was wearing a red jacket that night.",
     "The weather report predicted rain for the weekend.", "UNRELATED"),
    ("I saw her leave the building before 9 PM.",
     "The courthouse cafeteria closes at noon on Fridays.", "UNRELATED"),
    ("The gun was found on the kitchen table.",
     "My commute to work usually takes forty minutes.", "UNRELATED"),
    ("He told me he had never met the victim.",
     "The judge asked for a ten minute recess.", "UNRELATED"),
    ("The car was parked outside the entire evening.",
     "The jury consists of twelve members.", "UNRELATED"),
    ("I was alone in the office when it happened.",
     "The defense attorney graduated from law school in 2010.", "UNRELATED"),
]

# ── Real end-to-end pairs from the running system's own synthetic ground truth ──
SYSTEM_VERIFIED = [
    ("I saw the car go through the red light at the intersection.",
     "I never saw the car near the intersection at all.", "CONTRADICT"),
    ("I never saw the car near the intersection at all.",
     "Yes, the car definitely ran the red light, I saw it clearly from where I was standing.", "CONTRADICT"),
    ("I saw the car go through the red light at the intersection.",
     "Yes, the car definitely ran the red light, I saw it clearly from where I was standing.", "ENTAIL"),
]

LABELS = ["CONTRADICT", "ENTAIL", "UNRELATED"]


def main():
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    plt = apply_paper_style()

    all_pairs = [(a, b, gt, "hand-authored") for a, b, gt in HAND_LABELED] + \
                [(a, b, gt, "system end-to-end") for a, b, gt in SYSTEM_VERIFIED]

    y_true, y_pred, records = [], [], []
    for a, b, gt, source in all_pairs:
        result = contradiction_service.check_contradiction(a, b)
        pred = result.get("label", "UNRELATED")
        y_true.append(gt)
        y_pred.append(pred)
        records.append({
            "statement_a": a, "statement_b": b, "ground_truth": gt,
            "predicted": pred, "confidence": result.get("confidence"),
            "source": source, "correct": pred == gt,
        })

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    accuracy = sum(r["correct"] for r in records) / len(records)

    # False-positive rate on CONTRADICT specifically (the costly failure mode
    # in a legal-evidence tool: flagging a contradiction that isn't real)
    contra_idx = LABELS.index("CONTRADICT")
    fp_contradict = sum(1 for t, p in zip(y_true, y_pred) if p == "CONTRADICT" and t != "CONTRADICT")
    n_non_contradict_truth = sum(1 for t in y_true if t != "CONTRADICT")
    contradict_fpr = fp_contradict / n_non_contradict_truth if n_non_contradict_truth else None

    results = {
        "n_pairs": len(records),
        "accuracy": round(accuracy, 3),
        "per_class": {
            LABELS[i]: {"precision": round(float(precision[i]), 3),
                        "recall": round(float(recall[i]), 3),
                        "f1": round(float(f1[i]), 3),
                        "support": int(support[i])}
            for i in range(len(LABELS))
        },
        "contradict_false_positive_rate": None if contradict_fpr is None else round(contradict_fpr, 3),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": cm.tolist(),
        "records": records,
    }

    # ── Figure: confusion matrix ──
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=20)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=13, fontweight="bold")
    maybe_title(ax, f"Eval 5 — Contradiction classifier confusion matrix\n(n={len(records)}, accuracy={accuracy:.2f})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    savefig(fig, "eval05_contradiction_confusion_matrix")

    # ── Figure: per-class P/R/F1 ──
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    width = 0.25
    x = range(len(LABELS))
    ax2.bar([i - width for i in x], precision, width=width, color=OI["blue"], label="Precision")
    ax2.bar([i for i in x], recall, width=width, color=OI["vermillion"], label="Recall")
    ax2.bar([i + width for i in x], f1, width=width, color=OI["green"], label="F1")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(LABELS)
    ax2.set_ylim(0, 1.15)
    maybe_title(ax2, "Eval 5 — Precision / Recall / F1 by class")
    ax2.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig2.tight_layout()
    savefig(fig2, "eval05_contradiction_prf1")

    save_metrics("eval05_contradiction_detection", results)
    import json
    print(json.dumps({k: v for k, v in results.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
