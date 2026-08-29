"""
Eval 4 — Cross-session speaker verification: does the Resemblyzer voiceprint
+ cosine-similarity mechanism (Novelty 1) correctly separate same-speaker
pairs from different-speaker pairs?

Ground truth pairs, built from what actually has a captured voiceprint in the
running system right now:
  - Same speaker (positive): session1/session2/session3 — three synthetic
    clips using the identical TTS voice, so any pair among them is a true
    same-speaker pair.
  - Different speaker (negative): every cross pair between {session1,
    session2, session3} and {obama, clancy2}, plus obama vs. clancy2 —
    genuinely distinct real/synthetic voices.

Small n (10 pairs) — this is an exploratory-scale benchmark, not a
large-scale one; see EVALUATION.md for why (Clancy video 1's 4 speakers were
processed before the voiceprint-OOM fix and have no captured voiceprint).
"""
import sqlite3
import struct
import numpy as np
from itertools import combinations
from eval_common import REPO_ROOT, VIDEOS, apply_paper_style, save_metrics, savefig, maybe_title, OI

DB_PATH = REPO_ROOT / "data" / "testimony.db"

SAME_SPEAKER_GROUP = ["session1", "session2", "session3"]
OTHER_SPEAKERS = ["obama", "clancy2"]


def blob_to_vector(blob: bytes):
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob))


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_voiceprints():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT video_id, voiceprint_blob FROM speaker_video_map WHERE voiceprint_blob IS NOT NULL"
    ).fetchall()
    conn.close()
    by_video_id = {r["video_id"]: blob_to_vector(r["voiceprint_blob"]) for r in rows}
    label_to_id = {label: vid for label, vid in VIDEOS.items() if vid in by_video_id}
    return {label: by_video_id[vid] for label, vid in label_to_id.items()}


def main():
    from sklearn.metrics import roc_curve, auc
    plt = apply_paper_style()

    vps = load_voiceprints()
    print("voiceprints available for:", list(vps.keys()))

    pairs = []  # (label_a, label_b, score, is_same_speaker)
    for a, b in combinations(vps.keys(), 2):
        same = a in SAME_SPEAKER_GROUP and b in SAME_SPEAKER_GROUP
        score = cosine(vps[a], vps[b])
        pairs.append((a, b, score, same))

    scores = np.array([p[2] for p in pairs])
    labels = np.array([1 if p[3] else 0 for p in pairs])

    fpr, tpr, thresh = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr) if len(set(labels)) > 1 else float("nan")

    configured_threshold = 0.75  # match_voiceprint_to_registry default
    pred = scores >= configured_threshold
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    results = {
        "n_pairs": len(pairs),
        "n_same_speaker_pairs": int(labels.sum()),
        "n_different_speaker_pairs": int((1 - labels).sum()),
        "roc_auc": None if np.isnan(roc_auc) else round(float(roc_auc), 3),
        "configured_threshold": configured_threshold,
        "at_configured_threshold": {
            "precision": None if np.isnan(precision) else round(precision, 3),
            "recall": None if np.isnan(recall) else round(recall, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "pairs": [{"a": a, "b": b, "cosine_similarity": round(s, 4), "same_speaker_ground_truth": bool(sm)}
                   for a, b, s, sm in pairs],
    }

    # ── Figure: pairwise similarity matrix heatmap ──
    labels_order = list(vps.keys())
    n = len(labels_order)
    mat = np.eye(n)
    for a, b, s, _ in pairs:
        i, j = labels_order.index(a), labels_order.index(b)
        mat[i, j] = mat[j, i] = s

    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels_order, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels_order)
    for i in range(n):
        for j in range(n):
            same = (labels_order[i] in SAME_SPEAKER_GROUP and labels_order[j] in SAME_SPEAKER_GROUP)
            color = "white" if mat[i, j] > 0.6 else "black"
            marker = "•" if (i != j and same) else ""
            ax.text(j, i, f"{mat[i, j]:.2f}{marker}", ha="center", va="center", color=color, fontsize=9)
    maybe_title(ax, "Eval 4 — Cross-session voiceprint cosine similarity\n(• marks true same-speaker pairs)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")
    fig.tight_layout()
    savefig(fig, "eval04_speaker_similarity_matrix")

    # ── Figure: same- vs different-speaker score distributions ──
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    same_scores = scores[labels == 1]
    diff_scores = scores[labels == 0]
    ax2.scatter(same_scores, [1] * len(same_scores), color=OI["blue"], s=80, label=f"Same speaker (n={len(same_scores)})", zorder=3)
    ax2.scatter(diff_scores, [0] * len(diff_scores), color=OI["vermillion"], s=80, label=f"Different speaker (n={len(diff_scores)})", zorder=3)
    ax2.axvline(configured_threshold, color=OI["gray"], ls="--", lw=1, label=f"Configured threshold ({configured_threshold})")
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Different\nspeaker", "Same\nspeaker"])
    ax2.set_xlabel("Cosine similarity")
    maybe_title(ax2, "Eval 4 — Voiceprint similarity by pair type")
    ax2.set_xlim(-0.05, 1.05)
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig2.tight_layout()
    savefig(fig2, "eval04_speaker_score_by_class")

    save_metrics("eval04_speaker_verification", results)
    import json
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
