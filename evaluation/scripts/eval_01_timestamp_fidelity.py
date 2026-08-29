"""
Eval 1 — Timestamp fidelity (validates Bug #1's fix).

The pre-fix code computed `timestamp = saved_frame_index / FRAME_SAMPLE_FPS`,
silently assuming one saved frame == one real second. That's only true when
the sampling interval is never widened, which breaks the moment a video is
long/dense enough to need adaptive spacing or the MAX_FRAMES_PER_VIDEO cap —
exactly the case_clancy_trial videos (23 min compressed to a 300-frame budget).

This script pulls the REAL stored timestamps (post-fix) out of the running
system's image_index for every uploaded video, reconstructs what the OLD
formula would have produced for the same frames using each video's actual
duration/frame_count, and reports the resulting drift.
"""
import chromadb
from eval_common import (
    REPO_ROOT, VIDEOS, apply_paper_style, save_metrics, savefig, maybe_title, OI,
)

CHROMADB_DIR = REPO_ROOT / "data" / "chromadb"
FRAME_SAMPLE_FPS = 1.0  # matches backend/config.py default


def get_frames_for_video(video_id: str):
    client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
    col = client.get_or_create_collection("image_index")
    res = col.get(where={"video_id": video_id}, include=["metadatas"])
    frames = []
    for meta in res["metadatas"]:
        frames.append({
            "frame_number": meta.get("frame_number"),
            "timestamp": meta.get("timestamp"),
        })
    frames.sort(key=lambda f: f["frame_number"])
    return frames


def main():
    plt = apply_paper_style()
    import numpy as np

    results = {}
    for label, video_id in VIDEOS.items():
        frames = get_frames_for_video(video_id)
        if not frames:
            continue
        real_ts = [f["timestamp"] for f in frames]
        frame_nums = [f["frame_number"] for f in frames]

        # Reconstruct what the pre-fix formula would have output for these
        # same saved-frame indices.
        buggy_ts = [fn / FRAME_SAMPLE_FPS for fn in frame_nums]

        errors = [abs(r - b) for r, b in zip(real_ts, buggy_ts)]
        monotonic = all(real_ts[i] <= real_ts[i + 1] for i in range(len(real_ts) - 1))
        duration = max(real_ts) if real_ts else 0.0

        results[label] = {
            "video_id": video_id,
            "n_frames": len(frames),
            "duration_s": round(duration, 1),
            "mae_vs_buggy_formula_s": round(float(np.mean(errors)), 2),
            "max_error_vs_buggy_formula_s": round(float(np.max(errors)), 2),
            "monotonic_real_timestamps": monotonic,
            "within_duration_bounds": all(0 <= r <= duration + 1e-6 for r in real_ts),
        }
        results[label]["real_ts"] = real_ts
        results[label]["buggy_ts"] = buggy_ts
        results[label]["frame_nums"] = frame_nums

    # ── Figure: drift vs frame index, fixed vs reconstructed-buggy, for a
    # short/medium/long video to show the error scales with video length ──
    order = ["session1", "obama", "clancy1"]
    order = [o for o in order if o in results]
    fig, axes = plt.subplots(1, len(order), figsize=(5 * len(order), 4), sharey=False)
    if len(order) == 1:
        axes = [axes]
    for ax, label in zip(axes, order):
        r = results[label]
        ax.plot(r["frame_nums"], r["real_ts"], color=OI["blue"], lw=2, label="Fixed (real timestamp)")
        ax.plot(r["frame_nums"], r["buggy_ts"], color=OI["vermillion"], lw=2, ls="--", label="Pre-fix formula (reconstructed)")
        maybe_title(ax, f"{label}  ({r['duration_s']}s, {r['n_frames']} frames)")
        ax.set_xlabel("Saved frame index")
        ax.set_ylabel("Timestamp (s)")
    axes[0].legend(loc="upper left", fontsize=9)
    maybe_title(fig, "Eval 1 — Per-frame timestamp: fixed vs. pre-fix formula", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "eval01_timestamp_fidelity")

    # ── Figure: MAE summary bar chart across all videos ──
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    labels = list(results.keys())
    maes = [results[l]["mae_vs_buggy_formula_s"] for l in labels]
    bars = ax2.bar(labels, maes, color=OI["blue"], width=0.6)
    ax2.set_ylabel("Mean drift vs. pre-fix formula (s)")
    maybe_title(ax2, "Eval 1 — Timestamp drift the fix corrects, by video")
    ax2.set_yscale("symlog")
    for b, v in zip(bars, maes):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=9)
    fig2.tight_layout()
    savefig(fig2, "eval01_timestamp_mae_summary")

    # strip raw arrays before saving json summary (keep it small/readable)
    summary = {k: {kk: vv for kk, vv in v.items() if kk not in ("real_ts", "buggy_ts", "frame_nums")}
               for k, v in results.items()}
    save_metrics("eval01_timestamp_fidelity", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import json
    main()
