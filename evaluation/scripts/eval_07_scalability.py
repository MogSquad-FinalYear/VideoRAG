"""
Eval 7 — Processing time vs. video duration (scalability), and the
voiceprint-OOM-fix before/after comparison, both from REAL wall-clock
timestamps recorded in this project's own backend logs during development
and testing (not synthetic/simulated numbers).

"Cold" = first upload after a server (re)start, which pays a one-time model
loading cost (CLIP/BLIP/Whisper/YOLO/EasyOCR/Resemblyzer all load lazily on
first use). "Warm" = every later upload in the same server process, which
reuses already-loaded models — the fairer number for a scalability claim.
"""
from datetime import datetime
from eval_common import apply_paper_style, save_metrics, savefig, maybe_title, OI

FMT = "%Y-%m-%d %H:%M:%S,%f"


def secs(t0, t1):
    return (datetime.strptime(t1, FMT) - datetime.strptime(t0, FMT)).total_seconds()


# (video_id, duration_s, meta_ts, complete_ts, cold_start)
RUNS = [
    ("64676f3b-1c5", 4.0, "2026-08-12 17:16:41,414", "2026-08-12 17:38:25,786", True),
    ("346dd588-13e", 4.0, "2026-08-12 18:15:01,688", "2026-08-12 18:15:20,482", True),
    ("5963ac97-fdf", 4.0, "2026-08-12 18:18:44,272", "2026-08-12 18:19:05,820", True),
    ("83b964c5-b7d", 3.0, "2026-08-12 18:19:16,142", "2026-08-12 18:19:26,209", False),
    ("716d12b3-218", 3.0, "2026-08-12 18:23:51,335", "2026-08-12 18:23:54,224", True),
    ("15451208-79a", 214.2, "2026-08-12 18:51:19,888", "2026-08-12 18:52:05,122", True),
    ("6244a6c0-ff9", 6.0, "2026-08-12 18:53:21,467", "2026-08-12 18:53:32,554", False),
    ("b2a91a55-681", 1395.0, "2026-08-12 19:03:04,884", "2026-08-12 19:05:27,738", True),
    ("dac0ae13-04c (pre-fix, OOM'd internally)", 1316.5, "2026-08-12 19:05:54,408", "2026-08-12 19:08:18,650", False),
    ("b730910d-141 (post-fix rerun)", 1316.5, "2026-08-12 19:11:36,978", "2026-08-12 19:14:08,125", True),
    ("94731c1d-387", 4.0, "2026-08-13 17:46:07,718", "2026-08-13 17:46:40,734", True),
    ("be7b9715-a02", 3.0, "2026-08-13 17:46:41,253", "2026-08-13 17:46:51,555", False),
    ("e8a749f2-0bb", 6.0, "2026-08-13 17:46:53,392", "2026-08-13 17:47:04,653", False),
    ("9fa69b5d-e50", 214.2, "2026-08-13 17:47:23,709", "2026-08-13 17:48:02,402", False),
    ("a457b964-073", 3.0, "2026-08-13 17:48:16,686", "2026-08-13 17:48:17,611", False),
]


def main():
    plt = apply_paper_style()

    rows = []
    for vid, dur, t0, t1, cold in RUNS:
        rows.append({"video_id": vid, "duration_s": dur, "processing_s": round(secs(t0, t1), 1), "cold_start": cold})

    warm = [r for r in rows if not r["cold_start"]]
    cold = [r for r in rows if r["cold_start"]]

    results = {
        "n_runs": len(rows),
        "n_warm": len(warm),
        "n_cold": len(cold),
        "warm_mean_overhead_s_beyond_duration": round(
            sum(r["processing_s"] - 0 for r in warm) / len(warm), 1) if warm else None,
        "cold_start_extra_cost_s_estimate": round(
            (sum(r["processing_s"] for r in cold) / len(cold)) -
            (sum(r["processing_s"] for r in warm) / len(warm)), 1) if warm and cold else None,
        "runs": rows,
    }

    # ── Figure: processing time vs video duration, warm vs cold ──
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter([r["duration_s"] for r in warm], [r["processing_s"] for r in warm],
               color=OI["blue"], s=70, label=f"Warm (models cached, n={len(warm)})", zorder=3)
    ax.scatter([r["duration_s"] for r in cold], [r["processing_s"] for r in cold],
               color=OI["vermillion"], s=70, marker="^", label=f"Cold start (n={len(cold)})", zorder=3)
    lims = [0, max(r["duration_s"] for r in rows) * 1.05]
    ax.plot(lims, lims, color=OI["gray"], ls="--", lw=1, label="y = x (real-time)")
    ax.set_xlabel("Video duration (s)")
    ax.set_ylabel("Wall-clock processing time (s)")
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    maybe_title(ax, "Eval 7 — Processing time vs. video duration")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "eval07_scalability")

    # ── Figure: the OOM fix, before vs after, same video re-processed ──
    before = next(r for r in rows if r["video_id"].startswith("dac0ae13"))
    after = next(r for r in rows if r["video_id"].startswith("b730910d"))
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    bars = ax2.bar(["Pre-fix\n(voiceprint step OOM'd,\nsilently skipped)", "Post-fix\n(voiceprint captured\nsuccessfully)"],
                    [before["processing_s"], after["processing_s"]], color=[OI["vermillion"], OI["blue"]], width=0.5)
    for b, v in zip(bars, [before["processing_s"], after["processing_s"]]):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}s", ha="center", va="bottom", fontweight="bold")
    ax2.set_ylabel("Total processing time (s)")
    maybe_title(ax2, "Eval 7 — Same 1316s video, before vs. after voiceprint-OOM fix")
    fig2.tight_layout()
    savefig(fig2, "eval07_oom_fix_before_after")

    save_metrics("eval07_scalability", results)
    import json
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
