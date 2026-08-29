"""
Eval 2 — Multi-modal retrieval quality: Recall@k and MRR for the speech (ASR),
caption, and OCR indexes.

Methodology (self-supervised query construction, standard for evaluating an
existing index with no separately hand-labeled query set): for each stored
document in an index, build a short query from its own leading words (the way
a user skimming the footage would type a partial recollection), run the
production search function, and check whether the source document's own
(video_id, timestamp) is recovered in the top-k. This measures whether the
embedding+index pipeline can re-find content it already holds — a standard
lower bound on real-query retrieval quality.
"""
import random
from eval_common import REPO_ROOT, VIDEOS, apply_paper_style, save_metrics, savefig, maybe_title, OI

import sys
sys.path.insert(0, str(REPO_ROOT))
from backend.services import indexing_service  # noqa: E402

random.seed(42)
N_QUERIES_PER_MODALITY = 40
K_VALUES = [1, 3, 5, 10]


def build_query(text: str, n_words: int = 5) -> str:
    words = text.strip().split()
    return " ".join(words[:n_words]) if len(words) > n_words else text


def evaluate_modality(collection_name: str, search_fn, id_key="timestamp"):
    import chromadb
    client = chromadb.PersistentClient(path=str(REPO_ROOT / "data" / "chromadb"))
    col = client.get_or_create_collection(collection_name)
    res = col.get(include=["metadatas", "documents"])
    docs = [(m, d) for m, d in zip(res["metadatas"], res["documents"]) if d and d.strip()]
    if not docs:
        return None

    n_unique_docs = len(set(d for _, d in docs))
    sample = random.sample(docs, min(N_QUERIES_PER_MODALITY, len(docs)))
    ranks = []       # exact (video_id, timestamp) recovered — strict
    text_ranks = []  # any hit with the same document text recovered — content-aware
    for meta, doc in sample:
        query = build_query(doc)
        video_id = meta.get("video_id")
        target_ts = meta.get(id_key)
        hits = search_fn(query_text=query, n=10, video_id=None)
        rank, text_rank = None, None
        for i, h in enumerate(hits):
            same_video = h.get("video_id") == video_id
            same_time = abs((h.get(id_key) or -999) - (target_ts or -998)) < 0.6
            if rank is None and same_video and same_time:
                rank = i + 1
            if text_rank is None and h.get("content", "").strip() == doc.strip():
                text_rank = i + 1
        ranks.append(rank)
        text_ranks.append(text_rank)

    recall_at_k = {k: sum(1 for r in ranks if r is not None and r <= k) / len(ranks) for k in K_VALUES}
    text_recall_at_k = {k: sum(1 for r in text_ranks if r is not None and r <= k) / len(text_ranks) for k in K_VALUES}
    mrr = sum((1.0 / r) if r else 0.0 for r in ranks) / len(ranks)
    text_mrr = sum((1.0 / r) if r else 0.0 for r in text_ranks) / len(text_ranks)
    return {
        "n_queries": len(ranks),
        "n_unique_docs": n_unique_docs,
        "n_total_docs": len(docs),
        "recall_at_k_exact_frame": recall_at_k,
        "mrr_exact_frame": round(mrr, 3),
        "recall_at_k_content_match": text_recall_at_k,
        "mrr_content_match": round(text_mrr, 3),
    }


def main():
    plt = apply_paper_style()

    results = {}
    results["speech (ASR)"] = evaluate_modality(
        "speech_index_clip",
        lambda query_text, n, video_id: indexing_service.search_transcripts(query_text, n, video_id),
        id_key="start_time",
    )
    results["caption"] = evaluate_modality(
        "caption_index_clip",
        lambda query_text, n, video_id: indexing_service.search_captions(query_text, n, video_id),
        id_key="timestamp",
    )
    results["OCR"] = evaluate_modality(
        "ocr_index",
        lambda query_text, n, video_id: indexing_service.search_ocr(query_text, n, video_id),
        id_key="timestamp",
    )
    results = {k: v for k, v in results.items() if v is not None}

    # ── Figure: Recall@k grouped bar chart per modality (strict exact-frame) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    modalities = list(results.keys())
    width = 0.2
    colors = [OI["blue"], OI["vermillion"], OI["green"], OI["orange"]]
    x = range(len(modalities))
    for i, k in enumerate(K_VALUES):
        vals = [results[m]["recall_at_k_exact_frame"][k] for m in modalities]
        offs = [xi + (i - 1.5) * width for xi in x]
        ax.bar(offs, vals, width=width, label=f"Recall@{k}", color=colors[i])
    ax.set_xticks(list(x))
    ax.set_xticklabels(modalities)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    maybe_title(ax, "Eval 2 — Retrieval Recall@k by modality (exact source frame)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    savefig(fig, "eval02_recall_at_k")

    # ── Figure: exact-frame vs content-match MRR — shows how much of the
    # caption/OCR "miss" is actually a duplicate-content frame, not a failure ──
    fig1b, ax1b = plt.subplots(figsize=(7, 4.5))
    xw = 0.32
    xs = range(len(modalities))
    exact = [results[m]["mrr_exact_frame"] for m in modalities]
    content = [results[m]["mrr_content_match"] for m in modalities]
    ax1b.bar([i - xw / 2 for i in xs], exact, width=xw, label="MRR (exact source frame)", color=OI["blue"])
    ax1b.bar([i + xw / 2 for i in xs], content, width=xw, label="MRR (any matching-content frame)", color=OI["sky_blue"])
    ax1b.set_xticks(list(xs))
    ax1b.set_xticklabels(modalities)
    ax1b.set_ylabel("MRR")
    ax1b.set_ylim(0, 1.05)
    maybe_title(ax1b, "Eval 2 — Strict vs. content-aware MRR")
    ax1b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    fig1b.tight_layout()
    savefig(fig1b, "eval02_mrr_strict_vs_content_aware")

    save_metrics("eval02_retrieval", results)
    import json
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
