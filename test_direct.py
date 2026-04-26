"""Minimal test: pre-load CLIP then test the full pipeline"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.path.insert(0, ".")

import torch
torch.set_num_threads(1)

print("Step 1: Loading CLIP model first...")
from backend.services.embedding_service import load_clip_model, embed_text
load_clip_model()
print("  CLIP loaded OK")

print("Step 2: Init ChromaDB...")
from backend.services.indexing_service import get_chroma_client, search_captions, search_images, get_index_stats, get_caption_for_frame
get_chroma_client()
stats = get_index_stats()
print(f"  Stats: {stats}")

print("Step 3: Search captions for 'blue car'...")
caption_results = search_captions("blue car", n=2)
print(f"  Found {len(caption_results)} results")
for r in caption_results:
    print(f"    frame={r.get('frame_number')}, score={r.get('score'):.4f}")
    print(f"    content: {r.get('content','')[:100]}")
    print(f"    frame_path: {r.get('frame_path','')}")

print("\nStep 4: Search images for 'blue car'...")
emb = embed_text("blue car. Related visual concepts: car vehicle sedan hatchback")
img_results = search_images(emb, n=2)
print(f"  Found {len(img_results)} results")
for r in img_results:
    print(f"    frame={r.get('frame_number')}, score={r.get('score'):.4f}")
    print(f"    frame_path: {r.get('frame_path','')}")

print("\nStep 5: Enrichment test...")
from backend.agent.agent import _enrich_with_descriptions
all_results = (caption_results + img_results)[:2]
enriched = _enrich_with_descriptions(all_results, max_describe=2)
for i, r in enumerate(enriched):
    print(f"  Result {i+1}:")
    print(f"    description: {r.get('description','NONE')[:150]}")
    print(f"    content: {r.get('content','NONE')[:100]}")

print("\nStep 6: Full agent run...")
from backend.agent.agent import run_agent
result = run_agent(query="Find the blue car")
print(f"  Clips: {len(result.get('clips',[]))}")
print(f"  Sources: {len(result.get('sources',[]))}")
print(f"  Answer (first 400 chars):")
print(f"  {result.get('answer','')[:400]}")
for i, c in enumerate(result.get('clips',[])):
    print(f"\n  Clip {i+1}:")
    print(f"    Time: {c.get('start_time')}-{c.get('end_time')}")
    print(f"    Description: {c.get('description','')[:150]}")
    print(f"    Frames: {c.get('frame_paths',[])[0] if c.get('frame_paths') else 'none'}")

print("\n=== ALL TESTS PASSED ===")
