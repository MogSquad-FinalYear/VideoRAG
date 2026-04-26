"""Debug: test frame path resolution and BLIP description generation"""
import sys
sys.path.insert(0, ".")

from backend.config import FRAMES_DIR
from backend.services.indexing_service import search_captions, search_images, get_caption_for_frame
from backend.services.embedding_service import embed_text
from backend.agent.agent import _url_to_fs_path, _enrich_with_descriptions

# Test 1: Check what search returns
print("=== Test 1: Caption search for 'blue car' ===")
results = search_captions("blue car", n=3)
for r in results[:3]:
    print(f"  frame_path: {r.get('frame_path')}")
    print(f"  video_id: {r.get('video_id')}")
    print(f"  frame_number: {r.get('frame_number')}")
    print(f"  content: {r.get('content')}")
    print(f"  score: {r.get('score')}")
    print()

# Test 2: Check frame path resolution
if results:
    first = results[0]
    frame_url = first.get("frame_path", "")
    print(f"=== Test 2: Frame path resolution ===")
    print(f"  URL path: {frame_url}")
    fs_path = _url_to_fs_path(frame_url)
    print(f"  FS path: {fs_path}")
    
    if fs_path:
        from pathlib import Path
        print(f"  File exists: {Path(fs_path).exists()}")

# Test 3: Test BLIP description generation
if results:
    first = results[0]
    frame_url = first.get("frame_path", "")
    fs_path = _url_to_fs_path(frame_url)
    if fs_path:
        print(f"\n=== Test 3: BLIP description generation ===")
        print(f"  Generating BLIP description for: {fs_path}")
        from backend.services.captioning_service import describe_frame_detailed
        desc = describe_frame_detailed(fs_path)
        print(f"  Description: {desc}")

# Test 4: Test enrichment
print(f"\n=== Test 4: Enrichment ===")
enriched = _enrich_with_descriptions(results[:2], max_describe=2)
for r in enriched:
    print(f"  description: {r.get('description', 'NONE')}")
    print(f"  content: {r.get('content', 'NONE')}")
    print()

print("Done!")
