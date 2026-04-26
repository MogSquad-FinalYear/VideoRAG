"""Quick verification script for Strategy A and B pipeline."""
import sys
import os
import traceback
import logging

# Suppress logging to keep output clean
logging.disable(logging.CRITICAL)

output_file = "d:/VideoRAG/test_output.txt"
f = open(output_file, "w", encoding="utf-8")

def p(msg):
    f.write(msg + "\n")
    f.flush()

p("=" * 60)
p("VideoRAG -- Strategy A & B Verification")
p("=" * 60)

# Test 1: Import all modules
p("\n[1] Testing imports...")
try:
    from backend.main import app
    from backend.agent.agent import run_agent
    from backend.services.indexing_service import get_index_stats, get_caption_for_frame
    from backend.services.captioning_service import describe_frame_clip
    from backend.services.embedding_service import embed_text
    p("   [OK] All imports successful")
except Exception as e:
    p(f"   [FAIL] Import failed: {e}")
    traceback.print_exc(file=f)
    f.close()
    sys.exit(1)

# Test 2: Check index stats
p("\n[2] Checking index stats...")
try:
    stats = get_index_stats()
    p(f"   Image index:   {stats['image_index']} items")
    p(f"   Caption index: {stats['caption_index']} items")
    p(f"   Speech index:  {stats['speech_index']} items")
except Exception as e:
    p(f"   [FAIL] Index stats failed: {e}")
    traceback.print_exc(file=f)
    stats = {'image_index': 0}

# Test 3: Test CLIP description function
p("\n[3] Testing CLIP frame description...")
try:
    from pathlib import Path
    frames_dir = Path("d:/VideoRAG/data/frames")
    frame_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
    if frame_dirs:
        first_dir = frame_dirs[0]
        frame_files = sorted(first_dir.glob("frame_*.jpg"))
        if frame_files:
            test_frame = str(frame_files[0])
            clip_desc = describe_frame_clip(test_frame)
            p(f"   Frame: {test_frame}")
            p(f"   CLIP desc: {clip_desc}")
            p("   [OK] CLIP description works")
        else:
            p("   [SKIP] No frame files")
    else:
        p("   [SKIP] No frame dirs")
except Exception as e:
    p(f"   [FAIL] CLIP description failed: {e}")
    traceback.print_exc(file=f)

# Test 4: Test Strategy A
p("\n[4] Testing Strategy A: Text-to-Video ('Find the blue car')...")
try:
    if stats['image_index'] > 0:
        result = run_agent(query="Find the blue car")
        p(f"   Sources: {len(result['sources'])}")
        p(f"   Clips: {len(result['clips'])}")
        for i, src in enumerate(result['sources'][:2]):
            desc = src.get('description', src.get('content', 'N/A'))
            p(f"   Match {i+1}: score={src.get('score', 0):.3f}, ts={src.get('timestamp')}")
            p(f"     Desc ({len(desc)} chars): {desc[:150]}...")
        p(f"   Answer:\n{result['answer'][:500]}")
        p("   [OK] Strategy A completed")
    else:
        p("   [SKIP] No indexed data")
except Exception as e:
    p(f"   [FAIL] Strategy A: {e}")
    traceback.print_exc(file=f)

# Test 5: Test Strategy B
p("\n[5] Testing Strategy B: Image-to-Video (simulated)...")
try:
    if stats['image_index'] > 0 and frame_dirs:
        first_dir = frame_dirs[0]
        frame_files = sorted(first_dir.glob("frame_*.jpg"))
        if len(frame_files) > 2:
            test_image = str(frame_files[len(frame_files)//2])
            result = run_agent(query="Is this person in the video?", image_path=test_image)
            p(f"   Sources: {len(result['sources'])}")
            p(f"   Clips: {len(result['clips'])}")
            for i, src in enumerate(result['sources'][:3]):
                desc = src.get('description', src.get('content', 'N/A'))
                p(f"   Match {i+1}: score={src.get('score', 0):.3f}, idx={src.get('source_index')}")
                p(f"     Desc ({len(desc)} chars): {desc[:150]}...")
            p(f"   Answer:\n{result['answer'][:500]}")
            p("   [OK] Strategy B completed")
        else:
            p("   [SKIP] Not enough frames")
    else:
        p("   [SKIP] No indexed data")
except Exception as e:
    p(f"   [FAIL] Strategy B: {e}")
    traceback.print_exc(file=f)

p("\n" + "=" * 60)
p("Verification complete!")
p("=" * 60)
f.close()
print("Done. Results in test_output.txt")
