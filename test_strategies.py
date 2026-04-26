"""Test Strategy A & B against the running server"""
import requests
import json

BASE = "http://127.0.0.1:8000"

print("=" * 60)
print("STRATEGY A: Text-to-Video Search")
print("Query: 'Find the blue car'")
print("=" * 60)

r = requests.post(f"{BASE}/chat", json={"query": "Find the blue car"}, timeout=120)
d = r.json()
print(f"HTTP Status: {r.status_code}")
print(f"Clips count: {len(d.get('clips', []))}")
print(f"Sources count: {len(d.get('sources', []))}")

print(f"\n--- ANSWER ---")
print(d.get("answer", "")[:600])
print("--- END ANSWER ---")

for i, c in enumerate(d.get("clips", [])):
    desc = c.get("description", "(no description)")
    fp = c.get("frame_paths", [])
    print(f"\n  Clip {i+1}:")
    print(f"    Time: {c.get('start_time')} - {c.get('end_time')}")
    print(f"    Frame paths: {fp[:2]}")
    print(f"    Description: {desc[:200]}")

print(f"\n--- SOURCES ---")
for i, s in enumerate(d.get("sources", [])):
    print(f"  Src {i+1}: score={s.get('score')}, idx={s.get('source_index')}, content={s.get('content','')[:80]}")

print("\n\nDone!")
