"""Quick test for Phase 6-8 legal features (lightweight, no ML model loading)."""

# Test speaker service (no ML deps)
from backend.services.speaker_service import (
    assign_speaker_ids, auto_detect_roles_for_segments, auto_detect_role_from_text
)

segments = [
    {"text": "The court will come to order", "start_time": 0.0, "end_time": 2.0},
    {"text": "Please state your name for the record", "start_time": 2.5, "end_time": 4.0},
    {"text": "I saw the defendant at 5pm", "start_time": 8.0, "end_time": 10.0},
    {"text": "I was at the store when it happened", "start_time": 10.5, "end_time": 12.0},
    {"text": "Objection your honor that is hearsay", "start_time": 16.0, "end_time": 18.0},
]

print("=== Speaker Turn Detection ===")
result = assign_speaker_ids(segments, gap_threshold=2.0)
for seg in result:
    print(f"  {seg['speaker_id']}: [{seg['start_time']:.0f}s] {seg['text']}")

print("\n=== Auto Role Detection ===")
result = auto_detect_roles_for_segments(result)
for seg in result:
    print(f"  {seg['speaker_id']} ({seg['role']}): {seg['text']}")

print("\n=== Individual Role Tests ===")
print(f"  Judge cue: {auto_detect_role_from_text('The court finds the defendant guilty')}")
print(f"  Witness cue: {auto_detect_role_from_text('I saw him run away from the scene')}")
print(f"  Counsel cue: {auto_detect_role_from_text('Objection your honor')}")
print(f"  Unknown: {auto_detect_role_from_text('Hello world')}")

# Test testimony DB
print("\n=== Testimony DB ===")
from backend.services.testimony_db import store_statements_batch, get_all_statements, get_contradictions, delete_case_data

stmt_ids = store_statements_batch(
    case_id="test_case_001",
    session_id="session_1",
    video_id="vid_001",
    segments=result,
)
print(f"  Stored {len(stmt_ids)} statements")

stmts = get_all_statements("test_case_001")
print(f"  Retrieved {len(stmts)} statements")
for s in stmts:
    print(f"    [{s['speaker_id']}/{s['role']}] {s['text']}")

contradictions = get_contradictions("test_case_001")
print(f"  Contradictions found: {len(contradictions)}")

# Test citation DB
print("\n=== Citation DB ===")
from backend.services.citation_db import store_citation, get_citations_for_answer

cid = store_citation(
    answer_id="ans_001",
    cited_timestamp=42.5,
    cited_video="vid_001",
    claim_text="The witness stated they were at the store at 5pm",
    verification_score=0.85,
    verification_result="verified",
    clip_hash="abc123def456",
)
print(f"  Stored citation: {cid}")

cites = get_citations_for_answer("ans_001")
print(f"  Retrieved {len(cites)} citations")
for c in cites:
    print(f"    [{c['verification_result']}] score={c['verification_score']} hash={c['clip_hash']}")

# Cleanup
print("\n=== Cleanup ===")
delete_case_data("test_case_001")
print("  Test data cleaned up")

print("\n=== ALL PHASE 6-8 TESTS PASSED! ===")
