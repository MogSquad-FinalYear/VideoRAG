import sys
import json
import logging
logging.disable(logging.CRITICAL)

from backend.services.indexing_service import search_images, search_captions, search_captions_by_keyword, get_index_stats
from backend.services.embedding_service import embed_text

print('Stats:', get_index_stats())

queries = ['show me a clip where fighting started', 'show me a clip where person fell down']
video_id = 'da5ac56e-8f7'

for q in queries:
    print(f'\n--- Query: {q} ---')
    emb = embed_text(q)
    
    img_hits = search_images(emb, n=5, video_id=video_id, min_score=0.0)
    print("Image Search:")
    for h in img_hits:
        print(f"  Score: {h['score']}, Frame: {h['frame_number']}")
        
    cap_hits = search_captions(q, n=5, video_id=video_id)
    print("Caption Search:")
    for h in cap_hits:
        print(f"  Score: {h['score']}, Content: {h['content'][:50]}")
        
    key_hits = search_captions_by_keyword(q, n=5, video_id=video_id)
    print("Keyword Search:")
    for h in key_hits:
        print(f"  Score: {h['score']}, Content: {h['content'][:50]}")
