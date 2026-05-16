"""
Before/after retrieval comparison — filtered vs unfiltered.

Run:
  python scripts/test_filtered_retrieval.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from retrieval import embed_query, retrieve, retrieve_filtered, deduplicate_chunks

test_cases = [
    ("What is the standard warranty on electronics?", "product_info"),
    ("Can I convert from gold to premium member?",     "membership"),
]

print("\n" + "="*65)
print("DEDUPLICATION TEST — membership query")
print("="*65)
dedup_query = "Can I convert from gold to premium member?"
emb = embed_query(dedup_query)
chunks = retrieve_filtered(emb, "membership", top_k=10)
deduped, removed_log = deduplicate_chunks(chunks)

print(f"\nBefore dedup: {len(chunks)} chunks")
for c in chunks:
    print(f"  cosine={c['similarity']:.4f}  {c['doc_name']}  (chunk {c['chunk_index']})")

print(f"\nAfter dedup : {len(deduped)} chunks")
for c in deduped:
    print(f"  cosine={c['similarity']:.4f}  {c['doc_name']}  (chunk {c['chunk_index']})")

print(f"\nRemoved ({len(removed_log)} duplicates):")
for entry in removed_log:
    c  = entry["chunk"]
    d  = entry["duplicate_of"]
    print(f"  REMOVED  cosine={c['similarity']:.4f}  {c['doc_name']} chunk {c['chunk_index']}")
    print(f"    ↳ duplicate of chunk {d['chunk_index']}  jaccard={entry['jaccard']:.4f}")

print("\n" + "="*65)
print("FILTERED RETRIEVAL BEFORE/AFTER")
print("="*65)

for query, intent in test_cases:
    print(f"\n{'='*65}")
    print(f"Query  : {query}")
    print(f"Intent : {intent}")
    print(f"{'='*65}")

    emb = embed_query(query)

    unfiltered = retrieve(emb)
    print(f"\nWITHOUT filter ({len(unfiltered)} chunks):")
    for c in unfiltered:
        print(f"  {c['similarity']:.4f}  {c['doc_name']}  (chunk {c['chunk_index']})")

    filtered = retrieve_filtered(emb, intent)
    print(f"\nWITH filter — intent='{intent}' ({len(filtered)} chunks):")
    for c in filtered:
        print(f"  {c['similarity']:.4f}  {c['doc_name']}  (chunk {c['chunk_index']})")
