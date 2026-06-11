"""
rag/build_index.py — build the PGVector index from the corpus.

Pipeline:
  data/raw_messages.jsonl  ->  sentence-aware chunks  ->  local embeddings (Ollama/phi3.5)
  ->  entwin_chunks table  ->  IVFFlat ANN index

Incremental by default: only messages whose source_id isn't already indexed get added.
Use --rebuild to wipe and re-index everything (needed if you change the embedding model).

Run:
    python -m rag.build_index            # incremental
    python -m rag.build_index --rebuild  # wipe + full reindex
"""
import os, sys, json
from . import config, embeddings, chunking, db

CORPUS = os.environ.get("ENTWIN_CORPUS", "data/raw_messages.jsonl")

def load_corpus():
    if not os.path.exists(CORPUS):
        sys.exit(f"[error] corpus not found: {CORPUS}. Run the watcher/extract first.")
    rows = []
    for line in open(CORPUS, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def main():
    rebuild = "--rebuild" in sys.argv

    # 1. detect embedding dimension from the live model, and guard against model mismatch
    print(f"[rag] embedding model: {config.EMBED_MODEL} (local via Ollama)")
    dim = embeddings.detect_dim()
    print(f"[rag] detected embedding dimension: {dim}")

    db.init_schema(dim)
    prev_model = db.get_meta("embed_model")
    prev_dim = db.get_meta("embed_dim")
    if not rebuild and prev_dim and int(prev_dim) != dim:
        sys.exit(f"[error] index was built with dim {prev_dim} ({prev_model}) but current "
                 f"model gives dim {dim}. Re-run with --rebuild to switch embedding models.")

    if rebuild:
        print("[rag] --rebuild: clearing existing chunks")
        db.clear_all()
        db.init_schema(dim)

    # 2. figure out which messages still need indexing
    rows = load_corpus()
    done = set() if rebuild else db.existing_source_ids()
    todo = [r for r in rows if r.get("id") not in done]
    print(f"[rag] corpus: {len(rows)} messages | already indexed: {len(done)} | "
          f"to index: {len(todo)}")
    if not todo:
        print("[rag] nothing to index. Up to date.")
        db.create_ann_index()
        return

    # 3. chunk
    records = []
    for r in todo:
        records.extend(chunking.chunk_message(r))
    print(f"[rag] produced {len(records)} chunks (sentence-aware, "
          f"~{config.CHUNK_TARGET_WORDS}w target, {config.CHUNK_OVERLAP_WORDS}w overlap)")
    if not records:
        print("[rag] no usable chunks.")
        return

    # 4. embed locally, in order
    print(f"[rag] embedding {len(records)} chunks locally...")
    vecs = embeddings.embed_many([rec["text"] for rec in records])

    # 5. store + index
    n = db.insert_chunks(records, vecs)
    print(f"[rag] inserted {n} chunk vectors into PGVector")
    print("[rag] building IVFFlat ANN index...")
    db.create_ann_index()
    print("[rag] done. Index ready for retrieval.")

if __name__ == "__main__":
    main()
