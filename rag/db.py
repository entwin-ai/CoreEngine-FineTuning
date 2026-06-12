"""
rag/db.py — ChromaDB storage + retrieval (replaces the PGVector backend).

Chroma is a local, embedded vector DB: no server, no Postgres, no extension. The database is
just a folder on disk (config.CHROMA_PATH). We keep the SAME function names the rest of the RAG
layer already calls (init_schema, get_meta, clear_all, existing_source_ids, insert_chunks,
create_ann_index, search) so build_index.py needs no changes.

Pillar filters: instead of SQL WHERE strings, search() takes a Chroma `where` dict
(e.g. {"is_decision": True} or {"recipient_hint": "ab12cd34"}). retrieve.py passes these.

We store our OWN embeddings (computed locally via Ollama in embeddings.py), so Chroma never
embeds anything itself — we always pass precomputed vectors.
"""
import chromadb
from . import config

_CLIENT = None
_COLLECTION = "entwin_chunks"

def connect():
    """Return a persistent Chroma client rooted at config.CHROMA_PATH (a folder on disk)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return _CLIENT

def _collection():
    client = connect()
    # cosine space matches how we score similarity; metadata holds pillar-filter fields.
    return client.get_or_create_collection(
        name=_COLLECTION, metadata={"hnsw:space": "cosine"})

def init_schema(dim):
    """Chroma has no fixed schema; the collection is created lazily. We just record the
    embedding model + dim in a tiny meta collection so a model-switch is detectable."""
    _collection()  # ensure it exists
    client = connect()
    meta = client.get_or_create_collection("entwin_meta")
    try:
        meta.upsert(ids=["meta"], embeddings=[[0.0]],
                    metadatas=[{"embed_model": config.EMBED_MODEL, "embed_dim": int(dim)}],
                    documents=["meta"])
    except Exception:
        meta.add(ids=["meta"], embeddings=[[0.0]],
                 metadatas=[{"embed_model": config.EMBED_MODEL, "embed_dim": int(dim)}],
                 documents=["meta"])

def get_meta(key):
    client = connect()
    try:
        meta = client.get_collection("entwin_meta")
        got = meta.get(ids=["meta"], include=["metadatas"])
        if got and got.get("metadatas"):
            return got["metadatas"][0].get(key)
    except Exception:
        return None
    return None

def create_ann_index(lists=100):
    """No-op for Chroma: it builds/maintains its HNSW index automatically on insert.
    Kept so build_index.py can call it unchanged."""
    return

def clear_all():
    """Drop and recreate the chunks collection."""
    client = connect()
    try:
        client.delete_collection(_COLLECTION)
    except Exception:
        pass
    _collection()

def existing_source_ids():
    """Return the set of source_ids already indexed (for incremental builds)."""
    col = _collection()
    try:
        got = col.get(include=["metadatas"])
    except Exception:
        return set()
    ids = set()
    for md in (got.get("metadatas") or []):
        if md and "source_id" in md:
            ids.add(md["source_id"])
    return ids

def insert_chunks(records, vectors):
    """records: chunk dicts from chunking.chunk_message; vectors: parallel precomputed embeddings."""
    col = _collection()
    ids, embs, docs, metas = [], [], [], []
    for rec, vec in zip(records, vectors):
        cid = f"{rec['source_id']}::{rec['chunk_ix']}"
        ids.append(cid)
        embs.append(list(vec))
        docs.append(rec["text"])
        metas.append({
            "source_id": rec["source_id"],
            "chunk_ix": rec["chunk_ix"],
            "source": rec.get("source", ""),
            "ts": int(rec.get("ts", 0) or 0),
            "recipient_hint": rec.get("recipient_hint", "") or "",
            "thread_id": rec.get("thread_id", "") or "",
            "is_decision": bool(rec.get("is_decision", False)),
        })
    B = 256  # batch inserts to stay under Chroma's max batch size on large corpora
    for i in range(0, len(ids), B):
        col.upsert(ids=ids[i:i+B], embeddings=embs[i:i+B],
                   documents=docs[i:i+B], metadatas=metas[i:i+B])
    return len(ids)

def search(query_vec, k=6, where=None):
    """Cosine similarity search with an optional Chroma metadata filter `where`
    (e.g. {"is_decision": True} or {"recipient_hint": "ab12cd34"}).
    Returns rows shaped like the old PGVector backend: text, metadata, similarity."""
    col = _collection()
    kwargs = {"query_embeddings": [list(query_vec)], "n_results": k,
              "include": ["documents", "metadatas", "distances"]}
    if where:
        kwargs["where"] = where
    try:
        res = col.query(**kwargs)
    except Exception:
        return []
    out = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for doc, md, dist in zip(docs, metas, dists):
        md = md or {}
        out.append({
            "text": doc,
            "source": md.get("source", ""),
            "ts": md.get("ts", 0),
            "recipient_hint": md.get("recipient_hint", ""),
            "thread_id": md.get("thread_id", ""),
            "is_decision": md.get("is_decision", False),
            "similarity": 1.0 - float(dist) if dist is not None else 0.0,
        })
    return out
