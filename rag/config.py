"""
rag/config.py — central configuration for the Entwin RAG layer.

Everything tunable lives here: DB connection, embedding model/dimension, chunking sizes,
and the pillar->retrieval mapping that comes straight from the Excel (CORE sheet).
"""
import os

# ---------------- PGVector / Postgres connection ----------------
PG = {
    "host": os.environ.get("ENTWIN_PG_HOST", "localhost"),
    "port": int(os.environ.get("ENTWIN_PG_PORT", "5432")),
    "dbname": os.environ.get("ENTWIN_PG_DB", "entwin"),
    "user": os.environ.get("ENTWIN_PG_USER", "entwin"),
    "password": os.environ.get("ENTWIN_PG_PASSWORD", "entwin"),
}
def pg_dsn():
    return (f"host={PG['host']} port={PG['port']} dbname={PG['dbname']} "
            f"user={PG['user']} password={PG['password']}")

# ---------------- Embeddings (LOCAL via Ollama) ----------------
# "Reuse the same locally installed SLM": default to phi3.5 via Ollama's /api/embeddings.
# NOTE: phi3.5 is a generative model; it CAN embed, but a purpose-built embedder retrieves
# noticeably better. If you want the upgrade (still 100% local), run:
#     ollama pull nomic-embed-text
# and set ENTWIN_EMBED_MODEL=nomic-embed-text. The dimension auto-detects on first embed.
EMBED_MODEL = os.environ.get("ENTWIN_EMBED_MODEL", "phi3.5")
OLLAMA_URL = os.environ.get("ENTWIN_OLLAMA_URL", "http://localhost:11434")
# Embedding dimension is detected at index-build time and stored in the meta table,
# so you don't have to hardcode it. These are common values for reference:
#   nomic-embed-text -> 768,  phi3.5 -> 3072,  mxbai-embed-large -> 1024
EMBED_DIM_FALLBACK = int(os.environ.get("ENTWIN_EMBED_DIM", "0")) or None

# ---------------- Chunking ----------------
# Messages are short; we chunk by a token-ish budget with overlap so retrieval units are
# coherent. Sentence-aware splitting keeps a thought intact rather than cutting mid-clause.
CHUNK_TARGET_WORDS = int(os.environ.get("ENTWIN_CHUNK_WORDS", "180"))
CHUNK_OVERLAP_WORDS = int(os.environ.get("ENTWIN_CHUNK_OVERLAP", "40"))
CHUNK_MIN_WORDS = 8

# ---------------- Pillar -> retrieval mapping (from the Excel CORE sheet) ----------------
# Each RAG-relevant pillar gets: which "lens" it retrieves through, and how many chunks.
# The Excel assigns RAG these jobs explicitly:
#   Cognitive Style      -> retrieve similar PAST REASONING when the situation is novel
#   Affective Register   -> RAG for RELATIONSHIP-SPECIFIC calibration (per recipient)
#   Decision & Judgment  -> retrieve PRECEDENT so the twin doesn't contradict past decisions
#   (Knowledge & Context -> the underweighted foundation the IMPORTANT note insists on first)
PILLAR_RETRIEVAL = {
    "knowledge_context": {
        "k": 6,
        "purpose": "Facts, project status, who's who — the foundation the Excel says to nail first.",
        "filter": None,
    },
    "decision_judgment": {
        "k": 5,
        "purpose": "Precedent decisions: how a similar yes/no/scope call was handled before.",
        "filter": {"is_decision": True},
    },
    "cognitive_style": {
        "k": 4,
        "purpose": "Past reasoning on analogous problems, for novel situations needing precedent.",
        "filter": None,
    },
    "affective_register": {
        "k": 4,
        "purpose": "Recipient-specific tone calibration: how warmth/directness shows up with them.",
        "filter": "by_recipient",   # special: filter to the same recipient_hint when known
    },
}

# Total budget assembled into the final context block handed to the SLM.
MAX_CONTEXT_CHUNKS = int(os.environ.get("ENTWIN_MAX_CONTEXT_CHUNKS", "12"))
