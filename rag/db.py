"""
rag/db.py — PGVector storage + retrieval.

Schema (one table + a meta table):
  entwin_meta(key text primary key, value text)         -- stores embed_model + embed_dim
  entwin_chunks(
      id bigserial primary key,
      source_id text, chunk_ix int,
      text text,
      source text, ts bigint, recipient_hint text, thread_id text,
      is_decision boolean,
      embedding vector(<dim>)
  )

Index: IVFFlat on embedding with cosine ops for fast ANN search.

Retrieval is pillar-aware: callers pass an optional WHERE filter (e.g. is_decision = true for
the Decision & Judgment pillar, or recipient_hint = '...' for Affective Register calibration).
"""
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from . import config

def connect():
    return psycopg2.connect(config.pg_dsn())

def init_schema(dim):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS entwin_meta(
                key text PRIMARY KEY, value text);""")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS entwin_chunks(
                id bigserial PRIMARY KEY,
                source_id text, chunk_ix int,
                text text NOT NULL,
                source text, ts bigint, recipient_hint text, thread_id text,
                is_decision boolean DEFAULT false,
                embedding vector({dim})
            );""")
        # store the embedding model + dim so re-runs detect mismatch
        cur.execute("""INSERT INTO entwin_meta(key,value) VALUES('embed_model',%s)
                       ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value;""",
                    (config.EMBED_MODEL,))
        cur.execute("""INSERT INTO entwin_meta(key,value) VALUES('embed_dim',%s)
                       ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value;""",
                    (str(dim),))
        conn.commit()

def get_meta(key):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT value FROM entwin_meta WHERE key=%s;", (key,))
        row = cur.fetchone()
        return row[0] if row else None

def create_ann_index(lists=100):
    """Build the IVFFlat index AFTER bulk insert (recommended order for IVFFlat)."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""CREATE INDEX IF NOT EXISTS entwin_chunks_emb_idx
                       ON entwin_chunks USING ivfflat (embedding vector_cosine_ops)
                       WITH (lists = %s);""", (lists,))
        cur.execute("CREATE INDEX IF NOT EXISTS entwin_chunks_decision_idx ON entwin_chunks(is_decision);")
        cur.execute("CREATE INDEX IF NOT EXISTS entwin_chunks_recipient_idx ON entwin_chunks(recipient_hint);")
        conn.commit()

def clear_all():
    with connect() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE entwin_chunks RESTART IDENTITY;")
        conn.commit()

def existing_source_ids():
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT source_id FROM entwin_chunks;")
        return {r[0] for r in cur.fetchall()}

def insert_chunks(records, vectors):
    """records: list of chunk dicts (from chunking.chunk_message); vectors: parallel embeddings."""
    rows = []
    for rec, vec in zip(records, vectors):
        rows.append((
            rec["source_id"], rec["chunk_ix"], rec["text"], rec["source"],
            rec["ts"], rec["recipient_hint"], rec["thread_id"], rec["is_decision"],
            _vec_literal(vec),
        ))
    with connect() as conn, conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO entwin_chunks
              (source_id, chunk_ix, text, source, ts, recipient_hint, thread_id, is_decision, embedding)
            VALUES %s;""", rows, template="(%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)")
        conn.commit()
    return len(rows)

def _vec_literal(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def search(query_vec, k=6, where_sql=None, where_params=None):
    """Cosine-similarity search with an optional pillar filter.
    Returns rows with text, metadata, and similarity (1 - cosine_distance)."""
    where = f"WHERE {where_sql}" if where_sql else ""
    sql = f"""
        SELECT text, source, ts, recipient_hint, thread_id, is_decision,
               1 - (embedding <=> %s::vector) AS similarity
        FROM entwin_chunks
        {where}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;"""
    # query vector appears in SELECT and ORDER BY; filter params come first in the WHERE
    qlit = _vec_literal(query_vec)
    params = [qlit] + list(where_params or []) + [qlit, k]
    with connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return cur.fetchall()
