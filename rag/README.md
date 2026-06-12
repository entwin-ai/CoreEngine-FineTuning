# Entwin RAG Layer — Setup & Usage

This is the **RAG layer** the Excel says to build *first* ("nail the RAG layer first … then
invest in fine-tuning last to varnish the voice"). It grounds **what** the twin says — facts,
past decisions, prior reasoning, recipient-specific tone — so the stylistic twin doesn't
"sound exactly like the user while saying things the user would never say."

It reuses your **locally installed SLM** (Phi-3.5 via Ollama) for embeddings, stores vectors in
**ChromaDB** (a local, embedded vector store — no server, no Postgres), and retrieves through the
lenses the Excel assigns to RAG.

---

## How the pillars map to retrieval (from the Excel CORE sheet)

| Pillar | RAG's job (per Excel) | How this code does it |
|---|---|---|
| **Knowledge & Context** | The underweighted foundation — build first | Semantic similarity over all chunks |
| **Decision & Judgment** | Retrieve **precedent** so the twin doesn't contradict past decisions | Similarity search filtered to `is_decision = True` |
| **Cognitive Style** | Retrieve **similar past reasoning** for novel situations | Semantic similarity (reasoning-bearing chunks) |
| **Affective Register** | RAG for **relationship-specific calibration** | Similarity filtered to the same `recipient_hint` |

The retriever runs all four lenses, dedupes, and assembles one context block (`rag/retrieve.py`).
`rag/respond.py` then asks the local SLM to draft a reply grounded in that block.

---

## Part 1 — Install ChromaDB (no database server!)

Chroma is an **embedded** vector database: it runs inside your Python process and stores data as
a folder on disk. There is **no PostgreSQL, no extension, no Docker, no service to start.** This
is why we switched to it — it sidesteps the entire native-database install problem on Windows.

### 1.1 SQLite — already included, nothing to install
Chroma uses SQLite (>= 3.35) under the hood, and **SQLite ships inside Python** — you do not
install it separately. Confirm your version is fine (in your 3.12 venv):
```powershell
.venv312\Scripts\Activate.ps1
python -c "import sqlite3; print('SQLite engine:', sqlite3.sqlite_version)"
```
Any value **3.35 or higher** is good (Python 3.12 ships ~3.45). If it were ever lower, the fix is
to use the 3.12 venv — not to install SQLite by hand.

### 1.2 Install Chroma
```powershell
python -m pip install --upgrade pip
python -m pip install -r rag\requirements.txt   # installs chromadb
```
Use `python -m pip` so it lands in *this* venv. Chroma installs from prebuilt wheels — no
compiler, unlike the PGVector route.

### 1.3 Verify it works (storage + search, fully local)
```powershell
python -c "import chromadb; c=chromadb.PersistentClient(path='./chroma_db'); col=c.get_or_create_collection('smoke'); col.add(ids=['a','b'], embeddings=[[0.1,0.2,0.3],[0.9,0.8,0.7]], documents=['hello','world']); print('CHROMA OK:', col.query(query_embeddings=[[0.1,0.2,0.3]], n_results=1)['documents'][0][0])"
```
`CHROMA OK: hello` means storage, metadata, and similarity search all work. A `./chroma_db`
folder now exists — that folder **is** the database (copy it to back up, delete it to reset).

---

## Part 2 — Where the database lives

The Chroma database is just a folder, by default `chroma_db/` in your project root. Override the
location with `ENTWIN_CHROMA_PATH` if you want it elsewhere:
```powershell
$env:ENTWIN_CHROMA_PATH = "C:\entwin\data\chroma_db"
```
No connection string, port, user, or password — there's no server to connect to.

---

## Part 3 — The local SLM for embeddings

You already have `phi3.5` in Ollama, which is the default embedder. **Strongly recommended
upgrade (still 100% local):** a purpose-built embedding model retrieves much better than a
generative one. Pull it once:
```powershell
ollama pull nomic-embed-text
$env:ENTWIN_EMBED_MODEL = "nomic-embed-text"
```
Either way, the embedding dimension is **auto-detected** at index-build time and stored in the
DB, so you don't configure it by hand. If you switch embedding models later, rebuild the index
(`--rebuild`) because vectors from different models aren't comparable.

Make sure Ollama is running (the desktop app, or `ollama serve`).

---

## Part 4 — Build the index

The RAG layer reads the same corpus your watcher produces: `data/raw_messages.jsonl`.
So ingest documents first (drop them in the LandingZone, let `watch.py` parse them), then:

```powershell
# from project root, venv active, Ollama running (no DB server needed)
python -m rag.build_index
```
You'll see it detect the embedding dimension, chunk the corpus (sentence-aware, ~180-word
chunks with 40-word overlap), embed locally, store vectors, and build the IVFFlat ANN index.

Re-run anytime after new files arrive — it's **incremental** (only new messages get indexed).
Use `python -m rag.build_index --rebuild` to wipe and start over (needed if you change the
embedding model).

### Multi-source, speaker-aware ingestion (email / writing / WhatsApp / transcripts)

The upstream parser keeps only the **person's own words**, which is what makes "write like the
person" safe:

- **Email, documents, writing** (`.eml/.docx/.pdf/.md/.txt`) → the author's prose.
- **WhatsApp exports** (`.txt`) → auto-detected; only the author's messages are kept.
- **Meeting transcripts** (`.txt` with `Speaker: ...` turns) and **captions** (`.vtt/.srt`) →
  auto-detected; **only the author's turns are ingested.** Indexing every voice in a transcript
  would teach the twin other people's words. Tell the parser who the author is:
  ```powershell
  $env:ENTWIN_MY_NAMES = "Nishit,Nishit Ghosh,Nishit K Ghosh"
  ```

So you can drop a Zoom/Teams transcript or WhatsApp export straight into the LandingZone and
trust that only your half of the conversation reaches the index.

---

## Part 5 — Use it

**Inspect what gets retrieved** for a situation:
```powershell
python -m rag.retrieve --query "Can you get me the Q3 numbers by Friday?" --show
# add --recipient ab12cd34 to enable per-recipient tone calibration
```

**Draft a grounded reply** with the local SLM:
```powershell
python -m rag.respond --query "Can you get me the Q3 numbers by Friday?" --show-context
```

**The full twin (RAG + voice together)** — RAG grounds the content, the fine-tune restyles it:
```powershell
python -m rag.respond --query "..." --then-restyle
```
This is the Excel's intended end-state: RAG decides **what** to say (grounded, no contradicted
decisions, no hallucinated status), the fine-tune shapes **how** it's said. Until you've
trained the adapter, `--then-restyle` falls back to the base model + style guardrail.

---

## Environment variables (all optional; sensible defaults)

| Var | Default | Meaning |
|---|---|---|
| `ENTWIN_CHROMA_PATH` | `<project>/chroma_db` | folder where the Chroma DB lives |
| `ENTWIN_EMBED_MODEL` | `phi3.5` | local embedding model (try `nomic-embed-text`) |
| `ENTWIN_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `ENTWIN_SLM_MODEL` | `phi3.5` | local model used for grounded generation |
| `ENTWIN_CHUNK_WORDS` / `ENTWIN_CHUNK_OVERLAP` | 180 / 40 | chunk sizing |
| `ENTWIN_MAX_CONTEXT_CHUNKS` | 12 | total chunks assembled into context |

---

## Files

```
rag/
  config.py        connection, embedding model, pillar->retrieval mapping
  embeddings.py    local embeddings via Ollama
  chunking.py      sentence-aware splitting + overlap + decision/recipient tagging
  db.py            ChromaDB storage, upsert, pillar-filtered similarity search
  build_index.py   corpus -> chunks -> embeddings -> ChromaDB
  retrieve.py      pillar-aware multi-lens retrieval, assembles context block
  respond.py       RAG-grounded generation with the local SLM (+ optional restyle)
  requirements.txt chromadb
```
