# Entwin RAG Layer — Setup & Usage

This is the **RAG layer** the Excel says to build *first* ("nail the RAG layer first … then
invest in fine-tuning last to varnish the voice"). It grounds **what** the twin says — facts,
past decisions, prior reasoning, recipient-specific tone — so the stylistic twin doesn't
"sound exactly like the user while saying things the user would never say."

It reuses your **locally installed SLM** (Phi-3.5 via Ollama) for embeddings, stores vectors in
**PostgreSQL + PGVector**, and retrieves through the lenses the Excel assigns to RAG.

---

## How the pillars map to retrieval (from the Excel CORE sheet)

| Pillar | RAG's job (per Excel) | How this code does it |
|---|---|---|
| **Knowledge & Context** | The underweighted foundation — build first | Semantic similarity over all chunks |
| **Decision & Judgment** | Retrieve **precedent** so the twin doesn't contradict past decisions | Similarity search filtered to `is_decision = true` |
| **Cognitive Style** | Retrieve **similar past reasoning** for novel situations | Semantic similarity (reasoning-bearing chunks) |
| **Affective Register** | RAG for **relationship-specific calibration** | Similarity filtered to the same `recipient_hint` |

The retriever runs all four lenses, dedupes, and assembles one context block (`rag/retrieve.py`).
`rag/respond.py` then asks the local SLM to draft a reply grounded in that block.

---

## Part 1 — Install PostgreSQL + PGVector (Windows)

### 1.1 Install PostgreSQL
1. Download the **PostgreSQL 16** Windows installer from EDB:
   `https://www.enterprisedb.com/downloads/postgres-postgresql-downloads`
2. Run it. When prompted:
   - Set a **password** for the `postgres` superuser (remember it).
   - Keep the default **port 5432**.
   - Include **"Command Line Tools"** and **"pgAdmin 4"** (handy GUI).
3. After install, add PostgreSQL's `bin` to PATH (so `psql` works in PowerShell), e.g.:
   ```powershell
   $env:Path += ";C:\Program Files\PostgreSQL\16\bin"
   ```
   (Add it permanently via System → Environment Variables for future sessions.)

### 1.2 Install the PGVector extension
PGVector isn't bundled with the Windows installer, so install the prebuilt binary:

**Easiest path — StackBuilder / prebuilt DLL:**
1. Download the PGVector Windows release matching your PG major version from
   `https://github.com/pgvector/pgvector/releases` (look for a Windows `.zip`, e.g.
   `vector-vX.X.X-pg16-windows-x64.zip`).
2. Unzip. Copy the files into your PostgreSQL install:
   - `vector.dll` → `C:\Program Files\PostgreSQL\16\lib\`
   - `vector.control` and `vector--*.sql` → `C:\Program Files\PostgreSQL\16\share\extension\`
3. Restart the PostgreSQL service:
   ```powershell
   Restart-Service postgresql-x64-16
   ```

> If no prebuilt binary exists for your version, the alternative is building from source with
> MSVC + `nmake` (the pgvector README documents this), or running Postgres+pgvector via Docker
> (`docker run -e POSTGRES_PASSWORD=entwin -p 5432:5432 pgvector/pgvector:pg16`). Docker is the
> least painful if the DLL route gives trouble.

### 1.3 Create the database and user
Open `psql` as the `postgres` superuser:
```powershell
psql -U postgres
```
Then run:
```sql
CREATE USER entwin WITH PASSWORD 'entwin';
CREATE DATABASE entwin OWNER entwin;
\c entwin
CREATE EXTENSION vector;     -- enables PGVector in this DB
\q
```
If `CREATE EXTENSION vector;` succeeds, PGVector is installed correctly. (The build script
also runs `CREATE EXTENSION IF NOT EXISTS vector;`, so this is just a confirmation.)

---

## Part 2 — Python frameworks

From your project root, in the **3.12 venv** you already use:
```powershell
.venv312\Scripts\Activate.ps1
python -m pip install -r rag\requirements.txt
```
That installs `psycopg2-binary` (the Postgres driver). Embeddings and generation go through
Ollama over HTTP, so there's no heavyweight ML dependency here — keeps it light.

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
# from project root, venv active, Ollama running, Postgres running
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
| `ENTWIN_PG_HOST/PORT/DB/USER/PASSWORD` | localhost/5432/entwin/entwin/entwin | Postgres connection |
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
  db.py            PGVector schema, upsert, pillar-filtered similarity search
  build_index.py   corpus -> chunks -> embeddings -> PGVector -> ANN index
  retrieve.py      pillar-aware multi-lens retrieval, assembles context block
  respond.py       RAG-grounded generation with the local SLM (+ optional restyle)
  requirements.txt psycopg2-binary
```
