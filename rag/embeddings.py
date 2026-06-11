"""
rag/embeddings.py — LOCAL embeddings via Ollama (reuses the locally installed SLM).

Calls Ollama's /api/embeddings endpoint, so nothing leaves the machine. Works with the
generative phi3.5 (default) or a purpose-built local embedder like nomic-embed-text.

Exposes:
    embed_one(text) -> list[float]
    embed_many(texts) -> list[list[float]]   (sequential; Ollama embeds one at a time)
    detect_dim() -> int                       (embeds a probe string to learn the vector size)
"""
import json, urllib.request, time
from . import config

def _post(path, payload, timeout=120):
    req = urllib.request.Request(
        config.OLLAMA_URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def _check_ollama():
    try:
        urllib.request.urlopen(config.OLLAMA_URL + "/api/tags", timeout=10)
    except Exception:
        raise SystemExit(
            f"[error] Can't reach Ollama at {config.OLLAMA_URL}. "
            f"Start it (`ollama serve` or the Ollama app), and make sure model "
            f"'{config.EMBED_MODEL}' is pulled (`ollama pull {config.EMBED_MODEL}`).")

def embed_one(text):
    text = (text or "").strip()
    if not text:
        text = " "
    # Ollama supports both /api/embeddings (single) and /api/embed (batch in newer versions).
    try:
        resp = _post("/api/embeddings", {"model": config.EMBED_MODEL, "prompt": text})
        vec = resp.get("embedding")
        if vec:
            return vec
    except Exception:
        pass
    # newer endpoint shape
    resp = _post("/api/embed", {"model": config.EMBED_MODEL, "input": text})
    embs = resp.get("embeddings") or resp.get("embedding")
    if embs and isinstance(embs[0], list):
        return embs[0]
    return embs

def embed_many(texts, progress_every=50):
    _check_ollama()
    out = []
    t0 = time.time()
    for i, t in enumerate(texts):
        out.append(embed_one(t))
        if progress_every and (i + 1) % progress_every == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"  embedded {i+1}/{len(texts)} ({rate:.1f}/s)")
    return out

def detect_dim():
    _check_ollama()
    v = embed_one("dimension probe")
    return len(v)
