"""
rag/retrieve.py — pillar-aware retrieval (the heart of the RAG layer).

Given a new situation (the draft/prompt you want the twin to respond to, plus optional
recipient), this retrieves through the lenses the Excel assigns to RAG and assembles a single
context block to hand the local SLM:

  * knowledge_context  — general semantically-similar precedent (facts/status/who's-who)
  * decision_judgment  — past DECISIONS on similar matters (filter is_decision=true)
  * cognitive_style    — past REASONING on analogous problems
  * affective_register — same-recipient messages for tone calibration (when recipient known)

De-duplicates across lenses, trims to MAX_CONTEXT_CHUNKS, and formats with provenance so the
SLM (and you) can see WHY each snippet was retrieved.

Usage (library):
    from rag.retrieve import retrieve_context
    ctx = retrieve_context("Can you get the Q3 numbers to me by Friday?", recipient_hint="ab12cd34")

Usage (CLI):
    python -m rag.retrieve --query "..." [--recipient ab12cd34] [--show]
"""
import sys, argparse, datetime
from . import config, embeddings, db

def _fmt_ts(ts):
    if not ts:
        return "unknown date"
    try:
        return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return "unknown date"

def _dedupe(rows):
    seen, out = set(), []
    for r in rows:
        key = r["text"][:160]
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def retrieve_context(query, recipient_hint=None, max_chunks=None):
    """Return a dict with the assembled context string and the raw per-lens hits."""
    max_chunks = max_chunks or config.MAX_CONTEXT_CHUNKS
    qvec = embeddings.embed_one(query)

    lenses = {}

    # --- knowledge_context: plain semantic similarity ---
    spec = config.PILLAR_RETRIEVAL["knowledge_context"]
    lenses["knowledge_context"] = db.search(qvec, k=spec["k"])

    # --- decision_judgment: precedent decisions only ---
    spec = config.PILLAR_RETRIEVAL["decision_judgment"]
    lenses["decision_judgment"] = db.search(qvec, k=spec["k"],
                                            where={"is_decision": True})

    # --- cognitive_style: reasoning on analogous problems (semantic, longer chunks) ---
    spec = config.PILLAR_RETRIEVAL["cognitive_style"]
    lenses["cognitive_style"] = db.search(qvec, k=spec["k"])

    # --- affective_register: same-recipient calibration, only if recipient is known ---
    spec = config.PILLAR_RETRIEVAL["affective_register"]
    if recipient_hint:
        lenses["affective_register"] = db.search(
            qvec, k=spec["k"], where={"recipient_hint": recipient_hint})
    else:
        lenses["affective_register"] = []

    # assemble: keep lens provenance, dedupe across lenses, trim to budget
    blocks, used, total = [], set(), 0
    lens_titles = {
        "knowledge_context": "KNOWLEDGE & CONTEXT (facts / status / precedent)",
        "decision_judgment": "PAST DECISIONS (don't contradict these)",
        "cognitive_style": "PAST REASONING (how similar problems were thought through)",
        "affective_register": "TONE WITH THIS RECIPIENT (match the warmth/directness)",
    }
    for lens, rows in lenses.items():
        rows = _dedupe(rows)
        kept = []
        for r in rows:
            if total >= max_chunks:
                break
            key = r["text"][:160]
            if key in used:
                continue
            used.add(key); total += 1
            kept.append(r)
        if kept:
            lines = [f"## {lens_titles[lens]}"]
            for r in kept:
                sim = r.get("similarity", 0) or 0
                lines.append(f"- ({_fmt_ts(r['ts'])}, sim {sim:.2f}) {r['text']}")
            blocks.append("\n".join(lines))

    context = "\n\n".join(blocks) if blocks else "(no relevant precedent found)"
    return {"context": context, "lenses": lenses, "n_used": total}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--recipient", default=None, help="recipient_hint to calibrate tone")
    ap.add_argument("--show", action="store_true", help="print the full assembled context")
    args = ap.parse_args()
    res = retrieve_context(args.query, recipient_hint=args.recipient)
    print(f"[retrieve] assembled {res['n_used']} chunks across lenses\n")
    if args.show:
        print(res["context"])

if __name__ == "__main__":
    main()
