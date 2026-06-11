"""
rag/chunking.py — sentence-aware splitting with overlap, plus light pillar tagging.

"Proper splitting": we don't cut mid-sentence. We accumulate sentences up to a word budget,
then start a new chunk carrying a small overlap (the trailing sentences of the previous chunk)
so a thought that straddles a boundary is retrievable from either side.

Each chunk also gets cheap metadata used by the pillar-specific retrieval filters:
  - is_decision: does this text contain decision/commitment language? (Decision & Judgment)
  - recipient_hint: carried from the source message (Affective Register recipient calibration)
"""
import re
from . import config

_SENT = re.compile(r"(?<=[.!?])\s+")
_DECISION = re.compile(
    r"\b(approve|approved|reject|decline|let'?s (go|do|punt|hold)|decision|i'?ll commit|"
    r"deadline|budget|hire|fire|sign ?off|green ?light|no-go|prioriti[sz]e|"
    r"we (should|will|won'?t|can'?t)|going with|let'?s not|i (decided|chose))\b", re.I)

def split_sentences(text):
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]

def chunk_text(text):
    """Yield (chunk_text) units, sentence-aware, word-budgeted, with overlap."""
    sents = split_sentences(text)
    if not sents:
        return []
    chunks, cur, cur_words = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur_words + w > config.CHUNK_TARGET_WORDS and cur:
            chunks.append(" ".join(cur))
            # build overlap: trailing sentences up to CHUNK_OVERLAP_WORDS
            ov, ov_words = [], 0
            for prev in reversed(cur):
                pw = len(prev.split())
                if ov_words + pw > config.CHUNK_OVERLAP_WORDS:
                    break
                ov.insert(0, prev); ov_words += pw
            cur, cur_words = list(ov), ov_words
        cur.append(s); cur_words += w
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c.split()) >= config.CHUNK_MIN_WORDS]

def chunk_message(row):
    """Turn one corpus message (dict) into chunk records ready for indexing."""
    out = []
    for j, ctext in enumerate(chunk_text(row.get("text", ""))):
        out.append({
            "source_id": row.get("id", ""),
            "chunk_ix": j,
            "text": ctext,
            "source": row.get("source", ""),
            "ts": int(row.get("ts", 0) or 0),
            "recipient_hint": row.get("recipient_hint", ""),
            "thread_id": row.get("thread_id", ""),
            "is_decision": bool(_DECISION.search(ctext)),
        })
    return out
