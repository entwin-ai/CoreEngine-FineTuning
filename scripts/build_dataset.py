"""
Step 3 — BUILD DATASET
Turns authored messages into instruction->response pairs that teach STYLE, not FACTS.

The central design decision (straight from the Excel): fine-tuning owns the PRIMARY pillars
(Linguistic Fingerprint, Voice & Identity) and the CADENCE of the secondary pillars
(Cognitive Style, Affective Register, Meta-cognitive). It must NOT memorize knowledge,
project status, or specific decisions — those belong to RAG, and the Excel explicitly marks
Decision & Judgment as the LEAST reliable thing to fine-tune ("out-of-distribution").

Two complementary training formats are produced and mixed:

  A) STYLE-TRANSFER pairs (primary): a neutral paraphrase of the user's message is the
     INPUT; the user's real message is the TARGET. The model learns the mapping
     neutral-meaning -> user's-voice. Because the input already contains the content,
     the model is rewarded for STYLE, not for inventing facts. This is the cleanest way to
     isolate the fingerprint + voice.

  B) CONTINUATION pairs (secondary): given the opening of the user's message, complete it.
     Teaches openings/closings, argument cadence, and meta-cognitive phrasing.

Decision-heavy messages are DOWN-WEIGHTED (kept for cadence, but flagged) so the model
doesn't learn to assert stale judgments.

Paraphrase generation uses a local/cheap model (Ollama or the Anthropic API) — swap as needed.
"""
import json, os, re, random, subprocess
random.seed(7)

RAW = "data/raw_messages.jsonl"
TRAIN, VAL = "data/train.jsonl", "data/val.jsonl"

SYSTEM = ("You are Entwin, the user's digital twin. Write in the user's authentic voice: "
          "match their sentence rhythm, punctuation habits, idiolect, warmth, and structure. "
          "Convey only the meaning given to you — never invent facts, names, dates, or status.")

DECISION_MARKERS = re.compile(
    r"\b(approve|approved|reject|decline|let'?s (go|do|punt|hold)|decision|i'?ll commit|"
    r"deadline|budget|hire|fire|sign off|green ?light|no-go|prioriti[sz]e)\b", re.I)

# ---------------- paraphraser backends ----------------
def paraphrase_ollama(text, model="qwen2.5:7b-instruct"):
    prompt = ("Rewrite the message below in plain, neutral, voice-less English. Keep ALL "
              "facts, names, and intent identical. Strip personality, idiom, and stylistic "
              "flourish. Output only the rewrite.\n\nMESSAGE:\n" + text)
    try:
        out = subprocess.run(["ollama", "run", model],
                             input=prompt.encode(), capture_output=True, timeout=120)
        r = out.stdout.decode(errors="ignore").strip()
        return r or None
    except Exception:
        return None

def paraphrase_stub(text):
    # Deterministic fallback so the pipeline runs without a model:
    # crude neutralization (collapse em-dashes, expand a couple contractions).
    t = re.sub(r"\s*—\s*", ", ", text)
    t = re.sub(r";", ".", t)
    t = re.sub(r"\.\.\.|…", ".", t)
    return "Convey this: " + t.strip()

PARAPHRASE = paraphrase_ollama if os.environ.get("ENTWIN_USE_OLLAMA") else paraphrase_stub

# ---------------- builders ----------------
def style_transfer_pair(text):
    neutral = PARAPHRASE(text) or paraphrase_stub(text)
    instr = ("Express the following in the user's own voice and style. Keep the meaning "
             f"exactly; add no new facts.\n\nMeaning to convey:\n{neutral}")
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instr},
        {"role": "assistant", "content": text}]}

def continuation_pair(text):
    toks = text.split()
    cut = max(4, int(len(toks) * random.uniform(0.2, 0.4)))
    opening = " ".join(toks[:cut])
    instr = ("Continue this message in the user's natural voice, completing the thought "
             f"as they would:\n\n{opening}")
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instr},
        {"role": "assistant", "content": " ".join(toks[cut:])}]}

def weight_for(text):
    # Down-weight decision content (Excel: tertiary/unreliable for fine-tune).
    return 0.3 if DECISION_MARKERS.search(text) else 1.0

def main():
    rows = [json.loads(l) for l in open(RAW, encoding="utf-8")]
    random.shuffle(rows)
    samples = []
    for r in rows:
        t = r["text"].strip()
        w = weight_for(t)
        # weighted sampling: keep all full-weight, subsample decision-heavy ones
        if w < 1.0 and random.random() > w:
            continue
        samples.append(style_transfer_pair(t))
        if len(t.split()) >= 25:          # only longer msgs make good continuations
            samples.append(continuation_pair(t))
    random.shuffle(samples)
    n_val = max(20, int(0.05 * len(samples)))
    val, train = samples[:n_val], samples[n_val:]
    _dump(TRAIN, train); _dump(VAL, val)
    print(f"train={len(train)}  val={len(val)}  (from {len(rows)} messages)")
    print("NOTE: style-transfer pairs teach voice without facts; "
          "decision content down-weighted to 0.3 per the pillar guidance.")

def _dump(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
