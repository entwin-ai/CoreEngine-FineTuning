"""
Step 2 — PROFILE (the Linguistic Fingerprint measurement)
The Excel says: fine-tuning learns the fingerprint by mimicry, but the SYSTEM PROMPT must
hold a STATISTICAL SUMMARY "as a guardrail ... for when the fine-tune drifts."

This script measures that profile from the user's real corpus and emits:
  data/fingerprint.json          -> machine-readable, used by evaluate.py to detect drift
  prompts/style_guardrail.md     -> human-readable guardrail to paste into the system prompt

We measure EXACTLY the surface dimensions the Excel lists for the Linguistic Fingerprint pillar.
"""
import json, re, statistics as st
from collections import Counter

RAW = "data/raw_messages.jsonl"

def load():
    return [json.loads(l) for l in open(RAW, encoding="utf-8")]

def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def measure(rows):
    texts = [r["text"] for r in rows]
    blob = "\n".join(texts)
    words = re.findall(r"\b[\w']+\b", blob)
    n_words = max(len(words), 1)
    sents = [s for t in texts for s in sentences(t)]
    sent_lens = [len(s.split()) for s in sents] or [0]

    # contractions
    contractions = len(re.findall(r"\b\w+'(t|re|ll|ve|s|d|m)\b", blob, re.I))

    # punctuation habits
    def per1k(pat):
        return round(1000 * len(re.findall(pat, blob)) / n_words, 2)

    # active vs passive (cheap heuristic: 'was/were/been + past participle')
    passive = len(re.findall(r"\b(was|were|been|being|is|are)\s+\w+ed\b", blob, re.I))

    # prose vs bullets
    bullet_lines = len(re.findall(r"(?m)^\s*[-*•]\s+", blob))
    total_lines = max(blob.count("\n"), 1)

    # questions vs statements
    q = blob.count("?")
    terminal = blob.count(".") + blob.count("!") + q

    # vocabulary register: ratio of long (Latinate-ish) words
    long_words = sum(1 for w in words if len(w) >= 9)

    fp = {
        "n_messages": len(rows),
        "n_words": n_words,
        "sentence_length": {
            "mean": round(st.mean(sent_lens), 1),
            "median": st.median(sent_lens),
            "stdev": round(st.pstdev(sent_lens), 1),
            "p90": sorted(sent_lens)[int(0.9 * (len(sent_lens) - 1))],
        },
        "contraction_rate_per_1k_words": round(1000 * contractions / n_words, 2),
        "punctuation_per_1k_words": {
            "em_dash": per1k(r"—|--"),
            "semicolon": per1k(r";"),
            "ellipsis": per1k(r"\.\.\.|…"),
            "exclamation": per1k(r"!"),
            "comma": per1k(r","),
        },
        "passive_voice_ratio": round(passive / max(len(sents), 1), 3),
        "bullet_line_fraction": round(bullet_lines / total_lines, 3),
        "question_to_terminal_ratio": round(q / max(terminal, 1), 3),
        "long_word_fraction": round(long_words / n_words, 3),
        "avg_paragraph_words": round(n_words / max(len([t for t in texts]), 1), 1),
        "top_openings": _top_ngrams([t for t in texts], where="start"),
        "top_closings": _top_ngrams([t for t in texts], where="end"),
        "signature_phrases": _idiolect(blob),
    }
    return fp

def _top_ngrams(texts, where, n=15):
    c = Counter()
    for t in texts:
        toks = t.split()
        if len(toks) < 3:
            continue
        seg = toks[:6] if where == "start" else toks[-6:]
        c[" ".join(seg).lower()] += 1
    return [{"phrase": p, "count": k} for p, k in c.most_common(n) if k > 1]

def _idiolect(blob):
    # recurring multi-word patterns = Voice & Identity signal worth preserving verbatim
    c = Counter()
    toks = re.findall(r"\b[\w']+\b", blob.lower())
    for size in (2, 3):
        for i in range(len(toks) - size):
            c[" ".join(toks[i:i + size])] += 1
    # filter generic function-word grams
    stop = {"of the", "in the", "to the", "for the", "i am", "and the", "this is", "i will"}
    ranked = [(p, k) for p, k in c.most_common(80) if p not in stop and k >= 5]
    return [{"phrase": p, "count": k} for p, k in ranked[:25]]

def to_guardrail_md(fp):
    p = fp["punctuation_per_1k_words"]
    sl = fp["sentence_length"]
    def freq(v, hi, lo):  return "liberally" if v >= hi else ("rarely" if v <= lo else "occasionally")
    lines = [
        "# Entwin — Linguistic Fingerprint Guardrail",
        "_Statistical summary of the user's writing. Inject into the system prompt so the",
        "fine-tuned model has a hard reference if its voice drifts. Do NOT over-apply as rigid",
        "rules — the fine-tune carries the mimicry; this is the safety rail._\n",
        f"- Sentence length: averages **{sl['mean']} words** (median {sl['median']}, "
        f"varies up to ~{sl['p90']}). Mix short and long; avoid uniform length.",
        f"- Em-dashes: uses **{freq(p['em_dash'],3,0.5)}** "
        f"(~{p['em_dash']}/1k words).",
        f"- Semicolons: uses **{freq(p['semicolon'],2,0.3)}** (~{p['semicolon']}/1k words).",
        f"- Exclamation marks: uses **{freq(p['exclamation'],3,0.5)}** "
        f"(~{p['exclamation']}/1k words).",
        f"- Ellipses: uses **{freq(p['ellipsis'],1.5,0.2)}** (~{p['ellipsis']}/1k words).",
        f"- Contractions: **{freq(fp['contraction_rate_per_1k_words'],15,4)}** "
        f"({fp['contraction_rate_per_1k_words']}/1k words) — "
        f"{'casual register' if fp['contraction_rate_per_1k_words']>=15 else 'formal register'}.",
        f"- Voice: {'leans active' if fp['passive_voice_ratio']<0.15 else 'notable passive usage'} "
        f"(passive ratio {fp['passive_voice_ratio']}).",
        f"- Structure: {'prose-first' if fp['bullet_line_fraction']<0.1 else 'uses bullets readily'} "
        f"(bullet fraction {fp['bullet_line_fraction']}).",
        f"- Register: long-word fraction {fp['long_word_fraction']} "
        f"({'Latinate/formal' if fp['long_word_fraction']>0.18 else 'plain/Anglo-Saxon'}).",
        "",
        "## Preserve these signature phrases verbatim where natural (Voice & Identity):",
    ]
    for s in fp["signature_phrases"][:12]:
        lines.append(f'- "{s["phrase"]}"  (×{s["count"]})')
    lines.append("\n## Common openings:")
    for o in fp["top_openings"][:6]:
        lines.append(f'- "{o["phrase"]}"')
    lines.append("\n## Common closings:")
    for o in fp["top_closings"][:6]:
        lines.append(f'- "{o["phrase"]}"')
    return "\n".join(lines)

def main():
    import os; os.makedirs("prompts", exist_ok=True); os.makedirs("data", exist_ok=True)
    rows = load()
    fp = measure(rows)
    json.dump(fp, open("data/fingerprint.json", "w"), indent=2, ensure_ascii=False)
    open("prompts/style_guardrail.md", "w").write(to_guardrail_md(fp))
    print("wrote data/fingerprint.json and prompts/style_guardrail.md")
    print(json.dumps({k: fp[k] for k in
          ["n_messages", "sentence_length", "punctuation_per_1k_words",
           "contraction_rate_per_1k_words"]}, indent=2))

if __name__ == "__main__":
    main()
