"""
Step 5 — EVALUATE (fingerprint drift detection)
The Excel's whole reason for keeping the statistical fingerprint in the system prompt is to
catch "when the fine-tune drifts." This script closes that loop: it generates from the
fine-tuned model on held-out prompts, measures the SAME fingerprint dimensions as profile.py,
and reports how far each dimension drifted from the user's true profile.

It also runs a "fact-leak" probe: prompts with NO factual content should not produce specific
names/dates/numbers. If they do, the fine-tune memorized content it shouldn't have (a sign
LoRA rank/epochs are too high) — exactly the failure mode the Excel warns about.
"""
import json, re, statistics as st, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

CFG = json.load(open("configs/qlora.json"))
TRUE_FP = json.load(open("data/fingerprint.json"))

EVAL_PROMPTS = [
    "Express the following in the user's own voice: let the team know the review is pushed to next week.",
    "Continue this message in the user's natural voice: Thanks for sending that over —",
    "Express in the user's voice: politely decline a meeting request that has no clear agenda.",
    "Express in the user's voice: acknowledge you made a mistake on a shared document and will fix it.",
    "Express in the user's voice: agree with a proposal but flag one concern.",
    "Continue in the user's voice: Quick thought on the architecture —",
]

def gen(model, tok, prompt, n=3):
    out = []
    for _ in range(n):
        msgs = [{"role": "system", "content": "You are the user's digital twin. Write in their voice."},
                {"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, return_tensors="pt",
                                      add_generation_prompt=True).to(model.device)
        y = model.generate(ids, max_new_tokens=180, do_sample=True, temperature=0.8,
                           top_p=0.9, pad_token_id=tok.eos_token_id)
        out.append(tok.decode(y[0][ids.shape[1]:], skip_special_tokens=True).strip())
    return out

def measure_gen(texts):
    blob = "\n".join(texts)
    words = re.findall(r"\b[\w']+\b", blob); n = max(len(words), 1)
    sents = [s for t in texts for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    sl = [len(s.split()) for s in sents] or [0]
    per1k = lambda p: round(1000 * len(re.findall(p, blob)) / n, 2)
    return {
        "sent_mean": round(st.mean(sl), 1),
        "em_dash": per1k(r"—|--"), "semicolon": per1k(r";"),
        "exclamation": per1k(r"!"), "ellipsis": per1k(r"\.\.\.|…"),
        "contraction": round(1000 * len(re.findall(r"\b\w+'(t|re|ll|ve|s|d|m)\b", blob)) / n, 2),
        "long_word_frac": round(sum(1 for w in words if len(w) >= 9) / n, 3),
    }

def drift_report(gen_fp):
    t = TRUE_FP
    target = {
        "sent_mean": t["sentence_length"]["mean"],
        "em_dash": t["punctuation_per_1k_words"]["em_dash"],
        "semicolon": t["punctuation_per_1k_words"]["semicolon"],
        "exclamation": t["punctuation_per_1k_words"]["exclamation"],
        "ellipsis": t["punctuation_per_1k_words"]["ellipsis"],
        "contraction": t["contraction_rate_per_1k_words"],
        "long_word_frac": t["long_word_fraction"],
    }
    rep = {}
    for k, tv in target.items():
        gv = gen_fp[k]
        denom = abs(tv) if abs(tv) > 1e-6 else 1.0
        rel = round(abs(gv - tv) / denom, 2)
        rep[k] = {"target": tv, "generated": gv, "rel_drift": rel,
                  "flag": "DRIFT" if rel > 0.5 else "ok"}
    return rep

FACT_PAT = re.compile(r"\b(\d{4}|\$\d|Q[1-4]\b|[A-Z][a-z]+ \d{1,2})\b")
def fact_leak(texts):
    # crude: count specific dates/money/quarters that should NOT appear from content-free prompts
    hits = sum(len(FACT_PAT.findall(t)) for t in texts)
    return {"specific_tokens": hits,
            "flag": "LEAK" if hits > len(texts) else "ok"}

def main():
    tok = AutoTokenizer.from_pretrained(CFG["output_dir"])
    base = AutoModelForCausalLM.from_pretrained(CFG["base_model"], device_map="auto",
                                                torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, CFG["output_dir"]).eval()

    all_gen = []
    for p in EVAL_PROMPTS:
        all_gen += gen(model, tok, p)
    gen_fp = measure_gen(all_gen)
    report = {
        "generated_fingerprint": gen_fp,
        "drift_vs_true_profile": drift_report(gen_fp),
        "fact_leak_probe": fact_leak(all_gen),
        "samples": all_gen[:6],
    }
    json.dump(report, open("out/eval_report.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(report["drift_vs_true_profile"], indent=2))
    print("fact_leak:", report["fact_leak_probe"])
    flags = [k for k, v in report["drift_vs_true_profile"].items() if v["flag"] != "ok"]
    if flags:
        print(f"\n⚠ drifted dimensions: {flags} -> lean on prompts/style_guardrail.md "
              f"in the system prompt, or retrain with adjusted data balance.")
    if report["fact_leak_probe"]["flag"] == "LEAK":
        print("⚠ fact leak detected -> lower lora_r or num_train_epochs; the model is "
              "memorizing content that belongs in RAG.")

if __name__ == "__main__":
    main()
