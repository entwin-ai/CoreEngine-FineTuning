"""
rewrite.py — STYLE REWRITE (the payoff: use your trained twin)

Takes a paragraph and rewrites it in the EXACT style of the author whose data trained the SLM,
WITHOUT changing the meaning. This is the inference counterpart to the whole training pipeline:
it pairs the fine-tuned weights (which carry the voice) with the measured fingerprint guardrail
(prompts/style_guardrail.md, which holds the voice steady if the fine-tune drifts) — exactly
the two-layer design the project is built around.

Three backends, auto-selected (or forced with --backend):
  1. adapter  : base model + LoRA adapter from out/adapter/   (works right after train_qlora.py,
                no merge needed). Default if the adapter exists.
  2. merged   : the merged model from out/entwin-merged/      (after merge_serve.py).
  3. ollama   : the local `entwin` Ollama model               (after GGUF + ollama create).
                Best on your ARM Surface — no CUDA, no torch needed for inference.

Usage:
    # rewrite text given on the command line
    python scripts/rewrite.py --text "Paste the paragraph to restyle here."

    # rewrite text from a file
    python scripts/rewrite.py --infile draft.txt

    # pipe text in
    echo "some paragraph" | python scripts/rewrite.py

    # force a backend / tune sampling
    python scripts/rewrite.py --backend ollama --text "..." --temperature 0.7

The model is instructed to preserve meaning and invent no new facts — style only.
"""
import os, sys, json, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CFG = json.load(open(os.path.join(ROOT, "configs", "qlora.json")))
ADAPTER_DIR = os.path.join(ROOT, CFG.get("output_dir", "out/adapter"))
MERGED_DIR = os.path.join(ROOT, "out", "entwin-merged")
GUARDRAIL = os.path.join(ROOT, "prompts", "style_guardrail.md")
OLLAMA_MODEL = os.environ.get("ENTWIN_OLLAMA_SERVE_MODEL", "entwin")

SYSTEM_BASE = (
    "You are the author's digital twin. Rewrite the user's paragraph so it reads as if the "
    "author wrote it themselves: match their sentence rhythm, punctuation habits, idiolect, "
    "warmth, and structure. Preserve the meaning EXACTLY. Do not add, remove, or change any "
    "facts, names, numbers, or claims. Output only the rewritten paragraph — no preamble, no "
    "explanation, no quotes around it."
)

def _read_text_tolerant(path):
    """Read a text file even if it was written with a non-UTF-8 Windows encoding.
    Tries utf-8, utf-8-sig (BOM), then cp1252, then a lossy utf-8 as last resort."""
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()

def _system_prompt():
    """System = style instruction + the measured fingerprint guardrail (if present)."""
    sp = SYSTEM_BASE
    if os.path.exists(GUARDRAIL):
        sp += "\n\n# The author's measured style (follow as a guardrail):\n" + \
              _read_text_tolerant(GUARDRAIL)
    return sp

def _user_prompt(paragraph):
    return ("Rewrite the following paragraph in the author's exact style. Keep the meaning "
            f"identical; change only the voice and phrasing.\n\nPARAGRAPH:\n{paragraph}")

# ---------------- backend: transformers (adapter or merged) ----------------
def run_hf(paragraph, source, temperature, max_new_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if source == "adapter":
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)
        base = AutoModelForCausalLM.from_pretrained(
            CFG["base_model"], torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu")
        model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
    else:  # merged
        tok = AutoTokenizer.from_pretrained(MERGED_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_DIR, torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu").eval()

    msgs = [{"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_prompt(paragraph)}]
    ids = tok.apply_chat_template(msgs, return_tensors="pt",
                                  add_generation_prompt=True).to(model.device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                         temperature=max(temperature, 1e-5), top_p=0.9,
                         repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

# ---------------- backend: ollama ----------------
def run_ollama(paragraph, temperature, max_new_tokens):
    payload = {
        "model": OLLAMA_MODEL,
        "system": _system_prompt(),
        "prompt": _user_prompt(paragraph),
        "stream": False,
        "options": {"temperature": temperature, "top_p": 0.9, "num_predict": max_new_tokens},
    }
    # Prefer the HTTP API; fall back to the CLI if the server isn't reachable.
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/generate",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())["response"].strip()
    except Exception:
        full = _system_prompt() + "\n\n" + _user_prompt(paragraph)
        out = subprocess.run(["ollama", "run", OLLAMA_MODEL],
                             input=full.encode(), capture_output=True, timeout=180)
        return out.stdout.decode(errors="ignore").strip()

# ---------------- backend selection ----------------
def pick_backend():
    if os.path.isdir(ADAPTER_DIR) and os.listdir(ADAPTER_DIR):
        return "adapter"
    if os.path.isdir(MERGED_DIR) and os.listdir(MERGED_DIR):
        return "merged"
    return "ollama"

def read_input(args):
    if args.text:
        return args.text
    if args.infile:
        return open(args.infile, encoding="utf-8").read().strip()
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return data
    sys.exit("No input. Use --text \"...\", --infile path, or pipe text via stdin.")

def main():
    ap = argparse.ArgumentParser(description="Rewrite a paragraph in the trained author's style.")
    ap.add_argument("--text", help="paragraph to rewrite")
    ap.add_argument("--infile", help="file containing the paragraph to rewrite")
    ap.add_argument("--backend", choices=["adapter", "merged", "ollama", "auto"], default="auto")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="0 = deterministic, higher = more varied (default 0.8)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    paragraph = read_input(args)
    backend = pick_backend() if args.backend == "auto" else args.backend
    print(f"[rewrite] backend={backend}  temp={args.temperature}\n", file=sys.stderr)

    if backend in ("adapter", "merged"):
        result = run_hf(paragraph, backend, args.temperature, args.max_new_tokens)
    else:
        result = run_ollama(paragraph, args.temperature, args.max_new_tokens)

    print(result)

if __name__ == "__main__":
    main()
