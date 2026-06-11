"""
rag/respond.py — RAG-grounded response with the local SLM.

This is the RAG counterpart to rewrite.py. Where rewrite.py applies STYLE (the fine-tune),
this applies KNOWLEDGE + PRECEDENT (the RAG layer the Excel says to build first): it retrieves
relevant facts, past decisions, reasoning, and recipient tone, then asks the local SLM to draft
a response grounded in that precedent — so the twin doesn't hallucinate status or contradict
past decisions.

The Excel's intended end-state is BOTH together: RAG grounds WHAT is said, the fine-tune shapes
HOW it's said. See respond.py --then-restyle to chain into rewrite.py.

Usage:
    python -m rag.respond --query "Can you send Q3 numbers by Friday?" [--recipient ab12cd34]
    python -m rag.respond --query "..." --then-restyle   # pipe the draft through rewrite.py
"""
import sys, json, argparse, subprocess, os, urllib.request
from . import config, retrieve

SLM_MODEL = os.environ.get("ENTWIN_SLM_MODEL", "phi3.5")

SYSTEM = (
    "You are the user's digital twin, drafting a reply on their behalf. You are given "
    "RETRIEVED CONTEXT from the user's own past messages: facts, past decisions, prior "
    "reasoning, and how they speak to this recipient. Ground your reply in this context. "
    "Rules: (1) Never contradict a past decision shown in context. (2) Never invent facts, "
    "project status, names, or dates that aren't supported by the context — if unknown, say so "
    "or ask. (3) Match the tone shown for this recipient. Draft only the reply, no preamble.")

def build_prompt(query, context):
    return (f"RETRIEVED CONTEXT FROM PAST MESSAGES:\n{context}\n\n"
            f"NEW SITUATION TO RESPOND TO:\n{query}\n\n"
            f"Draft the user's reply, grounded in the context above.")

def call_slm(system, prompt):
    payload = {"model": SLM_MODEL, "system": system, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.6, "top_p": 0.9, "num_predict": 600}}
    req = urllib.request.Request(config.OLLAMA_URL + "/api/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read()).get("response", "").strip()
    except Exception as e:
        sys.exit(f"[error] SLM call failed ({SLM_MODEL} via Ollama): {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--recipient", default=None)
    ap.add_argument("--show-context", action="store_true")
    ap.add_argument("--then-restyle", action="store_true",
                    help="pass the grounded draft through scripts/rewrite.py for voice")
    args = ap.parse_args()

    res = retrieve.retrieve_context(args.query, recipient_hint=args.recipient)
    if args.show_context:
        print("=== RETRIEVED CONTEXT ===\n" + res["context"] + "\n=========================\n",
              file=sys.stderr)

    draft = call_slm(SYSTEM, build_prompt(args.query, res["context"]))

    if args.then_restyle:
        # chain: RAG grounds the content, rewrite.py applies the fine-tuned voice
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rw = os.path.join(here, "scripts", "rewrite.py")
        p = subprocess.run([sys.executable, rw, "--text", draft],
                           capture_output=True, text=True)
        print(p.stdout.strip() or draft)
    else:
        print(draft)

if __name__ == "__main__":
    main()
