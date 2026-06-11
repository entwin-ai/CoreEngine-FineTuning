"""
Step 6 — MERGE & SERVE
Merges the LoRA adapter into the base weights, then prepares for local inference via Ollama
on your ARM Surface (inference-only, no CUDA needed).

Run the merge on the same machine that trained (CUDA). Then convert to GGUF and pull the
.gguf to the Surface for Ollama.
"""
import json, torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

CFG = json.load(open("configs/qlora.json"))
MERGED = "out/entwin-merged"

def merge():
    tok = AutoTokenizer.from_pretrained(CFG["output_dir"])
    base = AutoModelForCausalLM.from_pretrained(CFG["base_model"], torch_dtype=torch.bfloat16,
                                                device_map="cpu")
    model = PeftModel.from_pretrained(base, CFG["output_dir"])
    model = model.merge_and_unload()
    os.makedirs(MERGED, exist_ok=True)
    model.save_pretrained(MERGED, safe_serialization=True)
    tok.save_pretrained(MERGED)
    print(f"merged model -> {MERGED}")

GGUF_AND_OLLAMA = """
# ---- Convert to GGUF (run once, on the training/cloud box) ----
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
pip install -r requirements.txt
python convert_hf_to_gguf.py ../out/entwin-merged --outfile ../out/entwin.gguf --outtype f16
# Optional quantize for the Surface (smaller, faster, ~no quality loss for chat):
./llama-quantize ../out/entwin.gguf ../out/entwin-q4_k_m.gguf q4_k_m

# ---- Pull entwin-q4_k_m.gguf to the Surface, then register with Ollama ----
# Create out/Modelfile:
#   FROM ./entwin-q4_k_m.gguf
#   SYSTEM \"\"\"<paste prompts/style_guardrail.md here as the voice guardrail>\"\"\"
#   PARAMETER temperature 0.8
#   PARAMETER top_p 0.9
#
# ollama create entwin -f out/Modelfile
# ollama run entwin
"""

def write_modelfile():
    guard = open("prompts/style_guardrail.md", encoding="utf-8", errors="replace").read() \
        if os.path.exists("prompts/style_guardrail.md") else "Write in the user's authentic voice."
    mf = (f"FROM ./entwin-q4_k_m.gguf\n"
          f'SYSTEM """{guard}\n\n'
          f'Convey only the meaning you are given. Never invent facts, names, dates, '
          f'or project status — those come from retrieval, not from you."""\n'
          f"PARAMETER temperature 0.8\nPARAMETER top_p 0.9\n")
    open("out/Modelfile", "w").write(mf)
    print("wrote out/Modelfile (system prompt = the fingerprint guardrail)")

if __name__ == "__main__":
    merge()
    write_modelfile()
    print(GGUF_AND_OLLAMA)
