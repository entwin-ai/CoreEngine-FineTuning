"""
Step 4 — TRAIN (QLoRA)
Fine-tunes the base model on the style-transfer dataset. Key choices for STYLE fine-tuning:
  - 4-bit NF4 quant + LoRA r=16  -> fits a 7-8B model on a single 24GB GPU.
  - train_on_responses_only      -> loss only on the assistant turn (the user's real voice),
                                     never on the instruction. This is what teaches voice
                                     rather than instruction-following.
  - low rank + few epochs        -> capture the thin "style manifold" without memorizing facts
                                     (facts belong to RAG, per the Excel build-order warning).

Run on CUDA. On your ARM Surface, run this step on a rented GPU, then pull the adapter back
and serve locally via Ollama (see merge_serve.py).
"""
import json, torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from trl import DataCollatorForCompletionOnlyLM

CFG = json.load(open("configs/qlora.json"))

def main():
    tok = AutoTokenizer.from_pretrained(CFG["base_model"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=CFG["load_in_4bit"],
        bnb_4bit_quant_type=CFG["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, CFG["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=CFG["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        CFG["base_model"], quantization_config=bnb, device_map="auto",
        torch_dtype=getattr(torch, CFG["bnb_4bit_compute_dtype"]))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora = LoraConfig(
        r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"],
        target_modules=CFG["target_modules"], bias="none", task_type="CAUSAL_LM")

    ds = load_dataset("json", data_files={"train": CFG["train_file"],
                                          "validation": CFG["val_file"]})

    def fmt(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False,
                                                add_generation_prompt=False)}
    ds = ds.map(fmt, remove_columns=ds["train"].column_names)

    # Response-only loss: mask everything before the assistant turn.
    # Each model family marks the assistant turn differently — must match the base model's
    # chat template exactly, or masking fails and the model trains on instructions too.
    bm = CFG["base_model"].lower()
    if "phi-3" in bm or "phi3" in bm or "phi-4" in bm:
        resp_template = "<|assistant|>\n"
    elif "qwen" in bm:
        resp_template = "<|im_start|>assistant\n"
    else:  # llama-style
        resp_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(response_template=resp_template, tokenizer=tok)

    args = SFTConfig(
        output_dir=CFG["output_dir"],
        num_train_epochs=CFG["num_train_epochs"],
        per_device_train_batch_size=CFG["per_device_train_batch_size"],
        gradient_accumulation_steps=CFG["gradient_accumulation_steps"],
        learning_rate=CFG["learning_rate"],
        lr_scheduler_type=CFG["lr_scheduler_type"],
        warmup_ratio=CFG["warmup_ratio"],
        weight_decay=CFG["weight_decay"],
        logging_steps=CFG["logging_steps"],
        eval_strategy=CFG["eval_strategy"], eval_steps=CFG["eval_steps"],
        save_steps=CFG["save_steps"], save_total_limit=CFG["save_total_limit"],
        load_best_model_at_end=CFG["load_best_model_at_end"],
        metric_for_best_model=CFG["metric_for_best_model"],
        bf16=CFG["bf16"], gradient_checkpointing=CFG["gradient_checkpointing"],
        max_seq_length=CFG["max_seq_len"], seed=CFG["seed"], report_to="none")

    trainer = SFTTrainer(model=model, args=args, peft_config=lora,
                         train_dataset=ds["train"], eval_dataset=ds["validation"],
                         processing_class=tok, data_collator=collator)
    trainer.train()
    trainer.save_model(CFG["output_dir"])
    tok.save_pretrained(CFG["output_dir"])
    print(f"adapter saved -> {CFG['output_dir']}")

if __name__ == "__main__":
    main()
