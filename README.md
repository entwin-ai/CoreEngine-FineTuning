# Entwin — Fine-Tuning Implementation (Pillar-Driven)

This package implements **only the fine-tuning layer** of Entwin, exactly as scoped in
`Entwin-Coding.xlsx` (CORE sheet). RAG, system-prompt engineering, and recipient
calibration are out of scope here — but the design respects the Excel's build-order warning:
**fine-tuning is the *last* varnish, not the foundation.** Run this after your RAG +
knowledge layer is working, or you ship "an eloquent twin that embarrasses you in week three."

## What fine-tuning is responsible for (per the Excel)

| Pillar | Role of fine-tuning | What we train |
|---|---|---|
| **Linguistic Fingerprint** | **PRIMARY** | Sentence-length distribution, punctuation habits (em-dashes, semicolons), contraction rate, prose-vs-bullets, active/passive, capitalization quirks — surface mechanics that live below conscious articulation. |
| **Voice & Identity** | **PRIMARY** | Recurring multi-word patterns, signature openings/closings, idiolect, humor/sarcasm cadence — patterns that recur across *thousands* of messages. |
| **Cognitive Style** | SECONDARY (reflex structure) | The habitual argument architecture when it's automatic, not chosen. |
| **Affective Register** | SECONDARY (cadence) | The word-by-word *way* warmth actually shows up. |
| **Meta-cognitive Patterns** | SECONDARY (cadence) | The specific sound of admitting error / signalling uncertainty. |
| **Decision & Judgment** | TERTIARY (least reliable) | Explicitly de-weighted — new decisions are out-of-distribution. We *exclude* decision content from style training to avoid hardcoding stale judgments. |

**Core principle the whole pipeline enforces:** fine-tune **HOW** the person writes, never
**WHAT** they know. Facts, project status, client identity → RAG. Style → fine-tune. Every
step below is built to strip semantic content and preserve only stylistic signal.

## Pipeline

```
0. watch.py          Watch a hardcoded inbox folder; on each new file, parse + trigger 1-6
1. extract.py        Pull your sent messages (Gmail / Drive / WhatsApp / Calendar notes)
2. profile.py        Compute the quantitative Linguistic Fingerprint (the guardrail summary)
3. build_dataset.py  Turn messages into instruction→response style-transfer pairs
4. train_qlora.py    QLoRA fine-tune a local base model (Phi-3.5-mini default; Qwen/Llama opt.)
5. evaluate.py       Score the fine-tune against the measured fingerprint; detect drift
6. merge_serve.py    Merge adapter, export GGUF for Ollama, ready for local inference
```

## Streaming ingest (watch.py) — the auto-trigger front end

Instead of (or in addition to) the one-shot `extract.py` bulk pull, you can drop files into a
**hardcoded inbox folder** and have everything run automatically. `watch.py`:

1. watches the folder (default `~/entwin_inbox`, override with `ENTWIN_INBOX`),
2. waits until each new file is fully written (size-stable — no half-copied reads),
3. parses it via `file_parser.py` into authored stylistic units (same schema as `extract.py`),
4. appends them to `data/raw_messages.jsonl`, de-duplicated by normalized text,
5. once reading is complete, **moves** the file to a separate hardcoded processed folder
   (`~/entwin_processed`, override with `ENTWIN_PROCESSED`), **renaming it with a
   monotonically-increasing alphanumeric prefix** (`<time36>-<seq36>__originalname`) so that
   a plain name-sort of that folder always reflects the chronological order files arrived;
   it also records the file in a ledger so it's never re-run,
6. after a short debounce (batches a burst of files), triggers the downstream pipeline.

### The sortable identifier (`monotonic_id.py`)

The prefix is `<TIME>-<SEQ>` in base36 (alphabet `0-9a-z`), each field zero-padded to a fixed
width so lexicographic ordering equals numeric ordering:
- **TIME** — milliseconds since epoch, 9 base36 chars (good past the year 3000).
- **SEQ** — a counter that breaks ties when multiple files land in the same millisecond.

It's guaranteed strictly increasing even across restarts and backwards clock jumps (NTP
corrections): the last-issued value is persisted, and the generator clamps forward rather
than ever emitting a smaller ID. Net effect: `ls | sort` over the processed folder is always
true chronological order, regardless of the original filenames.

**Supported file formats** (auto-detected): `.txt`, `.md`, `.json`, `.jsonl`, `.eml`,
`.csv` (text/body/message/content column), `.html`, `.pdf`, `.docx`, plus WhatsApp `.txt`
exports (auto-detected; keeps only your own lines). The same cleaning bar as the bulk pull
applies — signatures/quoted chains stripped, boilerplate and out-of-length-window units dropped.

```bash
python scripts/watch.py            # watch; run profile + build_dataset on each batch
python scripts/watch.py --full     # also run train + evaluate + merge after each batch
python scripts/watch.py --once     # process whatever's already in the folder, then exit
ENTWIN_INBOX=/data/dropzone python scripts/watch.py   # custom folder
ENTWIN_PROCESSED=/data/done python scripts/watch.py   # custom processed/archive folder
```

Backend is auto-selected: real OS filesystem events via `watchdog` if installed, otherwise a
dependency-free polling fallback. Default mode reruns only `profile.py` + `build_dataset.py`
per batch (cheap — keeps your fingerprint and dataset current as files stream in); pass
`--full` to retrain end-to-end each time, which you'll want only once enough new data has
accumulated.


## Hardware note (matches your stack)
- **Snapdragon X / Surface (ARM, no CUDA):** train in the cloud (one rented A100/4090 for
  a couple hours), then run inference locally via Ollama (`ollama run phi3.5` style) with the
  merged GGUF. QLoRA training needs CUDA; ARM is fine for inference only.
- **Current base: `microsoft/Phi-3.5-mini-instruct` (3.8B).** Being small, it QLoRA-tunes on
  a single 8-12GB GPU (even a 3060/4060) and runs comfortably for local inference — a good
  match for "running phi3.5 locally." Trade-off: less long-context voice retention than a 7-8B
  model, so lean a bit harder on the system-prompt fingerprint guardrail.
- **Heavier alternatives** (set in `configs/qlora.json` → `base_model`, and swap
  `target_modules` back to the split-projection names noted there): Qwen2.5-7B-Instruct or
  Meta-Llama-3.1-8B-Instruct on a 24GB GPU for stronger voice fidelity.

## Quickstart
```bash
pip install -r requirements.txt
python scripts/extract.py        # produces data/raw_messages.jsonl
python scripts/profile.py        # produces data/fingerprint.json + prompts/style_guardrail.md
python scripts/build_dataset.py  # produces data/train.jsonl / data/val.jsonl
python scripts/train_qlora.py    # produces out/adapter/
python scripts/evaluate.py       # produces out/eval_report.json
python scripts/merge_serve.py    # produces out/entwin-merged + GGUF instructions
```
